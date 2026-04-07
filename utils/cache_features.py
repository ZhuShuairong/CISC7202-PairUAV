"""
Pre-extract ResNet-50 spatial features and cache to disk (fp16).
Provides CachedDataset for training without the backbone in GPU memory.
"""
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from data.dataset import resolve_train_view_dir


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_and_cache(university_release_root: str, cache_dir: str,
                      device: str = "cuda", batch_size: int = 32):
    """Extract layer4 features for every image and save as .npz files.

    Each file: cache/<building_id>.npz  (fp16, compressed)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir)

    # ResNet-50 up to layer4 → (B, 2048, 7, 7)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-2]).eval().to(device)

    transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    drone_dir = resolve_train_view_dir(Path(university_release_root))
    buildings = sorted(os.listdir(drone_dir))

    for bld in tqdm(buildings, desc="Caching buildings"):
        bld_dir = drone_dir / bld
        if not bld_dir.is_dir():
            continue
        out_file = cache_path / f"{bld}.npz"
        if out_file.exists():
            continue

        imgs = sorted(f for f in os.listdir(bld_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not imgs:
            continue

        feats: dict = {}
        for start in range(0, len(imgs), batch_size):
            batch_names = imgs[start:start + batch_size]
            batch = torch.stack([
                transform(Image.open(bld_dir / n).convert("RGB"))
                for n in batch_names
            ]).to(device)
            with torch.no_grad():
                f = backbone(batch).cpu().numpy().astype(np.float16)  # (B,2048,7,7)
            for i, n in enumerate(batch_names):
                feats[n] = f[i]

        np.savez_compressed(str(out_file), **feats)

    count = len(list(cache_path.glob("*.npz")))
    print(f"[cache] Cached {count}/{len(buildings)} buildings → {cache_dir}")


# ---------------------------------------------------------------------------
# Cached dataset
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """Loads pre-extracted features instead of raw images.

    Generates pseudo-labels from image indices (azimuth proxy).
    """
    def __init__(self, cache_dir: str, max_pairs: int = 960_000,
                 buildings: list | None = None, seed: int = 42,
                 is_val: bool = False, preload: bool = True):
        import random
        self.cache_dir = Path(cache_dir)
        self.max_pairs = max_pairs
        self.seed = seed
        self.is_val = is_val
        self.preload = preload

        npz_files = sorted(self.cache_dir.glob("*.npz"))
        cached_ids = set(f.stem for f in npz_files)

        if buildings is None:
            buildings = sorted(cached_ids)
        self.buildings = [b for b in buildings if b in cached_ids]

        # Build per-building image index
        self.info: dict[str, list[str]] = {}
        self.feature_store: dict[str, dict[str, torch.Tensor]] = {}
        for bld in self.buildings:
            with np.load(self.cache_dir / f"{bld}.npz") as data:
                names = sorted(k for k in data.keys()
                               if k.lower().endswith((".png", ".jpg", ".jpeg")))
                if self.preload and names:
                    self.feature_store[bld] = {
                        name: torch.tensor(data[name], dtype=torch.float16)
                        for name in names
                    }
            if names:
                self.info[bld] = names

        self.pairs = self._generate_pairs()
        print(f"[CachedDataset] {len(self.buildings)} buildings, "
              f"{len(self.pairs)} pairs")

    # ------------------------------------------------------------------
    def _generate_pairs(self):
        import random
        rng = random.Random(self.seed)
        pairs = []

        total_possible = sum(
            n * (n - 1) // 2 for n in (len(v) for v in self.info.values())
        )

        for bld, imgs in self.info.items():
            n = len(imgs)
            if n < 2:
                continue
            possible = n * (n - 1) // 2

            if self.is_val or possible <= max(1, possible * self.max_pairs // max(total_possible, 1)):
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append({"building": bld,
                                      "source": imgs[i], "target": imgs[j]})
            else:
                n_sample = max(1, int(possible * self.max_pairs / total_possible))
                for _ in range(n_sample):
                    i, j = rng.sample(range(n), 2)
                    if i > j:
                        i, j = j, i
                    pairs.append({"building": bld,
                                  "source": imgs[i], "target": imgs[j]})

        if not self.is_val and len(pairs) > self.max_pairs:
            pairs = rng.sample(pairs, self.max_pairs)
        return pairs

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        bld = pair["building"]
        if self.preload:
            f_s = self.feature_store[bld][pair["source"]]
            f_t = self.feature_store[bld][pair["target"]]
        else:
            with np.load(self.cache_dir / f"{bld}.npz") as data:
                f_s = torch.from_numpy(data[pair["source"]]).half()  # (2048, 7, 7)
                f_t = torch.from_numpy(data[pair["target"]]).half()

        src_idx = int(pair["source"].split("-")[1].split(".")[0])
        tgt_idx = int(pair["target"].split("-")[1].split(".")[0])

        src_az = (src_idx - 1) // 3
        tgt_az = (tgt_idx - 1) // 3

        heading = (tgt_az - src_az) * 20.0
        heading = ((heading + 180) % 360) - 180

        az_diff = abs(tgt_az - src_az)
        distance = az_diff * 70.0 / 18.0

        return {
            "feat_s": f_s,
            "feat_t": f_t,
            "heading": torch.tensor(heading, dtype=torch.float32),
            "distance": torch.tensor(distance, dtype=torch.float32),
            "building": bld,
            "source": pair["source"],
            "target": pair["target"],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--university-release", "--data", dest="university_release",
                   type=str, required=True,
                   help="Path to the University-Release root or processed PairUAV root")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    extract_and_cache(args.university_release, args.cache, args.device, args.batch_size)
