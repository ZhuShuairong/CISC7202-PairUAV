"""Generate CodaBench submission files for PairUAV."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
import torchvision.models as models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baseline import siamese_fusion
from models.harp_dual_path import HARPDualPath
from models.harp_pose_lite import HARPPoseLite


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PAIR_MANIFEST_NAMES = (
    "pairs.txt",
    "pair.txt",
    "pair_list.txt",
    "test_pairs.txt",
    "query_pairs.txt",
    "pairs.csv",
    "test.csv",
    "annotations.csv",
)

DEFAULT_PAIRUAV_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/university/University-Release/University-Release"),
    Path("/root/autodl-tmp/university/PairUAV"),
    Path("/root/autodl-tmp/university/PairUAV-Processed"),
    Path("/root/autodl-pub/PairUAV"),
)


def resolve_pairuav_root(root: str | None) -> Path:
    checked: list[Path] = []
    seen: set[str] = set()
    candidates: list[Path] = []
    for env_name in ("PAIRUAV_ROOT", "PAIRUAV_DATA_ROOT", "PAIRUAV_PROCESSED_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    if root:
        explicit = Path(root).expanduser()
        checked.append(explicit)
        seen.add(str(explicit))
        if explicit.is_dir():
            return explicit.resolve()

    candidates.extend(DEFAULT_PAIRUAV_ROOT_CANDIDATES)

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not determine a usable PairUAV root. Checked: "
        + ", ".join(str(path) for path in checked)
        + ". Set --pairuav-root or the PAIRUAV_ROOT environment variable to the mounted data path."
    )


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _scan_image_files(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if _is_image_file(path)],
        key=lambda path: str(path.relative_to(root)).replace("\\", "/").lower(),
    )


def _build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]], list[Path]]:
    images = _scan_image_files(root)
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in images:
        relative = str(path.relative_to(root)).replace("\\", "/")
        by_relative[relative] = path
        by_name.setdefault(path.name.lower(), []).append(path)
    return by_relative, by_name, images


def _resolve_image_ref(ref: str, root: Path, by_relative: dict[str, Path],
                       by_name: dict[str, list[Path]]) -> Path | None:
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    normalized = ref.strip().lstrip("./").replace("\\", "/")
    direct = root / normalized
    if direct.exists():
        return direct

    for prefix in ("test/", "test_tour/", "train/", "train_tour/"):
        prefixed = root / prefix / normalized
        if prefixed.exists():
            return prefixed

    if normalized in by_relative:
        return by_relative[normalized]

    matches = by_name.get(Path(normalized).name.lower(), [])
    if len(matches) == 1:
        return matches[0]

    return None


def _parse_manifest(manifest: Path, root: Path,
                    by_relative: dict[str, Path], by_name: dict[str, list[Path]]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                fields = [field.strip() for field in next(csv.reader([line]))]
            elif "\t" in line:
                fields = [field.strip() for field in line.split("\t")]
            else:
                fields = line.split()

            if len(fields) < 2:
                continue

            source = _resolve_image_ref(fields[0], root, by_relative, by_name)
            target = _resolve_image_ref(fields[1], root, by_relative, by_name)
            if source is not None and target is not None:
                pairs.append((source, target))

    return pairs


def _group_image_sets(directory: Path) -> dict[str, list[Path]]:
    subdirs = [path for path in directory.iterdir() if path.is_dir()]
    if not subdirs:
        images = _scan_image_files(directory)
        return {"": images} if images else {}

    grouped: dict[str, list[Path]] = {}
    for subdir in sorted(subdirs, key=lambda path: path.name.lower()):
        grouped[subdir.name] = _scan_image_files(subdir)
    return grouped


def _find_split_pair_dirs(root: Path) -> tuple[Path, Path] | None:
    split_roots = [root / "test", root / "test_tour", root]
    directory_name_pairs = [
        ("query_drone", "gallery_drone"),
        ("query", "gallery"),
        ("source_drone", "target_drone"),
        ("source", "target"),
    ]

    for split_root in split_roots:
        for query_name, gallery_name in directory_name_pairs:
            query_dir = split_root / query_name
            gallery_dir = split_root / gallery_name
            if query_dir.is_dir() and gallery_dir.is_dir():
                return query_dir, gallery_dir

    return None


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    by_relative, by_name, _ = _build_image_index(root)

    test_root = root / "test"
    if test_root.is_dir():
        grouped: dict[str, set[str]] = defaultdict(set)
        json_found = False
        for subdir in sorted(test_root.iterdir(), key=lambda path: path.name.lower()):
            if not subdir.is_dir():
                continue
            for json_file in sorted(subdir.glob("*.json")):
                json_found = True
                with json_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                image_a = data.get("image_a")
                image_b = data.get("image_b")
                if not image_a or not image_b:
                    continue
                grouped[str(image_a)].add(str(image_b))

        if json_found and grouped:
            ordered_pairs: list[tuple[Path, Path]] = []
            for image_a, image_b_set in sorted(grouped.items()):
                image_a_ref = str(Path(image_a).with_suffix(".webp"))
                for image_b in sorted(image_b_set):
                    source = _resolve_image_ref(image_a_ref, root, by_relative, by_name)
                    target = _resolve_image_ref(image_b, root, by_relative, by_name)
                    if source is not None and target is not None:
                        ordered_pairs.append((source, target))
            if ordered_pairs:
                return ordered_pairs

    for base in (root, root / "test", root / "test_tour"):
        for name in PAIR_MANIFEST_NAMES:
            candidate = base / name
            if candidate.is_file():
                pairs = _parse_manifest(candidate, root, by_relative, by_name)
                if pairs:
                    return pairs

    split_dirs = _find_split_pair_dirs(root)
    if split_dirs is None:
        raise FileNotFoundError(
            f"Could not find test pair data under {root}. "
            "Expected a manifest file or query/gallery split directories."
        )

    query_dir, gallery_dir = split_dirs
    query_groups = _group_image_sets(query_dir)
    gallery_groups = _group_image_sets(gallery_dir)

    pairs: list[tuple[Path, Path]] = []
    common_group_names = [name for name in sorted(query_groups) if name in gallery_groups]

    if common_group_names:
        for group_name in common_group_names:
            query_images = query_groups[group_name]
            gallery_images = gallery_groups[group_name]
            if not query_images or not gallery_images:
                continue
            pair_count = min(len(query_images), len(gallery_images))
            pairs.extend(zip(query_images[:pair_count], gallery_images[:pair_count]))
        if pairs:
            return pairs

    query_flat = _scan_image_files(query_dir)
    gallery_flat = _scan_image_files(gallery_dir)
    pair_count = min(len(query_flat), len(gallery_flat))
    return list(zip(query_flat[:pair_count], gallery_flat[:pair_count]))


def _load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    dual_path_keys = any(
        key.startswith("wide.") or key.startswith("deep.") or key.startswith("fusion.")
        for key in state_dict.keys()
    )
    pose_lite_keys = any(
        key.startswith("regression_head.") or key.startswith("conf_head.")
        for key in state_dict.keys()
    )

    if dual_path_keys and not pose_lite_keys:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path"

    if pose_lite_keys:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "pose_lite"

    try:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "pose_lite"
    except Exception:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path"


def _feature_cache_path(cache_root: Path, root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
        relative = Path("external") / digest / image_path.name
    return (cache_root / relative).with_suffix(".pt")


def _extract_features(model, image_paths: Sequence[Path],
                      device: torch.device, root: Path,
                      cache_dir: str | None = None,
                      batch_size: int = 32) -> dict[Path, torch.Tensor]:
    transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    backbone = model.backbone
    backbone.eval()

    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    features: dict[Path, torch.Tensor] = {}
    pending_paths: list[Path] = []

    for path in image_paths:
        if cache_root is not None:
            cache_file = _feature_cache_path(cache_root, root, path)
            if cache_file.is_file():
                features[path] = torch.load(cache_file, map_location="cpu").half()
                continue
        pending_paths.append(path)

    for start in range(0, len(pending_paths), batch_size):
        batch_paths = list(pending_paths[start:start + batch_size])
        batch = torch.stack([
            transform(Image.open(path).convert("RGB")) for path in batch_paths
        ]).to(device)
        with torch.inference_mode():
            batch_features = backbone(batch)

        for path, feat in zip(batch_paths, batch_features):
            stored = feat.detach().cpu().half()
            features[path] = stored
            if cache_root is not None:
                cache_file = _feature_cache_path(cache_root, root, path)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stored, cache_file)

    return features


def _predict_pair_batch(model, model_kind: str, batch_pairs: Sequence[tuple[Path, Path]],
                        feature_cache: dict[Path, torch.Tensor],
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    source_features = torch.stack([feature_cache[source].float() for source, _ in batch_pairs]).to(device)
    target_features = torch.stack([feature_cache[target].float() for _, target in batch_pairs]).to(device)

    with torch.inference_mode():
        if model_kind == "dual_path":
            pred = model.forward_features(source_features, target_features)
            heading = pred["heading"]
            distance = pred["distance"]
        else:
            fused = siamese_fusion(source_features, target_features)
            raw = model.regression_head(fused)
            heading = raw[:, 0]
            distance = torch.clamp(raw[:, 1], min=0.0)

    return heading, distance


def generate_submission(checkpoint: str, cache_dir: str | None = None,
                        pairuav_root: str | None = None,
                        output: str = "result.txt", split: str = "query"):
    """Generate result.txt for CodaBench submission."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_pairuav_root(pairuav_root)

    model, model_kind = _load_model(checkpoint, device)
    pairs = _discover_pairs(root)

    if not pairs:
        raise RuntimeError(f"No test pairs were discovered under {root}.")

    unique_images = sorted({path for pair in pairs for path in pair}, key=lambda path: str(path).lower())
    feature_cache = _extract_features(model, unique_images, device, root, cache_dir=cache_dir)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 64 if model_kind == "pose_lite" else 32
    print(f"Generating submission -> {output_path}")
    print(f"  Pairs: {len(pairs)} | Unique images: {len(unique_images)} | Model: {model_kind}")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            heading, distance = _predict_pair_batch(model, model_kind, batch_pairs, feature_cache, device)
            heading = heading.detach().cpu().tolist()
            distance = distance.detach().cpu().tolist()
            for angle, dist in zip(heading, distance):
                angle_value = ((float(angle) + 180.0) % 360.0) - 180.0
                distance_value = max(0.0, float(dist))
                handle.write(f"{angle_value:.6f} {distance_value:.6f}\n")

    print(f"Result written to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache", type=str)
    p.add_argument("--pairuav-root", "--university-release", "--data", dest="pairuav_root",
                   type=str, default=None,
                   help="Path to the PairUAV data root used for submission discovery; when omitted the script auto-detects the mounted AutoDL dataset")
    p.add_argument("--output", type=str, default="result.txt")
    args = p.parse_args()
    generate_submission(args.checkpoint, args.cache, args.pairuav_root, args.output)
