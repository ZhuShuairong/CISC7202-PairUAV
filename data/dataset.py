import os
import random
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def resolve_train_view_dir(root: Path) -> Path:
    """Resolve the training image directory for raw or processed PairUAV data."""
    candidates = [
        root / "train" / "drone",
        root / "train_tour",
        root / "train",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find a training image directory under {root}. "
        "Expected one of: train/drone, train_tour, train."
    )


def resolve_train_annotation_dir(root: Path) -> Path | None:
    """Locate PairUAV train JSON annotations if available."""
    candidates = [
        root / "train",
        root / "pairUAV" / "train",
        root / "PairUAV" / "train",
    ]

    for candidate in candidates:
        if not candidate.is_dir():
            continue

        for child in candidate.iterdir():
            if child.is_file() and child.suffix.lower() == ".json":
                return candidate
            if child.is_dir() and any(path.is_file() for path in child.glob("*.json")):
                return candidate

    return None


def _extract_int(value: str) -> int:
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group())
    return 10**12


def _json_path_sort_key(json_path: Path) -> tuple[int, str, int, str]:
    group_name = json_path.parent.name
    json_name = json_path.stem
    return (
        _extract_int(group_name),
        group_name,
        _extract_int(json_name),
        json_name,
    )


def collect_annotation_json_paths(annotation_dir: Path,
                                 groups: Optional[List[str]] = None) -> List[Path]:
    """Collect JSON annotations from train folder (flat or grouped)."""
    paths: List[Path] = []

    if groups is None:
        paths.extend(path for path in annotation_dir.glob("*.json") if path.is_file())
        for subdir in sorted([p for p in annotation_dir.iterdir() if p.is_dir()],
                             key=lambda p: p.name.lower()):
            paths.extend(path for path in subdir.glob("*.json") if path.is_file())
    else:
        for group in groups:
            if group == "__root__":
                paths.extend(path for path in annotation_dir.glob("*.json") if path.is_file())
                continue
            group_dir = annotation_dir / group
            if group_dir.is_dir():
                paths.extend(path for path in group_dir.glob("*.json") if path.is_file())

    return sorted(paths, key=_json_path_sort_key)


class PairUAVAnnotationDataset(Dataset):
    """PairUAV training dataset using official JSON annotations.

    Each sample reads one train JSON item and resolves `image_a`, `image_b`,
    `heading_num`, and `range_num` as supervision.
    """

    def __init__(
        self,
        root: str,
        max_pairs: int = 960_000,
        groups: Optional[List[str]] = None,
        json_paths: Optional[List[Path]] = None,
        seed: int = 42,
        is_val: bool = False,
    ):
        self.root = Path(root)
        self.max_pairs = max_pairs
        self.seed = seed
        self.is_val = is_val

        annotation_dir = resolve_train_annotation_dir(self.root)
        if annotation_dir is None:
            raise FileNotFoundError(
                f"Could not find official train annotations under {self.root}. "
                "Expected train/*.json or train/*/*.json"
            )
        self.annotation_dir = annotation_dir
        self.view_dir = resolve_train_view_dir(self.root)

        if json_paths is None:
            all_json = collect_annotation_json_paths(self.annotation_dir, groups=groups)
        else:
            all_json = [Path(path) for path in json_paths]

        if not all_json:
            raise FileNotFoundError(
                f"No annotation JSON files found in {self.annotation_dir} "
                f"for groups={groups}."
            )

        rng = random.Random(self.seed)
        if len(all_json) > self.max_pairs:
            if self.is_val:
                all_json = all_json[:self.max_pairs]
            else:
                all_json = rng.sample(all_json, self.max_pairs)
                all_json = sorted(all_json, key=_json_path_sort_key)

        self.json_paths = all_json
        self._image_path_cache: Dict[str, Path] = {}

        if is_val:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        self.target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        print(
            "[PairUAVAnnotationDataset] "
            f"json={len(self.json_paths)} root={self.annotation_dir} view={self.view_dir}"
        )

    def __len__(self):
        return len(self.json_paths)

    def _resolve_image_path(self, image_ref: str) -> Path:
        normalized = str(image_ref).strip().replace("\\", "/")
        cached = self._image_path_cache.get(normalized)
        if cached is not None and cached.is_file():
            return cached

        rel = Path(normalized)
        image_name = rel.name
        image_stem = Path(image_name).stem
        suffix = rel.suffix.lower()

        candidates = [
            self.view_dir / rel,
            self.view_dir / image_name,
        ]

        if rel.parent != Path("."):
            candidates.append(self.view_dir / rel.parent.name / image_name)

        if suffix == ".webp":
            for alt_suffix in (".jpeg", ".jpg", ".png"):
                candidates.append(self.view_dir / f"{image_stem}{alt_suffix}")
                if rel.parent != Path("."):
                    candidates.append(self.view_dir / rel.parent.name / f"{image_stem}{alt_suffix}")

        for candidate in candidates:
            if candidate.is_file():
                self._image_path_cache[normalized] = candidate
                return candidate

        raise FileNotFoundError(
            f"Image not found for reference '{image_ref}'. "
            f"Tried {len(candidates)} candidate paths under {self.view_dir}."
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        json_path = self.json_paths[idx]
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_a = data.get("image_a")
        image_b = data.get("image_b")
        if not image_a or not image_b:
            raise ValueError(f"Invalid annotation (missing image_a/image_b): {json_path}")

        source_path = self._resolve_image_path(str(image_a))
        target_path = self._resolve_image_path(str(image_b))

        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        source = self.transform(source_img)
        target = self.target_transform(target_img)

        heading = float(data.get("heading_num", data.get("heading", 0.0)))
        distance = float(data.get("range_num", data.get("distance", 0.0)))

        return source, target, {
            "heading": heading,
            "distance": distance,
            "building": json_path.parent.name,
            "source_name": source_path.name,
            "target_name": target_path.name,
            "json_path": str(json_path),
        }


