from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataset import (
    collect_annotation_json_paths,
    resolve_train_annotation_dir,
    resolve_train_view_dir,
)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class PairRecord:
    source_ref: str
    target_ref: str
    heading: float
    distance: float
    source_id: str
    target_id: str
    geometry: np.ndarray
    metadata: dict[str, Any]


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group())


def _normalize_ref(image_ref: str) -> str:
    return image_ref.strip().replace("\\", "/").lstrip("./")


def build_pair_key(source_id: str, target_id: str) -> str:
    return f"{_normalize_ref(source_id)}||{_normalize_ref(target_id)}"


def _view_index_from_name(image_name: str) -> int | None:
    stem = Path(image_name).stem
    parsed = _extract_first_int(stem)
    return parsed


def build_geometry_features(
    source_name: str,
    target_name: str,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build lightweight geometry priors from metadata or view-index proxy."""
    metadata = metadata or {}

    source_az = metadata.get("azimuth_a")
    target_az = metadata.get("azimuth_b")
    source_alt = metadata.get("altitude_a")
    target_alt = metadata.get("altitude_b")

    source_idx = _view_index_from_name(source_name)
    target_idx = _view_index_from_name(target_name)

    if source_az is None and source_idx is not None:
        source_az = (source_idx - 1) // 3
    if target_az is None and target_idx is not None:
        target_az = (target_idx - 1) // 3

    if source_alt is None and source_idx is not None:
        source_alt = (source_idx - 1) % 3
    if target_alt is None and target_idx is not None:
        target_alt = (target_idx - 1) % 3

    source_az = float(source_az) if source_az is not None else 0.0
    target_az = float(target_az) if target_az is not None else 0.0
    source_alt = float(source_alt) if source_alt is not None else 0.0
    target_alt = float(target_alt) if target_alt is not None else 0.0

    azimuth_delta = target_az - source_az
    azimuth_delta_rad = math.radians(20.0 * azimuth_delta)

    overlap_proxy = float(max(0.0, math.cos(azimuth_delta_rad)))

    explicit_overlap = metadata.get("overlap")
    if explicit_overlap is not None:
        overlap_proxy = float(explicit_overlap)

    features = np.asarray(
        [
            math.sin(azimuth_delta_rad),
            math.cos(azimuth_delta_rad),
            (target_alt - source_alt) / 2.0,
            overlap_proxy,
            source_alt / 3.0,
            target_alt / 3.0,
        ],
        dtype=np.float32,
    )
    return features


def _safe_load_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_official_json_paths(
    annotation_dir: Path,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    paths = collect_annotation_json_paths(annotation_dir)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 annotation files for split, got {len(paths)}")

    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.parent.name, []).append(path)

    if len(grouped) > 1:
        groups = sorted(grouped)
        rng = random.Random(seed)
        rng.shuffle(groups)
        split_idx = max(1, int(len(groups) * (1.0 - val_ratio)))
        split_idx = min(split_idx, len(groups) - 1)

        train_groups = set(groups[:split_idx])
        train_paths = [path for group in train_groups for path in grouped[group]]
        val_paths = [path for group in groups[split_idx:] for path in grouped[group]]
    else:
        rng = random.Random(seed)
        paths = paths.copy()
        rng.shuffle(paths)
        split_idx = max(1, int(len(paths) * (1.0 - val_ratio)))
        split_idx = min(split_idx, len(paths) - 1)
        train_paths, val_paths = paths[:split_idx], paths[split_idx:]

    return train_paths, val_paths


class OfflineMatchFeatureStore:
    """Lazy loader for precomputed local correspondence summaries."""

    def __init__(
        self,
        match_root: str | None,
        index_file: str | None = None,
        feature_dim: int = 8,
    ) -> None:
        self.feature_dim = feature_dim
        self.default = np.zeros(self.feature_dim, dtype=np.float32)

        self.match_root = Path(match_root).expanduser().resolve() if match_root else None
        self.index_file = Path(index_file).expanduser().resolve() if index_file else None
        self.index: dict[str, Path] = {}
        self.cache: dict[str, np.ndarray] = {}

        if self.match_root and self.match_root.is_dir() and self.index_file and self.index_file.is_file():
            self._load_index()

    @property
    def enabled(self) -> bool:
        return self.match_root is not None and self.match_root.is_dir()

    def _load_index(self) -> None:
        assert self.index_file is not None
        assert self.match_root is not None
        with self.index_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = row.get("source") or row.get("image_a")
                target = row.get("target") or row.get("image_b")
                file_ref = row.get("path") or row.get("match_file")
                if not source or not target or not file_ref:
                    continue
                key = build_pair_key(source, target)
                self.index[key] = (self.match_root / file_ref).resolve()

    def _candidate_paths(self, source_id: str, target_id: str) -> list[Path]:
        assert self.match_root is not None

        src = Path(source_id)
        tgt = Path(target_id)

        source_tokens = [src.stem, src.name]
        target_tokens = [tgt.stem, tgt.name]

        candidates: list[Path] = []
        for src_token in source_tokens:
            for tgt_token in target_tokens:
                base = f"{src_token}__{tgt_token}"
                for ext in (".json", ".npz", ".npy"):
                    candidates.append(self.match_root / f"{base}{ext}")

        return candidates

    def _summary_from_dense_dict(self, payload: dict[str, Any]) -> np.ndarray:
        match_count = float(payload.get("match_count", payload.get("num_matches", 0.0)))
        inlier_ratio = float(payload.get("inlier_ratio", payload.get("inlier", 0.0)))
        mean_dx = float(payload.get("mean_dx", payload.get("dx", 0.0)))
        mean_dy = float(payload.get("mean_dy", payload.get("dy", 0.0)))
        std_dx = float(payload.get("std_dx", payload.get("dx_std", 0.0)))
        std_dy = float(payload.get("std_dy", payload.get("dy_std", 0.0)))
        scale_proxy = float(payload.get("scale_proxy", payload.get("scale", 0.0)))
        confidence = float(payload.get("confidence", payload.get("mean_conf", 0.0)))

        return np.asarray(
            [
                min(2.0, math.log1p(max(match_count, 0.0)) / 6.0),
                float(np.clip(inlier_ratio, 0.0, 1.0)),
                float(np.clip(mean_dx / 256.0, -4.0, 4.0)),
                float(np.clip(mean_dy / 256.0, -4.0, 4.0)),
                float(np.clip(std_dx / 256.0, 0.0, 4.0)),
                float(np.clip(std_dy / 256.0, 0.0, 4.0)),
                float(np.clip(scale_proxy, -4.0, 4.0)),
                float(np.clip(confidence, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _summary_from_arrays(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        matches: np.ndarray,
        confidence: np.ndarray | None,
    ) -> np.ndarray:
        valid = matches >= 0
        count = int(valid.sum())
        if count <= 0:
            return self.default.copy()

        src_points = keypoints0[valid]
        tgt_points = keypoints1[matches[valid]]
        displacement = tgt_points - src_points

        norms = np.linalg.norm(displacement, axis=1)
        conf = confidence[valid] if confidence is not None and confidence.shape[0] == matches.shape[0] else None

        inlier_ratio = float(np.mean(norms < 15.0))
        mean_conf = float(np.mean(conf)) if conf is not None else 0.5

        summary = np.asarray(
            [
                min(2.0, math.log1p(float(count)) / 6.0),
                float(np.clip(inlier_ratio, 0.0, 1.0)),
                float(np.clip(np.mean(displacement[:, 0]) / 256.0, -4.0, 4.0)),
                float(np.clip(np.mean(displacement[:, 1]) / 256.0, -4.0, 4.0)),
                float(np.clip(np.std(displacement[:, 0]) / 256.0, 0.0, 4.0)),
                float(np.clip(np.std(displacement[:, 1]) / 256.0, 0.0, 4.0)),
                float(np.clip(np.mean(norms) / 256.0, 0.0, 4.0)),
                float(np.clip(mean_conf, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return summary

    def _read_match_file(self, file_path: Path) -> np.ndarray:
        if file_path.suffix.lower() == ".json":
            payload = _safe_load_json(file_path)
            if isinstance(payload, dict) and {"keypoints0", "keypoints1", "matches"}.issubset(payload.keys()):
                keypoints0 = np.asarray(payload["keypoints0"], dtype=np.float32)
                keypoints1 = np.asarray(payload["keypoints1"], dtype=np.float32)
                matches = np.asarray(payload["matches"], dtype=np.int64)
                confidence = np.asarray(payload.get("confidence"), dtype=np.float32) if "confidence" in payload else None
                return self._summary_from_arrays(keypoints0, keypoints1, matches, confidence)
            if isinstance(payload, dict):
                return self._summary_from_dense_dict(payload)
            return self.default.copy()

        if file_path.suffix.lower() == ".npz":
            with np.load(file_path, allow_pickle=False) as payload:
                keys = set(payload.keys())
                if {"keypoints0", "keypoints1", "matches"}.issubset(keys):
                    confidence = payload["confidence"] if "confidence" in keys else None
                    return self._summary_from_arrays(
                        payload["keypoints0"],
                        payload["keypoints1"],
                        payload["matches"],
                        confidence,
                    )
                dense_payload = {key: payload[key].item() if payload[key].shape == () else payload[key] for key in payload.keys()}
                dense_dict = {
                    key: float(value)
                    for key, value in dense_payload.items()
                    if np.asarray(value).size == 1
                }
                return self._summary_from_dense_dict(dense_dict)

        if file_path.suffix.lower() == ".npy":
            array = np.load(file_path, allow_pickle=False)
            flat = np.asarray(array, dtype=np.float32).reshape(-1)
            if flat.size >= self.feature_dim:
                return flat[: self.feature_dim].astype(np.float32)
            output = self.default.copy()
            output[: flat.size] = flat
            return output

        return self.default.copy()

    def get(self, source_id: str, target_id: str) -> np.ndarray:
        if not self.enabled:
            return self.default.copy()

        key = build_pair_key(source_id, target_id)
        if key in self.cache:
            return self.cache[key].copy()

        file_path = self.index.get(key)
        reverse_used = False

        if file_path is None and self.match_root is not None:
            for candidate in self._candidate_paths(source_id, target_id):
                if candidate.is_file():
                    file_path = candidate
                    break

            if file_path is None:
                reverse_key = build_pair_key(target_id, source_id)
                file_path = self.index.get(reverse_key)
                reverse_used = file_path is not None
                if file_path is None:
                    for candidate in self._candidate_paths(target_id, source_id):
                        if candidate.is_file():
                            file_path = candidate
                            reverse_used = True
                            break

        if file_path is None or not file_path.is_file():
            self.cache[key] = self.default.copy()
            return self.cache[key].copy()

        summary = self._read_match_file(file_path)
        if reverse_used:
            summary = summary.copy()
            summary[2] *= -1.0
            summary[3] *= -1.0

        self.cache[key] = summary
        return summary.copy()


class PairUAVDataset(Dataset):
    """Official-label or pseudo-label PairUAV dataset with optional match summaries."""

    def __init__(
        self,
        root: str,
        mode: str = "auto",
        json_paths: list[Path] | None = None,
        buildings: list[str] | None = None,
        max_pairs: int | None = None,
        seed: int = 42,
        image_size: int = 224,
        augment: bool = True,
        is_val: bool = False,
        min_distance: float = 1.0,
        match_root: str | None = None,
        match_index_file: str | None = None,
        strict_official_only: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.seed = seed
        self.image_size = image_size
        self.augment = augment and not is_val
        self.is_val = is_val
        self.min_distance = min_distance
        self.strict_official_only = strict_official_only
        self.mode = "official"

        if self.annotation_dir is None:
            raise FileNotFoundError(
                f"Official mode requested but no annotation JSON folder exists under {self.root}"
            )

        self.view_dir = resolve_train_view_dir(self.root)
        self.records = self._build_records(
            annotation_dir=self.annotation_dir,
            json_paths=json_paths,
            buildings=buildings,
            max_pairs=max_pairs,
        )

        if not self.records:
            raise RuntimeError(f"No training pairs were built for mode={self.mode} root={self.root}")

        self.match_store = OfflineMatchFeatureStore(
            match_root=match_root,
            index_file=match_index_file,
            feature_dim=8,
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.augment:
            self.source_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if max_pairs is not None and len(records) > max_pairs:
            rng = random.Random(self.seed)
            records = rng.sample(records, max_pairs)

        records.sort(key=lambda item: (item.source_id, item.target_id))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def sample_decoded_labels(self, sample_count: int = 3, seed: int | None = None) -> list[dict[str, Any]]:
        if not self.records:
            return []

        rng = random.Random(self.seed if seed is None else seed)
        take = min(sample_count, len(self.records))
        indices = sorted(rng.sample(range(len(self.records)), take))

        samples: list[dict[str, Any]] = []
        for idx in indices:
            record = self.records[idx]
            samples.append(
                {
                    "index": idx,
                    "source_id": record.source_id,
                    "target_id": record.target_id,
                    "heading_deg": float(record.heading),
                    "distance_m": float(record.distance),
                    "json_path": str(record.metadata.get("json_path", "")),
                }
            )
        return samples

    def diagnostics(self, sample_count: int = 3, seed: int | None = None) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "strict_official_only": self.strict_official_only,
            "annotation_dir": str(self.annotation_dir) if self.annotation_dir is not None else None,
            "annotation_source_count": len(self.annotation_source_paths),
            "annotation_sources": [str(path) for path in self.annotation_source_paths[:5]],
            "pair_count": len(self.records),
            "label_samples": self.sample_decoded_labels(sample_count=sample_count, seed=seed),
            "match_feature_store_enabled": bool(self.match_store.enabled),
        }

    def _resolve_image_path(self, image_ref: str) -> Path:
        normalized = _normalize_ref(image_ref)
        cached = self._image_path_cache.get(normalized)
        if cached is not None and cached.is_file():
            return cached

        rel = Path(normalized)
        candidates: list[Path] = [
            self.view_dir / rel,
            self.view_dir / rel.name,
        ]

        if rel.parent != Path("."):
            candidates.append(self.view_dir / rel.parent.name / rel.name)

        if rel.suffix.lower() == ".webp":
            for suffix in (".jpeg", ".jpg", ".png"):
                candidates.append(self.view_dir / rel.with_suffix(suffix))
                if rel.parent != Path("."):
                    candidates.append(self.view_dir / rel.parent.name / rel.with_suffix(suffix).name)

        for candidate in candidates:
            if candidate.is_file():
                self._image_path_cache[normalized] = candidate
                return candidate

        raise FileNotFoundError(
            f"Could not resolve image ref '{image_ref}' under {self.view_dir} (tried {len(candidates)} candidates)."
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[index]

        source_path = self._resolve_image_path(record.source_ref)
        target_path = self._resolve_image_path(record.target_ref)

        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        source_tensor = self.source_transform(source_img)
        target_tensor = self.target_transform(target_img)

        match_features = self.match_store.get(record.source_id, record.target_id)
        geometry_features = record.geometry

        heading = float(record.heading)
        distance = max(self.min_distance, float(record.distance))

        target = {
            "heading": torch.tensor(heading, dtype=torch.float32),
            "distance": torch.tensor(distance, dtype=torch.float32),
            "log_distance": torch.tensor(math.log(distance), dtype=torch.float32),
            "match_features": torch.tensor(match_features, dtype=torch.float32),
            "geometry_features": torch.tensor(geometry_features, dtype=torch.float32),
            "source_id": record.source_id,
            "target_id": record.target_id,
            "pair_key": build_pair_key(record.source_id, record.target_id),
            "source_path": str(source_path),
            "target_path": str(target_path),
            "mode": self.mode,
        }
        return source_tensor, target_tensor, target


