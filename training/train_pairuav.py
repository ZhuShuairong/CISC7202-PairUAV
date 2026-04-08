from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import resolve_train_annotation_dir, resolve_train_view_dir
from data.dataset_pairuav import (
    OfflineMatchFeatureStore,
    PairUAVDataset,
    collect_official_test_pairs,
    count_official_test_pairs,
    resolve_test_annotation_dir,
    split_official_json_paths,
)
from models.geopairnet import GeoPairNet
from training.losses import PairUAVLoss
from utils.metrics import comprehensive_metrics, evaluate_result_files, is_better_result, write_result_file


@dataclass
class StageRuntimeState:
    optimizer_steps: int = 0


class CosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)

    def _lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)

        effective_steps = max(1, self.total_steps - self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(effective_steps)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, step: int) -> float:
        lr = self._lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if not value.dtype.is_floating_point:
                self.shadow[key] = value.detach().clone()
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


class FrozenFeatureCache:
    """CPU cache for frozen-trunk embeddings keyed by image id."""

    def __init__(self, max_items: int = 60_000) -> None:
        self.max_items = max_items
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _insert(self, key: str, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

    @torch.no_grad()
    def encode(
        self,
        model: GeoPairNet,
        images: torch.Tensor,
        image_ids: list[str],
        device: torch.device,
        channels_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_missing_ids: list[str] = []
        unique_missing_positions: list[int] = []
        seen_missing: set[str] = set()

        for idx, image_id in enumerate(image_ids):
            if image_id in self.cache or image_id in seen_missing:
                continue
            seen_missing.add(image_id)
            unique_missing_ids.append(image_id)
            unique_missing_positions.append(idx)

        if unique_missing_positions:
            missing_images = images[unique_missing_positions].to(device, non_blocking=True)
            if channels_last and missing_images.ndim == 4:
                missing_images = missing_images.contiguous(memory_format=torch.channels_last)

            global_embedding, spatial_embedding = model.encode(missing_images)
            global_embedding = global_embedding.detach().cpu().half()
            spatial_embedding = spatial_embedding.detach().cpu().half()

            for idx, image_id in enumerate(unique_missing_ids):
                self._insert(image_id, (global_embedding[idx], spatial_embedding[idx]))

        global_stack = torch.stack([self.cache[image_id][0] for image_id in image_ids])
        spatial_stack = torch.stack([self.cache[image_id][1] for image_id in image_ids])

        return global_stack.to(device, non_blocking=True).float(), spatial_stack.to(device, non_blocking=True).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoPairNet with staged PairUAV pipeline")
    parser.add_argument("--config", type=str, default="configs/geopairnet_default.json")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--match-root", type=str, default=None)
    parser.add_argument("--match-index-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/geopairnet")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stages", type=str, default="A,B,C")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_amp_dtype(enabled: bool) -> torch.dtype | None:
    if not enabled or not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_grad_scaler(amp_dtype: torch.dtype | None) -> torch.cuda.amp.GradScaler:
    use_fp16 = amp_dtype == torch.float16
    return torch.cuda.amp.GradScaler(enabled=use_fp16)


def _stage_filter(stages: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    allowed = {item.strip().upper() for item in requested.split(",") if item.strip()}
    selected = [stage for stage in stages if stage["name"].upper() in allowed]
    if not selected:
        raise ValueError(f"No stages matched --stages={requested}")
    return selected


def _resolve_data_split(
    root: Path,
    dataset_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    mode = str(dataset_cfg.get("mode", "auto")).lower()
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    strict_official_only = bool(dataset_cfg.get("strict_official_only", False))
    allow_pseudo_warmup_only = bool(dataset_cfg.get("allow_pseudo_warmup_only", True))

    annotation_dir = resolve_train_annotation_dir(root)
    if strict_official_only and annotation_dir is None:
        raise FileNotFoundError(
            "strict_official_only=True but no official train annotations were found under "
            f"{root}."
        )

    if mode == "auto":
        mode = "official" if annotation_dir is not None else "pseudo"

    if strict_official_only and mode != "official":
        raise RuntimeError(
            "strict_official_only=True requires official dataset mode, but mode resolved to "
            f"{mode}."
        )

    if mode == "official":
        if annotation_dir is None:
            raise FileNotFoundError(f"Official mode requested but no annotations found under {root}")
        train_json, val_json = split_official_json_paths(annotation_dir, val_ratio=val_ratio, seed=seed)
        return {
            "mode": "official",
            "requested_mode": str(dataset_cfg.get("mode", "auto")).lower(),
            "strict_official_only": strict_official_only,
            "allow_pseudo_warmup_only": allow_pseudo_warmup_only,
            "annotation_source": str(annotation_dir),
            "train_json": train_json,
            "val_json": val_json,
        }

    view_dir = resolve_train_view_dir(root)
    buildings = sorted(path.name for path in view_dir.iterdir() if path.is_dir())
    if len(buildings) < 2:
        raise RuntimeError(f"Pseudo mode needs at least 2 building folders under {view_dir}")

    rng = random.Random(seed)
    rng.shuffle(buildings)

    split_idx = max(1, int(len(buildings) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(buildings) - 1)
    train_buildings = buildings[:split_idx]
    val_buildings = buildings[split_idx:]

    return {
        "mode": "pseudo",
        "requested_mode": str(dataset_cfg.get("mode", "auto")).lower(),
        "strict_official_only": strict_official_only,
        "allow_pseudo_warmup_only": allow_pseudo_warmup_only,
        "annotation_source": str(view_dir),
        "train_buildings": train_buildings,
        "val_buildings": val_buildings,
    }


def build_dataloaders(
    root: Path,
    dataset_cfg: dict[str, Any],
    split: dict[str, Any],
    stage_cfg: dict[str, Any],
    workers_override: int | None,
    match_root_override: str | None,
    match_index_override: str | None,
    seed: int,
    strict_official_only: bool,
    use_match_features: bool,
) -> tuple[DataLoader, DataLoader]:
    max_train_pairs = int(dataset_cfg.get("max_train_pairs", 960_000))
    max_val_pairs = int(dataset_cfg.get("max_val_pairs", 20_000))
    image_size = int(dataset_cfg.get("image_size", 224))

    train_augment = not bool(stage_cfg.get("disable_augmentation", False))

    match_root = match_root_override if match_root_override is not None else dataset_cfg.get("match_root")
    match_index = match_index_override if match_index_override is not None else dataset_cfg.get("match_index_file")
    if not use_match_features:
        match_root = None
        match_index = None

    if split["mode"] == "official":
        train_ds = PairUAVDataset(
            root=str(root),
            mode="official",
            json_paths=split["train_json"],
            max_pairs=max_train_pairs,
            seed=seed,
            image_size=image_size,
            augment=train_augment,
            is_val=False,
            match_root=match_root,
            match_index_file=match_index,
            strict_official_only=strict_official_only,
            split_name="train",
        )
        val_ds = PairUAVDataset(
            root=str(root),
            mode="official",
            json_paths=split["val_json"],
            max_pairs=max_val_pairs,
            seed=seed + 1,
            image_size=image_size,
            augment=False,
            is_val=True,
            match_root=match_root,
            match_index_file=match_index,
            strict_official_only=strict_official_only,
            split_name="val",
        )
    else:
        train_ds = PairUAVDataset(
            root=str(root),
            mode="pseudo",
            buildings=split["train_buildings"],
            max_pairs=max_train_pairs,
            seed=seed,
            image_size=image_size,
            augment=train_augment,
            is_val=False,
            match_root=match_root,
            match_index_file=match_index,
            strict_official_only=strict_official_only,
            split_name="train",
        )
        val_ds = PairUAVDataset(
            root=str(root),
            mode="pseudo",
            buildings=split["val_buildings"],
            max_pairs=max_val_pairs,
            seed=seed + 1,
            image_size=image_size,
            augment=False,
            is_val=True,
            match_root=match_root,
            match_index_file=match_index,
            strict_official_only=strict_official_only,
            split_name="val",
        )

    workers = workers_override if workers_override is not None else int(dataset_cfg.get("num_workers", 8))
    loader_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(dataset_cfg.get("pin_memory", True)),
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(dataset_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(dataset_cfg.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(stage_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(stage_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def _wrap_heading_deg(values: torch.Tensor) -> torch.Tensor:
    return ((values + 180.0) % 360.0) - 180.0


def _assert_target_encode_decode_roundtrip(meta: dict[str, torch.Tensor]) -> None:
    heading = meta["heading"].detach().float()
    distance = meta["distance"].detach().float().clamp(min=1e-6)

    encoded_sin = torch.sin(torch.deg2rad(heading))
    encoded_cos = torch.cos(torch.deg2rad(heading))
    decoded_heading = torch.rad2deg(torch.atan2(encoded_sin, encoded_cos))
    decoded_heading = _wrap_heading_deg(decoded_heading)

    heading_error = (decoded_heading - _wrap_heading_deg(heading)).abs()
    heading_error = torch.minimum(heading_error, 360.0 - heading_error)
    max_heading_error = float(heading_error.max().item())
    if max_heading_error > 1e-4:
        raise RuntimeError(
            "Heading encode/decode roundtrip failed: "
            f"max error {max_heading_error:.6f} deg"
        )

    encoded_log_distance = torch.log(distance)
    decoded_distance = torch.exp(encoded_log_distance)
    relative_error = (decoded_distance - distance).abs() / distance.clamp(min=1e-6)
    max_distance_rel_error = float(relative_error.max().item())
    if max_distance_rel_error > 1e-6:
        raise RuntimeError(
            "Distance encode/decode roundtrip failed: "
            f"max relative error {max_distance_rel_error:.6e}"
        )


def _assert_prediction_units(prediction: dict[str, torch.Tensor], context: str) -> None:
    heading_raw = prediction["heading_deg"].detach().float()
    distance = prediction["distance"].detach().float()

    if not torch.isfinite(heading_raw).all():
        raise RuntimeError(f"Non-finite heading prediction encountered during {context}")
    if not torch.isfinite(distance).all():
        raise RuntimeError(f"Non-finite distance prediction encountered during {context}")
    if (distance <= 0.0).any():
        minimum = float(distance.min().item())
        raise RuntimeError(
            f"Non-positive metric distance encountered during {context}: min={minimum:.6f}"
        )

    heading_wrapped = _wrap_heading_deg(heading_raw)
    min_heading = float(heading_wrapped.min().item())
    max_heading = float(heading_wrapped.max().item())
    if min_heading < -180.0001 or max_heading > 180.0001:
        raise RuntimeError(
            f"Decoded heading out of benchmark-compatible range during {context}: "
            f"[{min_heading:.4f}, {max_heading:.4f}]"
        )


def _print_validation_debug_samples(
    prediction: dict[str, torch.Tensor],
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    count = min(max_samples, int(prediction["heading_deg"].shape[0]))
    if count <= 0:
        return

    heading_sin = prediction.get("heading_sin")
    heading_cos = prediction.get("heading_cos")
    log_distance = prediction.get("log_distance")
    heading_deg = prediction["heading_deg"].detach().float()
    distance = prediction["distance"].detach().float()

    print("Validation decode debug samples:")
    for idx in range(count):
        sin_text = "n/a" if heading_sin is None else f"{float(heading_sin[idx].item()):.4f}"
        cos_text = "n/a" if heading_cos is None else f"{float(heading_cos[idx].item()):.4f}"
        log_dist_text = "n/a" if log_distance is None else f"{float(log_distance[idx].item()):.4f}"
        heading_text = float(_wrap_heading_deg(heading_deg[idx : idx + 1])[0].item())
        distance_text = float(distance[idx].item())
        print(
            f"  sample[{idx}] raw(sin={sin_text}, cos={cos_text}, log_d={log_dist_text}) "
            f"-> heading_deg={heading_text:.4f}, distance_m={distance_text:.4f}"
        )


def _log_dataset_debug(
    split: dict[str, Any],
    train_dataset: PairUAVDataset,
    val_dataset: PairUAVDataset,
    data_root: Path,
    seed: int,
) -> None:
    train_summary = train_dataset.describe()
    val_summary = val_dataset.describe()

    train_annotation_dir = resolve_train_annotation_dir(data_root)
    train_image_dir = resolve_train_view_dir(data_root)
    test_annotation_dir = resolve_test_annotation_dir(data_root)
    tour_annotation_dir: Path | None = None
    for candidate in (data_root / "test_tour", data_root / "tour"):
        if candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*.json")):
            tour_annotation_dir = candidate
            break

    test_image_dir: Path | None = None
    for candidate in (
        data_root / "test" / "drone",
        data_root / "test",
    ):
        if candidate.is_dir():
            test_image_dir = candidate
            break

    tour_image_dir: Path | None = None
    for candidate in (
        data_root / "test_tour" / "drone",
        data_root / "test_tour",
        data_root / "tour" / "drone",
        data_root / "tour",
    ):
        if candidate.is_dir():
            tour_image_dir = candidate
            break

    print("Resolved PairUAV layout:")
    print(f"  data_root={data_root}")
    print(f"  train_annotation_dir={train_annotation_dir}")
    print(f"  train_image_dir={train_image_dir}")
    print(f"  test_annotation_dir={test_annotation_dir}")
    print(f"  test_image_dir={test_image_dir}")
    print(f"  tour_annotation_dir={tour_annotation_dir}")
    print(f"  tour_image_dir={tour_image_dir}")

    print("Dataset summary:")
    print(
        f"  mode={split['mode']} (requested={split.get('requested_mode', 'auto')}) "
        f"strict_official_only={split.get('strict_official_only', False)}"
    )
    print(f"  annotation_sources(train)={train_summary['annotation_sources']}")
    print(f"  annotation_sources(val)={val_summary['annotation_sources']}")

    test_pair_count = count_official_test_pairs(data_root)
    print(
        f"  pair_counts train={train_summary['pair_count']} "
        f"val={val_summary['pair_count']} test={test_pair_count}"
    )

    preview_seed = seed + 17
    train_preview = train_dataset.sample_label_preview(k=5, seed=preview_seed)
    val_preview = val_dataset.sample_label_preview(k=5, seed=preview_seed + 1)

    print("  random decoded train label samples (heading_deg, distance):")
    for item in train_preview:
        print(
            f"    {item['pair_key']} -> "
            f"heading_deg={item['heading_deg']:.4f}, distance={item['distance']:.4f}"
        )

    print("  random decoded val label samples (heading_deg, distance):")
    for item in val_preview:
        print(
            f"    {item['pair_key']} -> "
            f"heading_deg={item['heading_deg']:.4f}, distance={item['distance']:.4f}"
        )

    test_preview = collect_official_test_pairs(data_root, limit=5)
    if test_preview:
        print("  first official test pair ids:")
        for item in test_preview:
            print(f"    {item.pair_id} -> {item.source_ref} || {item.target_ref}")


def _auto_match_root(data_root: Path, split_tag: str) -> str | None:
    candidates = (
        data_root / f"{split_tag}_matches_data",
        data_root / "matches" / f"{split_tag}_matches_data",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def _auto_match_index_file(match_root: str | None) -> str | None:
    if not match_root:
        return None

    root = Path(match_root)
    for name in ("index.csv", "match_index.csv", "matches.csv", "pairs.csv"):
        candidate = root / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _resolve_runtime_match_store(
    use_match_features: bool,
    model: GeoPairNet,
    data_root: Path,
    match_root: str | None,
    match_index_file: str | None,
    match_missing_policy: str,
) -> tuple[str | None, str | None, bool]:
    if not use_match_features:
        return None, None, False

    policy = str(match_missing_policy).lower().strip()
    if policy not in {"disable", "error"}:
        policy = "disable"

    if not match_root:
        message = (
            "Match features requested but no match_root was provided "
            "(expected train_matches_data/ for official pipeline)."
        )
        if policy == "error":
            raise FileNotFoundError(message)
        print(f"WARNING: {message} Entering no-match-features fallback mode.")
        return None, None, False

    match_root_path = Path(match_root).expanduser().resolve()
    if not match_root_path.is_dir():
        message = f"Match root does not exist: {match_root_path}"
        if policy == "error":
            raise FileNotFoundError(message)
        print(f"WARNING: {message}. Entering no-match-features fallback mode.")
        return None, None, False

    match_index_path: Path | None = None
    if match_index_file:
        match_index_path = Path(match_index_file).expanduser().resolve()
        if not match_index_path.is_file():
            message = f"Match index file does not exist: {match_index_path}"
            if policy == "error":
                raise FileNotFoundError(message)
            print(f"WARNING: {message}. Continuing without index file.")
            match_index_path = None

    store = OfflineMatchFeatureStore(
        match_root=str(match_root_path),
        index_file=str(match_index_path) if match_index_path is not None else None,
        feature_dim=int(getattr(model, "match_feature_dim", 8)),
    )

    official_pairs = collect_official_test_pairs(data_root, limit=5)
    missing = 0
    reverse_failures = 0
    reverse_checked = 0

    if official_pairs:
        print("Match alignment check (official test samples):")
    for pair in official_pairs:
        probe = store.probe(pair.source_ref, pair.target_ref)
        status = "FOUND" if probe.found else "MISSING"
        reverse_text = "reverse" if probe.used_reverse else "direct"
        print(f"  {pair.pair_id}: {status} ({reverse_text})")
        if not probe.found:
            missing += 1
            continue

        reverse_probe = store.probe(pair.target_ref, pair.source_ref)
        if reverse_probe.found and (probe.used_reverse ^ reverse_probe.used_reverse):
            forward_feat = store.get(pair.source_ref, pair.target_ref)
            reverse_feat = store.get(pair.target_ref, pair.source_ref)
            reverse_checked += 1
            if abs(float(forward_feat[2] + reverse_feat[2])) > 2e-2:
                reverse_failures += 1
            if abs(float(forward_feat[3] + reverse_feat[3])) > 2e-2:
                reverse_failures += 1

    if reverse_checked > 0:
        print(
            f"  reverse-pair consistency checks: {reverse_checked} samples, "
            f"failures={reverse_failures}"
        )

    if missing > 0 or reverse_failures > 0:
        message = f"Match alignment failed (missing={missing}, reverse_failures={reverse_failures})"
        if policy == "error":
            raise RuntimeError(message)
        print(f"WARNING: {message}. Entering no-match-features fallback mode.")
        return None, None, False

    print(
        "Match features enabled for training: "
        f"root={match_root_path}, index={match_index_path}"
    )
    return str(match_root_path), (str(match_index_path) if match_index_path is not None else None), True


def _should_enable_feature_cache(stage_cfg: dict[str, Any], train_augment: bool) -> tuple[bool, str]:
    if not bool(stage_cfg.get("feature_cache", False)):
        return False, "disabled"

    backbone_mode = str(stage_cfg.get("backbone_mode", "full")).lower()
    if backbone_mode != "frozen":
        return False, "requires backbone_mode=frozen"

    if train_augment:
        return False, "requires disable_augmentation=true for deterministic cached inputs"

    return True, "enabled"


@torch.no_grad()
def _assert_cached_vs_uncached_equivalence(
    model: GeoPairNet,
    batch: tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
    device: torch.device,
    channels_last: bool,
    tolerance: float,
) -> None:
    source, target, meta = batch
    source, target, meta = _to_device_batch(source, target, meta, device, channels_last)

    previous_mode = model.training
    model.eval()
    direct_pred = model(
        source,
        target,
        match_features=meta["match_features"],
        geometry_features=meta["geometry_features"],
    )

    feature_cache = FrozenFeatureCache(max_items=max(128, source.shape[0] * 4))
    source_global, source_spatial = feature_cache.encode(
        model,
        source,
        meta["source_id"],
        device=device,
        channels_last=channels_last,
    )
    target_global, target_spatial = feature_cache.encode(
        model,
        target,
        meta["target_id"],
        device=device,
        channels_last=channels_last,
    )
    cached_pred = model.forward_from_embeddings(
        source_global=source_global,
        source_spatial=source_spatial,
        target_global=target_global,
        target_spatial=target_spatial,
        match_features=meta["match_features"],
        geometry_features=meta["geometry_features"],
    )

    worst_diff = 0.0
    worst_name = ""
    for key in ("heading_deg", "distance", "log_distance"):
        direct_value = direct_pred[key].detach().float()
        cached_value = cached_pred[key].detach().float()
        max_abs_diff = float((direct_value - cached_value).abs().max().item())
        if max_abs_diff > worst_diff:
            worst_diff = max_abs_diff
            worst_name = key

    if previous_mode:
        model.train()

    if worst_diff > tolerance:
        raise RuntimeError(
            "Cached vs uncached equivalence failed: "
            f"max abs diff {worst_diff:.6e} on '{worst_name}' exceeds tolerance {tolerance:.6e}"
        )


def _to_device_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    meta: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = source.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    if channels_last and source.ndim == 4:
        source = source.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)

    batch_meta = {
        "heading": meta["heading"].to(device, non_blocking=True),
        "distance": meta["distance"].to(device, non_blocking=True),
        "match_features": meta["match_features"].to(device, non_blocking=True),
        "geometry_features": meta["geometry_features"].to(device, non_blocking=True),
        "source_id": [str(item) for item in meta["source_id"]],
        "target_id": [str(item) for item in meta["target_id"]],
    }
    return source, target, batch_meta


def train_one_epoch(
    model: GeoPairNet,
    loader: DataLoader,
    criterion: PairUAVLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
    stage_name: str,
    global_epoch_idx: int,
    total_epochs: int,
    runtime_state: StageRuntimeState,
    feature_cache: FrozenFeatureCache | None,
    ema: ModelEMA | None,
) -> dict[str, float]:
    model.train()

    loss_total = 0.0
    loss_rotation = 0.0
    loss_distance = 0.0
    weight_rotation = 0.0
    weight_distance = 0.0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        source, target, meta = batch
        source, target, meta = _to_device_batch(source, target, meta, device, channels_last)

        progress = (global_epoch_idx + float(step + 1) / max(1, len(loader))) / max(1, total_epochs)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            if feature_cache is not None:
                source_global, source_spatial = feature_cache.encode(
                    model,
                    source,
                    meta["source_id"],
                    device=device,
                    channels_last=channels_last,
                )
                target_global, target_spatial = feature_cache.encode(
                    model,
                    target,
                    meta["target_id"],
                    device=device,
                    channels_last=channels_last,
                )
                prediction = model.forward_from_embeddings(
                    source_global=source_global,
                    source_spatial=source_spatial,
                    target_global=target_global,
                    target_spatial=target_spatial,
                    match_features=meta["match_features"],
                    geometry_features=meta["geometry_features"],
                )
            else:
                prediction = model(
                    source,
                    target,
                    match_features=meta["match_features"],
                    geometry_features=meta["geometry_features"],
                )

            losses = criterion(
                prediction=prediction,
                target={"heading": meta["heading"], "distance": meta["distance"]},
                progress=progress,
                stage_name=stage_name,
            )

            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"Encountered non-finite training loss at stage={stage_name} step={step}."
                )

            scaled_loss = losses["total"] / grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader)
        if should_step:
            lr = scheduler.step(runtime_state.optimizer_steps)
            _ = lr

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            grad_norm_sum += grad_norm_value
            grad_norm_max = max(grad_norm_max, grad_norm_value)
            grad_norm_steps += 1

            optimizer.zero_grad(set_to_none=True)
            runtime_state.optimizer_steps += 1

            if ema is not None:
                ema.update(model)

        loss_total += losses["total"].item()
        loss_rotation += losses["rotation"].item()
        loss_distance += losses["distance"].item()
        weight_rotation += losses["weight_rotation"].item()
        weight_distance += losses["weight_distance"].item()

    num_steps = max(1, len(loader))
    return {
        "train_total_loss": loss_total / num_steps,
        "train_rotation_loss": loss_rotation / num_steps,
        "train_distance_loss": loss_distance / num_steps,
        "train_weight_rotation": weight_rotation / num_steps,
        "train_weight_distance": weight_distance / num_steps,
        "train_grad_norm": grad_norm_sum / max(1, grad_norm_steps),
        "train_grad_norm_max": grad_norm_max,
    }


@torch.no_grad()
def validate_one_epoch(
    model: GeoPairNet,
    loader: DataLoader,
    criterion: PairUAVLoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    stage_name: str,
    global_epoch_idx: int,
    total_epochs: int,
    feature_cache: FrozenFeatureCache | None,
    debug_samples: int = 0,
    file_result_path: Path | None = None,
    file_truth_path: Path | None = None,
    file_metric_parity_tolerance: float = 1e-6,
) -> dict[str, float]:
    model.eval()

    val_loss = 0.0
    pred_heading: list[torch.Tensor] = []
    pred_distance: list[torch.Tensor] = []
    target_heading: list[torch.Tensor] = []
    target_distance: list[torch.Tensor] = []

    for step, batch in enumerate(loader):
        source, target, meta = batch
        source, target, meta = _to_device_batch(source, target, meta, device, channels_last)

        if step == 0:
            _assert_target_encode_decode_roundtrip(meta)

        progress = (global_epoch_idx + float(step + 1) / max(1, len(loader))) / max(1, total_epochs)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            if feature_cache is not None:
                source_global, source_spatial = feature_cache.encode(
                    model,
                    source,
                    meta["source_id"],
                    device=device,
                    channels_last=channels_last,
                )
                target_global, target_spatial = feature_cache.encode(
                    model,
                    target,
                    meta["target_id"],
                    device=device,
                    channels_last=channels_last,
                )
                prediction = model.forward_from_embeddings(
                    source_global=source_global,
                    source_spatial=source_spatial,
                    target_global=target_global,
                    target_spatial=target_spatial,
                    match_features=meta["match_features"],
                    geometry_features=meta["geometry_features"],
                )
            else:
                prediction = model(
                    source,
                    target,
                    match_features=meta["match_features"],
                    geometry_features=meta["geometry_features"],
                )

            losses = criterion(
                prediction=prediction,
                target={"heading": meta["heading"], "distance": meta["distance"]},
                progress=progress,
                stage_name=stage_name,
            )

        _assert_prediction_units(prediction, context=f"validation stage={stage_name} step={step}")
        if step == 0 and debug_samples > 0:
            _print_validation_debug_samples(prediction, max_samples=debug_samples)

        if not torch.isfinite(losses["total"]):
            raise RuntimeError(
                f"Encountered non-finite validation loss at stage={stage_name} step={step}."
            )

        val_loss += losses["total"].item()
        pred_heading.append(prediction["heading_deg"].detach().cpu())
        pred_distance.append(prediction["distance"].detach().cpu())
        target_heading.append(meta["heading"].detach().cpu())
        target_distance.append(meta["distance"].detach().cpu())

    pred = {
        "heading": torch.cat(pred_heading, dim=0),
        "distance": torch.cat(pred_distance, dim=0),
    }
    target = {
        "heading": torch.cat(target_heading, dim=0),
        "distance": torch.cat(target_distance, dim=0),
    }

    metrics = comprehensive_metrics(pred, target)
    metrics["val_total_loss"] = val_loss / max(1, len(loader))

    if file_result_path is not None and file_truth_path is not None:
        write_result_file(file_result_path, pred["heading"], pred["distance"], delimiter="comma")
        write_result_file(file_truth_path, target["heading"], target["distance"], delimiter="comma")
        file_metrics = evaluate_result_files(file_result_path, file_truth_path)

        parity_keys = ("angle_rel_error", "distance_rel_error", "final_score")
        max_delta = 0.0
        for key in parity_keys:
            tensor_value = float(metrics[key])
            file_value = float(file_metrics[key])
            delta = abs(tensor_value - file_value)
            max_delta = max(max_delta, delta)
            metrics[f"file_{key}"] = file_value

        metrics["file_metric_parity_max_delta"] = max_delta
        if max_delta > file_metric_parity_tolerance:
            print(
                "WARNING: file metric parity mismatch "
                f"max_delta={max_delta:.3e} tol={file_metric_parity_tolerance:.3e}"
            )

    return metrics


def save_checkpoint(
    path: Path,
    model: GeoPairNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    ema: ModelEMA | None,
    config: dict[str, Any],
    epoch: int,
    stage_name: str,
    metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "stage": stage_name,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.shadow if ema is not None else None,
        "model_config": config.get("model", {}),
        "full_config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_cfg = dict(config.get("dataset", {}))
    model_cfg = dict(config.get("model", {}))
    loss_cfg = dict(config.get("loss", {}))
    training_cfg = dict(config.get("training", {}))

    safe_submission_mode = bool(training_cfg.get("safe_submission_mode", False))
    safe_baseline_mode = bool(training_cfg.get("safe_baseline_mode", False))
    safe_overrides: list[str] = []
    if safe_submission_mode or safe_baseline_mode:
        if safe_submission_mode:
            print("safe_submission_mode enabled")
        else:
            print("safe_baseline_mode enabled (legacy key; prefer safe_submission_mode)")

        dataset_cfg["mode"] = "official"
        dataset_cfg["strict_official_only"] = True
        dataset_cfg["allow_pseudo_warmup_only"] = False
        dataset_cfg["match_root"] = None
        dataset_cfg["match_index_file"] = None
        safe_overrides.extend(
            [
                "dataset.mode=official",
                "dataset.strict_official_only=True",
                "dataset.allow_pseudo_warmup_only=False",
                "dataset.match_root=None",
                "dataset.match_index_file=None",
            ]
        )

        training_cfg["deterministic"] = True
        training_cfg["use_ema"] = False
        safe_overrides.extend(
            [
                "training.deterministic=True",
                "training.use_ema=False",
            ]
        )

        if safe_baseline_mode and not safe_submission_mode:
            model_cfg["no_match_features"] = True
            safe_overrides.append("model.no_match_features=True")

        for stage in config.get("stages", []):
            stage["feature_cache"] = False
            stage["disable_augmentation"] = True
        safe_overrides.extend(
            [
                "stages[*].feature_cache=False",
                "stages[*].disable_augmentation=True",
            ]
        )

        print("safe mode overrides:")
        for item in safe_overrides:
            print(f"  - {item}")

        training_cfg["safe_submission_mode"] = True

    if bool(model_cfg.pop("no_uncertainty", False)):
        model_cfg["use_uncertainty"] = False

    config["dataset"] = dataset_cfg
    config["model"] = model_cfg
    config["loss"] = loss_cfg
    config["training"] = training_cfg

    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 42))
    deterministic = bool(training_cfg.get("deterministic", False))
    set_seed(seed=seed, deterministic=deterministic)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")

    split = _resolve_data_split(data_root, dataset_cfg, seed=seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(enabled=bool(training_cfg.get("amp", True)))
    scaler = make_grad_scaler(amp_dtype)

    model = GeoPairNet(**model_cfg).to(device)

    channels_last = bool(training_cfg.get("channels_last", True))
    model.summary()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded checkpoint: {args.resume}")

    distance_cls_weight = float(loss_cfg.get("distance_cls_weight", 0.35))
    if bool(model_cfg.get("no_distance_bins", False)):
        distance_cls_weight = 0.0

    criterion = PairUAVLoss(
        log_distance_min=float(loss_cfg.get("log_distance_min", model_cfg.get("log_distance_min", 0.0))),
        log_distance_max=float(loss_cfg.get("log_distance_max", model_cfg.get("log_distance_max", 5.0))),
        num_bins=int(loss_cfg.get("distance_bins", model_cfg.get("distance_bins", 24))),
        min_distance=float(loss_cfg.get("min_distance", 1.0)),
        smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 0.05)),
        distance_cls_weight=distance_cls_weight,
    )

    all_stages = config.get("stages", [])
    if not all_stages:
        raise RuntimeError("Config must include a non-empty 'stages' list.")
    stages = _stage_filter(all_stages, args.stages)

    total_epochs = sum(int(stage["epochs"]) for stage in stages)
    global_epoch_idx = 0

    ema = ModelEMA(model, decay=float(training_cfg.get("ema_decay", 0.999))) if bool(training_cfg.get("use_ema", True)) else None
    if args.resume and ema is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("ema_state_dict"):
            ema.shadow = checkpoint["ema_state_dict"]

    best_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []

    optimizer_defaults = config.get("optimizer", {})
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    strict_official_only = bool(split.get("strict_official_only", dataset_cfg.get("strict_official_only", False)))
    allow_pseudo_warmup_only = bool(
        split.get("allow_pseudo_warmup_only", dataset_cfg.get("allow_pseudo_warmup_only", True))
    )
    use_match_features = not bool(model_cfg.get("no_match_features", False))

    match_missing_policy = str(dataset_cfg.get("match_missing_policy", "disable"))
    runtime_match_root = args.match_root if args.match_root is not None else dataset_cfg.get("match_root")
    runtime_match_index = args.match_index_file if args.match_index_file is not None else dataset_cfg.get("match_index_file")
    if use_match_features and runtime_match_root is None:
        runtime_match_root = _auto_match_root(data_root, split_tag="train")
        if runtime_match_root is not None:
            print(f"Detected train match root: {runtime_match_root}")
    if use_match_features and runtime_match_index is None:
        runtime_match_index = _auto_match_index_file(runtime_match_root)
        if runtime_match_index is not None:
            print(f"Detected train match index: {runtime_match_index}")

    runtime_match_root, runtime_match_index, use_match_features = _resolve_runtime_match_store(
        use_match_features=use_match_features,
        model=model,
        data_root=data_root,
        match_root=runtime_match_root,
        match_index_file=runtime_match_index,
        match_missing_policy=match_missing_policy,
    )

    val_debug_samples = int(training_cfg.get("validation_debug_samples", 0))
    val_debug_every_epoch = bool(training_cfg.get("validation_debug_every_epoch", False))
    cache_equivalence_check = bool(training_cfg.get("cache_equivalence_check", True))
    cache_equivalence_tolerance = float(training_cfg.get("cache_equivalence_tolerance", 1e-4))
    file_metric_parity_check = bool(training_cfg.get("file_metric_parity_check", False))
    file_metric_parity_tolerance = float(training_cfg.get("file_metric_parity_tolerance", 1e-6))
    file_metric_parity_dir = output_dir / "_metric_parity" if file_metric_parity_check else None
    if file_metric_parity_dir is not None:
        file_metric_parity_dir.mkdir(parents=True, exist_ok=True)

    dataset_debug_logged = False

    for stage in stages:
        stage_name = str(stage["name"]).upper()
        stage_epochs = int(stage["epochs"])
        stage_lr = float(stage["lr"])
        stage_min_lr = float(stage.get("min_lr", stage_lr * 0.05))
        stage_warmup_epochs = int(stage.get("warmup_epochs", 1))

        if split["mode"] == "pseudo" and allow_pseudo_warmup_only and stage_name != "A":
            raise RuntimeError(
                "Pseudo mode is enabled with allow_pseudo_warmup_only=True, so only stage A is allowed. "
                f"Got stage {stage_name}."
            )

        model.set_backbone_trainable(str(stage.get("backbone_mode", "full")))
        model.set_shared_projectors_trainable(bool(stage.get("train_projectors", True)))

        train_loader, val_loader = build_dataloaders(
            root=data_root,
            dataset_cfg=dataset_cfg,
            split=split,
            stage_cfg=stage,
            workers_override=args.workers,
            match_root_override=runtime_match_root,
            match_index_override=runtime_match_index,
            seed=seed,
            strict_official_only=strict_official_only,
            use_match_features=use_match_features,
        )

        train_dataset = cast(PairUAVDataset, train_loader.dataset)
        val_dataset = cast(PairUAVDataset, val_loader.dataset)

        if not dataset_debug_logged:
            _log_dataset_debug(
                split=split,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_root=data_root,
                seed=seed,
            )
            dataset_debug_logged = True

        trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=stage_lr,
            betas=tuple(optimizer_defaults.get("betas", [0.9, 0.999])),
            weight_decay=float(stage.get("weight_decay", optimizer_defaults.get("weight_decay", 1e-4))),
        )

        updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
        stage_total_updates = updates_per_epoch * stage_epochs
        warmup_steps = updates_per_epoch * stage_warmup_epochs
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            base_lr=stage_lr,
            min_lr=stage_min_lr,
            total_steps=stage_total_updates,
            warmup_steps=warmup_steps,
        )

        runtime_state = StageRuntimeState(optimizer_steps=0)
        patience = int(stage.get("early_stop_patience", 0))
        stale = 0

        train_augment = not bool(stage.get("disable_augmentation", False))
        cache_enabled, cache_reason = _should_enable_feature_cache(stage, train_augment=train_augment)
        if bool(stage.get("feature_cache", False)) and not cache_enabled:
            print(f"  * Feature cache disabled for stage {stage_name}: {cache_reason}")

        feature_cache_size = int(stage.get("feature_cache_size", 60_000))
        train_feature_cache = FrozenFeatureCache(max_items=feature_cache_size) if cache_enabled else None
        val_feature_cache = FrozenFeatureCache(max_items=feature_cache_size) if cache_enabled else None

        if cache_enabled and cache_equivalence_check and len(train_loader) > 0:
            sample_batch = next(iter(train_loader))
            _assert_cached_vs_uncached_equivalence(
                model=model,
                batch=sample_batch,
                device=device,
                channels_last=channels_last,
                tolerance=cache_equivalence_tolerance,
            )
            print(
                f"  * Cache equivalence check passed for stage {stage_name} "
                f"(tol={cache_equivalence_tolerance:.1e})"
            )

        print(
            f"\n[Stage {stage_name}] epochs={stage_epochs} lr={stage_lr:.2e} "
            f"batch={int(stage['batch_size'])} backbone_mode={stage.get('backbone_mode', 'full')} "
            f"cache={'on' if cache_enabled else 'off'} match_features={use_match_features} "
            f"train_pairs={len(train_dataset)} val_pairs={len(val_dataset)}"
        )

        for local_epoch in range(stage_epochs):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
                stage_name=stage_name,
                global_epoch_idx=global_epoch_idx,
                total_epochs=total_epochs,
                runtime_state=runtime_state,
                feature_cache=train_feature_cache,
                ema=ema,
            )

            if ema is not None:
                ema.store(model)
                ema.copy_to(model)

            epoch_number = global_epoch_idx + 1
            file_result_path = None
            file_truth_path = None
            if file_metric_parity_dir is not None:
                file_result_path = file_metric_parity_dir / f"stage_{stage_name.lower()}_epoch_{epoch_number:03d}_pred.txt"
                file_truth_path = file_metric_parity_dir / f"stage_{stage_name.lower()}_epoch_{epoch_number:03d}_truth.txt"

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
                stage_name=stage_name,
                global_epoch_idx=global_epoch_idx,
                total_epochs=total_epochs,
                feature_cache=val_feature_cache,
                debug_samples=val_debug_samples if (val_debug_every_epoch or local_epoch == 0) else 0,
                file_result_path=file_result_path,
                file_truth_path=file_truth_path,
                file_metric_parity_tolerance=file_metric_parity_tolerance,
            )

            if ema is not None:
                ema.restore(model)

            elapsed = time.time() - epoch_start
            epoch_report = {
                "epoch": global_epoch_idx + 1,
                "stage": stage_name,
                **train_metrics,
                **val_metrics,
                "elapsed_sec": elapsed,
            }
            history.append(epoch_report)

            print(
                f"Epoch {global_epoch_idx + 1:03d} [{stage_name}] ({elapsed:.0f}s) "
                f"train_total={train_metrics['train_total_loss']:.4f} "
                f"val_total={val_metrics['val_total_loss']:.4f} "
                f"grad_norm={train_metrics['train_grad_norm']:.4f} "
                f"grad_norm_max={train_metrics['train_grad_norm_max']:.4f} "
                f"angle_mae={val_metrics['mae_heading']:.3f} "
                f"distance_mae={val_metrics['mae_distance']:.3f} "
                f"angle_rel={val_metrics['angle_rel_error']:.4f} "
                f"distance_rel={val_metrics['distance_rel_error']:.4f} "
                f"final_score={val_metrics['final_score']:.4f}"
            )

            save_checkpoint(
                path=output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                config=config,
                epoch=global_epoch_idx + 1,
                stage_name=stage_name,
                metrics=epoch_report,
            )

            if is_better_result(val_metrics, best_metrics):
                best_metrics = val_metrics
                stale = 0
                save_checkpoint(
                    path=output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    config=config,
                    epoch=global_epoch_idx + 1,
                    stage_name=stage_name,
                    metrics=epoch_report,
                )
                print("  * Updated best checkpoint")
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    print(f"  * Early stopping Stage {stage_name} (patience={patience})")
                    global_epoch_idx += 1
                    break

            global_epoch_idx += 1

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best_metrics is None:
        raise RuntimeError("Training finished but no validation metrics were produced.")

    print("\nTraining completed.")
    print(
        f"Best final_score={best_metrics['final_score']:.4f} "
        f"distance_rel_error={best_metrics['distance_rel_error']:.4f} "
        f"angle_rel_error={best_metrics['angle_rel_error']:.4f}"
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()












