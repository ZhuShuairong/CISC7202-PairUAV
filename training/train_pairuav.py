from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import resolve_train_annotation_dir, resolve_train_view_dir
from data.dataset_pairuav import PairUAVDataset, split_official_json_paths
from models.geopairnet import GeoPairNet
from training.losses import LossWeightConfig, PairUAVLoss
from utils.metrics import comprehensive_metrics, is_better_result


@dataclass
class StageRuntimeState:
    optimizer_steps: int = 0


@dataclass(frozen=True)
class AblationFlags:
    no_match_features: bool = False
    no_geometry_features: bool = False
    no_distance_bins: bool = False
    no_uncertainty: bool = False


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
    parser.add_argument(
        "--safe-baseline-mode",
        action="store_true",
        help=(
            "Correctness-debug mode: official labels only, no cache, no EMA, no match summaries, "
            "deterministic behavior"
        ),
    )
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


def _resolve_ablation_flags(ablation_cfg: dict[str, Any]) -> AblationFlags:
    return AblationFlags(
        no_match_features=bool(ablation_cfg.get("no_match_features", False)),
        no_geometry_features=bool(ablation_cfg.get("no_geometry_features", False)),
        no_distance_bins=bool(ablation_cfg.get("no_distance_bins", False)),
        no_uncertainty=bool(ablation_cfg.get("no_uncertainty", False)),
    )


def _apply_safe_baseline_mode(config: dict[str, Any]) -> None:
    dataset_cfg = config.setdefault("dataset", {})
    training_cfg = config.setdefault("training", {})

    dataset_cfg["mode"] = "official"
    dataset_cfg["strict_official_only"] = True
    dataset_cfg["match_root"] = None
    dataset_cfg["match_index_file"] = None

    training_cfg["use_ema"] = False
    training_cfg["deterministic"] = True

    for stage in config.get("stages", []):
        stage["feature_cache"] = False
        stage["disable_augmentation"] = True


def _stage_rank(stage_name: str) -> int:
    order = {"A": 1, "B": 2, "C": 3}
    return order.get(stage_name.upper().strip(), 999)


def _count_expected_test_pairs(data_root: Path) -> int | None:
    try:
        from scripts.generate_submission import _discover_pairs

        pairs, _ = _discover_pairs(data_root, pair_order="official")
        return len(pairs)
    except Exception:
        return None


def _print_dataset_diagnostics(
    *,
    train_dataset: PairUAVDataset,
    val_dataset: PairUAVDataset,
    split: dict[str, Any],
    dataset_cfg: dict[str, Any],
    seed: int,
    expected_test_pairs: int | None,
) -> None:
    requested_mode = str(dataset_cfg.get("mode", "auto")).lower()
    strict_only = bool(dataset_cfg.get("strict_official_only", False))
    print(
        "[DatasetDiagnostics] "
        f"mode_requested={requested_mode} mode_resolved={split['mode']} strict_official_only={strict_only}"
    )

    print(
        "[DatasetDiagnostics] "
        f"pair_counts train={len(train_dataset)} val={len(val_dataset)} "
        f"test={expected_test_pairs if expected_test_pairs is not None else 'unknown'}"
    )

    train_diag = train_dataset.diagnostics(sample_count=3, seed=seed)
    annotation_dir = train_diag["annotation_dir"]
    print(f"[DatasetDiagnostics] annotation_dir={annotation_dir}")

    annotation_sources = train_diag["annotation_sources"]
    if annotation_sources:
        print("[DatasetDiagnostics] annotation_sources_preview=")
        for path in annotation_sources:
            print(f"  - {path}")
    else:
        print("[DatasetDiagnostics] annotation_sources_preview=<none>")

    print("[DatasetDiagnostics] decoded_label_samples=")
    for sample in train_diag["label_samples"]:
        print(
            "  - "
            f"idx={sample['index']} "
            f"src={sample['source_id']} "
            f"tgt={sample['target_id']} "
            f"heading_deg={sample['heading_deg']:.3f} "
            f"distance_m={sample['distance_m']:.3f}"
        )


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
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))

    annotation_dir = resolve_train_annotation_dir(root)
    if annotation_dir is None:
        raise FileNotFoundError(f"Official mode requested but no annotations found under {root}")
    train_json, val_json = split_official_json_paths(annotation_dir, val_ratio=val_ratio, seed=seed)
    return {
        "mode": "official",
        "annotation_dir": annotation_dir,
        "train_json": train_json,
        "val_json": val_json,
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
) -> tuple[DataLoader, DataLoader]:
    max_train_pairs = int(dataset_cfg.get("max_train_pairs", 960_000))
    max_val_pairs = int(dataset_cfg.get("max_val_pairs", 20_000))
    image_size = int(dataset_cfg.get("image_size", 224))

    train_augment = not bool(stage_cfg.get("disable_augmentation", False))

    match_root = match_root_override if match_root_override is not None else dataset_cfg.get("match_root")
    match_index = match_index_override if match_index_override is not None else dataset_cfg.get("match_index_file")

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


def _to_device_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    meta: dict[str, Any],
    device: torch.device,
    channels_last: bool,
    ablation: AblationFlags,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = source.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    if channels_last and source.ndim == 4:
        source = source.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)

    match_features = meta["match_features"].to(device, non_blocking=True)
    geometry_features = meta["geometry_features"].to(device, non_blocking=True)

    if ablation.no_match_features:
        match_features = torch.zeros_like(match_features)
    if ablation.no_geometry_features:
        geometry_features = torch.zeros_like(geometry_features)

    batch_meta = {
        "heading": meta["heading"].to(device, non_blocking=True),
        "distance": meta["distance"].to(device, non_blocking=True),
        "match_features": match_features,
        "geometry_features": geometry_features,
        "source_id": [str(item) for item in meta["source_id"]],
        "target_id": [str(item) for item in meta["target_id"]],
    }
    return source, target, batch_meta


def _assert_prediction_units(prediction: dict[str, torch.Tensor], context: str) -> None:
    heading = prediction["heading_deg"]
    distance = prediction["distance"]

    if not torch.isfinite(heading).all() or not torch.isfinite(distance).all():
        raise RuntimeError(f"Non-finite heading/distance predictions in {context}")

    wrapped_heading = ((heading + 180.0) % 360.0) - 180.0
    if torch.any(wrapped_heading < -180.0001) or torch.any(wrapped_heading > 180.0001):
        raise RuntimeError(f"Heading decode out of range in {context}")

    if torch.any(distance < 0.0):
        raise RuntimeError(f"Distance decode produced negative values in {context}")

    export_heading = ((heading + 180.0) % 360.0) - 180.0
    export_distance = torch.clamp(distance, min=1e-6)
    if torch.any(export_heading < -180.0001) or torch.any(export_heading > 180.0001):
        raise RuntimeError(f"Heading export range check failed in {context}")
    if torch.any(export_distance <= 0.0):
        raise RuntimeError(f"Distance export positivity check failed in {context}")

    if "log_distance" in prediction:
        reconstructed = torch.exp(prediction["log_distance"])
        max_abs = float((reconstructed - distance).abs().max().item())
        if max_abs > 1e-3:
            raise RuntimeError(
                "Distance decode inconsistency: exp(log_distance) does not match distance "
                f"in {context} (max_abs_diff={max_abs:.6f})"
            )


def _print_decode_debug(
    prediction: dict[str, torch.Tensor],
    target_heading: torch.Tensor,
    target_distance: torch.Tensor,
    sample_count: int,
) -> None:
    count = min(sample_count, prediction["heading_deg"].shape[0])
    if count <= 0:
        return

    heading_sin = prediction.get("heading_sin")
    heading_cos = prediction.get("heading_cos")
    log_distance = prediction.get("log_distance")

    print("[DecodeDebug] samples=")
    for idx in range(count):
        sin_value = float(heading_sin[idx].item()) if heading_sin is not None else float("nan")
        cos_value = float(heading_cos[idx].item()) if heading_cos is not None else float("nan")
        log_d_value = float(log_distance[idx].item()) if log_distance is not None else float("nan")
        heading_deg = float(prediction["heading_deg"][idx].item())
        distance_m = float(prediction["distance"][idx].item())
        tgt_heading = float(target_heading[idx].item())
        tgt_distance_m = float(target_distance[idx].item())

        print(
            "  - "
            f"raw_sin={sin_value:.5f} raw_cos={cos_value:.5f} raw_log_distance={log_d_value:.5f} "
            f"decoded_heading_deg={heading_deg:.5f} decoded_distance_m={distance_m:.5f} "
            f"target_heading_deg={tgt_heading:.5f} target_distance_m={tgt_distance_m:.5f}"
        )


@torch.no_grad()
def _assert_cached_vs_uncached_equivalence(
    model: GeoPairNet,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    ablation: AblationFlags,
    tolerance: float,
) -> None:
    if len(loader) == 0:
        return

    batch = next(iter(loader))
    source, target, meta = batch
    source, target, meta = _to_device_batch(source, target, meta, device, channels_last, ablation)

    model.eval()
    uncached = model(
        source,
        target,
        match_features=meta["match_features"],
        geometry_features=meta["geometry_features"],
    )

    cache = FrozenFeatureCache(max_items=max(4, source.shape[0] * 2))
    source_global, source_spatial = cache.encode(
        model,
        source,
        meta["source_id"],
        device=device,
        channels_last=channels_last,
    )
    target_global, target_spatial = cache.encode(
        model,
        target,
        meta["target_id"],
        device=device,
        channels_last=channels_last,
    )
    cached = model.forward_from_embeddings(
        source_global=source_global,
        source_spatial=source_spatial,
        target_global=target_global,
        target_spatial=target_spatial,
        match_features=meta["match_features"],
        geometry_features=meta["geometry_features"],
    )

    heading_diff = float((uncached["heading_deg"] - cached["heading_deg"]).abs().max().item())
    distance_diff = float((uncached["distance"] - cached["distance"]).abs().max().item())
    max_diff = max(heading_diff, distance_diff)

    if max_diff > tolerance:
        raise RuntimeError(
            "Cached-vs-uncached mismatch exceeded tolerance: "
            f"max_diff={max_diff:.6f}, heading_diff={heading_diff:.6f}, "
            f"distance_diff={distance_diff:.6f}, tolerance={tolerance:.6f}"
        )

    print(
        "[CacheCheck] cached_vs_uncached passed: "
        f"heading_diff={heading_diff:.6f} distance_diff={distance_diff:.6f} tolerance={tolerance:.6f}"
    )


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
    ablation: AblationFlags,
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
        source, target, meta = _to_device_batch(source, target, meta, device, channels_last, ablation)

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

            _assert_prediction_units(prediction, context=f"train:{stage_name}:step={step}")
            if not torch.isfinite(losses["total"]).all():
                raise RuntimeError(
                    f"Non-finite training loss detected at stage={stage_name} step={step}: "
                    f"loss={losses['total'].detach().cpu().item()}"
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
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at stage={stage_name} step={step}")
                grad_norm_sum += grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
                grad_norm_steps += 1
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at stage={stage_name} step={step}")
                grad_norm_sum += grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
                grad_norm_steps += 1
                optimizer.step()

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
        "train_grad_norm_mean": grad_norm_sum / max(1, grad_norm_steps),
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
    ablation: AblationFlags,
    debug_decode_samples: int,
) -> dict[str, float]:
    model.eval()

    val_loss = 0.0
    pred_heading: list[torch.Tensor] = []
    pred_distance: list[torch.Tensor] = []
    target_heading: list[torch.Tensor] = []
    target_distance: list[torch.Tensor] = []

    for step, batch in enumerate(loader):
        source, target, meta = batch
        source, target, meta = _to_device_batch(source, target, meta, device, channels_last, ablation)

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

        if not torch.isfinite(losses["total"]).all():
            raise RuntimeError(
                f"Non-finite validation loss detected at stage={stage_name} step={step}: "
                f"loss={losses['total'].detach().cpu().item()}"
            )

        _assert_prediction_units(prediction, context=f"val:{stage_name}:step={step}")
        if step == 0 and debug_decode_samples > 0:
            _print_decode_debug(
                prediction=prediction,
                target_heading=meta["heading"],
                target_distance=meta["distance"],
                sample_count=debug_decode_samples,
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

    if args.safe_baseline_mode:
        _apply_safe_baseline_mode(config)
        print("Safe baseline mode is enabled.")

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    ablation = _resolve_ablation_flags(config.get("ablation", {}))

    print(
        "[Ablation] "
        f"no_match_features={ablation.no_match_features} "
        f"no_geometry_features={ablation.no_geometry_features} "
        f"no_distance_bins={ablation.no_distance_bins} "
        f"no_uncertainty={ablation.no_uncertainty}"
    )

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

    model_cfg = dict(config.get("model", {}))
    if ablation.no_uncertainty:
        model_cfg["use_uncertainty"] = False
    model = GeoPairNet(**model_cfg).to(device)

    channels_last = bool(training_cfg.get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    model.summary()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded checkpoint: {args.resume}")

    loss_cfg = config.get("loss", {})
    loss_weights = LossWeightConfig(distance_cls_weight=0.0) if ablation.no_distance_bins else None
    criterion = PairUAVLoss(
        log_distance_min=float(loss_cfg.get("log_distance_min", model_cfg.get("log_distance_min", 0.0))),
        log_distance_max=float(loss_cfg.get("log_distance_max", model_cfg.get("log_distance_max", 5.0))),
        num_bins=int(loss_cfg.get("distance_bins", model_cfg.get("distance_bins", 24))),
        min_distance=float(loss_cfg.get("min_distance", 1.0)),
        smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 0.05)),
        weights=loss_weights,
    )

    all_stages = config.get("stages", [])
    if not all_stages:
        raise RuntimeError("Config must include a non-empty 'stages' list.")
    stages = _stage_filter(all_stages, args.stages)


    total_epochs = sum(int(stage["epochs"]) for stage in stages)
    global_epoch_idx = 0

    ema_start_stage = str(training_cfg.get("ema_start_stage", "B")).upper()
    debug_decode_samples = int(training_cfg.get("debug_decode_samples", 3))
    cache_equivalence_tolerance = float(training_cfg.get("cache_equivalence_tolerance", 5e-2))

    expected_test_pairs = _count_expected_test_pairs(data_root)

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

    match_root_override = None if args.safe_baseline_mode else args.match_root
    match_index_override = None if args.safe_baseline_mode else args.match_index_file
    if args.safe_baseline_mode and (args.match_root is not None or args.match_index_file is not None):
        print("[SafeMode] ignoring --match-root and --match-index-file")

    for stage in stages:
        stage_name = str(stage["name"]).upper()
        stage_epochs = int(stage["epochs"])
        stage_lr = float(stage["lr"])
        stage_min_lr = float(stage.get("min_lr", stage_lr * 0.05))
        stage_warmup_epochs = int(stage.get("warmup_epochs", 1))

        model.set_backbone_trainable(str(stage.get("backbone_mode", "full")))
        model.set_shared_projectors_trainable(bool(stage.get("train_projectors", True)))

        train_loader, val_loader = build_dataloaders(
            root=data_root,
            dataset_cfg=dataset_cfg,
            split=split,
            stage_cfg=stage,
            workers_override=args.workers,
            match_root_override=match_root_override,
            match_index_override=match_index_override,
            seed=seed,
            strict_official_only=bool(dataset_cfg.get("strict_official_only", False)),
        )

        _print_dataset_diagnostics(
            train_dataset=train_loader.dataset,
            val_dataset=val_loader.dataset,
            split=split,
            dataset_cfg=dataset_cfg,
            seed=seed,
            expected_test_pairs=expected_test_pairs,
        )

        train_augment_enabled = not bool(stage.get("disable_augmentation", False))
        backbone_mode = str(stage.get("backbone_mode", "full")).lower()
        cache_requested = bool(stage.get("feature_cache", False))
        cache_allowed = cache_requested and backbone_mode == "frozen" and not train_augment_enabled
        if cache_requested and not cache_allowed:
            reasons: list[str] = []
            if backbone_mode != "frozen":
                reasons.append("backbone is not frozen")
            if train_augment_enabled:
                reasons.append("stochastic train augmentation is enabled")
            print("[Cache] disabled automatically because " + ", ".join(reasons))

        train_feature_cache = (
            FrozenFeatureCache(max_items=int(stage.get("feature_cache_size", 60_000)))
            if cache_allowed
            else None
        )
        val_feature_cache = (
            FrozenFeatureCache(max_items=int(stage.get("feature_cache_size", 60_000)))
            if cache_allowed
            else None
        )

        if cache_allowed:
            _assert_cached_vs_uncached_equivalence(
                model=model,
                loader=val_loader,
                device=device,
                channels_last=channels_last,
                ablation=ablation,
                tolerance=cache_equivalence_tolerance,
            )

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

        ema_active_this_stage = ema is not None and _stage_rank(stage_name) >= _stage_rank(ema_start_stage)

        print(
            f"\n[Stage {stage_name}] epochs={stage_epochs} lr={stage_lr:.2e} "
            f"batch={int(stage['batch_size'])} backbone_mode={stage.get('backbone_mode', 'full')} "
            f"cache={'on' if cache_allowed else 'off'} ema={'on' if ema_active_this_stage else 'off'} "
            f"train_pairs={len(train_loader.dataset)} val_pairs={len(val_loader.dataset)}"
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
                ablation=ablation,
                ema=ema if ema_active_this_stage else None,
            )

            if ema_active_this_stage and ema is not None:
                ema.store(model)
                ema.copy_to(model)

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
                ablation=ablation,
                debug_decode_samples=debug_decode_samples,
            )

            if ema_active_this_stage and ema is not None:
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
                f"grad_norm_mean={train_metrics['train_grad_norm_mean']:.3f} "
                f"grad_norm_max={train_metrics['train_grad_norm_max']:.3f} "
                f"val_total={val_metrics['val_total_loss']:.4f} "
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
                ema=ema if ema_active_this_stage else None,
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
                    ema=ema if ema_active_this_stage else None,
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




