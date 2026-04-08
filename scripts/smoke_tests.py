#!/usr/bin/env python3
"""Remote-oriented smoke tests for PairUAV.

These checks are designed for the Ubuntu/CUDA runtime used for training and
submission generation. They are safe to keep in the repository even when local
Windows development environments cannot execute them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
except Exception as exc:  # noqa: BLE001
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    _OPTIONAL_IMPORT_ERROR: Exception | None = exc
else:
    _OPTIONAL_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

collect_official_test_pairs: Any = None
OfflineMatchFeatureStore: Any = None
GeoPairNet: Any = None
resolve_pairuav_root: Any = None
_resolve_submission_pairs: Any = None
PairUAVLoss: Any = None
_assert_cached_vs_uncached_equivalence: Any = None
_assert_prediction_units: Any = None
_assert_target_encode_decode_roundtrip: Any = None
_resolve_data_split: Any = None
build_dataloaders: Any = None
evaluate_result_files: Any = None
write_result_file: Any = None


def _lazy_imports() -> None:
    global collect_official_test_pairs
    global OfflineMatchFeatureStore
    global GeoPairNet
    global resolve_pairuav_root
    global _resolve_submission_pairs
    global PairUAVLoss
    global _assert_cached_vs_uncached_equivalence
    global _assert_prediction_units
    global _assert_target_encode_decode_roundtrip
    global _resolve_data_split
    global build_dataloaders
    global evaluate_result_files
    global write_result_file

    if GeoPairNet is not None:
        return

    from data.dataset_pairuav import (
        OfflineMatchFeatureStore as _OfflineMatchFeatureStore,
        collect_official_test_pairs as _collect_official_test_pairs,
    )
    from models.geopairnet import GeoPairNet as _GeoPairNet
    from scripts.generate_submission import (
        _resolve_submission_pairs as _resolve_submission_pairs_impl,
        resolve_pairuav_root as resolve_pairuav_root_impl,
    )
    from training.losses import PairUAVLoss as _PairUAVLoss
    from training.train_pairuav import (
        _assert_cached_vs_uncached_equivalence as _assert_cached_vs_uncached_equivalence_impl,
        _assert_prediction_units as _assert_prediction_units_impl,
        _assert_target_encode_decode_roundtrip as _assert_target_encode_decode_roundtrip_impl,
        _resolve_data_split as _resolve_data_split_impl,
        build_dataloaders as build_dataloaders_impl,
    )
    from utils.metrics import (
        evaluate_result_files as evaluate_result_files_impl,
        write_result_file as write_result_file_impl,
    )

    collect_official_test_pairs = _collect_official_test_pairs
    OfflineMatchFeatureStore = _OfflineMatchFeatureStore
    GeoPairNet = _GeoPairNet
    resolve_pairuav_root = resolve_pairuav_root_impl
    _resolve_submission_pairs = _resolve_submission_pairs_impl
    PairUAVLoss = _PairUAVLoss
    _assert_cached_vs_uncached_equivalence = _assert_cached_vs_uncached_equivalence_impl
    _assert_prediction_units = _assert_prediction_units_impl
    _assert_target_encode_decode_roundtrip = _assert_target_encode_decode_roundtrip_impl
    _resolve_data_split = _resolve_data_split_impl
    build_dataloaders = build_dataloaders_impl
    evaluate_result_files = evaluate_result_files_impl
    write_result_file = write_result_file_impl


TEST_NAMES = (
    "test_one_batch_forward",
    "test_overfit_16_samples",
    "test_submission_order",
    "test_decode_units",
    "test_cached_vs_uncached",
    "test_result_txt_parser",
    "test_match_alignment",
)

GATE_NAME = "pre_submission_gate"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PairUAV smoke tests")
    parser.add_argument("--config", type=str, default="configs/geopairnet_default.json")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--pairuav-root", type=str, default=None)
    parser.add_argument("--tests", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overfit-steps", type=int, default=25)
    parser.add_argument("--cache-tolerance", type=float, default=1e-4)
    parser.add_argument("--match-root", type=str, default=None)
    parser.add_argument("--match-index-file", type=str, default=None)
    parser.add_argument(
        "--match-policy",
        choices=["optional", "required"],
        default="optional",
        help="Whether match-alignment failures should fail when checking match artifacts.",
    )
    parser.add_argument(
        "--gate-include-match-check",
        action="store_true",
        help="Include test_match_alignment inside pre_submission_gate even when --match-root is not set.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_loss(config: dict[str, Any], model_cfg: dict[str, Any]) -> PairUAVLoss:
    loss_cfg = dict(config.get("loss", {}))
    distance_cls_weight = float(loss_cfg.get("distance_cls_weight", 0.35))
    if bool(model_cfg.get("no_distance_bins", False)):
        distance_cls_weight = 0.0

    return PairUAVLoss(
        log_distance_min=float(loss_cfg.get("log_distance_min", model_cfg.get("log_distance_min", 0.0))),
        log_distance_max=float(loss_cfg.get("log_distance_max", model_cfg.get("log_distance_max", 5.0))),
        num_bins=int(loss_cfg.get("distance_bins", model_cfg.get("distance_bins", 24))),
        min_distance=float(loss_cfg.get("min_distance", 1.0)),
        smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 0.05)),
        distance_cls_weight=distance_cls_weight,
    )


def _build_train_loader(
    config: dict[str, Any],
    data_root: Path,
    seed: int,
    batch_size: int,
    force_official: bool,
) -> tuple[torch.utils.data.DataLoader, dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_cfg = dict(config.get("dataset", {}))
    model_cfg = dict(config.get("model", {}))

    dataset_cfg["max_train_pairs"] = max(64, batch_size)
    dataset_cfg["max_val_pairs"] = max(32, batch_size)
    if force_official:
        dataset_cfg["mode"] = "official"
        dataset_cfg["strict_official_only"] = True

    split = _resolve_data_split(data_root, dataset_cfg, seed=seed)
    strict_official_only = bool(split.get("strict_official_only", dataset_cfg.get("strict_official_only", False)))
    use_match_features = not bool(model_cfg.get("no_match_features", False))

    stage_cfg = dict(config.get("stages", [])[0])
    stage_cfg["batch_size"] = batch_size
    stage_cfg["disable_augmentation"] = True
    stage_cfg["feature_cache"] = False

    train_loader, _ = build_dataloaders(
        root=data_root,
        dataset_cfg=dataset_cfg,
        split=split,
        stage_cfg=stage_cfg,
        workers_override=0,
        match_root_override=None,
        match_index_override=None,
        seed=seed,
        strict_official_only=strict_official_only,
        use_match_features=use_match_features,
    )
    return train_loader, dataset_cfg, model_cfg, split


def _to_device_batch(
    batch: tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    source, target, meta = batch
    source = source.to(device)
    target = target.to(device)
    packed = {
        "heading": meta["heading"].to(device),
        "distance": meta["distance"].to(device),
        "match_features": meta["match_features"].to(device),
        "geometry_features": meta["geometry_features"].to(device),
    }
    return source, target, packed


def test_one_batch_forward(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> str:
    train_loader, _, model_cfg, _ = _build_train_loader(
        config=config,
        data_root=Path(args.data_root).expanduser().resolve(),
        seed=args.seed,
        batch_size=args.batch_size,
        force_official=False,
    )
    batch = next(iter(train_loader))
    source, target, meta = _to_device_batch(batch, device)

    model = GeoPairNet(**model_cfg).to(device)
    model.eval()
    criterion = _build_loss(config, model_cfg)

    with torch.no_grad():
        prediction = model(
            source,
            target,
            match_features=meta["match_features"],
            geometry_features=meta["geometry_features"],
        )
        _assert_prediction_units(prediction, context="smoke_one_batch_forward")
        losses = criterion(
            prediction=prediction,
            target={"heading": meta["heading"], "distance": meta["distance"]},
            progress=0.0,
            stage_name="A",
        )

    if not torch.isfinite(losses["total"]):
        raise RuntimeError("Non-finite total loss in one-batch forward smoke test")

    return f"batch={source.shape[0]} total_loss={float(losses['total'].item()):.4f}"


def test_overfit_16_samples(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> str:
    train_loader, _, model_cfg, _ = _build_train_loader(
        config=config,
        data_root=Path(args.data_root).expanduser().resolve(),
        seed=args.seed,
        batch_size=16,
        force_official=False,
    )
    batch = next(iter(train_loader))
    source, target, meta = _to_device_batch(batch, device)

    model = GeoPairNet(**model_cfg).to(device)
    model.set_backbone_trainable("frozen")
    model.set_shared_projectors_trainable(True)
    model.train()

    criterion = _build_loss(config, model_cfg)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-3,
        weight_decay=1e-4,
    )

    with torch.no_grad():
        initial = criterion(
            prediction=model(source, target, match_features=meta["match_features"], geometry_features=meta["geometry_features"]),
            target={"heading": meta["heading"], "distance": meta["distance"]},
            progress=0.0,
            stage_name="A",
        )["total"].item()

    for step in range(max(1, args.overfit_steps)):
        prediction = model(
            source,
            target,
            match_features=meta["match_features"],
            geometry_features=meta["geometry_features"],
        )
        losses = criterion(
            prediction=prediction,
            target={"heading": meta["heading"], "distance": meta["distance"]},
            progress=min(1.0, (step + 1) / max(1, args.overfit_steps)),
            stage_name="A",
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

    with torch.no_grad():
        final = criterion(
            prediction=model(source, target, match_features=meta["match_features"], geometry_features=meta["geometry_features"]),
            target={"heading": meta["heading"], "distance": meta["distance"]},
            progress=1.0,
            stage_name="A",
        )["total"].item()

    if final >= initial * 0.95:
        raise RuntimeError(
            f"Overfit test failed: initial={initial:.4f}, final={final:.4f} (expected >=5% loss drop)."
        )

    return f"initial={initial:.4f} final={final:.4f}"


def test_submission_order(args: argparse.Namespace, _config: dict[str, Any], _device: torch.device) -> str:
    root_hint = args.pairuav_root if args.pairuav_root else args.data_root
    root = resolve_pairuav_root(root_hint)

    official_pairs = collect_official_test_pairs(root)
    if not official_pairs:
        raise RuntimeError(f"No official test annotation JSON pairs found under {root}")

    resolved_pairs, source = _resolve_submission_pairs(root=root, require_official_order=True)
    if source != "official_test_json":
        raise RuntimeError(f"Unexpected pair source for strict order mode: {source}")

    if len(resolved_pairs) != len(official_pairs):
        raise RuntimeError(
            f"Pair count mismatch: official={len(official_pairs)} resolved={len(resolved_pairs)}"
        )

    preview = min(50, len(official_pairs))
    for idx in range(preview):
        expected_pair_id = official_pairs[idx].pair_id
        resolved_pair_id = resolved_pairs[idx].pair_id.split("::", 1)[0]
        if expected_pair_id != resolved_pair_id:
            raise RuntimeError(
                "Official order mismatch at index "
                f"{idx}: expected={expected_pair_id} got={resolved_pair_id}"
            )

    return f"pairs={len(official_pairs)} verified_prefix={preview}"


def test_decode_units(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> str:
    train_loader, _, model_cfg, _ = _build_train_loader(
        config=config,
        data_root=Path(args.data_root).expanduser().resolve(),
        seed=args.seed,
        batch_size=args.batch_size,
        force_official=False,
    )
    batch = next(iter(train_loader))
    source, target, meta = _to_device_batch(batch, device)

    # Validate target-space encode/decode consistency used in training.
    _assert_target_encode_decode_roundtrip({"heading": meta["heading"], "distance": meta["distance"]})

    model = GeoPairNet(**model_cfg).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(
            source,
            target,
            match_features=meta["match_features"],
            geometry_features=meta["geometry_features"],
        )
    _assert_prediction_units(prediction, context="smoke_decode_units")

    heading = prediction["heading_deg"].detach().float()
    distance = prediction["distance"].detach().float()
    return (
        f"heading_range=[{float(heading.min().item()):.3f},{float(heading.max().item()):.3f}] "
        f"distance_min={float(distance.min().item()):.3f}"
    )


def test_cached_vs_uncached(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> str:
    train_loader, _, model_cfg, _ = _build_train_loader(
        config=config,
        data_root=Path(args.data_root).expanduser().resolve(),
        seed=args.seed,
        batch_size=args.batch_size,
        force_official=False,
    )
    batch = next(iter(train_loader))

    model = GeoPairNet(**model_cfg).to(device)
    model.eval()

    _assert_cached_vs_uncached_equivalence(
        model=model,
        batch=batch,
        device=device,
        channels_last=False,
        tolerance=float(args.cache_tolerance),
    )
    return f"tolerance={float(args.cache_tolerance):.1e}"


def test_result_txt_parser(args: argparse.Namespace, _config: dict[str, Any], _device: torch.device) -> str:
    with tempfile.TemporaryDirectory(prefix="pairuav_parser_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        pred_path = tmp_root / "result.txt"
        truth_path = tmp_root / "truth.txt"

        pred_heading = torch.tensor([10.0, -20.0, 179.9, -179.5], dtype=torch.float32)
        pred_distance = torch.tensor([10.0, 20.5, 30.25, 40.75], dtype=torch.float32)
        truth_heading = torch.tensor([12.0, -18.5, 178.0, -178.0], dtype=torch.float32)
        truth_distance = torch.tensor([11.0, 19.5, 29.75, 42.0], dtype=torch.float32)

        write_result_file(pred_path, pred_heading, pred_distance, delimiter="comma")
        write_result_file(truth_path, truth_heading, truth_distance, delimiter="space")

        # Mix comma- and whitespace-separated lines in one file to validate parser robustness.
        pred_lines = pred_path.read_text(encoding="utf-8").splitlines()
        if len(pred_lines) >= 2:
            values = pred_lines[1].split(",")
            pred_lines[1] = f"{float(values[0]):.6f} {float(values[1]):.6f}"
        pred_path.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

        metrics = evaluate_result_files(pred_path, truth_path)
        required_keys = {"angle_rel_error", "distance_rel_error", "final_score", "mae_heading", "mae_distance"}
        missing_keys = sorted(required_keys - set(metrics))
        if missing_keys:
            raise RuntimeError(f"Parser metrics missing keys: {missing_keys}")

    return (
        f"final_score={metrics['final_score']:.6f} "
        f"angle_rel={metrics['angle_rel_error']:.6f} "
        f"distance_rel={metrics['distance_rel_error']:.6f}"
    )


def test_match_alignment(args: argparse.Namespace, _config: dict[str, Any], _device: torch.device) -> str:
    root_hint = args.pairuav_root if args.pairuav_root else args.data_root
    root = resolve_pairuav_root(root_hint)
    official_pairs = collect_official_test_pairs(root, limit=5)
    if not official_pairs:
        raise RuntimeError(f"No official test pairs found under {root}")

    if not args.match_root:
        if args.match_policy == "required":
            raise RuntimeError("Match alignment required but --match-root was not provided")
        return "skipped (optional): --match-root not provided"

    store = OfflineMatchFeatureStore(
        match_root=args.match_root,
        index_file=args.match_index_file,
        feature_dim=8,
    )
    if not store.enabled:
        if args.match_policy == "required":
            raise RuntimeError(f"Match root is not usable: {args.match_root}")
        return f"skipped (optional): match root not usable ({args.match_root})"

    missing = 0
    reverse_checked = 0
    reverse_failures = 0
    for pair in official_pairs:
        probe = store.probe(pair.source_ref, pair.target_ref)
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

    if missing > 0 or reverse_failures > 0:
        message = (
            f"match alignment issues: missing={missing}, "
            f"reverse_failures={reverse_failures}, checked={len(official_pairs)}"
        )
        if args.match_policy == "required":
            raise RuntimeError(message)
        return f"warning (optional): {message}"

    return (
        f"aligned_pairs={len(official_pairs)} "
        f"reverse_checks={reverse_checked} reverse_failures={reverse_failures}"
    )


def pre_submission_gate(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> str:
    details: list[str] = []

    details.append("submission_order=" + test_submission_order(args, config, device))
    details.append("decode_units=" + test_decode_units(args, config, device))
    details.append("result_parser=" + test_result_txt_parser(args, config, device))

    root_hint = args.pairuav_root if args.pairuav_root else args.data_root
    root = resolve_pairuav_root(root_hint)
    official_count = len(collect_official_test_pairs(root))
    resolved_pairs, _ = _resolve_submission_pairs(root=root, require_official_order=True)
    if official_count != len(resolved_pairs):
        raise RuntimeError(
            "Pair count check failed: "
            f"official={official_count}, resolved={len(resolved_pairs)}"
        )
    details.append(f"pair_count_check=official:{official_count}")

    include_match_check = bool(args.match_root) or bool(args.gate_include_match_check)
    if include_match_check:
        details.append("match_alignment=" + test_match_alignment(args, config, device))
    else:
        details.append("match_alignment=skipped(optional)")

    return " | ".join(details)


def _resolve_requested_tests(tests_arg: str) -> list[str]:
    raw = tests_arg.strip().lower()
    if raw == "all":
        return list(TEST_NAMES)
    if raw == GATE_NAME:
        return [GATE_NAME]

    names = [item.strip() for item in tests_arg.split(",") if item.strip()]
    allowed = set(TEST_NAMES) | {GATE_NAME}
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown tests: {unknown}. Available: {list(TEST_NAMES)} + [{GATE_NAME}]")
    return names


def main() -> None:
    args = _parse_args()
    if _OPTIONAL_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing Python runtime dependencies for smoke tests. "
            "Install requirements first (for example: pip install -r requirements.txt). "
            f"Original import error: {_OPTIONAL_IMPORT_ERROR}"
        )

    _lazy_imports()
    _set_seed(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    config = _load_config(config_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_tests = _resolve_requested_tests(args.tests)

    runners = {
        "test_one_batch_forward": test_one_batch_forward,
        "test_overfit_16_samples": test_overfit_16_samples,
        "test_submission_order": test_submission_order,
        "test_decode_units": test_decode_units,
        "test_cached_vs_uncached": test_cached_vs_uncached,
        "test_result_txt_parser": test_result_txt_parser,
        "test_match_alignment": test_match_alignment,
        GATE_NAME: pre_submission_gate,
    }

    print(f"Running {len(selected_tests)} smoke test(s) on device={device}...")
    failures: list[str] = []

    for name in selected_tests:
        start = time.time()
        print(f"[START] {name}")
        try:
            detail = runners[name](args, config, device)
            elapsed = time.time() - start
            print(f"[PASS] {name} ({elapsed:.1f}s) -> {detail}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            print(f"[FAIL] {name} ({elapsed:.1f}s) -> {exc}")
            traceback.print_exc()
            failures.append(name)

    if failures:
        raise SystemExit(f"Smoke tests failed: {failures}")

    print("All selected smoke tests passed.")


if __name__ == "__main__":
    main()