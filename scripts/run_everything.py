#!/usr/bin/env python3
"""Verbose end-to-end runner for PairUAV.

The script performs a CPU smoke test first, then runs the selected GPU
training pipeline, generates result.txt, and finally packages result.zip.

Default mode: official data prep + raw dual-path training (phases 1, 2, 3)
with annotation supervision enabled when available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
sys.path.insert(0, str(REPO_ROOT))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def elapsed_text(seconds: float) -> str:
    minutes, rem_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{rem_seconds:02d}s"
    if minutes:
        return f"{minutes}m{rem_seconds:02d}s"
    return f"{rem_seconds}s"


def emit(message: str, log_file=None) -> None:
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def resolve_hf_endpoint() -> str:
    hf_endpoint = os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = hf_endpoint
    return hf_endpoint


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_root(value: str | None, *, description: str,
                      env_names: tuple[str, ...],
                      candidates: tuple[Path, ...],
                      allow_fallback_if_missing: bool = False) -> Path:
    checked: list[Path] = []
    seen: set[str] = set()

    if value:
        explicit = Path(value).expanduser()
        key = str(explicit)
        seen.add(key)
        checked.append(explicit)
        if explicit.is_dir():
            return explicit.resolve()
        if not allow_fallback_if_missing:
            raise FileNotFoundError(f"{description} does not exist: {explicit}")

    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        candidate = Path(env_value).expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    checked_text = ", ".join(str(path) for path in checked) if checked else "<none>"
    raise FileNotFoundError(f"Could not determine {description}. Checked: {checked_text}")


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_command(command: list[str], cwd: Path, log_path: Path, stage_name: str,
                env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        emit(f"[{stage_name}] Command: {format_command(command)}", log_file)
        emit(f"[{stage_name}] Working directory: {cwd}", log_file)

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            emit(f"[{stage_name}] {line}", log_file)

        return_code = process.wait()
        duration = elapsed_text(time.time() - start)
        emit(f"[{stage_name}] Exit code: {return_code} after {duration}", log_file)

        if return_code != 0:
            raise RuntimeError(f"Stage {stage_name} failed with exit code {return_code}")


def validate_pairuav_root(pairuav_root: Path) -> tuple[int, int]:
    from scripts.generate_submission import _discover_pairs

    pairs, _ = _discover_pairs(pairuav_root, pair_order="official")
    if not pairs:
        raise FileNotFoundError(
            f"Could not discover any submission pairs under {pairuav_root}. "
            "Expected a manifest file, test JSON pairs, or query/gallery split directories."
        )

    unique_images = {pair.source for pair in pairs}.union({pair.target for pair in pairs})
    return len(pairs), len(unique_images)


def _find_first_json_file(root: Path) -> Path | None:
    if not root.is_dir():
        return None

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda path: path.name.lower())
        except FileNotFoundError:
            continue

        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == ".json":
                return entry

        for entry in reversed(entries):
            if entry.is_dir():
                stack.append(entry)

    return None


def has_annotation_json(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    if any(path.is_file() for path in directory.glob("*.json")):
        return True
    for child in directory.iterdir():
        if child.is_dir() and any(path.is_file() for path in child.glob("*.json")):
            return True
    return False


def has_training_images(root: Path) -> bool:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    candidate_dirs = [root / "train_tour", root / "train" / "drone", root / "train"]
    for candidate in candidate_dirs:
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                return True
    return False


def _candidate_nested_roots(root: Path) -> list[Path]:
    return [
        root,
        root / "University-Release",
        root / "University-Release" / "University-Release",
        root / "PairUAV",
        root / "pairUAV",
        root / "pairuav",
    ]


def resolve_training_layout_root(root: Path) -> Path:
    """Resolve common nested AutoDL layouts to the directory that exposes train views."""
    from data.dataset import resolve_train_view_dir

    seen: set[str] = set()
    for candidate in _candidate_nested_roots(root):
        key = str(candidate)
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        try:
            resolve_train_view_dir(candidate)
            return candidate.resolve()
        except FileNotFoundError:
            continue
    return root.resolve()


def resolve_annotation_supervision_root(preferred_roots: Iterable[Path]) -> Path | None:
    from data.dataset import resolve_train_annotation_dir

    seen: set[str] = set()
    for root in preferred_roots:
        for candidate in _candidate_nested_roots(root):
            key = str(candidate)
            if key in seen or not candidate.is_dir():
                continue
            seen.add(key)
            if resolve_train_annotation_dir(candidate) is not None:
                return candidate.resolve()
    return None


def cpu_smoke_check(raw_root: Path, pairuav_root: Path, smoke_log: Path,
                    official_annotations: bool = False,
                    annotation_supervision_root: Path | None = None) -> None:
    from data.dataset import (
        PairUAVAnnotationDataset,
        resolve_train_annotation_dir,
        resolve_train_view_dir,
    )
    from models.baseline import PairUAVBaseline, baseline_total_loss
    from models.harp_dual_path import HARPDualPath
    from models.harp_pose_lite import HARPPoseLite, harp_pose_lite_loss
    from scripts.generate_submission import _discover_pairs
    from training.loss import laplace_nll, phase2_loss, phase3_loss

    start = time.time()
    failures: list[str] = []

    def check(name: str, fn: Callable[[], object]) -> None:
        emit(f"[CPU] BEGIN {name}", smoke_log)
        check_start = time.time()
        try:
            result = fn()
            duration = elapsed_text(time.time() - check_start)
            emit(f"[CPU] PASS {name} ({duration}) -> {result}", smoke_log)
        except Exception as exc:  # noqa: BLE001
            duration = elapsed_text(time.time() - check_start)
            emit(f"[CPU] FAIL {name} ({duration}) -> {exc}", smoke_log)
            emit(traceback.format_exc(), smoke_log)
            failures.append(name)

    check(
        "dependency versions",
        lambda: {
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "torchvision": __import__("torchvision").__version__,
            "timm": __import__("timm").__version__,
        },
    )

    check("raw dataset root", lambda: str(resolve_train_view_dir(raw_root)))

    if official_annotations:
        def annotation_dataset_sample() -> str:
            supervision_root = annotation_supervision_root or pairuav_root
            annotation_root = resolve_train_annotation_dir(supervision_root)
            if annotation_root is None:
                raise RuntimeError(
                    "Official annotation mode requested, but train annotations were not found under "
                    f"{supervision_root}"
                )
            first_json = _find_first_json_file(annotation_root)
            if first_json is None:
                raise RuntimeError(f"No annotation JSON files were found under {annotation_root}")
            dataset = PairUAVAnnotationDataset(
                str(supervision_root),
                max_pairs=1,
                json_paths=[first_json],
                seed=42,
                is_val=True,
            )
            source, target, meta = dataset[0]
            return (
                f"source={tuple(source.shape)} target={tuple(target.shape)} "
                f"heading={meta['heading']:.3f} distance={meta['distance']:.3f}"
            )

        check("official annotation sample", annotation_dataset_sample)

    official_pairs_cache: tuple[list[object], str] | None = None

    def _get_official_pairs() -> tuple[list[object], str]:
        nonlocal official_pairs_cache
        if official_pairs_cache is None:
            official_pairs_cache = _discover_pairs(pairuav_root, pair_order="official")
        return official_pairs_cache

    def pairuav_layout() -> str:
        pairs, source = _get_official_pairs()
        unique_images = {pair.source for pair in pairs}.union({pair.target for pair in pairs})
        return f"pairs={len(pairs)} source={source} test_images={len(unique_images)}"

    check("pairuav layout", pairuav_layout)

    def submission_order_check() -> str:
        pairs, source = _get_official_pairs()
        if not pairs:
            raise RuntimeError("No submission pairs discovered")
        first_pair = pairs[0]
        return (
            f"pairs={len(pairs)} source={source} "
            f"first=({first_pair.source.name}, {first_pair.target.name})"
        )

    check("submission pair order", submission_order_check)

    def baseline_forward() -> str:
        import torch

        model = PairUAVBaseline().eval()
        source = torch.randn(1, 3, 224, 224)
        target = torch.randn(1, 3, 224, 224)
        pred = model(source, target)
        target_batch = {
            "heading": torch.zeros(1),
            "distance": torch.ones(1),
        }
        loss = baseline_total_loss(pred, target_batch)
        return f"keys={sorted(pred.keys())} total={loss['total'].item():.4f}"

    def lite_forward() -> str:
        import torch

        model = HARPPoseLite().eval()
        source = torch.randn(1, 3, 224, 224)
        target = torch.randn(1, 3, 224, 224)
        pred = model(source, target)
        target_batch = {
            "heading": torch.zeros(1),
            "distance": torch.ones(1),
        }
        loss = harp_pose_lite_loss(pred, target_batch)
        return f"keys={sorted(pred.keys())} total={loss['total'].item():.4f}"

    def dual_path_forward() -> str:
        import torch

        model = HARPDualPath(frozen=True, use_gate=True).eval()
        model.phase = 2
        feat_s = torch.randn(1, 2048, 7, 7)
        feat_t = torch.randn(1, 2048, 7, 7)
        pred = model.forward_features(feat_s, feat_t)
        target_batch = {
            "heading": torch.zeros(1),
            "distance": torch.ones(1),
        }
        phase2 = phase2_loss(pred, target_batch)
        phase3 = phase3_loss(
            pred,
            target_batch,
            model_state=model.state_dict(),
            ewc_state=model.state_dict(),
            ewc_lambda=0.01,
        )
        return (
            f"keys={sorted(pred.keys())} phase2={phase2['total'].item():.4f} "
            f"phase3={phase3['total'].item():.4f}"
        )

    check("baseline model forward", baseline_forward)
    check("HARP-Pose-Lite forward", lite_forward)
    check("HARP dual-path forward", dual_path_forward)

    def loss_suite() -> str:
        import torch

        preds = {
            "heading": torch.tensor([0.0]),
            "distance": torch.tensor([1.0]),
            "confidence": torch.tensor([1.0]),
        }
        targets = {
            "heading": torch.tensor([0.0]),
            "distance": torch.tensor([1.0]),
        }
        laplace = laplace_nll(preds, targets)
        phase2 = phase2_loss(preds, targets)
        phase3 = phase3_loss(
            preds,
            targets,
            model_state={"w": torch.tensor([1.0])},
            ewc_state={"w": torch.tensor([1.0])},
            ewc_lambda=0.01,
        )
        return (
            f"laplace={laplace['total'].item():.4f} "
            f"phase2={phase2['total'].item():.4f} phase3={phase3['total'].item():.4f}"
        )

    check("loss functions", loss_suite)

    emit(f"[CPU] smoke check complete in {elapsed_text(time.time() - start)}", smoke_log)

    if failures:
        raise RuntimeError("CPU smoke checks failed: " + ", ".join(failures))


def package_submission(result_txt: Path, result_zip: Path) -> None:
    if not result_txt.is_file():
        raise FileNotFoundError(f"Missing result.txt: {result_txt}")

    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_txt, arcname="result.txt")


def has_prepared_pairuav_layout(root: Path) -> bool:
    """Return True when root looks like official processed PairUAV data."""
    train_dir = root / "train"
    test_dir = root / "test"
    has_train_view = has_training_images(root)
    has_train_annotations = has_annotation_json(train_dir)
    has_test_annotations = has_annotation_json(test_dir)
    has_test_tour = (root / "test_tour").is_dir()
    return has_train_view and has_train_annotations and has_test_annotations and has_test_tour


def resolve_superglue_root(explicit: str | None, pairuav_root: Path) -> Path:
    checked: list[Path] = []
    seen: set[str] = set()
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend([
        REPO_ROOT / "baseline" / "SuperGlue",
        REPO_ROOT.parent / "baseline" / "SuperGlue",
        pairuav_root / "baseline" / "SuperGlue",
    ])

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    checked_text = ", ".join(str(path) for path in checked) if checked else "<none>"
    raise FileNotFoundError(
        "Could not locate SuperGlue workspace. Checked: "
        f"{checked_text}. Provide --superglue-root to baseline/SuperGlue."
    )


def find_match_dir(split_tag: str, pairuav_root: Path, superglue_root: Path) -> Path | None:
    candidates = [
        pairuav_root / f"{split_tag}_matches_data",
        pairuav_root / "matches" / f"{split_tag}_matches_data",
        superglue_root / f"{split_tag}_matches_data",
        superglue_root.parent / f"{split_tag}_matches_data",
        superglue_root.parent.parent / f"{split_tag}_matches_data",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verbose end-to-end PairUAV runner")
    parser.add_argument("--train-root", "--university-release", "--data", dest="train_root", default=None,
                        help="Path to PairUAV training root (processed HF layout preferred); legacy aliases are kept for compatibility")
    parser.add_argument("--pairuav-root", default=None,
                        help="Path to the PairUAV submission root; auto-detected from mounted AutoDL data when omitted")
    parser.add_argument("--prepare-data", type=parse_bool_arg, default=True,
                        help="Run official PairUAV data preparation stage before training (true/false)")
    parser.add_argument("--force-data-prep", action="store_true",
                        help="Force rerunning official data prep even when processed layout already exists")
    parser.add_argument("--prepare-workdir", default=None,
                        help="Work directory for official data prep (defaults to --pairuav-root or /root/autodl-tmp/university/PairUAV)")
    parser.add_argument("--university-zip", default=None,
                        help="Deprecated no-op in HF-only prep mode")
    parser.add_argument("--prep-download-tool", choices=["auto", "hf", "huggingface-cli", "python"],
                        default="auto",
                        help="Download backend for official data prep")
    parser.add_argument("--prep-dataset-repo", default="YaxuanLi/UAVM_2026_test",
                        help="Hugging Face dataset repo used by official data prep")
    parser.add_argument("--hf-token", default=None,
                        help="Optional Hugging Face token for data prep download")
    parser.add_argument("--skip-prep-download", action="store_true",
                        help="Skip Hugging Face download in official data prep")
    parser.add_argument("--skip-prep-extract", action="store_true",
                        help="Skip tar extraction in official data prep")
    parser.add_argument("--prepare-matches", type=parse_bool_arg, default=False,
                        help="Run SuperGlue match-data preparation stage (true/false)")
    parser.add_argument("--superglue-root", default=None,
                        help="Path to baseline/SuperGlue workspace used for match preparation")
    parser.add_argument("--superglue-mode", choices=["download", "run"], default="download",
                        help="SuperGlue stage mode: download precomputed matches or run matching scripts")
    parser.add_argument("--mode", choices=["dual-path", "phase1-lite"], default="dual-path",
                        help="Training pipeline to run")
    parser.add_argument("--phases", default="1,2,3",
                        help="Comma-separated dual-path phases to run, e.g. 1,2,3")
    parser.add_argument("--run-dir", default=None,
                        help="Output directory for logs, checkpoints, cache, and submissions")
    parser.add_argument("--result-name", default="result.txt",
                        help="Submission filename to generate inside the run directory")
    parser.add_argument("--zip-name", default="result.zip",
                        help="Submission zip filename to generate inside the run directory")
    parser.add_argument("--cache-batch-size", type=int, default=64,
                        help="Batch size for feature extraction when building the cache")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip feature cache extraction if cache files already exist")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Stop after training without generating a submission")
    parser.add_argument("--skip-training", action="store_true",
                        help="Run only CPU checks and inference")
    parser.add_argument("--phase1-model", choices=["baseline", "harp-lite"], default="harp-lite",
                        help="Model to train in phase1-lite mode")
    parser.add_argument("--workers", type=int, default=16,
                        help="DataLoader workers to pass to training scripts")
    parser.add_argument("--raw", type=parse_bool_arg, default=True,
                        help="Use raw-image dual-path training instead of cached features (true/false)")
    parser.add_argument("--official-annotations", type=parse_bool_arg, default=True,
                        help="Enable official annotation supervision for raw dual-path training (true/false)")
    args = parser.parse_args()
    hf_endpoint = resolve_hf_endpoint()

    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (REPO_ROOT / "runs" / now_stamp())
    run_root = ensure_dir(run_root)
    logs_dir = ensure_dir(run_root / "logs")
    cache_train_dir = ensure_dir(run_root / "cache_train")
    cache_infer_dir = ensure_dir(run_root / "cache_infer")
    checkpoints_dir = ensure_dir(run_root / "checkpoints")
    submission_dir = ensure_dir(run_root / "submission")

    summary: dict[str, object] = {
        "run_root": str(run_root),
        "mode": args.mode,
        "raw_training": bool(args.raw),
        "prepare_data": bool(args.prepare_data),
        "hf_endpoint": hf_endpoint,
        "stages": [],
    }

    requested_pairuav_root = Path(args.pairuav_root).expanduser() if args.pairuav_root else None
    requested_train_root = Path(args.train_root).expanduser() if args.train_root else None

    if args.prepare_data:
        if args.prepare_workdir:
            prep_workdir = Path(args.prepare_workdir).expanduser().resolve()
        elif requested_pairuav_root is not None:
            prep_workdir = requested_pairuav_root.resolve()
        elif requested_train_root is not None:
            prep_workdir = requested_train_root.resolve()
        else:
            prep_workdir = Path(
                os.environ.get("PAIRUAV_ROOT", "/root/autodl-tmp/university/PairUAV")
            ).expanduser().resolve()

        ensure_dir(prep_workdir)

        if (
            has_prepared_pairuav_layout(prep_workdir)
            and not args.force_data_prep
            and not args.skip_prep_download
            and not args.skip_prep_extract
        ):
            print(
                "Official prep detected existing processed layout at "
                f"{prep_workdir}; skipping data-prep stage. "
                "Use --force-data-prep to rerun prep."
            )
        else:
            prep_log = logs_dir / "00_data_prep.log"
            prep_cmd = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts" / "prepare_pairuav_data.py"),
                "--workdir",
                str(prep_workdir),
                "--download-tool",
                args.prep_download_tool,
                "--dataset-repo",
                args.prep_dataset_repo,
            ]
            if args.hf_token:
                prep_cmd.extend(["--hf-token", args.hf_token])
            if args.skip_prep_download:
                prep_cmd.append("--skip-download")
            if args.skip_prep_extract:
                prep_cmd.append("--skip-extract")

            if args.university_zip:
                print("Note: --university-zip is ignored in HF-only prep mode.")

            run_command(prep_cmd, REPO_ROOT, prep_log, "data-prep")
            summary["stages"].append({
                "name": "data_prep",
                "status": "passed",
                "log": str(prep_log),
                "workdir": str(prep_workdir),
            })

        raw_root = prep_workdir
        pairuav_root = prep_workdir
    else:
        raw_root = resolve_data_root(
            args.train_root,
            description="PairUAV training root",
            env_names=(
                "PAIRUAV_TRAIN_ROOT",
                "PAIRUAV_ROOT",
                "PAIRUAV_DATA_ROOT",
                "PAIRUAV_PROCESSED_ROOT",
                "UNIVERSITY_RELEASE_ROOT",
                "UNIVERSITY_RELEASE",
            ),
            candidates=(
                Path("/root/autodl-tmp/university/University-Release/University-Release"),
                Path("/root/autodl-tmp/university/University-Release"),
                Path("/root/autodl-tmp/university/PairUAV"),
                Path("/root/autodl-tmp/university/PairUAV-Processed"),
                Path("/root/autodl-pub/PairUAV"),
            ),
        )
        resolved_raw_root = resolve_training_layout_root(raw_root)
        if resolved_raw_root != raw_root.resolve():
            print(
                "Note: resolved training root layout "
                f"from {raw_root} to {resolved_raw_root}"
            )
            raw_root = resolved_raw_root
        else:
            raw_root = resolved_raw_root

        pairuav_root = resolve_data_root(
            args.pairuav_root,
            description="PairUAV submission root",
            env_names=("PAIRUAV_ROOT", "PAIRUAV_DATA_ROOT", "PAIRUAV_PROCESSED_ROOT"),
            candidates=(
                raw_root,
                Path("/root/autodl-tmp/university/PairUAV"),
                Path("/root/autodl-tmp/university/PairUAV-Processed"),
                Path("/root/autodl-pub/PairUAV"),
            ),
            allow_fallback_if_missing=True,
        )
        if requested_pairuav_root is not None and not requested_pairuav_root.is_dir():
            print(f"Note: --pairuav-root {requested_pairuav_root} was not found; using detected root {pairuav_root}")

    annotation_supervision_root: Path | None = None
    if args.raw and args.official_annotations:
        annotation_supervision_root = resolve_annotation_supervision_root(
            preferred_roots=(pairuav_root, raw_root),
        )
        if annotation_supervision_root is None:
            raise FileNotFoundError(
                "Official annotation supervision was requested, but no train JSON annotations were found "
                f"under {pairuav_root} or {raw_root}."
            )
        if annotation_supervision_root != pairuav_root:
            print(
                "Note: using official train annotations from "
                f"{annotation_supervision_root}"
            )

    summary["raw_root"] = str(raw_root)
    summary["pairuav_root"] = str(pairuav_root)
    if annotation_supervision_root is not None:
        summary["annotation_supervision_root"] = str(annotation_supervision_root)

    if args.prepare_matches:
        superglue_root = resolve_superglue_root(args.superglue_root, pairuav_root)
        summary["superglue_root"] = str(superglue_root)
        superglue_env = {
            "PAIRUAV_ROOT": str(pairuav_root),
            "PAIRUAV_DATA_ROOT": str(pairuav_root),
        }

        if args.superglue_mode == "download":
            download_script = superglue_root / "download_results.sh"
            if not download_script.is_file():
                raise FileNotFoundError(
                    f"SuperGlue download script not found: {download_script}"
                )

            match_log = logs_dir / f"{len(summary['stages']):02d}_superglue_download.log"
            run_command(
                ["bash", str(download_script)],
                superglue_root,
                match_log,
                "superglue-download",
                env=superglue_env,
            )
            summary["stages"].append({
                "name": "superglue_download",
                "status": "passed",
                "log": str(match_log),
            })
        else:
            required = [
                ("gen_test_pairs.py", [sys.executable, "-u", str(superglue_root / "gen_test_pairs.py")], "superglue-gen-pairs"),
                ("run_train.sh", ["bash", str(superglue_root / "run_train.sh")], "superglue-run-train"),
                ("run_test.sh", ["bash", str(superglue_root / "run_test.sh")], "superglue-run-test"),
            ]

            for script_name, command, stage_name in required:
                script_path = superglue_root / script_name
                if not script_path.is_file():
                    raise FileNotFoundError(f"SuperGlue script not found: {script_path}")

                stage_log = logs_dir / f"{len(summary['stages']):02d}_{stage_name}.log"
                run_command(command, superglue_root, stage_log, stage_name, env=superglue_env)
                summary["stages"].append({
                    "name": stage_name.replace("-", "_"),
                    "status": "passed",
                    "log": str(stage_log),
                })

        train_match_root = find_match_dir("train", pairuav_root, superglue_root)
        test_match_root = find_match_dir("test", pairuav_root, superglue_root)
        if train_match_root is None or test_match_root is None:
            raise FileNotFoundError(
                "SuperGlue stage finished, but match directories were not discovered. "
                "Expected train_matches_data/ and test_matches_data/ under PairUAV or SuperGlue workspace."
            )

        print(f"Detected train match root: {train_match_root}")
        print(f"Detected test match root: {test_match_root}")
        summary["train_match_root"] = str(train_match_root)
        summary["test_match_root"] = str(test_match_root)

    print(f"Run directory: {run_root}")
    print(f"Mode: {args.mode}")
    print(f"Training root: {raw_root}")
    print(f"PairUAV root: {pairuav_root}")
    print(f"HF endpoint: {hf_endpoint}")
    print()

    cpu_log = logs_dir / f"{len(summary['stages']):02d}_cpu_smoke.log"
    cpu_start = time.time()
    try:
        with cpu_log.open("w", encoding="utf-8") as smoke_handle:
            cpu_smoke_check(
                raw_root,
                pairuav_root,
                smoke_handle,
                official_annotations=bool(args.raw and args.official_annotations),
                annotation_supervision_root=annotation_supervision_root,
            )
        cpu_status = "passed"
    except Exception as exc:  # noqa: BLE001
        cpu_status = "failed"
        print(f"CPU smoke check failed: {exc}")
        summary["stages"].append({
            "name": "cpu_smoke",
            "status": cpu_status,
            "elapsed": elapsed_text(time.time() - cpu_start),
            "log": str(cpu_log),
        })
        summary_path = run_root / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise

    summary["stages"].append({
        "name": "cpu_smoke",
        "status": cpu_status,
        "elapsed": elapsed_text(time.time() - cpu_start),
        "log": str(cpu_log),
    })

    if args.skip_training and args.skip_inference:
        summary_path = run_root / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Skipping training and inference as requested.")
        return

    print()
    if shutil.which("nvidia-smi") is not None:
        try:
            gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            print("GPU preflight (nvidia-smi):")
            print(gpu_info.stdout.rstrip())
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: nvidia-smi check failed: {exc}")

    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is not available, so GPU training cannot start.")

    import torch

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    if args.mode == "dual-path":
        train_script = REPO_ROOT / "training" / "train_dual_path.py"
        phases = [int(phase.strip()) for phase in args.phases.split(",") if phase.strip()]
        if not phases:
            raise ValueError("--phases must include at least one phase id from {1,2,3}")
        if not args.raw and 3 in phases:
            raise ValueError(
                "--raw false cannot run phase 3. Cached-feature mode bypasses backbone image encoding, "
                "so phase-3 backbone fine-tuning would be ineffective."
            )
        if not args.raw and args.official_annotations:
            print(
                "Note: --official-annotations applies only to --raw true. "
                "Cached mode uses pseudo labels from cached feature pairs."
            )
        checkpoint_path = checkpoints_dir / "dual_path.pt"

        if args.raw:
            print("Raw dual-path mode enabled: skipping training cache extraction.")
        else:
            cache_npz = list(cache_train_dir.glob("*.npz"))
            if args.skip_cache and not cache_npz:
                raise RuntimeError(
                    f"--skip-cache was set, but no cached .npz files were found in {cache_train_dir}."
                )

            if not args.skip_cache:
                if cache_npz:
                    print(f"Cache already present ({len(cache_npz)} .npz files); skipping extraction.")
                else:
                    cache_log = logs_dir / "01_cache_train.log"
                    cache_cmd = [
                        sys.executable, "-u", str(REPO_ROOT / "utils" / "cache_features.py"),
                        "--data-root", str(raw_root),
                        "--cache", str(cache_train_dir),
                        "--batch-size", str(args.cache_batch_size),
                    ]
                    run_command(cache_cmd, REPO_ROOT, cache_log, "cache-train")
                    summary["stages"].append({
                        "name": "cache_train",
                        "status": "passed",
                        "log": str(cache_log),
                    })

        if not args.skip_training:
            for index, phase in enumerate(phases, start=1):
                stage_name = f"train_phase_{phase}"
                train_log = logs_dir / f"{index + 1:02d}_{stage_name}.log"
                train_cmd = [
                    sys.executable, "-u", str(train_script),
                    "--phase", str(phase),
                    "--checkpoint", str(checkpoint_path),
                    "--workers", str(args.workers),
                ]
                if args.raw:
                    annotations_root = annotation_supervision_root or pairuav_root
                    train_cmd.extend([
                        "--raw", "true",
                        "--data-root", str(raw_root),
                        "--annotations-root", str(annotations_root),
                        "--official-annotations", "true" if args.official_annotations else "false",
                    ])
                else:
                    train_cmd.extend(["--cache", str(cache_train_dir)])
                run_command(train_cmd, REPO_ROOT, train_log, stage_name)
                summary["stages"].append({
                    "name": stage_name,
                    "status": "passed",
                    "log": str(train_log),
                })

    else:
        train_script = REPO_ROOT / "training" / "train_phase1.py"
        checkpoint_path = checkpoints_dir / "phase1_lite.pt"

        if not args.skip_training:
            train_log = logs_dir / "01_train_phase1_lite.log"
            train_cmd = [
                sys.executable, "-u", str(train_script),
                "--data-root", str(raw_root),
                "--model", args.phase1_model,
                "--output", str(checkpoint_path),
                "--num-workers", str(args.workers),
            ]
            run_command(train_cmd, REPO_ROOT, train_log, "train-phase1-lite")
            summary["stages"].append({
                "name": "train_phase1_lite",
                "status": "passed",
                "log": str(train_log),
            })

    if args.skip_inference:
        summary_path = run_root / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Training finished. Inference skipped by request.")
        return

    infer_log = logs_dir / f"{len(summary['stages']) + 1:02d}_inference.log"
    result_txt = submission_dir / args.result_name
    infer_cmd = [
        sys.executable, "-u", str(REPO_ROOT / "scripts" / "generate_submission.py"),
        "--checkpoint", str(checkpoint_path),
        "--cache", str(cache_infer_dir),
        "--pairuav-root", str(pairuav_root),
        "--pair-order", "official",
        "--safe-submission-mode",
        "--output", str(result_txt),
    ]
    run_command(infer_cmd, REPO_ROOT, infer_log, "inference")
    summary["stages"].append({
        "name": "inference",
        "status": "passed",
        "log": str(infer_log),
        "result_txt": str(result_txt),
    })

    result_zip = submission_dir / args.zip_name
    package_submission(result_txt, result_zip)
    print(f"Packaged submission: {result_zip}")

    summary["stages"].append({
        "name": "package",
        "status": "passed",
        "result_zip": str(result_zip),
    })

    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

