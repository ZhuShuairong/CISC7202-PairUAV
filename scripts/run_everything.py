#!/usr/bin/env python3
"""Verbose end-to-end runner for PairUAV.

The script performs a CPU smoke test first, then runs the selected GPU
training pipeline, generates result.txt, and finally packages result.zip.

Default mode: cached dual-path training (phases 1, 2, 3).
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

    pairs = _discover_pairs(pairuav_root)
    if not pairs:
        raise FileNotFoundError(
            f"Could not discover any submission pairs under {pairuav_root}. "
            "Expected a manifest file, test JSON pairs, or query/gallery split directories."
        )

    unique_images = {path for pair in pairs for path in pair}
    return len(pairs), len(unique_images)


def cpu_smoke_check(raw_root: Path, pairuav_root: Path, smoke_log: Path) -> None:
    from data.dataset import PairDataset, resolve_train_view_dir
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

    def raw_dataset_sample() -> str:
        view_dir = resolve_train_view_dir(raw_root)
        buildings = sorted(os.listdir(view_dir))
        if not buildings:
            raise RuntimeError(f"No building directories found in {view_dir}")
        dataset = PairDataset(str(raw_root), max_pairs=2, buildings=[buildings[0]], seed=42, is_val=True)
        source, target, meta = dataset[0]
        return (
            f"source={tuple(source.shape)} target={tuple(target.shape)} "
            f"heading={meta['heading']:.3f} distance={meta['distance']:.3f}"
        )

    check("raw dataset sample", raw_dataset_sample)

    def pairuav_layout() -> str:
        test_json_files, test_images = validate_pairuav_root(pairuav_root)
        return f"test_json_files={test_json_files} test_images={test_images}"

    check("pairuav layout", pairuav_layout)

    def submission_order_check() -> str:
        pairs = _discover_pairs(pairuav_root)
        if not pairs:
            raise RuntimeError("No submission pairs discovered")
        first_source, first_target = pairs[0]
        return f"pairs={len(pairs)} first=({first_source.name}, {first_target.name})"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verbose end-to-end PairUAV runner")
    parser.add_argument("--university-release", default=None,
                        help="Path to the raw University-Release root; auto-detected from the mounted AutoDL dataset when omitted")
    parser.add_argument("--pairuav-root", default=None,
                        help="Path to the PairUAV submission root; auto-detected from mounted AutoDL data when omitted")
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
    parser.add_argument("--raw", type=parse_bool_arg, default=False,
                        help="Use raw-image dual-path training instead of cached features (true/false)")
    args = parser.parse_args()

    raw_root = resolve_data_root(
        args.university_release,
        description="University-Release root",
        env_names=("PAIRUAV_UNIVERSITY_RELEASE", "UNIVERSITY_RELEASE_ROOT", "UNIVERSITY_RELEASE"),
        candidates=(
            Path("/root/autodl-tmp/university/University-Release/University-Release"),
            Path("/root/autodl-tmp/university/University-Release"),
            Path("/root/autodl-tmp/university/PairUAV"),
            Path("/root/autodl-tmp/university/pairUAV"),
            Path("/root/autodl-tmp/university/PairUAV-Processed"),
        ),
    )
    requested_pairuav_root = Path(args.pairuav_root).expanduser() if args.pairuav_root else None
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
        "raw_root": str(raw_root),
        "pairuav_root": str(pairuav_root),
        "stages": [],
    }

    print(f"Run directory: {run_root}")
    print(f"Mode: {args.mode}")
    print(f"Raw root: {raw_root}")
    print(f"PairUAV root: {pairuav_root}")
    print()

    cpu_log = logs_dir / "00_cpu_smoke.log"
    cpu_start = time.time()
    try:
        with cpu_log.open("w", encoding="utf-8") as smoke_handle:
            cpu_smoke_check(raw_root, pairuav_root, smoke_handle)
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
                        "--university-release", str(raw_root),
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
                    train_cmd.extend([
                        "--raw", "true",
                        "--university-release", str(raw_root),
                        "--annotations-root", str(pairuav_root),
                        "--official-annotations", "auto",
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
                "--university-release", str(raw_root),
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