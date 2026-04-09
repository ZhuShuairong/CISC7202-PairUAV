#!/usr/bin/env python3
"""Run the current dual-path pipeline and the original baseline, then compare result.txt files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_everything import (  # noqa: E402
    ensure_dir,
    has_prepared_pairuav_layout,
    now_stamp,
    parse_bool_arg,
    resolve_hf_endpoint,
    run_command,
)
from utils.metrics import evaluate_result_files, read_result_file  # noqa: E402


@dataclass(frozen=True)
class PipelineRun:
    name: str
    run_root: Path
    checkpoint: Path
    result_txt: Path


def parse_phase_list(value: str) -> list[int]:
    phases: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            phases.append(int(part))
        except ValueError as exc:  # noqa: BLE001
            raise argparse.ArgumentTypeError(f"Invalid phase value: {part}") from exc

    if not phases:
        raise argparse.ArgumentTypeError("At least one phase must be provided")
    return phases


def resolve_workdir(explicit: str | None) -> Path:
    checked: list[Path] = []
    seen: set[str] = set()

    if explicit:
        return Path(explicit).expanduser().resolve()

    for env_name in (
        "PAIRUAV_ROOT",
        "PAIRUAV_DATA_ROOT",
        "PAIRUAV_PROCESSED_ROOT",
        "UNIVERSITY_RELEASE_ROOT",
        "UNIVERSITY_RELEASE",
    ):
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

    for candidate in (
        Path("/root/autodl-tmp/university/PairUAV"),
        Path("/root/autodl-tmp/university/University-Release/University-Release"),
        Path("/root/autodl-tmp/university/University-Release"),
        Path("/root/autodl-pub/PairUAV"),
    ):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    checked_text = ", ".join(str(path) for path in checked) if checked else "<none>"
    raise FileNotFoundError(
        "Could not determine a PairUAV workdir. Checked: "
        f"{checked_text}. Pass --workdir explicitly."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_result_file(path: Path) -> dict[str, object]:
    parsed = read_result_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "path": str(path),
        "line_count": int(parsed["heading"].numel()),
        "sha256": sha256_file(path),
        "first_line": lines[0] if lines else None,
        "last_line": lines[-1] if lines else None,
    }


def compare_result_files(current_path: Path, baseline_path: Path, max_differences: int = 10) -> dict[str, object]:
    current_lines = current_path.read_text(encoding="utf-8").splitlines()
    baseline_lines = baseline_path.read_text(encoding="utf-8").splitlines()

    current_pred = read_result_file(current_path)
    baseline_pred = read_result_file(baseline_path)

    shared_line_count = min(len(current_lines), len(baseline_lines))
    differing_lines: list[dict[str, object]] = []
    for line_index in range(shared_line_count):
        current_line = current_lines[line_index].strip()
        baseline_line = baseline_lines[line_index].strip()
        if current_line == baseline_line:
            continue
        differing_lines.append(
            {
                "line": line_index + 1,
                "current": current_line,
                "baseline": baseline_line,
            }
        )
        if len(differing_lines) >= max_differences:
            break

    common_count = min(current_pred["heading"].numel(), baseline_pred["heading"].numel())
    delta_summary: dict[str, object] = {}
    if common_count > 0:
        heading_delta = (current_pred["heading"][:common_count] - baseline_pred["heading"][:common_count]).abs()
        distance_delta = (current_pred["distance"][:common_count] - baseline_pred["distance"][:common_count]).abs()
        delta_summary = {
            "shared_prediction_count": int(common_count),
            "heading_abs_delta_mean": float(heading_delta.mean().item()),
            "heading_abs_delta_max": float(heading_delta.max().item()),
            "distance_abs_delta_mean": float(distance_delta.mean().item()),
            "distance_abs_delta_max": float(distance_delta.max().item()),
        }

    return {
        "same_line_count": len(current_lines) == len(baseline_lines),
        "current_line_count": len(current_lines),
        "baseline_line_count": len(baseline_lines),
        "identical_text": current_lines == baseline_lines,
        "shared_line_count": shared_line_count,
        "current_extra_lines": max(0, len(current_lines) - shared_line_count),
        "baseline_extra_lines": max(0, len(baseline_lines) - shared_line_count),
        "first_differences": differing_lines,
        **delta_summary,
    }


def prepare_pairuav_layout(workdir: Path, args: argparse.Namespace, logs_root: Path) -> str:
    prep_log = logs_root / "00_data_prep.log"

    if not args.prepare_data:
        if not has_prepared_pairuav_layout(workdir):
            raise FileNotFoundError(
                f"PairUAV layout not found at {workdir}. Enable --prepare-data or point --workdir at a prepared root."
            )
        print(f"Using existing prepared PairUAV layout at {workdir}")
        return "skipped"

    if (
        has_prepared_pairuav_layout(workdir)
        and not args.force_data_prep
        and not args.skip_prep_download
        and not args.skip_prep_extract
    ):
        print(f"Official prep detected existing processed layout at {workdir}; skipping data-prep stage.")
        return "skipped"

    prep_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "prepare_pairuav_data.py"),
        "--workdir",
        str(workdir),
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

    run_command(prep_cmd, REPO_ROOT, prep_log, "data-prep")
    return "passed"


def run_current_dual_path(workdir: Path, run_root: Path, phases: list[int], workers: int) -> PipelineRun:
    logs_dir = ensure_dir(run_root / "logs")
    cache_infer_dir = ensure_dir(run_root / "cache_infer")
    checkpoints_dir = ensure_dir(run_root / "checkpoints")
    submission_dir = ensure_dir(run_root / "submission")
    checkpoint_path = checkpoints_dir / "dual_path.pt"

    train_script = REPO_ROOT / "training" / "train_dual_path.py"
    for index, phase in enumerate(phases, start=1):
        train_log = logs_dir / f"{index:02d}_train_phase_{phase}.log"
        train_cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--phase",
            str(phase),
            "--checkpoint",
            str(checkpoint_path),
            "--workers",
            str(workers),
            "--raw",
            "true",
            "--data-root",
            str(workdir),
            "--annotations-root",
            str(workdir),
            "--official-annotations",
            "true",
            "--strict-official-only",
            "true",
        ]
        run_command(train_cmd, REPO_ROOT, train_log, f"current-train-phase-{phase}")

    infer_log = logs_dir / f"{len(phases) + 1:02d}_inference.log"
    result_txt = submission_dir / "result.txt"
    infer_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "generate_submission.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--cache",
        str(cache_infer_dir),
        "--pairuav-root",
        str(workdir),
        "--pair-order",
        "official",
        "--safe-submission-mode",
        "--output",
        str(result_txt),
    ]
    run_command(infer_cmd, REPO_ROOT, infer_log, "current-inference")
    return PipelineRun("current-dual-path", run_root, checkpoint_path, result_txt)


def run_original_baseline(workdir: Path, run_root: Path, workers: int) -> PipelineRun:
    logs_dir = ensure_dir(run_root / "logs")
    cache_infer_dir = ensure_dir(run_root / "cache_infer")
    checkpoints_dir = ensure_dir(run_root / "checkpoints")
    submission_dir = ensure_dir(run_root / "submission")
    checkpoint_path = checkpoints_dir / "phase1_baseline.pt"

    train_log = logs_dir / "01_train_phase1_baseline.log"
    train_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "training" / "train_phase1.py"),
        "--data-root",
        str(workdir),
        "--model",
        "baseline",
        "--dataset-mode",
        "official",
        "--strict-official-only",
        "--output",
        str(checkpoint_path),
        "--num-workers",
        str(workers),
    ]
    run_command(train_cmd, REPO_ROOT, train_log, "baseline-train-phase1")

    infer_log = logs_dir / "02_inference.log"
    result_txt = submission_dir / "result.txt"
    infer_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "generate_submission.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--cache",
        str(cache_infer_dir),
        "--pairuav-root",
        str(workdir),
        "--pair-order",
        "official",
        "--safe-submission-mode",
        "--output",
        str(result_txt),
    ]
    run_command(infer_cmd, REPO_ROOT, infer_log, "baseline-inference")
    return PipelineRun("original-baseline", run_root, checkpoint_path, result_txt)


def build_report(
    workdir: Path,
    run_root: Path,
    preparation_status: str,
    current_run: PipelineRun,
    baseline_run: PipelineRun,
    truth_path: Path | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "workdir": str(workdir),
        "run_root": str(run_root),
        "preparation_status": preparation_status,
        "current": {
            "name": current_run.name,
            "run_root": str(current_run.run_root),
            "checkpoint": str(current_run.checkpoint),
            "result": summarize_result_file(current_run.result_txt),
        },
        "baseline": {
            "name": baseline_run.name,
            "run_root": str(baseline_run.run_root),
            "checkpoint": str(baseline_run.checkpoint),
            "result": summarize_result_file(baseline_run.result_txt),
        },
        "comparison": compare_result_files(current_run.result_txt, baseline_run.result_txt),
    }

    if truth_path is not None:
        current_truth_metrics = evaluate_result_files(current_run.result_txt, truth_path)
        baseline_truth_metrics = evaluate_result_files(baseline_run.result_txt, truth_path)
        metric_keys = sorted(current_truth_metrics.keys())
        report["truth"] = {
            "path": str(truth_path),
            "current": current_truth_metrics,
            "baseline": baseline_truth_metrics,
            "delta_current_minus_baseline": {
                key: float(current_truth_metrics[key]) - float(baseline_truth_metrics[key])
                for key in metric_keys
            },
        }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current PairUAV pipeline and the original baseline, then compare result.txt files",
    )
    parser.add_argument(
        "--workdir",
        "--pairuav-root",
        "--data-root",
        dest="workdir",
        default=None,
        help="Prepared PairUAV root used for both training and submission discovery",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output directory for the current, baseline, and comparison artifacts",
    )
    parser.add_argument(
        "--prepare-data",
        type=parse_bool_arg,
        default=True,
        help="Prepare official PairUAV data before running the two pipelines",
    )
    parser.add_argument(
        "--force-data-prep",
        action="store_true",
        help="Force rerunning official data prep even when a processed layout already exists",
    )
    parser.add_argument(
        "--prep-download-tool",
        choices=["auto", "hf", "huggingface-cli", "python"],
        default="auto",
        help="Download backend for official data prep",
    )
    parser.add_argument(
        "--prep-dataset-repo",
        default="YaxuanLi/UAVM_2026_test",
        help="Hugging Face dataset repo used by official data prep",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for data prep download",
    )
    parser.add_argument(
        "--skip-prep-download",
        action="store_true",
        help="Skip Hugging Face download in official data prep",
    )
    parser.add_argument(
        "--skip-prep-extract",
        action="store_true",
        help="Skip tar extraction in official data prep",
    )
    parser.add_argument(
        "--current-phases",
        type=parse_phase_list,
        default=parse_phase_list("1,2,3"),
        help="Comma-separated dual-path phases to run, e.g. 1,2,3",
    )
    parser.add_argument(
        "--current-workers",
        type=int,
        default=16,
        help="DataLoader workers for the current dual-path pipeline",
    )
    parser.add_argument(
        "--baseline-workers",
        type=int,
        default=16,
        help="DataLoader workers for the original baseline pipeline",
    )
    parser.add_argument(
        "--truth",
        default=None,
        help="Optional truth result file to score both result.txt files against",
    )
    parser.add_argument(
        "--report-name",
        default="comparison_report.json",
        help="Filename for the saved comparison report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_endpoint = resolve_hf_endpoint()

    workdir = resolve_workdir(args.workdir)
    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (REPO_ROOT / "runs" / f"compare_{now_stamp()}")
    run_root = ensure_dir(run_root)

    logs_root = ensure_dir(run_root / "logs")
    current_root = ensure_dir(run_root / "current")
    baseline_root = ensure_dir(run_root / "baseline")

    print(f"Run directory: {run_root}")
    print(f"Workdir: {workdir}")
    print(f"HF endpoint: {hf_endpoint}")
    print()

    preparation_status = prepare_pairuav_layout(workdir, args, logs_root)

    current_run = run_current_dual_path(
        workdir=workdir,
        run_root=current_root,
        phases=args.current_phases,
        workers=args.current_workers,
    )

    baseline_run = run_original_baseline(
        workdir=workdir,
        run_root=baseline_root,
        workers=args.baseline_workers,
    )

    truth_path = Path(args.truth).expanduser().resolve() if args.truth else None
    report = build_report(
        workdir=workdir,
        run_root=run_root,
        preparation_status=preparation_status,
        current_run=current_run,
        baseline_run=baseline_run,
        truth_path=truth_path,
    )

    report_path = run_root / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["comparison"], indent=2))
    if truth_path is not None:
        print()
        print(json.dumps(report["truth"], indent=2))
    print(f"Comparison report written to {report_path}")


if __name__ == "__main__":
    main()