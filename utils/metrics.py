"""Evaluation metrics and checkpoint ranking utilities for PairUAV."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch


def wrapped_heading_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise wrapped absolute heading error in degrees."""
    error = (pred - target).abs()
    return torch.where(error > 180.0, 360.0 - error, error)


def mae_heading(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(wrapped_heading_error(pred, target).mean().item())


def mae_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean().item())


def avg_error(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    pred_d: torch.Tensor,
    target_d: torch.Tensor,
) -> float:
    return 0.5 * (mae_heading(pred_h, target_h) + mae_distance(pred_d, target_d))


def angle_rel_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    wrapped = wrapped_heading_error(pred, target)
    return float((wrapped / 180.0).mean().item())


def distance_rel_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_distance_denominator: float = 1.0,
) -> float:
    denominator = target.abs().clamp(min=min_distance_denominator)
    relative = (pred - target).abs() / denominator
    return float(relative.mean().item())


def final_score(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    pred_d: torch.Tensor,
    target_d: torch.Tensor,
    min_distance_denominator: float = 1.0,
) -> float:
    angle = angle_rel_error(pred_h, target_h)
    distance = distance_rel_error(pred_d, target_d, min_distance_denominator=min_distance_denominator)
    return 0.5 * (angle + distance)


def success_rate(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    pred_d: torch.Tensor,
    target_d: torch.Tensor,
    tol_heading: float = 10.0,
    tol_distance: float = 10.0,
) -> float:
    err_h = wrapped_heading_error(pred_h, target_h)
    err_d = (pred_d - target_d).abs()
    success = ((err_h <= tol_heading) & (err_d <= tol_distance)).float()
    return float(success.mean().item() * 100.0)


def sr_at_10m(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    pred_d: torch.Tensor,
    target_d: torch.Tensor,
) -> float:
    _ = pred_h, target_h
    err_d = (pred_d - target_d).abs()
    return float((err_d <= 10.0).float().mean().item() * 100.0)


def comprehensive_metrics(
    pred: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    min_distance_denominator: float = 1.0,
) -> dict[str, float]:
    pred_h = pred["heading"]
    tgt_h = target["heading"]
    pred_d = pred["distance"]
    tgt_d = target["distance"]

    angle_rel = angle_rel_error(pred_h, tgt_h)
    distance_rel = distance_rel_error(
        pred_d,
        tgt_d,
        min_distance_denominator=min_distance_denominator,
    )

    return {
        "mae_heading": mae_heading(pred_h, tgt_h),
        "mae_distance": mae_distance(pred_d, tgt_d),
        "avg": avg_error(pred_h, tgt_h, pred_d, tgt_d),
        "angle_rel_error": angle_rel,
        "distance_rel_error": distance_rel,
        "final_score": 0.5 * (angle_rel + distance_rel),
        "sr_10m": sr_at_10m(pred_h, tgt_h, pred_d, tgt_d),
        "sr_5m": float(((pred_d - tgt_d).abs() <= 5.0).float().mean().item() * 100.0),
    }


def is_better_result(
    current: Mapping[str, float],
    best: Mapping[str, float] | None,
    eps: float = 1e-9,
) -> bool:
    """Leaderboard-priority comparator: final_score > dist_rel > angle_rel (all lower is better)."""
    if best is None:
        return True

    keys = ("final_score", "distance_rel_error", "angle_rel_error")
    for key in keys:
        current_value = float(current[key])
        best_value = float(best[key])
        if current_value < best_value - eps:
            return True
        if current_value > best_value + eps:
            return False

    if "val_total_loss" in current and "val_total_loss" in best:
        return float(current["val_total_loss"]) < float(best["val_total_loss"]) - eps
    return False


def parse_result_line(line: str) -> tuple[float, float]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty line is not allowed in result files")

    if "," in stripped:
        parts = [part.strip() for part in stripped.split(",")]
    else:
        parts = stripped.split()

    if len(parts) != 2:
        raise ValueError(
            "Each result line must contain exactly two numeric values: angle,distance "
            f"(got {len(parts)} fields from line '{line.rstrip()}')"
        )

    try:
        angle = float(parts[0])
        distance = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Non-numeric value in result line: '{line.rstrip()}'") from exc

    if not math.isfinite(angle) or not math.isfinite(distance):
        raise ValueError(f"Non-finite value in result line: '{line.rstrip()}'")

    return angle, distance


def read_result_file(path: Path | str) -> dict[str, torch.Tensor]:
    result_path = Path(path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    headings: list[float] = []
    distances: list[float] = []

    with result_path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            angle, distance = parse_result_line(raw_line)
            headings.append(angle)
            distances.append(distance)

    if not headings:
        raise ValueError(f"Result file is empty: {result_path}")

    return {
        "heading": torch.tensor(headings, dtype=torch.float32),
        "distance": torch.tensor(distances, dtype=torch.float32),
    }


def write_result_file(
    path: Path | str,
    heading: torch.Tensor,
    distance: torch.Tensor,
    delimiter: str = "comma",
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heading_cpu = heading.detach().float().cpu()
    distance_cpu = distance.detach().float().cpu()
    if heading_cpu.numel() != distance_cpu.numel():
        raise ValueError("Heading and distance tensors must have the same length")

    if delimiter not in {"comma", "space"}:
        raise ValueError(f"Unsupported delimiter: {delimiter}")
    sep = ", " if delimiter == "comma" else " "

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for angle_value, distance_value in zip(heading_cpu.tolist(), distance_cpu.tolist()):
            handle.write(f"{float(angle_value):.6f}{sep}{float(distance_value):.6f}\n")


def evaluate_result_files(
    result_path: Path | str,
    truth_path: Path | str,
    min_distance_denominator: float = 1.0,
) -> dict[str, float]:
    prediction = read_result_file(result_path)
    truth = read_result_file(truth_path)

    if prediction["heading"].numel() != truth["heading"].numel():
        raise ValueError(
            "Prediction/truth line count mismatch: "
            f"pred={prediction['heading'].numel()} truth={truth['heading'].numel()}"
        )

    return comprehensive_metrics(
        prediction,
        truth,
        min_distance_denominator=min_distance_denominator,
    )
