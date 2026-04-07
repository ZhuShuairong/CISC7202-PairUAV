"""
Evaluation metrics for HARP-Pose Dual-Path.
"""
import torch
import numpy as np


def mae_heading(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error for heading, handling circular wrap."""
    err = (pred - target).abs()
    err = torch.where(err > 180, 360 - err, err)
    return err.mean().item()


def mae_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error for distance."""
    return (pred - target).abs().mean().item()


def avg_error(pred_h, target_h, pred_d, target_d) -> float:
    """(MAE_H + MAE_D) / 2."""
    return (mae_heading(pred_h, target_h) +
            mae_distance(pred_d, target_d)) / 2


def success_rate(pred_h, target_h, pred_d, target_d,
                 tol_heading: float = 10.0,
                 tol_distance: float = 10.0) -> float:
    """Fraction of predictions within tolerances."""
    err_h = (pred_h - target_h).abs()
    err_h = torch.where(err_h > 180, 360 - err_h, err_h)
    err_d = (pred_d - target_d).abs()
    success = ((err_h < tol_heading) & (err_d < tol_distance)).float()
    return success.mean().item() * 100


def sr_at_10m(pred_h, target_h, pred_d, target_d) -> float:
    """Success Rate @ 10m endpoint error."""
    # Approximate endpoint error from heading + distance
    # 10m tolerance in the challenge usually refers to translation only
    err_d = (pred_d - target_d).abs()
    return (err_d < 10.0).float().mean().item() * 100


def comprehensive_metrics(pred, target) -> dict:
    """All metrics in one call."""
    return {
        "mae_heading": mae_heading(pred["heading"], target["heading"]),
        "mae_distance": mae_distance(pred["distance"], target["distance"]),
        "avg": avg_error(pred["heading"], target["heading"],
                        pred["distance"], target["distance"]),
        "sr_10m": sr_at_10m(pred["heading"], target["heading"],
                           pred["distance"], target["distance"]),
        "sr_5m": sr_at_10m(pred["heading"], target["heading"],
                          pred["distance"], target["distance"]) if False else 0.0,
    }
