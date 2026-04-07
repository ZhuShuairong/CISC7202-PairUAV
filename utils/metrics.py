"""
Evaluation metrics for HARP-Pose Dual-Path.
"""
import torch


def wrapped_heading_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise wrapped absolute heading error in degrees."""
    err = (pred - target).abs()
    return torch.where(err > 180, 360 - err, err)


def mae_heading(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error for heading, handling circular wrap."""
    return wrapped_heading_error(pred, target).mean().item()


def mae_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error for distance."""
    return (pred - target).abs().mean().item()


def avg_error(pred_h, target_h, pred_d, target_d) -> float:
    """(MAE_H + MAE_D) / 2."""
    return (mae_heading(pred_h, target_h) +
            mae_distance(pred_d, target_d)) / 2


def angle_rel_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative angular error using wrapped error normalised by 180°."""
    err = wrapped_heading_error(pred, target)
    return (err / 180.0).mean().item()


def distance_rel_error(pred: torch.Tensor, target: torch.Tensor,
                       min_distance_denominator: float = 1.0) -> float:
    """Relative distance error: |d_hat-d| / max(|d|, min_distance_denominator)."""
    denom = target.abs().clamp(min=min_distance_denominator)
    rel = (pred - target).abs() / denom
    return rel.mean().item()


def final_score(pred_h: torch.Tensor, target_h: torch.Tensor,
                pred_d: torch.Tensor, target_d: torch.Tensor,
                min_distance_denominator: float = 1.0) -> float:
    """Official-style score proxy: (angle_rel_error + distance_rel_error) / 2."""
    a = angle_rel_error(pred_h, target_h)
    d = distance_rel_error(pred_d, target_d,
                           min_distance_denominator=min_distance_denominator)
    return (a + d) / 2.0


def success_rate(pred_h, target_h, pred_d, target_d,
                 tol_heading: float = 10.0,
                 tol_distance: float = 10.0) -> float:
    """Fraction of predictions within tolerances."""
    err_h = wrapped_heading_error(pred_h, target_h)
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
    """Compute MAE metrics and benchmark-style relative errors."""
    pred_h = pred["heading"]
    tgt_h = target["heading"]
    pred_d = pred["distance"]
    tgt_d = target["distance"]

    return {
        "mae_heading": mae_heading(pred_h, tgt_h),
        "mae_distance": mae_distance(pred_d, tgt_d),
        "avg": avg_error(pred_h, tgt_h, pred_d, tgt_d),
        "angle_rel_error": angle_rel_error(pred_h, tgt_h),
        "distance_rel_error": distance_rel_error(pred_d, tgt_d),
        "final_score": final_score(pred_h, tgt_h, pred_d, tgt_d),
        "sr_10m": sr_at_10m(pred_h, tgt_h, pred_d, tgt_d),
        "sr_5m": sr_at_10m(pred_h, tgt_h, pred_d, tgt_d) if False else 0.0,
    }
