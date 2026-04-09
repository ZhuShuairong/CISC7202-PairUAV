from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeightConfig:
    distance_cls_weight: float = 0.35
    early_rotation_weight: float = 1.25
    early_distance_weight: float = 0.55
    mid_rotation_weight: float = 1.0
    mid_distance_weight: float = 1.0
    late_rotation_weight: float = 0.8
    late_distance_weight: float = 1.35


def _wrapped_angle_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    delta = pred_deg - target_deg
    return ((delta + 180.0) % 360.0) - 180.0


def _log_distance(target_distance: torch.Tensor, min_distance: float) -> torch.Tensor:
    return torch.log(target_distance.clamp(min=min_distance))


def _apply_uncertainty(loss_per_sample: torch.Tensor, log_var: torch.Tensor | None) -> torch.Tensor:
    if log_var is None:
        return loss_per_sample.mean()
    bounded_log_var = log_var.clamp(-6.0, 6.0)
    precision = torch.exp(-bounded_log_var)
    return (precision * loss_per_sample + bounded_log_var).mean()


class PairUAVLoss:
    """Multi-task loss for angle and scale-aware distance prediction."""

    def __init__(
        self,
        log_distance_min: float,
        log_distance_max: float,
        num_bins: int,
        min_distance: float = 1.0,
        smooth_l1_beta: float = 0.05,
        weights: LossWeightConfig | None = None,
    ) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")

        self.log_distance_min = log_distance_min
        self.log_distance_max = log_distance_max
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.smooth_l1_beta = smooth_l1_beta
        self.weights = weights or LossWeightConfig()

        self.bin_centers = torch.linspace(log_distance_min, log_distance_max, num_bins, dtype=torch.float32)

    def _distance_bin_targets(self, target_log_distance: torch.Tensor) -> torch.Tensor:
        centers = self.bin_centers.to(device=target_log_distance.device, dtype=target_log_distance.dtype)
        edges = (centers[:-1] + centers[1:]) * 0.5
        indices = torch.bucketize(target_log_distance, edges)
        return indices.clamp(min=0, max=self.num_bins - 1)

    def _task_weights(self, progress: float, stage_name: str) -> tuple[float, float]:
        if progress < 1.0 / 3.0:
            rotation_weight = self.weights.early_rotation_weight
            distance_weight = self.weights.early_distance_weight
        elif progress < 2.0 / 3.0:
            rotation_weight = self.weights.mid_rotation_weight
            distance_weight = self.weights.mid_distance_weight
        else:
            rotation_weight = self.weights.late_rotation_weight
            distance_weight = self.weights.late_distance_weight

        stage = stage_name.upper().strip()
        if stage == "A":
            distance_weight *= 0.9
        elif stage == "C":
            distance_weight *= 1.15

        return rotation_weight, distance_weight

    def __call__(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        progress: float,
        stage_name: str,
    ) -> dict[str, torch.Tensor]:
        heading_target_deg = target["heading"]
        distance_target = target["distance"].clamp(min=self.min_distance)
        log_distance_target = _log_distance(distance_target, min_distance=self.min_distance)

        target_rad = torch.deg2rad(heading_target_deg)
        target_sin = torch.sin(target_rad)
        target_cos = torch.cos(target_rad)

        pred_sin = prediction["heading_sin"]
        pred_cos = prediction["heading_cos"]
        pred_heading_deg = prediction["heading_deg"]

        cosine_alignment = pred_sin * target_sin + pred_cos * target_cos
        rotation_loss_per_sample = 1.0 - cosine_alignment.clamp(min=-1.0, max=1.0)
        rotation_loss = _apply_uncertainty(
            rotation_loss_per_sample,
            prediction.get("rotation_log_var"),
        )

        angle_l1 = _wrapped_angle_deg(pred_heading_deg, heading_target_deg).abs().mean()

        pred_log_distance = prediction["log_distance"]
        distance_reg_per_sample = F.smooth_l1_loss(
            pred_log_distance,
            log_distance_target,
            reduction="none",
            beta=self.smooth_l1_beta,
        )

        distance_logits = prediction["distance_logits"]
        distance_bins = self._distance_bin_targets(log_distance_target)
        distance_cls_per_sample = F.cross_entropy(distance_logits, distance_bins, reduction="none")

        distance_base_per_sample = distance_reg_per_sample + self.weights.distance_cls_weight * distance_cls_per_sample
        distance_loss = _apply_uncertainty(
            distance_base_per_sample,
            prediction.get("distance_log_var"),
        )

        rotation_weight, distance_weight = self._task_weights(progress, stage_name)
        total = rotation_weight * rotation_loss + distance_weight * distance_loss

        return {
            "total": total,
            "rotation": rotation_loss,
            "distance": distance_loss,
            "distance_reg": distance_reg_per_sample.mean(),
            "distance_cls": distance_cls_per_sample.mean(),
            "angle_l1": angle_l1,
            "weight_rotation": torch.tensor(rotation_weight, device=total.device),
            "weight_distance": torch.tensor(distance_weight, device=total.device),
        }
