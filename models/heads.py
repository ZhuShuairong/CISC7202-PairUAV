from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DistanceHeadConfig:
    log_distance_min: float = -0.5
    log_distance_max: float = 5.5
    num_bins: int = 48


class RotationHead(nn.Module):
    """Specialized circular-angle head that predicts normalized sin/cos."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        with_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.with_uncertainty = with_uncertainty

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.vector_head = nn.Linear(hidden_dim, 2)
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if with_uncertainty else None

    def forward(self, fused_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(fused_features)
        raw_vector = self.vector_head(hidden)
        unit_vector = F.normalize(raw_vector, p=2, dim=-1, eps=1e-6)

        heading_sin = unit_vector[:, 0]
        heading_cos = unit_vector[:, 1]
        heading_rad = torch.atan2(heading_sin, heading_cos)
        heading_deg = torch.rad2deg(heading_rad)

        output: dict[str, torch.Tensor] = {
            "heading_sin": heading_sin,
            "heading_cos": heading_cos,
            "heading_rad": heading_rad,
            "heading_deg": heading_deg,
        }

        if self.with_uncertainty and self.uncertainty_head is not None:
            output["rotation_log_var"] = self.uncertainty_head(hidden).squeeze(-1).clamp(-6.0, 6.0)

        return output


class DistanceHead(nn.Module):
    """Scale-aware distance head with bin classification and residual regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        config: DistanceHeadConfig | None = None,
        with_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or DistanceHeadConfig()
        self.with_uncertainty = with_uncertainty

        if self.config.num_bins < 2:
            raise ValueError("DistanceHead requires at least 2 bins.")

        bin_centers = torch.linspace(
            self.config.log_distance_min,
            self.config.log_distance_max,
            steps=self.config.num_bins,
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", bin_centers, persistent=True)
        self.bin_width = float(
            (self.config.log_distance_max - self.config.log_distance_min) / (self.config.num_bins - 1)
        )

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.bin_logits_head = nn.Linear(hidden_dim, self.config.num_bins)
        self.bin_residual_head = nn.Linear(hidden_dim, self.config.num_bins)
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if with_uncertainty else None

    def target_to_bins(self, target_log_distance: torch.Tensor) -> torch.Tensor:
        """Map target log-distance to nearest distance-bin index."""
        edges = (self.bin_centers[:-1] + self.bin_centers[1:]) * 0.5
        bin_index = torch.bucketize(target_log_distance, edges)
        return bin_index.clamp(min=0, max=self.config.num_bins - 1)

    def forward(self, fused_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(fused_features)

        distance_logits = self.bin_logits_head(hidden)
        distance_prob = F.softmax(distance_logits, dim=-1)

        # Residual is constrained to roughly half a bin to keep predictions stable.
        raw_residual = self.bin_residual_head(hidden)
        residual_scale = 0.5 * self.bin_width
        distance_residual_bins = torch.tanh(raw_residual) * residual_scale

        bin_centers = self.bin_centers.unsqueeze(0)
        log_distance_soft = torch.sum(distance_prob * (bin_centers + distance_residual_bins), dim=-1)

        hard_bin = torch.argmax(distance_logits, dim=-1)
        hard_bin_center = self.bin_centers.index_select(0, hard_bin)
        hard_residual = torch.gather(distance_residual_bins, 1, hard_bin.unsqueeze(1)).squeeze(1)
        log_distance_hard = hard_bin_center + hard_residual

        # Clamp log_distance to valid range before exp to prevent NaN/Inf
        log_distance_clamped = log_distance_soft.clamp(self.config.log_distance_min, self.config.log_distance_max)
        distance = torch.exp(log_distance_clamped)

        output: dict[str, torch.Tensor] = {
            "distance_logits": distance_logits,
            "distance_prob": distance_prob,
            "distance_residual_bins": distance_residual_bins,
            "log_distance": log_distance_clamped,
            "log_distance_hard": log_distance_hard,
            "distance": distance,
            "distance_bin_centers": self.bin_centers,
        }

        if self.with_uncertainty and self.uncertainty_head is not None:
            output["distance_log_var"] = self.uncertainty_head(hidden).squeeze(-1).clamp(-6.0, 6.0)

        return output
