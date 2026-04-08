from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairFusion(nn.Module):
    """Fuse global features with local correspondence and geometry summaries."""

    def __init__(
        self,
        global_dim: int,
        spatial_dim: int,
        match_dim: int = 8,
        geometry_dim: int = 6,
        hidden_dim: int = 1024,
        fused_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.spatial_dim = spatial_dim
        self.match_dim = match_dim
        self.geometry_dim = geometry_dim
        self.local_summary_dim = 8

        global_in_dim = global_dim * 4
        meta_in_dim = self.local_summary_dim + match_dim + geometry_dim

        self.global_branch = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.meta_branch = nn.Sequential(
            nn.Linear(meta_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
        )

    @staticmethod
    def _fit_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
        if features.size(-1) == target_dim:
            return features
        if features.size(-1) > target_dim:
            return features[:, :target_dim]

        pad = torch.zeros(
            features.size(0),
            target_dim - features.size(-1),
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, pad], dim=-1)

    def _local_correspondence_summary(
        self,
        source_spatial: torch.Tensor,
        target_spatial: torch.Tensor,
    ) -> torch.Tensor:
        source_flat = source_spatial.flatten(2)
        target_flat = target_spatial.flatten(2)

        cosine_map = F.cosine_similarity(source_flat, target_flat, dim=1)
        abs_diff = (source_spatial - target_spatial).abs().flatten(1)
        interaction = (source_spatial * target_spatial).flatten(1)

        summary = torch.stack(
            [
                cosine_map.mean(dim=-1),
                cosine_map.std(dim=-1, unbiased=False),
                cosine_map.min(dim=-1).values,
                cosine_map.max(dim=-1).values,
                abs_diff.mean(dim=-1),
                abs_diff.std(dim=-1, unbiased=False),
                interaction.mean(dim=-1),
                interaction.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        return summary

    def forward(
        self,
        source_global: torch.Tensor,
        target_global: torch.Tensor,
        source_spatial: torch.Tensor,
        target_spatial: torch.Tensor,
        match_features: torch.Tensor | None = None,
        geometry_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = source_global.size(0)
        device = source_global.device
        dtype = source_global.dtype

        if match_features is None:
            match_features = torch.zeros(batch_size, self.match_dim, device=device, dtype=dtype)
        if geometry_features is None:
            geometry_features = torch.zeros(batch_size, self.geometry_dim, device=device, dtype=dtype)

        match_features = self._fit_feature_dim(match_features, self.match_dim)
        geometry_features = self._fit_feature_dim(geometry_features, self.geometry_dim)

        global_fusion = torch.cat(
            [
                source_global,
                target_global,
                source_global - target_global,
                source_global * target_global,
            ],
            dim=-1,
        )

        local_summary = self._local_correspondence_summary(source_spatial, target_spatial)
        meta = torch.cat([local_summary, match_features, geometry_features], dim=-1)

        global_latent = self.global_branch(global_fusion)
        meta_latent = self.meta_branch(meta)

        gate = torch.sigmoid(self.gate(torch.cat([global_latent, meta_latent], dim=-1)))
        blended = gate * global_latent + (1.0 - gate) * meta_latent
        fused = self.output_branch(torch.cat([blended, global_latent * meta_latent], dim=-1))

        return {
            "fused_features": fused,
            "global_fusion": global_fusion,
            "local_summary": local_summary,
            "match_features": match_features,
            "geometry_features": geometry_features,
        }
