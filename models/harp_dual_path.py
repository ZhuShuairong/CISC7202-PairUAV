"""
HARP Dual-Path Architecture — Wide Path + Deep Path with Cross-Residual Gating
~7.5M trainable params (frozen backbone), ~33M trainable (full finetune)
"""
import torch
import torch.nn as nn
import torchvision.models as models
from models.baseline import siamese_fusion
from models.fusion import DenseCorrelationVolume


class BackboneSpatial(nn.Module):
    """ResNet-50 variant that returns 7×7×2048 spatial features."""
    def __init__(self, frozen: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool and fc — return layer4 output (7×7×2048)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.frozen = frozen
        if frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True):
        if self.frozen:
            mode = False
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, 2048, 7, 7)"""
        with torch.set_grad_enabled(not self.frozen):
            return self.features(x)


class WidePath(nn.Module):
    """Fast global estimate from fused features.

    (B, 512, 7, 7) → Conv 512→256→128 → GAP → MLP(128→64→2)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # [heading, distance]
        )
        self.conf_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv2(self.conv1(x))       # (B, 128, 7, 7)
        pooled = x.mean(dim=[2, 3])          # (B, 128)
        out = self.regressor(pooled)          # (B, 2)
        confidence = torch.exp(self.conf_head(pooled) / 10.0)  # always > 0
        return {
            'heading': out[:, 0],             # unbounded
            'distance': torch.clamp(out[:, 1], min=0.0),
            'confidence': confidence.squeeze(-1),
        }


class DeepPath(nn.Module):
    """Spatial refinement path — processes full 7×7 grid.

    (B, 512, 7, 7) → Conv 512→256→128→256 → per-region heads
    Outputs: precision-weighted aggregated heading & distance.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pose_head = nn.Conv2d(256, 2, 1)       # (B, 2, 7, 7)
        self.precision_head = nn.Conv2d(256, 1, 1)  # (B, 1, 7, 7)

    def forward(self, x: torch.Tensor):
        x = self.conv3(self.conv2(self.conv1(x)))   # (B, 256, 7, 7)
        pose = self.pose_head(x)                     # (B, 2, 7, 7)
        raw_prec = self.precision_head(x)            # (B, 1, 7, 7)
        precision = torch.exp(raw_prec)              # always positive

        w = precision.squeeze(1)                     # (B, 7, 7)
        w_sum = w.sum(dim=[1, 2], keepdim=True) + 1e-8
        w_norm = w / w_sum                           # normalised

        h = (pose[:, 0] * w_norm).sum(dim=[1, 2])   # (B,)
        d = (pose[:, 1] * w_norm).sum(dim=[1, 2]).clamp(min=0.0)
        return h, d, w.squeeze(1)


class CrossResidualGate(nn.Module):
    """fXReNet-style gated-tanh.

    gate = σ(W_g · y_wide) ⊙ tanh(W_c · y_wide)  →  projects to [δθ, δd]

    Bidirectional: gate activation also modulates wide-path confidence.
    """
    def __init__(self, wide_out_dim: int = 3, gate_hidden: int = 64):
        super().__init__()
        self.w_g = nn.Linear(wide_out_dim, gate_hidden)
        self.w_c = nn.Linear(wide_out_dim, gate_hidden)
        self.proj = nn.Sequential(
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2),
        )
        self.conf_feedback = nn.Linear(gate_hidden, 1)

    def forward(self, wide_out: dict):
        y = torch.stack([
            wide_out['heading'],
            wide_out['distance'],
            wide_out['confidence'],
        ], dim=-1)                                          # (B, 3)
        gate_sig = torch.sigmoid(self.w_g(y))                # (B, H)
        gate_tanh = torch.tanh(self.w_c(y))                  # (B, H)
        gate = gate_sig * gate_tanh                          # (B, H)
        delta = self.proj(gate)                              # (B, 2)
        conf_mod = torch.sigmoid(self.conf_feedback(gate)).squeeze(-1)
        return delta, conf_mod


class HARPDualPath(nn.Module):
    """Complete dual-path: Wide + Deep + Cross-Residual Gates.

    Phases:
        1 — wide path only (frozen backbone / cached features)
        2 — full dual-path with gates (frozen backbone)
        3 — joint fine-tuning (unfrozen backbone, low LR)
    """
    def __init__(self, frozen: bool = True, use_gate: bool = True):
        super().__init__()
        self.backbone = BackboneSpatial(frozen=frozen)
        
        self.corr_volume = DenseCorrelationVolume(in_channels=2048, downsample_dim=256)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(49, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.wide = WidePath()
        self.deep = DeepPath()
        self.use_gate = use_gate
        if use_gate:
            self.gate = CrossResidualGate(wide_out_dim=3, gate_hidden=64)
        self.phase = 1

    # ----- raw-image forward (Phase 3, unfrozen backbone) -----
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        f_s = self.backbone(source)    # (B, 2048, 7, 7)
        f_t = self.backbone(target)
        volume = self.corr_volume(f_s, f_t)  # (B, 49, 7, 7)
        spatial = self.fusion(volume)
        return self._forward_spatial(spatial)

    # ----- cached-feature forward (Phase 1-2) -----
    def forward_features(self, feat_s: torch.Tensor,
                         feat_t: torch.Tensor) -> dict:
        """Accept pre-extracted 7×7×2048 tensors directly."""
        volume = self.corr_volume(feat_s, feat_t)
        spatial = self.fusion(volume)
        return self._forward_spatial(spatial)

    def _forward_spatial(self, spatial: torch.Tensor) -> dict:
        wide_out = self.wide(spatial)

        if self.phase == 1:
            return wide_out

        deep_h, deep_d, prec_map = self.deep(spatial)

        if self.use_gate:
            delta, conf_mod = self.gate(wide_out)
            return {
                'heading': wide_out['heading'] + delta[:, 0],
                'distance': (wide_out['distance'] + delta[:, 1]).clamp(min=0.0),
                'confidence': wide_out['confidence'] * conf_mod,
                'wide_heading': wide_out['heading'],
                'wide_distance': wide_out['distance'],
                'deep_heading': deep_h,
                'deep_distance': deep_d,
                'gate_delta': delta,
                'precision_map': prec_map,
            }

        return {
            'heading': wide_out['heading'] + deep_h,
            'distance': (wide_out['distance'] + deep_d).clamp(min=0.0),
            'confidence': wide_out['confidence'],
            'wide_heading': wide_out['heading'],
            'wide_distance': wide_out['distance'],
            'deep_heading': deep_h,
            'deep_distance': deep_d,
        }


def siamese_fusion_spatial(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """4-way spatial fusion: [f_s, f_t, f_s-f_t, f_s⊙f_t] → (B, 8192, 7, 7)."""
    diff = f_s - f_t
    interaction = f_s * f_t
    return torch.cat([f_s, f_t, diff, interaction], dim=1)
