"""
Official PairUAV 2026 Baseline re-implementation (reference).

Architecture:
  Siamese ResNet-50 (ImageNet pretrained, shared weights)
    → 3-way feature fusion [Fs, Ft, Fs−Ft, Fs⊙Ft]  (8192-D)
    → MLP: 8192→4096(Relu)→1024(Relu)→2 [angle(°), distance(m)]

Loss:
  L_θ = |sin(θ̂ − θ*)|     (circular-aware — handles 360° wrapping)
  L_d = |d̂ − d*|           (L1 distance)
  L   = L_θ + 0.01 · L_d    (distance weighted down to not dominate)

Training: Adam, lr=1e-4, batch=32
Output: result.txt with "angle, distance" per line
"""
import torch
import torch.nn as nn


class BaselineBackbone(nn.Module):
    """Single ResNet-50 branch (ImageNet pretrained, features only)."""
    
    def __init__(self, frozen: bool = False):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove classification head — keep up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
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
        """(B, 3, 224, 224) → (B, 2048)"""
        with torch.set_grad_enabled(not self.frozen):
            return self.features(x).squeeze(-1).squeeze(-1)  # (B, 2048)


def siamese_fusion(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """3-way fusion mimicking relative motion reasoning.
    
    Concatenates: [Fs, Ft, Fs−Ft, Fs⊙Ft]
    - Fs, Ft: absolute appearance features
    - Fs−Ft: relative difference (motion cue)
    - Fs⊙Ft: element-wise interaction (correlation)
    
    Each is 2048-D → output is 8192-D.
    """
    diff = fs - ft          # relative motion
    interaction = fs * ft   # appearance correlation
    return torch.cat([fs, ft, diff, interaction], dim=1)  # (B, 8192)


class BaselineHead(nn.Module):
    """MLP regression head: 8192 → 4096 → 1024 → 2."""
    
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        # Explicit final regression layer avoids accidental hidden-state misuse.
        self.out = nn.Linear(1024, 2)  # [angle, distance]
    
    def forward(self, x: torch.Tensor) -> dict:
        hidden = self.hidden(x)
        raw = self.out(hidden)  # (B, 2)
        if raw.ndim != 2 or raw.size(1) != 2:
            raise RuntimeError(f"BaselineHead must output shape (B, 2), got {tuple(raw.shape)}")
        heading_deg = raw[:, 0]  # unbounded (angle in degrees)
        distance = torch.clamp(raw[:, 1], min=0.0)  # distance ≥ 0
        return {'heading_deg': heading_deg, 'distance': distance}


class PairUAVBaseline(nn.Module):
    """Complete baseline: Siamese R50 + fusion + MLP."""
    
    def __init__(self):
        super().__init__()
        self.backbone = BaselineBackbone()  # frozen=False by default
        self.head = BaselineHead()
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            source: (B, 3, 224, 224) — reference drone view
            target: (B, 3, 224, 224) — target drone view
        """
        fs = self.backbone(source)  # (B, 2048)
        ft = self.backbone(target)  # (B, 2048) — shared weights
        fused = siamese_fusion(fs, ft)  # (B, 8192)
        return self.head(fused)
    
    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[Baseline] Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M, Frozen={frozen/1e6:.2f}M")
        return f"Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M"


# ----- Loss functions (official baseline) -----

def baseline_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Circular-aware angular loss: L_θ = |sin(θ̂ − θ*)|.
    
    Automatically handles 360° wrapping: sin(179°−(−181°)) ≈ sin(0°) = 0.
    This is better than MSE for angular regression.
    """
    diff_rad = torch.deg2rad(pred - target)
    return torch.abs(torch.sin(diff_rad)).mean()


def baseline_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance loss."""
    return nn.functional.l1_loss(pred, target)


def baseline_total_loss(pred: dict, target: dict, lambda_dist: float = 0.01) -> dict:
    """L = L_θ + 0.01 · L_d.
    
    Why λ_dist = 0.01?
    Without it, L1 distance (metres, range ~0-130) dominates the sine loss (range 0-1),
    so the network learns to predict distance well but ignores heading entirely.
    """
    l_angle = baseline_angle_loss(pred['heading_deg'], target['heading'])
    l_dist  = baseline_distance_loss(pred['distance'], target['distance'])
    l_total = l_angle + lambda_dist * l_dist
    return {'total': l_total, 'angle': l_angle, 'dist': l_dist}
