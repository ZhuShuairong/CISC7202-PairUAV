"""Baseline model and loss functions for PairUAV."""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineBackbone(nn.Module):
    """ResNet-50 backbone returning 2048-d feature vector."""
    
    def __init__(self, frozen: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc
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
        with torch.set_grad_enabled(not self.frozen):
            out = self.features(x)
            return out.squeeze(-1).squeeze(-1)  # (B, 2048)


def siamese_fusion(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """[fs, ft, fs-ft, fs⊙ft] → (B, 8192)"""
    diff = fs - ft
    interaction = fs * ft
    return torch.cat([fs, ft, diff, interaction], dim=1)


class BaselineHead(nn.Module):
    """Simple MLP head for baseline."""
    
    def __init__(self, fused_dim: int = 8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2),  # [heading, distance]
        )
    
    def forward(self, fused: torch.Tensor) -> dict:
        out = self.net(fused)
        return {
            'heading_deg': out[:, 0],
            'distance': torch.clamp(out[:, 1], min=0.0),
        }


class PairUAVBaseline(nn.Module):
    """Baseline: Siamese R50 + fusion + MLP."""
    
    def __init__(self):
        super().__init__()
        self.backbone = BaselineBackbone(frozen=True)
        self.head = BaselineHead()
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        fs = self.backbone(source)
        ft = self.backbone(target)
        fused = siamese_fusion(fs, ft)
        return self.head(fused)
    
    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Baseline] Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M")
        return f"Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M"


def baseline_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """|sin(θ̂ − θ*)| — circular-aware angular loss."""
    diff_rad = torch.deg2rad(pred - target)
    return torch.abs(torch.sin(diff_rad)).mean()


def baseline_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance loss."""
    return torch.abs(pred - target).mean()


def baseline_total_loss(pred: dict, target: dict, 
                        lambda_dist: float = 0.01) -> torch.Tensor:
    """Total loss = L_angle + λ_dist · L_dist."""
    l_angle = baseline_angle_loss(pred['heading_deg'], target['heading'])
    l_dist = baseline_distance_loss(pred['distance'], target['distance'])
    return l_angle + lambda_dist * l_dist
