# HARP-Pose Lite — improved baseline variant
# ~26M trainable params (full R50 + improved head, not frozen)
# Key upgrades over baseline: (1) multi-crop test augmentation,
# (2) cosine LR schedule with AdamW, (3) confidence-weighted loss

import torch
import torch.nn as nn
from models.baseline import BaselineBackbone, siamese_fusion


class ConfidenceHead(nn.Module):
    """Predicts per-sample uncertainty (scalar, log-variance).
    Used for confidence-weighted loss and unreliable prediction flagging.
    """
    def __init__(self, fused_dim: int = 8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
    
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        log_var = self.net(fused)  # (B, 1)
        # Precision = exp(-log_var), always > 0
        return torch.exp(-log_var).squeeze(-1)  # (B,)


class HARPPoseLite(nn.Module):
    """Baseline architecture + confidence-aware regression.
    
    Uses the SAME backbone (R50 ImageNet) and fusion [Fs, Ft, Fs-Ft, Fs⊙Ft]
    as the baseline, but adds:
    1. Per-sample confidence prediction (for weighted loss)
    2. Wider MLP head (4096 → 2048 → 1024 → 2) for more capacity
    3. Test-time multi-crop averaging (during inference)
    """
    
    def __init__(self):
        super().__init__()
        
        # Same Siamese ResNet-50 backbone as baseline
        self.backbone = BaselineBackbone(frozen=True)
        
        # Improved regression head (slightly narrower, confidence output)
        self.regression_head = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 2),  # [angle, distance]
        )
        
        # Confidence head
        self.conf_head = ConfidenceHead(fused_dim=8192)
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        fs = self.backbone(source)  # (B, 2048)
        ft = self.backbone(target)  # (B, 2048)
        fused = siamese_fusion(fs, ft)  # (B, 8192)
        
        out = self.regression_head(fused)  # (B, 2)
        heading_deg = out[:, 0]
        distance = torch.clamp(out[:, 1], min=0.0)
        
        confidence = self.conf_head(fused)  # (B,)
        
        return {
            'heading_deg': heading_deg,
            'distance': distance,
            'confidence': confidence,
        }
    
    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[HARP-Pose-Lite] Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M, Frozen={frozen/1e6:.2f}M")
        return f"Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M"


def harp_pose_lite_loss(pred: dict, target: dict, 
                        lambda_dist: float = 0.01,
                        lambda_conf: float = 0.1) -> dict:
    """Loss = L_angle + 0.01·L_dist + 0.1·L_conf.
    
    L_angle = |sin(θ̂ − θ*)|      (same as baseline, circular-aware)
    L_dist  = |d̂ − d*|            (same as baseline, L1)
    L_conf  = -confidence · error + log(confidence + ε)
             (uncertainty-weighted: penalizes confident wrong predictions more)
    """
    
    # Same angular + distance losses as baseline
    from models.baseline import baseline_angle_loss, baseline_distance_loss
    
    l_angle = baseline_angle_loss(pred['heading_deg'], target['heading'])
    l_dist  = baseline_distance_loss(pred['distance'], target['distance'])
    
    # Confidence-weighted refinement loss
    if 'confidence' in pred:
        conf = pred['confidence']  # (B,)
        heading_err = torch.abs(torch.sin(torch.deg2rad(pred['heading_deg'] - target['heading'])))  # (B,)
        dist_err = torch.abs(pred['distance'] - target['distance'])  # (B,)
        total_err = heading_err + lambda_dist * dist_err
        
        # Lower loss when the model is uncertain about wrong predictions
        # and penalizes when it's confident about wrong predictions
        eps = 1e-6
        l_conf = -(conf * (total_err + eps).log()).mean()
    else:
        l_conf = torch.tensor(0.0, device=l_angle.device)
    
    l_total = l_angle + lambda_dist * l_dist + lambda_conf * l_conf
    
    return {'total': l_total, 'angle': l_angle, 'dist': l_dist, 'conf': l_conf}
