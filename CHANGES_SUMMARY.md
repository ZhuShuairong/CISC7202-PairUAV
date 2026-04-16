# Training Improvements Summary

## Problem
- Angle error: 0.071837 (good)
- Distance error: 0.632662 (bad - 9x worse than angle)
- NaN appearing after ~8 epochs in training

## Root Causes Identified
1. **Loss weight imbalance**: Distance loss weighted at 0.01 vs rotation at 1.25
2. **Coarse distance bins**: Only 24 bins for log-distance range [0, 5]
3. **Numerical instability**: Unclamped log-distance values causing exp() overflow
4. **No gradient clipping**: Leading to exploding gradients
5. **No NaN detection**: Training continues with invalid gradients

## Changes Made

### 1. `/workspace/training/losses.py` - Increased Distance Loss Weights
```python
# Before:
distance_cls_weight: float = 0.35
early_rotation_weight: float = 1.25
early_distance_weight: float = 0.55
mid_rotation_weight: float = 1.0
mid_distance_weight: float = 1.0
late_rotation_weight: float = 0.8
late_distance_weight: float = 1.35

# After:
distance_cls_weight: float = 0.5
early_rotation_weight: float = 0.8
early_distance_weight: float = 1.2      # 2.2x increase
mid_rotation_weight: float = 0.75
mid_distance_weight: float = 1.2        # 1.2x increase
late_rotation_weight: float = 0.6
late_distance_weight: float = 1.5       # 1.1x increase
```

### 2. `/workspace/models/heads.py` - More Distance Bins + Numerical Stability
```python
# Before:
log_distance_min: float = 0.0
log_distance_max: float = 5.0
num_bins: int = 24

# After:
log_distance_min: float = -0.5          # Extended range: exp(-0.5)=0.61m
log_distance_max: float = 5.5           # Extended range: exp(5.5)=245m
num_bins: int = 48                      # 2x more bins for finer resolution

# Added clamping in forward():
log_distance_clamped = log_distance_soft.clamp(
    self.config.log_distance_min, 
    self.config.log_distance_max
)
distance = torch.exp(log_distance_clamped)
```

### 3. `/workspace/training/train_dual_path.py` - NaN Detection + Gradient Clipping
```python
# Added NaN check before backward:
loss_val = losses["total"].item()
if not torch.isfinite(losses["total"]):
    print(f"  ⚠️  NaN/Inf detected... Skipping batch.")
    continue

# Explicit max_norm parameter:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduced batch sizes for 12GB VRAM:
PHASE_CFG = {
    1: dict(epochs=20, lr=5e-4,  bs=192, frozen=True,  gate=False),  # was 256
    2: dict(epochs=15, lr=1e-4,  bs=192, frozen=True,  gate=True),   # was 256
    3: dict(epochs=10, lr=1e-5,  bs=96,  frozen=False, gate=True),   # was 128
}
```

### 4. `/workspace/training/loss.py` - Higher Distance Weight + Confidence Clamping
```python
# Increased lambda_dist from 0.01 to 0.5 (50x increase!)
def laplace_nll(pred: dict, target: dict,
                lambda_dist: float = 0.5,  # was 0.01
                lambda_conf_reg: float = 0.01) -> dict:
    
    # Added confidence clamping to prevent numerical issues
    c = pred['confidence'].clamp(min=1e-4, max=1e4)  # was min=1e-4 only
```

### 5. `/workspace/training/train_phase1.py` - Matching Updates
```python
# Default batch size reduced for 12GB VRAM
p.add_argument('--batch-size', type=int, default=48)  # was 64

# Distance loss weight increased
p.add_argument('--lambda-dist', type=float, default=0.5)  # was 0.01
```

### 6. `/workspace/models/baseline.py` - Created Missing File
Created complete baseline model implementation that was referenced but missing.

## Expected Impact

| Metric | Before | Expected After | Notes |
|--------|--------|----------------|-------|
| Angle Error | 0.072 | 0.05-0.06 | Slight improvement |
| Distance Error | 0.633 | 0.20-0.30 | 2-3x improvement |
| Final Score | ~0.35 | ~0.13-0.18 | Significant improvement |
| NaN Issues | Yes (~8 epochs) | No | Fixed with clamping + detection |

## Is 0.05 Achievable?

**Realistic expectations:**
- **Short-term (these fixes)**: 0.12-0.18 final score
- **Medium-term (more tuning)**: 0.08-0.12
- **Long-term (ensembling, TTA)**: 0.05-0.08

The 0.05 target requires:
1. These foundational fixes ✓
2. Hyperparameter tuning (learning rates, bin counts)
3. Test-time augmentation (multi-crop, flips)
4. Model ensembling (3-5 checkpoints)
5. Possibly larger backbone or attention mechanisms

## Usage on AutoDL

```bash
# Phase 1 (wide path only, cached features)
python training/train_dual_path.py \
    --cache /path/to/features \
    --phase 1 \
    --checkpoint checkpoints/phase1.pt

# Phase 2 (dual-path with gates)
python training/train_dual_path.py \
    --cache /path/to/features \
    --phase 2 \
    --checkpoint checkpoints/phase1.pt

# Phase 3 (full finetuning, raw images)
python training/train_dual_path.py \
    --data-root /path/to/data \
    --phase 3 \
    --raw true \
    --checkpoint checkpoints/phase2.pt
```

## Monitoring

Watch for these signs of healthy training:
- ✅ Loss decreases smoothly without spikes
- ✅ No "NaN/Inf detected" warnings
- ✅ Distance error converges faster than before
- ✅ Validation metrics improve consistently

If you see NaN warnings frequently:
- Reduce learning rate by 2x
- Check data for outliers (distance < 0.5m or > 200m)
- Verify label quality
