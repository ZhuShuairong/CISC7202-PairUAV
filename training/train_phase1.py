"""
Train Phase 1: Baseline + HARP-Pose-Lite

Both use the same architecture (Siamese R50 + fusion + MLP).
HARP-Pose-Lite adds confidence prediction and cosine LR schedule.

Optimiser: Adam, lr=1e-4, batch=32
Loss: |sin(θ̂−θ*)| + 0.01·|d̂−d*|

Usage:
    python training/train_phase1.py \
    --data-root /path/to/PairUAV \
        --model baseline  # or harp-lite
        --epochs 30
"""

import argparse
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PairDataset, resolve_train_view_dir
from models.baseline import PairUAVBaseline, baseline_total_loss
from models.harp_pose_lite import HARPPoseLite, harp_pose_lite_loss


def get_parser():
    p = argparse.ArgumentParser(description='Train baseline or HARP-Pose-Lite')
    p.add_argument('--data-root', '--university-release', '--data', dest='data_root',
                   type=str, required=True, help='Path to PairUAV training root')
    p.add_argument('--model', type=str, default='baseline',
                   choices=['baseline', 'harp-lite'],
                   help='Model variant')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4, help='Adam lr')
    p.add_argument('--weight-decay', type=float, default=0.01, help='AdamW wd')
    p.add_argument('--lambda-dist', type=float, default=0.01, help='Distance loss weight')
    p.add_argument('--lambda-conf', type=float, default=0.1, help='Conf loss weight (Lite only)')
    p.add_argument('--warmup-epochs', type=int, default=3, help='Linear warmup')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None, help='Checkpoint path')
    p.add_argument('--num-workers', type=int, default=16)
    return p


def cosine_lr(epoch, total, warmup, base_lr):
    if warmup > 0 and epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    if total <= warmup:
        return base_lr
    progress = (epoch - warmup) / (total - warmup)
    progress = max(0.0, min(1.0, progress))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def get_amp_dtype():
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train_epoch(model, loader, optim, loss_fn, device, args,
                global_step_offset: int, total_steps: int, warmup_steps: int,
                amp_dtype, scaler):
    if args.model == 'baseline':
        model.train()
    else:
        # For Lite, backbone can be frozen or not
        model.train()
    
    epoch_loss = 0.0
    epoch_angle = 0.0
    epoch_dist = 0.0
    n_steps = len(loader)
    
    for step, (sources, targets, batch_targets) in enumerate(loader):
        current_step = global_step_offset + step
        sources = sources.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if sources.ndim == 4:
            sources = sources.contiguous(memory_format=torch.channels_last)
            targets = targets.contiguous(memory_format=torch.channels_last)
        
        bt = {
            'heading': batch_targets['heading'].to(device),
            'distance': batch_targets['distance'].to(device),
        }
        
        lr = cosine_lr(current_step, total_steps, warmup_steps, args.lr)
        for pg in optim.param_groups:
            pg['lr'] = lr
        
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(sources, targets)

            if args.model == 'baseline':
                losses = baseline_total_loss(pred, bt, lambda_dist=args.lambda_dist)
            else:
                losses = harp_pose_lite_loss(pred, bt,
                                             lambda_dist=args.lambda_dist,
                                             lambda_conf=args.lambda_conf)

        if scaler.is_enabled():
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        
        epoch_loss += losses['total'].item()
        epoch_angle += losses['angle'].item()
        epoch_dist += losses.get('dist', torch.tensor(0.0)).item()
        
        if step % 50 == 0:
            print(f"  Step {step}/{n_steps}  "
                  f"L={losses['total'].item():.4f}  "
                  f"Lθ={losses['angle'].item():.4f}  "
                  f"Ld={losses.get('dist', 0):.4f}  "
                  f"LR={lr:.2e}")
    
    return {
        'loss': epoch_loss / n_steps,
        'angle': epoch_angle / n_steps,
        'dist': epoch_dist / n_steps,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, args, amp_dtype):
    model.eval()
    total_err_h = 0.0
    total_err_d = 0.0
    total_angle_loss = 0.0
    n = 0
    
    for sources, targets, batch_targets in loader:
        sources = sources.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if sources.ndim == 4:
            sources = sources.contiguous(memory_format=torch.channels_last)
            targets = targets.contiguous(memory_format=torch.channels_last)
        
        bt = {
            'heading': batch_targets['heading'].to(device),
            'distance': batch_targets['distance'].to(device),
        }
        
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(sources, targets)
        
        # MAE heading (degrees, wrapped to [0, 180])
        err_h = (pred['heading_deg'] - bt['heading']).abs()
        err_h = torch.where(err_h > 180, 360 - err_h, err_h)
        
        # MAE distance
        err_d = (pred['distance'] - bt['distance']).abs()
        
        B = sources.shape[0]
        total_err_h += err_h.mean().item() * B
        total_err_d += err_d.mean().item() * B
        n += B
    
    return {
        'mae_heading': total_err_h / n,
        'mae_distance': total_err_d / n,
        'avg': (total_err_h / n + total_err_d / n) / 2,
    }


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("❌ GPU required. Exiting (would take years on CPU).")
        sys.exit(1)

    amp_dtype = get_amp_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    
    print(f"🖥  {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB VRAM")
    
    # Dataset
    print(f"\n📁 Dataset: {args.data_root}")
    all_buildings = sorted(os.listdir(resolve_train_view_dir(Path(args.data_root))))
    train_blds = all_buildings[:560]  # ~80%
    val_blds = all_buildings[560:]    # ~20%
    
    train_ds = PairDataset(args.data_root, max_pairs=960000, buildings=train_blds, 
                           seed=args.seed, is_val=False)
    val_ds = PairDataset(args.data_root, max_pairs=20000, buildings=val_blds,
                         seed=args.seed + 1, is_val=True)
    
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)
    
    print(f"  Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    # Model
    print(f"\n🔧 Model: {args.model}")
    if args.model == 'baseline':
        model = PairUAVBaseline().to(device)
    else:
        model = HARPPoseLite().to(device)
    model = model.to(memory_format=torch.channels_last)
    
    model.summary()
    
    # Optimiser
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss (baseline or Lite)
    if args.model == 'baseline':
        def loss_fn(p, t):
            return baseline_total_loss(p, t, lambda_dist=args.lambda_dist)
    else:
        def loss_fn(p, t):
            return harp_pose_lite_loss(p, t, lambda_dist=args.lambda_dist, lambda_conf=args.lambda_conf)
    
    print(f"\n🚀 Training {args.epochs} epochs")
    print(f"  LR={args.lr} (cosine, warmup={args.warmup_epochs}) | "
          f"Batch={args.batch_size} | λ_dist={args.lambda_dist}")
    
    best_avg = float('inf')
    best_ckpt = None
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optim,
            loss_fn,
            device,
            args,
            global_step_offset=epoch * steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        elapsed = time.time() - t0
        
        val_metrics = validate(model, val_loader, loss_fn, device, args, amp_dtype)
        
        print(f"\nEpoch {epoch+1:2d}/{args.epochs} ({elapsed:.0f}s) "
              f"Train L={train_metrics['loss']:.4f} | "
              f"Val MAE_H={val_metrics['mae_heading']:.1f}° "
              f"MAE_D={val_metrics['mae_distance']:.1f}m "
              f"AVG={val_metrics['avg']:.2f}")
        
        if val_metrics['avg'] < best_avg:
            best_avg = val_metrics['avg']
            best_ckpt = {
                'epoch': epoch, 'val_avg': best_avg,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'args': vars(args), 'model': args.model,
            }
            print(f"  ★ Best so far!")
    
    # Save checkpoint
    out_path = args.output or f"checkpoints/{args.model}_{args.epochs}ep.pt"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    torch.save(best_ckpt or {'args': vars(args), 'model': args.model}, out_path)
    print(f"\n💾 Saved best checkpoint to {out_path} (AVG={best_avg:.2f})")


if __name__ == '__main__':
    main()
