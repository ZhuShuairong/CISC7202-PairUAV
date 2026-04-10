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
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import resolve_train_annotation_dir, resolve_train_view_dir
from data.dataset_pairuav import PairUAVDataset, split_official_json_paths
from models.baseline import PairUAVBaseline, baseline_total_loss
from models.harp_pose_lite import HARPPoseLite, harp_pose_lite_loss
from utils.metrics import comprehensive_metrics, is_better_result


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
    p.add_argument('--dataset-mode', type=str, default='auto', choices=['auto', 'official', 'pseudo'])
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--strict-official-only', action='store_true')
    p.add_argument('--allow-pseudo-warmup', action='store_true')
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

    print(
        '[DatasetDiagnostics] '
        f'mode={dataset_mode} annotation_source={annotation_source} '
        f'pair_counts train={len(train_ds)} val={len(val_ds)} test=unknown'
    )
    print('[DatasetDiagnostics] decoded_label_samples=')
    for sample in train_ds.sample_decoded_labels(sample_count=3, seed=args.seed):
        print(
            '  - '
            f"idx={sample['index']} src={sample['source_id']} tgt={sample['target_id']} "
            f"heading_deg={sample['heading_deg']:.3f} distance_m={sample['distance_m']:.3f}"
        )
    
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
    best_rank_metrics = None
    pseudo_warmup_mode = False
    best_ckpt = None
    last_ckpt = None
    
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

        last_ckpt = {
            'epoch': epoch,
            'val_avg': val_metrics['avg'],
            'val_final_score': val_metrics['final_score'],
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'args': vars(args),
            'model': args.model,
            'dataset_mode': dataset_mode,
        }
        
        print(f"\nEpoch {epoch+1:2d}/{args.epochs} ({elapsed:.0f}s) "
              f"Train L={train_metrics['loss']:.4f} | "
              f"Val MAE_H={val_metrics['mae_heading']:.1f}° "
              f"MAE_D={val_metrics['mae_distance']:.1f}m "
              f"AVG={val_metrics['avg']:.2f} "
              f"Final={val_metrics['final_score']:.4f} "
              f"AngRel={val_metrics['angle_rel_error']:.4f} "
              f"DistRel={val_metrics['distance_rel_error']:.4f}")

        if not pseudo_warmup_mode:
            current_rank = {
                'final_score': val_metrics['final_score'],
                'distance_rel_error': val_metrics['distance_rel_error'],
                'angle_rel_error': val_metrics['angle_rel_error'],
            }
            if is_better_result(current_rank, best_rank_metrics):
                best_rank_metrics = current_rank
                best_avg = min(best_avg, val_metrics['avg'])
                best_ckpt = {
                    'epoch': epoch,
                    'val_avg': val_metrics['avg'],
                    'val_final_score': val_metrics['final_score'],
                    'val_distance_rel_error': val_metrics['distance_rel_error'],
                    'val_angle_rel_error': val_metrics['angle_rel_error'],
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'args': vars(args),
                    'model': args.model,
                    'dataset_mode': dataset_mode,
                }
                print("  ★ Best so far (leaderboard-priority metric improved)")
        else:
            print("  Pseudo warmup mode: checkpoint ranking disabled for final selection")
    
    # Save checkpoint
    out_path = args.output or f"checkpoints/{args.model}_{args.epochs}ep.pt"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    payload = best_ckpt if best_ckpt is not None else last_ckpt
    if payload is None:
        payload = {'args': vars(args), 'model': args.model, 'dataset_mode': dataset_mode}
    torch.save(payload, out_path)
    if pseudo_warmup_mode:
        print(f"\n💾 Saved warmup checkpoint to {out_path}")
    else:
        print(f"\n💾 Saved best checkpoint to {out_path} (AVG={best_avg:.2f})")


if __name__ == '__main__':
    main()

