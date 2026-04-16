"""
3-Phase Training for HARP-Pose Dual-Path
"""
import argparse, os, sys, time
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cache_features import CachedDataset
from utils.metrics import comprehensive_metrics, is_better_result
from models.harp_dual_path import HARPDualPath
from training.loss import phase1_loss, phase2_loss, laplace_nll

PHASE_CFG = {
    1: dict(epochs=20, lr=5e-4,  bs=192, frozen=True,  gate=False),
    2: dict(epochs=15, lr=1e-4,  bs=192, frozen=True,  gate=True),
    3: dict(epochs=10, lr=1e-5,  bs=64,  frozen=False, gate=True),
}


def get_amp_dtype():
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_grad_scaler(use_fp16: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_fp16)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_fp16)
    return torch.cuda.amp.GradScaler(enabled=use_fp16)


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def cosine_lr(epoch, total, warmup, base):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / (total - warmup)
    return base * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * prog)).item())


# ---------- wrappers that accept feature dicts ---------- #

def wide_forward(model, batch, dev):
    fs, ft = batch["feat_s"].to(dev, non_blocking=True), batch["feat_t"].to(dev, non_blocking=True)
    if fs.ndim == 4:
        fs = fs.contiguous(memory_format=torch.channels_last)
        ft = ft.contiguous(memory_format=torch.channels_last)
    th = batch["heading"].to(dev).float()
    td = batch["distance"].to(dev).float()
    pred = model.forward_features(fs, ft)
    return pred, {"heading": th, "distance": td}


def raw_forward(model, batch, dev):
    source, target, meta = batch
    source = source.to(dev, non_blocking=True)
    target = target.to(dev, non_blocking=True)
    if source.ndim == 4:
        source = source.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)
    th = torch.as_tensor(meta["heading"], dtype=torch.float32, device=dev)
    td = torch.as_tensor(meta["distance"], dtype=torch.float32, device=dev)
    pred = model(source, target)
    
    # Clamp distance predictions to prevent numerical instability
    # Distance is modeled as exp(log_distance), so clamp log_distance to reasonable range
    if "distance" in pred:
        pred["distance"] = pred["distance"].clamp(min=0.1, max=200.0)
    
    return pred, {"heading": th, "distance": td}


def forward_batch(model, batch, dev, use_raw: bool):
    if use_raw:
        return raw_forward(model, batch, dev)
    return wide_forward(model, batch, dev)


def compute_phase_losses(pred: dict, tgt: dict, phase: int) -> dict:
    if phase == 1:
        return phase1_loss(pred, tgt)
    if phase == 2:
        return phase2_loss(pred, tgt)
    return laplace_nll(pred, tgt)


def train_epoch(model, loader, optim, dev, phase, ep, total_ep, amp_dtype, scaler, use_raw: bool):
    model.train()
    model.phase = 1 if phase == 1 else 2
    t_loss, t_angle, n = 0.0, 0.0, 0
    for batch_idx, batch in enumerate(loader):
        lr = cosine_lr(ep, total_ep, 3, list(optim.param_groups)[0]["lr"])
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred, tgt = forward_batch(model, batch, dev, use_raw)
            losses = compute_phase_losses(pred, tgt, phase)

        # Check for NaN in loss before backward
        loss_val = losses["total"].item()
        if not torch.isfinite(losses["total"]):
            print(f"  ⚠️  NaN/Inf detected in loss at epoch {ep+1}, step {batch_idx+1}. Skipping batch.")
            # Reset scaler state to prevent carrying over bad gradients
            if scaler.is_enabled():
                scaler.update()  # Skip this batch's gradients
            continue

        if scaler.is_enabled():
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optim)
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Check for NaN in gradients after clipping
            has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
            if has_nan_grad:
                print(f"  ⚠️  NaN detected in gradients at epoch {ep+1}, step {batch_idx+1}. Skipping optimizer step.")
                scaler.update()
                continue
            scaler.step(optim)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        t_loss += loss_val
        t_angle += losses.get("angle", torch.tensor(0.0)).item()
        n += 1
        if n % 100 == 0:
            print(f"  Ep {ep+1} Step {n}/{len(loader)}  "
                  f"L={loss_val:.4f}  Lθ={t_angle/n:.4f}  LR={lr:.2e}")
    return {"loss": t_loss / max(n, 1), "angle": t_angle / max(n, 1)}


@torch.no_grad()
def validate(model, loader, dev, amp_dtype, phase: int, use_raw: bool):
    model.eval()
    model.phase = 1 if phase == 1 else 2
    v_loss, v_angle, n = 0.0, 0.0, 0
    pred_heading_batches = []
    pred_distance_batches = []
    tgt_heading_batches = []
    tgt_distance_batches = []

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred, tgt = forward_batch(model, batch, dev, use_raw)
            losses = compute_phase_losses(pred, tgt, phase)

        pred_heading_batches.append(pred["heading"].detach().cpu())
        pred_distance_batches.append(pred["distance"].detach().cpu())
        tgt_heading_batches.append(tgt["heading"].detach().cpu())
        tgt_distance_batches.append(tgt["distance"].detach().cpu())

        v_loss += losses["total"].item()
        v_angle += losses.get("angle", torch.tensor(0.0)).item()
        n += 1

    pred_h = torch.cat(pred_heading_batches)
    pred_d = torch.cat(pred_distance_batches)
    tgt_h = torch.cat(tgt_heading_batches)
    tgt_d = torch.cat(tgt_distance_batches)

    metrics = comprehensive_metrics(
        {"heading": pred_h, "distance": pred_d},
        {"heading": tgt_h, "distance": tgt_d},
    )
    return {
        "mae_heading": metrics["mae_heading"],
        "mae_distance": metrics["mae_distance"],
        "avg": metrics["avg"],
        "angle_rel_error": metrics["angle_rel_error"],
        "distance_rel_error": metrics["distance_rel_error"],
        "final_score": metrics["final_score"],
        "sr10": metrics["sr_10m"],
        "val_loss": v_loss / max(n, 1),
        "val_total_loss": v_loss / max(n, 1),
        "val_angle_loss": v_angle / max(n, 1),
    }


def preview_label_samples(dataset, sample_count: int, seed: int) -> list[dict]:
    if len(dataset) == 0:
        return []
    rng = random.Random(seed)
    take = min(sample_count, len(dataset))
    indices = sorted(rng.sample(range(len(dataset)), take))
    samples: list[dict] = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, tuple):
            _, _, meta = item
        else:
            meta = item
        heading = float(meta["heading"]) if "heading" in meta else float("nan")
        distance = float(meta["distance"]) if "distance" in meta else float("nan")
        source = str(meta.get("source_name", meta.get("source", "")))
        target = str(meta.get("target_name", meta.get("target", "")))
        samples.append(
            {
                "index": idx,
                "source": source,
                "target": target,
                "heading_deg": heading,
                "distance_m": distance,
            }
        )
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--data-root", "--university-release", "--data", dest="data_root",
                   type=str, default=None,
                   help="Path to PairUAV training root (required when --raw true)")
    p.add_argument("--phase", type=int, choices=[1,2,3])
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--checkpoint", type=str, default="checkpoints/dual_path.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--raw", type=parse_bool_arg, default=False,
                   help="Use raw-image training path instead of cached features (true/false)")
    p.add_argument("--annotations-root", type=str, default=None,
                   help="Optional PairUAV root containing official train annotations")
    p.add_argument("--official-annotations", choices=["auto", "true", "false"],
                   default="auto",
                   help="Use official PairUAV train JSON labels in raw mode")
    p.add_argument("--strict-official-only", type=parse_bool_arg, default=False,
                   help="Abort if official annotations are unavailable (true/false)")
    p.add_argument("--early-stop", type=parse_bool_arg, default=True,
                   help="Enable MAE+loss early stopping (true/false)")
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience in epochs without validation improvement")
    p.add_argument("--min-delta-avg", type=float, default=0.05,
                   help="Minimum validation AVG improvement to reset patience")
    p.add_argument("--min-delta-score", type=float, default=1e-3,
                   help="Minimum final_score improvement to reset patience")
    p.add_argument("--min-delta-loss", type=float, default=1e-3,
                   help="Minimum validation loss improvement to reset patience")
    args = p.parse_args()

    if args.raw and not args.data_root:
        raise ValueError("--data-root is required when --raw true")
    if not args.raw and not args.cache:
        raise ValueError("--cache is required when --raw false")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype()
    scaler = make_grad_scaler(amp_dtype == torch.float16)

    c = PHASE_CFG[args.phase]
    ep = args.epochs or c["epochs"]
    bs = args.batch_size or c["bs"]
    lr = args.lr or c["lr"]
    data_mode_text = "cached-features"
    annotation_source_text = "none"

    if args.raw:
        from data.dataset import (
            PairUAVAnnotationDataset,
            collect_annotation_json_paths,
            resolve_train_annotation_dir,
            resolve_train_view_dir,
        )

        annotation_root = Path(args.annotations_root).resolve() if args.annotations_root else Path(args.data_root)
        annotation_dir = resolve_train_annotation_dir(annotation_root)
        if args.official_annotations == "true" and annotation_dir is None:
            raise FileNotFoundError(
                "--official-annotations true was requested, but no train annotations were found under "
                f"{annotation_root}"
            )

        use_official_annotations = (
            annotation_dir is not None if args.official_annotations == "auto"
            else args.official_annotations == "true"
        )

        if use_official_annotations:
            assert annotation_dir is not None
            group_dirs = [
                path.name
                for path in annotation_dir.iterdir()
                if path.is_dir() and any(p.is_file() for p in path.glob("*.json"))
            ]
            group_dirs = sorted(
                group_dirs,
                key=lambda name: (int(name) if name.isdigit() else 10**12, name),
            )

            if len(group_dirs) >= 2:
                split_idx = max(1, int(len(group_dirs) * 0.8))
                split_idx = min(split_idx, len(group_dirs) - 1)
                tr_groups = group_dirs[:split_idx]
                va_groups = group_dirs[split_idx:]
                ds_tr = PairUAVAnnotationDataset(
                    str(annotation_root),
                    groups=tr_groups,
                    max_pairs=960_000,
                    seed=args.seed,
                    is_val=False,
                )
                ds_va = PairUAVAnnotationDataset(
                    str(annotation_root),
                    groups=va_groups,
                    max_pairs=20_000,
                    seed=args.seed + 1,
                    is_val=True,
                )
            else:
                all_json = collect_annotation_json_paths(annotation_dir)
                if len(all_json) < 2:
                    raise RuntimeError(
                        "Official annotation mode requires at least 2 JSON files for train/val split."
                    )
                split_idx = max(1, int(len(all_json) * 0.8))
                split_idx = min(split_idx, len(all_json) - 1)
                tr_json = all_json[:split_idx]
                va_json = all_json[split_idx:]
                ds_tr = PairUAVAnnotationDataset(
                    str(annotation_root),
                    json_paths=tr_json,
                    max_pairs=960_000,
                    seed=args.seed,
                    is_val=False,
                )
                ds_va = PairUAVAnnotationDataset(
                    str(annotation_root),
                    json_paths=va_json,
                    max_pairs=20_000,
                    seed=args.seed + 1,
                    is_val=True,
                )

            data_mode_text = "official-annotations"
            annotation_source_text = str(annotation_dir)
            print(f"Using official annotation supervision from {annotation_dir}")
        else:
            raise RuntimeError("strict_official_only is conceptually true now. Provide --annotations-root with official labels or set --official-annotations true.")
    else:
        all_bld = sorted(f.stem for f in Path(args.cache).glob("*.npz"))
        tr = all_bld[:560]
        va = all_bld[560:]
        ds_tr = CachedDataset(args.cache, buildings=tr, max_pairs=960_000,
                              seed=args.seed, preload=True)
        ds_va = CachedDataset(args.cache, buildings=va, max_pairs=20_000,
                              seed=args.seed+1, is_val=True, preload=True)

    loader_kwargs = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, **loader_kwargs)

    print(
        "[DatasetDiagnostics] "
        f"mode={data_mode_text} annotation_source={annotation_source_text} "
        f"pair_counts train={len(ds_tr)} val={len(ds_va)}"
    )
    print("[DatasetDiagnostics] decoded_label_samples=")
    for sample in preview_label_samples(ds_tr, sample_count=3, seed=args.seed):
        print(
            "  - "
            f"idx={sample['index']} src={sample['source']} tgt={sample['target']} "
            f"heading_deg={sample['heading_deg']:.3f} distance_m={sample['distance_m']:.3f}"
        )

    model = HARPDualPath(frozen=c["frozen"], use_gate=c["gate"]).to(dev)
    model = model.to(memory_format=torch.channels_last)

    # Load previous phase
    if args.phase > 1 and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint → {args.checkpoint}")

    tp = sum(p.numel() for p in model.parameters())
    tr_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {tp/1e6:.1f}M total, {tr_/1e6:.1f}M trainable")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)

    mode_text = data_mode_text if args.raw else "cached-features"
    print(f"\nPhase {args.phase}: {ep} epochs, LR={lr}, BS={bs} | input={mode_text}")
    best_avg = float("inf")
    best_final_score = float("inf")
    best_val_loss = float("inf")
    best_rank_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0

    for e in range(ep):
        for pg in optim.param_groups:
            pg["lr"] = cosine_lr(e, ep, 3, lr)
        t0 = time.time()
        tm = train_epoch(model, dl_tr, optim, dev, args.phase, e, ep, amp_dtype, scaler, args.raw)
        vm = validate(model, dl_va, dev, amp_dtype, args.phase, args.raw)
        print(f"Epoch {e+1}/{ep} ({time.time()-t0:.0f}s)  "
              f"Train L={tm['loss']:.4f} |  "
              f"Val L={vm['val_loss']:.4f} |  "
              f"Val MAE_H={vm['mae_heading']:.1f}°  "
              f"MAE_D={vm['mae_distance']:.1f}m  "
              f"AVG={vm['avg']:.2f}  "
              f"Final={vm['final_score']:.4f}  "
              f"AngRel={vm['angle_rel_error']:.4f}  "
              f"DistRel={vm['distance_rel_error']:.4f}  "
              f"SR@10={vm['sr10']:.1f}%")

        current_rank = {
            "final_score": vm["final_score"],
            "distance_rel_error": vm["distance_rel_error"],
            "angle_rel_error": vm["angle_rel_error"],
            "val_total_loss": vm["val_total_loss"],
        }
        if is_better_result(current_rank, best_rank_metrics):
            stale_epochs = 0
            best_rank_metrics = current_rank
            best_avg = min(best_avg, vm["avg"])
            best_final_score = min(best_final_score, vm["final_score"])
            best_val_loss = min(best_val_loss, vm["val_loss"])
            best_epoch = e + 1
            torch.save({
                "phase": args.phase,
                "epoch": e,
                "val_avg": vm["avg"],
                "val_final_score": vm["final_score"],
                "val_loss": vm["val_loss"],
                "best_val_avg": best_avg,
                "best_val_final_score": best_final_score,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
            }, args.checkpoint)
            print("  ★ Best checkpoint (leaderboard-priority metric improved)")
        else:
            stale_epochs += 1
            print(f"  No validation improvement for {stale_epochs}/{args.patience} epochs")
            if args.early_stop and stale_epochs >= args.patience:
                print(
                    f"Early stopping at epoch {e+1}: "
                    f"validation score/AVG/loss did not improve (patience={args.patience})."
                )
                break

    print(
        f"\nDone. Best FinalScore = {best_final_score:.4f} | "
        f"Best AVG = {best_avg:.2f} | "
        f"Best ValLoss = {best_val_loss:.4f} | Best Epoch = {best_epoch}"
    )


if __name__ == "__main__":
    main()

