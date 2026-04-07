"""
3-Phase Training for HARP-Pose Dual-Path
"""
import argparse, os, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cache_features import CachedDataset
from models.harp_dual_path import HARPDualPath
from training.loss import phase1_loss, phase2_loss, laplace_nll

PHASE_CFG = {
    1: dict(epochs=20, lr=5e-4,  bs=256, frozen=True,  gate=False),
    2: dict(epochs=15, lr=1e-4,  bs=256, frozen=True,  gate=True),
    3: dict(epochs=10, lr=1e-5,  bs=128, frozen=False, gate=True),
}


def get_amp_dtype():
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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


def train_epoch(model, loader, optim, dev, phase, ep, total_ep, amp_dtype, scaler):
    model.train()
    t_loss, t_angle, n = 0.0, 0.0, 0
    for batch in loader:
        lr = cosine_lr(ep, total_ep, 3, list(optim.param_groups)[0]["lr"])
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred, tgt = wide_forward(model, batch, dev)
            if phase == 1:
                losses = phase1_loss(pred, tgt)
            elif phase == 2:
                losses = phase2_loss(pred, tgt)
            else:
                losses = laplace_nll(pred, tgt)

        if scaler.is_enabled():
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        t_loss += losses["total"].item()
        t_angle += losses.get("angle", torch.tensor(0.0)).item()
        n += 1
        if n % 100 == 0:
            print(f"  Ep {ep+1} Step {n}/{len(loader)}  "
                  f"L={losses['total'].item():.4f}  Lθ={t_angle/n:.4f}  LR={lr:.2e}")
    return {"loss": t_loss / n, "angle": t_angle / n}


@torch.no_grad()
def validate(model, loader, dev, amp_dtype):
    model.eval(); model.phase = 2
    e_h, e_d, sr, total = 0.0, 0.0, 0, 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred, tgt = wide_forward(model, batch, dev)
        eh = (pred["heading"] - tgt["heading"]).abs()
        eh = torch.where(eh > 180, 360 - eh, eh)
        ed = (pred["distance"] - tgt["distance"]).abs()
        ep_dist = torch.sqrt((pred["heading"] - tgt["heading"]).pow(2) * 0.01
                             + (pred["distance"] - tgt["distance"]).pow(2))
        sr += (ep_dist < 10).sum().item()
        B = tgt["heading"].shape[0]
        e_h += eh.sum().item(); e_d += ed.sum().item(); total += B
    mae_h = e_h / total; mae_d = e_d / total
    return {"mae_heading": mae_h, "mae_distance": mae_d,
            "avg": (mae_h + mae_d) / 2, "sr10": sr / total * 100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--phase", type=int, choices=[1,2,3])
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--checkpoint", type=str, default="checkpoints/dual_path.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    c = PHASE_CFG[args.phase]
    ep = args.epochs or c["epochs"]
    bs = args.batch_size or c["bs"]
    lr = args.lr or c["lr"]

    all_bld = sorted(f.stem for f in Path(args.cache).glob("*.npz"))
    tr = all_bld[:560]; va = all_bld[560:]
    ds_tr = CachedDataset(args.cache, buildings=tr, max_pairs=960_000, seed=args.seed, preload=True)
    ds_va = CachedDataset(args.cache, buildings=va, max_pairs=20_000,
                          seed=args.seed+1, is_val=True, preload=True)
    loader_kwargs = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, **loader_kwargs)

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

    print(f"\nPhase {args.phase}: {ep} epochs, LR={lr}, BS={bs}")
    best = float("inf")

    for e in range(ep):
        for pg in optim.param_groups:
            pg["lr"] = cosine_lr(e, ep, 3, lr)
        t0 = time.time()
        tm = train_epoch(model, dl_tr, optim, dev, args.phase, e, ep, amp_dtype, scaler)
        vm = validate(model, dl_va, dev, amp_dtype)
        print(f"Epoch {e+1}/{ep} ({time.time()-t0:.0f}s)  "
              f"Train L={tm['loss']:.4f} |  "
              f"Val MAE_H={vm['mae_heading']:.1f}°  "
              f"MAE_D={vm['mae_distance']:.1f}m  "
              f"AVG={vm['avg']:.2f}  SR@10={vm['sr10']:.1f}%")

        if vm["avg"] < best:
            best = vm["avg"]
            torch.save({
                "phase": args.phase, "epoch": e, "val_avg": best,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
            }, args.checkpoint)
            print(f"  ★ Best checkpoint!")

    print(f"\nDone. Best AVG = {best:.2f}")


if __name__ == "__main__":
    main()
