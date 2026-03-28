"""Experiment 03 training: masked ECG+rPPG reconstruction (self-supervised)."""

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import build_masked_recon_dataloaders
from exp3_model import build_exp3_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 03: Masked ECG+rPPG reconstruction")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument("--checkpoint-tag", type=str, default="")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-block-min", type=int, default=8)
    parser.add_argument("--mask-block-max", type=int, default=32)
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--ecg-point-weight", type=float, default=1.0)
    parser.add_argument("--rppg-point-weight", type=float, default=1.0)
    parser.add_argument("--grad-loss-weight", type=float, default=0.0)
    parser.add_argument("--ecg-fft-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def build_visible_mask(x, mask_ratio, block_min, block_max):
    """Create visible mask (1=visible, 0=masked) with contiguous masked spans."""
    bsz, ch, length = x.shape
    visible = torch.ones((bsz, ch, length), device=x.device, dtype=x.dtype)

    target_mask_points = int(length * max(0.0, min(0.95, mask_ratio)))
    if target_mask_points <= 0:
        return visible

    for b in range(bsz):
        for c in range(ch):
            masked = 0
            attempts = 0
            while masked < target_mask_points and attempts < 64:
                attempts += 1
                seg_len = torch.randint(block_min, block_max + 1, (1,), device=x.device).item()
                seg_len = max(1, min(seg_len, length))
                start = torch.randint(0, max(1, length - seg_len + 1), (1,), device=x.device).item()
                end = start + seg_len
                newly = visible[b, c, start:end].sum().item()
                visible[b, c, start:end] = 0.0
                masked += int(newly)

    return visible


def weighted_masked_loss(
    pred,
    target,
    visible_mask,
    clean_score,
    criterion,
    context_weight,
    ecg_point_weight=1.0,
    rppg_point_weight=1.0,
    grad_loss_weight=0.0,
    ecg_fft_loss_weight=0.0,
):
    masked_mask = 1.0 - visible_mask

    ch_weight = torch.tensor([ecg_point_weight, rppg_point_weight], device=pred.device, dtype=pred.dtype).view(1, 2, 1)
    per_point = criterion(pred, target) * ch_weight

    masked_num = (per_point * masked_mask).sum(dim=(1, 2))
    masked_den = masked_mask.sum(dim=(1, 2)).clamp_min(1.0)
    masked_loss = masked_num / masked_den

    context_num = (per_point * visible_mask).sum(dim=(1, 2))
    context_den = visible_mask.sum(dim=(1, 2)).clamp_min(1.0)
    context_loss = context_num / context_den

    grad_loss = torch.zeros_like(masked_loss)
    if grad_loss_weight > 0:
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        grad_mask = torch.minimum(masked_mask[:, :, 1:], masked_mask[:, :, :-1])
        grad_err = (pred_diff - target_diff).abs() * grad_mask * ch_weight
        grad_num = grad_err.sum(dim=(1, 2))
        grad_den = grad_mask.sum(dim=(1, 2)).clamp_min(1.0)
        grad_loss = grad_num / grad_den

    ecg_fft_loss = torch.zeros_like(masked_loss)
    if ecg_fft_loss_weight > 0:
        pred_fft = torch.fft.rfft(pred[:, 0, :], dim=-1)
        target_fft = torch.fft.rfft(target[:, 0, :], dim=-1)
        ecg_fft_loss = torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)), dim=-1)

    # Keep all data, but trust high-quality windows slightly more.
    sample_weight = 0.5 + 0.5 * clean_score
    sample_loss = (
        masked_loss
        + context_weight * context_loss
        + grad_loss_weight * grad_loss
        + ecg_fft_loss_weight * ecg_fft_loss
    ) * sample_weight
    return sample_loss.mean()


def recon_mae_by_channel(pred, target, masked_mask):
    ecg_mae = ((pred[:, 0] - target[:, 0]).abs() * masked_mask[:, 0]).sum() / masked_mask[:, 0].sum().clamp_min(1.0)
    rppg_mae = ((pred[:, 1] - target[:, 1]).abs() * masked_mask[:, 1]).sum() / masked_mask[:, 1].sum().clamp_min(1.0)
    return ecg_mae.item(), rppg_mae.item()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    max_batches=None,
    mask_ratio=0.3,
    mask_block_min=8,
    mask_block_max=32,
    context_weight=0.2,
    ecg_point_weight=1.0,
    rppg_point_weight=1.0,
    grad_loss_weight=0.0,
    ecg_fft_loss_weight=0.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ecg_maes = []
    rppg_maes = []

    for batch_idx, (pair, clean_score, _, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pair = pair.to(DEVICE)
        clean_score = clean_score.to(DEVICE)

        visible = build_visible_mask(
            pair,
            mask_ratio=mask_ratio,
            block_min=mask_block_min,
            block_max=mask_block_max,
        )
        masked_mask = 1.0 - visible
        x_masked = pair * visible

        if is_train:
            optimizer.zero_grad()

        pred = model(x_masked, visible)
        loss = weighted_masked_loss(
            pred,
            pair,
            visible,
            clean_score,
            criterion,
            context_weight=context_weight,
            ecg_point_weight=ecg_point_weight,
            rppg_point_weight=rppg_point_weight,
            grad_loss_weight=grad_loss_weight,
            ecg_fft_loss_weight=ecg_fft_loss_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        ecg_mae, rppg_mae = recon_mae_by_channel(pred, pair, masked_mask)

        losses.append(loss.item())
        ecg_maes.append(ecg_mae)
        rppg_maes.append(rppg_mae)

    out = {
        "loss": sum(losses) / max(len(losses), 1),
        "ecg_mae": sum(ecg_maes) / max(len(ecg_maes), 1),
        "rppg_mae": sum(rppg_maes) / max(len(rppg_maes), 1),
    }
    return out


def save_history(history_rows, csv_path, plot_path, title):
    fieldnames = [
        "epoch",
        "tr_loss",
        "va_loss",
        "tr_ecg_mae",
        "tr_rppg_mae",
        "va_ecg_mae",
        "va_rppg_mae",
        "seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["tr_loss"] for r in history_rows]
    va_loss = [r["va_loss"] for r in history_rows]
    tr_ecg = [r["tr_ecg_mae"] for r in history_rows]
    va_ecg = [r["va_ecg_mae"] for r in history_rows]
    tr_rppg = [r["tr_rppg_mae"] for r in history_rows]
    va_rppg = [r["va_rppg_mae"] for r in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, tr_ecg, label="ECG train")
    axes[1].plot(epochs, va_ecg, label="ECG val")
    axes[1].plot(epochs, tr_rppg, label="rPPG train")
    axes[1].plot(epochs, va_rppg, label="rPPG val")
    axes[1].set_title("Masked MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=180)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np_seed = args.seed
    if np_seed is not None:
        import numpy as _np

        _np.random.seed(np_seed)

    print("Loading data ...")
    train_loader, val_loader = build_masked_recon_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    model = build_exp3_model(args.variant).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"Model variant: {args.variant}, params: {param_count:,}")

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_dir = os.path.join(ROOT_DIR, "train", "exp3", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    tag = args.checkpoint_tag.strip()
    ckpt_prefix = f"exp3_{args.variant}{tag}"

    best_val = float("inf")
    history_rows = []

    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrECG_MAE':>10}  {'TrRPPG_MAE':>11}  "
        f"{'VaECG_MAE':>10}  {'VaRPPG_MAE':>11}  {'Time':>6}"
    )
    print("-" * 96)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            mask_ratio=args.mask_ratio,
            mask_block_min=args.mask_block_min,
            mask_block_max=args.mask_block_max,
            context_weight=args.context_weight,
            ecg_point_weight=args.ecg_point_weight,
            rppg_point_weight=args.rppg_point_weight,
            grad_loss_weight=args.grad_loss_weight,
            ecg_fft_loss_weight=args.ecg_fft_loss_weight,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                max_batches=args.max_val_batches,
                mask_ratio=args.mask_ratio,
                mask_block_min=args.mask_block_min,
                mask_block_max=args.mask_block_max,
                context_weight=args.context_weight,
                ecg_point_weight=args.ecg_point_weight,
                rppg_point_weight=args.rppg_point_weight,
                grad_loss_weight=args.grad_loss_weight,
                ecg_fft_loss_weight=args.ecg_fft_loss_weight,
            )

        elapsed = time.time() - t0
        marker = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "checkpoint_tag": tag,
                    "mask_ratio": args.mask_ratio,
                    "target_length": args.target_length,
                    "ecg_point_weight": args.ecg_point_weight,
                    "rppg_point_weight": args.rppg_point_weight,
                    "grad_loss_weight": args.grad_loss_weight,
                    "ecg_fft_loss_weight": args.ecg_fft_loss_weight,
                },
                os.path.join(SAVE_DIR, f"{ckpt_prefix}_best.pt"),
            )
            marker = " *"

        history_rows.append(
            {
                "epoch": epoch,
                "tr_loss": tr["loss"],
                "va_loss": va["loss"],
                "tr_ecg_mae": tr["ecg_mae"],
                "tr_rppg_mae": tr["rppg_mae"],
                "va_ecg_mae": va["ecg_mae"],
                "va_rppg_mae": va["rppg_mae"],
                "seconds": elapsed,
            }
        )

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['ecg_mae']:10.4f}  {tr['rppg_mae']:11.4f}  "
            f"{va['ecg_mae']:10.4f}  {va['rppg_mae']:11.4f}  "
            f"{elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val,
            "checkpoint_tag": tag,
            "mask_ratio": args.mask_ratio,
            "target_length": args.target_length,
            "ecg_point_weight": args.ecg_point_weight,
            "rppg_point_weight": args.rppg_point_weight,
            "grad_loss_weight": args.grad_loss_weight,
            "ecg_fft_loss_weight": args.ecg_fft_loss_weight,
        },
        os.path.join(SAVE_DIR, f"{ckpt_prefix}_final.pt"),
    )

    history_csv = os.path.join(plot_dir, f"{ckpt_prefix}_history.csv")
    history_png = os.path.join(plot_dir, f"{ckpt_prefix}_history.png")
    save_history(history_rows, history_csv, history_png, title=f"Exp3 Training History ({ckpt_prefix})")

    print(f"Saved history: {history_csv}")
    print(f"Saved history plot: {history_png}")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
