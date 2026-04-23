"""Shared training loop for Exp3 split experiments (ECG-only / rPPG-only)."""

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from single_recon_dataloader import build_single_signal_dataloaders
from single_recon_model import build_single_recon_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args(exp_name):
    parser = argparse.ArgumentParser(description=f"{exp_name}: single-signal masked reconstruction")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
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
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--grad-loss-weight", type=float, default=0.1)
    parser.add_argument("--fft-loss-weight", type=float, default=0.02)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def build_single_window_visible_mask(x, mask_ratio):
    """Create one contiguous masked window per segment (1=visible, 0=masked)."""
    bsz, ch, length = x.shape
    visible = torch.ones((bsz, ch, length), device=x.device, dtype=x.dtype)

    ratio = max(0.0, min(0.95, float(mask_ratio)))
    if ratio <= 0.0:
        return visible

    mask_len = int(round(length * ratio))
    mask_len = max(1, min(mask_len, length - 1 if length > 1 else 1))

    for b in range(bsz):
        start = torch.randint(0, max(1, length - mask_len + 1), (1,), device=x.device).item()
        end = start + mask_len
        visible[b, :, start:end] = 0.0

    return visible


def weighted_masked_loss(pred, target, visible_mask, quality_score, criterion, context_weight, grad_loss_weight=0.1, fft_loss_weight=0.02):
    masked_mask = 1.0 - visible_mask
    per_point = criterion(pred, target)

    masked_num = (per_point * masked_mask).sum(dim=(1, 2))
    masked_den = masked_mask.sum(dim=(1, 2)).clamp_min(1.0)
    masked_loss = masked_num / masked_den

    context_num = (per_point * visible_mask).sum(dim=(1, 2))
    context_den = visible_mask.sum(dim=(1, 2)).clamp_min(1.0)
    context_loss = context_num / context_den

    sample_weight = 0.5 + 0.5 * quality_score
    sample_loss = (masked_loss + context_weight * context_loss) * sample_weight
    loss = sample_loss.mean()

    if grad_loss_weight > 0.0:
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + grad_loss_weight * F.l1_loss(pred_diff, target_diff)

    if fft_loss_weight > 0.0:
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        loss = loss + fft_loss_weight * F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

    return loss


def masked_mae(pred, target, masked_mask):
    mae = ((pred - target).abs() * masked_mask).sum() / masked_mask.sum().clamp_min(1.0)
    return mae.item()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    max_batches=None,
    mask_ratio=0.3,
    context_weight=0.2,
    grad_loss_weight=0.1,
    fft_loss_weight=0.02,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    maes = []

    for batch_idx, (signal, quality_score) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        signal = signal.to(DEVICE)
        quality_score = quality_score.to(DEVICE)

        visible = build_single_window_visible_mask(signal, mask_ratio=mask_ratio)
        masked_mask = 1.0 - visible
        x_masked = signal * visible

        if is_train:
            optimizer.zero_grad()

        pred = model(x_masked, visible)
        loss = weighted_masked_loss(
            pred,
            signal,
            visible,
            quality_score,
            criterion,
            context_weight=context_weight,
            grad_loss_weight=grad_loss_weight,
            fft_loss_weight=fft_loss_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        maes.append(masked_mae(pred, signal, masked_mask))

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


def save_history(history_rows, csv_path, plot_path, title):
    fieldnames = ["epoch", "tr_loss", "va_loss", "tr_mae", "va_mae", "seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["tr_loss"] for r in history_rows]
    va_loss = [r["va_loss"] for r in history_rows]
    tr_mae = [r["tr_mae"] for r in history_rows]
    va_mae = [r["va_mae"] for r in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, tr_mae, label="train")
    axes[1].plot(epochs, va_mae, label="val")
    axes[1].set_title("Masked MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=180)


def run_experiment(signal_type, exp_name):
    args = parse_args(exp_name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data ...")
    train_loader, val_loader = build_single_signal_dataloaders(
        ROOT_DIR,
        signal_type=signal_type,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    model = build_single_recon_model(args.variant).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"Signal: {signal_type}, model variant: {args.variant}, params: {param_count:,}")

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
    plot_dir = os.path.join(ROOT_DIR, "train", exp_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    tag = args.checkpoint_tag.strip()
    ckpt_prefix = f"{exp_name}_{args.variant}{tag}"

    best_val = float("inf")
    history_rows = []

    print(f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrMAE':>8}  {'VaMAE':>8}  {'Time':>6}")
    print("-" * 56)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            mask_ratio=args.mask_ratio,
            context_weight=args.context_weight,
            grad_loss_weight=args.grad_loss_weight,
            fft_loss_weight=args.fft_loss_weight,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                max_batches=args.max_val_batches,
                mask_ratio=args.mask_ratio,
                context_weight=args.context_weight,
                grad_loss_weight=args.grad_loss_weight,
                fft_loss_weight=args.fft_loss_weight,
            )

        elapsed = time.time() - t0
        marker = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "signal_type": signal_type,
                    "variant": args.variant,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "checkpoint_tag": tag,
                    "mask_ratio": args.mask_ratio,
                    "target_length": args.target_length,
                },
                os.path.join(SAVE_DIR, f"{ckpt_prefix}_best.pt"),
            )
            marker = " *"

        history_rows.append(
            {
                "epoch": epoch,
                "tr_loss": tr["loss"],
                "va_loss": va["loss"],
                "tr_mae": tr["mae"],
                "va_mae": va["mae"],
                "seconds": elapsed,
            }
        )

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['mae']:8.4f}  {va['mae']:8.4f}  {elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "signal_type": signal_type,
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val,
            "checkpoint_tag": tag,
            "mask_ratio": args.mask_ratio,
            "target_length": args.target_length,
        },
        os.path.join(SAVE_DIR, f"{ckpt_prefix}_final.pt"),
    )

    history_csv = os.path.join(plot_dir, f"{ckpt_prefix}_history.csv")
    history_png = os.path.join(plot_dir, f"{ckpt_prefix}_history.png")
    save_history(history_rows, history_csv, history_png, title=f"{exp_name} Training History ({ckpt_prefix})")

    print(f"Saved history: {history_csv}")
    print(f"Saved history plot: {history_png}")
    print(f"\nDone. Best val loss: {best_val:.4f}")
