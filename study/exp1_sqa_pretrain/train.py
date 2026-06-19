"""Unified training script for Exp1-SQAPreTrain masked reconstruction.

Supports all model architectures and signal types.

Usage examples:
    # Baseline UNet, ECG only
    python train.py --model baseline --variant light --signal-type ecg

    # TCN with curriculum learning
    python train.py --model tcn --variant tcn256 --signal-type ecg --curriculum

    # Joint ECG+rPPG model
    python train.py --model joint --variant full --signal-type joint

    # Mamba full
    python train.py --model mamba --variant full --signal-type ecg --epochs 100
"""

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure we can import from the parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_sqa_pretrain.dataloader import build_dataloaders
from exp1_sqa_pretrain.models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default directories relative to this package
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(_PKG_DIR, "checkpoints")
DEFAULT_PLOT_DIR = os.path.join(_PKG_DIR, "plots")


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp1-SQAPreTrain: masked signal reconstruction pre-training"
    )

    # Model
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "tcn"],
                        help="Model architecture.")

    # Data
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=["ecg", "rppg", "joint"],
                        help="Signal type for reconstruction.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data root directory. Defaults to project root.")
    parser.add_argument("--data-source", type=str, default="sqi",
                        choices=["sqi", "cleaned"],
                        help="Use mirror*_auto_cleaned_sqi or mirror*_auto_cleaned.")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Data windowing
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)

    # Masking
    parser.add_argument("--mask-ratio", type=float, default=0.30,
                        help="Fraction of points to mask.")
    parser.add_argument("--context-weight", type=float, default=0.20,
                        help="Weight for context (visible region) loss.")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (mask ratio increases over training).")
    parser.add_argument("--mask-min", type=float, default=0.10,
                        help="Starting mask ratio for curriculum.")
    parser.add_argument("--mask-max", type=float, default=0.40,
                        help="Ending mask ratio for curriculum.")
    parser.add_argument("--best-tolerance", type=float, default=0.008,
                        help="Tolerance for checkpoint saving in curriculum mode.")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints. Defaults to package checkpoints/.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for plots. Defaults to package plots/.")
    parser.add_argument("--checkpoint-tag", type=str, default="",
                        help="Optional tag for checkpoint filenames.")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from.")

    # Limits
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    return parser.parse_args()


# ──────────────────────────────────────────────
# Masking
# ──────────────────────────────────────────────

def build_visible_mask(x, mask_ratio):
    """Create visible mask (1=visible, 0=masked) with one contiguous masked span per channel.

    Args:
        x: (B, C, L) input tensor.
        mask_ratio: Fraction of time steps to mask [0, 1].

    Returns:
        visible: (B, C, L) mask tensor, same device/dtype as x.
    """
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


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def masked_recon_loss(pred, target, visible_mask, quality_score, criterion, context_weight):
    """Weighted masked reconstruction loss.

    Combines masked-region loss and context (visible-region) loss,
    weighted by quality score (higher quality → higher weight).
    """
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
    return sample_loss.mean()


def compute_masked_mae(pred, target, masked_mask):
    """Mean absolute error over masked regions only."""
    mae = ((pred - target).abs() * masked_mask).sum() / masked_mask.sum().clamp_min(1.0)
    return mae.item()


# ──────────────────────────────────────────────
# Training / validation epoch
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, mask_ratio, context_weight,
              optimizer=None, max_batches=None):
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    losses, maes = [], []

    for batch_idx, (signal, quality_score) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        signal = signal.to(DEVICE)
        quality_score = quality_score.to(DEVICE)

        visible = build_visible_mask(signal, mask_ratio=mask_ratio)
        masked_mask = 1.0 - visible
        x_masked = signal * visible

        if is_train:
            optimizer.zero_grad()

        pred = model(x_masked, visible)
        loss = masked_recon_loss(
            pred, signal, visible, quality_score, criterion,
            context_weight=context_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        maes.append(compute_masked_mae(pred, signal, masked_mask))

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


# ──────────────────────────────────────────────
# Curriculum schedule
# ──────────────────────────────────────────────

def get_curriculum_mask_ratio(epoch, total_epochs, min_ratio=0.10, max_ratio=0.40):
    """Linear curriculum: mask_ratio increases from min_ratio to max_ratio."""
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return min_ratio + (max_ratio - min_ratio) * progress


# ──────────────────────────────────────────────
# History saving
# ──────────────────────────────────────────────

def save_history(history_rows, csv_path, plot_path, title):
    """Save training history as CSV and plot."""
    fieldnames = ["epoch", "tr_loss", "va_loss", "tr_mae", "va_mae", "seconds"]
    if "mask_ratio" in history_rows[0]:
        fieldnames.append("mask_ratio")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["tr_loss"] for r in history_rows]
    va_loss = [r["va_loss"] for r in history_rows]
    tr_mae = [r["tr_mae"] for r in history_rows]
    va_mae = [r["va_mae"] for r in history_rows]

    ncols = 3 if "mask_ratio" in history_rows[0] else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 2:
        axes = [axes[0], axes[1]]

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

    if ncols >= 3 and "mask_ratio" in history_rows[0]:
        mask_ratios = [r["mask_ratio"] for r in history_rows]
        axes[2].plot(epochs, mask_ratios, color="darkorange", linewidth=1.5)
        axes[2].set_title("Curriculum mask_ratio")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("mask_ratio")
        axes[2].set_ylim(0, 1)
        axes[2].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve directories
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    checkpoint_dir = os.path.abspath(args.checkpoint_dir) if args.checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    plot_dir = os.path.abspath(args.plot_dir) if args.plot_dir else DEFAULT_PLOT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build checkpoint prefix
    model_tag = args.model
    signal_tag = args.signal_type
    tag = args.checkpoint_tag.strip()
    curriculum_suffix = "_curriculum" if args.curriculum else ""
    len_tag = f"L{args.target_length}"
    ckpt_prefix = f"exp1_{signal_tag}_{model_tag}_{len_tag}{tag}{curriculum_suffix}"

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Exp1-SQAPreTrain: {model_tag} on {signal_tag}  "
         f"{'(curriculum)' if args.curriculum else ''}")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Data:     {data_dir} (source={args.data_source})")
    print(f"║  Device:   {DEVICE}")
    print(f"║  Epochs:   {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"║  Window:   {args.window_sec}s  |  Target len: {args.target_length}")
    if args.curriculum:
        print(f"║  Curriculum: mask {args.mask_min:.2f} → {args.mask_max:.2f}")
    else:
        print(f"║  Mask ratio: {args.mask_ratio:.2f}")
    print(f"║  Checkpoints: {checkpoint_dir}")
    print(f"║  Plots:       {plot_dir}")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    # Build data
    print(f"\nLoading data...")
    train_loader, val_loader = build_dataloaders(
        data_dir,
        signal_type=args.signal_type,
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

    # Build model
    model = build_model(args.model, target_length=args.target_length).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume
    start_epoch = 1
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    # Training loop
    best_val = float("inf")
    history_rows = []

    col_headers = f"{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrMAE':>8}  {'VaMAE':>8}  {'Time':>6}"
    if args.curriculum:
        col_headers = f"{'Epoch':>5}  {'Mask':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrMAE':>8}  {'VaMAE':>8}  {'Time':>6}  Note"
    print(f"\n{col_headers}")
    print("-" * len(col_headers))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if args.curriculum:
            mask_ratio = get_curriculum_mask_ratio(
                epoch, args.epochs, min_ratio=args.mask_min, max_ratio=args.mask_max
            )
        else:
            mask_ratio = args.mask_ratio

        tr = run_epoch(
            model, train_loader, criterion,
            mask_ratio=mask_ratio,
            context_weight=args.context_weight,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            va = run_epoch(
                model, val_loader, criterion,
                mask_ratio=mask_ratio,
                context_weight=args.context_weight,
                optimizer=None,
                max_batches=args.max_val_batches,
            )

        elapsed = time.time() - t0

        # Checkpointing
        note = ""
        if args.curriculum:
            should_save = va["loss"] < best_val or va["loss"] <= best_val + args.best_tolerance
            if va["loss"] < best_val:
                best_val = va["loss"]
                note = "best"
            elif should_save:
                note = "tol"
        else:
            should_save = va["loss"] < best_val
            if should_save:
                best_val = va["loss"]
                note = "*"

        if should_save:
            torch.save({
                "model": args.model,
                "signal_type": args.signal_type,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
                "epoch_loss": va["loss"],
                "mask_ratio": mask_ratio,
                "context_weight": args.context_weight,
                "target_length": args.target_length,
                "curriculum": args.curriculum,
            }, os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt"))

        row = {
            "epoch": epoch,
            "tr_loss": tr["loss"],
            "va_loss": va["loss"],
            "tr_mae": tr["mae"],
            "va_mae": va["mae"],
            "seconds": elapsed,
        }
        if args.curriculum:
            row["mask_ratio"] = round(mask_ratio, 4)

        history_rows.append(row)

        if args.curriculum:
            print(f"{epoch:5d}  {mask_ratio:5.3f}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
                  f"{tr['mae']:8.4f}  {va['mae']:8.4f}  {elapsed:5.1f}s  {note:>4s}")
        else:
            print(f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
                  f"{tr['mae']:8.4f}  {va['mae']:8.4f}  {elapsed:5.1f}s{note}")

    # Save final checkpoint
    torch.save({
        "model": args.model,
        "signal_type": args.signal_type,
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val,
        "mask_ratio": mask_ratio,
        "context_weight": args.context_weight,
        "target_length": args.target_length,
        "curriculum": args.curriculum,
    }, os.path.join(checkpoint_dir, f"{ckpt_prefix}_final.pt"))

    # Save history
    history_csv = os.path.join(plot_dir, f"{ckpt_prefix}_history.csv")
    history_png = os.path.join(plot_dir, f"{ckpt_prefix}_history.png")
    title = f"Exp1 {signal_tag}/{model_tag}/{len_tag}{' curriculum' if args.curriculum else ''}"
    save_history(history_rows, history_csv, history_png, title=title)

    print(f"\n✓ Training complete. Best val loss: {best_val:.4f}")
    print(f"  History: {history_csv}")
    print(f"  Plot:    {history_png}")
    print(f"  Checkpoints: {checkpoint_dir}/{ckpt_prefix}_best.pt")
    print(f"               {checkpoint_dir}/{ckpt_prefix}_final.pt")


if __name__ == "__main__":
    main()
