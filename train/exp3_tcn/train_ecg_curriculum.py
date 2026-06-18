"""Experiment 03 ECG TCN Curriculum Learning — mask_ratio 0.15→0.60 over 300 epochs."""

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp3_common.single_recon_dataloader import build_single_signal_dataloaders
from exp3_common.single_recon_train import (
    build_single_window_visible_mask,
    weighted_masked_loss,
    masked_mae,
)
from exp3_tcn.single_recon_model import build_single_recon_tcn_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def get_curriculum_mask_ratio(epoch, total_epochs, min_ratio=0.15, max_ratio=0.60):
    """Linear curriculum: mask_ratio increases from min_ratio to max_ratio."""
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return min_ratio + (max_ratio - min_ratio) * progress


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp3 ECG TCN Curriculum Learning: mask_ratio 0.15→0.60 across epochs"
    )
    parser.add_argument("--variant", choices=["light", "full"], default="full")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Optional data folder override.",
    )
    parser.add_argument(
        "--data-source", choices=["sqi", "cleaned"], default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--mask-min", type=float, default=0.1)
    parser.add_argument("--mask-max", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--best-tolerance", type=float, default=0.008,
                        help="Save checkpoint as long as val_loss <= best_val + tolerance (curriculum-friendly).")
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    return parser.parse_args()


def run_epoch(
    model, loader, criterion, mask_ratio, context_weight, optimizer=None, max_batches=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses, maes = [], []

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
            pred, signal, visible, quality_score, criterion,
            context_weight=context_weight,
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
    fieldnames = ["epoch", "tr_loss", "va_loss", "tr_mae", "va_mae", "seconds", "mask_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["tr_loss"] for r in history_rows]
    va_loss = [r["va_loss"] for r in history_rows]
    tr_mae = [r["tr_mae"] for r in history_rows]
    va_mae = [r["va_mae"] for r in history_rows]
    mask_ratios = [r["mask_ratio"] for r in history_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

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

    axes[2].plot(epochs, mask_ratios, color="darkorange", linewidth=1.5)
    axes[2].set_title("Curriculum mask_ratio")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mask_ratio")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=180)


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[Curriculum] Loading data from: {data_dir} (source={args.data_source}) ...")
    train_loader, val_loader = build_single_signal_dataloaders(
        data_dir,
        signal_type="ecg",
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

    model = build_single_recon_tcn_model(args.variant).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"Model variant: {args.variant}, params: {param_count:,}")
    print(f"Curriculum: mask_ratio {args.mask_min:.2f} → {args.mask_max:.2f} over {args.epochs} epochs (linear)")

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
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp3_ecg_tcn_curriculum", "plots")
    plot_dir = os.path.normpath(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    ckpt_prefix = f"exp3_ecg_tcn_{args.variant}_curriculum"

    best_val = float("inf")
    history_rows = []

    print(f"[Curriculum] Checkpoint tolerance: {args.best_tolerance:.4f}")
    print(f"\n{'Epoch':>5}  {'Mask%':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrMAE':>8}  {'VaMAE':>8}  {'Time':>6}  Note")
    print("-" * 78)

    for epoch in range(start_epoch, args.epochs + 1):
        mask_ratio = get_curriculum_mask_ratio(
            epoch, args.epochs, min_ratio=args.mask_min, max_ratio=args.mask_max
        )

        t0 = time.time()

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

        # Curriculum-aware checkpointing:
        # Save if val_loss is an improvement OR within tolerance of best.
        should_save = va["loss"] < best_val or va["loss"] <= best_val + args.best_tolerance
        note = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            note = "best"
        elif should_save:
            note = "tol"

        if should_save:
            torch.save(
                {
                    "signal_type": "ecg",
                    "variant": args.variant,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "epoch_loss": va["loss"],
                    "mask_ratio": mask_ratio,
                    "mask_min": args.mask_min,
                    "mask_max": args.mask_max,
                    "best_tolerance": args.best_tolerance,
                    "target_length": args.target_length,
                    "model_family": "single_recon_tcn_v1",
                    "curriculum": True,
                },
                os.path.join(SAVE_DIR, f"{ckpt_prefix}_best.pt"),
            )

        history_rows.append({
            "epoch": epoch,
            "tr_loss": tr["loss"],
            "va_loss": va["loss"],
            "tr_mae": tr["mae"],
            "va_mae": va["mae"],
            "seconds": elapsed,
            "mask_ratio": round(mask_ratio, 4),
        })

        print(
            f"{epoch:5d}  {mask_ratio:5.3f}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['mae']:8.4f}  {va['mae']:8.4f}  {elapsed:5.1f}s  {note:>4s}"
        )

    torch.save(
        {
            "signal_type": "ecg",
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val,
            "mask_ratio": mask_ratio,
            "mask_min": args.mask_min,
            "mask_max": args.mask_max,
            "target_length": args.target_length,
            "model_family": "single_recon_tcn_v1",
            "curriculum": True,
        },
        os.path.join(SAVE_DIR, f"{ckpt_prefix}_final.pt"),
    )

    history_csv = os.path.join(plot_dir, f"{ckpt_prefix}_history.csv")
    history_png = os.path.join(plot_dir, f"{ckpt_prefix}_history.png")
    save_history(history_rows, history_csv, history_png,
                 title=f"exp3_ecg_tcn Curriculum ({args.mask_min}→{args.mask_max}) Training History")

    print(f"\nSaved history: {history_csv}")
    print(f"Saved history plot: {history_png}")
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
