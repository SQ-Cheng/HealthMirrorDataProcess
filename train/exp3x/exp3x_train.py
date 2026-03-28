"""Experiment 03X training for full-scale masked ECG+rPPG reconstruction."""

import argparse
import csv
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
EXP3_DIR = os.path.join(ROOT_DIR, "train", "exp3")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")
PLOT_DIR = os.path.join(ROOT_DIR, "train", "exp3x", "plots")

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, EXP3_DIR)

from exp3x_model import build_exp3x_model

try:
    from train.exp3.exp3_dataloader import build_masked_recon_dataloaders
    from train.exp3.exp3_train import build_visible_mask, recon_mae_by_channel, weighted_masked_loss
except ModuleNotFoundError:
    from exp3_dataloader import build_masked_recon_dataloaders  # type: ignore[import-not-found]
    from exp3_train import build_visible_mask, recon_mae_by_channel, weighted_masked_loss  # type: ignore[import-not-found]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 03X: full-scale training for candidate models")
    parser.add_argument(
        "--model",
        choices=["unet_gated", "dual_head", "tcn_ssm", "cross_attention", "mamba"],
        default="tcn_ssm",
    )
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-block-min", type=int, default=8)
    parser.add_argument("--mask-block-max", type=int, default=32)
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--ecg-point-weight", type=float, default=1.25)
    parser.add_argument("--rppg-point-weight", type=float, default=1.0)
    parser.add_argument("--grad-loss-weight", type=float, default=0.1)
    parser.add_argument("--ecg-fft-loss-weight", type=float, default=0.02)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, args, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ecg_maes = []
    rppg_maes = []

    max_batches = args.max_train_batches if is_train else args.max_val_batches

    for batch_idx, (pair, clean_score, _, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pair = pair.to(DEVICE)
        clean_score = clean_score.to(DEVICE)

        visible = build_visible_mask(
            pair,
            mask_ratio=args.mask_ratio,
            block_min=args.mask_block_min,
            block_max=args.mask_block_max,
        )
        masked = 1.0 - visible
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
            context_weight=args.context_weight,
            ecg_point_weight=args.ecg_point_weight,
            rppg_point_weight=args.rppg_point_weight,
            grad_loss_weight=args.grad_loss_weight,
            ecg_fft_loss_weight=args.ecg_fft_loss_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        ecg_mae, rppg_mae = recon_mae_by_channel(pred, pair, masked)
        losses.append(loss.item())
        ecg_maes.append(ecg_mae)
        rppg_maes.append(rppg_mae)

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "ecg_mae": float(sum(ecg_maes) / max(len(ecg_maes), 1)),
        "rppg_mae": float(sum(rppg_maes) / max(len(rppg_maes), 1)),
    }


def save_history(rows, csv_path, plot_path, title):
    fields = ["epoch", "tr_loss", "va_loss", "tr_ecg_mae", "tr_rppg_mae", "va_ecg_mae", "va_rppg_mae", "seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    epochs = [r["epoch"] for r in rows]
    tr_loss = [r["tr_loss"] for r in rows]
    va_loss = [r["va_loss"] for r in rows]
    tr_ecg = [r["tr_ecg_mae"] for r in rows]
    va_ecg = [r["va_ecg_mae"] for r in rows]
    tr_rppg = [r["tr_rppg_mae"] for r in rows]
    va_rppg = [r["va_rppg_mae"] for r in rows]

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
    set_seed(args.seed)

    print("Loading Exp3X data ...")
    train_loader, val_loader = build_masked_recon_dataloaders(
        ROOT_DIR,
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

    model = build_exp3x_model(args.model).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    tag = args.checkpoint_tag.strip()
    prefix = f"exp3x_{args.model}{tag}"
    best_val = float("inf")
    rows = []

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}, params: {params:,}")
    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrECG_MAE':>10}  {'TrRPPG_MAE':>11}  "
        f"{'VaECG_MAE':>10}  {'VaRPPG_MAE':>11}  {'Time':>6}"
    )
    print("-" * 96)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, args, optimizer=optimizer)
        with torch.no_grad():
            va = run_epoch(model, val_loader, criterion, args, optimizer=None)
        sec = time.time() - t0

        marker = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model": args.model,
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
                os.path.join(CHECKPOINT_DIR, f"{prefix}_best.pt"),
            )
            marker = " *"

        rows.append(
            {
                "epoch": epoch,
                "tr_loss": tr["loss"],
                "va_loss": va["loss"],
                "tr_ecg_mae": tr["ecg_mae"],
                "tr_rppg_mae": tr["rppg_mae"],
                "va_ecg_mae": va["ecg_mae"],
                "va_rppg_mae": va["rppg_mae"],
                "seconds": sec,
            }
        )

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['ecg_mae']:10.4f}  {tr['rppg_mae']:11.4f}  "
            f"{va['ecg_mae']:10.4f}  {va['rppg_mae']:11.4f}  "
            f"{sec:5.1f}s{marker}"
        )

    torch.save(
        {
            "model": args.model,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "checkpoint_tag": tag,
            "mask_ratio": args.mask_ratio,
            "target_length": args.target_length,
            "ecg_point_weight": args.ecg_point_weight,
            "rppg_point_weight": args.rppg_point_weight,
            "grad_loss_weight": args.grad_loss_weight,
            "ecg_fft_loss_weight": args.ecg_fft_loss_weight,
        },
        os.path.join(CHECKPOINT_DIR, f"{prefix}_final.pt"),
    )

    history_csv = os.path.join(PLOT_DIR, f"{prefix}_history.csv")
    history_png = os.path.join(PLOT_DIR, f"{prefix}_history.png")
    save_history(rows, history_csv, history_png, title=f"Exp3X Training History ({prefix})")

    print(f"Saved history: {history_csv}")
    print(f"Saved history plot: {history_png}")
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
