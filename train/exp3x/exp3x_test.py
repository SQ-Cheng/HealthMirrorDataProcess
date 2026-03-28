"""Experiment 03X smoke test runner for candidate model structures."""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
EXP3_DIR = os.path.join(ROOT_DIR, "train", "exp3")

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
    parser = argparse.ArgumentParser(description="Exp3X quick smoke tests for 4 model structures")
    parser.add_argument(
        "--models",
        type=str,
        default="unet_gated,dual_head,tcn_ssm,cross_attention",
        help="Comma-separated model list",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
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
    parser.add_argument("--max-patients", type=int, default=80)
    parser.add_argument("--max-windows-per-patient", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=10)
    parser.add_argument("--max-val-batches", type=int, default=4)
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


def main():
    args = parse_args()
    set_seed(args.seed)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided")

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

    out_dir = os.path.join(ROOT_DIR, "train", "exp3x", "plots")
    os.makedirs(out_dir, exist_ok=True)

    criterion = nn.SmoothL1Loss(reduction="none")
    rows = []

    print(f"Device: {DEVICE}")
    print(f"Models: {models}")

    for model_name in models:
        model = build_exp3x_model(model_name).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        t0 = time.time()
        tr = None
        va = None
        for _ in range(args.epochs):
            tr = run_epoch(model, train_loader, criterion, args, optimizer=optimizer)
            with torch.no_grad():
                va = run_epoch(model, val_loader, criterion, args, optimizer=None)
        sec = time.time() - t0

        row = {
            "model": model_name,
            "params": int(params),
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_ecg_mae": va["ecg_mae"],
            "val_rppg_mae": va["rppg_mae"],
            "seconds": sec,
        }
        rows.append(row)

        print(
            f"[{model_name}] params={params:,} "
            f"TrLoss={tr['loss']:.4f} VaLoss={va['loss']:.4f} "
            f"VaECG_MAE={va['ecg_mae']:.4f} VaRPPG_MAE={va['rppg_mae']:.4f} "
            f"time={sec:.1f}s"
        )

    rows = sorted(rows, key=lambda x: x["val_loss"])

    csv_path = os.path.join(out_dir, "exp3x_smoke_results.csv")
    json_path = os.path.join(out_dir, "exp3x_smoke_results.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "params", "train_loss", "val_loss", "val_ecg_mae", "val_rppg_mae", "seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "args": vars(args),
        "results": rows,
        "best_model_by_val_loss": rows[0]["model"] if rows else None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if rows:
        print(f"Best by val_loss: {rows[0]['model']} ({rows[0]['val_loss']:.4f})")


if __name__ == "__main__":
    main()
