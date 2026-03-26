"""Experiment 03 training: multitask rPPG regression for HR and SpO2."""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import (
    denormalize_hr,
    denormalize_spo2,
    build_vitals_dataloaders,
)
from exp3_model import build_exp3_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 03: rPPG -> HR + SpO2")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def masked_huber(pred, target, mask, criterion):
    loss = criterion(pred, target)
    weighted = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def compute_mae_metrics(pred, target, mask):
    pred_hr = denormalize_hr(pred[:, 0])
    gt_hr = denormalize_hr(target[:, 0])
    pred_spo2 = denormalize_spo2(pred[:, 1])
    gt_spo2 = denormalize_spo2(target[:, 1])

    hr_mask = mask[:, 0] > 0.5
    spo2_mask = mask[:, 1] > 0.5

    hr_mae = (pred_hr[hr_mask] - gt_hr[hr_mask]).abs().mean().item() if hr_mask.any() else float("nan")
    spo2_mae = (pred_spo2[spo2_mask] - gt_spo2[spo2_mask]).abs().mean().item() if spo2_mask.any() else float("nan")
    return hr_mae, spo2_mae


def run_epoch(model, loader, criterion, optimizer=None, max_batches=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n = 0
    hr_maes = []
    spo2_maes = []

    for batch_idx, (rppg, target, mask) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        rppg = rppg.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        pred = model(rppg)
        loss = masked_huber(pred, target, mask, criterion)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            hr_mae, spo2_mae = compute_mae_metrics(pred, target, mask)
            if not torch.isnan(torch.tensor(hr_mae)):
                hr_maes.append(hr_mae)
            if not torch.isnan(torch.tensor(spo2_mae)):
                spo2_maes.append(spo2_mae)

        total_loss += loss.item()
        n += 1

    out = {
        "loss": total_loss / max(n, 1),
        "hr_mae": sum(hr_maes) / max(len(hr_maes), 1),
        "spo2_mae": sum(spo2_maes) / max(len(spo2_maes), 1),
    }
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("Loading data ...")
    train_loader, val_loader = build_vitals_dataloaders(
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
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Model variant: {args.variant}, params: {param_count:,}")

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  "
        f"{'TrHR_MAE':>9}  {'TrSpO2':>8}  {'VaHR_MAE':>9}  {'VaSpO2':>8}  {'Time':>6}"
    )
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                max_batches=args.max_val_batches,
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
                },
                os.path.join(SAVE_DIR, f"exp3_{args.variant}_best.pt"),
            )
            marker = " *"

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['hr_mae']:9.3f}  {tr['spo2_mae']:8.3f}  "
            f"{va['hr_mae']:9.3f}  {va['spo2_mae']:8.3f}  "
            f"{elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val,
        },
        os.path.join(SAVE_DIR, f"exp3_{args.variant}_final.pt"),
    )

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
