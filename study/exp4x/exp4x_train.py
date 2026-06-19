"""Experiment 04-X training: full-data SNR-rank SQI regression."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp4x_dataloader import build_exp4x_dataloaders
from exp4x_model import build_exp4x_model, count_trainable_params


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 04-X: SQI regression from full rPPG")
    parser.add_argument("--model", choices=["exp4-1", "exp4-2", "exp4-3"], default="exp4-1")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional data root directory containing mirror*_auto_cleaned[_sqi] folders",
    )
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def _rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _safe_pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x, y):
    if len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _safe_pearson(rx, ry)


def run_epoch(model, loader, criterion, optimizer=None, max_batches=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    y_true = []
    y_pred = []
    snrs = []

    for batch_idx, (x, sqi, snr) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(DEVICE)
        sqi = sqi.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, sqi)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        y_true.extend(sqi.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        snrs.extend(snr.detach().cpu().numpy().tolist())

    if not losses:
        return {
            "loss": 0.0,
            "mae": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "snr_pred_corr": 0.0,
        }

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    snr_np = np.array(snrs)

    return {
        "loss": float(np.mean(losses)),
        "mae": float(np.mean(np.abs(y_true_np - y_pred_np))),
        "pearson": _safe_pearson(y_true_np, y_pred_np),
        "spearman": _safe_spearman(y_true_np, y_pred_np),
        "snr_pred_corr": _safe_pearson(snr_np, y_pred_np),
    }


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    torch.manual_seed(args.seed)

    print(f"Loading Exp4-X data from: {data_dir} (source={args.data_source}) ...")
    train_loader, val_loader = build_exp4x_dataloaders(
        data_dir,
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

    model = build_exp4x_model(args.model).to(DEVICE)
    param_count = count_trainable_params(model)
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}, trainable params: {param_count:,}")

    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaMAE':>8}  "
        f"{'VaPear':>8}  {'VaSpear':>8}  {'VaSNRcorr':>10}  {'Time':>6}"
    )
    print("-" * 88)

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
                    "model_name": args.model,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va["loss"],
                    "val_mae": va["mae"],
                    "target_length": args.target_length,
                    "window_sec": args.window_sec,
                },
                os.path.join(SAVE_DIR, f"exp4x_{args.model}_best.pt"),
            )
            marker = " *"

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  {va['mae']:8.4f}  "
            f"{va['pearson']:8.4f}  {va['spearman']:8.4f}  {va['snr_pred_corr']:10.4f}  {elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "model_name": args.model,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "target_length": args.target_length,
            "window_sec": args.window_sec,
        },
        os.path.join(SAVE_DIR, f"exp4x_{args.model}_final.pt"),
    )

    print(f"\nDone. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
