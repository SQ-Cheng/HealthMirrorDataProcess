"""Experiment 04 training: autoencoder-based artifact detector and reconstruction SQI."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp4_dataloader import build_artifact_dataloaders
from exp4_model import build_exp4_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 04: Autoencoder artifact detection")
    parser.add_argument("--variant", choices=["light", "full"], default="full")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--clean-percentile", type=float, default=90.0)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    max_batches=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    snrs = []
    rec_errors = []

    for batch_idx, (x, snr) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(DEVICE)
        snr = snr.to(DEVICE)
        target = x

        # No synthetic corruption/augmentation: use only observed windows.
        x_in = x

        if is_train:
            optimizer.zero_grad()

        pred = model(x_in)
        per_sample = ((pred - target) ** 2).mean(dim=(1, 2))
        loss = per_sample.mean()

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        snrs.extend(snr.detach().cpu().numpy().tolist())
        rec_errors.extend(per_sample.detach().cpu().numpy().tolist())

    if not losses:
        return {"loss": 0.0, "snr_rec_corr": 0.0, "q_sep": 0.0}

    snrs_np = np.array(snrs)
    errs_np = np.array(rec_errors)

    if len(snrs_np) >= 2:
        corr = float(np.corrcoef(snrs_np, errs_np)[0, 1])
    else:
        corr = 0.0

    q25 = np.percentile(snrs_np, 25)
    q75 = np.percentile(snrs_np, 75)
    low_err = errs_np[snrs_np <= q25].mean() if np.any(snrs_np <= q25) else errs_np.mean()
    high_err = errs_np[snrs_np >= q75].mean() if np.any(snrs_np >= q75) else errs_np.mean()
    q_sep = float(low_err - high_err)

    return {
        "loss": float(np.mean(losses)),
        "snr_rec_corr": corr,
        "q_sep": q_sep,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("Loading data ...")
    train_loader, val_loader, threshold = build_artifact_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        clean_percentile=args.clean_percentile,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    model = build_exp4_model(args.variant).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Model variant: {args.variant}, params: {param_count:,}")
    print(f"Train-clean SNR threshold: {threshold:.2f} dB")

    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  "
        f"{'VaCorr(SNR,Err)':>15}  {'VaQSep':>8}  {'Time':>6}"
    )
    print("-" * 70)

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
        score = va["loss"]
        if score < best_val:
            best_val = score
            torch.save(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_score": best_val,
                    "val_loss": va["loss"],
                    "snr_threshold": threshold,
                },
                os.path.join(SAVE_DIR, f"exp4_{args.variant}_best.pt"),
            )
            marker = " *"

        print(
            f"{epoch:5d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
            f"{va['snr_rec_corr']:15.4f}  {va['q_sep']:8.4f}  {elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_score": best_val,
            "snr_threshold": threshold,
        },
        os.path.join(SAVE_DIR, f"exp4_{args.variant}_final.pt"),
    )

    print(f"\nDone. Best val score: {best_val:.4f}")


if __name__ == "__main__":
    main()
