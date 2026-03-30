"""
Experiment 01T Training Script
==============================
Trains a CNN+Transformer model for blood pressure prediction
from ECG + rPPG windows.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam

# Add project root to path so imports work from any working directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, ROOT_DIR)

from train.exp1t.exp1t_model import CNNTransformerBP
from train.exp1t.exp1t_dataloader import build_dataloaders, BP_MIN, BP_MAX

BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
VAL_RATIO = 0.2
SEED = 43
WINDOW_SEC = 2.0
STEP_SEC = 1.0
TARGET_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 01T training (CNN+Transformer)")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    e_sbp = []
    e_dbp = []

    for ecg, rppg, bp in loader:
        if torch.isnan(ecg).any() or torch.isnan(rppg).any() or torch.isnan(bp).any():
            continue

        ecg = ecg.to(DEVICE)
        rppg = rppg.to(DEVICE)
        bp = bp.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(ecg, rppg)
        loss = criterion(pred, bp)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            bp_mmhg = bp * (BP_MAX - BP_MIN) + BP_MIN
            pred_mmhg = pred * (BP_MAX - BP_MIN) + BP_MIN
            err = (pred_mmhg - bp_mmhg).detach().cpu()
            e_sbp.append(err[:, 0])
            e_dbp.append(err[:, 1])

    e_sbp = torch.cat(e_sbp) if e_sbp else torch.zeros(1)
    e_dbp = torch.cat(e_dbp) if e_dbp else torch.zeros(1)

    return (
        total_loss / max(n_batches, 1),
        e_sbp.mean().item(), e_sbp.std().item(),
        e_dbp.mean().item(), e_dbp.std().item(),
    )


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    e_sbp = []
    e_dbp = []

    for ecg, rppg, bp in loader:
        ecg = ecg.to(DEVICE)
        rppg = rppg.to(DEVICE)
        bp = bp.to(DEVICE)

        pred = model(ecg, rppg)
        loss = criterion(pred, bp)
        total_loss += loss.item()
        n_batches += 1

        bp_mmhg = bp * (BP_MAX - BP_MIN) + BP_MIN
        pred_mmhg = pred * (BP_MAX - BP_MIN) + BP_MIN
        err = (pred_mmhg - bp_mmhg).detach().cpu()
        e_sbp.append(err[:, 0])
        e_dbp.append(err[:, 1])

    e_sbp = torch.cat(e_sbp) if e_sbp else torch.zeros(1)
    e_dbp = torch.cat(e_dbp) if e_dbp else torch.zeros(1)

    return (
        total_loss / max(n_batches, 1),
        e_sbp.mean().item(), e_sbp.std().item(),
        e_dbp.mean().item(), e_dbp.std().item(),
    )


def main():
    args = parse_args()
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print("Loading data ...")
    train_loader, val_loader = build_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=VAL_RATIO,
        seed=SEED,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        target_length=TARGET_LENGTH,
        data_source=args.data_source,
        debug=True,
    )

    if len(train_loader.dataset) == 0:
        print("[ERROR] No training data found. Exiting.")
        return

    model = CNNTransformerBP().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")

    print(
        f"\n{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  "
        f"[------- Train (mmHg) -------]  "
        f"[-------- Val (mmHg) --------]  {'Time':>6}"
    )
    print(
        f"{'':5}  {'':8}  {'':8}  "
        f"{'SBP ME':>9}{'SBP SD':>9}  {'DBP ME':>9}{'DBP SD':>9}  "
        f"{'SBP ME':>9}{'SBP SD':>9}  {'DBP ME':>9}{'DBP SD':>9}  {'':6}"
    )
    print("-" * 105)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer)
        va = validate(model, val_loader, criterion)

        train_loss, tr_me_sbp, tr_sd_sbp, tr_me_dbp, tr_sd_dbp = tr
        val_loss, va_me_sbp, va_sd_sbp, va_me_dbp, va_sd_dbp = va

        elapsed = time.time() - t0
        marker = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "exp1t_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_path,
            )
            marker = " *"

        print(
            f"{epoch:5d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{tr_me_sbp:7.2f}  {tr_sd_sbp:8.3f}  "
            f"{tr_me_dbp:7.2f}  {tr_sd_dbp:8.3f}  "
            f"| {va_me_sbp:7.2f}  {va_sd_sbp:8.3f}  "
            f"{va_me_dbp:7.2f}  {va_sd_dbp:8.3f}  "
            f"{elapsed:5.1f}s{marker}"
        )

    final_path = os.path.join(SAVE_DIR, "exp1t_final.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        final_path,
    )

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
