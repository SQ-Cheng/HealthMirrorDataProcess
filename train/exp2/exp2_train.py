"""
Experiment 02 Training Script
=============================
Train a bidirectional GAN for ECG <-> rPPG translation.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam


# Add train/ to path so imports work when run from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp2_dataloader import build_paired_dataloaders
from exp2_model import build_exp2_models, init_weights


LAMBDA_PAIR = 10.0
LAMBDA_CYCLE = 5.0
LAMBDA_IDENTITY = 2.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 02: Bidirectional ECG<->rPPG GAN")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def _adv_targets_like(tensor, is_real):
    fill_value = 1.0 if is_real else 0.0
    return torch.full_like(tensor, fill_value)


def train_one_epoch(
    g_e2r,
    g_r2e,
    d_rppg,
    d_ecg,
    loader,
    adv_criterion,
    l1_criterion,
    g_optimizer,
    d_optimizer,
    max_batches=None,
):
    g_e2r.train()
    g_r2e.train()
    d_rppg.train()
    d_ecg.train()

    stat = {
        "g_total": 0.0,
        "g_adv": 0.0,
        "g_pair": 0.0,
        "g_cycle": 0.0,
        "g_identity": 0.0,
        "d_total": 0.0,
        "n": 0,
    }

    for batch_idx, (real_ecg, real_rppg) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        real_ecg = real_ecg.to(DEVICE)
        real_rppg = real_rppg.to(DEVICE)

        # ---- Train discriminators ----
        d_optimizer.zero_grad()

        with torch.no_grad():
            fake_rppg_detached = g_e2r(real_ecg)
            fake_ecg_detached = g_r2e(real_rppg)

        pred_real_rppg = d_rppg(real_rppg)
        pred_fake_rppg = d_rppg(fake_rppg_detached)
        loss_d_rppg = 0.5 * (
            adv_criterion(pred_real_rppg, _adv_targets_like(pred_real_rppg, True))
            + adv_criterion(pred_fake_rppg, _adv_targets_like(pred_fake_rppg, False))
        )

        pred_real_ecg = d_ecg(real_ecg)
        pred_fake_ecg = d_ecg(fake_ecg_detached)
        loss_d_ecg = 0.5 * (
            adv_criterion(pred_real_ecg, _adv_targets_like(pred_real_ecg, True))
            + adv_criterion(pred_fake_ecg, _adv_targets_like(pred_fake_ecg, False))
        )

        loss_d = loss_d_rppg + loss_d_ecg
        loss_d.backward()
        d_optimizer.step()

        # ---- Train generators ----
        g_optimizer.zero_grad()

        fake_rppg = g_e2r(real_ecg)
        fake_ecg = g_r2e(real_rppg)

        rec_ecg = g_r2e(fake_rppg)
        rec_rppg = g_e2r(fake_ecg)

        id_ecg = g_r2e(real_ecg)
        id_rppg = g_e2r(real_rppg)

        pred_fake_rppg_for_g = d_rppg(fake_rppg)
        pred_fake_ecg_for_g = d_ecg(fake_ecg)

        loss_g_adv = (
            adv_criterion(pred_fake_rppg_for_g, _adv_targets_like(pred_fake_rppg_for_g, True))
            + adv_criterion(pred_fake_ecg_for_g, _adv_targets_like(pred_fake_ecg_for_g, True))
        )
        loss_g_pair = l1_criterion(fake_rppg, real_rppg) + l1_criterion(fake_ecg, real_ecg)
        loss_g_cycle = l1_criterion(rec_ecg, real_ecg) + l1_criterion(rec_rppg, real_rppg)
        loss_g_identity = l1_criterion(id_ecg, real_ecg) + l1_criterion(id_rppg, real_rppg)

        loss_g = (
            loss_g_adv
            + LAMBDA_PAIR * loss_g_pair
            + LAMBDA_CYCLE * loss_g_cycle
            + LAMBDA_IDENTITY * loss_g_identity
        )
        loss_g.backward()
        g_optimizer.step()

        stat["g_total"] += loss_g.item()
        stat["g_adv"] += loss_g_adv.item()
        stat["g_pair"] += loss_g_pair.item()
        stat["g_cycle"] += loss_g_cycle.item()
        stat["g_identity"] += loss_g_identity.item()
        stat["d_total"] += loss_d.item()
        stat["n"] += 1

    n = max(stat["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in stat.items()}


@torch.no_grad()
def validate(g_e2r, g_r2e, loader, l1_criterion, max_batches=None):
    g_e2r.eval()
    g_r2e.eval()

    total_pair = 0.0
    total_cycle = 0.0
    n = 0

    for batch_idx, (real_ecg, real_rppg) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        real_ecg = real_ecg.to(DEVICE)
        real_rppg = real_rppg.to(DEVICE)

        fake_rppg = g_e2r(real_ecg)
        fake_ecg = g_r2e(real_rppg)
        rec_ecg = g_r2e(fake_rppg)
        rec_rppg = g_e2r(fake_ecg)

        pair_loss = l1_criterion(fake_rppg, real_rppg) + l1_criterion(fake_ecg, real_ecg)
        cycle_loss = l1_criterion(rec_ecg, real_ecg) + l1_criterion(rec_rppg, real_rppg)

        total_pair += pair_loss.item()
        total_cycle += cycle_loss.item()
        n += 1

    n = max(n, 1)
    return {
        "pair": total_pair / n,
        "cycle": total_cycle / n,
        "combined": (total_pair + total_cycle) / n,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("Loading data ...")
    train_loader, val_loader = build_paired_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        debug=True,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    if len(train_loader.dataset) == 0:
        print("[ERROR] No training data found. Exiting.")
        return

    g_e2r, g_r2e, d_rppg, d_ecg = build_exp2_models(args.variant)
    g_e2r = g_e2r.to(DEVICE)
    g_r2e = g_r2e.to(DEVICE)
    d_rppg = d_rppg.to(DEVICE)
    d_ecg = d_ecg.to(DEVICE)

    g_e2r.apply(init_weights)
    g_r2e.apply(init_weights)
    d_rppg.apply(init_weights)
    d_ecg.apply(init_weights)

    g_params = sum(p.numel() for p in g_e2r.parameters()) + sum(p.numel() for p in g_r2e.parameters())
    d_params = sum(p.numel() for p in d_rppg.parameters()) + sum(p.numel() for p in d_ecg.parameters())
    print(f"Device: {DEVICE}")
    print(f"Generator params (2x): {g_params:,}")
    print(f"Discriminator params (2x): {d_params:,}")

    adv_criterion = nn.BCEWithLogitsLoss()
    l1_criterion = nn.L1Loss()

    g_optimizer = Adam(
        list(g_e2r.parameters()) + list(g_r2e.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    d_optimizer = Adam(
        list(d_rppg.parameters()) + list(d_ecg.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    print(
        f"\n{'Epoch':>5}  {'G_total':>9}  {'D_total':>9}  "
        f"{'G_adv':>8}  {'G_pair':>8}  {'G_cycle':>8}  {'G_id':>8}  "
        f"{'Val_pair':>9}  {'Val_cycle':>9}  {'Time':>6}"
    )
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stat = train_one_epoch(
            g_e2r,
            g_r2e,
            d_rppg,
            d_ecg,
            train_loader,
            adv_criterion,
            l1_criterion,
            g_optimizer,
            d_optimizer,
            max_batches=args.max_train_batches,
        )
        val_stat = validate(
            g_e2r,
            g_r2e,
            val_loader,
            l1_criterion,
            max_batches=args.max_val_batches,
        )

        elapsed = time.time() - t0
        marker = ""

        if val_stat["combined"] < best_val:
            best_val = val_stat["combined"]
            torch.save(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "g_e2r": g_e2r.state_dict(),
                    "g_r2e": g_r2e.state_dict(),
                    "d_rppg": d_rppg.state_dict(),
                    "d_ecg": d_ecg.state_dict(),
                    "g_optimizer": g_optimizer.state_dict(),
                    "d_optimizer": d_optimizer.state_dict(),
                    "val_combined": best_val,
                },
                os.path.join(SAVE_DIR, f"exp2_{args.variant}_best.pt"),
            )
            marker = " *"

        print(
            f"{epoch:5d}  "
            f"{train_stat['g_total']:9.4f}  {train_stat['d_total']:9.4f}  "
            f"{train_stat['g_adv']:8.4f}  {train_stat['g_pair']:8.4f}  "
            f"{train_stat['g_cycle']:8.4f}  {train_stat['g_identity']:8.4f}  "
            f"{val_stat['pair']:9.4f}  {val_stat['cycle']:9.4f}  "
            f"{elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "variant": args.variant,
            "epoch": args.epochs,
            "g_e2r": g_e2r.state_dict(),
            "g_r2e": g_r2e.state_dict(),
            "d_rppg": d_rppg.state_dict(),
            "d_ecg": d_ecg.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "val_combined": best_val,
        },
        os.path.join(SAVE_DIR, f"exp2_{args.variant}_final.pt"),
    )

    print(f"\nDone. Best validation combined loss: {best_val:.4f}")
    print(f"Checkpoints saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
