"""GAN training loop for Exp3 single-signal masked reconstruction."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from exp3_common.single_recon_dataloader import build_single_signal_dataloaders
from exp3_common.single_recon_train import build_single_window_visible_mask, masked_mae
from single_recon_model import build_exp3_gan_models, init_weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args(exp_name):
    parser = argparse.ArgumentParser(description=f"{exp_name}: GAN single-signal masked reconstruction")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Optional data folder override. Supports either a single mirror folder "
            "(contains patient_*.csv) or a parent folder that contains "
            "mirror*_auto_cleaned[_sqi] folders."
        ),
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
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--adv-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def reconstruction_loss(pred, target, visible_mask, quality_score, context_weight):
    criterion = nn.SmoothL1Loss(reduction="none")

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


def _adv_targets_like(tensor, is_real):
    fill_value = 1.0 if is_real else 0.0
    return torch.full_like(tensor, fill_value)


def train_epoch(
    generator,
    discriminator,
    loader,
    g_optimizer,
    d_optimizer,
    adv_criterion,
    max_batches=None,
    mask_ratio=0.3,
    context_weight=0.2,
    adv_loss_weight=0.5,
):
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []
    maes = []

    for batch_idx, (signal, quality_score) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        signal = signal.to(DEVICE)
        quality_score = quality_score.to(DEVICE)

        visible = build_single_window_visible_mask(signal, mask_ratio=mask_ratio)
        masked_mask = 1.0 - visible
        x_masked = signal * visible

        # ---- Train discriminator ----
        d_optimizer.zero_grad()
        with torch.no_grad():
            fake_signal_detached = generator(x_masked, visible)

        pred_real = discriminator(signal)
        pred_fake = discriminator(fake_signal_detached)
        d_loss = 0.5 * (
            adv_criterion(pred_real, _adv_targets_like(pred_real, True))
            + adv_criterion(pred_fake, _adv_targets_like(pred_fake, False))
        )
        d_loss.backward()
        d_optimizer.step()

        # ---- Train generator ----
        g_optimizer.zero_grad()
        fake_signal = generator(x_masked, visible)
        pred_fake_for_g = discriminator(fake_signal)

        rec_loss = reconstruction_loss(
            fake_signal,
            signal,
            visible,
            quality_score,
            context_weight=context_weight,
        )
        g_adv = adv_criterion(pred_fake_for_g, _adv_targets_like(pred_fake_for_g, True))
        g_loss = rec_loss + adv_loss_weight * g_adv

        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        maes.append(masked_mae(fake_signal, signal, masked_mask))

    return {
        "g_loss": sum(g_losses) / max(len(g_losses), 1),
        "d_loss": sum(d_losses) / max(len(d_losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


@torch.no_grad()
def validate_epoch(
    generator,
    loader,
    max_batches=None,
    mask_ratio=0.3,
    context_weight=0.2,
):
    generator.eval()

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

        pred = generator(x_masked, visible)
        loss = reconstruction_loss(
            pred,
            signal,
            visible,
            quality_score,
            context_weight=context_weight,
        )

        losses.append(loss.item())
        maes.append(masked_mae(pred, signal, masked_mask))

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


def save_history(history_rows, csv_path, plot_path, title):
    fieldnames = ["epoch", "tr_g_loss", "tr_d_loss", "va_loss", "tr_mae", "va_mae", "seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    epochs = [r["epoch"] for r in history_rows]
    tr_g_loss = [r["tr_g_loss"] for r in history_rows]
    tr_d_loss = [r["tr_d_loss"] for r in history_rows]
    va_loss = [r["va_loss"] for r in history_rows]
    tr_mae = [r["tr_mae"] for r in history_rows]
    va_mae = [r["va_mae"] for r in history_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, tr_g_loss, label="train G")
    axes[0].plot(epochs, tr_d_loss, label="train D")
    axes[0].plot(epochs, va_loss, label="val recon")
    axes[0].set_title("GAN/Reconstruction Loss")
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
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from: {data_dir} (source={args.data_source}) ...")
    train_loader, val_loader = build_single_signal_dataloaders(
        data_dir,
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

    generator, discriminator = build_exp3_gan_models(args.variant)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"Signal: {signal_type}, variant: {args.variant}, G params: {g_params:,}, D params: {d_params:,}")

    adv_criterion = nn.BCEWithLogitsLoss()
    g_optimizer = Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_epoch = 1
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=DEVICE)
        generator.load_state_dict(resume["model_state_dict"])
        if "discriminator_state_dict" in resume:
            discriminator.load_state_dict(resume["discriminator_state_dict"])
        if "g_optimizer_state_dict" in resume:
            g_optimizer.load_state_dict(resume["g_optimizer_state_dict"])
        if "d_optimizer_state_dict" in resume:
            d_optimizer.load_state_dict(resume["d_optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    tag = args.checkpoint_tag.strip()
    ckpt_prefix = f"{exp_name}_{args.variant}{tag}"

    best_val = float("inf")
    history_rows = []

    print(f"\n{'Epoch':>5}  {'TrG':>8}  {'TrD':>8}  {'VaLoss':>8}  {'TrMAE':>8}  {'VaMAE':>8}  {'Time':>6}")
    print("-" * 64)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr = train_epoch(
            generator,
            discriminator,
            train_loader,
            g_optimizer,
            d_optimizer,
            adv_criterion,
            max_batches=args.max_train_batches,
            mask_ratio=args.mask_ratio,
            context_weight=args.context_weight,
            adv_loss_weight=args.adv_loss_weight,
        )
        va = validate_epoch(
            generator,
            val_loader,
            max_batches=args.max_val_batches,
            mask_ratio=args.mask_ratio,
            context_weight=args.context_weight,
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
                    "model_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "val_loss": best_val,
                    "checkpoint_tag": tag,
                    "mask_ratio": args.mask_ratio,
                    "target_length": args.target_length,
                    "model_family": "single_recon_gan_v1",
                },
                os.path.join(SAVE_DIR, f"{ckpt_prefix}_best.pt"),
            )
            marker = " *"

        history_rows.append(
            {
                "epoch": epoch,
                "tr_g_loss": tr["g_loss"],
                "tr_d_loss": tr["d_loss"],
                "va_loss": va["loss"],
                "tr_mae": tr["mae"],
                "va_mae": va["mae"],
                "seconds": elapsed,
            }
        )

        print(
            f"{epoch:5d}  {tr['g_loss']:8.4f}  {tr['d_loss']:8.4f}  {va['loss']:8.4f}  "
            f"{tr['mae']:8.4f}  {va['mae']:8.4f}  {elapsed:5.1f}s{marker}"
        )

    torch.save(
        {
            "signal_type": signal_type,
            "variant": args.variant,
            "epoch": args.epochs,
            "model_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "val_loss": best_val,
            "checkpoint_tag": tag,
            "mask_ratio": args.mask_ratio,
            "target_length": args.target_length,
            "model_family": "single_recon_gan_v1",
        },
        os.path.join(SAVE_DIR, f"{ckpt_prefix}_final.pt"),
    )

    history_csv = os.path.join(plot_dir, f"{ckpt_prefix}_history.csv")
    history_png = os.path.join(plot_dir, f"{ckpt_prefix}_history.png")
    save_history(history_rows, history_csv, history_png, title=f"{exp_name} GAN Training History ({ckpt_prefix})")

    print(f"Saved history: {history_csv}")
    print(f"Saved history plot: {history_png}")
    print(f"\nDone. Best val recon loss: {best_val:.4f}")
