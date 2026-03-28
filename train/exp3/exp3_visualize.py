"""Visualize Exp3 masked reconstruction on ECG+rPPG segments."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import build_masked_recon_dataloaders
from exp3_train import build_visible_mask
from exp3_model import build_exp3_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Exp3 masked ECG+rPPG reconstruction")
    parser.add_argument("--variant", choices=["light", "full"], default="light")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-block-min", type=int, default=8)
    parser.add_argument("--mask-block-max", type=int, default=32)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def find_checkpoint(variant, explicit_path=None):
    if explicit_path:
        return explicit_path
    best = os.path.join(CHECKPOINT_DIR, f"exp3_{variant}_best.pt")
    final = os.path.join(CHECKPOINT_DIR, f"exp3_{variant}_final.pt")
    if os.path.exists(best):
        return best
    if os.path.exists(final):
        return final
    raise FileNotFoundError(f"No checkpoint found for {variant}. Tried {best} and {final}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    ckpt_path = find_checkpoint(args.variant, args.checkpoint)

    model = build_exp3_model(args.variant).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = build_masked_recon_dataloaders(
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

    batch = next(iter(val_loader))
    pair, clean_score, _, _ = batch
    n = min(args.num_segments, pair.shape[0])

    pair = pair[:n].to(DEVICE)
    clean_score = clean_score[:n].cpu().numpy()

    visible = build_visible_mask(
        pair,
        mask_ratio=args.mask_ratio,
        block_min=args.mask_block_min,
        block_max=args.mask_block_max,
    )
    masked = pair * visible

    with torch.no_grad():
        pred = model(masked, visible)

    pair_np = pair.cpu().numpy()
    masked_np = masked.cpu().numpy()
    pred_np = pred.cpu().numpy()
    masked_region = (1.0 - visible.cpu().numpy())

    ecg_mae = np.mean(np.abs((pred_np[:, 0] - pair_np[:, 0]) * masked_region[:, 0]))
    rppg_mae = np.mean(np.abs((pred_np[:, 1] - pair_np[:, 1]) * masked_region[:, 1]))

    t = np.arange(args.target_length) / (args.target_length / args.window_sec)
    fig, axes = plt.subplots(n, 2, figsize=(14, max(3 * n, 4)), sharex=True)
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        ax0.plot(t, pair_np[i, 0], label="ECG target", color="tab:blue", linewidth=1.0)
        ax0.plot(t, masked_np[i, 0], label="ECG masked input", color="tab:gray", linewidth=1.0, alpha=0.8)
        ax0.plot(t, pred_np[i, 0], label="ECG recon", color="tab:orange", linewidth=1.0)
        ax0.set_title(f"Seg#{i+1} ECG | clean_score={clean_score[i]:.3f}")
        ax0.grid(alpha=0.2)

        ax1.plot(t, pair_np[i, 1], label="rPPG target", color="tab:green", linewidth=1.0)
        ax1.plot(t, masked_np[i, 1], label="rPPG masked input", color="tab:gray", linewidth=1.0, alpha=0.8)
        ax1.plot(t, pred_np[i, 1], label="rPPG recon", color="tab:red", linewidth=1.0)
        ax1.set_title(f"Seg#{i+1} rPPG | clean_score={clean_score[i]:.3f}")
        ax1.grid(alpha=0.2)

        for ax in (ax0, ax1):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized amplitude")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.suptitle(
        "Exp3 Masked Reconstruction (ECG + rPPG)\n"
        f"Checkpoint={os.path.basename(ckpt_path)} | Masked MAE ECG={ecg_mae:.4f}, rPPG={rppg_mae:.4f}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp3", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"exp3_{args.variant}_masked_recon.png")
    else:
        out_path = args.output
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    print(f"Masked MAE ECG={ecg_mae:.6f}, rPPG={rppg_mae:.6f}")


if __name__ == "__main__":
    main()
