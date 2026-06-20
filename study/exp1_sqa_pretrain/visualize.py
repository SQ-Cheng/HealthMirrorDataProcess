"""Unified visualization script for Exp1-SQAPreTrain.

Quickly visualize masked reconstruction on a few validation samples.

Usage:
    python visualize.py --model tcn --variant tcn256 --signal-type ecg
    python visualize.py --model baseline --variant full --signal-type ecg \\
        --checkpoint checkpoints/exp1_ecg_baseline_full_best.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_sqa_pretrain.dataloader import build_dataloaders
from exp1_sqa_pretrain.models import build_model
from exp1_sqa_pretrain.train import build_visible_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(_PKG_DIR, "checkpoints")
DEFAULT_PLOT_DIR = os.path.join(_PKG_DIR, "plots")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp1-SQAPreTrain: quick visualization"
    )

    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "resnet", "tcn"])
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=["ecg", "rppg"])

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-source", type=str, default="sqi",
                        choices=["sqi", "cleaned"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5,
                        help="Number of samples to visualize.")

    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Auto-generated if not provided.")

    return parser.parse_args()


def find_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    signal_tag = args.signal_type
    model_tag = args.model
    len_tag = f"L{args.target_length}"
    ckpt_prefix = f"exp1_{signal_tag}_{model_tag}_{len_tag}"

    candidates = [
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt"),
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_final.pt"),
    ]

    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt

    raise FileNotFoundError(
        f"No checkpoint found. Tried: {', '.join(candidates)}"
    )


def main():
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    plot_dir = (os.path.dirname(os.path.abspath(args.output))
                if args.output else DEFAULT_PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = find_checkpoint(args)

    # Load model
    model = build_model(args.model, target_length=args.target_length).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load validation data
    _, val_loader = build_dataloaders(
        data_dir,
        signal_type=args.signal_type,
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

    signal, quality_score = next(iter(val_loader))
    n = min(args.num_segments, signal.shape[0])

    signal = signal[:n].to(DEVICE)
    quality_score = quality_score[:n].cpu().numpy()

    visible = build_visible_mask(signal, mask_ratio=args.mask_ratio)
    masked = signal * visible

    with torch.no_grad():
        pred = model(masked, visible)

    signal_np = signal.cpu().numpy()
    masked_np = masked.cpu().numpy()
    pred_np = pred.cpu().numpy()
    masked_region = (1.0 - visible.cpu().numpy())

    n_ch = signal_np.shape[1]
    ch_names = ["ECG", "rPPG"] if n_ch <= 2 else [f"Channel {i}" for i in range(n_ch)]
    colors_target = ["tab:blue", "tab:green"]
    colors_recon = ["tab:orange", "tab:red"]

    t = np.arange(args.target_length) / (args.target_length / args.window_sec)

    # Compute per-channel masked MAE
    mae_strs = []
    for ch in range(n_ch):
        masked_mae = np.mean(np.abs(
            (pred_np[:, ch] - signal_np[:, ch]) * masked_region[:, ch]
        ))
        mae_strs.append(f"{ch_names[ch]} MAE={masked_mae:.4f}")

    fig, axes = plt.subplots(n, n_ch, figsize=(6 * n_ch, 2.5 * n),
                              squeeze=False, sharex=True)

    for i in range(n):
        for ch in range(n_ch):
            ax = axes[i, ch]
            ax.plot(t, signal_np[i, ch], label="Target", color=colors_target[ch % 2],
                    linewidth=1.0)
            ax.plot(t, masked_np[i, ch], label="Masked input", color="tab:gray",
                    linewidth=1.0, alpha=0.85)
            ax.plot(t, pred_np[i, ch], label="Reconstruction", color=colors_recon[ch % 2],
                    linewidth=1.0)
            ax.set_title(f"Sample #{i+1} {ch_names[ch]} | q={quality_score[i]:.3f}")
            ax.grid(alpha=0.15)
            ax.set_ylabel("Norm. amplitude")

    for ch in range(n_ch):
        axes[-1, ch].set_xlabel("Time (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    len_tag = f"L{args.target_length}"
    fig.suptitle(
        f"Exp1 {args.signal_type}/{args.model} {len_tag} | "
        f"mask={args.mask_ratio:.2f} | "
        + ", ".join(mae_strs),
        y=1.01, fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is None:
        out_path = os.path.join(
            plot_dir,
            f"exp1_viz_{args.signal_type}_{args.model}_{len_tag}_mask{args.mask_ratio:.2f}.png"
        )
    else:
        out_path = args.output

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Masked MAE: {', '.join(mae_strs)}")


if __name__ == "__main__":
    main()
