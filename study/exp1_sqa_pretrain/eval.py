"""Unified evaluation script for Exp1-SQAPreTrain checkpoints.

Evaluates a trained model on the validation set, computing per-channel metrics
and generating visualization plots.

Usage:
    python eval.py --model tcn --variant tcn256 --signal-type ecg \\
        --checkpoint checkpoints/exp1_ecg_tcn_tcn256_best.pt

    # Evaluate at multiple mask ratios
    python eval.py --model baseline --variant full --signal-type ecg \\
        --mask-ratios 0.10 0.20 0.35
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_sqa_pretrain.dataloader import build_dataloaders
from exp1_sqa_pretrain.models import build_model
from exp1_sqa_pretrain.train import build_visible_mask, masked_recon_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(_PKG_DIR, "checkpoints")
DEFAULT_PLOT_DIR = os.path.join(_PKG_DIR, "plots")


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp1-SQAPreTrain: evaluation and visualization"
    )

    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "tcn"])
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=["ecg", "rppg"])

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Auto-detected if not provided.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to search for checkpoints.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for output plots.")

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-source", type=str, default="sqi",
                        choices=["sqi", "cleaned"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.30],
                        help="Mask ratios to evaluate at.")
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-vis-samples", type=int, default=5,
                        help="Number of samples to visualize.")

    return parser.parse_args()


# ──────────────────────────────────────────────
# Checkpoint discovery
# ──────────────────────────────────────────────

def find_checkpoint(args):
    """Find checkpoint path from arguments or auto-discovery."""
    if args.checkpoint:
        return args.checkpoint

    checkpoint_dir = (os.path.abspath(args.checkpoint_dir)
                      if args.checkpoint_dir else DEFAULT_CHECKPOINT_DIR)

    signal_tag = args.signal_type
    model_tag = args.model
    len_tag = f"L{args.target_length}"
    ckpt_prefix = f"exp1_{signal_tag}_{model_tag}_{len_tag}"

    candidates = [
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt"),
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_final.pt"),
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_curriculum_best.pt"),
        os.path.join(checkpoint_dir, f"{ckpt_prefix}_curriculum_final.pt"),
    ]

    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt

    raise FileNotFoundError(
        f"No checkpoint found for {model_tag}/{len_tag}/{signal_tag}. "
        f"Tried: {', '.join(candidates)}"
    )


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def channel_metrics(pred, target, masked):
    """Compute MAE, RMSE, Pearson correlation for masked region."""
    p = pred[masked]
    t = target[masked]
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = safe_corr(t, p)
    return mae, rmse, corr


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_at_mask_ratio(model, loader, mask_ratio, context_weight, max_batches=None):
    """Evaluate model at a specific mask ratio."""
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="none")

    losses = []
    all_pred = []
    all_target = []
    all_masked_mask = []

    for batch_idx, (signal, quality_score) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        signal = signal.to(DEVICE)
        quality_score = quality_score.to(DEVICE)

        visible = build_visible_mask(signal, mask_ratio=mask_ratio)
        masked = 1.0 - visible
        x_masked = signal * visible

        pred = model(x_masked, visible)
        loss = masked_recon_loss(
            pred, signal, visible, quality_score, criterion,
            context_weight=context_weight,
        )

        losses.append(loss.item())
        all_pred.append(pred.cpu().numpy())
        all_target.append(signal.cpu().numpy())
        all_masked_mask.append(masked.cpu().numpy())

    pred_cat = np.concatenate(all_pred, axis=0)
    target_cat = np.concatenate(all_target, axis=0)
    masked_cat = np.concatenate(all_masked_mask, axis=0) > 0.5

    avg_loss = float(np.mean(losses))

    # Per-channel metrics
    n_ch = target_cat.shape[1]
    metrics = {}
    for ch in range(n_ch):
        mae, rmse, corr = channel_metrics(
            pred_cat[:, ch], target_cat[:, ch], masked_cat[:, ch]
        )
        ch_name = ["ECG", "rPPG"][ch] if n_ch <= 2 else f"ch{ch}"
        metrics[ch_name] = {"mae": mae, "rmse": rmse, "corr": corr}

    return {
        "loss": avg_loss,
        "metrics": metrics,
        "predictions": pred_cat,
        "targets": target_cat,
        "masked": masked_cat,
    }


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_reconstruction(results, mask_ratio, args, save_path):
    """Plot reconstruction examples at a given mask ratio."""
    pred = results["predictions"]
    target = results["targets"]
    masked = results["masked"]

    n_vis = min(args.num_vis_samples, pred.shape[0])
    n_ch = target.shape[1]
    t = np.arange(args.target_length) / (args.target_length / args.window_sec)

    fig, axes = plt.subplots(n_vis, n_ch, figsize=(6 * n_ch, 2.5 * n_vis),
                              squeeze=False, sharex=True)

    ch_names = ["ECG", "rPPG"] if n_ch <= 2 else [f"Channel {i}" for i in range(n_ch)]
    colors_target = ["tab:blue", "tab:green"]
    colors_recon = ["tab:orange", "tab:red"]

    for i in range(n_vis):
        # Create masked input for visualization
        visible = 1.0 - masked[i]
        masked_input = target[i] * visible

        for ch in range(n_ch):
            ax = axes[i, ch]
            ax.plot(t, target[i, ch], label="Target", color=colors_target[ch % 2],
                    linewidth=1.0, alpha=0.7)
            ax.plot(t, masked_input[ch], label="Masked input", color="tab:gray",
                    linewidth=1.0, alpha=0.85)
            ax.plot(t, pred[i, ch], label="Reconstruction", color=colors_recon[ch % 2],
                    linewidth=1.0)
            ax.set_title(f"Sample #{i+1} {ch_names[ch]} | mask={mask_ratio:.2f}")
            ax.grid(alpha=0.15)
            ax.set_ylabel("Norm. amplitude")

    for ch in range(n_ch):
        axes[-1, ch].set_xlabel("Time (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)

    metrics_str = ", ".join(
        f"{name} MAE={m['mae']:.4f}" for name, m in results["metrics"].items()
    )
    fig.suptitle(
        f"Exp1 {args.signal_type}/{args.model} L{args.target_length} | "
        f"mask={mask_ratio:.2f} | {metrics_str}",
        y=1.01, fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    plot_dir = os.path.abspath(args.plot_dir) if args.plot_dir else DEFAULT_PLOT_DIR
    os.makedirs(plot_dir, exist_ok=True)

    variant_tag = args.variant if args.variant else "default"
    signal_tag = args.signal_type
    model_tag = args.model

    ckpt_path = find_checkpoint(args)
    print(f"Using checkpoint: {ckpt_path}")

    # Load model
    model = build_model(args.model, target_length=args.target_length).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # Handle legacy checkpoints that may have mismatched target_length
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load data
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

    # Evaluate at each mask ratio
    all_results = {}
    len_tag = f"L{args.target_length}"

    print(f"\n{'='*70}")
    print(f"Evaluation: {model_tag} on {signal_tag} ({len_tag})")
    print(f"{'='*70}")

    for mr in args.mask_ratios:
        print(f"\n--- mask_ratio={mr:.2f} ---")
        results = evaluate_at_mask_ratio(
            model, val_loader, mr, args.context_weight,
        )
        all_results[mr] = results

        print(f"  Loss: {results['loss']:.6f}")
        for name, m in results["metrics"].items():
            print(f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, Corr={m['corr']:.4f}")

        # Plot
        plot_path = os.path.join(
            plot_dir,
            f"exp1_eval_{signal_tag}_{model_tag}_{len_tag}_mask{mr:.2f}.png"
        )
        plot_reconstruction(results, mr, args, plot_path)
        print(f"  Plot saved: {plot_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary (masked MAE)")
    header = f"{'mask_ratio':>10}"
    for mr in args.mask_ratios:
        header += f"  | {mr:.2f}"
    print(header)
    print("-" * len(header))

    for ch_name in all_results[args.mask_ratios[0]]["metrics"].keys():
        row = f"{ch_name:>10}"
        for mr in args.mask_ratios:
            row += f"  | {all_results[mr]['metrics'][ch_name]['mae']:.4f}"
        print(row)

    # Save metrics JSON
    metrics_json = {}
    for mr, res in all_results.items():
        metrics_json[f"mask_{mr:.2f}"] = {
            "loss": res["loss"],
            "metrics": {k: v for k, v in res["metrics"].items()},
        }

    json_path = os.path.join(
        plot_dir,
        f"exp1_eval_{signal_tag}_{model_tag}_{len_tag}_metrics.json"
    )
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved: {json_path}")


if __name__ == "__main__":
    main()
