"""Shared visualization for single-signal masked reconstruction (Exp3 ECG/rPPG TCN split)."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from single_recon_dataloader import build_single_signal_dataloaders
from single_recon_model_tcn import build_single_recon_tcn_model
from single_recon_train_tcn import build_single_window_visible_mask


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args(exp_name):
    parser = argparse.ArgumentParser(description=f"Visualize {exp_name} masked reconstruction")
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
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def list_candidate_checkpoints(exp_name, variant, explicit_path=None):
    if explicit_path:
        return [explicit_path]

    best = os.path.join(CHECKPOINT_DIR, f"{exp_name}_{variant}_best.pt")
    final = os.path.join(CHECKPOINT_DIR, f"{exp_name}_{variant}_final.pt")
    candidates = []

    if os.path.exists(best):
        candidates.append(best)
    if os.path.exists(final):
        candidates.append(final)

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {exp_name}/{variant}. Tried {best} and {final}")
    return candidates


def load_first_compatible_checkpoint(model, checkpoint_candidates):
    errors = []
    for ckpt_path in checkpoint_candidates:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            return ckpt_path, ckpt
        except RuntimeError as exc:
            errors.append(f"{os.path.basename(ckpt_path)} -> {exc}")

    msg = "No compatible checkpoint found for current model definition.\n"
    msg += "Tried:\n  - " + "\n  - ".join(checkpoint_candidates)
    if errors:
        msg += "\nLoad errors:\n  - " + "\n  - ".join(errors)
    raise RuntimeError(msg)


def run_visualization(signal_type, exp_name):
    args = parse_args(exp_name)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_candidates = list_candidate_checkpoints(exp_name, args.variant, args.checkpoint)

    model = build_single_recon_tcn_model(args.variant).to(DEVICE)
    ckpt_path, _ = load_first_compatible_checkpoint(model, ckpt_candidates)
    model.eval()

    _, val_loader = build_single_signal_dataloaders(
        ROOT_DIR,
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

    signal, quality_score = next(iter(val_loader))
    n = min(args.num_segments, signal.shape[0])

    signal = signal[:n].to(DEVICE)
    quality_score = quality_score[:n].cpu().numpy()

    visible = build_single_window_visible_mask(signal, mask_ratio=args.mask_ratio)
    masked = signal * visible

    with torch.no_grad():
        pred = model(masked, visible)

    signal_np = signal.cpu().numpy()[:, 0]
    masked_np = masked.cpu().numpy()[:, 0]
    pred_np = pred.cpu().numpy()[:, 0]
    masked_region = (1.0 - visible.cpu().numpy())[:, 0]

    masked_mae = np.mean(np.abs((pred_np - signal_np) * masked_region))

    t = np.arange(args.target_length) / (args.target_length / args.window_sec)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.6 * n, 4)), sharex=True)
    if n == 1:
        axes = np.array([axes])

    channel_label = "ECG" if signal_type == "ecg" else "rPPG"
    target_color = "tab:blue" if signal_type == "ecg" else "tab:green"
    recon_color = "tab:orange" if signal_type == "ecg" else "tab:red"

    for i in range(n):
        ax = axes[i]
        ax.plot(t, signal_np[i], label=f"{channel_label} target", color=target_color, linewidth=1.0)
        ax.plot(t, masked_np[i], label=f"{channel_label} masked input", color="tab:gray", linewidth=1.0, alpha=0.85)
        ax.plot(t, pred_np[i], label=f"{channel_label} recon", color=recon_color, linewidth=1.0)
        ax.set_title(f"Seg#{i+1} {channel_label} | quality_score={quality_score[i]:.3f}")
        ax.grid(alpha=0.2)
        ax.set_ylabel("Norm. amplitude")

    axes[-1].set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.suptitle(
        f"{exp_name} TCN Masked Reconstruction ({channel_label})\\n"
        f"Checkpoint={os.path.basename(ckpt_path)} | Masked MAE={masked_mae:.4f}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is None:
        out_dir = os.path.join(ROOT_DIR, "train", exp_name, "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{exp_name}_{args.variant}_masked_recon.png")
    else:
        out_path = args.output
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    print(f"Masked MAE={masked_mae:.6f}")
