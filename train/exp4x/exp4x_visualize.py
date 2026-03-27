"""Experiment 04-X visualization for SQI regression quality."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp4x_dataloader import build_exp4x_dataloaders
from exp4x_model import build_exp4x_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Exp4-X SQI prediction")
    parser.add_argument("--model", choices=["exp4-1", "exp4-2", "exp4-3"], default="exp4-1")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def find_checkpoint(model_name, explicit_path=None):
    if explicit_path:
        return explicit_path

    best = os.path.join(CHECKPOINT_DIR, f"exp4x_{model_name}_best.pt")
    final = os.path.join(CHECKPOINT_DIR, f"exp4x_{model_name}_final.pt")
    if os.path.exists(best):
        return best
    if os.path.exists(final):
        return final
    raise FileNotFoundError(
        f"No checkpoint found for {model_name}. Expected one of: {best}, {final}"
    )


def safe_corr(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stacked_segments(ax, dataset, indices, title, window_sec):
    if len(indices) == 0:
        ax.axis("off")
        return

    target_length = len(dataset.windows[indices[0]])
    t = np.arange(target_length) / (target_length / window_sec)

    offset = 3.0
    for i, idx in enumerate(indices):
        seg = dataset.windows[idx]
        sqi = dataset.sqi[idx]
        snr = dataset.snr_db[idx]
        ax.plot(t, seg + i * offset, linewidth=1.0, label=f"#{i+1} SQI={sqi:.2f}, SNR={snr:.1f}")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")


def main():
    args = parse_args()

    ckpt_path = find_checkpoint(args.model, args.checkpoint)

    model = build_exp4x_model(args.model).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader, meta = build_exp4x_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
        return_meta=True,
    )

    dataset = meta["dataset"]
    val_indices = list(meta["val_indices"])

    preds = []
    trues = []
    snrs = []

    with torch.no_grad():
        for x, sqi, snr in val_loader:
            x = x.to(DEVICE)
            pred = model(x).detach().cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(sqi.numpy().tolist())
            snrs.extend(snr.numpy().tolist())

    preds_np = np.array(preds)
    trues_np = np.array(trues)
    snrs_np = np.array(snrs)

    mae = float(np.mean(np.abs(preds_np - trues_np))) if len(preds_np) else float("nan")
    pear = safe_corr(preds_np, trues_np)
    snr_corr = safe_corr(preds_np, snrs_np)

    val_true_by_index = np.array([dataset.sqi[idx] for idx in val_indices])
    order = np.argsort(val_true_by_index)
    k = max(1, min(args.num_segments, len(order)))
    worst_indices = [val_indices[i] for i in order[:k]]
    best_indices = [val_indices[i] for i in order[-k:]]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])

    ax_scatter = fig.add_subplot(gs[0, :])
    sc = ax_scatter.scatter(trues_np, preds_np, c=snrs_np, cmap="viridis", alpha=0.7, s=16)
    cbar = fig.colorbar(sc, ax=ax_scatter)
    cbar.set_label("SNR (dB)")
    ax_scatter.set_title("Exp4-X: Predicted SQI vs SNR-ranked Target SQI (Validation)")
    ax_scatter.set_xlabel("Target SQI (rank-normalized SNR)")
    ax_scatter.set_ylabel("Predicted SQI")
    ax_scatter.grid(alpha=0.2)

    low, high = 0.0, 1.0
    ax_scatter.plot([low, high], [low, high], "r--", linewidth=1.0, label="Ideal")
    ax_scatter.legend(loc="lower right")

    ax_best = fig.add_subplot(gs[1, :])
    stacked_segments(ax_best, dataset, best_indices, "Best-quality segments (highest target SQI)", args.window_sec)

    ax_worst = fig.add_subplot(gs[2, :])
    stacked_segments(ax_worst, dataset, worst_indices, "Worst-quality segments (lowest target SQI)", args.window_sec)

    fig.suptitle(
        "Exp4-X SQI Regression Visualization\n"
        f"Model={args.model} | Checkpoint={os.path.basename(ckpt_path)} | "
        f"MAE={mae:.4f}, Pearson={pear:.4f}, Corr(Pred,SNR)={snr_corr:.4f}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp4x", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"exp4x_{args.model}_viz.png")
    else:
        out_path = args.output
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    print(f"Val MAE={mae:.6f}, Pearson={pear:.6f}, Corr(Pred,SNR)={snr_corr:.6f}")


if __name__ == "__main__":
    main()
