"""Visualize Exp4 reconstruction on clean vs noisy rPPG segments."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp4_dataloader import RPPGWindowDataset
from exp4_model import build_exp4_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Exp4 clean/noisy rPPG reconstruction")
    parser.add_argument("--variant", choices=["light", "full"], default="full")
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def find_checkpoint(variant, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    best_path = os.path.join(CHECKPOINT_DIR, f"exp4_{variant}_best.pt")
    final_path = os.path.join(CHECKPOINT_DIR, f"exp4_{variant}_final.pt")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(final_path):
        return final_path
    raise FileNotFoundError(
        f"No checkpoint found for variant={variant}. "
        f"Expected one of: {best_path}, {final_path}"
    )


def reconstruct_segments(model, dataset, indices):
    raws = []
    recons = []
    snrs = []
    mses = []
    maes = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            x, snr = dataset[idx]
            raw = x.squeeze(0).numpy()

            inp = x.unsqueeze(0).to(DEVICE)
            rec = model(inp).squeeze(0).squeeze(0).cpu().numpy()
            mse = float(np.mean((rec - raw) ** 2))
            mae = float(np.mean(np.abs(rec - raw)))

            raws.append(raw)
            recons.append(rec)
            snrs.append(float(snr.item()))
            mses.append(mse)
            maes.append(mae)

    return raws, recons, snrs, mses, maes


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    rng = np.random.default_rng(args.seed)

    ckpt_path = find_checkpoint(args.variant, args.checkpoint)

    model = build_exp4_model(args.variant).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = RPPGWindowDataset(
        data_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    if len(dataset) == 0:
        raise RuntimeError("Exp4 visualization failed: dataset has no valid windows.")

    snr_all = np.array(dataset.snr_db)
    q25 = np.percentile(snr_all, 25)
    q75 = np.percentile(snr_all, 75)

    clean_pool = np.where(snr_all >= q75)[0]
    noisy_pool = np.where(snr_all <= q25)[0]
    if len(clean_pool) == 0 or len(noisy_pool) == 0:
        raise RuntimeError("Exp4 visualization failed: clean/noisy pools are empty.")

    n = max(1, args.num_segments)
    if len(clean_pool) < n or len(noisy_pool) < n:
        raise RuntimeError(
            "Not enough clean/noisy segments for requested visualization pairs. "
            f"Requested={n}, clean_pool={len(clean_pool)}, noisy_pool={len(noisy_pool)}"
        )

    clean_sel = rng.choice(clean_pool, size=n, replace=False)
    noisy_sel = rng.choice(noisy_pool, size=n, replace=False)

    clean_raw, clean_rec, clean_snr, clean_mse, clean_mae = reconstruct_segments(model, dataset, clean_sel)
    noisy_raw, noisy_rec, noisy_snr, noisy_mse, noisy_mae = reconstruct_segments(model, dataset, noisy_sel)

    rows = max(len(clean_raw), len(noisy_raw))
    t = np.arange(args.target_length) / (args.target_length / args.window_sec)

    fig, axes = plt.subplots(rows, 2, figsize=(14, max(3 * rows, 4)), sharex=True)
    if rows == 1:
        axes = np.array([axes])

    for r in range(rows):
        ax_clean = axes[r, 0]
        ax_noisy = axes[r, 1]

        if r < len(clean_raw):
            ax_clean.plot(t, clean_raw[r], color="tab:blue", linewidth=1.0, label="Raw")
            ax_clean.plot(t, clean_rec[r], color="tab:orange", linewidth=1.0, label="Reconstructed")
            ax_clean.set_title(
                f"Clean #{r+1} | SNR={clean_snr[r]:.2f} dB | "
                f"MSE={clean_mse[r]:.4f} | MAE={clean_mae[r]:.4f}"
            )
        else:
            ax_clean.axis("off")

        if r < len(noisy_raw):
            ax_noisy.plot(t, noisy_raw[r], color="tab:blue", linewidth=1.0, label="Raw")
            ax_noisy.plot(t, noisy_rec[r], color="tab:orange", linewidth=1.0, label="Reconstructed")
            ax_noisy.set_title(
                f"Noisy #{r+1} | SNR={noisy_snr[r]:.2f} dB | "
                f"MSE={noisy_mse[r]:.4f} | MAE={noisy_mae[r]:.4f}"
            )
        else:
            ax_noisy.axis("off")

        for ax in (ax_clean, ax_noisy):
            if ax.has_data():
                ax.grid(alpha=0.2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Normalized amplitude")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    clean_mean = float(np.mean(clean_mse)) if clean_mse else float("nan")
    noisy_mean = float(np.mean(noisy_mse)) if noisy_mse else float("nan")
    clean_mae_mean = float(np.mean(clean_mae)) if clean_mae else float("nan")
    noisy_mae_mean = float(np.mean(noisy_mae)) if noisy_mae else float("nan")
    fig.suptitle(
        "Exp4 Reconstruction: Clean vs Noisy rPPG Segments\n"
        f"Checkpoint={os.path.basename(ckpt_path)} | "
        f"Mean MSE clean={clean_mean:.4f}, noisy={noisy_mean:.4f} | "
        f"Mean MAE clean={clean_mae_mean:.4f}, noisy={noisy_mae_mean:.4f}",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp4", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"exp4_{args.variant}_clean_vs_noisy.png")
    else:
        out_path = args.output
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    fig.savefig(out_path, dpi=180)
    print(f"Saved figure: {out_path}")
    print(f"Mean MSE clean={clean_mean:.6f}, noisy={noisy_mean:.6f}")
    print(f"Mean MAE clean={clean_mae_mean:.6f}, noisy={noisy_mae_mean:.6f}")


if __name__ == "__main__":
    main()
