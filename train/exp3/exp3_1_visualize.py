"""Exp3-1: visualize ECG SQI by autocorrelation (good vs bad pairs)."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import MaskedReconDataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3-1 visualize ECG autocorr SQI")
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
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def autocorr_profile(ecg, fs):
    """Return normalized ACF and the HR-relevant peak in 0.33-1.50 s lag range."""
    x = ecg - np.mean(ecg)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(x) - 1 :]

    if len(acf) == 0 or acf[0] <= 1e-12:
        return acf, 1, 1, 1, 0.0

    acf = acf / acf[0]
    lag_lo = max(1, int(fs * 0.33))
    lag_hi = min(len(acf) - 1, int(fs * 1.50))

    if lag_hi <= lag_lo:
        return acf, lag_lo, lag_hi, lag_lo, 0.0

    peak_lag = lag_lo + int(np.argmax(acf[lag_lo : lag_hi + 1]))
    peak_val = float(acf[peak_lag])
    return acf, lag_lo, lag_hi, peak_lag, peak_val


def maybe_subsample(indices, max_samples, rng):
    if max_samples is None or max_samples >= len(indices):
        return indices
    return rng.choice(indices, size=max_samples, replace=False)


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    rng = np.random.default_rng(args.seed)

    dataset = MaskedReconDataset(
        data_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    candidate_indices = np.arange(len(dataset))
    candidate_indices = maybe_subsample(candidate_indices, args.max_samples, rng)

    sqi = np.array([dataset.ecg_autocorr_sqi[i] for i in candidate_indices], dtype=np.float64)
    order = np.argsort(sqi)

    n_pairs = min(max(1, args.num_pairs), len(order) // 2)
    worst_local = order[:n_pairs]
    best_local = order[-n_pairs:][::-1]

    worst_idx = candidate_indices[worst_local]
    best_idx = candidate_indices[best_local]

    fs_target = args.target_length / args.window_sec
    t = np.arange(args.target_length) / fs_target

    fig, axes = plt.subplots(n_pairs, 4, figsize=(20, max(3.2 * n_pairs, 6)))
    if n_pairs == 1:
        axes = np.array([axes])

    headers = [
        "Good ECG segment",
        "Good ECG autocorr",
        "Bad ECG segment",
        "Bad ECG autocorr",
    ]
    for col, title in enumerate(headers):
        axes[0, col].set_title(title)

    for r in range(n_pairs):
        g_idx = int(best_idx[r])
        b_idx = int(worst_idx[r])

        ecg_good = dataset.samples[g_idx][0]
        ecg_bad = dataset.samples[b_idx][0]

        g_acf, g_lo, g_hi, g_peak_lag, g_peak_val = autocorr_profile(ecg_good, fs_target)
        b_acf, b_lo, b_hi, b_peak_lag, b_peak_val = autocorr_profile(ecg_bad, fs_target)

        ax_sig_g = axes[r, 0]
        ax_acf_g = axes[r, 1]
        ax_sig_b = axes[r, 2]
        ax_acf_b = axes[r, 3]

        ax_sig_g.plot(t, ecg_good, color="tab:blue", linewidth=1.0)
        ax_sig_g.set_ylabel(f"Pair {r + 1}")
        ax_sig_g.set_xlabel("Time (s)")
        ax_sig_g.grid(alpha=0.2)
        ax_sig_g.text(
            0.02,
            0.92,
            f"idx={g_idx}\\nSQI={dataset.ecg_autocorr_sqi[g_idx]:.3f}",
            transform=ax_sig_g.transAxes,
            fontsize=9,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        lag_time_g = np.arange(len(g_acf)) / fs_target
        ax_acf_g.plot(lag_time_g, g_acf, color="tab:green", linewidth=1.0)
        ax_acf_g.axvspan(g_lo / fs_target, g_hi / fs_target, color="tab:orange", alpha=0.15)
        ax_acf_g.scatter([g_peak_lag / fs_target], [g_peak_val], color="tab:red", s=20)
        ax_acf_g.set_ylim(-0.2, 1.05)
        ax_acf_g.set_xlabel("Lag (s)")
        ax_acf_g.grid(alpha=0.2)

        ax_sig_b.plot(t, ecg_bad, color="tab:red", linewidth=1.0)
        ax_sig_b.set_xlabel("Time (s)")
        ax_sig_b.grid(alpha=0.2)
        ax_sig_b.text(
            0.02,
            0.92,
            f"idx={b_idx}\\nSQI={dataset.ecg_autocorr_sqi[b_idx]:.3f}",
            transform=ax_sig_b.transAxes,
            fontsize=9,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        lag_time_b = np.arange(len(b_acf)) / fs_target
        ax_acf_b.plot(lag_time_b, b_acf, color="tab:purple", linewidth=1.0)
        ax_acf_b.axvspan(b_lo / fs_target, b_hi / fs_target, color="tab:orange", alpha=0.15)
        ax_acf_b.scatter([b_peak_lag / fs_target], [b_peak_val], color="tab:red", s=20)
        ax_acf_b.set_ylim(-0.2, 1.05)
        ax_acf_b.set_xlabel("Lag (s)")
        ax_acf_b.grid(alpha=0.2)

    fig.suptitle(
        "Exp3-1 ECG SQI by Autocorr: 5 Good/Bad Pairs\n"
        "SQI = max normalized autocorr in lag window 0.33-1.50 s",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp3", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "exp3_1_autocorr_sqi_good_bad.png")
    else:
        out_path = args.output
        out_parent = os.path.dirname(out_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    fig.savefig(out_path, dpi=180)

    print(f"Saved figure: {out_path}")
    print(
        "Autocorr SQI summary: "
        f"min={np.min(sqi):.4f}, max={np.max(sqi):.4f}, mean={np.mean(sqi):.4f}, std={np.std(sqi):.4f}"
    )
    print("Selected good indices:", ",".join(str(int(x)) for x in best_idx))
    print("Selected bad indices:", ",".join(str(int(x)) for x in worst_idx))


if __name__ == "__main__":
    main()
