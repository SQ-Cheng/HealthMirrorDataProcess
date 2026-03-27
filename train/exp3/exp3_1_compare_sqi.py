"""Exp3-1: Compare multiple ECG SQI approaches by visualization."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import MaskedReconDataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3-1 ECG SQI approach comparison")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    lo = np.min(x)
    hi = np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dataset = MaskedReconDataset(
        ROOT_DIR,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    n_total = len(dataset)
    k = min(args.max_samples, n_total)
    indices = np.arange(n_total)
    if k < n_total:
        indices = rng.choice(indices, size=k, replace=False)

    template = np.array([dataset.ecg_template_sqi[i] for i in indices])
    autocorr = np.array([dataset.ecg_autocorr_sqi[i] for i in indices])
    morph = np.array([dataset.ecg_morph_sqi[i] for i in indices])
    artifact = np.array([dataset.ecg_artifact_sqi[i] for i in indices])
    composite = np.array([dataset.ecg_quality[i] for i in indices])
    legacy_snr = np.array([dataset.ecg_legacy_freq_snr[i] for i in indices])
    legacy_snr_norm = _normalize01(legacy_snr)
    rppg_snr = np.array([dataset.rppg_snr_db[i] for i in indices])

    method_names = [
        "template_corr",
        "autocorr",
        "morph_stability",
        "artifact_penalty",
        "composite_nonfreq",
        "legacy_freq_snr",
        "rppg_snr",
    ]
    method_values = [
        template,
        autocorr,
        morph,
        artifact,
        composite,
        legacy_snr_norm,
        _normalize01(rppg_snr),
    ]

    corr = np.zeros((len(method_names), len(method_names)), dtype=np.float64)
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            corr[i, j] = safe_corr(method_values[i], method_values[j])

    if args.output_dir is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp3", "plots")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: correlation matrix + method histograms
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0])

    ax_mat = fig.add_subplot(gs[0, :2])
    im = ax_mat.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax_mat.set_xticks(np.arange(len(method_names)))
    ax_mat.set_yticks(np.arange(len(method_names)))
    ax_mat.set_xticklabels(method_names, rotation=30, ha="right")
    ax_mat.set_yticklabels(method_names)
    ax_mat.set_title("Exp3-1 SQI Method Correlation Matrix")
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            ax_mat.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

    ax_sc = fig.add_subplot(gs[0, 2:])
    ax_sc.scatter(composite, legacy_snr_norm, s=10, alpha=0.5, label="composite vs legacy")
    ax_sc.scatter(composite, _normalize01(rppg_snr), s=10, alpha=0.5, label="composite vs rPPG SNR")
    ax_sc.set_title("Composite Non-Freq SQI Relation")
    ax_sc.set_xlabel("composite_nonfreq")
    ax_sc.set_ylabel("normalized reference")
    ax_sc.grid(alpha=0.2)
    ax_sc.legend()

    hist_specs = [
        (template, "template_corr"),
        (autocorr, "autocorr"),
        (morph, "morph_stability"),
        (artifact, "artifact_penalty"),
    ]
    for idx, (vals, name) in enumerate(hist_specs):
        ax = fig.add_subplot(gs[1, idx])
        ax.hist(vals, bins=30, alpha=0.8)
        ax.set_title(name)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    path_matrix = os.path.join(out_dir, "exp3_1_sqi_compare_matrix.png")
    fig.savefig(path_matrix, dpi=180)

    # Figure 2: best/worst examples per method
    example_methods = [
        ("template_corr", template),
        ("autocorr", autocorr),
        ("morph_stability", morph),
        ("artifact_penalty", artifact),
        ("composite_nonfreq", composite),
    ]

    fig2, axes = plt.subplots(len(example_methods), 2, figsize=(14, 3 * len(example_methods)), sharex=True)
    if len(example_methods) == 1:
        axes = np.array([axes])

    t = np.arange(args.target_length) / (args.target_length / args.window_sec)

    for r, (name, vals) in enumerate(example_methods):
        i_best = int(np.argmax(vals))
        i_worst = int(np.argmin(vals))
        idx_best = int(indices[i_best])
        idx_worst = int(indices[i_worst])

        ecg_best = dataset.samples[idx_best][0]
        ecg_worst = dataset.samples[idx_worst][0]

        ax0 = axes[r, 0]
        ax1 = axes[r, 1]

        ax0.plot(t, ecg_best, color="tab:blue", linewidth=1.0)
        ax1.plot(t, ecg_worst, color="tab:red", linewidth=1.0)

        ax0.set_title(f"{name} best | score={vals[i_best]:.3f}")
        ax1.set_title(f"{name} worst | score={vals[i_worst]:.3f}")

        for ax in (ax0, ax1):
            ax.grid(alpha=0.2)
            ax.set_ylabel("ECG amplitude")
            ax.set_xlabel("Time (s)")

    fig2.suptitle("Exp3-1 ECG SQI Methods: Best vs Worst Segments", y=0.995)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    path_examples = os.path.join(out_dir, "exp3_1_sqi_compare_examples.png")
    fig2.savefig(path_examples, dpi=180)

    print(f"Saved figure: {path_matrix}")
    print(f"Saved figure: {path_examples}")
    print(
        "Composite correlation: "
        f"vs legacy_freq_snr={safe_corr(composite, legacy_snr_norm):.4f}, "
        f"vs rppg_snr={safe_corr(composite, _normalize01(rppg_snr)):.4f}"
    )


if __name__ == "__main__":
    main()
