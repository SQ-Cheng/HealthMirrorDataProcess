"""Video-level Exp8 figures and comparison with the saved RGB control."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    BASE_OUTPUT, CACHE, FEATURE_SHAPE, GRID_SIZE, HERE, OUTPUT, TARGETS,
    TARGET_LABELS, VARIANT, WAVELENGTH_NM,
)
from .extract_features import load_index


COLORS = ("#1C7C80", "#B75C43", "#6A72A6")


def plot_results():
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    histories = pd.read_csv(OUTPUT / "history_all.csv")
    metrics = pd.read_csv(OUTPUT / "metrics_all.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for axis, target in zip(axes, TARGETS):
        values = histories.loc[histories.target.eq(target)]
        axis.plot(values.epoch, values.train_loss, label="Training", color=COLORS[0])
        axis.plot(values.epoch, values.val_loss, label="Validation", color=COLORS[1])
        axis.set_title(TARGET_LABELS[target][0])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Smooth L1 loss (scaled value)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "training_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for axis, target, color in zip(axes, TARGETS, COLORS):
        values = pd.read_csv(OUTPUT / "runs" / target / "predictions.csv")
        values = values.loc[values.split.eq("test")]
        actual = values.raw_value.to_numpy(np.float64)
        predicted = values.predicted_raw_value.to_numpy(np.float64)
        axis.scatter(actual, predicted, s=15, alpha=0.52, color=color,
                     edgecolors="none")
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        axis.plot([low, high], [low, high], color="#777777", linestyle="--",
                  linewidth=1, label="Identity")
        if np.ptp(actual) > 0:
            slope, intercept = np.polyfit(actual, predicted, 1)
            axis.plot([low, high], np.asarray([low, high]) * slope + intercept,
                      color="#242424", linewidth=1.4, label="Linear fit")
        score = metrics.loc[metrics.target.eq(target) & metrics.split.eq("test")].iloc[0]
        axis.set_title(f"{TARGET_LABELS[target][0]}  |  R²={score.r2:.2f}, n={int(score.n)}")
        axis.set_xlabel(f"Measured ({TARGET_LABELS[target][1]})")
        axis.set_ylabel(f"Predicted ({TARGET_LABELS[target][1]})")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "test_predicted_vs_true.png", dpi=180)
    plt.close(fig)

    reference = pd.read_csv(BASE_OUTPUT / "metrics_all.csv")
    reference = reference.loc[
        reference.architecture.eq("efficientnet_b0")
        & reference.split.eq("test")
        & reference.target.isin(TARGETS)
    ].set_index("target")
    spectral = metrics.loc[metrics.split.eq("test")].set_index("target")
    if (set(reference.index) != set(TARGETS)
            or not np.array_equal(reference.loc[list(TARGETS), "n"].to_numpy(),
                                  spectral.loc[list(TARGETS), "n"].to_numpy())):
        raise RuntimeError("RGB control and Exp8 do not share target/test counts")
    x = np.arange(len(TARGETS))
    names = [TARGET_LABELS[target][0] for target in TARGETS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, key, ylabel in zip(axes, ("r2", "mae"), ("Test R²", "Test MAE")):
        axis.bar(x - 0.18, reference.loc[list(TARGETS), key], width=0.35,
                 color=COLORS[0], label="RGB EfficientNet-B0")
        axis.bar(x + 0.18, spectral.loc[list(TARGETS), key], width=0.35,
                 color=COLORS[1], label="Estimated-spectrum MLP")
        axis.set_xticks(x, names, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "rgb_control_comparison.png", dpi=180)
    plt.close(fig)

    index = load_index()
    features = np.load(
        CACHE / f"mstpp_31band_grid{GRID_SIZE}x{GRID_SIZE}.npy", mmap_mode="r"
    )
    if features.shape[1:] != FEATURE_SHAPE:
        raise RuntimeError("Unexpected spectral feature shape")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for axis, target in zip(axes, TARGETS):
        records = pd.read_csv(BASE_OUTPUT / "task_records" / f"{target}.csv")
        test = records.loc[records.split.eq("test")].sort_values("raw_value")
        selected = test.iloc[[0, len(test) // 2, len(test) - 1]]
        for label, row, color in zip(("Low", "Median", "High"),
                                     selected.itertuples(index=False), COLORS):
            start, end = index.frame_range(row.video_id)
            curve = np.asarray(features[start:end], dtype=np.float32).mean(axis=(0, 2, 3))
            axis.plot(WAVELENGTH_NM, curve,
                      label=f"{label}: {row.raw_value:g} {TARGET_LABELS[target][1]}",
                      color=color)
        axis.set_title(TARGET_LABELS[target][0])
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Estimated normalized response")
        axis.legend(frameon=False, fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "estimated_spectrum_examples.png", dpi=180)
    plt.close(fig)
    if OUTPUT != HERE / "outputs":
        _plot_variant_comparison(figures, metrics)
    print(f"[exp8-figures] {figures}", flush=True)


def _plot_variant_comparison(figures, new_metrics):
    old_output = HERE / "outputs"
    old_metrics = pd.read_csv(old_output / "metrics_all.csv")
    for target in TARGETS:
        paths = [output / "runs" / target / "predictions.csv"
                 for output in (old_output, OUTPUT)]
        old, new = [pd.read_csv(path, dtype={"video_id": str, "hospital_id": str})
                    for path in paths]
        old = old.loc[old.split.eq("test"), ["hospital_id", "video_id", "raw_value"]]
        new = new.loc[new.split.eq("test"), ["hospital_id", "video_id", "raw_value"]]
        paired = old.merge(new, on=["hospital_id", "video_id"], how="outer",
                           validate="one_to_one", indicator=True,
                           suffixes=("_reference", "_new"))
        if (not paired._merge.eq("both").all()
                or not np.allclose(paired.raw_value_reference,
                                   paired.raw_value_new, atol=1e-6)):
            raise RuntimeError(f"Spectral variants have different test labels: {target}")

    old = old_metrics.loc[old_metrics.split.eq("test")].set_index("target")
    new = new_metrics.loc[new_metrics.split.eq("test")].set_index("target")
    if (set(old.index) != set(TARGETS) or set(new.index) != set(TARGETS)
            or not np.array_equal(old.loc[list(TARGETS), "n"].to_numpy(),
                                  new.loc[list(TARGETS), "n"].to_numpy())):
        raise RuntimeError("Spectral variants have different test counts")
    comparison = old.loc[list(TARGETS), ["n", "mae", "r2", "pearson_r"]].join(
        new.loc[list(TARGETS), ["mae", "r2", "pearson_r"]],
        lsuffix="_grid4_ntire", rsuffix="_new",
    )
    comparison["mae_change_pct"] = 100 * (
        comparison.mae_new / comparison.mae_grid4_ntire - 1
    )
    comparison.to_csv(OUTPUT / "reference_comparison.csv")
    x = np.arange(len(TARGETS))
    names = [TARGET_LABELS[target][0] for target in TARGETS]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    source = "Hyper-Skin" if VARIANT == "hyperskin" else "NTIRE 2022"
    new_label = f"{source} MST++ {GRID_SIZE}×{GRID_SIZE}"
    for axis, key, label in zip(axes[:2], ("r2", "pearson_r"),
                                ("Test R²", "Test Pearson r")):
        axis.bar(x - 0.18, old.loc[list(TARGETS), key], width=0.35,
                 color=COLORS[0], label="NTIRE 2022 MST++ 4×4")
        axis.bar(x + 0.18, new.loc[list(TARGETS), key], width=0.35,
                 color=COLORS[1], label=new_label)
        axis.set_xticks(x, names, rotation=18, ha="right")
        axis.set_ylabel(label)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[2].bar(x, comparison.mae_change_pct, width=0.55,
                color=[COLORS[0] if value < 0 else COLORS[1]
                       for value in comparison.mae_change_pct])
    axes[2].set_xticks(x, names, rotation=18, ha="right")
    axes[2].set_ylabel("Test MAE change vs 4×4 (%)")
    axes[2].axhline(0, color="#777777", linewidth=0.8)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    filename = ("feature_grid_comparison.png" if GRID_SIZE == 16 and VARIANT == "ntire2022"
                else "spectral_source_comparison.png")
    fig.savefig(figures / filename, dpi=180)
    plt.close(fig)
