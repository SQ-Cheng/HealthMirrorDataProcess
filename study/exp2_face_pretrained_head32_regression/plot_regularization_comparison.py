"""Plot the 30/40 control against two independent regularization changes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS
from .plot_patient_diverse_comparison import _test_metrics, _verify_same_test_videos
from .plot_results import TASK_LABELS


def plot_comparison(reference_dir, ablation_dir, variants):
    reference_dir, ablation_dir = Path(reference_dir), Path(ablation_dir)
    names = ("30/40 control", "Weight decay 1e-2", "Fixed BN statistics")
    paths = [reference_dir] + [ablation_dir / item[0] for item in variants]
    for path in paths[1:]:
        if not (path / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete ablation: {path}")
        _verify_same_test_videos(reference_dir, path)
    metrics = [_test_metrics(path) for path in paths]
    if any(not metrics[0].n.equals(frame.n) for frame in metrics[1:]):
        raise AssertionError("Test video counts differ between ablations")

    output = ablation_dir / "patient_diverse_regularization_comparison"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        row = {"target": target, "test_videos": int(metrics[0].loc[target, "n"])}
        for label, frame in zip(("control", "wd1e2", "bn_fixed"), metrics):
            for column in ("mae", "pearson_r", "r2"):
                row[f"{label}_{column}"] = float(frame.loc[target, column])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "test_metrics_comparison.csv", index=False)

    x = np.arange(len(TARGETS))
    labels = [TASK_LABELS[target] for target in TARGETS]
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for axis, column, title in (
        (axes[0], "mae", "Test MAE change from control (%)"),
        (axes[1], "pearson_r", "Test Pearson r change from control"),
        (axes[2], "r2", "Test R2 change from control"),
    ):
        for offset, (variant, color, label) in enumerate((
            ("wd1e2", "#D05B3E", names[1]),
            ("bn_fixed", "#278245", names[2]),
        )):
            values = table[f"{variant}_{column}"] - table[f"control_{column}"]
            if column == "mae":
                values = 100 * values / table["control_mae"]
            axis.bar(x + (offset - 0.5) * 0.36, values, width=0.36,
                     color=color, label=label)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    figure.suptitle("Independent regularization ablations | identical held-out videos")
    figure.tight_layout()
    figure.savefig(figures / "test_effects.png", dpi=180)
    plt.close(figure)

    histories = [pd.read_csv(path / "history_all.csv") for path in paths]
    grid_rows, grid_columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(
        grid_rows, grid_columns,
        figsize=target_grid_figsize(grid_rows, grid_columns), squeeze=False,
    )
    for axis, target in zip(axes.flat, TARGETS):
        for history, label, color in zip(
            histories, names, ("#2878B5", "#D05B3E", "#278245")
        ):
            selected = history.loc[history.target.eq(target)].sort_values("global_epoch")
            if selected.empty:
                raise RuntimeError(f"Missing history for {target}: {label}")
            axis.plot(selected.global_epoch, selected.val_mae, label=label, color=color)
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MAE")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Validation history | same patient-diverse 20-frame data")
    figure.tight_layout()
    figure.savefig(figures / "validation_history.png", dpi=180)
    plt.close(figure)
    (output / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[regularization-comparison-complete] output={output}", flush=True)


if __name__ == "__main__":
    from .run_ablations import ABLATION_DIR
    from .run_patient_diverse_regularization_ablations import REFERENCE, VARIANTS

    plot_comparison(REFERENCE, ABLATION_DIR, VARIANTS)
