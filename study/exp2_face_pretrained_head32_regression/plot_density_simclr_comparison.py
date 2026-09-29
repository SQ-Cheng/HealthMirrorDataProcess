"""Compare two independent ablations against the same completed 30/40 control."""

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


def plot_comparison(control_dir, density_dir, simclr_dir):
    paths = [Path(path) for path in (control_dir, density_dir, simclr_dir)]
    for path in paths:
        if not (path / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete comparison input: {path}")
    for path in paths[1:]:
        _verify_same_test_videos(paths[0], path)
    metrics = [_test_metrics(path) for path in paths]
    if any(not metrics[0].n.equals(frame.n) for frame in metrics[1:]):
        raise AssertionError("Ablations do not share held-out video counts")
    output = paths[0].parent / "patient_diverse_density_simclr_comparison"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = ("30/40 control", "Inverse-density loss", "Label-aware SimCLR")
    colors = ("#177E89", "#C9793F", "#66729C")
    rows = []
    for target in TARGETS:
        row = {"target": target, "test_videos": int(metrics[0].loc[target, "n"])}
        for prefix, frame in zip(("control", "density", "simclr"), metrics):
            for name in ("r2", "mae", "pearson_r"):
                row[f"{prefix}_{name}"] = float(frame.loc[target, name])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "test_metrics_comparison.csv", index=False)
    x = np.arange(len(TARGETS))
    display = [TASK_LABELS[target] for target in TARGETS]
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.0), sharex=True)
    for index, (prefix, label, color) in enumerate(zip(
        ("control", "density", "simclr"), labels, colors
    )):
        offset = (index - 1) * 0.25
        axes[0].bar(x + offset, table[f"{prefix}_r2"], width=0.23,
                    color=color, label=label)
        if index:
            change = 100 * (
                table[f"{prefix}_mae"] / table["control_mae"] - 1
            )
            axes[1].bar(x + (index - 1.5) * 0.30, change,
                        width=0.28, color=color, label=label)
    for axis in axes:
        axis.axhline(0, color="#636B70", linewidth=0.8)
        axis.grid(axis="y", color="#E3E6E8", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(r"Held-out video-level $R^2$")
    axes[1].set_ylabel("Test MAE change from control (%)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xticks(x, display, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(figures / "test_comparison.png", dpi=180)
    plt.close(fig)

    histories = [pd.read_csv(path / "history_all.csv") for path in paths]
    nrows, ncols = target_grid_shape(len(TARGETS))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=target_grid_figsize(nrows, ncols), squeeze=False
    )
    for axis, target in zip(axes.flat, TARGETS):
        for history, label, color in zip(histories, labels, colors):
            selection = history.loc[history.target.eq(target)].sort_values(
                "global_epoch"
            )
            axis.plot(selection.global_epoch, selection.val_mae,
                      color=color, linewidth=1.3, label=label)
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Downstream epoch")
        axis.set_ylabel("Validation MAE")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "validation_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=target_grid_figsize(nrows, ncols), squeeze=False
    )
    for axis, target in zip(axes.flat, TARGETS):
        path = paths[2] / "runs" / "efficientnet_b0" / target / "pretrain_history.csv"
        history = pd.read_csv(path)
        axis.plot(history.epoch, history.train_loss, color=colors[2], marker="o")
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Contrastive epoch")
        axis.set_ylabel("Training NT-Xent loss")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "simclr_pretraining_history.png", dpi=180)
    plt.close(fig)
    (output / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[density-simclr-comparison-complete] output={output}", flush=True)
