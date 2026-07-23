"""Plot per-backbone/task training curves for the pretrained face experiments."""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_history(input_csv, output_png, title):
    history = pd.read_csv(input_csv)
    if history.empty:
        raise ValueError(f"No history rows found in {input_csv}")

    architecture_order = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    architecture_rank = {name: index for index, name in enumerate(architecture_order)}
    target_order = list(history["target"].drop_duplicates())
    target_rank = {name: index for index, name in enumerate(target_order)}
    jobs = sorted(
        history[["architecture", "target"]].drop_duplicates().itertuples(index=False),
        key=lambda row: (
            architecture_rank.get(row.architecture, 999),
            target_rank.get(row.target, 999),
        ),
    )

    columns = min(3, len(jobs))
    rows = math.ceil(len(jobs) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 4.5 * rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for index, (architecture, target) in enumerate(jobs):
        axis = axes[index]
        task_history = history[
            history["architecture"].eq(architecture)
            & history["target"].eq(target)
        ].sort_values("global_epoch")
        epochs = task_history["global_epoch"].to_numpy()

        axis.plot(epochs, task_history["train_loss"], "b-", label="Train loss", alpha=0.75)
        axis.plot(epochs, task_history["val_loss"], "r-", label="Val loss", alpha=0.75)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.set_title(f"{architecture} / {target}", fontsize=9)

        second_axis = axis.twinx()
        second_axis.plot(epochs, task_history["val_bacc"], "g-", label="Val bACC")
        second_axis.plot(epochs, task_history["val_roc_auc"], "m-", label="Val ROC-AUC")
        second_axis.set_ylim(-0.05, 1.05)
        second_axis.set_ylabel("Score")

        lines_a, labels_a = axis.get_legend_handles_labels()
        lines_b, labels_b = second_axis.get_legend_handles_labels()
        axis.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="best")

    for axis in axes[len(jobs) :]:
        axis.set_visible(False)

    figure.suptitle(title, fontsize=14, y=1.01)
    figure.tight_layout()
    figure.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_png}: {len(jobs)} subplots")


if __name__ == "__main__":
    experiments = [
        (
            "exp2_face_pretrained",
            "Exp2 Pretrained RGB: Backbone/Task Models",
        ),
        (
            "exp2_face_pretrained_head32",
            "Exp2 Pretrained RGB Head32: Backbone/Task Models",
        ),
    ]
    for experiment, title in experiments:
        output_dir = os.path.join(ROOT, "study", experiment, "outputs")
        plot_history(
            os.path.join(output_dir, "history_all.csv"),
            os.path.join(output_dir, "loss_curves.png"),
            title,
        )
