"""SSL diagnostics and explicitly split-aware downstream reference comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def plot_all():
    figures = config.OUTPUT / "figures"; figures.mkdir(exist_ok=True)
    history = pd.read_csv(config.OUTPUT / "ssl/history.csv")
    actual_epochs = int(history.epoch.max())
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column, title, color in (
        (axes[0], "loss", "BYOL loss", "#2878B5"),
        (axes[1], "projection_std", "Normalized projection standard deviation", "#CB6547"),
        (axes[2], "ema", "Teacher EMA momentum", "#278245"),
    ):
        axis.plot(history.epoch, history[column], marker="o", color=color)
        axis.set(xlabel="SSL epoch", ylabel=title); axis.grid(alpha=.2)
    figure.suptitle(f"BYOL pretraining | {actual_epochs} completed epochs")
    figure.tight_layout(); figure.savefig(figures / "ssl_training_history.png", dpi=180); plt.close(figure)
    from study.exp2_face_shared_backbone_head32_regression.plot_comparison import plot_comparison
    plot_comparison(config.BASE, config.OUTPUT, names=("ImageNet initialization", "BYOL initialization"),
                    experiment_label=f"BYOL-pretrained independent regression models (SSL {actual_epochs} epochs)",
                    figure_prefix="byol_vs_imagenet")
