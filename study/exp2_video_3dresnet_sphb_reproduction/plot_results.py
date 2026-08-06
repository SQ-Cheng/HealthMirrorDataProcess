"""Paper-aligned regression figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(output_dir):
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(output / "history.csv")
    predictions = pd.read_csv(output / "video_predictions.csv")
    test = predictions.loc[predictions["split"].eq("test")]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.epoch, history.train_mse_g_dl2, label="Train MSE")
    axes[0].plot(history.epoch, history.val_mse_g_dl2, label="Validation MSE")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (g/dL)^2"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(history.epoch, history.val_rmse_g_dl, label="Validation RMSE")
    axes[1].plot(history.epoch, history.val_mae_g_dl, label="Validation MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Error (g/dL)"); axes[1].legend(); axes[1].grid(alpha=.2)
    figure.suptitle("Residual 3D CNN training history")
    figure.tight_layout(); figure.savefig(figures / "training_history.png", dpi=180); plt.close(figure)

    truth, prediction = test.y_true_g_dl.to_numpy(), test.y_pred_g_dl.to_numpy()
    difference, mean = prediction - truth, (prediction + truth) / 2
    bias, sd = difference.mean(), difference.std(ddof=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(truth, prediction, alpha=.7, s=24)
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    axes[0].plot(limits, limits, "--", color="black"); axes[0].set_xlabel("True Hb (g/dL)"); axes[0].set_ylabel("Predicted Hb (g/dL)")
    axes[1].scatter(mean, difference, alpha=.7, s=24)
    axes[1].axhline(bias, color="black"); axes[1].axhline(bias + 1.96 * sd, color="red", linestyle="--"); axes[1].axhline(bias - 1.96 * sd, color="red", linestyle="--")
    axes[1].set_xlabel("Mean Hb (g/dL)"); axes[1].set_ylabel("Predicted - true (g/dL)"); axes[1].set_title(f"Bland-Altman: bias={bias:.3f}")
    figure.suptitle("Video-level held-out test results")
    figure.tight_layout(); figure.savefig(figures / "test_regression_and_bland_altman.png", dpi=180); plt.close(figure)
    print(f"Saved 2 figures to {figures}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.output_dir)
