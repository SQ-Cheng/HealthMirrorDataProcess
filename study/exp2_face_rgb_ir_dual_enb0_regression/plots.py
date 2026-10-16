"""PNG training/scatter figures and matched-frame EN-B0 comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import plot_one, panels, LABELS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp2_face_pretrained_head32_regression.train import _regression_metrics
from . import config


def plot_results():
    root = config.OUTPUT / "regression/dual_enb0"
    histories, metrics = [], []
    rows, audits = [], []
    controls = ("RGB main / same frames", "RGB+IR / single 4-channel EN-B0", "RGB+IR / independent EN-B0")
    for target in config.TARGETS:
        run = root / f"runs/{target}"
        history = pd.read_csv(run / "history.csv")
        history["epoch"] = history.global_epoch
        histories.append(history)
        metrics.append(pd.read_csv(run / "metrics.csv"))
        candidate = pd.read_csv(run / "video_predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
        rgb = pd.read_csv(run / "main_rgb_same_frames_predictions.csv", dtype={"hospital_id": str, "video_id": str})
        previous = pd.read_csv(config.paired.OUTPUT / f"runs/efficientnet_b0/{target}/video_predictions.csv",
                               dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
        iqr = pd.read_csv(run / "main_rgb_same_frames_metrics.csv").reference_train_iqr.iloc[0]
        expected = candidate.sort_values("video_id").reset_index(drop=True)
        for label, values in zip(controls, (rgb, previous, candidate)):
            values = values.sort_values("video_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(expected[["hospital_id", "video_id"]], values[["hospital_id", "video_id"]], check_dtype=False)
            np.testing.assert_allclose(expected.y_true, values.y_true, rtol=1e-12, atol=1e-12)
            if not values.frame_count.eq(20).all():
                raise RuntimeError("Comparison dropped test frames")
            result = _regression_metrics(values.y_true, values.y_pred, values.score_threshold,
                                         config.paired.SCORE_DEFINITIONS[target]["direction"])
            rows.append({"target": target, "model": label, **result,
                         "explained_variance": explained_variance_score(values.y_true, values.y_pred),
                         "mae_reference_iqr": result["mae"]/iqr, "rmse_reference_iqr": result["rmse"]/iqr})
        audits.append({"target": target, "test_videos": len(expected), "test_patients": expected.hospital_id.nunique(),
                       "same_test_labels_and_rgb_frames": True, "rgb_main_training_cohort_is_larger": True,
                       "single_and_dual_rgb_ir_training_cohort_identical": True,
                       "dual_finetune_bn_microbatch": config.MICROBATCH})
    pd.concat(histories, ignore_index=True).to_csv(root / "history_all.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(root / "metrics_all.csv", index=False)
    LABELS["dual_enb0"] = "RGB+IR / independent EfficientNet-B0 encoders"
    plot_one(root, "regression", "dual_enb0", targets=config.TARGETS, loss_label="Frame-level SmoothL1")
    table = pd.DataFrame(rows)
    figures, tables = config.OUTPUT / "figures", config.OUTPUT / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    table.to_csv(tables / "matched_test_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(tables / "comparison_audit.csv", index=False)
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"),
                       ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            values = table.loc[table.target.eq(target)].set_index("model").loc[list(controls), key]
            bars = axis.bar(range(3), values, color=("#64737c", "#2878b5", "#278245"))
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
            axis.set_xticks(range(3), ("RGB main", "RGB+IR\nsingle EN-B0", "RGB+IR\ndual EN-B0"))
            axis.set(title=TASK_LABELS[target], ylabel=label+(f" ({TASK_UNITS[target]})" if key in ("mae", "rmse") else ""))
            axis.grid(axis="y", alpha=.2)
            axis.axhline(0, color=".5", linewidth=.6)
            axis.set_axisbelow(True)
            axis.margins(y=.2)
        figure.suptitle(f"Same paired20 test videos and RGB frames | {label}")
        figure.savefig(figures / f"{key}_comparison.png", dpi=180)
        plt.close(figure)
    (config.OUTPUT / "REPORT.md").write_text(
        "# Independent RGB/IR EfficientNet-B0 Regression\n\n"
        "Two independently initialized ImageNet EN-B0 encoders produce 1280 features each. "
        "A concat2560/head32 predicts robust-scaled raw lab values. Both encoders are frozen/eval in stage one "
        "and fully updated in stage two. IR is native gray repeated across three channels; both branches use ImageNet normalization.\n\n"
        "The paired cohort, labels, patient splits, target scalers, twenty frames and five views match the four-channel RGB+IR control. "
        "The optimizer sees twelve distinct lab events/240 frame pairs per update; microbatch60 limits memory. "
        "Finetune BatchNorm statistics update per microbatch, unlike the older single-backbone control. "
        "Both training stages use dynamic Inductor compilation without CUDA Graphs; evaluation is eager. "
        "The RGB-only main checkpoint was trained on a larger RGB cohort, but is re-evaluated on exactly the same paired RGB frames. "
        "These training differences must not be mistaken for a pure architecture-only ablation.\n\n"
        "Histories/scatters: `regression/dual_enb0/figures/`; matched comparisons: `figures/`; machine-readable audit: `tables/`.\n")
    print(f"[plots-complete] {figures}; histories, fitted scatter and five matched comparison PNGs", flush=True)


if __name__ == "__main__":
    plot_results()
