"""Per-head figures and six-architecture comparisons on identical test videos."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import plot_one, panels, LABELS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from . import config


def plot_results():
    tables, audit = {}, []
    group_labels = {"dino": "RGB+IR DINOv3-B", "dino_farl": "RGB+IR DINOv3-B + FaRL"}
    method_labels = {"concat": "Concat", "vector_gate": "Vector gate", "scalar_gate": "Scalar gate"}
    for variant in config.VARIANTS:
        group = "dino_farl" if variant.startswith("dino_farl_") else "dino"
        LABELS[variant] = group_labels[group] + " / " + method_labels[config.method_for(variant)]
        root = config.OUTPUT / f"regression/{variant}"
        plot_one(root, "regression", variant, targets=config.TARGETS, loss_label="Frame-level SmoothL1")
        tables[variant] = pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").copy()
    for target in config.TARGETS:
        reference = None
        for variant in config.VARIANTS:
            frame = pd.read_csv(config.OUTPUT / f"regression/{variant}/runs/{target}/video_predictions.csv",
                dtype={"hospital_id": str, "video_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True)
            if reference is None:
                reference = frame
            pd.testing.assert_frame_equal(reference[["hospital_id", "video_id", "y_true"]],
                                          frame[["hospital_id", "video_id", "y_true"]], check_dtype=False)
            if not frame.frame_count.eq(20).all():
                raise RuntimeError("Comparison requires exactly twenty paired test frames")
            tables[variant].loc[target, "explained_variance"] = explained_variance_score(frame.y_true, frame.y_pred)
        audit.append({"target": target, "test_videos": len(reference), "test_patients": reference.hospital_id.nunique(),
                      "identical_six_model_cohorts_labels_and_selected_frame_pairs": True})
    pd.DataFrame(audit).to_csv(config.OUTPUT / "paired_test_audit.csv", index=False)
    pd.concat([table.reset_index().assign(model=variant) for variant, table in tables.items()], ignore_index=True).to_csv(
        config.OUTPUT / "test_comparison.csv", index=False)
    figures = config.OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            for group, offset, color in (("dino", -.18, "#2878b5"), ("dino_farl", .18, "#cb6547")):
                values = [tables[f"{group}_{method}"].loc[target, metric] for method in config.METHODS]
                bars = axis.bar([i+offset for i in range(3)], values, width=.34, color=color, label=group_labels[group])
                axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=6)
            axis.set_xticks(range(3), [method_labels[method] for method in config.METHODS], rotation=20, ha="right")
            axis.set(title=TASK_LABELS[target], ylabel=label+(f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.axhline(0, color=".5", linewidth=.6)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            axis.margins(y=.2)
        axes.flat[0].legend(fontsize=7, frameon=False)
        figure.suptitle(f"Frozen RGB/IR encoders | same 24h / paired20 / five-view cohort | {label}")
        figure.savefig(figures / f"{metric}_comparison.png", dpi=180)
        plt.close(figure)
    print(f"[plots-complete] {figures}; six training histories/scatters and five paired metric comparisons (PNG only)", flush=True)


if __name__ == "__main__":
    plot_results()
    from .compare_rgb import main as compare_rgb
    compare_rgb()
