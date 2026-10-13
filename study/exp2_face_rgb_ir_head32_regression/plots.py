"""Matched-test and matched-frame RGB main comparisons; PNG output only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_pretrained_head32_regression.train import _regression_metrics
from .config import OUTPUT, REFERENCE, SCORE_DEFINITIONS


LABELS = {"oxyhemoglobin_fraction": "Oxyhemoglobin fraction", "lactate_high": "Lactate",
          "urea_high": "Urea", "total_bilirubin_high": "Total bilirubin", "platelet_count_low": "Platelets",
          "hemoglobin_low": "Hemoglobin", "aa_po2_ratio_low": "A/a PO2 ratio", "creatinine_high": "Creatinine",
          "hematocrit_low": "Hematocrit", "egfr_low": "eGFR"}
COLORS = {"RGB main, original frames": "#919da4", "RGB main, joint frames": "#397fa4", "RGB+IR, joint frames": "#247c65"}


def plot_results(output, targets):
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True); tables.mkdir(exist_ok=True)
    predictions, rows, audits, histories = {}, [], [], []
    for target in targets:
        run = output / f"runs/efficientnet_b0/{target}"
        new = pd.read_csv(run / "video_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        new = new.loc[new.split.eq("test")].copy()
        main = pd.read_csv(REFERENCE / f"runs/efficientnet_b0/{target}/video_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        main = main.loc[main.split.eq("test")].copy()
        same = pd.read_csv(run / "main_rgb_same_frames_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        cohorts = {"RGB+IR, joint frames": new, "RGB main, joint frames": same,
                   "RGB main, original frames": main.loc[main.video_id.isin(new.video_id)]}
        reference_iqr = pd.read_csv(run / "main_rgb_same_frames_metrics.csv").reference_train_iqr.iloc[0]
        for name, frame in cohorts.items():
            compared = new.merge(frame, on=["hospital_id", "video_id"], suffixes=("_new", "_comparison"), validate="one_to_one")
            if len(compared) != len(new) or len(frame) != len(new) or not np.allclose(compared.y_true_new, compared.y_true_comparison, rtol=0, atol=1e-9):
                raise ValueError(f"Comparison labels or test membership differ: {target}/{name}")
            metrics = _regression_metrics(frame.y_true, frame.y_pred, frame.score_threshold, SCORE_DEFINITIONS[target]["direction"])
            rows.append({"target": target, "model": name, **metrics,
                         "explained_variance": explained_variance_score(frame.y_true, frame.y_pred),
                         "mae_reference_iqr": metrics["mae"]/reference_iqr, "rmse_reference_iqr": metrics["rmse"]/reference_iqr})
        audits.append({"target": target, "main_full_test_videos": len(main), "common_test_videos": len(new),
                       "common_test_patients": new.hospital_id.nunique(), "same_labels": True,
                       "same_joint_rgb_frames_verified": True, "training_cohorts_identical": False})
        predictions[target] = new
        histories.append(pd.read_csv(run / "history.csv"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(tables / "matched_test_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(tables / "comparison_cohorts.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "training_history_all.csv", index=False)
    all_metrics = pd.concat([pd.read_csv(output / f"runs/efficientnet_b0/{target}/metrics.csv") for target in targets], ignore_index=True)
    all_metrics.to_csv(output / "metrics_all.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    panels = (("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("r2", "R squared"),
              ("explained_variance", "Explained variance"), ("mae_reference_iqr", "MAE / main train IQR"),
              ("rmse_reference_iqr", "RMSE / main train IQR"))
    y = np.arange(len(targets))
    for ax, (column, title) in zip(axes.flat, panels):
        for i, (model, color) in enumerate(COLORS.items()):
            values = metrics.loc[metrics.model.eq(model)].set_index("target").reindex(targets)[column]
            ax.barh(y+(i-1)*.24, values, height=.22, color=color, label=model)
        ax.set_yticks(y, [LABELS[t] for t in targets]); ax.invert_yaxis()
        ax.set_title(title); ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
        ax.axvline(0, color=".5", linewidth=.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(figures / "matched_rgb_vs_rgb_ir.png", dpi=180); plt.close(fig)
    columns = 5 if len(targets) == 10 else min(4, len(targets))
    nrows = int(np.ceil(len(targets)/columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(4.1*columns, 3.9*nrows), squeeze=False)
    for ax, target in zip(axes.flat, targets):
        frame = predictions[target]
        ax.scatter(frame.y_true, frame.y_pred, s=16, alpha=.55, color="#247c65", edgecolors="none")
        low, high = min(frame.y_true.min(), frame.y_pred.min()), max(frame.y_true.max(), frame.y_pred.max())
        ax.plot([low, high], [low, high], color=".5", linestyle="--", linewidth=1)
        if len(frame)>1 and frame.y_true.std()>0:
            x = np.linspace(frame.y_true.min(), frame.y_true.max(), 100)
            ax.plot(x, np.polyval(np.polyfit(frame.y_true, frame.y_pred, 1), x), color="#a34e44", linewidth=1.2)
        r = frame.y_true.corr(frame.y_pred)
        unit = SCORE_DEFINITIONS[target]["unit"]
        ax.set(title=f"{LABELS[target]}\nn={len(frame)}, r={r:.2f}", xlabel=f"True ({unit})", ylabel=f"Predicted ({unit})")
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(targets):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(figures / "test_predicted_vs_true.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(nrows, columns, figsize=(4.1*columns, 3.8*nrows), squeeze=False)
    for ax, target, history in zip(axes.flat, targets, histories):
        ax.plot(history.global_epoch, history.train_loss, color="#397fa4", label="Train loss")
        ax.plot(history.global_epoch, history.val_loss, color="#b76646", label="Validation loss")
        right = ax.twinx()
        right.plot(history.global_epoch, history.train_pearson_r, color="#397fa4", linestyle="--", alpha=.7, label="Train r")
        right.plot(history.global_epoch, history.val_pearson_r, color="#247c65", linestyle="--", label="Validation r")
        r = history[["train_pearson_r", "val_pearson_r"]].to_numpy().ravel(); r = r[np.isfinite(r)]
        if len(r):
            pad = max(.05, .1*np.ptp(r)); right.set_ylim(max(-1.02, r.min()-pad), min(1.02, r.max()+pad))
        ax.set(title=LABELS[target], xlabel="Epoch", ylabel="SmoothL1 loss"); right.set_ylabel("Pearson r")
        if history.stage.eq("finetune").any():
            ax.axvline(history.loc[history.stage.eq("finetune"), "global_epoch"].min()-.5, color=".5", linestyle=":")
        if target == targets[0]:
            h, l = ax.get_legend_handles_labels(); h2, l2 = right.get_legend_handles_labels()
            ax.legend(h+h2, l+l2, fontsize=7)
    for ax in list(axes.flat)[len(targets):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(figures / "training_history.png", dpi=180); plt.close(fig)
    lines = ["# Native224 RGB+IR Regression", "",
             "All comparisons use common test videos and identical raw labels. The existing RGB checkpoint is additionally evaluated on the exact same joint-selected RGB frames, without retraining. Main RGB and new RGB+IR training cohorts differ; the charts are not a same-training-cohort architecture-only ablation.", "",
             "Stage one freezes the zero IR stem and trains head32, so IR cannot influence predictions until full-backbone finetuning. RGB uses unchanged ImageNet normalization; IR uses fixed mean/std .5/.5, with synchronized five-view transformations.", "",
             "|Analyte|Model|n|MAE|RMSE|r|R2|Explained variance|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        lines.append(f"|{LABELS[row.target]}|{row.model}|{row.n}|{row.mae:.4f}|{row.rmse:.4f}|{row.pearson_r:.3f}|{row.r2:.3f}|{row.explained_variance:.3f}|")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    targets = tuple(json.loads((args.output_dir / "protocol.json").read_text())["targets"])
    plot_results(args.output_dir, targets)
