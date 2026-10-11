"""PNG-only, matched-test comparisons and residual diagnostics."""

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
from study.exp2_face_pretrained_head32_regression.config import SCORE_DEFINITIONS
from .data import HERE, REFERENCE, TASKS


LABELS = {"oxyhemoglobin_fraction": "Oxyhemoglobin fraction", "lactate_high": "Lactate",
          "urea_high": "Urea", "total_bilirubin_high": "Total bilirubin", "platelet_count_low": "Platelets",
          "hemoglobin_low": "Hemoglobin", "aa_po2_ratio_low": "A/a PO2 ratio", "creatinine_high": "Creatinine",
          "hematocrit_low": "Hematocrit", "egfr_low": "eGFR"}
COLORS = {"History baseline": "#3676a4", "History + face residual": "#21866b", "Face only": "#bd6544"}


def plot_results(output, targets):
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "axes.grid": True, "grid.alpha": .15})
    predictions, rows, comparison_audit = {}, [], []
    scaler_info = json.loads((output / "target_scalers.json").read_text())
    for target in targets:
        path = output / f"runs/{target}/face_residual"
        all_pred = pd.read_csv(path / "video_predictions.csv", dtype={"hospital_id": str})
        pred = all_pred.loc[all_pred.split.eq("test")].copy()
        if not len(pred) or not np.isfinite(pred[["y_true", "y_pred", "history_pred_raw"]]).all().all():
            raise ValueError(f"Invalid test predictions: {target}")
        predictions[target] = pred
        baseline_path = REFERENCE / f"runs/efficientnet_b0/{target}/video_predictions.csv"
        paired = pred
        model_columns = {"History baseline": "history_pred_raw", "History + face residual": "y_pred"}
        if baseline_path.exists():
            face = pd.read_csv(baseline_path, dtype={"hospital_id": str})
            face = face.loc[face.split.eq("test"), ["hospital_id", "video_id", "y_true", "y_pred"]]
            paired = pred.merge(face, on=["hospital_id", "video_id"], suffixes=("", "_face"), how="inner", validate="one_to_one")
            if len(paired) != len(pred) or not np.allclose(paired.y_true, paired.y_true_face, rtol=0, atol=1e-10):
                raise ValueError(f"Face-only test labels or split are not comparable: {target}")
            model_columns["Face only"] = "y_pred_face"
        comparison_audit.append({"target": target, "test_videos": len(pred), "test_patients": pred.hospital_id.nunique(),
                                 "face_only_available": "Face only" in model_columns, "same_test_labels": True})
        for model, column in model_columns.items():
            metrics = _regression_metrics(paired.y_true, paired[column], paired.score_threshold, SCORE_DEFINITIONS[target]["direction"])
            rows.append({"target": target, "model": model, **metrics,
                         "explained_variance": explained_variance_score(paired.y_true, paired[column]),
                         "mae_train_iqr": metrics["mae"] / scaler_info[target]["iqr"],
                         "rmse_train_iqr": metrics["rmse"] / scaler_info[target]["iqr"]})
        all_pred.to_csv(tables / f"{target}_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(tables / "matched_test_metrics.csv", index=False)
    pd.DataFrame(comparison_audit).to_csv(tables / "comparison_cohort_audit.csv", index=False)
    columns = 5 if len(targets) == 10 else min(4, len(targets))
    nrows = int(np.ceil(len(targets)/columns))
    for residual in (False, True):
        fig, axes = plt.subplots(nrows, columns, figsize=(4.1*columns, 3.9*nrows), squeeze=False)
        for ax, target in zip(axes.flat, targets):
            data = predictions[target]
            x = data.true_residual_raw if residual else data.y_true
            y = data.predicted_residual_raw if residual else data.y_pred
            ax.scatter(x, y, s=16, alpha=.55, color=COLORS["History + face residual"], edgecolors="none")
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([low, high], [low, high], color=".5", linestyle="--", linewidth=1)
            if len(x) > 1 and x.std() > 0:
                line_x = np.linspace(x.min(), x.max(), 100)
                ax.plot(line_x, np.polyval(np.polyfit(x, y, 1), line_x), color="#9c4343", linewidth=1.3)
            r = x.corr(y)
            unit = SCORE_DEFINITIONS[target]["unit"]
            ax.set(title=f"{LABELS[target]}\nn={len(x)}, r={r:.2f}",
                   xlabel=f"True {'residual' if residual else 'value'} ({unit})",
                   ylabel=f"Predicted {'residual' if residual else 'value'} ({unit})")
        for ax in list(axes.flat)[len(targets):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / ("test_residual_predicted_vs_true.png" if residual else "test_predicted_vs_true.png"))
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    names = (("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("r2", "R squared"),
             ("explained_variance", "Explained variance"), ("mae_train_iqr", "MAE / training IQR"),
             ("rmse_train_iqr", "RMSE / training IQR"))
    models = [model for model in COLORS if model in set(metrics.model)]
    positions = np.arange(len(targets))
    for ax, (column, title) in zip(axes.flat, names):
        for i, model in enumerate(models):
            values = metrics.loc[metrics.model.eq(model)].set_index("target").reindex(targets)[column]
            ax.barh(positions+(i-(len(models)-1)/2)*.24, values, height=.22, color=COLORS[model], label=model)
        ax.set_yticks(positions, [LABELS[t] for t in targets])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.axvline(0, color=".4", linewidth=.6)
    axes.flat[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "test_metrics_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(nrows, columns, figsize=(4.1*columns, 3.8*nrows), squeeze=False)
    for ax, target in zip(axes.flat, targets):
        h = pd.read_csv(output / f"runs/{target}/face_residual/history.csv")
        x = h.global_epoch
        ax.plot(x, h.train_loss, color="#3676a4", label="Train loss")
        ax.plot(x, h.val_loss, color="#bd6544", label="Validation loss")
        right = ax.twinx()
        right.plot(x, h.train_pearson_r, color="#3676a4", linestyle="--", alpha=.65, label="Train r")
        right.plot(x, h.val_pearson_r, color="#21866b", linestyle="--", label="Validation r")
        r = h[["train_pearson_r", "val_pearson_r"]].to_numpy().ravel()
        r = r[np.isfinite(r)]
        if len(r):
            pad = max(.05, .1*np.ptp(r))
            right.set_ylim(max(-1.02, r.min()-pad), min(1.02, r.max()+pad))
        right.set_ylabel("Pearson r")
        right.grid(False)
        if h.stage.eq("finetune").any():
            ax.axvline(h.loc[h.stage.eq("finetune"), "global_epoch"].min()-.5, color=".5", linestyle=":")
        ax.set(title=LABELS[target], xlabel="Epoch", ylabel="SmoothL1 loss")
        if target == targets[0]:
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = right.get_legend_handles_labels()
            ax.legend(handles+handles2, labels+labels2, fontsize=7)
    for ax in list(axes.flat)[len(targets):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "training_history.png")
    plt.close(fig)
    lines = ["# Sequential History + Face Residual Regression", "",
             "All test comparisons use identical video IDs and raw labels. No PDF files are generated.", "",
             "Training baselines are patient-OOF predictions. Validation/test baselines use the frozen full history model fitted on outer train, selected on outer validation. Final deployment uses that full history model plus the face residual model.", "",
             "History is defined relative to the current lab timestamp, not video acquisition time; this is not a prospective real-time forecast.", "",
             "Historical eight-task face+history results used an older lab table and are deliberately not treated as a matched control.", "",
             "|Analyte|Model|n|MAE|RMSE|r|R2|Explained variance|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        lines.append(f"|{LABELS[row.target]}|{row.model}|{row.n}|{row.mae:.4f}|{row.rmse:.4f}|{row.pearson_r:.3f}|{row.r2:.3f}|{row.explained_variance:.3f}|")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    targets = json.loads((args.output_dir / "protocol.json").read_text())["targets"]
    plot_results(args.output_dir, tuple(targets))
