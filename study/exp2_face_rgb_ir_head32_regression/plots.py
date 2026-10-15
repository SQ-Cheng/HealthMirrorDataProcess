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
    joint = json.loads((output / "protocol.json").read_text()).get("training_protocol") == "joint_ir_head"
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
    if joint:
        lines[4] = ("One joint stage trains only 288 independent IR filters and the head32. All pretrained RGB backbone parameters and BatchNorm buffers remain bitwise unchanged; backbone stochastic depth is disabled, head dropout remains enabled. IR/head learning rates start at 1e-4/5e-5, with cosine floor 1e-6, maximum 60 epochs and patience 12. Data, target scaling, five views and frame loss are unchanged.")
    for row in metrics.itertuples(index=False):
        lines.append(f"|{LABELS[row.target]}|{row.model}|{row.n}|{row.mae:.4f}|{row.rmse:.4f}|{row.pearson_r:.3f}|{row.r2:.3f}|{row.explained_variance:.3f}|")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")


def plot_initialization_comparison(output, targets, reference_output):
    protocol = json.loads((output / "protocol.json").read_text())
    joint = protocol.get("training_protocol") == "joint_ir_head"
    reference = json.loads((reference_output / "protocol.json").read_text())
    for key in ("lab_table_sha256", "paired_table_sha256", "normalization", "views", "frames"):
        if protocol[key] != reference[key]:
            raise ValueError(f"Initialization comparison protocol changed: {key}")
    rows, audits = [], []
    colors = {"RGB main": "#397fa4", "RGB+IR / ImageNet init": "#247c65", "RGB+IR / RGB-trained init": "#b57932"}
    warm_output = reference_output / "rgb_main_init"
    if joint:
        if not (warm_output / "COMPLETED.json").exists():
            raise RuntimeError("RGB-trained full-finetuning comparison must be complete")
        warm_protocol = json.loads((warm_output / "protocol.json").read_text())
        for key in ("lab_table_sha256", "paired_table_sha256", "normalization", "views", "frames"):
            if warm_protocol[key] != protocol[key]:
                raise ValueError(f"Warm-start comparison protocol changed: {key}")
        colors["RGB+IR / frozen RGB, train IR+head"] = "#b35c65"
    for target in targets:
        run = output / f"runs/efficientnet_b0/{target}"
        new = pd.read_csv(run / "video_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        new = new.loc[new.split.eq("test")].copy()
        previous = pd.read_csv(reference_output / f"runs/efficientnet_b0/{target}/video_predictions.csv",
                               dtype={"hospital_id": str}, float_precision="round_trip")
        previous = previous.loc[previous.split.eq("test") & previous.video_id.isin(new.video_id)].copy()
        rgb = pd.read_csv(run / "main_rgb_same_frames_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        iqr = pd.read_csv(run / "main_rgb_same_frames_metrics.csv").reference_train_iqr.iloc[0]
        cohorts = [("RGB main", rgb), ("RGB+IR / ImageNet init", previous), ("RGB+IR / RGB-trained init", new)]
        if joint:
            warm = pd.read_csv(warm_output / f"runs/efficientnet_b0/{target}/video_predictions.csv",
                               dtype={"hospital_id": str}, float_precision="round_trip")
            warm = warm.loc[warm.split.eq("test") & warm.video_id.isin(new.video_id)].copy()
            if warm_protocol["initialization_checkpoints"][target] != protocol["initialization_checkpoints"][target]:
                raise ValueError("Frozen/full comparison must start from the identical RGB checkpoint and scaling")
            if (warm_output / f"task_records/{target}.csv").read_bytes() != (reference_output / f"task_records/{target}.csv").read_bytes():
                raise ValueError("Warm-start reference uses different paired data")
            cohorts[-1] = ("RGB+IR / RGB-trained init", warm)
            cohorts.append(("RGB+IR / frozen RGB, train IR+head", new))
        for name, frame in cohorts:
            compared = new.merge(frame, on=["hospital_id", "video_id"], suffixes=("_new", "_other"), validate="one_to_one")
            if len(frame) != len(new) or len(compared) != len(new) or not np.allclose(compared.y_true_new, compared.y_true_other, rtol=0, atol=1e-9):
                raise ValueError(f"Initialization comparison labels/cohort mismatch: {target}/{name}")
            metrics = _regression_metrics(frame.y_true, frame.y_pred, frame.score_threshold, SCORE_DEFINITIONS[target]["direction"])
            rows.append({"target": target, "model": name, **metrics,
                "explained_variance": explained_variance_score(frame.y_true, frame.y_pred),
                "mae_reference_iqr": metrics["mae"]/iqr, "rmse_reference_iqr": metrics["rmse"]/iqr})
        paired_records = pd.read_csv(reference_output / f"task_records/{target}.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        new_records = pd.read_csv(output / f"task_records/{target}.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        matched = new_records.merge(paired_records, on="video_id", suffixes=("_new", "_previous"), validate="one_to_one")
        for column in ("hospital_id", "split", "raw_value", "clinical_event_id", "robust_scaled_raw_value"):
            if len(matched) != len(new_records) or not np.array_equal(matched[column+"_new"], matched[column+"_previous"]):
                raise ValueError(f"Initialization comparison data changed: {target}/{column}")
        if not protocol["smoke_test"] and len(new_records) != len(paired_records):
            raise ValueError("Full initialization comparison requires identical training data")
        audits.append({"target": target, "test_videos": len(new), "test_patients": new.hospital_id.nunique(),
            "same_labels_splits_frames_scaler": True, "same_rgb_ir_training_cohort": not protocol["smoke_test"],
            "rgb_pretraining_cohort_is_larger": True,
            "rgb_initialization_checkpoint_sha256": protocol["initialization_checkpoints"][target]["sha256"]})
    metrics = pd.DataFrame(rows)
    filename = "frozen_rgb_ir_head_comparison" if joint else "initialization_comparison"
    metrics.to_csv(output / f"tables/{filename}.csv", index=False)
    pd.DataFrame(audits).to_csv(output / f"tables/{filename}_audit.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    panels = (("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("r2", "R squared"),
              ("explained_variance", "Explained variance"), ("mae_reference_iqr", "MAE / RGB main train IQR"),
              ("rmse_reference_iqr", "RMSE / RGB main train IQR"))
    y = np.arange(len(targets))
    for ax, (column, title) in zip(axes.flat, panels):
        for i, (name, color) in enumerate(colors.items()):
            values = metrics.loc[metrics.model.eq(name)].set_index("target").reindex(targets)[column]
            spacing = .8/len(colors)
            ax.barh(y+(i-(len(colors)-1)/2)*spacing, values, height=spacing*.9, color=color, label=name)
        ax.set_yticks(y, [LABELS[t] for t in targets]); ax.invert_yaxis()
        ax.set_title(title); ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
        ax.axvline(0, color=".5", linewidth=.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2 if joint else 3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / f"figures/{filename}.png", dpi=180); plt.close(fig)
    lines = ["", "## Initialization Comparison", "",
        "The two RGB+IR variants use identical paired training records, labels, patient splits, twenty frames, five views, target scalers and training settings. The RGB-trained variant loads each analyte's selected RGB main backbone AND head. Its final linear layer is converted between train-only scalers to preserve raw predictions; IR filters start at zero. RGB pretraining used the larger original RGB training cohort and the same held-out patients; this is supervised warm-starting, not a pure additional-modality ablation.", "",
        "All three models are evaluated on identical joint-selected RGB frames and common test videos. Scalar errors are normalized by the same RGB main training IQR in the comparison chart.", "",
        "|Analyte|Model|n|MAE|RMSE|r|R2|", "|---|---|---:|---:|---:|---:|---:|"]
    if joint:
        lines[1] = "## Frozen RGB / Joint IR And Head Comparison"
        lines[3] = ("All RGB+IR variants use the same paired labels, patient splits, frames, views and target scalers. Both RGB-initialized variants load the identical per-analyte RGB main best checkpoint and equivalent head scaling. The new variant jointly trains only IR filters (lr 1e-4) and head (lr 5e-5) in one stage, max 60 epochs/patience 12; the earlier variants use head then full-backbone finetuning. Frozen RGB parameters and BatchNorm buffers are checked bitwise against initialization. RGB supervised pretraining used the larger original training cohort, with the same held-out patient split.")
        lines[5] = "All four models use common test videos and identical joint-selected RGB frames. Errors in the chart share the original RGB training IQR normalization."
    for row in metrics.itertuples(index=False):
        lines.append(f"|{LABELS[row.target]}|{row.model}|{row.n}|{row.mae:.4f}|{row.rmse:.4f}|{row.pearson_r:.3f}|{row.r2:.3f}|")
    with (output / "REPORT.md").open("a") as handle:
        handle.write("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    targets = tuple(json.loads((args.output_dir / "protocol.json").read_text())["targets"])
    plot_results(args.output_dir, targets)
    if json.loads((args.output_dir / "protocol.json").read_text()).get("initialization") == "rgb_main":
        plot_initialization_comparison(args.output_dir, targets, OUTPUT)
