"""CPU-only matched-frame reevaluation of existing RGB and RGB/IR checkpoints."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import explained_variance_score

from study.common.run_video_loss_12h import sha256
from study.exp2_face_architecture_ablation.plots import panels
from study.exp2_face_dinov3_frozen.backbone import Head
from study.exp2_face_dinov3_farl_regression.models import build_model as build_rgb_farl
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp2_face_pretrained_head32_regression.train import _regression_metrics
from . import config
from .models import build_model


RGB_DINO = config.reference.config.HERE / "outputs/main24h_frame_loss_vitb16"
RGB_FARL = config.farl_config.HERE / "outputs/vitb16"
METHOD_LABELS = {"concat": "Concat", "vector_gate": "Vector gate", "scalar_gate": "Scalar gate"}
GROUP_LABELS = {"dino": "DINOv3-B", "dino_farl": "DINOv3-B + FaRL"}
METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"),
           ("r2", "$R^2$"), ("explained_variance", "Explained variance"))
COLORS = ("#2878b5", "#278245")


def read_records(path):
    return pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")


def rgb_spec(group, method, target):
    if group == "dino":
        root = RGB_DINO / "regression/dinov3_vitb16_frozen"
        return root / f"runs/{target}/model.pt", ("rgb_dino",), Head(32, in_features=768)
    variant = {"concat": "concat", "vector_gate": "gated", "scalar_gate": "scalar_gated"}[method]
    return (RGB_FARL / f"regression/{variant}/runs/{target}/model.pt",
            ("rgb_dino", "rgb_farl"), build_rgb_farl(variant, 768))


@torch.inference_mode()
def predict_video_values(model, state, records, index, banks, branches):
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    scaler = RobustTargetScaler(**state["target_scaler"])
    frame_indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.video_id])
    if len(frame_indices) != len(records)*20:
        raise RuntimeError("Matched evaluation requires exactly twenty paired frames per video")
    outputs = []
    for start in range(0, len(frame_indices), 512):
        selected = frame_indices[start:start+512]
        features = np.concatenate([banks[branch][selected, 0] for branch in branches], axis=1)
        if not np.isfinite(features).all():
            raise RuntimeError("Nonfinite cached original-view features")
        outputs.append(model(torch.from_numpy(features)).squeeze(1).numpy())
    frame_scores = np.concatenate(outputs)
    if not np.isfinite(frame_scores).all():
        raise RuntimeError("Nonfinite frozen-head predictions")
    # Identical float32 head inference and float64 pooling for both modalities.
    scores = frame_scores.astype(np.float64).reshape(len(records), 20).mean(1)
    predictions = records[["hospital_id", "video_id", "raw_value", "score_threshold"]].copy()
    predictions["split"] = "test"
    predictions["y_true"] = predictions.raw_value
    predictions["y_pred_scaled"] = scores
    predictions["y_pred"] = scaler.inverse_transform(scores)
    predictions["frame_count"] = 20
    return predictions


def validate_training_source(state, records, paired_records, target):
    if (records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any()
            or paired_records.groupby("hospital_id").split.nunique().gt(1).any()):
        raise RuntimeError("Duplicated videos or patient split leakage")
    matched = records.set_index("video_id").loc[paired_records.video_id].reset_index()
    for column in ("hospital_id", "split", "clinical_event_id"):
        np.testing.assert_array_equal(matched[column], paired_records[column])
    np.testing.assert_allclose(matched.raw_value, paired_records.raw_value, rtol=1e-12, atol=1e-12)
    scaler = RobustTargetScaler(**state["target_scaler"])
    if scaler.target != target or state["seed"] != config.reference.reference.job_seed("regression", target):
        raise RuntimeError("Checkpoint target/seed mismatch")
    fitted = fit_robust_target_scaler(target, records, scaler.unit)
    if fitted.to_dict() != scaler.to_dict():
        raise RuntimeError("Checkpoint scaler does not match its own training population")


def evaluate_comparators():
    """No video decoding, pretrained encoder loading, training, or CUDA access."""
    torch.set_num_threads(1)
    for root in (RGB_DINO, RGB_FARL):
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete source experiment: {root}")
    experiment = json.loads((config.OUTPUT / "experiment_manifest.json").read_text())
    contract_hash = sha256(config.OUTPUT / "experiment_manifest.json")
    for target in config.TARGETS:
        for variant in config.VARIANTS:
            marker = config.OUTPUT / f"regression/{variant}/runs/{target}/job_complete.json"
            if json.loads(marker.read_text()) != {"contract": contract_hash}:
                raise RuntimeError("Incomplete or changed RGB/IR head")
    feature_manifest_path = config.CACHE / "manifest.json"
    cached = json.loads(feature_manifest_path.read_text())
    if cached["contract"] != {**experiment["feature_contract"], "frames": cached["contract"]["frames"]}:
        raise RuntimeError("Paired feature contract differs from trained RGB/IR models")
    if sha256(config.RGB_INDEX) != cached["contract"]["indexes"]["rgb"]:
        raise RuntimeError("Paired RGB packet index changed")
    index = FrameOffsetIndex.load(config.RGB_INDEX)
    banks = {}
    for branch in config.BRANCHES:
        path = config.CACHE / f"{branch}.npy"
        if sha256(path) != cached["sha256"][branch]:
            raise RuntimeError("Changed frozen feature cache")
        banks[branch] = np.load(path, mmap_mode="r")
        if banks[branch].shape != (len(index.starts), 5, config.DIMENSIONS[branch]):
            raise RuntimeError("Cache/index shape mismatch")
    references = {"dino": json.loads((RGB_DINO / "experiment_manifest.json").read_text()),
                  "dino_farl": json.loads((RGB_FARL / "experiment_manifest.json").read_text())}
    for group, manifest in references.items():
        weight_key = "backbone_weight_sha256" if group == "dino" else "dino_weight_sha256"
        if (manifest["lab_table_sha256"] != experiment["lab_table_sha256"]
                or manifest[weight_key] != cached["contract"]["weights"]["dino"]):
            raise RuntimeError("RGB clinical source or DINO backbone differs")
        if group == "dino_farl" and manifest["farl_weight_sha256"] != cached["contract"]["weights"]["farl"]:
            raise RuntimeError("RGB FaRL backbone differs")
    tables = config.OUTPUT / "tables"
    tables.mkdir(exist_ok=True)
    metrics, predictions, audits, checkpoints = [], [], [], {}
    for target in config.TARGETS:
        path = config.OUTPUT / f"source_records/{target}.csv"
        if sha256(path) != experiment["records_sha256"][target]:
            raise RuntimeError("Paired training labels/splits changed")
        paired = read_records(path)
        test = paired.loc[paired.split.eq("test")].sort_values("video_id").reset_index(drop=True)
        rgb_predictions = {}
        for group in config.GROUPS:
            reference_path = (RGB_DINO if group == "dino" else RGB_FARL) / f"source_records/{target}.csv"
            if sha256(reference_path) != references[group]["source_records_sha256"][target]:
                raise RuntimeError("RGB source labels/splits changed")
            original = read_records(reference_path)
            for method in config.METHODS:
                architecture = f"{group}_{method}"
                for modality in ("rgb", "rgb_ir"):
                    if modality == "rgb":
                        model_path, branches, model = rgb_spec(group, method, target)
                    else:
                        model_path = config.OUTPUT / f"regression/{architecture}/runs/{target}/model.pt"
                        branches, model = config.branches_for(architecture), build_model(architecture)
                    checkpoints[str(model_path)] = sha256(model_path)
                    state = torch.load(model_path, map_location="cpu", weights_only=True)
                    source_records = original if modality == "rgb" else paired
                    validate_training_source(state, source_records, paired, target)
                    if modality == "rgb":
                        if (state.get("backbone_weight_sha256", state.get("dino_weight_sha256")) != cached["contract"]["weights"]["dino"]
                                or (group == "dino_farl" and state["farl_weight_sha256"] != cached["contract"]["weights"]["farl"])):
                            raise RuntimeError("Checkpoint encoder weights differ from cached RGB features")
                        source_hash = state.get("source_records_sha256", state.get("clinical_source_sha256"))
                        if source_hash != sha256(reference_path):
                            raise RuntimeError("Checkpoint training records differ from comparator source")
                    elif (state["feature_manifest_sha256"] != sha256(feature_manifest_path)
                          or state["experiment_contract"] != sha256(config.OUTPUT / "experiment_manifest.json")):
                        raise RuntimeError("RGB/IR checkpoint does not use these paired features")
                    key = str(model_path)
                    if modality == "rgb" and key in rgb_predictions:
                        values = rgb_predictions[key].copy()
                    else:
                        values = predict_video_values(model, state, test, index, banks, branches)
                        if modality == "rgb":
                            rgb_predictions[key] = values.copy()
                    drift = np.nan
                    if modality == "rgb_ir":
                        saved = read_records(model_path.parent / "video_predictions.csv").query("split == 'test'").sort_values("video_id").reset_index(drop=True)
                        pd.testing.assert_frame_equal(test[["hospital_id", "video_id"]], saved[["hospital_id", "video_id"]])
                        np.testing.assert_allclose(test.raw_value, saved.y_true, rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(values.y_pred_scaled, saved.y_pred_scaled, rtol=5e-3, atol=1e-2)
                        drift = float(np.max(np.abs(values.y_pred_scaled-saved.y_pred_scaled)))
                    result = _regression_metrics(values.y_true, values.y_pred, values.score_threshold,
                                                 config.paired.SCORE_DEFINITIONS[target]["direction"])
                    metrics.append({"target": target, "family": group, "method": method, "modality": modality,
                        "training_videos": int(source_records.split.eq("train").sum()),
                        "training_patients": source_records.loc[source_records.split.eq("train"), "hospital_id"].nunique(),
                        **result, "explained_variance": explained_variance_score(values.y_true, values.y_pred)})
                    predictions.append(values.assign(target=target, family=group, method=method, modality=modality))
                    audits.append({"target": target, "family": group, "method": method, "modality": modality,
                        "checkpoint": str(model_path), "checkpoint_sha256": checkpoints[str(model_path)],
                        "test_videos": len(test), "test_patients": test.hospital_id.nunique(), "same_test_videos_labels_rgb_packets": True,
                        "scaler": "each checkpoint's own train-only median/IQR", "head_inference": "CPU float32 for both modalities",
                        "max_scaled_prediction_drift_vs_saved_amp": drift, "training_cohorts_identical": False,
                        "rgb_gate_axis": "none" if group == "dino" or method == "concat" else "DINO vs FaRL",
                        "rgb_ir_gate_axis": "none" if method == "concat" else "RGB vs IR"})
        print(f"[rgb-comparison] {target}: {len(test)} identical paired20 test videos, both families/all fusions", flush=True)
    for path, digest in checkpoints.items():
        if sha256(path) != digest:
            raise RuntimeError("Checkpoint changed during read-only comparison")
    result = pd.DataFrame(metrics)
    result.to_csv(tables / "rgb_only_vs_rgb_ir_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(tables / "rgb_only_vs_rgb_ir_predictions.csv", index=False)
    pd.DataFrame(audits).to_csv(tables / "rgb_only_vs_rgb_ir_audit.csv", index=False)
    (tables / "rgb_only_vs_rgb_ir_protocol.json").write_text(json.dumps({
        "training": "none; existing checkpoints only", "device": "CPU", "head_precision": "float32, identical both modalities",
        "feature_precision": "existing float32-stored encoder AMP features, same RGB packets in both modalities",
        "evaluation": "same common test videos; same paired20 RGB frames; original view only; per-video mean, own scaler inverse",
        "cache_manifest_sha256": sha256(feature_manifest_path), "checkpoints_sha256": checkpoints,
        "scope": "backbone-family matched, not a strict same-training-cohort/identical-fusion-graph IR ablation",
        "caveats": ["RGB-only models trained on larger original RGB cohorts; no retraining", "single DINO RGB baseline repeated across fusion methods",
                    "RGB DINO/FaRL gates act across encoders, RGB/IR gates act across modalities after within-modality concatenation"]}, indent=2)+"\n")
    return result


def plot_comparisons(table):
    figures = config.OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    footer = "Identical test videos and paired20 RGB frames; RGB-only models trained on a larger cohort."
    for group in config.GROUPS:
        extra = ("The same RGB DINOv3-B head32 baseline is repeated for all three fusion methods." if group == "dino" else
                 "RGB-only gates: DINO vs FaRL. RGB+IR gates: RGB vs IR, with DINO/FaRL concatenated within each modality.")
        for metric, label in METRICS:
            figure, axes = panels(config.TARGETS)
            for axis in axes.flat[len(config.TARGETS):]:
                axis.set_visible(False)
            for axis, target in zip(axes.flat, config.TARGETS):
                selected = table.loc[table.family.eq(group) & table.target.eq(target)].set_index(["method", "modality"])
                for modality, offset, color, name in (("rgb", -.18, COLORS[0], "RGB only"), ("rgb_ir", .18, COLORS[1], "RGB + IR")):
                    heights = [selected.loc[(method, modality), metric] for method in config.METHODS]
                    bars = axis.bar(np.arange(3)+offset, heights, width=.34, color=color, label=name)
                    axis.bar_label(bars, fmt="%.3f", fontsize=6, padding=3)
                axis.set_xticks(range(3), [METHOD_LABELS[method] for method in config.METHODS], rotation=20, ha="right")
                axis.set(title=TASK_LABELS[target], ylabel=label+(f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
                axis.axhline(0, color=".5", linewidth=.6)
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
                axis.margins(y=.2)
            axes.flat[0].legend(fontsize=8, frameon=False)
            figure.suptitle(f"{GROUP_LABELS[group]} | RGB only vs RGB+IR | {label}")
            figure.supxlabel(footer+"\n"+extra, fontsize=9)
            figure.savefig(figures / f"rgb_only_vs_rgb_ir_{group}_{metric}.png", dpi=180)
            plt.close(figure)
    # One compact figure covering both backbone families and all fusion choices.
    figure, axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
    for row, group in enumerate(config.GROUPS):
        for column, method in enumerate(config.METHODS):
            axis = axes[row, column]
            selected = table.loc[table.family.eq(group) & table.method.eq(method)].set_index(["target", "modality"])
            for modality, offset, color, label in (("rgb", -.18, COLORS[0], "RGB only"), ("rgb_ir", .18, COLORS[1], "RGB + IR")):
                values = [selected.loc[(target, modality), "pearson_r"] for target in config.TARGETS]
                bars = axis.barh(np.arange(len(config.TARGETS))+offset, values, height=.34, color=color, label=label)
                axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
            axis.set_yticks(range(len(config.TARGETS)), [TASK_LABELS[target] for target in config.TARGETS])
            axis.invert_yaxis()
            axis.axvline(0, color=".5", linewidth=.6)
            axis.grid(axis="x", alpha=.2)
            axis.set_axisbelow(True)
            axis.set(title=f"{GROUP_LABELS[group]} / {METHOD_LABELS[method]}", xlabel="Test Pearson r")
            axis.margins(x=.2)
    axes[0, 0].legend(fontsize=9, frameon=False)
    figure.suptitle("Frozen DINOv3-B (+FaRL): RGB-only versus RGB+IR")
    figure.supxlabel(footer+"\nRGB DINO/FaRL gates combine encoders; RGB+IR gates combine modalities. Single-DINO RGB baseline repeated.", fontsize=10)
    figure.savefig(figures / "rgb_only_vs_rgb_ir_pearson_r_overview.png", dpi=180)
    plt.close(figure)
    print(f"[rgb-comparison-plots] {figures}; 11 PNGs, no training or CUDA use", flush=True)


def main():
    plot_comparisons(evaluate_comparators())


if __name__ == "__main__":
    main()
