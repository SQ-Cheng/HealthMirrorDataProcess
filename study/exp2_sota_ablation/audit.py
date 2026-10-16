"""Read-only provenance validation and baseline diagnostics; no optimization."""

import json

import numpy as np
import pandas as pd
import torch

from study.common.run_video_loss_12h import sha256
from study.exp2_face_rgb_ir_head32_regression.index import load_indexes, signature
from study.exp2_face_rgb_ir_dinov3b_regression.features import cache_contract
from study.exp2_face_rgb_ir_dinov3b_regression.models import build_model
from study.exp2_face_dinov3_farl_regression import config as farl_config
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from . import config
from .data import read_records


def validate_sources():
    if not (config.BASELINE / "COMPLETE").is_file():
        raise RuntimeError("Completed baseline required")
    manifest_path = config.BASELINE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if cache_contract() != manifest["feature_contract"]:
        raise RuntimeError("Live encoder weights/normalization/index contract changed")
    if tuple(manifest["targets"]) != config.TARGETS:
        raise RuntimeError("Ten-analyte baseline contract differs")
    if sha256(config.reference.paired.ROOT / "merged_lab_tests.csv") != manifest["lab_table_sha256"]:
        raise RuntimeError("Live lab table changed; do not silently rematch this fixed-cohort ablation")
    rgb, ir = load_indexes(config.reference.RGB_INDEX, config.reference.IR_INDEX)
    sources = json.loads((config.reference.paired.CACHE / "manifest.json").read_text())
    if any(signature(path) != expected for path, expected in sources["source_signatures"].items()):
        raise RuntimeError("Native FFV1 video/sidecar changed; do not rebuild another experiment's index")
    cache_path = config.reference.CACHE / "manifest.json"
    cache = json.loads(cache_path.read_text())
    for mode, path in (("rgb", config.reference.RGB_INDEX), ("ir", config.reference.IR_INDEX)):
        if sha256(path) != manifest["feature_contract"]["indexes"][mode]:
            raise RuntimeError("Paired packet index changed")
    if cache["contract"] != {**manifest["feature_contract"], "frames": len(rgb.starts)}:
        raise RuntimeError("Feature cache differs from baseline training")
    for branch in config.reference.BRANCHES:
        path = config.reference.CACHE / f"{branch}.npy"
        if sha256(path) != cache["sha256"][branch]:
            raise RuntimeError("Frozen feature cache was changed")
    records_hashes, checkpoints, warmstarts = {}, {}, {}
    scalers = json.loads((config.BASELINE / "target_scalers.json").read_text())
    for target in config.TARGETS:
        path = config.BASELINE / f"source_records/{target}.csv"
        records_hashes[target] = sha256(path)
        if records_hashes[target] != manifest["records_sha256"][target]:
            raise RuntimeError("Baseline clinical records changed")
        records = read_records(target)
        if (records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any()
                or set(records.split) != {"train", "val", "test"} or records.clinical_event_id.isna().any()
                or not records.match_delta_h.between(0, 24+1e-9).all()):
            raise RuntimeError("Invalid fixed patient split/labels")
        if records.groupby("clinical_event_id").hospital_id.nunique().gt(1).any():
            raise RuntimeError("Clinical event crosses patients")
        scaler = RobustTargetScaler(**scalers[target])
        if fit_robust_target_scaler(target, records, scaler.unit).to_dict() != scaler.to_dict():
            raise RuntimeError("Baseline scaler is not train-only")
        run = config.BASELINE / f"regression/{config.ARCHITECTURE}/runs/{target}"
        checkpoints[target] = sha256(run / "model.pt")
        state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        if (state["experiment_contract"] != sha256(manifest_path) or state["feature_manifest_sha256"] != sha256(cache_path)
                or state["source_records_sha256"] != records_hashes[target] or state["seed"] != manifest["job_seeds"][target]):
            raise RuntimeError("Baseline checkpoint provenance changed")
        warm = farl_config.HERE / f"outputs/vitb16/regression/scalar_gated/runs/{target}/model.pt"
        warm_source = farl_config.HERE / f"outputs/vitb16/source_records/{target}.csv"
        old = torch.load(warm, map_location="cpu", weights_only=True)
        source = pd.read_csv(warm_source, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
        if old["clinical_source_sha256"] != sha256(warm_source) or old["target"] != target:
            raise RuntimeError("RGB warm-start source differs")
        source_scaler = RobustTargetScaler(**old["target_scaler"])
        if fit_robust_target_scaler(target, source, source_scaler.unit).to_dict() != source_scaler.to_dict():
            raise RuntimeError("RGB warm-start scaler is not train-only")
        inherited = source.set_index("video_id").loc[records.video_id].reset_index()
        for column in ("hospital_id", "split", "clinical_event_id"):
            np.testing.assert_array_equal(records[column], inherited[column])
        np.testing.assert_allclose(records.raw_value, inherited.raw_value, rtol=1e-12, atol=1e-12)
        held = set(records.loc[records.split.ne("train"), "hospital_id"])
        if held.intersection(source.loc[source.split.eq("train"), "hospital_id"]):
            raise RuntimeError("RGB warm-start trained on held-out paired patients")
        if (old["dino_weight_sha256"] != manifest["feature_contract"]["weights"]["dino"]
                or old["farl_weight_sha256"] != manifest["feature_contract"]["weights"]["farl"]):
            raise RuntimeError("Warm-start pretrained families differ")
        warmstarts[target] = {"path": str(warm), "sha256": sha256(warm), "source_records_sha256": sha256(warm_source),
                             "rgb_training_videos": int(source.split.eq("train").sum()),
                             "paired_training_videos": int(records.split.eq("train").sum()), "held_out_patient_leakage": False}
    return rgb, ir, {"baseline_manifest_sha256": sha256(manifest_path), "cache_manifest_sha256": sha256(cache_path),
        "records_sha256": records_hashes, "baseline_checkpoints_sha256": checkpoints,
        "job_seeds": manifest["job_seeds"], "warmstarts": warmstarts,
        "lab_table_sha256": manifest["lab_table_sha256"], "weights": manifest["feature_contract"]["weights"]}


@torch.inference_mode()
def write_baseline_audit(index, provenance):
    torch.set_num_threads(1)
    output = config.OUTPUT / "audit"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows, cohorts, gates, scales = [], [], [], []
    banks = {branch: np.load(config.reference.CACHE / f"{branch}.npy", mmap_mode="r") for branch in config.BRANCHES}
    for target in config.TARGETS:
        records = read_records(target)
        run = config.BASELINE / f"regression/{config.ARCHITECTURE}/runs/{target}"
        state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        history = pd.read_csv(run / "history.csv")
        metrics = pd.read_csv(run / "metrics.csv").set_index("split")
        best, last = history.loc[history.epoch.eq(state["best_epoch"])].iloc[0], history.iloc[-1]
        rows.append({"target": target, "best_epoch": state["best_epoch"], "last_epoch": int(last.epoch),
            "best_train_r": best.train_pearson_r, "best_val_r": best.val_pearson_r,
            "last_train_r": last.train_pearson_r, "last_val_r": last.val_pearson_r,
            "best_train_mae": best.train_mae, "best_val_mae": best.val_mae,
            "last_train_mae": last.train_mae, "last_val_mae": last.val_mae,
            "test_r": metrics.loc["test", "pearson_r"], "test_mae": metrics.loc["test", "mae"], "test_r2": metrics.loc["test", "r2"]})
        model = build_model(config.ARCHITECTURE).eval()
        model.load_state_dict(state["state_dict"], strict=True)
        for split, group in records.groupby("split"):
            cohorts.append({"target": target, "split": split, "videos": len(group), "patients": group.hospital_id.nunique(),
                "lab_events": group.clinical_event_id.nunique(), "median_match_h": group.match_delta_h.median(),
                "match_gt_12h_fraction": group.match_delta_h.gt(12).mean()})
            indices = np.concatenate([np.arange(*index.frame_range(video)) for video in group.video_id])
            weights = []
            for start in range(0, len(indices), 512):
                selected = indices[start:start+512]
                values = torch.from_numpy(np.concatenate([banks[branch][selected, 0] for branch in config.BRANCHES], 1))
                rgb, ir = values.split(1280, 1)
                rgb, ir = model.rgb_project(rgb), model.ir_project(ir)
                weights.append(model.gate(torch.cat((rgb, ir), 1)).sigmoid().numpy().ravel())
            values = np.concatenate(weights)
            gates.append({"target": target, "split": split, "gate_rgb_mean": values.mean(), "gate_rgb_std": values.std(),
                          "p10": np.quantile(values, .1), "median": np.median(values), "p90": np.quantile(values, .9),
                          "extreme_fraction": np.mean((values < .05) | (values > .95))})
            if split == "train":
                for branch in config.BRANCHES:
                    sums = np.zeros(banks[branch].shape[-1], dtype=np.float64)
                    for start in range(0, len(indices), 512):
                        sums += np.square(banks[branch][indices[start:start+512], 0].astype(np.float64)).sum(0)
                    scales.append({"target": target, "branch": branch, "train_original_feature_rms": np.sqrt(sums.sum()/len(indices)/len(sums))})
    pd.DataFrame(rows).to_csv(tables / "baseline_results.csv", index=False)
    pd.DataFrame(cohorts).to_csv(tables / "cohort_counts.csv", index=False)
    pd.DataFrame(gates).to_csv(tables / "baseline_gate_distribution.csv", index=False)
    pd.DataFrame(scales).to_csv(tables / "train_feature_scales.csv", index=False)
    from .models import Fusion
    plans = {name: {"settings": config.get_variant(name),
                   "fusion_parameters": sum(p.numel() for p in Fusion(config.get_variant(name)).parameters())}
             for name in config.VARIANTS}
    (tables / "ablation_plan.json").write_text(json.dumps({"training_started": False, "order": list(plans),
        "variants": plans, "fixed_patient_splits": True, "fixed_test_frames": 20,
        "ir_tail_trainable_parameters": 14573568, "test_based_selection": False}, indent=2)+"\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    baseline_settings = [
        ("Encoders", "independent RGB/IR DINOv3-B/16 (768) and FaRL ViT-B/16 (512); all four frozen"),
        ("Within modality", "concat768+512=1280; independent FC1280->64 + LN + SiLU"),
        ("Fusion", "input-dependent scalar g=sigmoid(FC128->1); g*RGB64+(1-g)*IR64; g init .5"),
        ("Task head", "FC64->32 + LN + SiLU + Dropout .25 + FC32->1"),
        ("Parameters", "343724544 frozen encoder parameters; 166530 trainable fusion/head parameters per analyte"),
        ("Target", "raw lab regression with paired train-only median/IQR; no clipping; ten independent models"),
        ("Data", "1311 unique paired videos /549 patients in union; per-analyte original patient split, no new search"),
        ("Matching", "nearest lab within +/-24h;20 non-adjacent mutual-nearest unique RGB/IR pairs <=33.333ms"),
        ("Input", "native224; IR repeated gray3; DINO ImageNet norm /FaRL CLIP norm; independent original crops"),
        ("Views", "original,hflip,center crop .9,brightness +.06,contrast +.08; all5 train, original only val/test"),
        ("Batch", "12 distinct clinical events x20 paired frames x1 view=240; all partials retained"),
        ("Loss", "unweighted frame-level SmoothL1 beta .5; no ranking, time/density/patient weighting"),
        ("Optimization", "one stage; AdamW lr2e-4, wd1e-3; cosine floor1e-6;80 epochs;patience12;clip1;AMP fp16"),
        ("Selection/evaluation", "best validation video-level raw MAE; mean20 original predictions then inverse scaler"),
        ("Execution", "frozen features cached once; train fusion/head only; eager, no compilation"),
    ]
    pd.DataFrame(baseline_settings, columns=["setting", "value"]).to_csv(tables / "baseline_architecture_hyperparameters.csv", index=False)
    lines = ["# Baseline Architecture and Audit", "", "| Setting | Actual baseline configuration |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in baseline_settings)
    columns = ["target", "best_epoch", "best_train_r", "best_val_r", "test_r", "test_mae", "test_r2"]
    result_lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for row in rows:
        result_lines.append("| "+" | ".join(f"{row[key]:.4g}" if isinstance(row[key], float) else str(row[key]) for key in columns)+" |")
    lines.extend(["", "## Observed Results", "", *result_lines, "",
        "Seven tasks selected epoch1; final train correlations are not the selected-checkpoint correlations. "
        "Gate diagnostics do not show general all-RGB/all-IR collapse. Feature scales differ; that is a hypothesis for branch standardization, not proof of causality."])
    (output / "BASELINE_REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"[audit-complete] {output}; no optimizer/GPU training invoked", flush=True)
