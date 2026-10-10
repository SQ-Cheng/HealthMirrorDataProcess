"""Add two directly reported targets, then fit the ten-task preoperative ablation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.common import run_video_loss_12h as worker_api
from study.common.time_alignment import local_naive_to_unix
from study.common.video_loss import DistinctLabViewBatchSampler
from study.exp2_face_history_head32_regression.source_data import (
    TARGET_ANALYTES, _binary_label, _load_hospital_episodes, _load_lab_data,
    _nearest_measurement, validate_analyte_source_policies,
)
from . import config
from .data import _distribution_audit, _plot_split_distributions, add_patient_split, validate_source_data
from .frame_index import FrameOffsetIndex, _index_is_reusable, build_or_reuse_frame_index
from .scaling import RobustTargetScaler, fit_robust_target_scaler, write_target_scalers


ROOT = Path(config.EXP_DIR).parents[1]
STUDY = ROOT / "study"
BASE = Path(config.OUTPUT_DIRS["20frame"])
TAG = "preoperative_nearest_unlimited_face224"
OUTPUT = Path(config.OUTPUT_ROOT) / "ablations" / TAG
LOGS = Path(config.LOG_DIR) / "ablations" / TAG
PREPARATION = OUTPUT / "source_data"
TARGETS = config.ALL_REGRESSION_TARGETS
ADDITIONAL = config.ADDITIONAL_REGRESSION_TARGETS
PROTOCOL = STUDY / "common/outputs/face_main_24h_frame_loss/protocol.json"


def select_measurement(measurements, start, end, admission, discharge, surgery_start=None):
    eligible = [(time, value) for time, value in measurements if admission <= time <= discharge]
    pre = surgery_start is not None and end <= surgery_start
    if pre:
        eligible = [(time, value) for time, value in eligible if time < surgery_start]
    selected = _nearest_measurement(eligible, start, end, float("inf") if pre else 24)
    return selected, pre


def extend_patient_split(records, reference, target):
    if reference.groupby("hospital_id").split.nunique().gt(1).any():
        raise ValueError("Reference split leaks patients")
    known = reference.groupby("hospital_id").split.first().to_dict()
    new = sorted(set(records.hospital_id) - set(known))
    offset = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")
    rng = np.random.default_rng((config.SEED + offset) % (2**32))
    shuffled = np.asarray(new, dtype=object)[rng.permutation(len(new))]
    if len(new) >= 3:
        train = min(max(1, round(.6 * len(new))), len(new) - 2)
        val = min(max(1, round(.2 * len(new))), len(new) - train - 1)
        splits = ["train"] * train + ["val"] * val + ["test"] * (len(new) - train - val)
    else:
        splits = rng.choice(["train", "val", "test"], len(new), p=[.6, .2, .2]).tolist()
    assignments = {**known, **dict(zip(shuffled, splits))}
    result = records.copy()
    result["split"] = result.hospital_id.map(assignments)
    result["split_origin"] = np.where(result.hospital_id.isin(known), "main_patient_split", "fixed_seed_new_patient")
    return result


def read_inventory(sex):
    audit = pd.read_csv(BASE / "source_data/raw_video_audit.csv", dtype={"hospital_id": str, "video_id": str})
    valid = audit.loc[audit.status.isin(("retained_24h_pool", "supported_lab_outside_24h", "patient_without_supported_lab"))]
    admissions = _load_hospital_episodes()
    rows = []
    for row in valid.itertuples(index=False):
        episodes = [ep for ep in admissions.get(row.hospital_id, ())
                    if ep[0] <= row.capture_start_unix and row.capture_end_unix <= ep[1]]
        if len(episodes) != 1:
            raise RuntimeError("Admission membership changed since reference preparation")
        mirror, local = row.video_id.split("_patient_")
        rows.append({"hospital_id": row.hospital_id, "video_id": row.video_id, "mirror": mirror,
                     "lab_patient_id": int(local), "sex": sex.get(row.hospital_id, ""),
                     "capture_start_unix": row.capture_start_unix, "capture_end_unix": row.capture_end_unix,
                     "admission_unix": episodes[0][0], "discharge_unix": episodes[0][1]})
    return pd.DataFrame(rows)


def attach_cabg(videos, digest):
    from study.exp4.build_dataset import load_surgical_episodes
    cache = STUDY / "exp4/outputs"
    manifest = cache / "experiment_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("source", {}).get("sha256") == digest:
        episodes = pd.read_csv(cache / "surgical_episodes.csv", dtype={"hospital_id": str})
        events = pd.read_csv(cache / "surgery_event_audit.csv", dtype={"hospital_id": str})
    else:
        episodes, events = load_surgical_episodes()
    keys = ["hospital_id", "admission_unix", "discharge_unix"]
    events = events.loc[events.is_cabg.eq(True) & events.valid_event.eq(True)].copy()
    for source, destination in (("admission_time", "admission_unix"), ("discharge_time", "discharge_unix")):
        events[destination] = events[source].map(local_naive_to_unix)
    events["surgery_start_unix"] = events.surgery_start.map(local_naive_to_unix)
    events["surgery_end_unix"] = events.surgery_end.map(local_naive_to_unix)
    # Count distinct operations, not repeated metadata attached to many assays.
    events = events.drop_duplicates(keys + ["surgery_start_unix", "surgery_end_unix"])
    counts = events.groupby(keys).size()
    valid = events.set_index(keys).loc[counts.eq(1).reindex(events.set_index(keys).index).to_numpy()]
    valid = valid.reset_index()[keys + ["surgery_start_unix", "surgery_end_unix"]]
    if valid.duplicated(keys).any():
        raise RuntimeError("Ambiguous surgery metadata")
    return videos.merge(valid, on=keys, how="left", validate="many_to_one")


def make_record(video, target, selected, pre=False):
    definition = config.SCORE_DEFINITIONS[target]
    threshold = definition["threshold"]
    if isinstance(threshold, dict):
        threshold = threshold["male"] if video.sex == "男" else threshold["other"]
    value = selected["value"]
    distance = (threshold - value) / definition["scale"] if definition["direction"] == "low" else (value - threshold) / definition["scale"]
    time = selected["timestamp_unix"]
    return {**video._asdict(), "binary_label": _binary_label(target, value, video.sex), "raw_value": value,
            "score_threshold": threshold, "score_scale": definition["scale"], "standardized_distance": distance,
            "abnormal_score": float(np.arcsinh(distance)), "source_sample_id": f"raw_video_{video.video_id}",
            "match_delta_h": selected["delta_h"], "match_signed_delta_h": selected["signed_delta_h"],
            "label_time_unix": time, "clinical_event_id": f"{video.hospital_id}@{time:.17g}",
            "systolic_blood_pressure": np.nan, "diastolic_blood_pressure": np.nan,
            "match_policy": "preoperative_nearest_unlimited" if pre else "unchanged_nearest_24h"}


def select_index(candidates):
    paths = [STUDY / "common/cache/face224_20frame_main",
             STUDY / "exp9_face_interpolated_lab_regression/cache/combined_frames20"]
    for directory in paths:
        path = directory / "frame_offsets.npz"
        if path.exists() and _index_is_reusable(directory, set(candidates.video_id), "20frame"):
            print(f"[frame-cache] reuse validated packet index: {path}", flush=True)
            return FrameOffsetIndex.load(path), path
    directory = Path(config.EXP_DIR) / "cache" / TAG
    return build_or_reuse_frame_index(candidates, str(directory), "20frame"), directory / "frame_offsets.npz"


def write_cohort(output, records, selections, merge=False):
    output.mkdir(parents=True, exist_ok=True)
    (output / "task_records").mkdir(exist_ok=True)
    summaries, audits, pairs, scalers = [], [], [], {}
    for target, table in records.items():
        if table.video_id.duplicated().any() or table.groupby("hospital_id").split.nunique().gt(1).any():
            raise RuntimeError(f"Duplicate video or patient leakage: {target}")
        scaler = fit_robust_target_scaler(target, table, config.SCORE_DEFINITIONS[target]["unit"])
        table["robust_scaled_raw_value"] = scaler.transform(table.raw_value)
        scalers[target] = scaler
        table.to_csv(output / f"task_records/{target}.csv", index=False)
        audit, pair = _distribution_audit(table, target)
        audits.extend(audit); pairs.extend(pair)
        row = {"target": target, "status": "ready", "clean_videos": len(table),
               "clean_patients": table.hospital_id.nunique(), "positive_videos": int(table.binary_label.sum()),
               "negative_videos": int(table.binary_label.eq(0).sum()),
               "preoperative_unlimited_videos": int(table.match_policy.eq("preoperative_nearest_unlimited").sum()),
               "matches_over24h": int(table.match_delta_h.gt(24).sum())}
        for split, group in table.groupby("split"):
            row.update({f"{split}_videos": len(group), f"{split}_patients": group.hospital_id.nunique(),
                        f"{split}_source_frames": 20 * len(group), f"{split}_lab_events": group.clinical_event_id.nunique()})
        row["train_augmented_inputs"] = row["train_source_frames"] * 5
        summaries.append(row)
    for name, rows in (("task_summary", summaries), ("split_distribution_audit", audits), ("split_distribution_pairwise", pairs)):
        current = pd.DataFrame(rows)
        path = output / f"{name}.csv"
        if merge and path.exists():
            old = pd.read_csv(path)
            current = pd.concat([old.loc[~old.target.isin(records)], current], ignore_index=True)
        current.to_csv(path, index=False)
    if merge:
        path = output / "target_scalers.json"
        saved = json.loads(path.read_text())
        saved["targets"].update({target: scaler.to_dict() for target, scaler in scalers.items()})
        path.write_text(json.dumps(saved, indent=2) + "\n")
        path = output / "split_assignment_manifest.json"
        saved = json.loads(path.read_text())
        saved["target_results"] = [row for row in saved["target_results"] if row["target"] not in records] + selections
        path.write_text(json.dumps(saved, indent=2) + "\n")
    else:
        write_target_scalers(scalers, output / "target_scalers.json")
        (output / "split_assignment_manifest.json").write_text(json.dumps({"target_results": selections}, indent=2) + "\n")
    _plot_split_distributions(records, output if not merge else BASE / "source_data/direct_fields")
    return scalers


def prepare():
    PREPARATION.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    digest = worker_api.sha256(ROOT / "merged_lab_tests.csv")
    if digest != protocol["lab_table_sha256"]:
        raise RuntimeError("Current Exp2 reference is not based on the latest lab table")
    quality = validate_source_data(BASE / "source_data", 24)
    lab_path = BASE / "source_data/lab_timeseries.csv"
    if worker_api.sha256(lab_path) != quality["source_fingerprints"]["lab_timeseries_cache"]["sha256"]:
        raise RuntimeError("Reference canonical labels changed")
    references = {}
    for target in config.TARGETS:
        path = BASE / f"task_records/{target}.csv"
        if worker_api.sha256(path) != protocol["source_records_sha256"][target]:
            raise RuntimeError(f"Original main task changed: {target}")
        references[target] = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
    extra_dir = BASE / "source_data/direct_fields"; extra_dir.mkdir(exist_ok=True)
    labs_extra, sex, extra_quality = _load_lab_data(ADDITIONAL, str(extra_dir))
    validate_analyte_source_policies(extra_quality, ADDITIONAL)
    (extra_dir / "data_quality_report.json").write_text(json.dumps(extra_quality, ensure_ascii=False, indent=2) + "\n")
    labs = pd.concat([pd.read_csv(lab_path, dtype={"hospital_id": str}, float_precision="round_trip"), labs_extra], ignore_index=True)
    if labs.duplicated(["hospital_id", "analyte", "timestamp_unix"]).any():
        raise RuntimeError("Canonical laboratory events are not unique")
    labs.to_csv(PREPARATION / "lab_timeseries.csv", index=False)
    videos = attach_cabg(read_inventory(sex), digest)
    lookup = {key: list(zip(group.timestamp_unix, group.value)) for key, group in
              labs.sort_values("timestamp_unix").groupby(["hospital_id", "analyte"])}
    direct, ablation, audit = {}, {}, []
    for target in TARGETS:
        rows, direct_rows = [], []
        old = references.get(target)
        old = old.set_index("video_id") if old is not None else None
        for video in videos.itertuples(index=False):
            measurements = lookup.get((video.hospital_id, TARGET_ANALYTES[target]), [])
            baseline, _ = select_measurement(measurements, video.capture_start_unix, video.capture_end_unix,
                                             video.admission_unix, video.discharge_unix)
            surgery = None if pd.isna(video.surgery_start_unix) else video.surgery_start_unix
            selected, pre = select_measurement(measurements, video.capture_start_unix, video.capture_end_unix,
                                               video.admission_unix, video.discharge_unix, surgery)
            if target in ADDITIONAL and baseline is not None:
                direct_rows.append(make_record(video, target, baseline))
            if not pre and old is not None:
                # Reuse the exact original record outside the changed preoperative rule.
                if video.video_id not in old.index:
                    continue
                record = old.loc[video.video_id].to_dict()
                record.update(video._asdict())
                record["match_policy"] = "unchanged_nearest_24h"
                rows.append(record)
            elif selected is not None:
                rows.append(make_record(video, target, selected, pre))
            audit.append({"target": target, "hospital_id": video.hospital_id, "video_id": video.video_id,
                          "is_preoperative": pre, "status": "retained" if selected is not None else
                          ("no_preoperative_lab" if pre else "no_lab_within24h")})
        ablation[target] = pd.DataFrame(rows)
        if target in ADDITIONAL:
            direct[target] = pd.DataFrame(direct_rows)
    candidates = pd.concat(list(ablation.values()) + list(direct.values()), ignore_index=True)
    index, index_path = select_index(candidates)
    valid = set(index.video_ids)
    candidates.loc[~candidates.video_id.isin(valid), ["hospital_id", "video_id"]].drop_duplicates().to_csv(
        PREPARATION / "frame_exclusions.csv", index=False)
    selections = []
    for target in ADDITIONAL:
        records = direct[target].loc[direct[target].video_id.isin(valid)].reset_index(drop=True)
        saved = BASE / f"task_records/{target}.csv"
        if saved.exists():
            existing = pd.read_csv(saved, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
            pd.testing.assert_frame_equal(records[["hospital_id", "video_id", "raw_value", "label_time_unix"]],
                                          existing[["hospital_id", "video_id", "raw_value", "label_time_unix"]], check_dtype=False)
            records["split"] = records.hospital_id.map(existing.groupby("hospital_id").split.first())
            saved_selections = json.loads((BASE / "split_assignment_manifest.json").read_text())["target_results"]
            selection = next(row for row in saved_selections if row["target"] == target)
        else:
            records, reason, _, _, selection = add_patient_split(records, target)
            if records is None:
                raise RuntimeError(f"{target}: {reason}")
        direct[target] = records; selections.append(selection); references[target] = records
    write_cohort(BASE, direct, selections, merge=True)
    selections = []
    for target in TARGETS:
        records = ablation[target].loc[ablation[target].video_id.isin(valid)].reset_index(drop=True)
        ablation[target] = extend_patient_split(records, references[target], target)
        selections.append({"target": target, "policy": "fixed reference patients; seeded extension only for newly eligible patients",
                           "new_patients": int(ablation[target].loc[ablation[target].split_origin.eq("fixed_seed_new_patient"), "hospital_id"].nunique())})
    write_cohort(OUTPUT, ablation, selections)
    videos.to_csv(PREPARATION / "video_inventory.csv", index=False)
    pd.DataFrame(audit).to_csv(PREPARATION / "matching_audit.csv", index=False)
    unchanged = {target: worker_api.sha256(BASE / f"task_records/{target}.csv") for target in config.TARGETS}
    if unchanged != protocol["source_records_sha256"]:
        raise RuntimeError("Preparation altered an original eight-task main record")
    settings = {key: getattr(config, key) for key in (
        "HEAD_LEARNING_RATE", "FINETUNE_LEARNING_RATE", "HEAD_MAX_EPOCHS", "FINETUNE_MAX_EPOCHS",
        "HEAD_PATIENCE", "FINETUNE_PATIENCE", "MIN_LEARNING_RATE", "WEIGHT_DECAY", "SMOOTH_L1_BETA",
        "HEAD_HIDDEN_FEATURES", "TORCH_COMPILE_MODE", "SEED", "VIEW_NAMES")}
    manifest = {"schema_version": 1, "targets": list(TARGETS), "additional_main_targets": list(ADDITIONAL),
                "lab_table_sha256": digest, "frame_index": str(index_path), "frame_index_sha256": worker_api.sha256(index_path),
                "original_eight_records_sha256": unchanged, "base_protocol_sha256": worker_api.sha256(PROTOCOL),
                "architecture": "one independent ImageNet EfficientNet-B0 + head32 per target",
                "training": settings, "frames_per_video": 20, "frame_batch_size": 240,
                "loss_level": "frame", "batch_policy": "distinct_lab_views", "distinct_lab_events_per_batch": 12,
                "preoperative_policy": "same admission, strictly pre-CABG reports only, nearest to video interval, unlimited distance",
                "unchanged_policy": "outside safely identified pre-CABG videos: exact original nearest24h rule",
                "ambiguous_or_missing_cabg": "no unrestricted extension; retain original 24h rule",
                "split_policy": "retain every main patient assignment; fixed-seed 60/20/20 extension for new patients",
                "direct_field_sources": extra_quality["analyte_source_policies"],
                "thresholds_are_auxiliary": "clinical-sign metrics and split stratification only; regression targets are raw values",
                "threshold_sources": ["https://www.medlineplus.gov/ency/article/003646.htm",
                                      "https://www.kidney.org/what-criteria-ckd"],
                "main_addition": "original eight model runs and matching source files preserved",
                "record_hashes": {str(root): {target: worker_api.sha256(root / f"task_records/{target}.csv")
                                              for target in (ADDITIONAL if root == BASE else TARGETS)} for root in (BASE, OUTPUT)}}
    (OUTPUT / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (BASE / "direct_field_extension_protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    preflight()
    print(pd.read_csv(OUTPUT / "task_summary.csv").to_string(index=False), flush=True)


def preflight():
    manifest = json.loads((OUTPUT / "experiment_manifest.json").read_text())
    if worker_api.sha256(ROOT / "merged_lab_tests.csv") != manifest["lab_table_sha256"]:
        raise RuntimeError("Lab table changed after preparation")
    if worker_api.sha256(PROTOCOL) != manifest["base_protocol_sha256"]:
        raise RuntimeError("Baseline training protocol changed")
    index_path = Path(manifest["frame_index"]); index = FrameOffsetIndex.load(index_path)
    if worker_api.sha256(index_path) != manifest["frame_index_sha256"] or not _index_is_reusable(index_path.parent, index.video_ids, "20frame"):
        raise RuntimeError("Native224 packet cache changed")
    for target, digest in manifest["original_eight_records_sha256"].items():
        if worker_api.sha256(BASE / f"task_records/{target}.csv") != digest:
            raise RuntimeError("Original main records changed")
    for root in (BASE, OUTPUT):
        scalers = json.loads((root / "target_scalers.json").read_text())["targets"]
        for target, digest in manifest["record_hashes"][str(root)].items():
            path = root / f"task_records/{target}.csv"
            if worker_api.sha256(path) != digest:
                raise RuntimeError(f"Prepared records changed: {path}")
            records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
            assert not records.video_id.duplicated().any()
            assert not records.groupby("hospital_id").split.nunique().gt(1).any()
            assert set(records.split) == {"train", "val", "test"}
            assert records.label_time_unix.between(records.admission_unix, records.discharge_unix).all()
            pre = records.match_policy.eq("preoperative_nearest_unlimited")
            assert records.loc[pre, "capture_end_unix"].le(records.loc[pre, "surgery_start_unix"]).all()
            assert records.loc[pre, "label_time_unix"].lt(records.loc[pre, "surgery_start_unix"]).all()
            assert records.loc[~pre, "match_delta_h"].between(0, 24 + 1e-9).all()
            fitted = fit_robust_target_scaler(target, records, config.SCORE_DEFINITIONS[target]["unit"])
            assert fitted.to_dict() == scalers[target]
            np.testing.assert_array_equal(fitted.transform(records.raw_value).astype(np.float32),
                                          records.robust_scaled_raw_value.to_numpy(np.float32))
            for video in records.video_id:
                start, end = index.frame_range(video)
                assert end - start == 20 and np.diff(index.source_indices[start:end]).min() >= 2
            if root == OUTPUT:
                reference = pd.read_csv(BASE / f"task_records/{target}.csv",
                                        dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
                common = records.merge(reference, on=["hospital_id", "video_id"], suffixes=("", "_reference"), validate="one_to_one")
                assert common.split.eq(common.split_reference).all()
                unchanged = common.loc[common.match_policy.eq("unchanged_nearest_24h")]
                for column in ("raw_value", "label_time_unix", "match_delta_h"):
                    np.testing.assert_array_equal(unchanged[column], unchanged[column + "_reference"])
                assignment = reference.groupby("hospital_id").split.first()
                inherited = records.hospital_id.isin(assignment.index)
                assert records.loc[inherited, "split"].eq(records.loc[inherited, "hospital_id"].map(assignment)).all()
            train = records.loc[records.split.eq("train")].reset_index(drop=True)
            assert train.groupby("clinical_event_id").raw_value.nunique().le(1).all()
            dataset = SimpleNamespace(expand_all_views=False, video_records=train,
                                      frame_video_rows=np.repeat(np.arange(len(train)), 20), views=config.VIEW_NAMES)
            batches = list(DistinctLabViewBatchSampler(dataset, 240))
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(train) * 100))
            for batch in batches:
                groups = np.asarray(batch).reshape(-1, 20)
                assert train.iloc[groups[:, 0] // 100].clinical_event_id.nunique() == len(groups)
                assert all(len(set(group % 5)) == 1 and len(set(group // 100)) == 1 for group in groups)
            assert all(len(batch) == 240 for batch in batches[:-1])
    print("[preflight-ok] ten fields; native224/20 frames/five views; no leakage; pre-CABG only; train-only scaling; 12 distinct labs/batch", flush=True)
    return index, manifest


def smoke():
    from . import train
    index, _ = preflight()
    for target in ADDITIONAL:
        records = pd.read_csv(BASE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
        subset = pd.concat([records.loc[records.split.eq("train")].drop_duplicates("clinical_event_id").head(12),
                            records.loc[records.split.ne("train")].groupby("split", group_keys=False).head(2)])
        scaler = RobustTargetScaler(**json.loads((BASE / "target_scalers.json").read_text())["targets"][target])
        with tempfile.TemporaryDirectory(prefix="direct_lab_regression_smoke_") as temporary:
            with patch.object(train, "TORCH_COMPILE_ENABLED", False), patch.object(train, "TRAIN_NUM_WORKERS", 0), patch.object(train, "EVAL_NUM_WORKERS", 0):
                train.train_task("efficientnet_b0", target, index, subset, scaler, config.WEIGHTS_DIR, temporary,
                                 head_epochs=1, finetune_epochs=1, max_batches=1,
                                 train_batch_policy="distinct_lab_views", loss_level="frame")
            checkpoint = torch.load(Path(temporary) / "model.pt", map_location="cpu", weights_only=True)
            assert checkpoint["target"] == target and checkpoint["loss_level"] == "frame"
    print("[smoke-ok] both new tasks: two stages, finite losses, saved checkpoints and predictions", flush=True)


def init_worker(queue, index_path):
    worker_api.GPU = int(queue.get())
    torch.cuda.set_device(worker_api.GPU)
    worker_api.INDEX = FrameOffsetIndex.load(index_path)
    torch.set_num_threads(1)


def finalize(root):
    targets = TARGETS
    for name in ("metrics", "history"):
        pd.concat([pd.read_csv(root / f"runs/efficientnet_b0/{target}/{name}.csv") for target in targets], ignore_index=True).to_csv(
            root / f"{name}_all.csv", index=False)
    from .plot_results import main as plot
    plot(root)
    if root == BASE:
        path = BASE / "experiment_manifest.json"
        saved = json.loads(path.read_text())
        saved["targets"] = list(TARGETS)
        saved["direct_field_extension"] = {
            "targets": list(ADDITIONAL), "protocol": str(BASE / "direct_field_extension_protocol.json"),
            "source_dir": str(BASE / "source_data/direct_fields"),
            "original_eight_model_runs_preserved": True,
        }
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    (root / "COMPLETE").write_text("ten independent regression models and figures completed\n")


def train_all():
    index, manifest = preflight()
    contract = worker_api.sha256(OUTPUT / "experiment_manifest.json")
    jobs, rows = [], {}
    finalized = set()
    for root, targets in ((BASE, ADDITIONAL), (OUTPUT, TARGETS)):
        rows[str(root)] = []
        if root == BASE:
            previous = pd.read_csv(BASE / "run_index.csv")
            rows[str(root)] = previous.loc[~previous.target.isin(ADDITIONAL)].to_dict("records")
        scalers = json.loads((root / "target_scalers.json").read_text())["targets"]
        for target in targets:
            run = root / f"runs/efficientnet_b0/{target}"
            marker = run / "job_complete.json"
            expected = {"contract": contract, "seed": worker_api.job_seed("regression", target)}
            if marker.exists() and json.loads(marker.read_text()) == expected and all(
                (run / name).exists() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")
            ):
                rows[str(root)].append({"family": "regression", "architecture": "efficientnet_b0", "target": target, "status": "ok", "seed": expected["seed"]})
                continue
            if marker.exists() and json.loads(marker.read_text()) != expected:
                raise RuntimeError(f"Different existing run protocol; not overwriting: {run}")
            jobs.append({"family": "regression", "target": target, "output": str(root), "batch_policy": "distinct_lab_views",
                         "loss_level": "frame", "frame_index_path": manifest["frame_index"], "reference_source": str(BASE),
                         "scaler": scalers[target], "contract": contract})
    workers = min(4, torch.cuda.device_count(), len(jobs))
    if jobs and workers == 0:
        raise RuntimeError("CUDA is required")
    print(f"[scheduler] two supplementary main models + ten preoperative models; pending={len(jobs)} GPUs={workers}", flush=True)
    ctx = mp.get_context("spawn")
    if jobs:
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers): queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                     initargs=(queue, manifest["frame_index"])) as pool:
                futures = {pool.submit(worker_api.train_one, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try: row = future.result()
                    except Exception:
                        row = {"family": "regression", "target": job["target"], "architecture": "efficientnet_b0",
                               "status": "failed", "error": traceback.format_exc()}
                        print(row["error"], flush=True)
                    rows[job["output"]].append(row)
                    pd.DataFrame(rows[job["output"]]).to_csv(Path(job["output"]) / "run_index.csv", index=False)
                    print(f"[task-finished] output={job['output']} target={job['target']} {row['status']}", flush=True)
                    main_rows = rows[str(BASE)]
                    if len(main_rows) == len(TARGETS) and all(item["status"] == "ok" for item in main_rows) and BASE not in finalized:
                        finalize(BASE)
                        finalized.add(BASE)
    if any(row["status"] != "ok" for group in rows.values() for row in group):
        raise RuntimeError("Training failed; inspect run_index.csv")
    for root in (BASE, OUTPUT):
        pd.DataFrame(rows[str(root)]).to_csv(root / "run_index.csv", index=False)
        if root not in finalized:
            finalize(root)
    from .plot_preoperative_comparison import plot_comparison
    (OUTPUT / "COMPLETE").unlink(missing_ok=True)
    plot_comparison(BASE, OUTPUT)
    preflight()
    (OUTPUT / "COMPLETE").write_text("ten-task ablation, two supplementary main models and comparison figures completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.reuse_prepared: prepare()
        if args.prepare_only: return
        if args.smoke: smoke(); return
        train_all()


if __name__ == "__main__":
    main()
