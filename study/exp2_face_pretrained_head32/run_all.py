"""One-process runner for all pretrained architecture/target experiments."""

import argparse
import json
import os
import random
import traceback

import numpy as np
import pandas as pd
import torch

from .config import (
    ARCHITECTURES,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    HEAD_HIDDEN_FEATURES,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    OUTPUT_DIR,
    SEED,
    SOURCE_DATA_DIR,
    TARGETS,
    WEIGHTS_DIR,
)
from .data import prepare_tasks, validate_source_data
from .models import WEIGHT_FILES
from .train import train_task


def _parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


LAB_TARGET_PREFIXES = {
    "lactate_high": "lactate",
    "lactate_moderate_high": "lactate",
    "troponin_high": "troponin",
    "troponin_extreme_high": "troponin",
    "glucose_high": "glucose",
    "glucose_marked_high": "glucose",
    "hemoglobin_low": "hemoglobin",
    "hemoglobin_moderate_low": "hemoglobin",
    "po2_low": "po2",
    "po2_moderate_low": "po2",
    "pco2_abnormal": "pco2",
    "pco2_low": "pco2",
    "pco2_high": "pco2",
}


def _validate_time_alignment(manifest, targets):
    for target in targets:
        prefix = LAB_TARGET_PREFIXES.get(target)
        if prefix is None:
            continue
        labelled = pd.to_numeric(manifest[target], errors="coerce").notna()
        if not labelled.any():
            continue
        delta = pd.to_numeric(manifest.loc[labelled, f"{prefix}_delta_h"], errors="coerce")
        signed = pd.to_numeric(
            manifest.loc[labelled, f"{prefix}_signed_delta_h"], errors="coerce"
        )
        invalid = (
            delta.isna()
            | signed.isna()
            | delta.lt(0.0)
            | delta.gt(24.0 + 1e-6)
            | ~np.isclose(delta.to_numpy(), np.abs(signed.to_numpy()), atol=1e-7)
        )
        if invalid.any():
            raise ValueError(
                f"Invalid 24-hour alignment for {target}: {int(invalid.sum())} rows"
            )


def _validate_weights(weights_dir, architectures):
    missing = [
        os.path.join(weights_dir, WEIGHT_FILES[name])
        for name in architectures
        if not os.path.exists(os.path.join(weights_dir, WEIGHT_FILES[name]))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing local pretrained files. Run "
            "`python -m study.exp2_face_pretrained.download_weights`: "
            + ", ".join(missing)
        )


def _collect_histories(output_dir):
    histories = []
    runs_dir = os.path.join(output_dir, "runs")
    if not os.path.isdir(runs_dir):
        return pd.DataFrame()
    for root, _, files in os.walk(runs_dir):
        if "history.csv" in files:
            histories.append(pd.read_csv(os.path.join(root, "history.csv")))
    return pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=SOURCE_DATA_DIR)
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--head-epochs", type=int, default=HEAD_MAX_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_MAX_EPOCHS)
    parser.add_argument("--head-patience", type=int, default=HEAD_PATIENCE)
    parser.add_argument("--finetune-patience", type=int, default=FINETUNE_PATIENCE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    architectures = _parse_csv(args.architectures)
    requested_targets = _parse_csv(args.targets)
    unknown_architectures = sorted(set(architectures) - set(ARCHITECTURES))
    unknown_targets = sorted(set(requested_targets) - set(TARGETS))
    if unknown_architectures or unknown_targets:
        raise ValueError(
            f"Unknown architectures={unknown_architectures}, targets={unknown_targets}"
        )
    if args.smoke_test:
        args.head_epochs = 1
        args.finetune_epochs = 1
        args.head_patience = 1
        args.finetune_patience = 1
        args.max_batches = args.max_batches or 2

    existing_index = os.path.join(args.output_dir, "run_index.csv")
    if os.path.exists(existing_index) and not args.overwrite:
        raise FileExistsError(
            f"Output already contains a run: {args.output_dir}. Use --overwrite explicitly."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
    _set_seed(args.seed)
    source_quality = validate_source_data(args.source_dir)
    _validate_weights(args.weights_dir, architectures)
    weight_manifest_path = os.path.join(args.weights_dir, "manifest.json")
    if not os.path.exists(weight_manifest_path):
        raise FileNotFoundError(f"Missing pretrained weight manifest: {weight_manifest_path}")
    with open(weight_manifest_path, encoding="utf-8") as handle:
        weight_manifest = json.load(handle)

    manifest_path = os.path.join(args.source_dir, "manifest.csv")
    features_path = os.path.join(args.source_dir, "features.npz")
    manifest = pd.read_csv(manifest_path, dtype={"hospital_id": str})
    _validate_time_alignment(manifest, requested_targets)
    with np.load(features_path, allow_pickle=False) as features:
        face = features["face"]
    if face.ndim != 5 or face.shape[1:] != (20, 3, 128, 128):
        raise ValueError(f"Expected native RGB [V,20,3,128,128], got {face.shape}")
    if manifest[["video_index", "frame_index"]].isna().any().any():
        raise ValueError("Missing feature indices in source manifest")
    if not manifest["video_index"].between(0, face.shape[0] - 1).all():
        raise ValueError("Source manifest contains out-of-range video_index values")
    if not manifest["frame_index"].between(0, face.shape[1] - 1).all():
        raise ValueError("Source manifest contains out-of-range frame_index values")
    for column in ("hospital_id", "video_index"):
        if manifest.groupby("video_id")[column].nunique().gt(1).any():
            raise ValueError(f"video_id maps to multiple {column} values")
    print(
        f"Loaded corrected source: rows={len(manifest)} face_shape={face.shape} "
        f"timezone={source_quality['lab_report_time']['source_timezone']}",
        flush=True,
    )

    task_records, task_summary, conflict_audit = prepare_tasks(
        manifest, args.output_dir, targets=requested_targets, seed=args.seed
    )
    ready_targets = tuple(
        task_summary.loc[task_summary["status"].eq("ready"), "target"].astype(str)
    )
    if args.smoke_test and ready_targets:
        ready_targets = ready_targets[:1]
    print(
        f"Task preparation: ready={len(ready_targets)} "
        f"skipped={int(task_summary['status'].eq('skipped').sum())}; "
        f"generalized_conflicting_video_target_pairs={len(conflict_audit)}",
        flush=True,
    )
    for row in task_summary.loc[task_summary["status"].eq("skipped")].itertuples():
        print(f"[task-skip] {row.target}: {row.reason}", flush=True)

    experiment_manifest = {
        "schema_version": 1,
        "source_dir": os.path.abspath(args.source_dir),
        "source_data_quality_report": source_quality,
        "preprocessing": {
            "source_frame_shape": [3, 128, 128],
            "time_alignment_validation": "delta_h <= 24 and delta_h == abs(signed_delta_h)",
            "model_input_shape": [3, 224, 224],
            "normalization": "ImageNet mean/std",
            "conflict_policy": "exclude a video per target if both labels occur",
            "duplicate_policy": "one copy of each video/frame per target",
            "split_policy": "patient-level stratified 60/20/20 shared by architectures",
            "evaluation_unit": "video; mean probability over 20 frames",
        },
        "training": {
            "stage_1": "frozen encoder, replacement single-task head, lr=1e-3",
            "stage_2": "all parameters unfrozen, lr=1e-4",
            "seed": args.seed,
            "head_max_epochs": args.head_epochs,
            "finetune_max_epochs": args.finetune_epochs,
        },
        "model_head": {
            "type": "Linear-LayerNorm-SiLU-Dropout-Linear",
            "hidden_features": HEAD_HIDDEN_FEATURES,
            "dropout": 0.25,
        },
        "parent_experiment": "study/exp2_face_pretrained",
        "architectures": list(architectures),
        "pretrained_weights": weight_manifest,
        "requested_targets": list(requested_targets),
        "ready_targets": list(ready_targets),
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(experiment_manifest, handle, ensure_ascii=False, indent=2)

    run_rows, metric_frames, failure_rows = [], [], []
    total_jobs = len(architectures) * len(ready_targets)
    completed = 0
    for architecture in architectures:
        for target in ready_targets:
            run_dir = os.path.join(args.output_dir, "runs", architecture, target)
            try:
                metrics = train_task(
                    architecture=architecture,
                    target=target,
                    face=face,
                    records=task_records[target],
                    weights_dir=args.weights_dir,
                    run_dir=run_dir,
                    head_epochs=args.head_epochs,
                    finetune_epochs=args.finetune_epochs,
                    head_patience=args.head_patience,
                    finetune_patience=args.finetune_patience,
                    max_batches=args.max_batches,
                )
                metric_frames.append(metrics)
                status, reason = "ok", ""
            except Exception as exc:
                status, reason = "failed", str(exc)
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "error.txt"), "w", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
                failure_rows.append({
                    "architecture": architecture,
                    "target": target,
                    "error": str(exc),
                })
                print(f"[job-failed] arch={architecture} task={target}: {exc}", flush=True)
            completed += 1
            run_rows.append({
                "architecture": architecture,
                "target": target,
                "status": status,
                "reason": reason,
                "run_dir": run_dir,
            })
            pd.DataFrame(run_rows).to_csv(existing_index, index=False)
            if metric_frames:
                pd.concat(metric_frames, ignore_index=True).to_csv(
                    os.path.join(args.output_dir, "metrics_all.csv"), index=False
                )
            _collect_histories(args.output_dir).to_csv(
                os.path.join(args.output_dir, "history_all.csv"), index=False
            )
            print(
                f"[scheduler] {completed}/{total_jobs} arch={architecture} "
                f"task={target} status={status}",
                flush=True,
            )
    pd.DataFrame(failure_rows).to_csv(
        os.path.join(args.output_dir, "failures.csv"), index=False
    )
    print(f"Experiment outputs saved to {args.output_dir}", flush=True)
    if failure_rows:
        raise RuntimeError(f"{len(failure_rows)} training jobs failed; see failures.csv")


if __name__ == "__main__":
    main()
