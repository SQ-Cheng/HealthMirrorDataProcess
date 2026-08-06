"""Prepare and run the controlled history-only Head32 ablation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import pandas as pd
import torch

from .config import (
    BATCH_SIZE,
    HEAD_HIDDEN_FEATURES,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_NAME,
    OUTPUT_DIR,
    PATIENCE,
    REFERENCE_DIR,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    TARGETS,
    WEIGHT_DECAY,
)
from .data import load_task
from .models import HistoryOnlyRegressor, parameter_counts
from .train import train_task


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _job_seed(target, seed):
    token = f"{seed}:{MODEL_NAME}:{target}".encode()
    return (seed + int.from_bytes(hashlib.sha256(token).digest()[:4], "little")) % (
        2**31 - 1
    )


def _copy_and_audit_inputs(reference_dir, output_dir, targets):
    task_dir = output_dir / "task_records"
    history_dir = output_dir / "history_records"
    task_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for target in targets:
        task_source = reference_dir / "task_records" / f"{target}.csv"
        history_source = reference_dir / "history_records" / f"{target}.npz"
        task_destination = task_dir / task_source.name
        history_destination = history_dir / history_source.name
        shutil.copy2(task_source, task_destination)
        shutil.copy2(history_source, history_destination)
        for suffix in (".csv", "_summary.csv"):
            source = reference_dir / "history_records" / f"{target}{suffix}"
            if source.is_file():
                shutil.copy2(source, history_dir / source.name)
        records, history = load_task(output_dir, target)
        reference_records, reference_history = load_task(reference_dir, target)
        pd.testing.assert_frame_equal(
            records.sort_values("video_id").reset_index(drop=True),
            reference_records.sort_values("video_id").reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
        if not (
            np.array_equal(history.video_ids, reference_history.video_ids)
            and np.array_equal(history.offsets, reference_history.offsets)
            and np.array_equal(history.features, reference_history.features)
        ):
            raise AssertionError(f"Copied history features differ for {target}")
        lookup = history.lookup()
        for split, group in records.groupby("split", sort=True):
            counts = []
            for video_id in group["video_id"].astype(str):
                row = lookup[video_id]
                counts.append(int(history.offsets[row + 1] - history.offsets[row]))
            audit_rows.append(
                {
                    "target": target,
                    "split": split,
                    "videos": len(group),
                    "patients": group["hospital_id"].nunique(),
                    "positive_videos": int(group["abnormal_score"].gt(0).sum()),
                    "negative_videos": int(group["abnormal_score"].lt(0).sum()),
                    "history_measurements": int(np.sum(counts)),
                    "videos_with_history": int(np.count_nonzero(counts)),
                    "median_history_count": float(np.median(counts)),
                    "max_history_count": int(np.max(counts)),
                    "task_record_sha256": _sha256(task_destination),
                    "history_feature_sha256": _sha256(history_destination),
                    "reference_task_record_sha256": _sha256(task_source),
                    "reference_history_feature_sha256": _sha256(history_source),
                    "exact_match": True,
                }
            )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "data_alignment_audit.csv", index=False)
    if not (
        audit["task_record_sha256"].eq(audit["reference_task_record_sha256"]).all()
        and audit["history_feature_sha256"]
        .eq(audit["reference_history_feature_sha256"])
        .all()
    ):
        raise AssertionError("Copied input hashes differ from the reference experiment")
    return audit


def _run_job(job):
    gpu_id = int(job["gpu_id"])
    torch.cuda.set_device(gpu_id)
    seed = int(job["seed"])
    _seed(seed)
    records, history = load_task(Path(job["output_dir"]), job["target"])
    return train_task(
        target=job["target"],
        records=records,
        history_store=history,
        run_dir=Path(job["output_dir"]) / "runs" / job["target"],
        device=torch.device(f"cuda:{gpu_id}"),
        seed=seed,
        max_epochs=job["max_epochs"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "runs").mkdir(parents=True)
    _seed(SEED)
    audit = _copy_and_audit_inputs(reference_dir, output_dir, targets)
    model = HistoryOnlyRegressor()
    counts = parameter_counts(model)
    parameter_names = [name for name, _ in model.named_parameters()]
    forbidden = [
        name for name in parameter_names if "image" in name or "backbone" in name
    ]
    if forbidden:
        raise AssertionError(f"Image parameters in history-only model: {forbidden}")
    manifest = {
        "schema_version": 1,
        "experiment": "exp2_history_only_head32_regression_ablation",
        "reference_experiment": str(reference_dir),
        "targets": list(targets),
        "seed": SEED,
        "controlled_variables": {
            "task_records": "byte-identical copies from reference",
            "patient_split": "exact reference train/val/test assignment",
            "regression_target": f"{SCORE_TRANSFORM} abnormal score",
            "history_features": "byte-identical compact NPZ from reference",
            "history_encoder": "same 2->16->16 masked-mean encoder",
            "head_hidden_features": HEAD_HIDDEN_FEATURES,
            "head_dropout": 0.25,
            "loss": f"unweighted SmoothL1 beta={SMOOTH_L1_BETA}",
        },
        "ablated_components": [
            "face image input",
            "20 selected video frames",
            "five image views",
            "image resize and ImageNet normalization",
            "pretrained image backbone",
            "backbone fine-tuning stage",
        ],
        "sample_unit": "one history sequence and one prediction per labelled video",
        "model": {
            "history_encoder": (
                f"Linear({HISTORY_INPUT_FEATURES},{HISTORY_HIDDEN_FEATURES})-SiLU-"
                f"Linear({HISTORY_HIDDEN_FEATURES},{HISTORY_OUTPUT_FEATURES})-"
                f"LayerNorm({HISTORY_OUTPUT_FEATURES})-SiLU-masked_mean"
            ),
            "head": (
                f"Linear({HISTORY_OUTPUT_FEATURES},{HEAD_HIDDEN_FEATURES})-"
                f"LayerNorm({HEAD_HIDDEN_FEATURES})-SiLU-Dropout(0.25)-"
                "Linear(32,1)"
            ),
            "parameter_counts": counts,
            "parameter_names": parameter_names,
            "forbidden_image_parameters": forbidden,
        },
        "training": {
            "stages": 1,
            "reason": "all remaining parameters are task-specific and trainable",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": 2 if args.smoke_test else MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "mixed_precision": "FP16 autocast with GradScaler",
            "torch_compile": False,
        },
        "data_alignment_audit": str(
            (output_dir / "data_alignment_audit.csv").resolve()
        ),
        "data_alignment_passed": bool(audit["exact_match"].all()),
    }
    with (output_dir / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"[prepared] targets={len(targets)} parameters={counts['total']} "
        f"history_encoder={counts['history_encoder']} head={counts['head']} "
        f"exact_reference_alignment={audit['exact_match'].all()}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal ablation run")
    jobs = [
        {
            "target": target,
            "gpu_id": index % torch.cuda.device_count(),
            "seed": _job_seed(target, SEED),
            "output_dir": str(output_dir),
            "max_epochs": 2 if args.smoke_test else MAX_EPOCHS,
        }
        for index, target in enumerate(targets)
    ]
    run_rows, metric_frames, failures = [], [], []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                metrics, run_row = future.result()
                metric_frames.append(metrics)
                run_row["job_seed"] = job["seed"]
                run_rows.append(run_row)
            except Exception as exc:
                failure = {"target": job["target"], "error": str(exc)}
                failures.append(failure)
                run_rows.append(
                    {
                        "target": job["target"],
                        "status": "failed",
                        "reason": str(exc),
                        "job_seed": job["seed"],
                    }
                )
                error_dir = output_dir / "runs" / job["target"]
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / "error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                print(f"[job-failed] task={job['target']}: {exc}", flush=True)
            pd.DataFrame(run_rows).to_csv(output_dir / "run_index.csv", index=False)
            if metric_frames:
                pd.concat(metric_frames, ignore_index=True).to_csv(
                    output_dir / "metrics_all.csv", index=False
                )
            print(
                f"[scheduler] {completed}/{len(jobs)} task={job['target']}",
                flush=True,
            )
    pd.DataFrame(failures, columns=("target", "error")).to_csv(
        output_dir / "failures.csv", index=False
    )
    histories = [
        pd.read_csv(path) for path in sorted((output_dir / "runs").glob("*/history.csv"))
    ]
    pd.concat(histories, ignore_index=True).to_csv(
        output_dir / "history_all.csv", index=False
    )
    if failures:
        raise RuntimeError(f"{len(failures)} history-only jobs failed")
    print("[plot] generating history-only and baseline comparison figures", flush=True)
    from .plot_results import main as plot_results

    plot_results(output_dir, reference_dir)
    print(f"[complete] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
