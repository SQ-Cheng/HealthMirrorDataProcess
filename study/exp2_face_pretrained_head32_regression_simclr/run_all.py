"""Four-GPU dynamic scheduler for the three-stage SimCLR ablation."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import traceback

import pandas as pd
import torch

from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex

from .config import (
    FINETUNE_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    OUTPUT_DIR,
    REFERENCE_INDEX_PATH,
    REFERENCE_OUTPUT_DIR,
    SEED,
    SIMCLR_BACKBONE_LR,
    SIMCLR_BATCH_SIZE,
    SIMCLR_EPOCHS,
    SIMCLR_MIN_LR_RATIO,
    SIMCLR_PROJECTOR_LR,
    SIMCLR_TEMPERATURE,
    TARGETS,
)
from .plot_results import plot_results
from .train import job_seed, train_three_stage


def _load_scalers():
    payload = json.loads(
        (Path(REFERENCE_OUTPUT_DIR) / "target_scalers.json").read_text(encoding="utf-8")
    )["targets"]
    return {target: RobustTargetScaler(**payload[target]) for target in TARGETS}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(device_id, tasks, results, seed, smoke):
    torch.cuda.set_device(device_id); torch.set_num_threads(4)
    frame_index = FrameOffsetIndex.load(REFERENCE_INDEX_PATH)
    scalers = _load_scalers()
    while True:
        target = tasks.get()
        if target is None:
            return
        run_dir = Path(OUTPUT_DIR) / "runs/efficientnet_b0" / target
        try:
            records = pd.read_csv(
                Path(REFERENCE_OUTPUT_DIR) / "task_records" / f"{target}.csv",
                dtype={"hospital_id": str, "video_id": str},
            )
            if records.groupby("hospital_id").split.nunique().max() != 1:
                raise AssertionError(f"Patient leakage in reference records for {target}")
            metrics = train_three_stage(
                target, frame_index, records, scalers[target], run_dir, seed, smoke
            )
            results.put((
                target,
                "ok",
                "",
                job_seed(seed, target),
                metrics.to_dict("records"),
            ))
        except Exception:
            results.put((target, "failed", traceback.format_exc(), None, []))


def _collect_csv(filename):
    frames = []
    for target in TARGETS:
        path = Path(OUTPUT_DIR) / "runs/efficientnet_b0" / target / filename
        if path.is_file():
            frame = pd.read_csv(path); frame["target"] = target; frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate_reference_contract():
    frame_index = FrameOffsetIndex.load(REFERENCE_INDEX_PATH)
    for target in TARGETS:
        records_path = Path(REFERENCE_OUTPUT_DIR) / "task_records" / f"{target}.csv"
        if not records_path.is_file():
            raise FileNotFoundError(f"Missing reference task records: {records_path}")
        records = pd.read_csv(
            records_path, dtype={"hospital_id": str, "video_id": str}
        )
        if records.groupby("hospital_id").split.nunique().max() != 1:
            raise AssertionError(f"Patient leakage in reference records for {target}")
        missing = sorted(set(records.video_id.astype(str)) - set(frame_index.video_lookup))
        if missing:
            raise ValueError(
                f"Reference frame index is missing {len(missing)} {target} videos; "
                f"examples={missing[:5]}"
            )
        counts = {
            str(video_id): frame_index.frame_range(str(video_id))[1]
            - frame_index.frame_range(str(video_id))[0]
            for video_id in records.video_id
        }
        invalid = [video_id for video_id, count in counts.items() if count != 20]
        if invalid:
            raise ValueError(
                f"Expected exactly 20 indexed frames for {target}; "
                f"invalid examples={invalid[:5]}"
            )
    return frame_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    devices = (
        [int(item) for item in args.gpus.split(",")]
        if args.gpus else list(range(torch.cuda.device_count()))
    )
    if not devices:
        raise RuntimeError("No CUDA devices available")
    _validate_reference_contract()
    targets = ("hemoglobin_low",) if args.smoke_test else TARGETS
    output_dir = Path(OUTPUT_DIR)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if (output_dir / "run_index.csv").exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already contains a run: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    tasks, results = context.Queue(), context.Queue()
    for target in targets: tasks.put(target)
    for _ in devices[:len(targets)]: tasks.put(None)
    workers = [
        context.Process(target=_worker, args=(device, tasks, results, args.seed, args.smoke_test))
        for device in devices[:len(targets)]
    ]
    for worker in workers: worker.start()
    rows = [results.get() for _ in targets]
    for worker in workers: worker.join()
    failures = [row for row in rows if row[1] == "failed"]
    pd.DataFrame(
        [
            {
                "architecture": "efficientnet_b0",
                "target": target,
                "status": status,
                "reason": error,
                "job_seed": current_job_seed,
                "run_dir": str(output_dir / "runs/efficientnet_b0" / target),
            }
            for target, status, error, current_job_seed, _ in rows
        ]
    ).to_csv(output_dir / "run_index.csv", index=False)
    if failures:
        for target, _, error, _, _ in failures:
            print(f"[job-failed] target={target}\n{error}", flush=True)
        raise RuntimeError(f"{len(failures)} SimCLR ablation jobs failed")
    if not args.smoke_test:
        _collect_csv("metrics.csv").to_csv(output_dir / "metrics_all.csv", index=False)
        _collect_csv("history.csv").to_csv(output_dir / "history_all.csv", index=False)
        _collect_csv("simclr_history.csv").to_csv(
            output_dir / "simclr_history_all.csv", index=False
        )
        (output_dir / "experiment_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "experiment": "exp2_face_pretrained_head32_regression_simclr",
            "targets": list(TARGETS),
            "reference_experiment": str(REFERENCE_OUTPUT_DIR),
            "data_and_split": "exact reference task records",
            "reference_fingerprints": {
                "frame_index_sha256": _sha256(REFERENCE_INDEX_PATH),
                "target_scalers_sha256": _sha256(
                    Path(REFERENCE_OUTPUT_DIR) / "target_scalers.json"
                ),
                "task_records_sha256": {
                    target: _sha256(
                        Path(REFERENCE_OUTPUT_DIR) / "task_records" / f"{target}.csv"
                    )
                    for target in TARGETS
                },
            },
            "stages": ["train-split-only SimCLR", "frozen-backbone head", "full fine-tune"],
            "optimization": {
                "simclr_epochs": SIMCLR_EPOCHS,
                "simclr_video_batch_size": SIMCLR_BATCH_SIZE,
                "simclr_temperature": SIMCLR_TEMPERATURE,
                "simclr_backbone_learning_rate": SIMCLR_BACKBONE_LR,
                "simclr_projector_learning_rate": SIMCLR_PROJECTOR_LR,
                "simclr_minimum_learning_rate_ratio": SIMCLR_MIN_LR_RATIO,
                "head_learning_rate": HEAD_LEARNING_RATE,
                "full_finetune_learning_rate": FINETUNE_LEARNING_RATE,
            },
            "gpu_scheduler": "dynamic task queue with one persistent worker per GPU",
        }, indent=2), encoding="utf-8")
        plot_results(output_dir)
    print(f"[all-complete] jobs={len(targets)} smoke={args.smoke_test}", flush=True)


if __name__ == "__main__":
    main()
