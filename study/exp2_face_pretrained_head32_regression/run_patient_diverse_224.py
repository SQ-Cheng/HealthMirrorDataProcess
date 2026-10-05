"""Wait for the native-224 queue, then reproduce the patient-diverse schedule."""

import argparse
import fcntl
import json
import os
import time

import numpy as np
import pandas as pd

from study.common.rerun_face224 import INDEX_DIR, STATE as QUEUE_STATE, sha256

from .config import TARGETS, VIEW_NAMES, WEIGHTS_DIR
from .models import WEIGHT_FILES
from .run_ablations import (
    ABLATION_DIR, BASE_DIR, _job_seed, _prepare_sources, _run_variant,
)
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG


VARIANT = "patient_diverse_schedule_30_40_face224"
LEGACY = ABLATION_DIR / "patient_diverse_schedule_30_40"
NATIVE = BASE_DIR.parent / "20frame_face224"
OUTPUT = ABLATION_DIR / VARIANT
STATE = QUEUE_STATE.parent / VARIANT


def validate_legacy():
    paths, _, _, source_hashes = _prepare_sources()
    manifest = json.loads((LEGACY / "experiment_manifest.json").read_text())
    index = pd.read_csv(LEGACY / "run_index.csv")
    if (not (LEGACY / "COMPLETE").is_file() or len(index) != len(TARGETS)
            or set(index.target) != set(TARGETS) or not index.status.eq("ok").all()):
        raise RuntimeError("Legacy patient-diverse schedule must be complete")
    if (manifest["architecture"] != "efficientnet_b0"
            or manifest["training_protocol"] != "two_stage_full"
            or manifest["train_batch_policy"] != "patient_diverse"
            or manifest["head_hidden_features"] != 32
            or manifest["pretrained_weight_sha256"] != sha256(os.path.join(WEIGHTS_DIR, WEIGHT_FILES["efficientnet_b0"]))
            or manifest["training_views"] != list(VIEW_NAMES)
            or manifest["baseline_source_sha256"] != source_hashes
            or manifest["job_seeds"] != {target: _job_seed(target) for target in TARGETS}
            or any(manifest[key] != value for key, value in STAGE_CONFIG.items())):
        raise RuntimeError("Legacy data, seeds or training settings differ from this protocol")
    fingerprints = {str(LEGACY / name): sha256(LEGACY / name) for name in
                    ("experiment_manifest.json", "run_index.csv", "metrics_all.csv", "history_all.csv")}
    for target in TARGETS:
        prediction = LEGACY / f"runs/efficientnet_b0/{target}/video_predictions.csv"
        fingerprints[str(prediction)] = sha256(prediction)
        fingerprints[str(paths[target])] = sha256(paths[target])
    return manifest, fingerprints


def queue_finished():
    if not (QUEUE_STATE / ".queue.lock").is_file():
        raise RuntimeError("The native-224 rerun queue has not been registered")
    with (QUEUE_STATE / ".queue.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (QUEUE_STATE / "COMPLETE").is_file():
            raise RuntimeError("The preceding queue stopped before completion; refusing to launch")
    return True


def native_sources():
    from .frame_index import FrameOffsetIndex, _index_is_reusable
    from .scaling import RobustTargetScaler

    if not (NATIVE / "COMPLETE").is_file():
        raise RuntimeError("The native-224 main regression is incomplete")
    run_index = pd.read_csv(NATIVE / "run_index.csv")
    if (len(run_index) != len(TARGETS) or set(run_index.target) != set(TARGETS)
            or not run_index.status.eq("ok").all()):
        raise RuntimeError("Native-224 main regression has failed or missing tasks")
    index_path = INDEX_DIR / "frame_offsets.npz"
    manifest = json.loads((NATIVE / "experiment_manifest.json").read_text())
    if (manifest["frame_index_sha256"] != sha256(index_path)
            or manifest["frame_index"] != str(index_path)):
        raise RuntimeError("The native main run uses a different frame index")
    index = FrameOffsetIndex.load(index_path)
    if set(index.video_formats) != {"ffv1"} or not _index_is_reusable(INDEX_DIR, index.video_ids, "20frame"):
        raise RuntimeError("The shared native FFV1 index is stale or has legacy inputs")
    scalers = json.loads((NATIVE / "target_scalers.json").read_text())["targets"]
    paths = {}
    for target in TARGETS:
        path = NATIVE / f"task_records/{target}.csv"
        records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        original = pd.read_csv(BASE_DIR / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
        if (records.video_id.duplicated().any() or set(records.split) != {"train", "val", "test"}
                or records.groupby("hospital_id").split.nunique().max() != 1):
            raise RuntimeError(f"Invalid native patient split: {target}")
        saved = original.set_index("video_id").loc[records.video_id].reset_index()
        identities = ["hospital_id", "video_id", "split", "binary_label", "source_sample_id"]
        pd.testing.assert_frame_equal(records[identities], saved[identities], check_dtype=False)
        np.testing.assert_allclose(records.raw_value, saved.raw_value, rtol=0, atol=1e-10)
        np.testing.assert_allclose(RobustTargetScaler(**scalers[target]).transform(records.raw_value),
                                   records.robust_scaled_raw_value, rtol=0, atol=1e-10)
        if any(index.frame_range(video_id)[1] - index.frame_range(video_id)[0] != 20 for video_id in records.video_id):
            raise RuntimeError(f"Non-20-frame native video: {target}")
        paths[target] = path
    source_hashes = {
        "run_index": sha256(NATIVE / "run_index.csv"),
        "target_scalers": sha256(NATIVE / "target_scalers.json"),
        "frame_index": sha256(index_path),
        "task_records": {target: sha256(path) for target, path in paths.items()},
    }
    return paths, scalers, index_path, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    os.environ["HEALTHMIRROR_FACE_SOURCE"] = "face224"
    manifest, frozen = validate_legacy()
    schedule = {
        **STAGE_CONFIG,
        "head_min_learning_rate": manifest["head_min_learning_rate"],
        "head_warmup_epochs": manifest.get("head_warmup_epochs", 0),
        "finetune_warmup_epochs": manifest.get("finetune_warmup_epochs", 0),
    }
    if args.check_only:
        paths, _, index, _ = native_sources()
        print(f"[preflight-ok] targets={len(paths)} native_index={index} schedule={STAGE_CONFIG}", flush=True)
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (STATE / "plan.json").write_text(json.dumps({
            "wait_for": str(QUEUE_STATE / "COMPLETE"), "output": str(OUTPUT),
            "baseline": str(LEGACY), "native_source": str(NATIVE),
            "stage_config": schedule, "legacy_sha256": frozen,
        }, indent=2))
        while not queue_finished():
            print("[waiting] existing face224 experiment queue is active; no GPU allocated", flush=True)
            time.sleep(args.poll_seconds)
        if validate_legacy()[1] != frozen:
            raise RuntimeError("Legacy reference changed while waiting")
        paths, scalers, index, hashes = native_sources()
        if not (OUTPUT / "COMPLETE").is_file():
            _run_variant(
                VARIANT, "efficientnet_b0", "two_stage_full", paths, scalers,
                index, hashes, train_batch_policy="patient_diverse",
                stage_config=schedule, weight_decay=manifest.get("weight_decay", 1e-4),
                freeze_batchnorm_stats=manifest.get("batchnorm_running_stats_frozen", False),
                baseline_dir=NATIVE,
            )
        current = json.loads((OUTPUT / "experiment_manifest.json").read_text())
        if (current["baseline_source_sha256"] != hashes
                or current["train_batch_policy"] != "patient_diverse"
                or any(current[key] != value for key, value in schedule.items())):
            raise RuntimeError("Completed ablation uses a different source or schedule")
        from study.common.plot_face224_comparison import plot_comparison as compare_resolution
        compare_resolution({"key": VARIANT, "family": "regression", "baseline": str(LEGACY), "output": str(OUTPUT)})
        from .plot_patient_diverse_comparison import plot_comparison
        plot_comparison(NATIVE, OUTPUT, reference_label="Native 224 main / original batches and schedule",
                        candidate_label="Native 224 patient-diverse / 30+40 epochs",
                        figure_prefix="native224_main")
        if validate_legacy()[1] != frozen or native_sources()[3] != hashes:
            raise RuntimeError("A comparison reference changed during the ablation")
        (STATE / "COMPLETE").write_text("training and both comparisons completed\n")
        print(f"[complete] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
