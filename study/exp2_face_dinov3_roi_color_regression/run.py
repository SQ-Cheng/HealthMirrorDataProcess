"""Run ten frozen-DINO/native-color regressors and matched-frame figures."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import pandas as pd
import torch

from study.common.run_video_loss_12h import Tee, sha256
from study.exp1_roi_color_regression.preview_rois import HERE as ROI_HERE
from study.exp1_roi_color_regression.views import CACHE as ROI_CACHE
from study.exp2_face_dinov3_frozen import run_main_regression as reference
from study.exp2_face_dinov3_frozen.features import ensure_cache
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from .train import train_job


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
INDEX = DEVICE = None


def init_worker(queue):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX = FrameOffsetIndex.load(reference.INDEX_PATH)


def read_records(target):
    return pd.read_csv(ROI_HERE / f"outputs/tables/records_{target}.csv", dtype={"hospital_id": str, "video_id": str})


def worker(job):
    root = OUTPUT / f"runs/{job['target']}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "train.log").open("w", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_job(job["target"], INDEX, read_records(job["target"]), DEVICE, root, job["contract"])
    torch.cuda.empty_cache()
    return {"target": job["target"], "status": "ok"}


def validate():
    index, _, hashes, _, lab_hash = reference.preflight()
    ensure_cache(reference.INDEX_PATH, reference.CACHE)
    prepared = json.loads((ROI_HERE / "outputs/tables/preparation_manifest.json").read_text())
    if (prepared["task_records_sha256"] != hashes or prepared["frame_index_sha256"] != sha256(reference.INDEX_PATH)
            or prepared["lab_table_sha256"] != lab_hash or prepared["views"] != list(reference.config.VIEWS)
            or prepared["protocol_sha256"] != sha256(ROI_HERE / "protocol.json")):
        raise RuntimeError("ROI/DINO/main data protocols do not match")
    roi_manifest = json.loads((ROI_CACHE / "manifest.json").read_text())
    for key, value in roi_manifest.items():
        if prepared.get(key) != value:
            raise RuntimeError(f"Prepared ROI cache differs: {key}")
    params = prepared["view_parameters"]
    dino_manifest = json.loads((reference.CACHE / "manifest.json").read_text())
    for key in ("crop_scale", "brightness_delta", "contrast_delta"):
        if params[key] != dino_manifest[key]:
            raise RuntimeError(f"View parameters differ: {key}")
    comparator = json.loads((reference.OUTPUT / "experiment_manifest.json").read_text())
    if not (reference.OUTPUT / "COMPLETE").is_file() or comparator["source_records_sha256"] != hashes:
        raise RuntimeError("Completed DINO-only comparator does not match")
    for target in reference.TARGETS:
        source = reference.read_records(reference.BASELINE / f"task_records/{target}.csv")
        actual = read_records(target)
        expected = source.loc[source.video_id.isin(actual.video_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
        if actual.video_id.duplicated().any() or actual.groupby("hospital_id").split.nunique().gt(1).any():
            raise RuntimeError("Duplicated video or leaked patient")
    manifest = {"targets": list(reference.TARGETS), "source_records_sha256": hashes, "lab_table_sha256": lab_hash,
                "roi_preparation_sha256": sha256(ROI_HERE / "outputs/tables/preparation_manifest.json"),
                "frame_index_sha256": sha256(reference.INDEX_PATH), "dino_cache": dino_manifest,
                "dino_comparator_manifest_sha256": sha256(reference.OUTPUT / "experiment_manifest.json"),
                "effective_record_hashes": {target: sha256(ROI_HERE / f"outputs/tables/records_{target}.csv") for target in reference.TARGETS},
                "views": list(reference.config.VIEWS), "evaluation_view": "original", "features": [384, 164],
                "head_hidden": 32, "trainable_parameters": 17665, "encoder_frozen": True, "matching_hours": 24,
                "candidate_frames_per_video": 20, "minimum_common_valid_frames": 10,
                "loss": "SmoothL1 beta=.5; per-frame on robust-scaled raw values; equal weight per video/view",
                "color_scaling": "training-frame/view-only weighted StandardScaler; each training video equal mass",
                "target_scaling": "effective training-video-only median/IQR", "batch_distinct_labs": 12,
                "batch_frame_views": "120..240 for full batches; each selected video has one view",
                "max_epochs": 80, "patience": 12, "learning_rate": .0002, "min_lr": .000001, "weight_decay": .001,
                "optimizer": "AdamW", "scheduler": "cosine; no warmup", "dropout": .25, "gradient_clip": 1.,
                "amp": True, "compile": False, "split_search": False,
                "comparison": "DINO-only checkpoint re-evaluated on identical common-valid original frames; trained on full unfiltered 20-frame cohort",
                "center_crop": "DINO uses resized 202x202 crop; color uses the same native crop/mask pixels without spatial interpolation"}
    return index, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    index, manifest = validate()
    if args.check_only:
        print(json.dumps(manifest, indent=2))
        return
    if args.smoke:
        target = "lactate_high"
        records = read_records(target)
        subset = pd.concat([records.query("split == 'train'").drop_duplicates("clinical_event_id").head(12),
                            records.query("split == 'val'").head(2), records.query("split == 'test'").head(2)], ignore_index=True)
        with tempfile.TemporaryDirectory(prefix="dino_roi_smoke_") as name:
            root = Path(name)
            train_job(target, index, subset, torch.device("cuda:0"), root, "smoke-only", max_epochs=1)
            assert torch.load(root / "model.pt", map_location="cpu", weights_only=True)["parameters"] == 17665
        print("[smoke-ok] real five-view features, distinct-lab loss, checkpoint reload and matched-frame DINO evaluation", flush=True)
        return
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUTPUT / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing fusion results have a different contract; not overwriting")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        (OUTPUT / "COMPLETE").unlink(missing_ok=True)
        contract = sha256(path)
        (OUTPUT / "source_records").mkdir(exist_ok=True)
        shutil.copy2(ROI_HERE / "outputs/tables/cohort_counts.csv", OUTPUT / "cohort_counts.csv")
        for target in reference.TARGETS:
            shutil.copy2(ROI_HERE / f"outputs/tables/records_{target}.csv", OUTPUT / f"source_records/{target}.csv")
        jobs, rows = [], []
        for target in reference.TARGETS:
            root = OUTPUT / f"runs/{target}"
            marker = root / "job_complete.json"
            if marker.exists() and json.loads(marker.read_text()) == {"contract": contract} and all((root / name).is_file() for name in ("model.pt", "history.csv", "metrics.csv", "video_predictions.csv", "selected_frames.npz")):
                rows.append({"target": target, "status": "ok"})
            else:
                jobs.append({"target": target, "contract": contract})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        if jobs and not workers:
            raise RuntimeError("CUDA required")
        print(f"[training-start] DINO+ROI independent heads={len(jobs)} GPUs={workers} views=5", flush=True)
        if jobs:
            context = mp.get_context("spawn")
            with context.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(worker, job): job for job in jobs}
                    for future in as_completed(futures):
                        try:
                            row = future.result()
                        except Exception:
                            row = {"target": futures[future]["target"], "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(OUTPUT / "run_index.csv", index=False)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Fusion task failure; see run_index.csv")
        for field in ("metrics", "history"):
            pd.concat([pd.read_csv(OUTPUT / f"runs/{target}/{field}.csv") for target in reference.TARGETS], ignore_index=True).to_csv(OUTPUT / f"{field}_all.csv", index=False)
        if reference.preflight()[2] != manifest["source_records_sha256"]:
            raise RuntimeError("Clinical reference changed during training")
        from .plots import plot_results
        plot_results()
        (OUTPUT / "COMPLETE").write_text("Ten independent frozen DINO+ROI regressors and matched-frame figures completed\n")
        print("[queue-complete] DINO+ROI five-view regression", flush=True)


if __name__ == "__main__":
    main()
