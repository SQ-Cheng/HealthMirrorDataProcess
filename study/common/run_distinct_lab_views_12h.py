"""Wait for current view-loss jobs, then train twelve-distinct-lab batches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import io
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import run_video_loss_12h as base
from .video_loss import DistinctLabViewBatchSampler


PREDECESSOR = base.STATE
BASELINES = dict(base.OUTPUTS)
STATE = base.STUDY / "common/outputs/view_loss_12_distinct_labs_12h"
OUTPUTS = {family: root.with_name("view_loss_12_distinct_labs") for family, root in BASELINES.items()}


def prepared_records():
    raw = pd.read_csv(base.SOURCE / "source_data/base_manifest.csv",
                      dtype={"hospital_id": str, "video_id": str}).set_index("video_id", verify_integrity=True)
    texts, audit = {}, []
    for target in base.config.TARGETS:
        path = base.SOURCE / f"task_records/{target}.csv"
        records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        column = target.removesuffix("_high").removesuffix("_low") + "_lab_time_unix"
        matched = raw.loc[records.video_id].reset_index()
        if (not np.array_equal(matched.hospital_id, records.hospital_id)
                or not np.isfinite(matched[column]).all()):
            raise ValueError(f"Invalid matched lab identity/time: {target}")
        events = records.hospital_id + "@" + matched[column].map(lambda x: format(x, ".17g"))
        records["clinical_event_id"] = events
        if records.groupby("clinical_event_id")[["raw_value", "binary_label"]].nunique().gt(1).any().any():
            raise ValueError(f"The same lab event has inconsistent labels: {target}")
        mapping = dict(zip(records.video_id, events))
        # Preserve original CSV numeric strings so identity checks stay bit-exact.
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames + ["clinical_event_id"], lineterminator="\n")
            writer.writeheader()
            for row in reader:
                writer.writerow({**row, "clinical_event_id": mapping[row["video_id"]]})
        texts[target] = output.getvalue()
        for split, group in records.groupby("split"):
            if split != "train":
                continue
            group = group.reset_index(drop=True)
            dataset = SimpleNamespace(expand_all_views=False, video_records=group,
                                      frame_video_rows=np.repeat(np.arange(len(group)), 20),
                                      views=base.config.VIEW_NAMES)
            torch.manual_seed(base.job_seed("regression", target))
            sampler = DistinctLabViewBatchSampler(dataset, 240)
            batches = list(sampler)
            order = np.concatenate(batches)
            if not np.array_equal(np.sort(order), np.arange(len(group) * 100)) or len(batches) != len(sampler):
                raise AssertionError(f"Dropped/duplicated frames or views: {target}")
            full, short = 0, 0
            for indices in batches:
                groups = np.asarray(indices).reshape(-1, 20)
                video_rows = groups[:, 0] // 100
                if (len(set(video_rows)) != len(video_rows)
                        or group.iloc[video_rows].clinical_event_id.nunique() != len(video_rows)):
                    raise AssertionError(f"A video or lab event repeats in one batch: {target}")
                for frames in groups:
                    if len(set(frames // 100)) != 1 or len(set(frames % 5)) != 1:
                        raise AssertionError("A loss group mixes video IDs or views")
                full += len(indices) == 240
                short += len(indices) < 240
            if short > 1 or any(len(batch) != 240 for batch in batches[:-1]):
                raise AssertionError(f"Unexpected underfilled batches in this cohort: {target}")
            audit.append({"target": target, "train_videos": len(group),
                          "train_lab_events": group.clinical_event_id.nunique(),
                          "full_240_frame_batches": full, "partial_final_batches": short,
                          "epoch_inputs": len(order), "expected_inputs": len(group) * 100,
                          "full_batch_distinct_lab_events": 12, "views_per_video": 5,
                          "views_per_video_per_batch": 1})
    print("[batch-audit-ok] all eight real cohorts: 12 distinct labs per full batch, full frame/view coverage", flush=True)
    return texts, pd.DataFrame(audit)


def predecessor_complete():
    with (PREDECESSOR / ".queue.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("The current view-loss queue stopped before completion; new jobs will not start")
    for root in BASELINES.values():
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete paired baseline: {root}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    _, scalers, hashes, counts = base.preflight()
    event_hash = base.sha256(base.SOURCE / "source_data/base_manifest.csv")
    texts, audit = prepared_records()
    if args.check_only:
        print(audit.to_string(index=False))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit.to_csv(STATE / "batch_audit.csv", index=False)
        while not predecessor_complete():
            print("[waiting] current classification/regression view-loss queue active; no GPU allocated", flush=True)
            time.sleep(60)
        if (base.preflight()[2] != hashes
                or base.sha256(base.SOURCE / "source_data/base_manifest.csv") != event_hash):
            raise RuntimeError("Clinical source changed while waiting")
        base.OUTPUTS = OUTPUTS
        contracts = base.prepare(scalers, hashes, counts, prepared_records=texts)
        jobs, rows = [], []
        for target in base.config.TARGETS:
            for family, root in OUTPUTS.items():
                marker = root / f"runs/efficientnet_b0/{target}/job_complete.json"
                expected = {"contract": contracts[family], "seed": base.job_seed(family, target)}
                if marker.is_file() and json.loads(marker.read_text()) == expected and all(
                    (marker.parent / name).is_file() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")
                ):
                    rows.append({"family": family, "target": target, "architecture": "efficientnet_b0", "status": "ok"})
                else:
                    jobs.append({"family": family, "target": target, "output": str(root),
                                 "batch_policy": "distinct_lab_views", "scaler": scalers[target], "contract": contracts[family]})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        if jobs and workers < 1:
            raise RuntimeError("CUDA is required")
        print(f"[scheduler] 12-distinct-lab view loss pending={len(jobs)} gpus={workers}", flush=True)
        ctx = mp.get_context("spawn")
        if jobs:
            with ctx.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=base.init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(base.train_one, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            row = future.result()
                        except Exception:
                            row = {"family": job["family"], "target": job["target"], "architecture": "efficientnet_b0",
                                   "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame([r for r in rows if r["family"] == job["family"]]).to_csv(OUTPUTS[job["family"]] / "run_index.csv", index=False)
                        print(f"[task-finished] {job['family']}/{job['target']} {row['status']}", flush=True)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Distinct-lab view-loss jobs failed")
        for family, root in OUTPUTS.items():
            pd.DataFrame([r for r in rows if r["family"] == family]).to_csv(root / "run_index.csv", index=False)
            base.finalize(family)
            try:
                if family == "classification":
                    from study.exp2_face_pretrained_head32_classification.plot_patient_diverse_schedule_comparison import plot_comparison
                    plot_comparison(BASELINES[family], root, base.config.TARGETS, candidate_label="12 distinct labs")
                else:
                    from study.exp2_face_pretrained_head32_regression.plot_patient_diverse_comparison import plot_comparison
                    plot_comparison(BASELINES[family], root, reference_label="Grouped views",
                                    candidate_label="12 distinct labs", figure_prefix="batch_composition")
            except Exception:
                (root / "COMPLETE").unlink(missing_ok=True)
                raise
            audit.to_csv(root / "batch_audit.csv", index=False)
        (STATE / "COMPLETE").write_text("both 12-distinct-lab view-loss experiments and comparisons completed\n")
        print("[queue-complete] 12-distinct-lab view-loss classification/regression", flush=True)


if __name__ == "__main__":
    main()
