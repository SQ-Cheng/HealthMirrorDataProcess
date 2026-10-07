"""Refresh the cohort, audit gains, and sequentially rerun the non-DINO queue."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .lab_run_version import OVERWRITE, RUN_TAG, STUDY
from . import run_video_loss_12h as reference


ROOT = STUDY.parent
STATE = STUDY / "common/outputs" / RUN_TAG
OLD = STUDY / "exp2_face_pretrained_head32_regression/outputs/ablations/lab_match_12h_face224"


def run(module, *arguments):
    print(f"[queue-stage] {module} {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT, check=True)


def copy_index(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("frame_offsets.npz", "index_manifest.json", "video_frame_summary.csv", "invalid_frames.csv"):
        old, new = source / name, destination / name
        if old.is_file() and not new.exists():
            shutil.copy2(old, new)


def event_set(records, manifest, target):
    prefix = reference.config.SCORE_DEFINITIONS[target]["value_column"].removesuffix("_value")
    rows = manifest.set_index("video_id", verify_integrity=True).loc[records.video_id]
    return set(zip(rows.hospital_id, rows[f"{prefix}_lab_time_unix"]))


def cohort_report():
    previous = pd.read_csv(OLD / "source_data/base_manifest.csv", dtype={"hospital_id": str, "video_id": str})
    current = pd.read_csv(reference.SOURCE / "source_data/base_manifest.csv", dtype={"hospital_id": str, "video_id": str})
    old_labs = pd.read_csv(OLD / "source_data/lab_timeseries.csv", dtype={"hospital_id": str})
    new_labs = pd.read_csv(reference.SOURCE / "source_data/lab_timeseries.csv", dtype={"hospital_id": str})
    rows = []
    for target in reference.config.TARGETS:
        old = pd.read_csv(OLD / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
        new = pd.read_csv(reference.SOURCE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
        old_videos, new_videos = set(old.video_id), set(new.video_id)
        old_patients, new_patients = set(old.hospital_id), set(new.hospital_id)
        old_events, new_events = event_set(old, previous, target), event_set(new, current, target)
        shared = old.merge(new, on=["video_id", "hospital_id"], suffixes=("_old", "_new"), validate="one_to_one")
        analyte = reference.config.SCORE_DEFINITIONS[target]["value_column"].removesuffix("_value")
        old_analyte = old_labs.loc[old_labs.analyte.eq(analyte)]
        new_analyte = new_labs.loc[new_labs.analyte.eq(analyte)]
        old_lab_events = set(zip(old_analyte.hospital_id, old_analyte.timestamp_unix))
        new_lab_events = set(zip(new_analyte.hospital_id, new_analyte.timestamp_unix))
        row = {
            "target": target, "old_videos": len(old), "new_videos": len(new),
            "net_added_videos": len(new)-len(old), "newly_usable_videos": len(new_videos-old_videos),
            "lost_videos": len(old_videos-new_videos),
            "old_patients": len(old_patients), "new_patients": len(new_patients),
            "net_added_patients": len(new_patients)-len(old_patients),
            "newly_usable_patients": len(new_patients-old_patients),
            "old_matched_lab_events": len(old_events), "new_matched_lab_events": len(new_events),
            "newly_matched_lab_events": len(new_events-old_events),
            "old_positive_videos": int(old.binary_label.sum()), "new_positive_videos": int(new.binary_label.sum()),
            "shared_videos_changed_value": int((~np.isclose(shared.raw_value_old,shared.raw_value_new,rtol=0,atol=1e-9)).sum()),
            "shared_videos_changed_binary_label": int(shared.binary_label_old.ne(shared.binary_label_new).sum()),
            "old_valid_lab_events": len(old_lab_events), "new_valid_lab_events": len(new_lab_events),
            "newly_valid_lab_events": len(new_lab_events-old_lab_events),
        }
        for split in ("train", "val", "test"):
            selected = new.loc[new.split.eq(split)]
            row[f"new_{split}_videos"] = len(selected)
            row[f"new_{split}_patients"] = selected.hospital_id.nunique()
        rows.append(row)
    report = pd.DataFrame(rows)
    report.to_csv(STATE / "cohort_update.csv", index=False)
    from study.exp4.config import OUTPUT_DIR
    previous_exp4 = STATE / "previous_exp4_records.csv"
    old = pd.read_csv(previous_exp4 if previous_exp4.exists() else STUDY / "exp4/outputs/records.csv", dtype={"hospital_id": str})
    new = pd.read_csv(OUTPUT_DIR / "records.csv", dtype={"hospital_id": str})
    exp4 = {"old_videos": len(old), "new_videos": len(new), "net_added_videos": len(new)-len(old),
            "old_patients": old.hospital_id.nunique(), "new_patients": new.hospital_id.nunique(),
            "net_added_patients": new.hospital_id.nunique()-old.hospital_id.nunique(),
            "newly_usable_videos": len(set(new.video_id)-set(old.video_id)),
            "lost_videos": len(set(old.video_id)-set(new.video_id))}
    pd.DataFrame([exp4]).to_csv(STATE / "exp4_cohort_update.csv", index=False)
    provenance = {
        "run_tag": RUN_TAG,
        "old_cohort": str(OLD),
        "new_shared_cohort": str(reference.SOURCE),
        "lab_table": str(ROOT / "merged_lab_tests.csv"),
        "lab_table_sha256": reference.sha256(ROOT / "merged_lab_tests.csv"),
        "frame_index": str(reference.INDEX_PATH),
        "frame_index_sha256": reference.sha256(reference.INDEX_PATH),
        "matching_hours": 12,
        "frames_per_video": 20,
        "training_views": list(reference.config.VIEW_NAMES),
        "data_gain_unit": "eligible video-target pairs after clinical and native224 frame checks; not raw CSV rows",
        "split_policy": "new 512-candidate patient-disjoint distribution search; exact reuse across all Exp2 groups",
        "source_alias_addition": "*total bilirubin Chinese field added; missing units remain excluded",
        "old_new_performance_comparison": "different cohorts and splits; not a controlled performance comparison",
        "overwrite_previous_results": OVERWRITE,
        "models": {"chunked_view_loss": 16, "distinct_lab_views": 16, "architecture_controls": 48, "exp4": 1},
        "dino": "excluded; no monitor or training launched",
        "outputs": {family: str(path) for family, path in reference.OUTPUTS.items()},
        "exp4_output": str(OUTPUT_DIR),
    }
    (STATE / "update_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(report[["target","old_videos","new_videos","net_added_videos","old_patients","new_patients","net_added_patients"]].to_string(index=False), flush=True)
    print(f"[exp4-cohort-update] {exp4}", flush=True)
    return report


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    lab_sha = reference.sha256(ROOT / "merged_lab_tests.csv")
    if (reference.SOURCE / "DATA_COMPLETE").exists():
        path = reference.SOURCE / "experiment_manifest.json"
        manifest = json.loads(path.read_text())
        if "frame_index_sha256" not in manifest:
            manifest["frame_index_sha256"] = reference.sha256(reference.INDEX_PATH)
            path.write_text(json.dumps(manifest, indent=2)+"\n")
        reference.preflight()
        return cohort_report()
    copy_index(STUDY / "common/cache/face224_20frame", reference.INDEX_PATH.parent)
    run("study.exp2_face_pretrained_head32_regression.run_all", "--prepare-only",
        "--frame-policy", "20frame", "--match-max-delta-hours", "12",
        "--output-dir", str(reference.SOURCE), "--index-dir", str(reference.INDEX_PATH.parent),
        "--reference-output-dir", "")
    if reference.sha256(ROOT / "merged_lab_tests.csv") != lab_sha:
        raise RuntimeError("Lab table changed during cohort construction")
    path = reference.SOURCE / "experiment_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["lab_table_sha256"] = lab_sha
    manifest["frame_index_sha256"] = reference.sha256(reference.INDEX_PATH)
    manifest["role"] = "data-only shared reference; not a trained frame-loss baseline"
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    from study.exp4.config import CACHE_DIR
    copy_index(STUDY / "exp4/cache/frames20", CACHE_DIR / "frames20")
    run("study.exp4.run_all", "--prepare-only")
    if reference.sha256(ROOT / "merged_lab_tests.csv") != lab_sha:
        raise RuntimeError("Lab table changed during Exp4 construction")
    (reference.SOURCE / "DATA_COMPLETE").write_text("fresh cohort, splits, train-only scalers and native224 index\n")
    reference.preflight()
    from .run_distinct_lab_views_12h import prepared_records
    prepared_records()
    return cohort_report()


def activate_overwrite():
    if not OVERWRITE or (STATE / "OVERWRITE_READY").exists():
        return
    if not (STATE / "cohort_update.csv").exists():
        raise RuntimeError("The gain report must exist before removing old results")
    exp4 = STUDY / "exp4/outputs"
    prepared = exp4 / RUN_TAG
    if not (prepared / "records.csv").exists():
        raise RuntimeError("Updated Exp4 cohort has not been prepared")
    shutil.copy2(exp4 / "records.csv", STATE / "previous_exp4_records.csv")
    temporary = STATE / "prepared_exp4"
    shutil.move(str(prepared), temporary)
    from .run_distinct_lab_views_12h import OUTPUTS, STATE as distinct_state
    from study.exp2_face_architecture_ablation.config import OUTPUT_DIR
    roots = [*reference.OUTPUTS.values(), *OUTPUTS.values(), reference.STATE,
             distinct_state, OUTPUT_DIR, exp4]
    for root in roots:
        if root.is_dir():
            print(f"[overwrite] remove previous results: {root}", flush=True)
            shutil.rmtree(root)
    shutil.move(str(temporary), exp4)
    (STATE / "OVERWRITE_READY").write_text("old result trees removed; refreshed Exp4 preparation installed\n")
    cohort_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not RUN_TAG:
        parser.error("Set HEALTHMIRROR_LAB_RUN_TAG to a new lab_update_YYYYMMDD namespace")
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
            return
        reference.preflight()
        if not (STATE / "cohort_update.csv").is_file():
            raise RuntimeError("Prepare and report the data changes before training")
        activate_overwrite()
        for module in ("study.common.run_video_loss_12h", "study.common.run_distinct_lab_views_12h",
                       "study.exp2_face_architecture_ablation.run", "study.exp4.run_all"):
            reference.preflight()
            run(module, *(["--reuse-prepared"] if module == "study.exp4.run_all" else []))
        (STATE / "COMPLETE").write_text("80 Exp2 models and one Exp4 model; figures completed; DINO excluded\n")
        print("[queue-complete] refreshed-lab experiments and figures", flush=True)


if __name__ == "__main__":
    main()
