"""Reuse validated Exp2 labels and splits while indexing continuous video clips."""

import hashlib
import json

import numpy as np
import pandas as pd

from study.exp2_face_pretrained_head32_regression.run_ablations import _prepare_sources
from study.exp2_face_pretrained_head32_regression.scaling import (
    fit_robust_target_scaler, write_target_scalers,
)

from .clips import build_or_reuse_index
from .config import INDEX_DIR, OUTPUT_DIR, TARGETS, WEIGHT_PATH


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    source_paths, source_scalers, _, source_hashes = _prepare_sources()
    sources = {
        target: pd.read_csv(
            path, dtype={"hospital_id": str, "video_id": str}
        ) for target, path in source_paths.items()
    }
    all_videos = pd.concat(
        [frame[["video_id", "mirror", "lab_patient_id"]] for frame in sources.values()],
        ignore_index=True,
    )
    index = build_or_reuse_index(all_videos, INDEX_DIR)
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(WEIGHT_PATH)
    index_hash = _sha256(INDEX_DIR / "clip_offsets.npz")
    source_contract = {
        "exp2_source_sha256": source_hashes,
        "clip_index_sha256": index_hash,
        "r3d18_weight_sha256": _sha256(WEIGHT_PATH),
    }
    manifest_path = OUTPUT_DIR / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_contract"] != source_contract:
            raise RuntimeError("Prepared Exp3 data has a different source contract")
        if not all((OUTPUT_DIR / "task_records" / f"{target}.csv").is_file()
                   for target in TARGETS):
            raise RuntimeError("Prepared Exp3 task records are incomplete")
        print(f"[prepare-reused] directory={OUTPUT_DIR}", flush=True)
        return index
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"Partial or unrelated Exp3 output exists: {OUTPUT_DIR}")

    record_dir = OUTPUT_DIR / "task_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    scalers, summary = {}, []
    for target in TARGETS:
        source = sources[target]
        if not np.isfinite(source.raw_value).all():
            raise ValueError(f"Nonfinite labels: {target}")
        if source.match_delta_h.gt(24.0 + 1e-6).any():
            raise ValueError(f"Video/laboratory mismatch exceeds 24 hours: {target}")
        records = source.loc[source.video_id.astype(str).isin(index.lookup)].copy()
        if records.groupby("hospital_id").split.nunique().max() != 1:
            raise AssertionError(f"Patient leakage after clip filtering: {target}")
        if set(records.split) != {"train", "val", "test"}:
            raise ValueError(f"Missing data split after clip filtering: {target}")
        unit = source_scalers[target]["unit"]
        scaler = fit_robust_target_scaler(target, records, unit)
        records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
        records.to_csv(record_dir / f"{target}.csv", index=False)
        scalers[target] = scaler
        for split in ("train", "val", "test"):
            subset = records.loc[records.split.eq(split)]
            summary.append({
                "target": target, "split": split,
                "source_videos": int(source.split.eq(split).sum()),
                "usable_videos": int(len(subset)),
                "excluded_no_contiguous_clip": int(
                    source.split.eq(split).sum() - len(subset)
                ),
                "patients": int(subset.hospital_id.nunique()),
                "clips": int(sum(index.clip_range(video_id)[1] - index.clip_range(video_id)[0]
                                 for video_id in subset.video_id)),
            })
    write_target_scalers(scalers, OUTPUT_DIR / "target_scalers.json")
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "data_summary.csv", index=False)
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "experiment": "exp3_video_lab_regression",
        "source_contract": source_contract,
        "data_policy": "Exp2 raw-value nearest-laboratory video records, within 24 h",
        "split_policy": "reused patient-disjoint Exp2 per-target split",
        "label": "raw laboratory value robust-scaled by usable training videos only",
        "clip_policy": "up to three disjoint, source-contiguous 16-frame windows per video",
        "eval_policy": "mean clip predictions per video, then inverse-scale and score",
        "targets": list(TARGETS),
    }, indent=2), encoding="utf-8")
    print(f"[prepare-complete] targets={len(TARGETS)} "
          f"indexed_videos={len(index.video_ids)} indexed_clips={len(index.starts)} "
          f"directory={OUTPUT_DIR}", flush=True)
    return index
