"""Run the controlled postoperative-face-only partial-CASUS protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.exp4.frame_index import build_or_reuse_frame_index

from .build_dataset import add_patient_split, prepare_post_only_records, write_label_outputs
from .config import (
    CASUS_ANALYTES, CASUS_MAX_SCORE, POST_ONLY_CACHE_DIR,
    POST_ONLY_OUTPUT_DIR, SEED, TRAIN_VIEWS,
)
from .data import PostOnlyCasusDataset
from .models import build_post_only_model, freeze_backbone, parameter_counts
from .plot_post_face_only import plot_post_face_only
from .train import _loss, _prepare, seed_everything, train


PROTOCOL = "post_face_only"
RUN_DIR = POST_ONLY_OUTPUT_DIR
TRAINING_FILES = (
    "history.csv", "metrics.csv", "model.pt", "run_manifest.json",
    "training_history.png", "video_predictions.csv",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data():
    records, frame_records, manifest = prepare_post_only_records(RUN_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(
        frame_records, POST_ONLY_CACHE_DIR,
    )
    usable = set(frame_index.video_lookup)
    invalid = records.loc[~records.video_id.isin(usable)].copy()
    if len(invalid):
        invalid.to_csv(RUN_DIR / "frame_exclusions.csv", index=False)
        records = records.loc[records.video_id.isin(usable)].copy()
        records = records.drop(columns="split")
        records, split_manifest = add_patient_split(records)
        records.to_csv(RUN_DIR / "records.csv", index=False)
        write_label_outputs(records, RUN_DIR)
        manifest["split"] = split_manifest
        print(
            f"[frames] excluded_videos={len(invalid)} "
            f"final_records={len(records)} final_patients={records.hospital_id.nunique()} "
            "split_refit=true",
            flush=True,
        )
    else:
        (RUN_DIR / "frame_exclusions.csv").unlink(missing_ok=True)
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Patient leakage in post-face-only CASUS records")
    if set(records.split) != {"train", "val", "test"}:
        raise AssertionError("Post-face-only records do not contain all splits")
    component_sum = sum(records[f"{name}_casus_points"] for name in CASUS_ANALYTES)
    if not component_sum.equals(records.casus_score):
        raise AssertionError("Invalid post-face-only partial-CASUS labels")
    manifest["counts"]["frame_excluded_videos"] = len(invalid)
    manifest["counts"]["final_labelled_postoperative_videos"] = len(records)
    manifest["counts"]["final_labelled_patients"] = records.hospital_id.nunique()
    manifest["frame_index"] = frame_manifest["policy"]
    (RUN_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        records, frame_index, RUN_DIR / "records.csv",
        POST_ONLY_CACHE_DIR / "frame_offsets.npz",
    )


def smoke_test(records, frame_index, device_id):
    seed_everything(SEED)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    sample = records.loc[records.split.eq("train")].head(4).reset_index(drop=True)
    dataset = PostOnlyCasusDataset(
        frame_index, sample, TRAIN_VIEWS, expand_views=True,
    )
    post, target, _, codes, weights = next(iter(DataLoader(dataset, batch_size=4)))
    post = _prepare(post, codes, device)
    repeat = codes.shape[1]
    target = target.repeat_interleave(repeat).to(device)
    weights = weights.repeat_interleave(repeat).to(device)
    model, _ = build_post_only_model()
    freeze_backbone(model)
    model.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(post).squeeze(1)
        loss = _loss(prediction, target, weights)
    loss.backward()
    total, trainable = parameter_counts(model)
    if not torch.isfinite(loss) or prediction.shape != target.shape:
        raise RuntimeError("Invalid post-face-only smoke output")
    print(
        f"[smoke-ok] protocol={PROTOCOL} inputs={tuple(post.shape)} "
        f"target_range=0-{CASUS_MAX_SCORE:g} normalized_loss={float(loss):.6f} "
        f"parameters={total} head_trainable={trainable}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    records, frame_index, records_path, frame_index_path = prepare_data()
    if args.smoke:
        smoke_test(records, frame_index, args.device)
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in TRAINING_FILES:
        (RUN_DIR / name).unlink(missing_ok=True)
    figure_dir = RUN_DIR / "figures"
    if figure_dir.is_dir():
        for path in figure_dir.glob("*.png"):
            path.unlink()
    train(records, frame_index, args.seed, RUN_DIR, protocol=PROTOCOL)
    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "cohort": "all eligible postoperative CABG videos; preoperative video not required",
        "split": "protocol-specific patient-level balanced split",
        "records_path": str(records_path),
        "records_sha256": _sha256(records_path),
        "frame_index_path": str(frame_index_path),
        "frame_index_sha256": _sha256(frame_index_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_post_face_only(RUN_DIR)
    print("[experiment-complete] post-face-only CASUS outputs and figures generated", flush=True)


if __name__ == "__main__":
    main()
