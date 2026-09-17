"""Prepare, validate, train, and summarize Exp5."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from study.exp4.frame_index import FrameOffsetIndex, build_or_reuse_frame_index

from .build_dataset import (
    add_patient_split,
    fit_trajectories_and_score,
    load_analytes,
    load_cabg_episodes,
    prepare_records,
)
from .config import CACHE_DIR, OUTPUT_DIR, SEED
from .data import PairedFrameDataset
from .models import build_model, freeze_backbone
from .plot_results import plot_results
from .train import _loss, _prepare, train


TRAINING_FILES = (
    "history.csv", "metrics.csv", "model.pt", "video_predictions.csv",
    "training_history.png", "run_manifest.json",
)


def prepare():
    records, frame_records, manifest = prepare_records(OUTPUT_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(frame_records, CACHE_DIR)
    usable = set(frame_index.video_lookup)
    invalid = records[
        ~records.video_id.isin(usable) | ~records.pre_video_id.isin(usable)
    ].copy()
    if len(invalid):
        invalid["post_frame_usable"] = invalid.video_id.isin(usable)
        invalid["pre_frame_usable"] = invalid.pre_video_id.isin(usable)
        invalid.to_csv(OUTPUT_DIR / "frame_pair_exclusions.csv", index=False)
        records = records[
            records.video_id.isin(usable) & records.pre_video_id.isin(usable)
        ].copy()
        generated = {"split", "recovery_score"}
        generated.update(column for column in records if column.endswith("_recovery_component"))
        records = records.drop(columns=[column for column in generated if column in records])
        records, split_manifest = add_patient_split(records)
        episodes, _ = load_cabg_episodes()
        labs, _ = load_analytes(episodes)
        records, trajectories = fit_trajectories_and_score(records, labs)
        records.to_csv(OUTPUT_DIR / "records.csv", index=False)
        trajectories.to_csv(OUTPUT_DIR / "recovery_trajectories.csv", index=False)
        split_rows = []
        for split, group in records.groupby("split"):
            split_rows.append({
                "split": split, "videos": len(group),
                "patients": group.hospital_id.nunique(),
                "score_mean": group.recovery_score.mean(),
                "score_std": group.recovery_score.std(),
                "score_q10": group.recovery_score.quantile(.1),
                "score_median": group.recovery_score.median(),
                "score_q90": group.recovery_score.quantile(.9),
            })
        pd.DataFrame(split_rows).to_csv(
            OUTPUT_DIR / "split_distribution.csv", index=False
        )
        manifest["split"] = split_manifest
        manifest["counts"]["frame_excluded_pairs"] = len(invalid)
        manifest["counts"]["final_labelled_postoperative_videos"] = len(records)
        manifest["counts"]["final_labelled_patients"] = records.hospital_id.nunique()
        print(
            f"[frames] excluded_pairs={len(invalid)} final_records={len(records)} "
            f"final_patients={records.hospital_id.nunique()} labels_and_split_refit=true",
            flush=True,
        )
    else:
        (OUTPUT_DIR / "frame_pair_exclusions.csv").unlink(missing_ok=True)
        manifest["counts"]["frame_excluded_pairs"] = 0
        manifest["counts"]["final_labelled_postoperative_videos"] = len(records)
        manifest["counts"]["final_labelled_patients"] = records.hospital_id.nunique()
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Patient leakage after frame validation")
    if not records.recovery_score.between(0, 1).all():
        raise AssertionError("Invalid recovery scores")
    component_columns = [column for column in records if column.endswith("_recovery_component")]
    if len(component_columns) != 5 or not torch.allclose(
        torch.tensor(records[component_columns].mean(axis=1).to_numpy()),
        torch.tensor(records.recovery_score.to_numpy()), atol=1e-7,
    ):
        raise AssertionError("Recovery score is not the equal-weight mean of five components")
    manifest["frame_index"] = frame_manifest["policy"]
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, frame_index


def smoke_test(records, frame_index, device_id):
    torch.cuda.set_device(device_id); device = torch.device(f"cuda:{device_id}")
    sample = records[records.split.eq("train")].head(4).reset_index(drop=True)
    dataset = PairedFrameDataset(frame_index, sample)
    pre, post, target, _, codes, weights = zip(*(dataset[index] for index in range(4)))
    pre = _prepare(torch.stack(pre), torch.stack(codes), device)
    post = _prepare(torch.stack(post), torch.stack(codes), device)
    target = torch.stack(target).to(device); weights = torch.stack(weights).to(device)
    model, _ = build_model(); freeze_backbone(model); model = model.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(pre, post).squeeze(1); loss = _loss(prediction, target, weights)
    loss.backward()
    if not torch.isfinite(loss) or not (
        prediction.detach().ge(0).all() and prediction.detach().le(1).all()
    ):
        raise RuntimeError("Smoke test produced invalid output")
    print(f"[smoke-ok] pairs={len(pre)} shape={tuple(pre.shape)} loss={float(loss):.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records, frame_index = prepare()
    if args.prepare_only: return
    if args.smoke:
        smoke_test(records, frame_index, args.device); return
    for name in TRAINING_FILES:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    train(records, frame_index, args.seed, args.device, OUTPUT_DIR)
    plot_results(OUTPUT_DIR)
    print("[experiment-complete] checkpoint, histories, predictions, and figures generated", flush=True)


if __name__ == "__main__":
    main()
