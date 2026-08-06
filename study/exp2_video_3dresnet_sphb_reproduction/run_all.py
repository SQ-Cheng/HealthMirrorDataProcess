"""Prepare and optionally train the paper residual 3D CNN reproduction."""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.frame_index import (
    build_or_reuse_frame_index,
)

from .config import (
    ALLFRAME_INDEX_DIR,
    EARLY_STOPPING_VAL_MSE,
    EFFECTIVE_BATCH_SIZE,
    FRAMES_PER_CLIP,
    GRADIENT_ACCUMULATION_STEPS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    OUTPUT_DIR,
    PAPER_DOI,
    PAPER_MODEL,
    REFERENCE_OUTPUT_DIR,
    SEED,
    TRAIN_MICRO_BATCH_SIZE,
)
from .data import prepare_records, sha256, validate_index, write_sampling_audit
from .models import PaperResidual3DRegressor, parameter_count
from .plot_results import main as plot_results
from .train import train


def _split_hash(records):
    rows = records[["hospital_id", "video_id", "split"]].sort_values(
        ["hospital_id", "video_id"]
    )
    payload = rows.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _model_smoke_test():
    model = PaperResidual3DRegressor().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 8, 32, 32))
    if output.shape != (1,) or not torch.isfinite(output).all():
        raise RuntimeError(f"3D model smoke test failed: shape={tuple(output.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--reference-output-dir", default=REFERENCE_OUTPUT_DIR)
    parser.add_argument("--index-dir", default=ALLFRAME_INDEX_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.output_dir, "run_index.csv")) and not args.overwrite:
        raise FileExistsError(f"Existing run in {args.output_dir}; use --overwrite explicitly")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records, reference_path, records_path = prepare_records(
        args.reference_output_dir, args.output_dir
    )
    build_or_reuse_frame_index(records, args.index_dir, frame_policy="allframes")
    index, index_path, index_manifest_path = validate_index(records, args.index_dir)
    audit = write_sampling_audit(records, index, args.output_dir)
    _model_smoke_test()
    split_counts = records.groupby("split").size().astype(int).to_dict()
    patient_counts = records.groupby("split")["hospital_id"].nunique().astype(int).to_dict()
    repeated = audit.loc[audit["repeated_positions"].gt(0)]
    manifest = {
        "schema_version": 1,
        "experiment": "biosensors_2025_15_485_residual_3d_cnn_reproduction",
        "paper": {
            "doi": PAPER_DOI,
            "url": "https://www.mdpi.com/2079-6374/15/8/485",
            "reproduced_model": PAPER_MODEL,
            "architecture_source": "Section 3.2.3 and Figure 5c",
            "training_source": "Section 3.2.3 and Table 2",
        },
        "target": {
            "name": "hemoglobin",
            "training_unit": "g/dL",
            "source_unit": "g/L",
            "conversion": "g/dL = g/L / 10",
            "one_nearest_lab_per_video": True,
        },
        "split": {
            "policy": "exact reuse of current patient-disjoint Exp2 split",
            "reference_task_records": os.path.abspath(reference_path),
            "reference_sha256": sha256(reference_path),
            "materialized_sha256": sha256(records_path),
            "assignment_sha256": _split_hash(records),
            "videos": split_counts,
            "patients": patient_counts,
            "patient_leakage": False,
        },
        "video": {
            "input_layout": "B,C,T,H,W",
            "input_shape": [3, FRAMES_PER_CLIP, IMAGE_SIZE, IMAGE_SIZE],
            "sampling": "deterministic uniform coverage of all decodable source frames",
            "temporal_interpolation": False,
            "short_video_policy": "nearest-neighbor repeated positions to preserve exact sample set",
            "videos_requiring_repeated_positions": int(len(repeated)),
            "maximum_repeated_positions": int(audit["repeated_positions"].max()),
            "spatial_resize": "bilinear with antialias, 128x128 to 224x224",
            "pixel_scaling": "uint8 / 255",
            "augmentation": None,
            "decoded_frame_cache_on_disk": False,
            "byte_offset_index": os.path.abspath(index_path),
            "byte_offset_index_sha256": sha256(index_path),
            "index_manifest_sha256": sha256(index_manifest_path),
        },
        "model": {
            "stem": "Conv3d(3,64,kernel=7,stride=2,padding=3)+BN+ReLU",
            "residual_blocks": [
                {"channels": 64, "stride": 1}, {"channels": 64, "stride": 1},
                {"channels": 128, "stride": 2}, {"channels": 128, "stride": 1},
                {"channels": 256, "stride": 2}, {"channels": 256, "stride": 1},
            ],
            "block": "two Conv3d(3x3x3)+BN with ReLU and identity/projection shortcut",
            "tail": "AdaptiveAvgPool3d(1)+Linear(256,1)",
            "parameters": parameter_count(),
            "pretraining": None,
        },
        "training": {
            "loss": "MSE in g/dL",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "maximum_epochs": MAX_EPOCHS,
            "paper_early_stopping_threshold_val_mse": EARLY_STOPPING_VAL_MSE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": TRAIN_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "mixed_precision": True,
            "scheduler": None,
        },
        "declared_adaptations": [
            "Reuse the current Exp2 patient split instead of the paper's random 64/16/20 split.",
            "Use deterministic full-duration uniform sampling because the paper does not specify its 224-frame sampler.",
            "Repeat positions only for source videos shorter than 224 decodable frames to avoid changing the sample set.",
            "Use gradient accumulation to reproduce effective batch 4 on 16 GB GPUs.",
            "The available face crops are 128x128 and are spatially resized to the paper's 224x224 input.",
        ],
        "paper_internal_inconsistencies": [
            "Table 2 states 224 frames while Section 4.2 later states 30 frames; this implementation follows Table 2 and Sections 3.2.1/3.2.3.",
            "Table 2 and Section 3.2.3 state 100 epochs while Figure 7 discussion states 50; this implementation uses 100.",
        ],
    }
    with open(os.path.join(args.output_dir, "experiment_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"Prepared paper 3D CNN: videos={len(records)} split={split_counts} "
        f"short_videos_repeated={len(repeated)} parameters={parameter_count():,}", flush=True
    )
    if args.prepare_only:
        print("Preparation and model smoke test complete; training not started", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 224-frame 3D CNN")
    metrics, checkpoint_path, checkpoint_sha256 = train(
        records, index, args.output_dir, args.seed
    )
    pd.DataFrame([{
        "model": "paper_residual_3d_cnn", "target": "hemoglobin", "status": "ok",
        "checkpoint": checkpoint_path, "checkpoint_sha256": checkpoint_sha256,
    }]).to_csv(os.path.join(args.output_dir, "run_index.csv"), index=False)
    plot_results(args.output_dir)


if __name__ == "__main__":
    main()
