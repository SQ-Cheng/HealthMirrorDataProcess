"""Run SimCLR, then the unchanged two-stage Head32 regression trainer."""

import hashlib
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression import models as base_models
from study.exp2_face_pretrained_head32_regression import train as base_train

from .config import (
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    SIMCLR_EPOCHS,
    WEIGHTS_DIR,
)
from .simclr import train_simclr


def job_seed(seed, target):
    token = f"{seed}:efficientnet_b0:{target}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return (seed + offset) % (2**31 - 1)


def _seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _plot_three_stages(run_dir, target):
    run_dir = Path(run_dir)
    simclr = pd.read_csv(run_dir / "simclr_history.csv")
    regression = pd.read_csv(run_dir / "history.csv")
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(simclr.epoch, simclr.loss, color="#2F6B8A")
    axes[0].set(title="Stage 1: SimCLR NT-Xent", xlabel="Epoch", ylabel="Loss")
    for stage, group in regression.groupby("stage", sort=False):
        axes[1].plot(group.global_epoch, group.train_loss, label=f"{stage} train")
        axes[1].plot(group.global_epoch, group.val_loss, "--", label=f"{stage} val")
        axes[2].plot(group.global_epoch, group.val_mae, label=f"{stage} MAE")
        axes[2].plot(group.global_epoch, group.val_pearson_r, "--", label=f"{stage} r")
    axes[1].set(title="Stages 2-3: regression loss", xlabel="Global epoch")
    axes[2].set(title="Stages 2-3: validation", xlabel="Global epoch")
    for axis in axes:
        axis.grid(alpha=.25)
        if axis is not axes[0]: axis.legend(fontsize=8)
    figure.suptitle(f"Three-stage training | {target}")
    figure.tight_layout(); figure.savefig(
        run_dir / "three_stage_history.png", dpi=180, bbox_inches="tight"
    ); plt.close(figure)


def train_three_stage(
    target, frame_index, records, target_scaler, run_dir, seed,
    smoke=False,
):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    current_seed = job_seed(seed, target)
    _seed(current_seed)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    backbone_state, _ = train_simclr(
        frame_index=frame_index,
        records=records,
        weights_dir=WEIGHTS_DIR,
        run_dir=run_dir,
        target=target,
        device=device,
        epochs=1 if smoke else SIMCLR_EPOCHS,
        max_batches=2 if smoke else None,
    )

    # Recreate the prediction head from the original job seed; only the encoder
    # receives the SimCLR state, keeping head initialization controlled.
    _seed(current_seed)
    original_builder = base_train.build_pretrained_model

    def simclr_initialized_builder(architecture, weights_dir):
        model, head, weight_path = original_builder(architecture, weights_dir)
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        unexpected = [key for key in unexpected if key.startswith("features.")]
        missing_features = [key for key in missing if key.startswith("features.")]
        if unexpected or missing_features:
            raise RuntimeError(
                f"SimCLR backbone load mismatch missing={missing_features} "
                f"unexpected={unexpected}"
            )
        return model, head, weight_path

    base_train.build_pretrained_model = simclr_initialized_builder
    try:
        metrics = base_train.train_task(
            architecture="efficientnet_b0",
            target=target,
            frame_index=frame_index,
            records=records,
            target_scaler=target_scaler,
            weights_dir=WEIGHTS_DIR,
            run_dir=str(run_dir),
            head_epochs=1 if smoke else HEAD_MAX_EPOCHS,
            finetune_epochs=1 if smoke else FINETUNE_MAX_EPOCHS,
            head_patience=1 if smoke else HEAD_PATIENCE,
            finetune_patience=1 if smoke else FINETUNE_PATIENCE,
            max_batches=2 if smoke else None,
        )
    finally:
        base_train.build_pretrained_model = original_builder

    checkpoint_path = run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update({
        "training_stages": ("simclr", "head", "finetune"),
        "encoder_initialization": "ImageNet followed by train-split-only SimCLR",
        "simclr_checkpoint": "stage_simclr.pt",
        "simclr_positive_pair": (
            "two different frames and two different views from the same video"
        ),
        "simclr_same_patient_other_video_policy": "excluded from negatives",
    })
    torch.save(checkpoint, checkpoint_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "experiment": "exp2_face_pretrained_head32_regression_simclr",
        "target": target, "job_seed": current_seed,
        "stages": ["simclr", "head", "finetune"],
        "data_contract": "exact task records and patient split from baseline regression",
        "simclr_scope": "training-split videos only",
        "positive_pair": "different frames and different views from the same video",
        "same_patient_other_video_policy": "excluded from negatives",
    }, indent=2), encoding="utf-8")
    _plot_three_stages(run_dir, target)
    return metrics
