"""Patient-aware SimCLR over two distinct frames and views from each video."""

import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model
from study.exp2_face_pretrained_head32_regression.train import _prepare_images

from .config import (
    GRAD_CLIP_NORM,
    SIMCLR_BACKBONE_LR,
    SIMCLR_BATCH_SIZE,
    SIMCLR_EPOCHS,
    SIMCLR_MIN_LR_RATIO,
    SIMCLR_NUM_WORKERS,
    SIMCLR_PROJECTION_DIM,
    SIMCLR_PROJECTION_HIDDEN,
    SIMCLR_PROJECTOR_LR,
    SIMCLR_TEMPERATURE,
    VIEW_NAMES,
    WEIGHT_DECAY,
)


class VideoPositivePairDataset(Dataset):
    """One pair per video; both source frame and deterministic view differ."""

    def __init__(self, frame_index, records):
        self.records = records.reset_index(drop=True).copy()
        missing = sorted(
            set(self.records.video_id.astype(str)) - set(frame_index.video_lookup)
        )
        if missing:
            raise ValueError(
                f"SimCLR frame index is missing {len(missing)} videos; "
                f"examples={missing[:5]}"
            )
        self.base = AllFramesDataset(
            frame_index, self.records, views=("original",),
            interpolation="bicubic", expand_all_views=False,
        )
        self.ranges = []
        for video_row, video_id in enumerate(self.records.video_id.astype(str)):
            start, end = frame_index.frame_range(video_id)
            if end - start < 2:
                raise ValueError(f"SimCLR requires two frames for {video_id}")
            positions = np.flatnonzero(self.base.frame_video_rows == video_row)
            self.ranges.append(positions)
        patient_values = sorted(self.records.hospital_id.astype(str).unique())
        patient_lookup = {value: index for index, value in enumerate(patient_values)}
        self.patient_codes = self.records.hospital_id.astype(str).map(
            patient_lookup
        ).to_numpy(np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, video_row):
        positions = self.ranges[video_row]
        selected = torch.randperm(len(positions))[:2].tolist()
        first_position, second_position = (
            int(positions[selected[0]]), int(positions[selected[1]])
        )
        first_global = int(self.base.frame_indices[first_position])
        second_global = int(self.base.frame_indices[second_position])
        view_codes = torch.randperm(len(VIEW_NAMES), dtype=torch.int64)[:2]
        return (
            self.base._decode(first_global, video_row),
            self.base._decode(second_global, video_row),
            view_codes[0].to(torch.uint8),
            view_codes[1].to(torch.uint8),
            torch.tensor(self.patient_codes[video_row], dtype=torch.long),
        )

    def close(self):
        base = getattr(self, "base", None)
        if base is not None:
            base.close()

    def __del__(self):
        self.close()


class SimCLRModel(nn.Module):
    def __init__(self, regression_model):
        super().__init__()
        self.backbone = regression_model
        feature_dim = regression_model.classifier[0].in_features
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, SIMCLR_PROJECTION_HIDDEN),
            nn.LayerNorm(SIMCLR_PROJECTION_HIDDEN),
            nn.SiLU(inplace=True),
            nn.Linear(SIMCLR_PROJECTION_HIDDEN, SIMCLR_PROJECTION_DIM),
        )

    def forward(self, images):
        features = self.backbone.features(images)
        features = self.backbone.avgpool(features).flatten(1)
        return F.normalize(self.projector(features), dim=1)


def patient_aware_nt_xent(first, second, patient_codes, temperature):
    batch = len(first)
    embeddings = torch.cat((first, second), dim=0)
    patients = torch.cat((patient_codes, patient_codes), dim=0)
    similarities = embeddings @ embeddings.T / temperature
    indices = torch.arange(2 * batch, device=embeddings.device)
    positives = (indices + batch) % (2 * batch)
    allowed = patients[:, None].ne(patients[None, :])
    allowed[indices, positives] = True
    allowed[indices, indices] = False
    denominator = torch.logsumexp(similarities.masked_fill(~allowed, -torch.inf), dim=1)
    positive_similarity = similarities[indices, positives]
    loss = (denominator - positive_similarity).mean()
    with torch.no_grad():
        raw_positive = (first * second).sum(dim=1).mean()
        negative_mask = patients[:, None].ne(patients[None, :])
        negative_similarity = (embeddings @ embeddings.T)[negative_mask].mean()
    return loss, raw_positive, negative_similarity


def _plot(history, path, target):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column, title in zip(
        axes,
        ("loss", "positive_cosine", "negative_cosine"),
        ("NT-Xent loss", "Positive-pair cosine", "Cross-patient cosine"),
    ):
        axis.plot(frame.epoch, frame[column], marker="o", ms=3)
        axis.set_title(title); axis.set_xlabel("SimCLR epoch"); axis.grid(alpha=.25)
    figure.suptitle(f"SimCLR initialization | {target}")
    figure.tight_layout(); figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def train_simclr(
    frame_index, records, weights_dir, run_dir, target, device,
    epochs=SIMCLR_EPOCHS, max_batches=None,
):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    train_records = records[records.split.eq("train")].reset_index(drop=True)
    dataset = VideoPositivePairDataset(frame_index, train_records)
    loader = DataLoader(
        dataset, batch_size=SIMCLR_BATCH_SIZE, shuffle=True,
        num_workers=SIMCLR_NUM_WORKERS, pin_memory=True,
        persistent_workers=SIMCLR_NUM_WORKERS > 0,
        prefetch_factor=2,
    )
    regression_model, _, weight_path = build_pretrained_model(
        "efficientnet_b0", str(weights_dir)
    )
    model = SimCLRModel(regression_model).to(
        device, memory_format=torch.channels_last
    )
    optimizer = AdamW([
        {"params": model.backbone.features.parameters(), "lr": SIMCLR_BACKBONE_LR},
        {"params": model.projector.parameters(), "lr": SIMCLR_PROJECTOR_LR},
    ], weight_decay=WEIGHT_DECAY)
    def cosine_multiplier(epoch):
        progress = min(epoch, epochs) / max(epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return SIMCLR_MIN_LR_RATIO + (1.0 - SIMCLR_MIN_LR_RATIO) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=cosine_multiplier)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    history = []
    print(
        f"[stage-start] task={target} stage=simclr videos={len(dataset)} "
        f"backbone_lr={SIMCLR_BACKBONE_LR:.1e} projector_lr={SIMCLR_PROJECTOR_LR:.1e} "
        f"temperature={SIMCLR_TEMPERATURE} epochs={epochs}", flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train(); totals = np.zeros(3, dtype=np.float64); batches = inputs = 0
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
        for batch_index, (first, second, first_view, second_view, patients) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            first = _prepare_images(first, first_view, "bicubic", device)
            second = _prepare_images(second, second_view, "bicubic", device)
            patients = patients.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                first_embedding = model(first)
                second_embedding = model(second)
                loss, positive, negative = patient_aware_nt_xent(
                    first_embedding, second_embedding, patients, SIMCLR_TEMPERATURE
                )
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer); scaler.update()
            totals += (float(loss.detach()), float(positive), float(negative))
            batches += 1; inputs += 2 * len(first)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
        row = {
            "stage": "simclr", "epoch": epoch,
            "loss": totals[0] / max(batches, 1),
            "positive_cosine": totals[1] / max(batches, 1),
            "negative_cosine": totals[2] / max(batches, 1),
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "projector_learning_rate": optimizer.param_groups[1]["lr"],
            "model_inputs": inputs, "seconds": elapsed,
            "inputs_per_second": inputs / max(elapsed, 1e-9),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "simclr_history.csv", index=False)
        print(
            f"[epoch] task={target} stage=simclr {epoch:03d}/{epochs} "
            f"loss={row['loss']:.4f} pos_cos={row['positive_cosine']:.4f} "
            f"neg_cos={row['negative_cosine']:.4f} "
            f"throughput={row['inputs_per_second']:.1f}/s "
            f"mem={row['peak_gpu_memory_gb']:.2f}GiB", flush=True,
        )
        scheduler.step()
    backbone_state = {
        key: value.detach().cpu().clone()
        for key, value in model.backbone.state_dict().items()
        if key.startswith("features.")
    }
    torch.save({
        "schema_version": 1, "stage": "simclr", "target": target,
        "backbone_state_dict": backbone_state,
        "projector_state_dict": model.projector.state_dict(),
        "positive_pair": "two distinct frames and two distinct views from one video",
        "same_patient_other_video_policy": "excluded from negatives",
        "temperature": SIMCLR_TEMPERATURE,
        "backbone_learning_rate": SIMCLR_BACKBONE_LR,
        "projector_learning_rate": SIMCLR_PROJECTOR_LR,
        "minimum_learning_rate_ratio": SIMCLR_MIN_LR_RATIO,
        "pretrained_weight_path": str(weight_path),
    }, run_dir / "stage_simclr.pt")
    _plot(history, run_dir / "simclr_history.png", target)
    dataset.close(); del loader, model
    torch.cuda.empty_cache()
    return backbone_state, history
