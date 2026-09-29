"""Train-only label-aware SimCLR initialization for the face regression backbone."""

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler
from torchvision.models import efficientnet_b0

from .config import PREFETCH_FACTOR, VIEW_NAMES, WEIGHTS_DIR
from .data import AllFramesDataset
from .models import WEIGHT_FILES
from .train import _prepare_images


NEAR_IQR = 0.25
FAR_IQR = 0.75
TEMPERATURE = 0.20
EPOCHS = 8
ANCHORS_PER_BATCH = 12
FRAMES_PER_VIDEO_IN_BATCH = 2
BACKBONE_LR = 1e-4
PROJECTOR_LR = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
PROJECTION_DIM = 128


class LabelPairBatchSampler(Sampler):
    """Use close-valued video pairs while retaining distant-valued negatives."""

    def __init__(self, dataset, anchors_per_batch=ANCHORS_PER_BATCH):
        self.dataset = dataset
        self.anchors_per_batch = anchors_per_batch
        records = dataset.video_records
        if set(records.split) != {"train"} or dataset.views != ("original",):
            raise ValueError("Contrastive sampler requires train-only original frames")
        counts = np.bincount(dataset.frame_video_rows, minlength=len(records))
        if np.any(counts != 20):
            raise ValueError("Contrastive sampler requires exactly 20 indexed frames per video")
        self.values = records.robust_scaled_raw_value.to_numpy(np.float64)
        self.patients = records.hospital_id.astype(str).to_numpy()
        self.events = records.source_sample_id.astype(str).to_numpy()
        self.same_event_candidates = []
        self.same_patient_new_event_candidates = []
        self.other_patient_candidates = []
        for index, value in enumerate(self.values):
            close = np.flatnonzero(np.abs(self.values - value) <= NEAR_IQR)
            close = close[close != index]
            same = close[self.patients[close] == self.patients[index]]
            self.same_patient_new_event_candidates.append(
                same[self.events[same] != self.events[index]]
            )
            self.same_event_candidates.append(
                same[self.events[same] == self.events[index]]
            )
            self.other_patient_candidates.append(
                close[self.patients[close] != self.patients[index]]
            )
        self.last_epoch_pair_counts = {}

    def __len__(self):
        return math.ceil(len(self.values) / self.anchors_per_batch)

    def _partner(self, anchor):
        same_new = self.same_patient_new_event_candidates[anchor]
        other = self.other_patient_candidates[anchor]
        same_event = self.same_event_candidates[anchor]
        if len(same_new) and (not len(other) or np.random.random() < 0.5):
            return int(np.random.choice(same_new)), "same_patient_new_event"
        if len(other):
            return int(np.random.choice(other)), "other_patient_close_value"
        if len(same_event):
            return int(np.random.choice(same_event)), "same_event_other_video"
        return anchor, "same_video_fallback"

    def __iter__(self):
        order = torch.randperm(len(self.values)).tolist()
        counts = Counter()
        for start in range(0, len(order), self.anchors_per_batch):
            anchors = order[start:start + self.anchors_per_batch]
            if len(anchors) < self.anchors_per_batch:
                anchors += order[:self.anchors_per_batch - len(anchors)]
            partners = []
            for anchor in anchors:
                partner, kind = self._partner(anchor)
                partners.append(partner)
                counts[kind] += 1
            indices = []
            for video_row in anchors + partners:
                frame_offset = 20 * video_row
                chosen = torch.randperm(20)[:FRAMES_PER_VIDEO_IN_BATCH].tolist()
                indices.extend(frame_offset + frame for frame in chosen)
            yield indices
        self.last_epoch_pair_counts = dict(counts)

    def audit(self):
        return {
            "train_videos": len(self.values),
            "train_patients": len(set(self.patients)),
            "videos_with_same_patient_different_event_close_partner": sum(
                len(candidates) > 0
                for candidates in self.same_patient_new_event_candidates
            ),
            "videos_with_other_patient_close_partner": sum(
                len(candidates) > 0 for candidates in self.other_patient_candidates
            ),
            "near_threshold_training_iqr": NEAR_IQR,
            "far_threshold_training_iqr": FAR_IQR,
            "anchors_per_batch": self.anchors_per_batch,
            "frames_per_video_in_batch": FRAMES_PER_VIDEO_IN_BATCH,
            "views_per_frame": 2,
        }


def contrastive_loss(embeddings, values, video_rows):
    embeddings = functional.normalize(embeddings.float(), dim=1)
    values = values.float()
    logits = embeddings @ embeddings.T / TEMPERATURE
    distance = (values[:, None] - values[None, :]).abs()
    identity = torch.eye(len(values), device=values.device, dtype=torch.bool)
    positive = ((distance <= NEAR_IQR) | (video_rows[:, None] == video_rows[None, :])) & ~identity
    negative = (distance >= FAR_IQR) & ~identity
    selected = positive | negative
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    valid = (positive_count > 0) & (negative_count > 0)
    if not valid.any():
        raise RuntimeError("Contrastive batch has no anchors with positives and far negatives")
    log_denominator = torch.logsumexp(
        logits.masked_fill(~selected, -torch.inf), dim=1
    )
    per_anchor = (
        (log_denominator[:, None] - logits).masked_fill(~positive, 0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return per_anchor[valid].mean(), {
        "positive_pairs": int(positive.sum().item()),
        "far_negative_pairs": int(negative.sum().item()),
        "valid_anchors": int(valid.sum().item()),
    }


class ContrastiveModel(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        backbone.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.encoder = backbone
        self.encoder.classifier = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, PROJECTION_DIM),
        )

    def forward(self, images):
        features = self.encoder.features(images)
        features = self.encoder.avgpool(features).flatten(1)
        return self.projector(features)


def pretrain_encoder(frame_index, train_records, run_dir, max_batches=None,
                     epochs=EPOCHS):
    if set(train_records.split) != {"train"}:
        raise ValueError("Contrastive pretraining must not receive validation/test records")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = AllFramesDataset(
        frame_index, train_records, views=("original",), interpolation="bicubic"
    )
    sampler = LabelPairBatchSampler(dataset)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
    )
    weight_path = Path(WEIGHTS_DIR) / WEIGHT_FILES["efficientnet_b0"]
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = ContrastiveModel(weight_path).to(
        device, memory_format=torch.channels_last
    )
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": BACKBONE_LR},
            {"params": model.projector.parameters(), "lr": PROJECTOR_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LR)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    history = []
    audit = {
        **sampler.audit(),
        "method": "label_aware_multi_positive_simclr",
        "positive_policy": "same video or absolute scaled value difference <= 0.25",
        "negative_policy": "absolute scaled value difference >= 0.75",
        "ambiguous_pairs": "excluded from contrastive denominator",
        "projector": "1280-256-128; discarded before regression training",
        "temperature": TEMPERATURE,
        "epochs": epochs,
        "backbone_learning_rate": BACKBONE_LR,
        "projector_learning_rate": PROJECTOR_LR,
        "minimum_learning_rate": MIN_LR,
        "weight_decay": WEIGHT_DECAY,
        "source_views": list(VIEW_NAMES),
    }
    (run_dir / "pretrain_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"[simclr-start] target={run_dir.name} train_videos={len(train_records)} "
        f"same_patient_new_event_candidates="
        f"{audit['videos_with_same_patient_different_event_close_partner']} "
        f"epochs={epochs} batch_images={2 * 2 * 2 * ANCHORS_PER_BATCH}",
        flush=True,
    )
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            started = time.perf_counter()
            loss_sum = batches = positive_pairs = far_negative_pairs = 0
            for batch_index, (images, labels, indices, _) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                first = torch.randint(len(VIEW_NAMES), (len(images),), dtype=torch.uint8)
                second = (first + torch.randint(1, len(VIEW_NAMES), (len(images),),
                                                dtype=torch.uint8)) % len(VIEW_NAMES)
                views = torch.stack((first, second), dim=1)
                images = _prepare_images(images, views, dataset.interpolation, device)
                values = labels.to(device, non_blocking=True).repeat_interleave(2)
                video_rows = torch.as_tensor(
                    dataset.frame_video_rows[indices.numpy()], device=device
                ).repeat_interleave(2)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    embeddings = model(images)
                loss, counts = contrastive_loss(embeddings, values, video_rows)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite contrastive loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach())
                batches += 1
                positive_pairs += counts["positive_pairs"]
                far_negative_pairs += counts["far_negative_pairs"]
            if not batches:
                raise RuntimeError("Contrastive pretraining processed no batches")
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / batches,
                "batches": batches,
                "positive_pairs": positive_pairs,
                "far_negative_pairs": far_negative_pairs,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "projector_learning_rate": optimizer.param_groups[1]["lr"],
                "seconds": time.perf_counter() - started,
                **{f"sampled_{name}": count
                   for name, count in sampler.last_epoch_pair_counts.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "pretrain_history.csv", index=False)
            print(
                f"[simclr-epoch] target={run_dir.name} {epoch}/{epochs} "
                f"loss={row['train_loss']:.4f} pairs={positive_pairs}/{far_negative_pairs} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
            scheduler.step()
        path = run_dir / "pretrain_encoder.pt"
        torch.save(
            {"features_state_dict": {
                key: value.detach().cpu()
                for key, value in model.encoder.features.state_dict().items()
            }},
            path,
        )
        return path
    finally:
        del loader
        dataset.close()
        del model
        torch.cuda.empty_cache()
