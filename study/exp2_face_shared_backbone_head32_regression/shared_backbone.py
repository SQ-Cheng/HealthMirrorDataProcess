"""A single face encoder, independent scalar heads, and masked task-balanced loss."""

import heapq

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler

from . import config
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model, SingleTaskHead


class SharedBackboneModel(nn.Module):
    def __init__(self, targets):
        super().__init__()
        backbone, _, self.weight_path = build_pretrained_model("efficientnet_b0", config.WEIGHTS_DIR)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.targets = tuple(targets)
        self.heads = nn.ModuleDict({target: SingleTaskHead(1280) for target in self.targets})

    def forward(self, images):
        features = self.avgpool(self.features(images)).flatten(1)
        return torch.cat([self.heads[target](features) for target in self.targets], dim=1)

    def freeze_encoder(self):
        for parameter in self.parameters(): parameter.requires_grad = False
        for parameter in self.heads.parameters(): parameter.requires_grad = True


def masked_losses(predictions, labels, mask, beta=config.SMOOTH_L1_BETA):
    mask = mask.bool()
    safe_labels = torch.where(mask, labels, predictions.detach())
    values = F.smooth_l1_loss(predictions.float(), safe_labels.float(), beta=beta, reduction="none")
    return torch.where(mask, values, torch.zeros_like(values))


def task_balanced_loss(predictions, labels, mask, weights):
    """Uniform-video batches estimate the equal mean of per-task observed losses."""
    return (masked_losses(predictions, labels, mask) * weights).sum(dim=1).mean()


class SharedFrameDataset(AllFramesDataset):
    def __init__(self, frame_index, videos, labels, masks, views):
        super().__init__(frame_index, videos.assign(robust_scaled_raw_value=0.), views=views,
                         interpolation="bicubic", expand_all_views=False)
        self.task_labels = np.asarray(labels, np.float32)
        self.task_masks = np.asarray(masks, bool)
        if self.task_labels.shape != self.task_masks.shape or len(self.task_labels) != len(videos):
            raise ValueError("Video/label/mask alignment is invalid")
        self.decode_cache_frames = 20

    def __getitem__(self, index):
        image, _, frame_row, code = super().__getitem__(index)
        video = int(self.frame_video_rows[int(frame_row)])
        return image, torch.from_numpy(self.task_labels[video]), torch.from_numpy(self.task_masks[video]), frame_row, code


class SharedLabBatchSampler(Sampler):
    """Retain every video/view while preventing repeated per-task assay events."""

    def __init__(self, event_sets, view_count=5, groups_per_batch=12, seed=config.SEED):
        self.events = [frozenset(events) for events in event_sets]
        if any(not events for events in self.events): raise ValueError("A video has no observed target")
        self.view_count, self.groups_per_batch, self.seed = view_count, groups_per_batch, seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        views = [rng.permutation(self.view_count).tolist() for _ in self.events]
        heap = [(-self.view_count, float(rng.random()), video) for video in range(len(self.events))]
        heapq.heapify(heap)
        self.batches = []
        while heap:
            selected, deferred, used = [], [], set()
            while heap and len(selected) < self.groups_per_batch:
                item = heapq.heappop(heap); video = item[2]
                if self.events[video] & used:
                    deferred.append(item)
                else:
                    selected.append(video); used.update(self.events[video])
            batch = []
            for video in selected:
                view = views[video].pop()
                batch.extend((video * 20 + frame) * self.view_count + view for frame in range(20))
                if views[video]: deferred.append((-len(views[video]), float(rng.random()), video))
            for item in deferred: heapq.heappush(heap, item)
            self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
