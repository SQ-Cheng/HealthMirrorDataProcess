"""Complete video/view batches and losses on twenty-frame predictions."""

import math
import heapq

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler


class VideoViewBatchSampler(Sampler):
    """Keep each video's frames together, without patient-diverse sampling."""

    def __init__(self, dataset, frame_budget, shuffle, frames=20, chunk_frames=256):
        if dataset.expand_all_views:
            raise ValueError("Video batches require individually indexed views")
        self.dataset, self.frames, self.shuffle = dataset, frames, shuffle
        self.groups_per_batch = frame_budget // frames
        self.frame_batch_size = self.groups_per_batch * frames
        self.videos_per_chunk = max(1, math.ceil(chunk_frames / frames))
        if self.groups_per_batch < 1:
            raise ValueError("Frame budget must fit one complete video")
        expected = np.repeat(np.arange(len(dataset.video_records)), frames)
        if not np.array_equal(dataset.frame_video_rows, expected):
            raise ValueError("Each video must have exactly twenty contiguous selected frames")

    def __iter__(self):
        n = len(self.dataset.video_records)
        chunks = math.ceil(n / self.videos_per_chunk)
        order = torch.randperm(chunks).tolist() if self.shuffle else range(chunks)
        view_count = len(self.dataset.views)
        batch = []
        for chunk in order:
            for video in range(chunk * self.videos_per_chunk, min(n, (chunk + 1) * self.videos_per_chunk)):
                views = torch.randperm(view_count).tolist() if self.shuffle else range(view_count)
                for view in views:
                    batch.extend((video * self.frames + frame) * view_count + view
                                 for frame in range(self.frames))
                    if len(batch) == self.frame_batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset.video_records) * len(self.dataset.views) / self.groups_per_batch)


class DistinctLabViewBatchSampler(VideoViewBatchSampler):
    """Use a different clinical measurement for every video/view in a batch."""

    def __init__(self, dataset, frame_budget, frames=20):
        super().__init__(dataset, frame_budget, True, frames=frames)
        records = dataset.video_records
        if "clinical_event_id" not in records or records.clinical_event_id.isna().any():
            raise ValueError("Distinct-lab batching requires verified clinical measurement IDs")
        self.events = records.clinical_event_id.astype(str).to_numpy()
        multiplicity = records.groupby("clinical_event_id").size().max() * len(dataset.views)
        self.batch_count = max(super().__len__(), int(multiplicity))

    def __iter__(self):
        view_count = len(self.dataset.views)
        group_count = len(self.events) * view_count
        pending = {}
        for group in torch.randperm(group_count).tolist():
            video, view = divmod(group, view_count)
            pending.setdefault(self.events[video], []).append((video, view))
        ties = iter(torch.rand(group_count + len(pending)).tolist())
        heap = [(-len(groups), next(ties), event) for event, groups in pending.items()]
        heapq.heapify(heap)
        while heap:
            selected = [heapq.heappop(heap) for _ in range(min(self.groups_per_batch, len(heap)))]
            batch = []
            for _, _, event in selected:
                video, view = pending[event].pop()
                batch.extend((video * self.frames + frame) * view_count + view
                             for frame in range(self.frames))
            # Reinsert only after selection, so a measurement cannot occur twice.
            for _, _, event in selected:
                if pending[event]:
                    heapq.heappush(heap, (-len(pending[event]), next(ties), event))
            yield batch

    def __len__(self):
        return self.batch_count


class VideoSmoothL1Loss(nn.Module):
    video_level = True

    def __init__(self, beta=0.5, frames=20):
        super().__init__()
        self.beta, self.frames = beta, frames

    def forward(self, prediction, labels):
        mean = prediction.float().reshape(-1, self.frames).mean(dim=1)
        truth = labels.float().reshape(-1, self.frames)[:, 0]
        return F.smooth_l1_loss(mean, truth, beta=self.beta, reduction="none")


class VideoBCELoss(nn.Module):
    video_level = True

    def __init__(self, pos_weight, frames=20):
        super().__init__()
        self.frames = frames
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight))

    def forward(self, logits, labels):
        values = logits.float().reshape(-1, self.frames)
        truth = labels.float().reshape(-1, self.frames)[:, 0]
        # Log-space averaging matches mean(sigmoid(frame logits)) without clipping.
        log_positive = torch.logsumexp(F.logsigmoid(values), dim=1) - math.log(self.frames)
        log_negative = torch.logsumexp(F.logsigmoid(-values), dim=1) - math.log(self.frames)
        return -(self.pos_weight * truth * log_positive + (1 - truth) * log_negative)
