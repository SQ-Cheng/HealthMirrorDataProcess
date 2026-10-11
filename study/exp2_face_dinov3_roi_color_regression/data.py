"""Same-source-frame/view fusion and variable-length distinct-lab batches."""

import heapq

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from study.exp1_roi_color_regression.views import CACHE as ROI_CACHE
from study.exp2_face_dinov3_frozen.run_main_regression import CACHE as DINO_CACHE


def load_features(index, records):
    dino = np.load(DINO_CACHE / "cls_features.npy", mmap_mode="r")
    colors = np.load(ROI_CACHE / "features.npy", mmap_mode="r")
    validity = np.load(ROI_CACHE / "valid.npy", mmap_mode="r")
    if (dino.shape != (len(index.starts), 5, 384) or colors.shape != (len(index.starts), 5, 4, 41)
            or validity.shape != (len(index.starts), 5, 4)):
        raise RuntimeError("Frame/view cache shape mismatch")
    frames, ptr = [], [0]
    for video in records.video_id:
        start, end = index.frame_range(video)
        selected = np.arange(start, end)[validity[start:end].all(axis=(1, 2))]
        if len(selected) < 10:
            raise RuntimeError("An ineligible ROI video entered fusion")
        frames.extend(selected)
        ptr.append(len(frames))
    frames, ptr = np.asarray(frames, dtype=np.int64), np.asarray(ptr, dtype=np.int64)
    features = np.concatenate((dino[frames], colors[frames].reshape(len(frames), 5, 164)), axis=2)
    if not np.isfinite(features).all():
        raise RuntimeError("Nonfinite fused feature")
    return features, ptr, frames


def fit_color_scaler(features, ptr, training):
    video_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    selected = np.isin(video_rows, training)
    weights = np.repeat(1. / (np.diff(ptr)[video_rows[selected]] * 5), 5)
    return StandardScaler().fit(features[selected, :, 384:].reshape(-1, 164), sample_weight=weights)


def lab_batches(records, ptr, groups_per_batch=12):
    """Every video/view once; no repeated clinical event within a batch."""
    pending = {}
    for group in torch.randperm(len(records) * 5).tolist():
        video, view = divmod(group, 5)
        pending.setdefault(str(records.iloc[video].clinical_event_id), []).append((video, view))
    ties = iter(torch.rand(len(records) * 5 + len(pending)).tolist())
    heap = [(-len(groups), next(ties), event) for event, groups in pending.items()]
    heapq.heapify(heap)
    while heap:
        picked = [heapq.heappop(heap) for _ in range(min(groups_per_batch, len(heap)))]
        samples = []
        for _, _, event in picked:
            video, view = pending[event].pop()
            samples.extend(np.arange(ptr[video], ptr[video + 1]) * 5 + view)
        for _, _, event in picked:
            if pending[event]:
                heapq.heappush(heap, (-len(pending[event]), next(ties), event))
        yield np.asarray(samples, dtype=np.int64)


def pooled_prediction(frame_predictions, ptr):
    return np.asarray([frame_predictions[start:end].mean(dtype=np.float64) for start, end in zip(ptr[:-1], ptr[1:])])
