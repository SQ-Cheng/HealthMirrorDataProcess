"""Read-only paired caches and training-only label/weight transformations."""

import heapq
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from . import config


class PatientEventBatchSampler(DistinctLabViewBatchSampler):
    """One patient/event per batch; exhaustive views, including short tails."""
    def __init__(self, dataset, frame_budget):
        super().__init__(dataset, frame_budget)
        if dataset.video_records.groupby("clinical_event_id").hospital_id.nunique().gt(1).any():
            raise ValueError("Clinical event cannot cross patient identities")
        self.patients = dataset.video_records.hospital_id.to_numpy()
        maximum = dataset.video_records.groupby("hospital_id").size().max()*len(dataset.views)
        self.batch_count = max(math.ceil(len(dataset.video_records)*len(dataset.views)/self.groups_per_batch), int(maximum))

    def __iter__(self):
        views = len(self.dataset.views)
        count = len(self.patients)*views
        pending = {}
        for group in torch.randperm(count).tolist():
            pending.setdefault(self.patients[group//views], []).append(group)
        ties = iter(torch.rand(count+len(pending)).tolist())
        heap = [(-len(groups), next(ties), patient) for patient, groups in pending.items()]
        heapq.heapify(heap)
        while heap:
            selected = [heapq.heappop(heap) for _ in range(min(self.groups_per_batch, len(heap)))]
            batch = []
            for _, _, patient in selected:
                video, view = divmod(pending[patient].pop(), views)
                batch.extend((video*20+frame)*views+view for frame in range(20))
            for _, _, patient in selected:
                if pending[patient]:
                    heapq.heappush(heap, (-len(pending[patient]), next(ties), patient))
            yield batch


def read_records(target):
    return pd.read_csv(config.BASELINE / f"source_records/{target}.csv", dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")


def fit_transform(records, target, settings):
    train = records.loc[records.split.eq("train")]
    logged = target in settings["log_targets"]
    raw = train.raw_value.to_numpy(float)
    if logged and (records.raw_value < 0).any():
        raise ValueError("log1p analyte contains negative lab values")
    values = np.log1p(raw) if logged else raw
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    if q3 <= q1:
        raise ValueError("Nonpositive training target IQR")
    return {"target": target, "log1p": logged, "median": float(median), "iqr": float(q3-q1),
            "unit": config.reference.paired.SCORE_DEFINITIONS[target]["unit"], "fit_scope": "train only"}


def transform(raw, scaler):
    values = np.asarray(raw, dtype=np.float64)
    if scaler["log1p"]:
        values = np.log1p(values)
    return ((values-scaler["median"])/scaler["iqr"]).astype(np.float32)


def inverse(values, scaler):
    raw = np.asarray(values, dtype=np.float64)*scaler["iqr"]+scaler["median"]
    raw = np.expm1(raw) if scaler["log1p"] else raw
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite inverse-transformed predictions; no silent clipping")
    return raw


def train_weights(records, settings):
    if set(records.split) != {"train"}:
        raise ValueError("Weights must be fitted on training records only")
    weights = np.ones(len(records), dtype=np.float64)
    policy = {"patient_event_balance": settings["balance"], "fit_scope": "train only"}
    if settings["balance"]:
        event_patients = records.groupby("clinical_event_id").hospital_id.nunique()
        if event_patients.gt(1).any():
            raise ValueError("Clinical event belongs to multiple patients")
        events = records.groupby("hospital_id").clinical_event_id.transform("nunique")
        videos = records.groupby("clinical_event_id").video_id.transform("size")
        weights /= (events*videos).to_numpy(float)
    if settings["density"]:
        lower, upper = np.quantile(records.raw_value, [.01, .99])
        if upper <= lower:
            raise ValueError("Density weighting requires a nonconstant target")
        edges = np.linspace(lower, upper, 6)
        bins = np.searchsorted(edges[1:-1], records.raw_value)
        counts = np.bincount(bins, minlength=5)
        factor = np.sqrt(len(records)/counts[bins])
        factor = np.minimum(factor/factor.mean(), 3.)
        weights *= factor
        policy.update(density_edges=edges.tolist(), density_counts=counts.tolist(), density_factor_cap=3.)
    if settings["time_tau"] is not None:
        delta = records.match_delta_h.to_numpy(float)
        if not np.isfinite(delta).all() or np.any(delta < 0):
            raise ValueError("Invalid matching intervals")
        weights *= .25+.75*np.exp(-delta/settings["time_tau"])
        policy.update(time_tau_hours=settings["time_tau"], time_floor=.25)
    weights /= weights.mean()
    policy.update(min_weight=float(weights.min()), max_weight=float(weights.max()),
                  effective_video_n=float(weights.sum()**2/np.square(weights).sum()))
    return weights.astype(np.float32), policy


class Dataset(torch.utils.data.Dataset):
    def __init__(self, index, records, settings, scaler, train=False, online=False, ir_index=None):
        self.index, self.video_records = index, records.reset_index(drop=True)
        self.views = tuple(settings["views"] if train else ("original",))
        self.codes = [config.reference.VIEWS.index(view) for view in self.views]
        self.expand_all_views = False
        self.frame_indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.video_id])
        self.frame_video_rows = np.repeat(np.arange(len(records)), 20)
        self.labels = transform(records.raw_value, scaler)
        self.banks = [np.load(config.reference.CACHE / f"{branch}.npy", mmap_mode="r") for branch in config.BRANCHES]
        self.ir = None
        if online:
            self.ir = AllFramesDataset(ir_index, records.assign(robust_scaled_raw_value=self.labels), views=("original",))
            self.ir.decode_cache_frames = 20

    def __len__(self):
        return len(self.frame_indices)*len(self.views)

    def __getitem__(self, sample):
        frame, local_view = divmod(sample, len(self.views))
        view, position, video = self.codes[local_view], self.frame_indices[frame], self.frame_video_rows[frame]
        features = np.concatenate([bank[position, view] for bank in self.banks])
        pixels = self.ir[frame][0] if self.ir is not None else torch.empty(0, dtype=torch.uint8)
        return torch.from_numpy(features), torch.tensor(self.labels[video]), torch.tensor(video), torch.tensor(view), pixels

    def close(self):
        if self.ir is not None:
            self.ir.close()


def loader(index, records, settings, scaler, train=False, online=False, ir_index=None):
    dataset = Dataset(index, records, settings, scaler, train, online, ir_index)
    sampler_class = PatientEventBatchSampler if settings["balance"] else DistinctLabViewBatchSampler
    sampler = sampler_class(dataset, 240) if train else VideoViewBatchSampler(dataset, 240, False)
    workers = 2 if online else 0
    options = dict(batch_sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available())
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return dataset, DataLoader(dataset, **options)


def fit_feature_stats(model, index, records):
    if set(records.split) != {"train"}:
        raise ValueError("Feature normalization uses train only")
    indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.video_id])
    mean, std = [], []
    for branch in config.BRANCHES:
        bank = np.load(config.reference.CACHE / f"{branch}.npy", mmap_mode="r")
        total = np.zeros(bank.shape[-1], dtype=np.float64)
        squares = total.copy()
        for start in range(0, len(indices), 512):
            values = bank[indices[start:start+512], 0].astype(np.float64)
            total += values.sum(0)
            squares += np.square(values).sum(0)
        current = total/len(indices)
        mean.extend(current)
        std.extend(np.sqrt(np.maximum(squares/len(indices)-current**2, 0)).clip(.05))
    model.feature_mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.feature_std.copy_(torch.tensor(std, dtype=torch.float32))
