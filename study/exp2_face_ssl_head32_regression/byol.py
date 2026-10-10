"""Label-free BYOL with EMA parameters and train-only BatchNorm state."""

import copy
import heapq
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from . import config


def bootstrap_loss(prediction, target):
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    return 2 - 2 * (prediction * target).sum(dim=-1)


def mlp(inputs):
    return nn.Sequential(nn.Linear(inputs, config.SSL_MLP_HIDDEN), nn.BatchNorm1d(config.SSL_MLP_HIDDEN),
                         nn.ReLU(inplace=True), nn.Linear(config.SSL_MLP_HIDDEN, config.SSL_PROJECTION_DIM))


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        backbone, _, _ = build_pretrained_model("efficientnet_b0", config.reference.WEIGHTS_DIR)
        self.online_features = backbone.features
        self.online_projector = mlp(1280)
        self.predictor = mlp(config.SSL_PROJECTION_DIM)
        self.target_features = copy.deepcopy(self.online_features)
        self.target_projector = copy.deepcopy(self.online_projector)
        for module in (self.target_features, self.target_projector):
            for parameter in module.parameters(): parameter.requires_grad = False

    def forward(self, first, second):
        images = torch.cat([first, second], dim=0)
        # Teacher BN uses only the same training batch; no held-out forward passes.
        with torch.no_grad():
            target = self.target_features(images).mean(dim=(-2, -1))
            target = self.target_projector(target)
        features = self.online_features(images).mean(dim=(-2, -1))
        projection = self.online_projector(features)
        prediction = self.predictor(projection)
        first_prediction, second_prediction = prediction.chunk(2)
        first_target, second_target = target.chunk(2)
        loss = .5 * (bootstrap_loss(first_prediction, second_target) + bootstrap_loss(second_prediction, first_target))
        return loss, F.normalize(projection[:len(first)].detach().float(), dim=1)

    @torch.no_grad()
    def update_target(self, momentum):
        for online, target in ((self.online_features, self.target_features),
                               (self.online_projector, self.target_projector)):
            for value, averaged in zip(online.parameters(), target.parameters()):
                averaged.lerp_(value, 1 - momentum)
        # BN running buffers are updated by teacher forward, not gradient/EMA.


class PairDataset(AllFramesDataset):
    def __init__(self, index, videos):
        super().__init__(index, videos.assign(robust_scaled_raw_value=0.), views=config.VIEWS,
                         expand_all_views=False, interpolation="bicubic")

    def __getitem__(self, selection):
        flat_index, epoch, valid = selection
        image, _, frame, anchor = super().__getitem__(flat_index)
        other = (int(anchor) + 1 + (int(frame) * 2027 + epoch * 1009) % 4) % len(config.VIEWS)
        return image, anchor, torch.tensor(other, dtype=torch.uint8), valid


class DiverseVideoBatchSampler(Sampler):
    """Prefer distinct videos, then fill scarce-video tails without dropping data."""

    def __init__(self, counts, batch_size, seed=config.SEED):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        if not len(self.counts) or self.counts.min() < 1 or batch_size < 1:
            raise ValueError("Positive per-video frame counts and batch size required")
        self.offsets = np.r_[0, np.cumsum(self.counts)]

    def __len__(self):
        return math.ceil(int(self.counts.sum()) * len(config.VIEWS) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        queues = [rng.permutation(int(count) * len(config.VIEWS)) for count in self.counts]
        remaining = self.counts * len(config.VIEWS)
        heap = [(-int(count), float(rng.random()), video) for video, count in enumerate(remaining)]
        heapq.heapify(heap)
        while heap:
            selected = [heapq.heappop(heap)[2] for _ in range(min(self.batch_size, len(heap)))]
            batch = []
            def take(video):
                position = len(queues[video]) - remaining[video]
                index = int(self.offsets[video]) * len(config.VIEWS) + int(queues[video][position])
                batch.append((index, self.epoch)); remaining[video] -= 1
            for video in selected:
                take(video)
                if remaining[video]: heapq.heappush(heap, (-int(remaining[video]), float(rng.random()), video))
            while heap and len(batch) < self.batch_size:
                video = heapq.heappop(heap)[2]; take(video)
                if remaining[video]: heapq.heappush(heap, (-int(remaining[video]), float(rng.random()), video))
            rng.shuffle(batch)
            yield batch


class RankBatchSampler(Sampler):
    """Shard common diverse batches; empty ranks use a loss-zero train-only slot."""

    def __init__(self, sampler, rank, world):
        self.sampler, self.rank, self.world = sampler, rank, world

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for global_batch in self.sampler:
            local = global_batch[self.rank::self.world]
            if local:
                yield [(index, epoch, True) for index, epoch in local]
            else:
                index, epoch = global_batch[0]
                yield [(index, epoch, False)]


def schedules(step, total, warmup):
    if step < warmup:
        factor = (step + 1) / max(1, warmup)
    else:
        factor = .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    momentum = 1 - (1 - config.SSL_EMA_BASE) * .5 * (1 + math.cos(math.pi * step / max(1, total)))
    return factor, momentum


def atomic_save(data, path):
    path = Path(path); temporary = path.with_suffix(".tmp")
    torch.save(data, temporary); temporary.replace(path)


def pretrain(index, videos, output, contract, epochs=config.SSL_EPOCHS, max_steps=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if set(videos.split) != {"train"}: raise ValueError("BYOL is strictly train-only")
    rank, world, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available(): raise RuntimeError("BYOL requires CUDA")
    torch.cuda.set_device(local_rank); torch.set_num_threads(1)
    if world > 1: dist.init_process_group("nccl")
    torch.manual_seed(config.SEED); torch.cuda.manual_seed(config.SEED + rank)
    device = torch.device(f"cuda:{local_rank}")
    dataset = PairDataset(index, videos)
    counts = np.bincount(dataset.frame_video_rows, minlength=len(videos))
    sampler = DiverseVideoBatchSampler(counts, config.SSL_ANCHORS_PER_GPU * world)
    loader = DataLoader(dataset, batch_sampler=RankBatchSampler(sampler, rank, world), num_workers=0 if max_steps else config.SSL_WORKERS,
                        pin_memory=True, persistent_workers=not bool(max_steps),
                        multiprocessing_context=None if max_steps else "spawn",
                        prefetch_factor=None if max_steps else config.SSL_PREFETCH)
    model = BYOL().to(device, memory_format=torch.channels_last)
    execution = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    optimizer = AdamW([{"params": model.online_features.parameters(), "lr": config.SSL_BACKBONE_LR},
                       {"params": list(model.online_projector.parameters()) + list(model.predictor.parameters()),
                        "lr": config.SSL_MLP_LR}], weight_decay=config.SSL_WEIGHT_DECAY)
    amp = torch.amp.GradScaler("cuda", init_scale=1024)
    step, start_epoch, history = 0, 0, []
    last = output / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["contract"] != contract or saved["gpu_count"] != world:
            raise RuntimeError("SSL resume contract/GPU count changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); amp.load_state_dict(saved["amp"])
        step, start_epoch, history = saved["step"], saved["epoch"], saved["history"]
        torch.set_rng_state(saved["rngs"][rank]["cpu"]); torch.cuda.set_rng_state(saved["rngs"][rank]["cuda"], device)
    total = epochs * len(sampler); warmup = min(config.SSL_WARMUP_EPOCHS, epochs) * len(sampler)
    if rank == 0: print(f"[byol-start] train_videos={len(videos)} patients={videos.hospital_id.nunique()} "
          f"frames={dataset.frame_count} anchors_per_epoch={len(dataset)} global_batch={sampler.batch_size} "
          f"gpus={world} DDP epochs={epochs}; no validation/test images or lab labels", flush=True)
    for epoch in range(start_epoch, epochs):
        sampler.epoch = epoch; model.train()
        loss_sum = std_sum = anchors = batches = skipped = 0
        started = time.monotonic()
        for batch, (images, anchor, partner, valid) in enumerate(loader):
            if max_steps is not None and batch >= max_steps: break
            factor, momentum = schedules(step, total, warmup)
            for group, base_lr in zip(optimizer.param_groups, (config.SSL_BACKBONE_LR, config.SSL_MLP_LR)):
                group["lr"] = config.SSL_MIN_LR + (base_lr - config.SSL_MIN_LR) * factor
            first = _prepare_images(images, anchor, "bicubic", device)
            second = _prepare_images(images, partner, "bicubic", device)
            valid = valid.to(device)
            count = valid.sum().float()
            if world > 1: dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                losses, embeddings = execution(first, second)
                loss = (losses * valid).sum() * world / count
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite BYOL loss")
            amp.scale(loss).backward(); amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP)
            scale = amp.get_scale(); amp.step(optimizer); amp.update()
            if amp.get_scale() >= scale: model.update_target(momentum)
            else: skipped += 1
            actual = embeddings[valid]
            stats = torch.cat([(losses.detach() * valid).sum().reshape(1),
                               actual.sum(dim=0), actual.square().sum(dim=0)])
            if world > 1: dist.all_reduce(stats)
            std = (stats[1 + config.SSL_PROJECTION_DIM:] / count -
                   (stats[1:1 + config.SSL_PROJECTION_DIM] / count).square()).clamp_min(0).sqrt().mean()
            loss_sum += float(stats[0]); std_sum += float(std * count)
            anchors += int(count); batches += 1; step += 1
            if rank == 0 and (batches % config.SSL_LOG_EVERY == 0 or max_steps):
                print(f"[byol-step] epoch={epoch+1}/{epochs} step={batches}/{len(sampler)} "
                      f"loss={float(stats[0]/count):.4f} tau={momentum:.6f} anchors={anchors} "
                      f"throughput={anchors/(time.monotonic()-started):.1f}/s", flush=True)
            del first, second, losses, embeddings, loss, actual, stats
        if max_steps is None and (anchors != len(dataset) or batches != len(sampler)):
            raise RuntimeError("SSL sampler did not cover every frame/view exactly once")
        row = {"epoch": epoch + 1, "loss": loss_sum / anchors, "projection_std": std_sum / anchors,
               "anchors": anchors, "optimizer_batches": batches, "amp_skipped_steps": skipped,
               "backbone_lr": optimizer.param_groups[0]["lr"], "mlp_lr": optimizer.param_groups[1]["lr"],
               "ema": momentum, "seconds": time.monotonic() - started}
        history.append(row)
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
        rngs = [None] * world if rank == 0 else None
        if world > 1: dist.gather_object(rng, rngs, dst=0)
        else: rngs = [rng]
        if rank == 0:
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "amp": amp.state_dict(),
                         "step": step, "epoch": epoch + 1, "history": history, "contract": contract, "gpu_count": world,
                         "rngs": rngs}, last)
            print(f"[byol-epoch] {row}", flush=True)
        if world > 1: dist.barrier()
    if history[-1]["projection_std"] < 1e-4: raise RuntimeError("Representation collapse detected; downstream training blocked")
    if rank == 0: atomic_save({"features_state_dict": {key: value.detach().cpu() for key, value in model.online_features.state_dict().items()},
                 "method": "BYOL", "contract": contract, "ssl_epochs": epochs,
                 "train_patients": sorted(set(videos.hospital_id))}, output / "encoder.pt")
    dataset.close()
    if rank == 0: print(f"[byol-complete] encoder={output/'encoder.pt'}", flush=True)
    if world > 1: dist.barrier(); dist.destroy_process_group()
