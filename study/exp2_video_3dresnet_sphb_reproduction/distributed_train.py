"""Four-GPU DDP training for the paper residual 3D CNN."""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex

from .config import (
    ALLFRAME_INDEX_DIR,
    EARLY_STOPPING_VAL_MSE,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PREFETCH_FACTOR,
    SEED,
    TRAIN_MICRO_BATCH_SIZE,
    WEIGHT_DECAY,
)
from .data import VideoClipDataset
from .models import PaperResidual3DRegressor, parameter_count
from .plot_results import main as plot_results
from .train import _metrics, _prepare, _sha256


class RankStrideSampler(Sampler):
    """Non-padding sampler for duplicate-free distributed evaluation."""

    def __init__(self, length, rank, world_size):
        self.indices = list(range(rank, length, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _loader(records, index, rank, world_size, train):
    dataset = VideoClipDataset(records, index)
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=SEED, drop_last=False,
        )
        if train
        else RankStrideSampler(len(dataset), rank, world_size)
    )
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_MICRO_BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    return dataset, loader, sampler


@torch.no_grad()
def _evaluate(model, loader, device, world_size):
    model.eval()
    local = []
    for video, target, row_index in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(_prepare(video, device))
        local.extend(
            (int(index), float(actual), float(estimate))
            for index, actual, estimate in zip(
                row_index.numpy(), target.numpy(), prediction.float().cpu().numpy()
            )
        )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    merged = sorted((row for rows in gathered for row in rows), key=lambda row: row[0])
    indices = [row[0] for row in merged]
    truth = np.asarray([row[1] for row in merged])
    prediction = np.asarray([row[2] for row in merged])
    return _metrics(truth, prediction), truth, prediction, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--index-dir", default=ALLFRAME_INDEX_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 4:
        raise RuntimeError(f"Formal reproduction requires four DDP ranks, found {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    records = pd.read_csv(
        os.path.join(args.output_dir, "task_records.csv"),
        dtype={"hospital_id": str, "video_id": str},
    )
    index = FrameOffsetIndex.load(os.path.join(args.index_dir, "frame_offsets.npz"))
    splits = {
        name: records.loc[records["split"].eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    datasets, loaders, samplers = {}, {}, {}
    for name in splits:
        datasets[name], loaders[name], samplers[name] = _loader(
            splits[name], index, rank, world_size, name == "train"
        )
    eager_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        PaperResidual3DRegressor()
    ).to(
        device, memory_format=torch.channels_last_3d
    )
    model = DistributedDataParallel(
        eager_model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False,
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    best_mse, best_state, history = np.inf, None, []
    if rank == 0:
        print(
            f"[ddp-start] world_size={world_size} global_batch={world_size * TRAIN_MICRO_BATCH_SIZE} "
            f"train_videos={len(splits['train'])} parameters={parameter_count(eager_model):,}",
            flush=True,
        )

    for epoch in range(1, MAX_EPOCHS + 1):
        samplers["train"].set_epoch(epoch)
        model.train()
        local_loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
        started = time.perf_counter()
        for video, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(_prepare(video, device))
                loss = torch.nn.functional.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            local_loss_sum[0] += loss.detach().double() * len(target)
            local_loss_sum[1] += len(target)
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        train_mse = float((local_loss_sum[0] / local_loss_sum[1]).cpu())
        val_metrics, _, _, _ = _evaluate(
            model.module, loaders["val"], device, world_size
        )
        elapsed = time.perf_counter() - started
        stop = val_metrics["mse_g_dl2"] < EARLY_STOPPING_VAL_MSE
        if rank == 0:
            history.append({
                "epoch": epoch,
                "train_mse_g_dl2": train_mse,
                "val_mse_g_dl2": val_metrics["mse_g_dl2"],
                "val_rmse_g_dl": val_metrics["rmse_g_dl"],
                "val_mae_g_dl": val_metrics["mae_g_dl"],
                "val_pearson_r": val_metrics["pearson_r"],
                "learning_rate": LEARNING_RATE,
                "train_seconds": elapsed,
            })
            pd.DataFrame(history).to_csv(
                os.path.join(args.output_dir, "history.csv"), index=False
            )
            marker = ""
            if val_metrics["mse_g_dl2"] < best_mse:
                best_mse = val_metrics["mse_g_dl2"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.module.state_dict().items()
                }
                marker = "*"
            print(
                f"[epoch] {epoch:03d}/{MAX_EPOCHS} train_MSE={train_mse:.4f} "
                f"val_MSE={val_metrics['mse_g_dl2']:.4f} "
                f"val_RMSE={val_metrics['rmse_g_dl']:.4f} "
                f"val_r={val_metrics['pearson_r']:.4f} seconds={elapsed:.1f}{marker}",
                flush=True,
            )
        stop_tensor = torch.tensor(int(stop), device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            if rank == 0:
                print(f"[paper-early-stop] val_MSE<{EARLY_STOPPING_VAL_MSE}", flush=True)
            break

    state_container = [best_state]
    dist.broadcast_object_list(state_container, src=0)
    model.module.load_state_dict(state_container[0])
    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        metrics, truth, prediction, indices = _evaluate(
            model.module, loaders[split], device, world_size
        )
        if rank == 0:
            metric_rows.append({"split": split, **metrics})
            selected = splits[split].iloc[indices].reset_index(drop=True)
            for row, actual, estimate in zip(selected.itertuples(index=False), truth, prediction):
                prediction_rows.append({
                    "hospital_id": row.hospital_id, "video_id": row.video_id,
                    "split": split, "y_true_g_dl": actual, "y_pred_g_dl": estimate,
                    "residual_g_dl": estimate - actual,
                })
    if rank == 0:
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
        pd.DataFrame(prediction_rows).to_csv(
            os.path.join(args.output_dir, "video_predictions.csv"), index=False
        )
        checkpoint_path = os.path.join(args.output_dir, "model.pt")
        torch.save({
            "model_state_dict": best_state,
            "model": "paper_residual_3d_cnn",
            "target": "hemoglobin_g_dl",
            "parameters": parameter_count(model.module),
            "distributed_data_parallel_world_size": world_size,
            "global_batch_size": world_size * TRAIN_MICRO_BATCH_SIZE,
            "best_validation_mse_g_dl2": best_mse,
        }, checkpoint_path)
        pd.DataFrame([{
            "model": "paper_residual_3d_cnn", "target": "hemoglobin",
            "status": "ok", "checkpoint": checkpoint_path,
            "checkpoint_sha256": _sha256(checkpoint_path),
        }]).to_csv(os.path.join(args.output_dir, "run_index.csv"), index=False)
        manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["training"]["distributed_data_parallel_world_size"] = world_size
        manifest["training"]["global_batch_size"] = world_size * TRAIN_MICRO_BATCH_SIZE
        manifest["training"]["gradient_accumulation_steps"] = 1
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        plot_results(args.output_dir)
    dist.barrier()
    for dataset in datasets.values():
        dataset.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
