"""Five-view, two-seed dynamic GPU training for Exp2 Aug20 24h."""

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .config import (
    AUGMENT_BRIGHTNESS,
    AUGMENT_CONTRAST,
    AUGMENT_CROP_MIN_SCALE,
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENT_TARGETS,
    FACE_SIZE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    MIN_TRAIN_PATIENTS_PER_CLASS,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    POS_WEIGHT_MAX,
    SEED,
    TARGETS,
    WEIGHT_DECAY,
)
from .models import SingleFrameRGBNet
from .train_eval import (
    _aggregate_event_predictions,
    _binary_metrics,
    _event_metric_rows,
    _optimal_threshold,
    _patient_level_split_for_task,
    _task_split_data,
)

VIEW_NAMES = ("original", "hflip", "center_crop", "brightness", "contrast")
DEFAULT_SEEDS = tuple(SEED + offset for offset in range(2))
_WORKER_FACE = None
_WORKER_MANIFEST = None


def _log(message):
    print(message, flush=True)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _worker_init(features_path, manifest_path):
    global _WORKER_FACE, _WORKER_MANIFEST
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_FACE = np.load(features_path, allow_pickle=True)["face"]
    _WORKER_MANIFEST = pd.read_csv(manifest_path, dtype={"hospital_id": str})


def _train_statistics(face, data, chunk_size=512):
    """Compute sample-weighted RGB statistics without materializing all float frames."""
    video_indices = data["video_indices"].astype(np.int64)
    frame_indices = data["frame_indices"].astype(np.int64)
    channel_sum = np.zeros(3, dtype=np.uint64)
    channel_square_sum = np.zeros(3, dtype=np.uint64)
    pixel_count = 0
    for start in range(0, len(video_indices), chunk_size):
        stop = start + chunk_size
        frames = face[video_indices[start:stop], frame_indices[start:stop]]
        channel_sum += frames.sum(axis=(0, 2, 3), dtype=np.uint64)
        integer_frames = frames.astype(np.uint32)
        channel_square_sum += np.square(integer_frames).sum(
            axis=(0, 2, 3), dtype=np.uint64
        )
        pixel_count += frames.shape[0] * frames.shape[2] * frames.shape[3]
    mean_raw = channel_sum.astype(np.float64) / pixel_count
    variance_raw = channel_square_sum.astype(np.float64) / pixel_count - np.square(mean_raw)
    mean = (mean_raw / 255.0).astype(np.float32)
    std = (np.sqrt(np.maximum(variance_raw, 1e-6)) / 255.0).astype(np.float32)
    return mean, std


class ViewDataset(Dataset):
    def __init__(self, face, data, mean, std, views):
        self.face = face
        self.video_indices = data["video_indices"].astype(np.int64)
        self.frame_indices = data["frame_indices"].astype(np.int64)
        self.labels = data["labels"].astype(np.float32)
        self.row_indices = data["row_indices"].astype(np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.views = tuple(views)

    def __len__(self):
        return len(self.labels) * len(self.views)

    def __getitem__(self, index):
        base_index = index // len(self.views)
        view_name = self.views[index % len(self.views)]
        face = torch.from_numpy(
            self.face[self.video_indices[base_index], self.frame_indices[base_index]]
        ).float().div_(255.0)
        if view_name == "hflip":
            face = torch.flip(face, dims=(-1,))
        elif view_name == "center_crop":
            height, width = face.shape[-2:]
            crop_h = max(8, int(round(height * AUGMENT_CROP_MIN_SCALE)))
            crop_w = max(8, int(round(width * AUGMENT_CROP_MIN_SCALE)))
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
            face = functional.interpolate(
                face[:, top:top + crop_h, left:left + crop_w].unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        elif view_name == "brightness":
            face = (face * (1.0 + AUGMENT_BRIGHTNESS)).clamp_(0.0, 1.0)
        elif view_name == "contrast":
            image_mean = face.mean(dim=(-2, -1), keepdim=True)
            face = ((face - image_mean) * (1.0 + AUGMENT_CONTRAST) + image_mean).clamp_(0.0, 1.0)
        elif view_name != "original":
            raise ValueError(f"Unknown view: {view_name}")
        face = (face - self.mean) / self.std
        return (
            face,
            torch.tensor(self.labels[base_index], dtype=torch.float32),
            torch.tensor(self.row_indices[base_index], dtype=torch.long),
        )


def _loader(face, data, mean, std, views, shuffle):
    return DataLoader(
        ViewDataset(face, data, mean, std, views),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def _train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss, batches, scores, labels_out = 0.0, 0, [], []
    use_amp = device.type == "cuda"
    for faces, labels, _ in loader:
        faces = faces.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(faces)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
        scores.append(torch.sigmoid(logits.detach().float()).cpu().numpy().ravel())
        labels_out.append(labels.detach().float().cpu().numpy().ravel())
    probabilities = np.concatenate(scores) if scores else np.asarray([], dtype=np.float32)
    observed_labels = (
        np.concatenate(labels_out) if labels_out else np.asarray([], dtype=np.float32)
    )
    metrics = _binary_metrics(observed_labels, probabilities)
    return total_loss / max(batches, 1), metrics


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, scores, labels_out, rows = [], [], [], []
    use_amp = device.type == "cuda"
    for faces, labels, row_indices in loader:
        faces = faces.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(faces)
            loss = criterion(logits, labels_device)
        losses.append(float(loss.cpu()))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        labels_out.append(labels.numpy())
        rows.append(row_indices.numpy())
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "probabilities": np.concatenate(scores) if scores else np.asarray([], dtype=np.float32),
        "labels": np.concatenate(labels_out) if labels_out else np.asarray([], dtype=np.float32),
        "row_indices": np.concatenate(rows) if rows else np.asarray([], dtype=np.int64),
    }


def _write_skip(run_dir, seed, target, reason):
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame([{
        "seed": seed,
        "target": target,
        "split": "test",
        "status": "skipped",
        "reason": reason,
    }]).to_csv(os.path.join(run_dir, "event_metrics.csv"), index=False)


def _job(seed, target, run_dir, max_epochs):
    global _WORKER_FACE, _WORKER_MANIFEST
    start_time = time.time()
    job_name = f"seed={seed} task={target}"
    _set_seed(seed)
    face, manifest = _WORKER_FACE, _WORKER_MANIFEST
    os.makedirs(run_dir, exist_ok=True)
    _log(f"[job-start] {job_name}")

    split = _patient_level_split_for_task(manifest, target, seed)
    if split is None:
        reason = "insufficient stratifiable patients"
        _write_skip(run_dir, seed, target, reason)
        _log(f"[job-skip] {job_name}: {reason}")
        return {"seed": seed, "target": target, "run_dir": run_dir, "status": "skipped"}

    data = _task_split_data(manifest, target, split)
    if any(len(data[name]["labels"]) == 0 for name in ("train", "val", "test")):
        reason = "empty patient split"
        _write_skip(run_dir, seed, target, reason)
        _log(f"[job-skip] {job_name}: {reason}")
        return {"seed": seed, "target": target, "run_dir": run_dir, "status": "skipped"}

    train_labels = data["train"]["labels"]
    n_pos, n_neg = int((train_labels > 0.5).sum()), int((train_labels < 0.5).sum())
    patient_frame = manifest.iloc[data["train"]["row_indices"]][["hospital_id"]].copy()
    patient_frame["label"] = train_labels
    grouped = patient_frame.groupby("hospital_id")["label"]
    n_pos_pat = int(grouped.max().gt(0.5).sum())
    n_neg_pat = int(grouped.min().lt(0.5).sum())
    if min(n_pos, n_neg) < MIN_TRAIN_SAMPLES_PER_CLASS or min(n_pos_pat, n_neg_pat) < MIN_TRAIN_PATIENTS_PER_CLASS:
        reason = f"insufficient train class: pos={n_pos}/{n_pos_pat}, neg={n_neg}/{n_neg_pat}"
        _write_skip(run_dir, seed, target, reason)
        _log(f"[job-skip] {job_name}: {reason}")
        return {"seed": seed, "target": target, "run_dir": run_dir, "status": "skipped"}

    mean, std = _train_statistics(face, data["train"])
    loaders = {
        "train": _loader(face, data["train"], mean, std, VIEW_NAMES, True),
        "val": _loader(face, data["val"], mean, std, ("original",), False),
        "test": _loader(face, data["test"], mean, std, ("original",), False),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SingleFrameRGBNet().to(device)
    pos_weight = min(POS_WEIGHT_MAX, max(1.0, n_neg / max(n_pos, 1)))
    _log(
        f"[job-data] {job_name}: "
        f"train={len(data['train']['labels'])}x{len(VIEW_NAMES)}={len(data['train']['labels']) * len(VIEW_NAMES)} "
        f"val={len(data['val']['labels'])} test={len(data['test']['labels'])}; "
        f"pos={n_pos}/{n_pos_pat} patients neg={n_neg}/{n_neg_pat} patients; "
        f"pos_weight={pos_weight:.2f}; device={device}"
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=MIN_LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_state, patience, history = -np.inf, None, 0, []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = _train_epoch(
            model, loaders["train"], optimizer, scaler, criterion, device
        )
        validation = _evaluate(model, loaders["val"], criterion, device)
        validation_metrics = _binary_metrics(validation["labels"], validation["probabilities"])
        validation_auc = validation_metrics["roc_auc"]
        validation_bacc = validation_metrics["balanced_accuracy"]
        score = validation_auc if np.isfinite(validation_auc) else -validation["loss"]
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "seed": seed,
            "target": target,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bacc": train_metrics["balanced_accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": validation["loss"],
            "val_bacc": validation_bacc,
            "val_roc_auc": validation_auc,
            "learning_rate": learning_rate,
            "pos_weight": float(pos_weight),
        })
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        if score > best_score + 1e-4:
            best_score, patience = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_marker = "*"
        else:
            patience += 1
            best_marker = ""
        _log(
            f"[epoch] {job_name} {epoch:03d}/{max_epochs}: "
            f"train_loss={train_loss:.4f} train_AUC={train_metrics['roc_auc']:.4f} "
            f"train_bACC={train_metrics['balanced_accuracy']:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_AUC={validation_auc:.4f} val_bACC={validation_bacc:.4f} "
            f"lr={learning_rate:.2e} patience={patience}/{EARLY_STOPPING_PATIENCE}{best_marker}"
        )
        scheduler.step()
        if patience >= EARLY_STOPPING_PATIENCE:
            _log(f"[early-stop] {job_name}: epoch={epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"{job_name}: no finite checkpoint")
    model.load_state_dict(best_state)
    evaluations = {name: _evaluate(model, loaders[name], criterion, device) for name in ("train", "val", "test")}
    threshold = _optimal_threshold(evaluations["val"]["labels"], evaluations["val"]["probabilities"])
    frame_rows, prediction_rows = [], []
    for split_name, evaluation in evaluations.items():
        metric = _binary_metrics(evaluation["labels"], evaluation["probabilities"], threshold)
        frame_rows.append({
            "seed": seed,
            "target": target,
            "split": split_name,
            "status": "ok",
            **{f"metric_{key}": value for key, value in metric.items()},
        })
        for local_index, row_index in enumerate(evaluation["row_indices"]):
            source = manifest.iloc[int(row_index)]
            score = float(evaluation["probabilities"][local_index])
            prediction_rows.append({
                "seed": seed,
                "target": target,
                "split": split_name,
                "sample_id": source["sample_id"],
                "event_type": source["event_type"],
                "base_event_id": source["base_event_id"],
                "frame_index": int(source["frame_index"]),
                "video_id": source["video_id"],
                "hospital_id": str(source["hospital_id"]),
                "y_true": int(evaluation["labels"][local_index]),
                "score": score,
                "threshold": float(threshold),
                "y_pred": int(score >= threshold),
                "match_delta_h": float(source["match_delta_h"]),
            })
    predictions = pd.DataFrame(prediction_rows)
    event_predictions = _aggregate_event_predictions(predictions)
    event_metrics = _event_metric_rows(event_predictions)
    event_metrics.insert(0, "seed", seed)
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    pd.DataFrame(frame_rows).to_csv(os.path.join(run_dir, "frame_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    event_predictions.to_csv(os.path.join(run_dir, "event_predictions.csv"), index=False)
    event_metrics.to_csv(os.path.join(run_dir, "event_metrics.csv"), index=False)
    pd.DataFrame([{"seed": seed, "target": target, "view": view} for view in VIEW_NAMES]).to_csv(
        os.path.join(run_dir, "views.csv"), index=False
    )
    torch.save({
        "model_state_dict": best_state,
        "seed": seed,
        "target": target,
        "threshold": threshold,
        "rgb_mean": mean,
        "rgb_std": std,
        "views": VIEW_NAMES,
    }, os.path.join(run_dir, "model.pt"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_event = event_metrics[(event_metrics["target"].eq(target)) & (event_metrics["split"].eq("test"))]
    if not test_event.empty:
        row = test_event.iloc[0]
        _log(
            f"[job-done] {job_name}: "
            f"test_event_auc={row.get('metric_roc_auc', np.nan):.4f} "
            f"test_event_bacc={row.get('metric_balanced_accuracy', np.nan):.4f} "
            f"elapsed_min={(time.time() - start_time) / 60.0:.1f}"
        )
    else:
        _log(f"[job-done] {job_name}: elapsed_min={(time.time() - start_time) / 60.0:.1f}")
    return {"seed": seed, "target": target, "run_dir": run_dir, "status": "ok"}


def _plot_histories(history, output_dir):
    if history.empty:
        return
    tasks = sorted(history.target.unique())
    columns = min(3, len(tasks))
    rows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    for axis, target in zip(axes.flatten(), tasks):
        task_history = history[history.target.eq(target)]
        for _, seed_history in task_history.groupby("seed"):
            axis.plot(seed_history.epoch, seed_history.train_loss, color="tab:blue", alpha=0.25)
            axis.plot(seed_history.epoch, seed_history.val_loss, color="tab:red", alpha=0.35)
        axis.set_title(target, fontsize=9)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        second_axis = axis.twinx()
        for _, seed_history in task_history.groupby("seed"):
            second_axis.plot(
                seed_history.epoch, seed_history.train_bacc,
                color="tab:green", linestyle="--", alpha=0.25,
            )
            second_axis.plot(
                seed_history.epoch, seed_history.train_roc_auc,
                color="tab:purple", linestyle="--", alpha=0.25,
            )
            second_axis.plot(seed_history.epoch, seed_history.val_bacc, color="tab:green", alpha=0.35)
            second_axis.plot(seed_history.epoch, seed_history.val_roc_auc, color="tab:purple", alpha=0.35)
        second_axis.set_ylim(-0.05, 1.05)
        second_axis.set_ylabel("Score")
        proxy_lines = [
            plt.Line2D([0], [0], color="tab:blue", label="Train loss"),
            plt.Line2D([0], [0], color="tab:red", label="Val loss"),
            plt.Line2D([0], [0], color="tab:green", linestyle="--", label="Train bACC"),
            plt.Line2D([0], [0], color="tab:purple", linestyle="--", label="Train ROC-AUC"),
            plt.Line2D([0], [0], color="tab:green", label="Val bACC"),
            plt.Line2D([0], [0], color="tab:purple", label="Val ROC-AUC"),
        ]
        axis.legend(handles=proxy_lines, fontsize=7, loc="best")
    for axis in axes.flatten()[len(tasks):]:
        axis.set_visible(False)
    figure.suptitle("Exp2 Aug20 24h Native-Resolution Hemoglobin Training", fontsize=14, y=1.01)
    figure.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    _log(f"Loss and metric curves saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input-dir", default=os.path.join(exp_dir, "outputs_aug20_24h"))
    parser.add_argument("--output-dir", default=os.path.join(exp_dir, "outputs_aug20_24h_views5_seed5"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--tasks", default=",".join(EXPERIMENT_TARGETS))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    tasks = tuple(value for value in args.tasks.split(",") if value)
    os.makedirs(args.output_dir, exist_ok=True)
    features_path = os.path.join(args.input_dir, "features.npz")
    manifest_path = os.path.join(args.input_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path, dtype={"hospital_id": str})
    with np.load(features_path, allow_pickle=True) as features:
        face_shape = features["face"].shape
    jobs = [
        (seed, target, os.path.join(args.output_dir, "runs", f"seed_{seed}", target), args.max_epochs)
        for seed in seeds
        for target in tasks
    ]
    workers = max(1, min(args.workers, len(jobs)))
    _log(
        f"Dynamic scheduler: {len(jobs)} jobs, {workers} GPU workers; "
        f"input_rows={len(manifest)} face_shape={face_shape}; "
        f"views={','.join(VIEW_NAMES)} seeds={','.join(str(seed) for seed in seeds)}"
    )
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(features_path, manifest_path),
    ) as executor:
        futures = [executor.submit(_job, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _log(f"[scheduler] {completed}/{len(jobs)} seed={result['seed']} task={result['target']} {result['status']}")

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "job_index.csv"), index=False)
    histories, event_metrics = [], []
    for result in results:
        run_dir = result["run_dir"]
        for name, bucket in (("history.csv", histories), ("event_metrics.csv", event_metrics)):
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                bucket.append(pd.read_csv(path))
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    metrics = pd.concat(event_metrics, ignore_index=True) if event_metrics else pd.DataFrame()
    history.to_csv(os.path.join(args.output_dir, "loss_all.csv"), index=False)
    metrics.to_csv(os.path.join(args.output_dir, "metrics_per_seed.csv"), index=False)
    metric_columns = [column for column in metrics.columns if column.startswith("metric_")]
    if not metrics.empty and metric_columns:
        ok_metrics = metrics[metrics.status.eq("ok")]
        summary = ok_metrics.groupby(["target", "split"])[metric_columns].agg(["mean", "std", "count"])
        summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
        summary = summary.reset_index()
    else:
        summary = pd.DataFrame()
    summary.to_csv(os.path.join(args.output_dir, "metrics_mean.csv"), index=False)
    _plot_histories(history, args.output_dir)
    _log(f"Saved two-seed results to {args.output_dir}")


if __name__ == "__main__":
    main()
