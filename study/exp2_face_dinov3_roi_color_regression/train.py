"""Independent frozen-feature heads with equal-video-weight frame loss."""

import json
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from study.common.run_video_loss_12h import job_seed
from study.exp2_face_dinov3_frozen.backbone import Head
from study.exp2_face_dinov3_frozen.run_main_regression import RESULTS as DINO_RESULTS
from study.exp2_face_pretrained_head32_regression import config
from study.exp2_face_pretrained_head32_regression.scaling import fit_robust_target_scaler, RobustTargetScaler
from study.exp6_face_pair_lab_delta.train import _metrics
from .data import load_features, fit_color_scaler, lab_batches, pooled_prediction
from .models import FusionHead


@torch.no_grad()
def predict_frames(model, features):
    model.eval()
    return torch.cat([model(features[start:start + 4096]).float().reshape(-1) for start in range(0, len(features), 4096)]).cpu().numpy()


def train_job(target, index, records, device, output, contract, max_epochs=80):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed = job_seed("regression", target)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    features, ptr, frames = load_features(index, records)
    groups = {split: np.flatnonzero(records.split.eq(split)) for split in ("train", "val", "test")}
    color = fit_color_scaler(features, ptr, groups["train"])
    scaler = fit_robust_target_scaler(target, records, config.SCORE_DEFINITIONS[target]["unit"])
    y = records.raw_value.to_numpy(float)
    labels = scaler.transform(y).astype(np.float32)
    video_rows = np.repeat(np.arange(len(records)), np.diff(ptr))
    x = torch.tensor(features.reshape(-1, 548), device=device)
    original_x = x[::5]
    frame_labels = torch.tensor(np.repeat(labels[video_rows], 5), device=device)
    weights = torch.tensor(np.repeat(1. / np.diff(ptr)[video_rows], 5), dtype=torch.float32, device=device)
    train_records = records.iloc[groups["train"]].reset_index(drop=True)
    train_ptr = np.concatenate(([0], np.cumsum(np.diff(ptr)[groups["train"]])))
    train_frames = np.concatenate([np.arange(ptr[row], ptr[row + 1]) for row in groups["train"]])
    train_samples = (train_frames[:, None] * 5 + np.arange(5)).reshape(-1)
    model = FusionHead(color.mean_, color.scale_).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, stale, history = np.inf, 0, []
    print(f"[job-start] fusion/{target} {device} videos=" + "/".join(str(len(groups[s])) for s in groups)
          + f" frames={len(frames)} views=5 parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total, mass, batches = 0., 0., 0
        for sample in lab_batches(train_records, train_ptr):
            selected = torch.tensor(train_samples[sample], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss_each = F.smooth_l1_loss(model(x[selected]).float(), frame_labels[selected], beta=.5, reduction="none")
                loss = (loss_each * weights[selected]).sum() / weights[selected].sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            amp.step(optimizer)
            amp.update()
            batch_mass = weights[selected].sum().item()
            total += loss.item() * batch_mass
            mass += batch_mass
            batches += 1
        predicted_scaled = predict_frames(model, original_x)
        predicted = scaler.inverse_transform(pooled_prediction(predicted_scaled, ptr))
        train_metric = _metrics(y[groups["train"]], predicted[groups["train"]])
        val_metric = _metrics(y[groups["val"]], predicted[groups["val"]])
        frame_error = F.smooth_l1_loss(torch.from_numpy(predicted_scaled), torch.from_numpy(labels[video_rows]), beta=.5, reduction="none").numpy()
        val_loss = np.mean([frame_error[ptr[row]:ptr[row + 1]].mean() for row in groups["val"]])
        row = {"target": target, "epoch": epoch, "train_loss": total / mass, "val_loss": val_loss,
               "train_mae": train_metric["mae"], "val_mae": val_metric["mae"], "train_pearson_r": train_metric["pearson_r"],
               "val_pearson_r": val_metric["pearson_r"], "learning_rate": optimizer.param_groups[0]["lr"],
               "batches": batches, "train_frame_views": len(train_samples), "seconds": time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if row["val_mae"] < best - 1e-8:
            best, stale = row["val_mae"], 0
            torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        "target_scaler": scaler.to_dict(), "seed": seed, "best_epoch": epoch, "contract": contract,
                        "parameters": 17665, "views": 5, "encoder_frozen": True}, output / "model.pt")
        else:
            stale += 1
        print(f"[epoch] {target} {epoch}/{max_epochs} train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} "
              f"train_r={row['train_pearson_r']:.3f} val_r={row['val_pearson_r']:.3f} val_MAE={row['val_mae']:.5g} "
              f"lr={row['learning_rate']:.3g} patience={stale}/12 seconds={row['seconds']:.2f}", flush=True)
        scheduler.step()
        if stale >= 12:
            break
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    predicted = scaler.inverse_transform(pooled_prediction(predict_frames(model, original_x), ptr))
    checkpoint = torch.load(DINO_RESULTS / f"runs/{target}/model.pt", map_location="cpu", weights_only=True)
    baseline = Head(32).to(device)
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline_scaled = pooled_prediction(predict_frames(baseline, original_x[:, :384]), ptr)
    baseline_prediction = RobustTargetScaler(**checkpoint["target_scaler"]).inverse_transform(baseline_scaled)
    metric_rows = []
    for name, values in (("DINO + ROI", predicted), ("DINO-only (matched frames)", baseline_prediction)):
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite raw prediction")
        for split, positions in groups.items():
            metric = {key: value for key, value in _metrics(y[positions], values[positions]).items() if not key.startswith("direction_")}
            metric_rows.append({"model": name, "target": target, "split": split, **metric})
    prediction = records[["hospital_id", "video_id", "split", "clinical_event_id"]].assign(
        y_true=y, y_pred=predicted, dino_only_pred=baseline_prediction, frame_count=np.diff(ptr))
    prediction.to_csv(output / "video_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    np.savez_compressed(output / "selected_frames.npz", global_frame_indices=frames, source_frame_indices=index.source_indices[frames], video_ptr=ptr)
    (output / "job_complete.json").write_text(json.dumps({"contract": contract}) + "\n")
    print(f"[job-complete] fusion/{target} best_epoch={saved['best_epoch']}", flush=True)
