"""Video-level regression from frozen estimated visible-spectrum features."""

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from study.exp2_face_pretrained_head32_regression.data import validate_source_data

from .config import (
    BASE_OUTPUT, BATCH_SIZE, FEATURE_SHAPE, HEAD_EPOCHS, INDEX_PATH,
    LEARNING_RATE, MATCHING_HOURS, MIN_LEARNING_RATE, OUTPUT, PATIENCE, SEED,
    SMOOTH_L1_BETA, TARGETS, VARIANT, WEIGHT_DECAY,
)
from .extract_features import load_or_extract_features
from .spectral import checkpoint_sha256, sha256


class SpectralRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(int(np.prod(FEATURE_SHAPE))),
            nn.Linear(int(np.prod(FEATURE_SHAPE)), 64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _load_task(target, index, features, scaler):
    path = BASE_OUTPUT / "task_records" / f"{target}.csv"
    records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    if set(records.split) != {"train", "val", "test"}:
        raise RuntimeError(f"Invalid saved split for {target}")
    if records.video_id.duplicated().any():
        raise RuntimeError(f"Duplicate video labels for {target}")
    if (not np.isfinite(records[["raw_value", "robust_scaled_raw_value", "match_delta_h"]]).all().all()
            or not records.match_delta_h.between(0, MATCHING_HOURS + 1e-9).all()):
        raise RuntimeError(f"Invalid labels or records outside the {MATCHING_HOURS}h window: {target}")
    control_path = (BASE_OUTPUT / "runs" / "efficientnet_b0" / target
                    / "video_predictions.csv")
    control = pd.read_csv(control_path, dtype={"hospital_id": str, "video_id": str})
    comparable = records[["hospital_id", "video_id", "split", "raw_value"]].merge(
        control[["hospital_id", "video_id", "split", "raw_value"]],
        on=["hospital_id", "video_id", "split"], how="outer",
        validate="one_to_one", indicator=True, suffixes=("_task", "_control"),
    )
    if (not comparable["_merge"].eq("both").all()
            or not np.allclose(comparable.raw_value_task,
                               comparable.raw_value_control, atol=1e-6)):
        raise RuntimeError(f"Saved RGB control is not paired with current {target} records")
    patient_splits = records.groupby("hospital_id").split.nunique()
    if patient_splits.max() != 1:
        raise RuntimeError(f"Patient leakage in saved {target} split")
    if not set(records.video_id).issubset(index.video_lookup):
        raise RuntimeError(f"Task videos missing from index for {target}")
    train = records.loc[records.split.eq("train")]
    if len(train) != scaler["train_videos"]:
        raise RuntimeError(f"Training count differs from reference scaler for {target}")
    med, iqr = float(scaler["median"]), float(scaler["iqr"])
    expected = (records.raw_value.to_numpy(np.float64) - med) / iqr
    if not np.allclose(expected, records.robust_scaled_raw_value.to_numpy(np.float64),
                       atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Saved raw values and robust-scaled labels disagree for {target}")
    video_features = []
    for video_id in records.video_id:
        start, end = index.frame_range(video_id)
        if end - start != 20:
            raise RuntimeError(f"Unexpected frame count for {video_id}")
        video_features.append(np.asarray(features[start:end], dtype=np.float32).mean(axis=0))
    x = np.stack(video_features)
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite spectral features for {target}")
    return records, x, path


def _predict(model, x, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            model(torch.from_numpy(x[start:start + 256]).to(device)).cpu().numpy()
            for start in range(0, len(x), 256)
        ])


def _metrics(actual, predicted):
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "pearson_r": (float(pearsonr(actual, predicted).statistic)
                      if np.std(actual) > 0 and np.std(predicted) > 0 else np.nan),
    }


def train_target(target, target_index, index, features, scaler, device):
    records, x, records_path = _load_task(target, index, features, scaler)
    y = records.robust_scaled_raw_value.to_numpy(np.float32)
    train_idx = np.flatnonzero(records.split.eq("train"))
    val_idx = np.flatnonzero(records.split.eq("val"))
    run_dir = OUTPUT / "runs" / target
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = SEED + target_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SpectralRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HEAD_EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    train_dataset = TensorDataset(torch.from_numpy(x[train_idx]),
                                  torch.from_numpy(y[train_idx]))
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    median, iqr = float(scaler["median"]), float(scaler["iqr"])
    best_mae = float("inf")
    best_epoch = 0
    history = []
    print(
        f"[target-start] {target} videos={records.split.value_counts().to_dict()} "
        f"patients={records.groupby('split').hospital_id.nunique().to_dict()} "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(inputs)
        val_scaled = _predict(model, x[val_idx], device)
        val_loss = float(nn.functional.smooth_l1_loss(
            torch.from_numpy(val_scaled), torch.from_numpy(y[val_idx]),
            beta=SMOOTH_L1_BETA,
        ))
        val_mae = float(mean_absolute_error(
            records.raw_value.to_numpy(np.float64)[val_idx],
            val_scaled * iqr + median,
        ))
        row = {
            "target": target, "epoch": epoch,
            "train_loss": loss_sum / len(train_dataset),
            "val_loss": val_loss, "val_mae": val_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        if val_mae < best_mae - 1e-8:
            best_mae, best_epoch = val_mae, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "target": target,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                "scaler": scaler,
                "mstpp_weight_sha256": checkpoint_sha256(),
                "spectral_variant": VARIANT,
                "frame_index_sha256": sha256(INDEX_PATH),
                "task_records_sha256": sha256(records_path),
            }, run_dir / "best_model.pt")
        scheduler.step()
        print(
            f"[epoch] {target} {epoch}/{HEAD_EPOCHS} "
            f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_mae={val_mae:.3f} best_epoch={best_epoch}", flush=True,
        )
        if epoch - best_epoch >= PATIENCE:
            break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = []
    predictions = []
    for split in ("train", "val", "test"):
        selected = np.flatnonzero(records.split.eq(split))
        raw_predictions = _predict(model, x[selected], device) * iqr + median
        raw_actual = records.raw_value.to_numpy(np.float64)[selected]
        rows.append({"target": target, "split": split, **_metrics(raw_actual, raw_predictions)})
        frame = records.iloc[selected][["hospital_id", "video_id", "split", "raw_value"]].copy()
        frame["target"] = target
        frame["predicted_raw_value"] = raw_predictions
        predictions.append(frame)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    print(f"[target-complete] {target} best_epoch={best_epoch} test={rows[-1]}", flush=True)
    return rows, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_source_data(BASE_OUTPUT / "source_data", expected_max_delta_hours=MATCHING_HOURS)
    if not (BASE_OUTPUT / "COMPLETE").is_file():
        raise RuntimeError("The paired 12h RGB control is incomplete")
    if args.check_only:
        from .config import CACHE, GRID_SIZE
        if not (CACHE / f"mstpp_31band_grid{GRID_SIZE}x{GRID_SIZE}.npy").is_file():
            raise RuntimeError("CPU-only checks require the existing spectral cache")
        index, features = load_or_extract_features(device="cpu")
        scalers = json.loads((BASE_OUTPUT / "target_scalers.json").read_text())["targets"]
        for target in TARGETS:
            records, _, _ = _load_task(target, index, features, scalers[target])
            print(f"[check-ok] {target} videos={records.split.value_counts().to_dict()} "
                  f"patients={records.groupby('split').hospital_id.nunique().to_dict()}", flush=True)
        return
    if (OUTPUT / "COMPLETE").exists():
        raise RuntimeError("Exp8 is already complete; refusing to overwrite its results")
    index, features = load_or_extract_features()
    scaler_data = json.loads((BASE_OUTPUT / "target_scalers.json").read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matching_hours": MATCHING_HOURS, "source_resolution": 224,
        "base_output": str(BASE_OUTPUT), "frame_index_sha256": sha256(INDEX_PATH),
        "task_records_sha256": {target: sha256(BASE_OUTPUT / f"task_records/{target}.csv") for target in TARGETS},
        "target_scalers_sha256": sha256(BASE_OUTPUT / "target_scalers.json"),
        "mstpp_weight_sha256": checkpoint_sha256(), "spectral_variant": VARIANT,
        "feature_shape": list(FEATURE_SHAPE), "frames_per_video": 20,
        "split_policy": "reuse paired native224 12h RGB control; patient-disjoint",
        "seed": SEED, "learning_rate": LEARNING_RATE, "min_learning_rate": MIN_LEARNING_RATE,
        "max_epochs": HEAD_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY, "smooth_l1_beta": SMOOTH_L1_BETA,
    }
    (OUTPUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metrics, histories = [], []
    for target_index, target in enumerate(TARGETS):
        task_metrics, task_history = train_target(
            target, target_index, index, features, scaler_data["targets"][target], device
        )
        metrics.extend(task_metrics)
        histories.extend(task_history)
    pd.DataFrame(metrics).to_csv(OUTPUT / "metrics_all.csv", index=False)
    pd.DataFrame(histories).to_csv(OUTPUT / "history_all.csv", index=False)
    from .plot_results import plot_results
    plot_results()
    (OUTPUT / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[exp8-complete] output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
