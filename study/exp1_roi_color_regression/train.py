"""Video-level MLP and patient-group CV ridge, sharing identical feature cohorts."""

import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from study.common.run_video_loss_12h import job_seed
from study.exp6_face_pair_lab_delta.train import _metrics
from .preview_rois import HERE


class MLP(nn.Sequential):
    def __init__(self, dimension):
        super().__init__(nn.Linear(dimension, 64), nn.SiLU(), nn.Dropout(.2),
                         nn.Linear(64, 32), nn.SiLU(), nn.Dropout(.2), nn.Linear(32, 1))


def target_scale(y):
    q1, median, q3 = np.quantile(y, [.25, .5, .75])
    if q3 <= q1:
        raise ValueError("Training target IQR is zero")
    return float(median), float(q3 - q1)


def fit_ridge(x, y, patients, alphas):
    if len(set(patients)) < 5:
        raise ValueError("Five-fold ridge selection needs at least five training patients")
    scores = []
    folds = list(GroupKFold(n_splits=5).split(x, y, groups=patients))
    for alpha in alphas:
        errors = []
        for training, held in folds:
            assert not set(patients[training]) & set(patients[held])
            transform = StandardScaler().fit(x[training])
            median, iqr = target_scale(y[training])
            model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(transform.transform(x[training]), (y[training] - median) / iqr)
            predicted = model.predict(transform.transform(x[held])) * iqr + median
            errors.append(float(np.mean(np.abs(predicted - y[held]))))
        scores.append({"alpha": alpha, "cv_raw_mae": np.mean(errors)})
    chosen = min(scores, key=lambda row: (row["cv_raw_mae"], row["alpha"]))["alpha"]
    transform = StandardScaler().fit(x)
    median, iqr = target_scale(y)
    model = Ridge(alpha=chosen, fit_intercept=True, solver="svd").fit(transform.transform(x), (y - median) / iqr)
    return model, transform, median, iqr, scores


def fit_mlp(x, y, x_val, y_val, device, options, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transform = StandardScaler().fit(x)
    median, iqr = target_scale(y)
    train_x = torch.tensor(transform.transform(x), dtype=torch.float32, device=device)
    train_y = torch.tensor((y - median) / iqr, dtype=torch.float32, device=device)
    val_x = torch.tensor(transform.transform(x_val), dtype=torch.float32, device=device)
    model = MLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=options["scheduler_factor"], patience=options["scheduler_patience"], min_lr=options["min_lr"])
    best, patience, history = np.inf, 0, []
    for epoch in range(1, options["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), options["batch_videos"]):
            selected = order[start:start + options["batch_videos"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(train_x[selected]).squeeze(1), train_y[selected], beta=.5)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite MLP loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_pred = model(train_x).squeeze(1)
            val_pred = model(val_x).squeeze(1)
            train_loss = nn.functional.smooth_l1_loss(train_pred, train_y, beta=.5).item()
            val_loss = nn.functional.smooth_l1_loss(val_pred, torch.tensor((y_val - median) / iqr, dtype=torch.float32, device=device), beta=.5).item()
        train_metrics, val_metrics = _metrics(y, train_pred.cpu().numpy() * iqr + median), _metrics(y_val, val_pred.cpu().numpy() * iqr + median)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_mae": train_metrics["mae"], "val_mae": val_metrics["mae"],
               "train_pearson_r": train_metrics["pearson_r"], "val_pearson_r": val_metrics["pearson_r"],
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if row["val_mae"] < best - 1e-8:
            best, patience = row["val_mae"], 0
            torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        "feature_mean": torch.from_numpy(transform.mean_), "feature_scale": torch.from_numpy(transform.scale_),
                        "target_median": median, "target_iqr": iqr, "input_dimension": x.shape[1], "seed": seed,
                        "best_epoch": epoch}, output / "model.pt")
        else:
            patience += 1
        print(f"[epoch] {output.name} {epoch}/{options['max_epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_MAE={row['val_mae']:.4g} val_r={row['val_pearson_r']:.3f} patience={patience}/{options['patience']}", flush=True)
        scheduler.step(row["val_mae"])
        if patience >= options["patience"]:
            break
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, transform, median, iqr


def train_job(job, device="cpu"):
    protocol = json.loads((HERE / "protocol.json").read_text())
    target, roi_mode, kind = job["target"], job["roi_mode"], job["kind"]
    output = HERE / f"outputs/models/{kind}/{roi_mode}/{target}"
    output.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(HERE / f"outputs/tables/records_{target}.csv", dtype={"hospital_id": str, "video_id": str})
    features = np.load(HERE / f"outputs/tables/features_{target}.npy", allow_pickle=False)[:, job["roi_indices"], :].reshape(len(records), -1)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite effective video features")
    groups = {split: np.flatnonzero(records.split.eq(split)) for split in ("train", "val", "test")}
    y = records.raw_value.to_numpy(float)
    training, val = groups["train"], groups["val"]
    print(f"[job-start] {kind}/{roi_mode}/{target} device={device} dimension={features.shape[1]} "
          f"videos={len(training)}/{len(val)}/{len(groups['test'])}", flush=True)
    if kind == "ridge":
        model, transform, median, iqr, cv = fit_ridge(features[training], y[training], records.hospital_id.to_numpy()[training], protocol["ridge"]["alphas"])
        pd.DataFrame(cv).to_csv(output / "cv_scores.csv", index=False)
        joblib.dump({"model": model, "transform": transform, "target_median": median, "target_iqr": iqr}, output / "model.joblib")
        restored = joblib.load(output / "model.joblib")
        np.testing.assert_array_equal(model.predict(transform.transform(features)), restored["model"].predict(restored["transform"].transform(features)))
        prediction = model.predict(transform.transform(features)) * iqr + median
    else:
        model, transform, median, iqr = fit_mlp(features[training], y[training], features[val], y[val], torch.device(device),
                                              protocol["mlp"], job_seed("regression", target), output)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(transform.transform(features), dtype=torch.float32, device=device)).squeeze(1).cpu().numpy() * iqr + median
    metric_rows = []
    baseline = float(np.median(y[training]))
    for split, positions in groups.items():
        metrics = _metrics(y[positions], prediction[positions])
        metrics = {key: value for key, value in metrics.items() if not key.startswith("direction_")}
        metric_rows.append({"kind": kind, "roi_mode": roi_mode, "target": target, "split": split,
                            "training_median_mae": float(np.mean(np.abs(y[positions] - baseline))), **metrics})
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    records[["hospital_id", "video_id", "split", "raw_value"]].assign(y_true=y, y_pred=prediction).to_csv(output / "predictions.csv", index=False)
    (output / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}) + "\n")
    print(f"[job-complete] {kind}/{roi_mode}/{target}", flush=True)
    return {"kind": kind, "roi_mode": roi_mode, "target": target, "status": "ok"}
