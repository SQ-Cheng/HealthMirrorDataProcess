"""CABG postoperative time-only controls for the history-only regressor."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
import torch
from torch import nn
from torch.utils.data import Dataset

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.common.time_alignment import local_naive_to_unix
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler
from study.exp5_face_pair_recovery.build_dataset import load_cabg_episodes

from .config import (
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    OUTPUT_DIR,
    REFERENCE_DIR,
    SCORE_DEFINITIONS,
    SEED,
    SMOOTH_L1_BETA,
    TARGETS,
)
from .data import load_task
from .models import HistoryOnlyRegressor, parameter_counts
from .train import (
    _evaluate,
    _loader,
    _plot_history,
    _raw_evaluation,
    _run_stage,
    regression_metrics,
    train_task,
)


EXPERIMENT_DIR = OUTPUT_DIR / "trajectory_ablations"
NEURAL_VARIANTS = ("surgery_time_mlp", "sampling_times_mlp")
SPLINE_VARIANTS = ("surgery_time_spline", "sampling_times_spline")
VARIANTS = ("matched_history_only", *NEURAL_VARIANTS, *SPLINE_VARIANTS)


class SurgeryTimeMLP(nn.Module):
    """A 993-parameter time-only counterpart to the original 993-parameter model."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 24), nn.LayerNorm(24), nn.SiLU(),
            nn.Linear(24, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Dropout(0.25), nn.Linear(32, 1),
        )

    def forward(self, history, mask):
        return self.network(history[:, 0, :])


class SamplingTimesMLP(nn.Module):
    """The original mean-pooling architecture with its value channel removed."""

    def __init__(self):
        super().__init__()
        self.measurement_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16),
            nn.LayerNorm(16), nn.SiLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Dropout(0.25), nn.Linear(32, 1),
        )

    def forward(self, history, mask):
        encoded = self.measurement_mlp(history)
        valid = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.regressor(pooled)


class TemporalDataset(Dataset):
    def __init__(self, records, features, masks):
        self.records = records.reset_index(drop=True)
        self.features = features.astype(np.float32, copy=False)
        self.mask = masks.astype(np.bool_, copy=False)
        self.labels = self.records.robust_scaled_raw_value.to_numpy(np.float32)
        self.history_count = self.records.history_count.to_numpy(np.int64)
        if len(self.records) != len(self.features) or self.features.shape[:2] != self.mask.shape:
            raise ValueError("Temporal feature/record alignment mismatch")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.mask[index]),
            torch.tensor(self.labels[index]),
            torch.tensor(index, dtype=torch.long),
        )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cabg_episodes():
    episodes, audit = load_cabg_episodes()
    keys = ["hospital_id", "admission_time", "discharge_time"]
    valid = audit.loc[audit.valid_event & audit.is_cabg].copy()
    first = valid.loc[valid.position.eq(valid.groupby(keys).position.transform("min"))]
    conflicts = first.groupby(keys).surgery_end.nunique().gt(1)
    bad_keys = set(conflicts.index[conflicts])
    episodes = episodes.loc[
        ~episodes[keys].apply(tuple, axis=1).isin(bad_keys)
    ].copy()
    for column in ("admission_time", "discharge_time", "surgery_end"):
        episodes[f"{column}_unix"] = episodes[column].map(local_naive_to_unix)
    if episodes.duplicated(["hospital_id", "admission_time_unix", "discharge_time_unix"]).any():
        raise AssertionError("Duplicate CABG episode anchors")
    return episodes, len(bad_keys)


def _make_cohort(target, episodes):
    records, history = load_task(REFERENCE_DIR, target)
    summary = pd.read_csv(
        REFERENCE_DIR / "history_records" / f"{target}_summary.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    if summary.video_id.duplicated().any():
        raise AssertionError(f"Duplicate history summaries for {target}")
    merged = records.merge(
        summary[[
            "hospital_id", "video_id", "split", "current_lab_time_unix",
            "episode_admission_time_unix", "episode_discharge_time_unix",
            "history_count",
        ]], on=["hospital_id", "video_id", "split"], validate="one_to_one"
    )
    if len(merged) != len(records):
        raise AssertionError(f"Missing reference history summaries for {target}")
    merged = merged.merge(
        episodes[[
            "hospital_id", "admission_time_unix", "discharge_time_unix",
            "surgery_end_unix",
        ]],
        left_on=["hospital_id", "episode_admission_time_unix", "episode_discharge_time_unix"],
        right_on=["hospital_id", "admission_time_unix", "discharge_time_unix"],
        how="inner", validate="many_to_one",
    )
    matched_cabg = len(merged)
    merged = merged.loc[
        merged.current_lab_time_unix.ge(merged.surgery_end_unix)
        & merged.current_lab_time_unix.le(merged.episode_discharge_time_unix)
    ].copy()
    postoperative = len(merged)
    merged = merged.sort_values(["match_delta_h", "video_id"], kind="stable")
    merged = merged.drop_duplicates(
        ["hospital_id", "current_lab_time_unix"], keep="first"
    ).sort_values(["hospital_id", "current_lab_time_unix", "video_id"])
    merged = merged.reset_index(drop=True)
    merged["hours_since_cabg"] = (
        merged.current_lab_time_unix - merged.surgery_end_unix
    ) / 3600.0
    merged["log_days_since_cabg"] = np.log1p(merged.hours_since_cabg / 24.0)
    if not np.isfinite(merged.log_days_since_cabg).all():
        raise AssertionError(f"Invalid CABG-relative times for {target}")
    if set(merged.split) != {"train", "val", "test"}:
        raise RuntimeError(f"Missing split after CABG filtering for {target}")
    if merged.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError(f"Patient leakage after CABG filtering for {target}")
    audit = {
        "target": target, "reference_videos": len(records),
        "matched_cabg_episode": matched_cabg,
        "postoperative_videos": postoperative,
        "unique_lab_events": len(merged),
        "deduplicated_videos": postoperative - len(merged),
        **{f"{split}_videos": int(merged.split.eq(split).sum())
           for split in ("train", "val", "test")},
        "reference_task_sha256": _sha256(
            REFERENCE_DIR / "task_records" / f"{target}.csv"
        ),
        "reference_history_sha256": _sha256(
            REFERENCE_DIR / "history_records" / f"{target}.npz"
        ),
    }
    return merged, history, audit


def _feature_arrays(records, store, variant, time_center=0.0, time_scale=1.0):
    if variant == "surgery_time_mlp":
        values = (records.log_days_since_cabg.to_numpy(float) - time_center) / time_scale
        return values.astype(np.float32).reshape(-1, 1, 1), np.ones((len(records), 1), bool)
    lookup = store.lookup()
    lengths = []
    for video_id in records.video_id.astype(str):
        row = lookup[video_id]
        lengths.append(int(store.offsets[row + 1] - store.offsets[row]))
    width = max(1, max(lengths))
    features = np.zeros((len(records), width, 1), dtype=np.float32)
    masks = np.zeros((len(records), width), dtype=np.bool_)
    for index, (video_id, count) in enumerate(zip(records.video_id.astype(str), lengths)):
        if count:
            row = lookup[video_id]
            features[index, :count, 0] = store.features[
                store.offsets[row]:store.offsets[row + 1], 1
            ]
            masks[index, :count] = True
    if not np.array_equal(np.asarray(lengths), records.history_count.to_numpy(int)):
        raise AssertionError("History counts differ from source summary")
    return features, masks


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _neural_run(target, records, store, scaler, variant, root, device, seed,
                head_epochs, finetune_epochs):
    _seed(seed)
    model = SurgeryTimeMLP() if variant == "surgery_time_mlp" else SamplingTimesMLP()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 900 <= parameter_count <= 1100:
        raise AssertionError(f"Unmatched model size: {variant}={parameter_count}")
    model = model.to(device)
    train_time = records.loc[records.split.eq("train"), "log_days_since_cabg"].to_numpy(float)
    time_center = float(np.median(train_time))
    time_scale = float(np.subtract(*np.quantile(train_time, [0.75, 0.25])))
    time_scale = max(time_scale, 1e-6)
    datasets, loaders = {}, {}
    for split in ("train", "val", "test"):
        subset = records.loc[records.split.eq(split)].reset_index(drop=True)
        features, masks = _feature_arrays(
            subset, store, variant, time_center, time_scale
        )
        datasets[split] = TemporalDataset(subset, features, masks)
        loaders[split] = _loader(datasets[split], False, seed)
    loaders["train_shuffle"] = _loader(datasets["train"], True, seed)
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA, reduction="none")
    run_dir = root / "runs" / target / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    print(
        f"[job-start] target={target} variant={variant} device={device} "
        f"parameters={parameter_count} videos={len(datasets['train'])}/"
        f"{len(datasets['val'])}/{len(datasets['test'])}", flush=True,
    )
    head_state, head_mae = _run_stage(
        "head", model, loaders, datasets, criterion, device, scaler,
        HEAD_LEARNING_RATE, head_epochs, HEAD_PATIENCE, history_rows,
        run_dir, target, architecture=variant,
    )
    torch.save(head_state, run_dir / "stage_head_best.pt")
    model.load_state_dict(head_state, strict=True)
    finetune_state, finetune_mae = _run_stage(
        "finetune", model, loaders, datasets, criterion, device, scaler,
        FINETUNE_LEARNING_RATE, finetune_epochs, FINETUNE_PATIENCE,
        history_rows, run_dir, target, architecture=variant,
    )
    torch.save(finetune_state, run_dir / "stage_finetune_best.pt")
    selected = "finetune" if finetune_mae <= head_mae else "head"
    best_state = finetune_state if selected == "finetune" else head_state
    model.load_state_dict(best_state, strict=True)
    torch.save({
        "model_state_dict": best_state, "target": target,
        "architecture": variant, "selected_stage": selected,
        "parameter_count": parameter_count, "target_scaler": scaler.to_dict(),
        "time_center": time_center, "time_scale": time_scale,
    }, run_dir / "model.pt")
    _plot_history(history_rows, run_dir / "history.png", target, variant)
    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        evaluation = _evaluate(model, loaders[split], criterion, device)
        metrics, selected_records, y_scaled, pred_scaled, y_true, y_pred = _raw_evaluation(
            evaluation, datasets[split], target, scaler
        )
        metric_rows.append({
            "architecture": variant, "target": target, "split": split,
            "selected_stage": selected, "parameters": parameter_count, **metrics,
        })
        prediction_rows.append(pd.DataFrame({
            "architecture": variant, "target": target, "split": split,
            "hospital_id": selected_records.hospital_id.astype(str),
            "video_id": selected_records.video_id.astype(str),
            "current_lab_time_unix": selected_records.current_lab_time_unix,
            "hours_since_cabg": selected_records.hours_since_cabg,
            "history_count": selected_records.history_count,
            "y_true": y_true, "y_pred": y_pred,
            "y_true_scaled": y_scaled, "y_pred_scaled": pred_scaled,
        }))
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    print(
        f"[job-complete] target={target} variant={variant} "
        f"test_MAE={metrics_frame.loc[metrics_frame.split.eq('test'), 'mae'].iloc[0]:.4f}",
        flush=True,
    )
    return metrics_frame


def _spline_design(records, store, variant, transformer):
    if variant == "surgery_time_spline":
        return transformer.transform(records.log_days_since_cabg.to_numpy(float).reshape(-1, 1))
    lookup = store.lookup()
    output = np.zeros((len(records), transformer.n_features_out_), dtype=np.float64)
    for index, video_id in enumerate(records.video_id.astype(str)):
        row = lookup[video_id]
        lags = store.features[store.offsets[row]:store.offsets[row + 1], 1]
        if len(lags):
            output[index] = transformer.transform(lags.reshape(-1, 1)).mean(axis=0)
    return output


def _spline_run(target, records, store, scaler, variant, root):
    run_dir = root / "runs" / target / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        split: records.loc[records.split.eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    if variant == "surgery_time_spline":
        fit_values = groups["train"].log_days_since_cabg.to_numpy(float)
    else:
        lookup = store.lookup()
        fit_values = np.concatenate([
            store.features[store.offsets[lookup[video_id]]:
                           store.offsets[lookup[video_id] + 1], 1]
            for video_id in groups["train"].video_id.astype(str)
        ])
    if len(np.unique(fit_values)) < 4:
        raise RuntimeError(f"Insufficient unique training times for {target}/{variant}")
    best = None
    for knots in (4, 6, 8):
        transformer = SplineTransformer(
            n_knots=knots, degree=3, knots="quantile",
            extrapolation="constant", include_bias=False,
        ).fit(fit_values.reshape(-1, 1))
        train_x = _spline_design(groups["train"], store, variant, transformer)
        val_x = _spline_design(groups["val"], store, variant, transformer)
        y_train = groups["train"].robust_scaled_raw_value.to_numpy(float)
        y_val = groups["val"].raw_value.to_numpy(float)
        for alpha in (0.01, 1.0, 100.0):
            model = Ridge(alpha=alpha).fit(train_x, y_train)
            val_pred = scaler.inverse_transform(model.predict(val_x))
            val_mae = float(np.mean(np.abs(y_val - val_pred)))
            if best is None or val_mae < best[0]:
                best = (val_mae, knots, alpha, transformer, model)
    val_mae, knots, alpha, transformer, model = best
    from joblib import dump

    dump({
        "transformer": transformer, "regressor": model,
        "target_scaler": scaler.to_dict(), "variant": variant,
    }, run_dir / "model.joblib")
    metric_rows, prediction_rows = [], []
    for split, group in groups.items():
        x = _spline_design(group, store, variant, transformer)
        y_true = group.raw_value.to_numpy(float)
        y_pred = scaler.inverse_transform(model.predict(x))
        metrics = regression_metrics(
            y_true, y_pred, group.score_threshold.to_numpy(float),
            SCORE_DEFINITIONS[target]["direction"],
        )
        metric_rows.append({
            "architecture": variant, "target": target, "split": split,
            "selected_stage": "val_selected_spline", "parameters": x.shape[1] + 1,
            **metrics,
        })
        prediction_rows.append(pd.DataFrame({
            "architecture": variant, "target": target, "split": split,
            "hospital_id": group.hospital_id.astype(str),
            "video_id": group.video_id.astype(str),
            "current_lab_time_unix": group.current_lab_time_unix,
            "hours_since_cabg": group.hours_since_cabg,
            "history_count": group.history_count,
            "y_true": y_true, "y_pred": y_pred,
            "y_true_scaled": group.robust_scaled_raw_value,
            "y_pred_scaled": model.predict(x),
        }))
    pd.DataFrame({
        "target": [target], "variant": [variant], "knots": [knots],
        "degree": [3], "alpha": [alpha], "val_mae": [val_mae],
    }).to_csv(run_dir / "selected_hyperparameters.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    print(
        f"[job-complete] target={target} variant={variant} "
        f"knots={knots} alpha={alpha} "
        f"test_MAE={metrics_frame.loc[metrics_frame.split.eq('test'), 'mae'].iloc[0]:.4f}",
        flush=True,
    )
    return metrics_frame


_GPU_ID = None


def _worker_init(gpu_queue):
    global _GPU_ID
    _GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    print(f"[worker-ready] gpu=cuda:{_GPU_ID}", flush=True)


def _run_target(job):
    target = job["target"]
    root = Path(job["output_dir"])
    records = pd.read_csv(
        root / "cohort" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    _, history = load_task(REFERENCE_DIR, target)
    scaler = RobustTargetScaler(**job["scaler"])
    seed = int(job["seed"])
    device = torch.device(f"cuda:{_GPU_ID}")
    print(f"[target-start] target={target} device={device} rows={len(records)}", flush=True)
    _seed(seed)
    reference_dir = root / "runs" / target / "matched_history_only"
    matched, _ = train_task(
        target, records, history, scaler, reference_dir, device, seed,
        head_epochs=job["head_epochs"],
        finetune_epochs=job["finetune_epochs"],
    )
    matched["architecture"] = "matched_history_only"
    matched.insert(0, "parameters", parameter_counts(HistoryOnlyRegressor())["total"])
    frames = [matched]
    for variant in NEURAL_VARIANTS:
        frames.append(_neural_run(
            target, records, history, scaler, variant, root, device, seed,
            job["head_epochs"], job["finetune_epochs"],
        ))
    for variant in SPLINE_VARIANTS:
        frames.append(_spline_run(target, records, history, scaler, variant, root))
    print(f"[target-complete] target={target} device={device}", flush=True)
    return pd.concat(frames, ignore_index=True)


def _frozen_existing(target, cohort):
    path = OUTPUT_DIR / "runs" / target / "video_predictions.csv"
    if not path.is_file():
        return None
    existing = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    test = cohort.loc[cohort.split.eq("test")]
    matched = test[["hospital_id", "video_id", "raw_value", "score_threshold"]].merge(
        existing.loc[existing.split.eq("test"), ["hospital_id", "video_id", "y_pred"]],
        on=["hospital_id", "video_id"], how="left", validate="one_to_one",
    )
    if matched.y_pred.isna().any():
        raise AssertionError(f"Original test predictions missing for {target}")
    result = regression_metrics(
        matched.raw_value, matched.y_pred, matched.score_threshold,
        SCORE_DEFINITIONS[target]["direction"],
    )
    return {
        "architecture": "existing_history_only_frozen", "target": target,
        "split": "test", "selected_stage": "existing", "parameters": 993,
        **result,
    }


def _plot_results(root, metrics):
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    test = metrics.loc[metrics.split.eq("test")].copy()
    labels = {
        "matched_history_only": "Full history (matched)",
        "existing_history_only_frozen": "Full history (existing)",
        "surgery_time_mlp": "CABG time MLP",
        "sampling_times_mlp": "Sampling times MLP",
        "surgery_time_spline": "CABG time spline",
        "sampling_times_spline": "Sampling times spline",
    }
    colors = {
        "matched_history_only": "#285f85",
        "existing_history_only_frozen": "#7898ae",
        "surgery_time_mlp": "#c0634c",
        "sampling_times_mlp": "#ad8139",
        "surgery_time_spline": "#d69275",
        "sampling_times_spline": "#c5aa66",
    }
    targets = [target for target in TARGETS if target in set(test.target)]
    rows, columns = target_grid_shape(len(targets))
    fig, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False
    )
    for axis, target in zip(axes.flat, targets):
        subset = test.loc[test.target.eq(target)].set_index("architecture")
        baseline = float(subset.loc["matched_history_only", "mae"])
        order = [key for key in labels if key in subset.index]
        values = [float(subset.loc[key, "mae"]) / baseline for key in order]
        axis.barh([labels[key] for key in order], values,
                  color=[colors[key] for key in order])
        axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(target)
        axis.set_xlabel("Test MAE / matched full-history MAE")
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "test_relative_mae.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    residual_rows = []
    for target in targets:
        baseline_path = root / "runs" / target / "surgery_time_spline" / "video_predictions.csv"
        baseline = pd.read_csv(baseline_path)
        baseline = baseline.loc[baseline.split.eq("test"), ["video_id", "y_true", "y_pred"]]
        baseline = baseline.rename(columns={"y_pred": "trajectory_pred"})
        denominator = float(np.square(baseline.y_true - baseline.trajectory_pred).sum())
        for variant in VARIANTS:
            path = root / "runs" / target / variant / "video_predictions.csv"
            pred = pd.read_csv(path)
            joined = baseline.merge(
                pred.loc[pred.split.eq("test"), ["video_id", "y_true", "y_pred"]],
                on="video_id", validate="one_to_one", suffixes=("_base", ""),
            )
            if len(joined) != len(baseline) or not np.allclose(
                joined.y_true_base, joined.y_true, rtol=0, atol=1e-7
            ):
                raise AssertionError(f"Residual comparison cohort differs: {target}/{variant}")
            residual_rows.append({
                "target": target, "architecture": variant, "n": len(joined),
                "trajectory_residual_r2": (
                    1.0 - float(np.square(joined.y_true - joined.y_pred).sum()) / denominator
                    if denominator > 0 else np.nan
                ),
                "trajectory_residual_pearson_r": (
                    float(np.corrcoef(
                        joined.y_true - joined.trajectory_pred,
                        joined.y_pred - joined.trajectory_pred,
                    )[0, 1])
                    if (joined.y_true - joined.trajectory_pred).std() > 0
                    and (joined.y_pred - joined.trajectory_pred).std() > 0 else np.nan
                ),
            })
    residuals = pd.DataFrame(residual_rows)
    residuals.to_csv(root / "trajectory_residual_metrics.csv", index=False)
    fig, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False
    )
    for axis, target in zip(axes.flat, targets):
        subset = residuals.loc[residuals.target.eq(target)].set_index("architecture")
        order = [key for key in labels if key in subset.index and key != "surgery_time_spline"]
        axis.barh(
            [labels[key] for key in order],
            [subset.loc[key, "trajectory_residual_r2"] for key in order],
            color=[colors[key] for key in order],
        )
        axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(target)
        axis.set_xlabel("Test R2 relative to CABG-time spline")
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "trajectory_residual_skill.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots-complete] directory={figures}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--head-epochs", type=int, default=HEAD_MAX_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_MAX_EPOCHS)
    parser.add_argument("--targets", default=",".join(TARGETS))
    args = parser.parse_args()
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(TARGETS):
        raise ValueError(f"Invalid target set: {targets}")
    if args.output_dir.exists():
        raise FileExistsError(f"Trajectory ablation output already exists: {args.output_dir}")
    root = args.output_dir.resolve()
    (root / "cohort").mkdir(parents=True)
    episodes, ambiguous_count = _cabg_episodes()
    scalers = json.loads(
        (REFERENCE_DIR / "target_scalers.json").read_text(encoding="utf-8")
    )["targets"]
    audits, jobs, cohorts = [], [], {}
    for target in targets:
        cohort, _, audit = _make_cohort(target, episodes)
        cohort.to_csv(root / "cohort" / f"{target}.csv", index=False)
        audits.append(audit)
        cohorts[target] = cohort
        token = f"{SEED}:history_only_head32:{target}".encode()
        seed = (SEED + int.from_bytes(hashlib.sha256(token).digest()[:4], "little")) % (2**31 - 1)
        jobs.append({
            "target": target, "output_dir": str(root), "scaler": scalers[target],
            "seed": seed, "head_epochs": args.head_epochs,
            "finetune_epochs": args.finetune_epochs,
        })
    pd.DataFrame(audits).to_csv(root / "cohort_audit.csv", index=False)
    manifest = {
        "schema_version": 1,
        "experiment": "exp2_history_only_cabg_trajectory_ablations",
        "targets": list(targets), "seed": SEED,
        "cohort": "first valid CABG end to discharge in same admission",
        "sample_policy": "nearest video per patient/analyte/lab timestamp",
        "split_policy": "unchanged patient-disjoint reference split",
        "label_scalers": "unchanged reference train-only robust scalers",
        "ambiguous_cabg_episodes_excluded": ambiguous_count,
        "reference_history_policy": "same-analyte strictly prior measurements",
        "surgery_time_input": "log1p(hours since CABG end / 24); MLP train-only median/IQR normalization",
        "sampling_time_input": "reference -log1p(history age hours / 24), no lab values",
        "spline": "cubic B-spline Ridge; history basis mean pooled; knots/alpha selected on val MAE",
        "neural_schedule": {
            "head_lr": HEAD_LEARNING_RATE, "finetune_lr": FINETUNE_LEARNING_RATE,
            "head_epochs": args.head_epochs, "finetune_epochs": args.finetune_epochs,
        },
        "reference_task_hashes": {item["target"]: item["reference_task_sha256"] for item in audits},
        "reference_history_hashes": {item["target"]: item["reference_history_sha256"] for item in audits},
        "model_parameters": {
            "matched_history_only": parameter_counts(HistoryOnlyRegressor())["total"],
            "surgery_time_mlp": sum(p.numel() for p in SurgeryTimeMLP().parameters()),
            "sampling_times_mlp": sum(p.numel() for p in SamplingTimesMLP().parameters()),
        },
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the neural ablations")
    worker_count = min(4, torch.cuda.device_count(), len(jobs))
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for device_id in range(worker_count):
        gpu_queue.put(device_id)
    print(
        f"[prepared] targets={len(targets)} workers={worker_count} "
        f"CABG_episodes={len(episodes)} output={root}", flush=True,
    )
    metric_frames, run_rows, failures = [], [], []
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context,
        initializer=_worker_init, initargs=(gpu_queue,),
    ) as executor:
        futures = {executor.submit(_run_target, job): job for job in jobs}
        for future in as_completed(futures):
            target = futures[future]["target"]
            try:
                metric_frames.append(future.result())
                run_rows.append({"target": target, "status": "ok"})
                print(f"[scheduler-complete] target={target}", flush=True)
            except Exception as exc:
                failures.append({"target": target, "error": repr(exc), "traceback": traceback.format_exc()})
                run_rows.append({"target": target, "status": "failed", "error": repr(exc)})
                print(f"[scheduler-failed] target={target}: {exc}", flush=True)
            pd.DataFrame(run_rows).to_csv(root / "run_index.csv", index=False)
            if metric_frames:
                pd.concat(metric_frames, ignore_index=True).to_csv(root / "metrics_all.csv", index=False)
    if failures:
        (root / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} trajectory ablation targets failed")
    frozen_rows = [row for target in targets if (row := _frozen_existing(target, cohorts[target]))]
    if frozen_rows:
        metric_frames.append(pd.DataFrame(frozen_rows))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(root / "metrics_all.csv", index=False)
    _plot_results(root, metrics)
    print(f"[experiment-complete] targets={len(targets)} output={root}", flush=True)


if __name__ == "__main__":
    main()
