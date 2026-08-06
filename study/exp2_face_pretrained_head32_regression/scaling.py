"""Leakage-safe train-only robust scaling for raw regression targets."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class RobustTargetScaler:
    target: str
    unit: str
    median: float
    q1: float
    q3: float
    iqr: float
    train_videos: int
    train_video_ids_sha256: str
    quantile_method: str = "linear"

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return (values - self.median) / self.iqr

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values * self.iqr + self.median

    def to_dict(self):
        return asdict(self)


def fit_robust_target_scaler(target, records, unit):
    train = records.loc[records["split"].eq("train")].copy()
    if train.empty:
        raise ValueError(f"No training rows available to fit scaler for {target}")
    values = train["raw_value"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite training raw values for {target}")
    q1, median, q3 = np.quantile(values, (0.25, 0.50, 0.75), method="linear")
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0.0:
        raise ValueError(f"Non-positive training IQR for {target}: {iqr}")
    video_ids = "\n".join(sorted(train["video_id"].astype(str))).encode()
    return RobustTargetScaler(
        target=target,
        unit=unit,
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=iqr,
        train_videos=len(train),
        train_video_ids_sha256=hashlib.sha256(video_ids).hexdigest(),
    )


def write_target_scalers(scalers, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "method": "train-only median/IQR robust scaling",
                "formula": "scaled=(raw_value-train_median)/train_IQR",
                "inverse_formula": "raw_value=scaled*train_IQR+train_median",
                "clipping": None,
                "targets": {
                    target: scaler.to_dict() for target, scaler in scalers.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
