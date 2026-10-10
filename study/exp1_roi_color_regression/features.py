"""Compact native-pixel color descriptors; no spatial resize or histogram bins."""

import cv2
import numpy as np

from .roi import NAMES


EPSILON = 1. / 255.
FEATURE_NAMES = tuple(
    [f"rgb_{channel}_{stat}" for channel in ("r", "g", "b") for stat in ("mean", "std", "p10", "p50", "p90")]
    + [f"lab_{channel}_{stat}" for channel in ("l", "a", "b") for stat in ("mean", "std", "median")]
    + [f"hsv_{channel}_{stat}" for channel in ("s", "v") for stat in ("mean", "std")]
    + ["hue_weighted_mean_sin", "hue_weighted_mean_cos", "hue_concentration"]
    + [f"chromaticity_{channel}_{stat}" for channel in ("r", "g") for stat in ("mean", "std")]
    + [f"log_{ratio}_{stat}" for ratio in ("r_over_g", "b_over_g") for stat in ("mean", "std")]
    + ["dark_pixel_fraction", "saturated_pixel_fraction"]
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _mean_std(values):
    return float(np.mean(values, dtype=np.float64)), float(np.std(values.astype(np.float64), ddof=0))


def roi_features(rgb, mask):
    if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
        raise ValueError("Features require unmodified native224 uint8 RGB")
    if mask.shape != (224, 224) or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ValueError("Invalid native ROI mask")
    # Select original pixels before color conversion; mask holes/padding never enter statistics.
    raw = rgb[mask.astype(bool)]
    if not len(raw):
        raise ValueError("Empty native ROI mask")
    pixels = raw.astype(np.float32) / 255.
    lab = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_RGB2LAB)[:, 0, :]
    hsv = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, :]
    result = []
    for channel in pixels.T:
        result.extend(_mean_std(channel))
        result.extend(np.percentile(channel, [10, 50, 90], method="linear"))
    for channel in lab.T:
        result.extend(_mean_std(channel))
        result.append(float(np.median(channel)))
    for channel in (hsv[:, 1], hsv[:, 2]):
        result.extend(_mean_std(channel))
    angle = np.deg2rad(hsv[:, 0].astype(np.float64))
    weights = hsv[:, 1].astype(np.float64)
    if weights.sum() <= 1e-8:
        sine = cosine = concentration = 0.
    else:
        sine = float(np.dot(weights, np.sin(angle)) / weights.sum())
        cosine = float(np.dot(weights, np.cos(angle)) / weights.sum())
        concentration = min(1., float(np.hypot(sine, cosine)))
    result.extend((sine, cosine, concentration))
    values = pixels.astype(np.float64)
    total = np.maximum(values.sum(axis=1), EPSILON)
    for channel in (values[:, 0] / total, values[:, 1] / total):
        result.extend(_mean_std(channel))
    for numerator in (values[:, 0], values[:, 2]):
        result.extend(_mean_std(np.log((numerator + EPSILON) / (values[:, 1] + EPSILON))))
    result.extend((float((raw <= 5).all(axis=1).mean()), float((raw >= 250).any(axis=1).mean())))
    vector = np.asarray(result, np.float32)
    if vector.shape != (41,) or not np.isfinite(vector).all():
        raise ValueError("Nonfinite or incorrectly sized color descriptor")
    return vector


def frame_features(rgb, rois):
    descriptors = np.full((len(NAMES), FEATURE_COUNT), np.nan, np.float32)
    valid = np.zeros(len(NAMES), bool)
    for row, name in enumerate(NAMES):
        roi = rois.get(name)
        if roi is not None and roi.accepted:
            descriptors[row] = roi_features(rgb, roi.mask)
            valid[row] = True
    return descriptors, valid


def aggregate_video(descriptors, valid, minimum_frames=10):
    descriptors, valid = np.asarray(descriptors), np.asarray(valid, dtype=bool)
    if descriptors.ndim != 3 or descriptors.shape[1:] != (len(NAMES), FEATURE_COUNT) or valid.shape != descriptors.shape[:2]:
        raise ValueError("Expected frame x ROI x feature descriptors and matching validity mask")
    if not np.isfinite(descriptors[valid]).all():
        raise ValueError("A supposedly valid ROI has nonfinite features")
    common = valid.all(axis=1)
    if common.sum() < minimum_frames:
        raise ValueError("Too few common-valid native frames")
    return descriptors[common].mean(axis=0, dtype=np.float64).astype(np.float32), common


def feature_schema():
    return {"schema_version": 2, "dimensions_per_roi": FEATURE_COUNT, "names": list(FEATURE_NAMES),
            "source": "unmodified uint8 RGB pixels selected by original 224x224 ROI mask",
            "spatial_resize": False, "population_std": True, "percentile_method": "linear",
            "rgb_units": "channel intensity divided by 255", "lab_units": "OpenCV float Lab; L in [0,100], signed a/b",
            "hsv_units": "float HSV; hue degrees converted to radians, S/V in [0,1]",
            "hue_weights": "pixel saturation; zero sine/cosine/concentration for achromatic regions",
            "chromaticity": "r=R/max(R+G+B,1/255); g=G/max(R+G+B,1/255)",
            "log_ratios": "natural log((R+1/255)/(G+1/255)), natural log((B+1/255)/(G+1/255))",
            "pixel_quality": "dark: all raw channels<=5; saturated: any raw channel>=250",
            "dark_and_saturated_pixels": "retained, not removed", "histograms": False,
            "aggregation": "equal frame mean using the same common-valid frames for every ROI"}
