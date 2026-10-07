"""Label-independent, compact color features from unchanged native face views."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from study.exp2_face_pretrained_head32_regression.config import IMAGENET_MEAN, IMAGENET_STD, CROP_SCALE, BRIGHTNESS_DELTA, CONTRAST_DELTA
from study.common.run_video_loss_12h import sha256

from .config import HERE, INDEX_PATH, VIEWS, HISTOGRAM_BINS, HISTOGRAM_FEATURES, STATISTICS_FEATURES


DATASET = None
CACHE = HERE / "cache"


def color_features(rgb):
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 1), dtype=np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    normalized = [rgb, hsv / np.array([360., 1., 1.], np.float32),
                  lab / np.array([100., 128., 128.], np.float32)]
    histogram = []
    for space_index, space in enumerate(normalized):
        for channel in range(3):
            values = space[..., channel]
            if space_index == 2 and channel > 0:
                values = (values + 1) / 2
            bins = np.histogram(np.clip(values, 0, 1), HISTOGRAM_BINS, range=(0, 1))[0]
            histogram.extend(bins / bins.sum())

    def summarize(space):
        pixels = space.reshape(-1, space.shape[-1])
        quantiles = np.quantile(pixels, [.1, .5, .9], axis=0)
        return np.concatenate([pixels.mean(0), pixels.std(0), *quantiles])

    angle = normalized[1][..., 0] * (2 * np.pi)
    saturation = hsv[..., 1]
    denominator = max(float(saturation.sum()), 1e-6)
    sine = float((np.sin(angle) * saturation).sum() / denominator)
    cosine = float((np.cos(angle) * saturation).sum() / denominator)
    hue = [sine, cosine, np.hypot(sine, cosine)]
    statistics = np.concatenate([summarize(rgb), summarize(normalized[1][..., 1:]),
                                 hue, summarize(normalized[2])])
    histogram = np.asarray(histogram, np.float32)
    statistics = statistics.astype(np.float32)
    if histogram.shape != (HISTOGRAM_FEATURES,) or statistics.shape != (STATISTICS_FEATURES,):
        raise AssertionError("Color feature dimensions changed")
    if not np.isfinite(histogram).all() or not np.isfinite(statistics).all():
        raise ValueError("Non-finite color features")
    return histogram, statistics


def _init_worker():
    global DATASET
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    index = FrameOffsetIndex.load(INDEX_PATH)
    records = pd.DataFrame({"video_id": index.video_ids.astype(str),
                            "robust_scaled_raw_value": np.zeros(len(index.video_ids))})
    DATASET = AllFramesDataset(index, records, views=("original",))


def _extract_video(video):
    index = DATASET.index
    start, end = index.frame_range(index.video_ids[video])
    hist, stats = [], []
    mean = np.array(IMAGENET_MEAN, np.float32)
    std = np.array(IMAGENET_STD, np.float32)
    for local in range(start, end):
        image = DATASET[local][0]
        normalized = _prepare_images(image.unsqueeze(0), torch.arange(len(VIEWS)).unsqueeze(0),
                                     "bicubic", torch.device("cpu"))
        images = normalized.numpy().transpose(0, 2, 3, 1) * std + mean
        values = [color_features(rgb) for rgb in images]
        hist.append(np.stack([value[0] for value in values]))
        stats.append(np.stack([value[1] for value in values]))
    return start, end, np.stack(hist), np.stack(stats)


def ensure_cache():
    CACHE.mkdir(exist_ok=True)
    index = FrameOffsetIndex.load(INDEX_PATH)
    expected = {
        "index_sha256": sha256(INDEX_PATH), "frames": len(index.starts),
        "views": list(VIEWS), "bins": HISTOGRAM_BINS, "opencv_version": cv2.__version__,
        "crop_scale": CROP_SCALE, "brightness_delta": BRIGHTNESS_DELTA,
        "imagenet_mean": list(IMAGENET_MEAN), "imagenet_std": list(IMAGENET_STD),
        "contrast_delta": CONTRAST_DELTA, "image_size": 224, "roi": "full face frame",
        "histogram_features": HISTOGRAM_FEATURES, "statistics_features": STATISTICS_FEATURES,
        "dtype": "float32", "label_independent": True,
        "hue_statistics": "saturation-weighted sine/cosine and resultant length",
    }
    manifest = CACHE / "manifest.json"
    with (CACHE / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.is_file():
            if json.loads(manifest.read_text()) != expected:
                raise RuntimeError("Existing color cache uses a different protocol")
            for name, width in (("histogram", HISTOGRAM_FEATURES), ("statistics", STATISTICS_FEATURES)):
                values = np.load(CACHE / f"{name}.npy", mmap_mode="r")
                if values.shape != (len(index.starts), 5, width) or not np.isfinite(values).all():
                    raise RuntimeError(f"Incomplete color cache: {name}")
            print("[color-cache-reused] verified compact frame/view features", flush=True)
            return
        arrays = {name: np.lib.format.open_memmap(CACHE / f"{name}.tmp.npy", mode="w+", dtype=np.float32,
                                                  shape=(len(index.starts), 5, width))
                  for name, width in (("histogram", HISTOGRAM_FEATURES), ("statistics", STATISTICS_FEATURES))}
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx, initializer=_init_worker) as pool:
            futures = [pool.submit(_extract_video, video) for video in range(len(index.video_ids))]
            for completed, future in enumerate(as_completed(futures), 1):
                start, end, hist, stats = future.result()
                arrays["histogram"][start:end] = hist
                arrays["statistics"][start:end] = stats
                if completed % 50 == 0:
                    print(f"[color-cache] videos={completed}/{len(index.video_ids)}", flush=True)
        for name, array in arrays.items():
            array.flush()
            if not np.isfinite(array).all():
                raise RuntimeError("Feature extraction failed")
        del arrays, array
        for name in ("histogram", "statistics"):
            (CACHE / f"{name}.tmp.npy").replace(CACHE / f"{name}.npy")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
        print("[color-cache-complete] no decoded images or spectral cubes persisted", flush=True)
