"""One-time frozen CLS features; no backbone updates or patch-cube storage."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.train import _prepare_images

from . import config
from .backbone import load_encoder, weight_sha256


CACHE = config.HERE / "cache"
ENCODER = DATASET = DEVICE = None


def _init_worker(queue):
    global ENCODER, DATASET, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    ENCODER = load_encoder().to(DEVICE).eval()
    index = FrameOffsetIndex.load(config.INDEX_PATH)
    records = pd.DataFrame({"video_id": index.video_ids.astype(str),
                            "robust_scaled_raw_value": np.zeros(len(index.video_ids))})
    DATASET = AllFramesDataset(index, records, views=("original",))


@torch.inference_mode()
def _extract(video):
    start, end = DATASET.index.frame_range(DATASET.index.video_ids[video])
    images = torch.stack([DATASET[frame][0] for frame in range(start, end)])
    views = torch.arange(5).repeat(20, 1)
    normalized = _prepare_images(images, views, "bicubic", DEVICE)
    features = []
    for offset in range(0, len(normalized), 40):
        with torch.autocast("cuda", dtype=torch.float16):
            features.append(ENCODER(normalized[offset:offset + 40]).float().cpu().numpy())
    result = np.concatenate(features).reshape(20, 5, 384)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite frozen DINOv3 feature")
    return start, end, result


def ensure_cache():
    CACHE.mkdir(exist_ok=True)
    index = FrameOffsetIndex.load(config.INDEX_PATH)
    expected = {
        "encoder": "official dinov3_vits16", "repo_revision": config.REPO_REVISION,
        "weight_sha256": weight_sha256(), "frame_index_sha256": config.reference.sha256(config.INDEX_PATH),
        "shape": [len(index.starts), 5, 384], "dtype": "float32", "input_size": [224, 224],
        "feature": "final normalized CLS token", "encoder_frozen": True,
        "encoder_mode": "eval", "views": list(config.VIEWS), "label_independent": True,
        "crop_scale": config.reference.config.CROP_SCALE,
        "brightness_delta": config.reference.config.BRIGHTNESS_DELTA,
        "contrast_delta": config.reference.config.CONTRAST_DELTA,
        "mean": list(config.reference.config.IMAGENET_MEAN),
        "std": list(config.reference.config.IMAGENET_STD),
    }
    manifest = CACHE / "manifest.json"
    with (CACHE / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            features = np.load(CACHE / "cls_features.npy", mmap_mode="r")
            if json.loads(manifest.read_text()) != expected or list(features.shape) != expected["shape"] or not np.isfinite(features).all():
                raise RuntimeError("Frozen-feature cache no longer matches the model or frames")
            print("[dino-cache-reused] verified final CLS features", flush=True)
            return
        workers = min(4, torch.cuda.device_count())
        if workers < 1:
            raise RuntimeError("Frozen feature extraction requires CUDA")
        temporary = CACHE / "cls_features.tmp.npy"
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(expected["shape"]))
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers):
                queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(queue,)) as pool:
                futures = [pool.submit(_extract, video) for video in range(len(index.video_ids))]
                for completed, future in enumerate(as_completed(futures), 1):
                    start, end, value = future.result()
                    features[start:end] = value
                    if completed % 50 == 0:
                        print(f"[dino-cache] videos={completed}/{len(index.video_ids)}", flush=True)
        features.flush()
        if not np.isfinite(features).all():
            raise RuntimeError("Frozen feature extraction failed")
        del features
        temporary.replace(CACHE / "cls_features.npy")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
        print("[dino-cache-complete] frozen encoder; compact CLS vectors only", flush=True)
