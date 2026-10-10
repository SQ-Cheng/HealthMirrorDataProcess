"""Cache FaRL once; reuse the exact existing DINO frame/view feature cache."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from study.exp2_face_pretrained_head32_regression import config as image_config
from . import config
from .backbone import load_farl


MODEL = DATA = DEVICE = None


def init_worker(queue):
    global MODEL, DATA, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    MODEL = load_farl()[0].to(DEVICE).eval()
    index = FrameOffsetIndex.load(config.INDEX_PATH)
    DATA = AllFramesDataset(index, pd.DataFrame({"video_id": index.video_ids, "robust_scaled_raw_value": 0.}), views=("original",))


@torch.inference_mode()
def extract_video(video):
    start, end = DATA.index.frame_range(DATA.index.video_ids[video])
    pixels = torch.stack([DATA[frame][0] for frame in range(start, end)])
    inputs = _prepare_images(pixels, torch.arange(5).repeat(20, 1), "bicubic", DEVICE,
                             normalization_mean=config.CLIP_MEAN, normalization_std=config.CLIP_STD)
    values = []
    for offset in range(0, 100, 20):
        with torch.autocast("cuda", dtype=torch.float16):
            values.append(MODEL(inputs[offset:offset + 20]).float().cpu().numpy())
    features = np.concatenate(values).reshape(20, 5, 512)
    if not np.isfinite(features).all():
        raise RuntimeError("Nonfinite frozen FaRL feature")
    return start, end, features


def ensure_cache():
    index = FrameOffsetIndex.load(config.INDEX_PATH)
    expected = {"frame_index_sha256": config.reference.reference.sha256(config.INDEX_PATH),
                "weight_sha256": config.reference.reference.sha256(config.WEIGHTS), "clip_revision": config.CLIP_REVISION,
                "shape": [len(index.starts), 5, 512], "feature": "official CLIP-style ln_post CLS @ visual.proj; no L2 post-normalization",
                "views": list(config.reference.config.VIEWS), "mean": list(config.CLIP_MEAN), "std": list(config.CLIP_STD),
                "crop_scale": image_config.CROP_SCALE, "brightness_delta": image_config.BRIGHTNESS_DELTA,
                "contrast_delta": image_config.CONTRAST_DELTA, "interpolation": "bicubic",
                "input": "same native224 frame and deterministic view as DINO; branch-specific normalization", "backbone_frozen": True}
    config.CACHE.mkdir(parents=True, exist_ok=True)
    manifest = config.CACHE / "manifest.json"
    with (config.CACHE / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            values = np.load(config.CACHE / "features.npy", mmap_mode="r")
            if json.loads(manifest.read_text()) != expected or list(values.shape) != expected["shape"] or not np.isfinite(values).all():
                raise RuntimeError("FaRL cache contract changed")
            print("[farl-cache-reused] frozen image encoder features", flush=True)
            return
        workers = min(4, torch.cuda.device_count())
        if not workers:
            raise RuntimeError("CUDA required for feature extraction")
        temporary = config.CACHE / "features.tmp.npy"
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(expected["shape"]))
        context = mp.get_context("spawn")
        with context.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers):
                queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(queue,)) as pool:
                futures = [pool.submit(extract_video, video) for video in range(len(index.video_ids))]
                for i, future in enumerate(as_completed(futures), 1):
                    start, end, values = future.result()
                    features[start:end] = values
                    if i % 50 == 0:
                        print(f"[farl-cache] videos={i}/{len(index.video_ids)}", flush=True)
        features.flush()
        del features
        temporary.replace(config.CACHE / "features.npy")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
        print("[farl-cache-complete] shared by both fusion heads; DINO cache unchanged", flush=True)
