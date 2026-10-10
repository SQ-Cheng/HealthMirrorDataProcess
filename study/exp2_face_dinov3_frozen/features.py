"""One-time frozen frame features; no backbone updates or decoded-pixel storage."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.train import _prepare_images

from . import config
from .backbone import load_encoder, weight_sha256


CACHE = config.HERE / "cache"
ENCODER = DATASET = DEVICE = None


def load_frozen_encoder(architecture="dinov3_vits16"):
    if architecture == "dinov3_vits16":
        return load_encoder()
    if architecture != "efficientnet_b0":
        raise ValueError(f"Unsupported frozen encoder: {architecture}")
    from torchvision.models import efficientnet_b0
    path = config.WEIGHTS.parent / "efficientnet_b0_rwightman-7f5810bc.pth"
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.classifier = torch.nn.Identity()
    return model.eval().requires_grad_(False)


def _init_worker(queue, index_path, architecture):
    global ENCODER, DATASET, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    ENCODER = load_frozen_encoder(architecture).to(DEVICE).eval()
    index = FrameOffsetIndex.load(index_path)
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
    result = np.concatenate(features).reshape(20, 5, -1)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite frozen DINOv3 feature")
    return start, end, result


def ensure_cache(index_path=None, cache_dir=None, reuse_from=None, architecture="dinov3_vits16"):
    if architecture not in ("dinov3_vits16", "efficientnet_b0"):
        raise ValueError(f"Unsupported frozen encoder: {architecture}")
    prefix = "dino-cache" if architecture == "dinov3_vits16" else "en-b0-cache"
    index_path = index_path or config.INDEX_PATH
    cache_dir = cache_dir or CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = FrameOffsetIndex.load(index_path)
    expected = {
        "encoder": "official dinov3_vits16", "repo_revision": config.REPO_REVISION,
        "weight_sha256": weight_sha256() if architecture == "dinov3_vits16" else config.reference.sha256(config.WEIGHTS.parent / "efficientnet_b0_rwightman-7f5810bc.pth"),
        "frame_index_sha256": config.reference.sha256(index_path),
        "shape": [len(index.starts), 5, 384], "dtype": "float32", "input_size": [224, 224],
        "feature": "final normalized CLS token", "encoder_frozen": True,
        "encoder_mode": "eval", "views": list(config.VIEWS), "label_independent": True,
        "crop_scale": config.reference.config.CROP_SCALE,
        "brightness_delta": config.reference.config.BRIGHTNESS_DELTA,
        "contrast_delta": config.reference.config.CONTRAST_DELTA,
        "mean": list(config.reference.config.IMAGENET_MEAN),
        "std": list(config.reference.config.IMAGENET_STD),
    }
    if architecture == "efficientnet_b0":
        import torchvision
        digest = config.reference.sha256(config.WEIGHTS.parent / "efficientnet_b0_rwightman-7f5810bc.pth")
        if not digest.startswith("7f5810bc"):
            raise RuntimeError("EfficientNet-B0 ImageNet checkpoint hash mismatch")
        expected.update(encoder="torchvision efficientnet_b0 IMAGENET1K_V1", repo_revision=torchvision.__version__,
                        weight_sha256=digest, shape=[len(index.starts), 5, 1280],
                        feature="global-average-pooled feature; classifier omitted; frozen BatchNorm statistics")
    manifest = cache_dir / "manifest.json"
    with (cache_dir / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            features = np.load(cache_dir / "cls_features.npy", mmap_mode="r")
            if json.loads(manifest.read_text()) != expected or list(features.shape) != expected["shape"] or not np.isfinite(features).all():
                raise RuntimeError("Frozen-feature cache no longer matches the model or frames")
            print(f"[{prefix}-reused] verified frozen frame features", flush=True)
            return
        workers = min(4, torch.cuda.device_count())
        if workers < 1:
            raise RuntimeError("Frozen feature extraction requires CUDA")
        temporary = cache_dir / "cls_features.tmp.npy"
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(expected["shape"]))
        pending = list(range(len(index.video_ids)))
        reused = 0
        if reuse_from is not None:
            old_index_path, old_cache = reuse_from
            old_manifest_path = old_cache / "manifest.json"
            if old_manifest_path.is_file():
                old_manifest = json.loads(old_manifest_path.read_text())
                ignored = {"shape", "frame_index_sha256"}
                if all(old_manifest.get(key) == value for key, value in expected.items() if key not in ignored):
                    if old_manifest["frame_index_sha256"] != config.reference.sha256(old_index_path):
                        raise RuntimeError("Reusable DINO cache index changed")
                    old_index = FrameOffsetIndex.load(old_index_path)
                    if not _index_is_reusable(old_index_path.parent, old_index.video_ids, "20frame"):
                        raise RuntimeError("Reusable DINO cache video/sidecar fingerprints changed")
                    old_features = np.load(old_cache / "cls_features.npy", mmap_mode="r")
                    if list(old_features.shape) != old_manifest["shape"] or not np.isfinite(old_features).all():
                        raise RuntimeError("Reusable DINO cache is invalid")
                    pending = []
                    for video, video_id in enumerate(index.video_ids):
                        old_video = old_index.video_lookup.get(str(video_id))
                        same = old_video is not None and all(
                            getattr(index, key)[video] == getattr(old_index, key)[old_video]
                            for key in ("video_paths", "video_formats", "codec_extradata"))
                        start, end = index.frame_range(video_id)
                        if same:
                            old_start, old_end = old_index.frame_range(video_id)
                            same = all(np.array_equal(getattr(index, key)[start:end],
                                                     getattr(old_index, key)[old_start:old_end])
                                       for key in ("starts", "ends", "source_indices"))
                        if same:
                            features[start:end] = old_features[old_start:old_end]
                            reused += 1
                        else:
                            pending.append(video)
                    del old_features
            print(f"[{prefix}-reuse] reused_videos={reused} extract_videos={len(pending)}", flush=True)
        ctx = mp.get_context("spawn")
        if pending:
            with ctx.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(queue, index_path, architecture)) as pool:
                    futures = [pool.submit(_extract, video) for video in pending]
                    for completed, future in enumerate(as_completed(futures), 1):
                        start, end, value = future.result()
                        features[start:end] = value
                        if completed % 50 == 0:
                            print(f"[{prefix}] videos={completed + reused}/{len(index.video_ids)}", flush=True)
        features.flush()
        if not np.isfinite(features).all():
            raise RuntimeError("Frozen feature extraction failed")
        del features
        temporary.replace(cache_dir / "cls_features.npy")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
        (cache_dir / "build_audit.json").write_text(json.dumps({"reused_videos": reused,
                                                               "extracted_videos": len(pending)}, indent=2) + "\n")
        print(f"[{prefix}-complete] frozen encoder; compact feature vectors only", flush=True)
