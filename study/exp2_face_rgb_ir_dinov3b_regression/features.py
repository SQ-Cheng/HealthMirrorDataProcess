"""Eight-GPU extraction from paired FFV1 packets; no decoded image copies."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from study.common.run_video_loss_12h import sha256
from study.exp2_face_dinov3_frozen.backbone import load_encoder, weight_sha256
from study.exp2_face_dinov3_farl_regression.backbone import load_farl
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from study.exp2_face_pretrained_head32_regression import config as image_config
from . import config


MODELS = DATA = DEVICE = REUSE = OLD = None


def cache_contract():
    return {"schema_version": 1, "indexes": {"rgb": sha256(config.RGB_INDEX), "ir": sha256(config.IR_INDEX)},
            "weights": {"dino": weight_sha256("dinov3_vitb16"), "farl": sha256(config.farl_config.WEIGHTS)},
            "dino_revision": config.reference.config.REPO_REVISION,
            "clip_revision": config.farl_config.CLIP_REVISION,
            "views": list(config.VIEWS), "crop_scale": image_config.CROP_SCALE,
            "brightness_delta": image_config.BRIGHTNESS_DELTA, "contrast_delta": image_config.CONTRAST_DELTA,
            "mean": {"dino": list(image_config.IMAGENET_MEAN), "farl": list(config.farl_config.CLIP_MEAN)},
            "std": {"dino": list(image_config.IMAGENET_STD), "farl": list(config.farl_config.CLIP_STD)},
            "interpolation": "bicubic", "dtype": "float32", "encoders_frozen": True, "encoder_mode": "eval",
            "ir_input": "validate three equal gray channels; repeat native gray channel three times, no fitted normalization",
            "spatial_registration": "original independent RGB/IR crops; synchronized views, no added registration",
            "features": {"dino": "final normalized CLS", "farl": "ln_post CLS @ visual.proj; no L2 post-normalization"}}


def reuse_rows(paired, original):
    """Match actual encoded packets, never merely a video ID or frame slot."""
    mapping = np.full(len(paired.starts), -1, dtype=np.int64)
    for video in paired.video_ids:
        if video not in original.video_lookup:
            continue
        a, b = paired.video_lookup[video], original.video_lookup[video]
        if (paired.video_paths[a] != original.video_paths[b]
                or paired.codec_extradata[a] != original.codec_extradata[b]):
            continue
        begin, end = original.frame_range(video)
        packets = {(int(original.source_indices[i]), int(original.starts[i]), int(original.ends[i])): i
                   for i in range(begin, end)}
        begin, end = paired.frame_range(video)
        for i in range(begin, end):
            mapping[i] = packets.get((int(paired.source_indices[i]), int(paired.starts[i]), int(paired.ends[i])), -1)
    return mapping


def reusable_rgb_caches(expected, rgb):
    original = FrameOffsetIndex.load(config.reference.INDEX_PATH)
    candidates = {"rgb_dino": (config.reference.config.HERE / "cache/main24h_frame_loss_vitb16", "cls_features.npy"),
                  "rgb_farl": (config.farl_config.CACHE, "features.npy")}
    result = {}
    for branch, (root, filename) in candidates.items():
        if not (root / "manifest.json").exists():
            continue
        saved = json.loads((root / "manifest.json").read_text())
        encoder = branch.split("_")[1]
        checks = {"frame_index_sha256": sha256(config.reference.INDEX_PATH), "weight_sha256": expected["weights"][encoder],
                  "views": expected["views"], "crop_scale": expected["crop_scale"],
                  "brightness_delta": expected["brightness_delta"], "contrast_delta": expected["contrast_delta"],
                  "mean": expected["mean"][encoder], "std": expected["std"][encoder]}
        if any(saved.get(key) != value for key, value in checks.items()):
            raise RuntimeError(f"Cannot reuse changed RGB feature contract: {branch}")
        values = np.load(root / filename, mmap_mode="r")
        if values.dtype != np.float32 or values.shape != (len(original.starts), 5, config.DIMENSIONS[branch]):
            raise RuntimeError("RGB cache shape/dtype changed")
        result[branch] = str(root / filename)
    return reuse_rows(rgb, original), result


def create_encoders(device):
    # Separate modules even when initial weights are identical; all remain frozen.
    result = {}
    for modality in ("rgb", "ir"):
        result[f"{modality}_dino"] = load_encoder(architecture="dinov3_vitb16").to(device).eval()
        result[f"{modality}_farl"] = load_farl()[0].to(device).eval()
    if len({id(model) for model in result.values()}) != 4 or any(p.requires_grad for model in result.values() for p in model.parameters()):
        raise RuntimeError("Four independent frozen encoders are required")
    return result


def init_worker(queue, mapping, old_paths):
    global MODELS, DATA, DEVICE, REUSE, OLD
    gpu = int(queue.get())
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    MODELS = create_encoders(DEVICE)
    DATA = {}
    for modality, path in (("rgb", config.RGB_INDEX), ("ir", config.IR_INDEX)):
        index = FrameOffsetIndex.load(path)
        DATA[modality] = AllFramesDataset(index, pd.DataFrame({"video_id": index.video_ids, "robust_scaled_raw_value": 0.}), views=("original",))
        DATA[modality].decode_cache_frames = 20
    REUSE, OLD = mapping, {branch: np.load(path, mmap_mode="r") for branch, path in old_paths.items()}
    print(f"[encoder-worker] cuda:{gpu} four independent frozen encoders; inference batch=20", flush=True)


def validate_ir(pixels):
    if pixels.ndim != 4 or pixels.shape[1:] != (3, 224, 224) or pixels.dtype != torch.uint8:
        raise ValueError("Expected native224 three-channel encoded IR")
    if not torch.equal(pixels[:, 0], pixels[:, 1]) or not torch.equal(pixels[:, 0], pixels[:, 2]):
        raise ValueError("IR is not genuine replicated grayscale")
    return pixels[:, :1].expand(-1, 3, -1, -1).contiguous()


@torch.inference_mode()
def extract_video(video):
    start, end = DATA["rgb"].index.frame_range(video)
    result = {}
    for modality in ("rgb", "ir"):
        # Decode each modality only once, then apply identical five views.
        pixels = torch.stack([DATA[modality][frame][0] for frame in range(start, end)])
        if modality == "ir":
            pixels = validate_ir(pixels)
        for encoder in ("dino", "farl"):
            branch = f"{modality}_{encoder}"
            values = np.empty((20, 5, config.DIMENSIONS[branch]), dtype=np.float32)
            reused = REUSE[start:end] >= 0 if branch in OLD else np.zeros(20, dtype=bool)
            if reused.any():
                values[reused] = OLD[branch][REUSE[start:end][reused]]
            needed = np.flatnonzero(~reused)
            if len(needed):
                mean = image_config.IMAGENET_MEAN if encoder == "dino" else config.farl_config.CLIP_MEAN
                std = image_config.IMAGENET_STD if encoder == "dino" else config.farl_config.CLIP_STD
                inputs = _prepare_images(pixels[needed], torch.arange(5).repeat(len(needed), 1), "bicubic", DEVICE,
                                         normalization_mean=mean, normalization_std=std)
                outputs = []
                for offset in range(0, len(inputs), 20):
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs.append(MODELS[branch](inputs[offset:offset+20]).float().cpu().numpy())
                values[needed] = np.concatenate(outputs).reshape(len(needed), 5, config.DIMENSIONS[branch])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite features: {video}/{branch}")
            result[branch] = values
    return start, end, result


def ensure_cache(workers=8):
    from study.exp2_face_rgb_ir_head32_regression.index import load_indexes
    rgb, _ = load_indexes(config.RGB_INDEX, config.IR_INDEX)
    expected = {**cache_contract(), "frames": len(rgb.starts)}
    config.CACHE.mkdir(parents=True, exist_ok=True)
    manifest = config.CACHE / "manifest.json"
    with (config.CACHE / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["contract"] != expected:
                raise RuntimeError("Existing paired features have a changed contract")
            for branch, dimension in config.DIMENSIONS.items():
                path = config.CACHE / f"{branch}.npy"
                values = np.load(path, mmap_mode="r")
                if values.shape != (len(rgb.starts), 5, dimension) or values.dtype != np.float32 or sha256(path) != saved["sha256"][branch]:
                    raise RuntimeError(f"Changed or incomplete feature cache: {branch}")
            print("[cache-reuse] four verified frozen feature banks", flush=True)
            return sha256(manifest)
        mapping, old_paths = reusable_rgb_caches(expected, rgb)
        banks = {branch: np.lib.format.open_memmap(config.CACHE / f"{branch}.tmp.npy", mode="w+", dtype=np.float32,
                                                   shape=(len(rgb.starts), 5, dimension)) for branch, dimension in config.DIMENSIONS.items()}
        workers = min(workers, torch.cuda.device_count(), len(rgb.video_ids))
        if not workers:
            raise RuntimeError("CUDA required")
        print(f"[cache-start] GPUs={workers} videos={len(rgb.video_ids)} RGB packet-identical reuse={int((mapping >= 0).sum())} frames per RGB encoder", flush=True)
        context = mp.get_context("spawn")
        done = np.zeros(len(rgb.starts), dtype=bool)
        with context.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers):
                queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(queue, mapping, old_paths)) as pool:
                videos = iter(rgb.video_ids)
                pending = {pool.submit(extract_video, video) for video in list(rgb.video_ids[:workers*2])}
                for _ in range(min(workers*2, len(rgb.video_ids))):
                    next(videos)
                count = 0
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        start, end, values = future.result()
                        if done[start:end].any() or end-start != 20:
                            raise RuntimeError("Duplicate or incomplete video feature result")
                        for branch in banks:
                            banks[branch][start:end] = values[branch]
                        done[start:end] = True
                        count += 1
                        video = next(videos, None)
                        if video is not None:
                            pending.add(pool.submit(extract_video, video))
                        if count == 1 or count % 25 == 0 or count == len(rgb.video_ids):
                            print(f"[cache-progress] paired videos={count}/{len(rgb.video_ids)}; all four branches", flush=True)
                    del completed
        if not done.all():
            raise RuntimeError("Feature extraction omitted frames")
        hashes = {}
        for branch in list(banks):
            bank = banks.pop(branch)
            if not np.isfinite(bank).all():
                raise RuntimeError("Nonfinite complete feature bank")
            bank.flush()
            del bank
            path = config.CACHE / f"{branch}.npy"
            (config.CACHE / f"{branch}.tmp.npy").replace(path)
            hashes[branch] = sha256(path)
        manifest.write_text(json.dumps({"contract": expected, "sha256": hashes,
            "reused_rgb_frames_per_encoder": int((mapping >= 0).sum())}, indent=2)+"\n")
        print("[cache-complete] four float32 banks; shared across all sixty heads", flush=True)
        return sha256(manifest)
