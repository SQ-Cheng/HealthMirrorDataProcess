"""One-time MST++ inference over the retained native224 20-frame FFV1 index."""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import (
    FrameOffsetIndex,
    _index_is_reusable,
)

from .config import CACHE, FEATURE_SHAPE, GRID_SIZE, INDEX_PATH
from .spectral import checkpoint_sha256, load_reconstructor, reconstruct_features, sha256


def load_index():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Missing reference 20-frame index: {INDEX_PATH}")
    index = FrameOffsetIndex.load(INDEX_PATH)
    if set(index.video_formats) != {"ffv1"}:
        raise RuntimeError("Exp8 requires the native-224 FFV1 frame index")
    if not _index_is_reusable(INDEX_PATH.parent, index.video_ids, "20frame"):
        raise RuntimeError("Reference 20-frame index is stale; do not silently resample videos")
    if np.any(np.diff(index.video_ptr) != 20):
        raise RuntimeError("Reference index does not contain exactly 20 frames per video")
    return index


def load_or_extract_features(device=None):
    index = load_index()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"mstpp_31band_grid{GRID_SIZE}x{GRID_SIZE}.npy"
    manifest_file = CACHE / "feature_manifest.json"
    expected = {
        "weight_sha256": checkpoint_sha256(),
        "index_sha256": sha256(INDEX_PATH),
        "index_video_count": len(index.video_ids),
        "frames": len(index.starts),
        "feature_shape": list(FEATURE_SHAPE),
        "dtype": "float16",
        "rgb_preprocess": "per-frame min-max across all RGB pixels, as in official MST++ training",
        "feature_policy": (
            "clip 0-1; central pixels [28:196,28:196]; " +
            f"{GRID_SIZE}x{GRID_SIZE} spatial means"
        ),
    }
    if cache_file.is_file() and manifest_file.is_file():
        saved = json.loads(manifest_file.read_text(encoding="utf-8"))
        if saved == expected:
            features = np.load(cache_file, mmap_mode="r")
            if (features.shape == (len(index.starts), *FEATURE_SHAPE)
                    and features.dtype == np.float16
                    and np.isfinite(features).all()):
                print(f"[spectral-cache-reused] frames={len(features)} path={cache_file}", flush=True)
                return index, features
        raise RuntimeError("Existing spectral cache does not match the source index or checkpoint")
    if cache_file.exists() or manifest_file.exists():
        raise RuntimeError("Incomplete spectral cache exists; inspect before rerunning")
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_reconstructor(device)
    records = pd.DataFrame({
        "video_id": index.video_ids.astype(str),
        "robust_scaled_raw_value": np.zeros(len(index.video_ids), dtype=np.float32),
    })
    dataset = AllFramesDataset(index, records, views=("original",))
    if len(dataset) != len(index.starts):
        raise AssertionError("One cache row per indexed frame is required")
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=device.type == "cuda", persistent_workers=True,
    )
    temporary = CACHE / f"mstpp_31band_grid{GRID_SIZE}x{GRID_SIZE}.tmp.npy"
    features = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16,
        shape=(len(index.starts), *FEATURE_SHAPE),
    )
    processed = 0
    try:
        for images, _, local_indices, _ in loader:
            global_indices = dataset.frame_indices[local_indices.numpy()]
            batch = reconstruct_features(model, images, device).numpy()
            features[global_indices] = batch.astype(np.float16)
            processed += len(batch)
            if processed % 1600 < len(batch):
                print(f"[spectral-extract] {processed}/{len(dataset)} frames", flush=True)
        features.flush()
        if processed != len(dataset) or not np.isfinite(features).all():
            raise RuntimeError("Spectral extraction is incomplete or non-finite")
    finally:
        del loader
        dataset.close()
        del features
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    temporary.replace(cache_file)
    manifest_file.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"[spectral-cache-complete] frames={processed} path={cache_file}", flush=True)
    return index, np.load(cache_file, mmap_mode="r")
