"""Five-view native ROI statistics, reusing the reviewed original landmarks."""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from study.common.face_video import decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from .features import color_pixel_features
from .preview_rois import HERE, MODEL, sha256
from .roi import NAMES, extract_rois


VIEWS = ("original", "hflip", "center_crop", "brightness", "contrast")
CACHE = HERE / "cache/five_views41"
INDEX = OPTIONS = None


def augmented_descriptors(rgb, rois, original, valid, parameters, roi_options):
    """Select augmented pixels from transformed masks, never resize an ROI."""
    output = np.full((5, 4, 41), np.nan, np.float32)
    accepted = np.tile(valid, (5, 1))
    output[0] = original
    # Flipping changes image-left/right ROI identity but not pixel distributions.
    permutation = [0, 2, 1, 3]
    output[1], accepted[1] = original[permutation], valid[permutation]
    pixels = rgb.astype(np.float32) / 255.
    bright = np.clip(pixels * (1 + parameters["brightness_delta"]), 0, 1)
    mean = pixels.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
    contrast = np.clip((pixels - mean) * (1 + parameters["contrast_delta"]) + mean, 0, 1)
    size = int(round(224 * parameters["crop_scale"]))
    top = (224 - size) // 2
    section = slice(top, top + size)
    for j, name in enumerate(NAMES):
        roi = rois[name]
        mask = roi.mask.astype(bool)
        crop_mask = mask[section, section]
        minimum = roi_options["minimum_native_valid_pixels_lips" if name == "lips" else "minimum_native_valid_pixels_skin"]
        accepted[2, j] = bool(valid[j] and crop_mask.sum() >= minimum)
        if accepted[2, j]:
            output[2, j] = color_pixel_features(rgb[section, section][crop_mask])
        if valid[j]:
            output[3, j] = color_pixel_features(bright[mask])
            output[4, j] = color_pixel_features(contrast[mask])
    if not np.isfinite(output[accepted]).all():
        raise RuntimeError("Nonfinite accepted augmented ROI feature")
    return output, accepted


def init_worker(index_path, protocol):
    global INDEX, OPTIONS
    INDEX = FrameOffsetIndex.load(index_path)
    OPTIONS = protocol


def extract_video(video):
    start, end = INDEX.frame_range(video)
    with np.load(HERE / f"cache/native41/{video}.npz", allow_pickle=False) as shard:
        np.testing.assert_array_equal(shard["source_frame_indices"], INDEX.source_indices[start:end])
        original, validity, points = shard["features"], shard["valid"], shard["landmarks"]
    if original.shape != (20, 4, 41) or validity.shape != (20, 4):
        raise RuntimeError("Invalid original ROI cache")
    result = np.full((20, 5, 4, 41), np.nan, np.float32)
    valid = np.zeros((20, 5, 4), bool)
    decoders = OrderedDict()
    position = INDEX.video_lookup[video]
    with open(str(INDEX.video_paths[position]), "rb") as handle:
        for i, frame in enumerate(range(start, end)):
            if not np.isfinite(points[i]).all():
                continue
            rgb = np.ascontiguousarray(decode_indexed_frame(INDEX, position, frame, handle, decoders).permute(1, 2, 0).numpy())
            rois = extract_rois(rgb, points[i], OPTIONS["roi"])
            np.testing.assert_array_equal([rois[name].accepted for name in NAMES], validity[i])
            result[i], valid[i] = augmented_descriptors(rgb, rois, original[i], validity[i], OPTIONS["features"]["augmentation_parameters"], OPTIONS["roi"])
    return start, end, result, valid


def prepare_views(protocol, workers=6):
    from study.exp2_face_dinov3_frozen.run_main_regression import preflight, BASELINE, INDEX_PATH
    index, _, hashes, _, lab_hash = preflight()
    old = json.loads((HERE / "cache/native41/manifest.json").read_text())
    contract = {"protocol_sha256": sha256(HERE / "protocol.json"), "roi_schema_sha256": sha256(HERE / "roi_schema.json"),
                "feature_schema_sha256": sha256(HERE / "feature_schema.json"), "landmarker_sha256": sha256(MODEL),
                "frame_index_sha256": sha256(INDEX_PATH), "lab_table_sha256": lab_hash, "task_records_sha256": hashes,
                "views": list(VIEWS), "pixel_statistics": "native transformed pixels; no ROI spatial resize",
                "view_parameters": protocol["features"]["augmentation_parameters"], "common_validity": "all 5 views x 4 ROIs",
                "extractor_sha256": sha256(__file__), "features_code_sha256": sha256(HERE / "features.py")}
    for key in ("roi_schema_sha256", "feature_schema_sha256", "landmarker_sha256", "frame_index_sha256", "lab_table_sha256", "task_records_sha256"):
        if old[key] != contract[key]:
            raise RuntimeError(f"Reviewed original ROI cache is stale: {key}")
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text()) != contract:
            raise RuntimeError("Five-view cache contract changed; remove only the derived cache to rebuild")
        bank = np.load(CACHE / "features.npy", mmap_mode="r")
        valid = np.load(CACHE / "valid.npy", mmap_mode="r")
        if bank.shape != (len(index.starts), 5, 4, 41) or valid.shape != (len(index.starts), 5, 4):
            raise RuntimeError("Incorrect five-view cache shape")
        print("[roi-five-view-cache-reused]", flush=True)
    else:
        bank = np.lib.format.open_memmap(CACHE / "features.tmp.npy", mode="w+", dtype=np.float32, shape=(len(index.starts), 5, 4, 41))
        valid = np.lib.format.open_memmap(CACHE / "valid.tmp.npy", mode="w+", dtype=bool, shape=(len(index.starts), 5, 4))
        bank[:] = np.nan
        valid[:] = False
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                                 initargs=(str(INDEX_PATH), protocol)) as pool:
            futures = [pool.submit(extract_video, str(video)) for video in index.video_ids]
            for count, future in enumerate(as_completed(futures), 1):
                start, end, values, flags = future.result()
                bank[start:end], valid[start:end] = values, flags
                if count % 25 == 0:
                    print(f"[roi-five-view-extract] videos={count}/{len(index.video_ids)}", flush=True)
        bank.flush()
        valid.flush()
        del bank, valid
        (CACHE / "features.tmp.npy").replace(CACHE / "features.npy")
        (CACHE / "valid.tmp.npy").replace(CACHE / "valid.npy")
        manifest.write_text(json.dumps(contract, indent=2) + "\n")
        bank = np.load(CACHE / "features.npy", mmap_mode="r")
        valid = np.load(CACHE / "valid.npy", mmap_mode="r")
    common = valid.all(axis=(1, 2))
    if not np.isfinite(bank[common]).all():
        raise RuntimeError("Invalid common-valid augmented features")
    vectors, audit = {}, []
    for video in index.video_ids:
        start, end = index.frame_range(video)
        accepted = common[start:end]
        eligible = int(accepted.sum()) >= protocol["roi"]["minimum_common_valid_frames_per_video"]
        if eligible:
            vectors[str(video)] = bank[start:end][accepted].mean(axis=0, dtype=np.float64).astype(np.float32)
        audit.append({"video_id": str(video), "common_valid_frames": int(accepted.sum()), "included": eligible})
    tables = HERE / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(tables / "roi_video_audit.csv", index=False)
    counts = []
    for target in protocol["targets"]:
        records = pd.read_csv(BASELINE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
        effective = records.loc[records.video_id.isin(vectors)].reset_index(drop=True)
        if set(effective.split) != {"train", "val", "test"} or effective.groupby("hospital_id").split.nunique().gt(1).any():
            raise RuntimeError("Invalid effective ROI splits")
        effective.to_csv(tables / f"records_{target}.csv", index=False)
        records.loc[~records.video_id.isin(vectors)].to_csv(tables / f"excluded_{target}.csv", index=False)
        np.save(tables / f"features_{target}.npy", np.stack([vectors[video] for video in effective.video_id]))
        for split, group in effective.groupby("split"):
            counts.append({"target": target, "split": split, "videos": len(group), "patients": group.hospital_id.nunique(),
                           "source_videos": int(records.split.eq(split).sum()), "lab_events": group.clinical_event_id.nunique()})
    pd.DataFrame(counts).to_csv(tables / "cohort_counts.csv", index=False)
    (tables / "preparation_manifest.json").write_text(json.dumps({**contract, "included_videos": len(vectors),
                                                                  "source_videos": len(index.video_ids)}, indent=2) + "\n")
    print(f"[roi-data-ready] 5 views; included={len(vectors)}/{len(index.video_ids)} videos", flush=True)
    return contract
