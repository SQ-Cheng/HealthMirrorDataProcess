"""One native41 ROI extraction pass per video with bounded CPU concurrency."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
import json
import multiprocessing
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from study.common.face_video import decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from .features import FEATURE_COUNT, frame_features, aggregate_video
from .preview_rois import HERE, MODEL, ROOT, sha256
from .roi import NAMES, extract_rois


INDEX = DETECTOR = OPTIONS = CACHE = REVIEWED = None


def init_worker(index_path, options, cache):
    global INDEX, DETECTOR, OPTIONS, CACHE, REVIEWED
    cv2.setNumThreads(1)
    INDEX, OPTIONS, CACHE = FrameOffsetIndex.load(index_path), options, Path(cache)
    configured = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=options["landmarker"]["min_face_detection_confidence"],
        min_face_presence_confidence=options["landmarker"]["min_face_presence_confidence"])
    DETECTOR = mp.tasks.vision.FaceLandmarker.create_from_options(configured)
    selected = pd.read_csv(HERE / "outputs/tables/roi_preview_100/selected_frames.csv")
    reviewed_manifest = json.loads((HERE / "outputs/tables/roi_preview_100/manifest.json").read_text())
    if reviewed_manifest["frame_index_sha256"] != sha256(index_path) or reviewed_manifest["model_sha256"] != sha256(MODEL):
        raise RuntimeError("Reviewed landmark source or model changed")
    with np.load(HERE / "outputs/tables/roi_preview_100/landmarks.npz", allow_pickle=False) as values:
        REVIEWED = dict(zip(selected.global_frame_index.astype(int), values["xy"]))


def extract_video(video_id):
    path = CACHE / f"{video_id}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            if values["features"].shape != (20, 4, FEATURE_COUNT) or values["valid"].shape != (20, 4):
                raise RuntimeError("Invalid native feature shard")
        return video_id
    start, end = INDEX.frame_range(video_id)
    if end - start != 20:
        raise RuntimeError("Expected twenty selected native frames")
    descriptors = np.full((20, 4, FEATURE_COUNT), np.nan, np.float32)
    valid = np.zeros((20, 4), bool)
    reasons = np.full((20, 4), "no_face_landmarks", dtype="U100")
    counts = np.zeros((20, 4), np.int32)
    points = np.full((20, 478, 2), np.nan, np.float32)
    video = INDEX.video_lookup[video_id]
    decoders = OrderedDict()
    with open(str(INDEX.video_paths[video]), "rb") as handle:
        for i, frame in enumerate(range(start, end)):
            rgb = np.ascontiguousarray(decode_indexed_frame(INDEX, video, frame, handle, decoders).permute(1, 2, 0).numpy())
            if frame in REVIEWED:
                xy = REVIEWED[frame]
            else:
                detection = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                xy = (np.asarray([(p.x * 224, p.y * 224) for p in detection.face_landmarks[0]], np.float32)
                      if detection.face_landmarks else points[i])
            points[i] = xy
            if not np.isfinite(xy).all():
                continue
            rois = extract_rois(rgb, xy, OPTIONS["roi"])
            descriptors[i], valid[i] = frame_features(rgb, rois)
            for j, name in enumerate(NAMES):
                reasons[i, j], counts[i, j] = rois[name].reason, rois[name].native_pixels
    if not np.isfinite(descriptors[valid]).all():
        raise RuntimeError("Nonfinite accepted ROI descriptor")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=descriptors, valid=valid, reasons=reasons, native_pixels=counts,
                        landmarks=points, source_frame_indices=INDEX.source_indices[start:end])
    temporary.replace(path)
    return video_id


def prepare(protocol, workers=6):
    from study.exp2_face_dinov3_frozen.run_main_regression import preflight, BASELINE, INDEX_PATH
    index, _, hashes, _, lab_hash = preflight()
    cache = HERE / "cache/native41"
    cache.mkdir(parents=True, exist_ok=True)
    contract = {"protocol_sha256": sha256(HERE / "protocol.json"), "roi_schema_sha256": sha256(HERE / "roi_schema.json"),
                "feature_schema_sha256": sha256(HERE / "feature_schema.json"), "landmarker_sha256": sha256(MODEL),
                "frame_index_sha256": sha256(INDEX_PATH), "lab_table_sha256": lab_hash, "task_records_sha256": hashes}
    manifest = cache / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise RuntimeError("Existing ROI cache contract changed")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")
    records = {target: pd.read_csv(BASELINE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
               for target in protocol["targets"]}
    video_ids = sorted(set(pd.concat(list(records.values())).video_id))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker,
                             initargs=(str(INDEX_PATH), protocol, str(cache))) as pool:
        futures = [pool.submit(extract_video, video) for video in video_ids]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 25 == 0:
                print(f"[roi-extract] videos={completed}/{len(video_ids)}", flush=True)
    vectors, valid_videos, audit = {}, set(), []
    for video in video_ids:
        with np.load(cache / f"{video}.npz", allow_pickle=False) as shard:
            common = shard["valid"].all(axis=1)
            if common.sum() >= protocol["roi"]["minimum_common_valid_frames_per_video"]:
                vectors[video], _ = aggregate_video(shard["features"], shard["valid"], protocol["roi"]["minimum_common_valid_frames_per_video"])
                valid_videos.add(video)
            audit.append({"video_id": video, "common_valid_frames": int(common.sum()), "included": video in valid_videos,
                          **{f"{name}_valid_frames": int(shard["valid"][:, j].sum()) for j, name in enumerate(NAMES)}})
    tables = HERE / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(tables / "roi_video_audit.csv", index=False)
    summaries = []
    for target, original in records.items():
        effective = original.loc[original.video_id.isin(valid_videos)].reset_index(drop=True)
        if set(effective.split) != {"train", "val", "test"} or effective.groupby("hospital_id").split.nunique().gt(1).any():
            raise RuntimeError(f"Invalid effective cohort: {target}")
        original.loc[~original.video_id.isin(valid_videos)].to_csv(tables / f"excluded_{target}.csv", index=False)
        effective.to_csv(tables / f"records_{target}.csv", index=False)
        np.save(tables / f"features_{target}.npy", np.stack([vectors[video] for video in effective.video_id]))
        for split, group in effective.groupby("split"):
            summaries.append({"target": target, "split": split, "videos": len(group), "patients": group.hospital_id.nunique(),
                              "source_videos": int(original.split.eq(split).sum()), "lab_events": group.clinical_event_id.nunique()})
    pd.DataFrame(summaries).to_csv(tables / "cohort_counts.csv", index=False)
    (tables / "preparation_manifest.json").write_text(json.dumps({**contract, "included_videos": len(valid_videos),
                                                                  "source_videos": len(video_ids)}, indent=2) + "\n")
    print(f"[roi-data-ready] included={len(valid_videos)}/{len(video_ids)} videos; common 41/82/164 features", flush=True)
    return contract
