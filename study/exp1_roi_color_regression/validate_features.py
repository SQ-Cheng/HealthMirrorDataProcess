"""Validate native41 descriptors using the existing 100-frame landmark audit."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.common.face_video import decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from .features import FEATURE_NAMES, FEATURE_COUNT, feature_schema, frame_features
from .preview_rois import sha256, HERE, ROOT
from .roi import NAMES, extract_rois


def main():
    source = HERE / "outputs/tables/roi_preview_100"
    preview_manifest = json.loads((source / "manifest.json").read_text())
    selected = pd.read_csv(source / "selected_frames.csv", dtype={"hospital_id": str, "video_id": str})
    old_audit = pd.read_csv(source / "roi_audit.csv")
    protocol = json.loads((HERE / "protocol.json").read_text())
    reviewed_geometry = json.loads((source / "roi_schema_snapshot.json").read_text())
    current_geometry = json.loads((HERE / "roi_schema.json").read_text())
    if any(current_geometry.get(key) != value for key, value in reviewed_geometry.items() if key != "stage"):
        raise RuntimeError("User-reviewed ROI definition changed")
    reference = ROOT / protocol["reference"]
    reference_protocol = json.loads((reference / "main_protocol.json").read_text())
    index_path = Path(reference_protocol["frame_index"])
    index = FrameOffsetIndex.load(index_path)
    if sha256(index_path) != preview_manifest["frame_index_sha256"] or not _index_is_reusable(index_path.parent, index.video_ids, "20frame"):
        raise RuntimeError("Reviewed preview source frames changed")
    with np.load(source / "landmarks.npz", allow_pickle=False) as values:
        points = values["xy"]
        np.testing.assert_array_equal(values["preview_id"], selected.preview_id)
    if json.loads((HERE / "feature_schema.json").read_text()) != feature_schema():
        raise RuntimeError("Published feature schema differs from implementation")
    descriptors = np.full((len(selected), len(NAMES), FEATURE_COUNT), np.nan, np.float32)
    validity = np.zeros((len(selected), len(NAMES)), bool)
    records, handles, decoders = [], {}, OrderedDict()
    try:
        for i, row in enumerate(selected.itertuples(index=False)):
            video = index.video_lookup[row.video_id]
            path = str(index.video_paths[video])
            if path not in handles:
                handles[path] = open(path, "rb")
            rgb = np.ascontiguousarray(decode_indexed_frame(index, video, row.global_frame_index, handles[path], decoders).permute(1, 2, 0).numpy())
            rois = extract_rois(rgb, points[i], protocol["roi"]) if np.isfinite(points[i]).all() else {}
            descriptors[i], validity[i] = frame_features(rgb, rois)
            for j, name in enumerate(NAMES):
                roi = rois.get(name)
                original = old_audit.loc[old_audit.preview_id.eq(row.preview_id) & old_audit.roi.eq(name)].iloc[0]
                if bool(original.accepted) != bool(validity[i, j]) or int(original.native_pixels) != (roi.native_pixels if roi else 0):
                    raise RuntimeError("Reviewed ROI geometry/acceptance changed")
                record = {**row._asdict(), "roi": name, "accepted": bool(validity[i, j]),
                          "native_pixels": roi.native_pixels if roi else 0,
                          "reason": roi.reason if roi else "no_cached_face_landmarks"}
                record.update(dict(zip(FEATURE_NAMES, descriptors[i, j])))
                records.append(record)
    finally:
        for handle in handles.values():
            handle.close()
    if not np.isfinite(descriptors[validity]).all() or not np.isnan(descriptors[~validity]).all():
        raise RuntimeError("Invalid feature/validity handling")
    destination = HERE / "outputs/tables/native41_preview_validation"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(destination / "native_roi_features.csv", index=False)
    np.savez_compressed(destination / "native_roi_features.npz", features=descriptors, valid=validity,
                        feature_names=np.asarray(FEATURE_NAMES), roi_names=np.asarray(NAMES),
                        preview_id=selected.preview_id.to_numpy())
    report = {"frames": len(selected), "dimensions_per_roi": FEATURE_COUNT,
              "accepted_roi_vectors": int(validity.sum()), "all_four_valid_frames": int(validity.all(axis=1).sum()),
              "accepted_by_roi": {name: int(validity[:, j].sum()) for j, name in enumerate(NAMES)},
              "finite_features_for_every_accepted_roi": True, "geometry_and_native_pixel_counts_match_review": True,
              "source": "same reviewed native224 frames; reused saved landmarks; no MediaPipe rerun",
              "spatial_resize": False, "invalid_roi_policy": "explicit validity mask with NaN descriptors; no imputation",
              "protocol_sha256": sha256(HERE / "protocol.json"), "feature_schema_sha256": sha256(HERE / "feature_schema.json"),
              "roi_schema_sha256": sha256(HERE / "roi_schema.json"), "training_started": False}
    (destination / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
