"""Sample 100 patient-distinct native224 frames and render full-face/ROI panels."""

import argparse
from collections import Counter, OrderedDict, deque
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from study.common.face_video import decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from .roi import NAMES, extract_rois, landmark_schema


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / "preprocess/face_crop_comparison/models/face_landmarker.task"
COLORS = {"forehead": (218, 165, 36), "image_left_cheek": (38, 133, 201),
          "image_right_cheek": (26, 163, 123), "lips": (218, 68, 112)}
LABELS = {"forehead": "Forehead (top = y0)", "image_left_cheek": "Left cheek (image)",
          "image_right_cheek": "Right cheek (image)", "lips": "Lips (mouth excluded)"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def sample_frames(reference, index, targets, count, seed):
    tables = [pd.read_csv(reference / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
              for target in targets]
    records = pd.concat(tables, ignore_index=True)
    if records.groupby("video_id").hospital_id.nunique().gt(1).any():
        raise ValueError("A video has conflicting patient identities")
    records = records.drop_duplicates("video_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    records = records.iloc[rng.permutation(len(records))].drop_duplicates("hospital_id")
    groups = {mirror: deque(group.to_dict("records")) for mirror, group in records.groupby("mirror")}
    order = sorted(groups)
    selected = []
    while len(selected) < count:
        progressed = False
        for mirror in order:
            if groups[mirror] and len(selected) < count:
                row = groups[mirror].popleft()
                start, end = index.frame_range(row["video_id"])
                frame = int(rng.integers(start, end))
                selected.append({"preview_id": len(selected) + 1, "hospital_id": row["hospital_id"],
                                 "video_id": row["video_id"], "mirror": mirror, "global_frame_index": frame,
                                 "source_frame_index": int(index.source_indices[frame])})
                progressed = True
        if not progressed:
            raise ValueError("Too few distinct patients for requested preview count")
    return pd.DataFrame(selected)


def render(rgb, rois, row, failure):
    canvas = Image.new("RGB", (1280, 650), "#f0f3f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 17), f"Face {row.preview_id:03d} | {row.video_id} | source frame {row.source_frame_index}",
              fill="#263238", font=font(23))
    overlay = rgb.copy()
    for name, roi in rois.items():
        cv2.polylines(overlay, [np.rint(roi.polygon).astype(np.int32)], True, COLORS[name], 1, cv2.LINE_AA)
        if roi.hole is not None:
            cv2.polylines(overlay, [np.rint(roi.hole).astype(np.int32)], True, COLORS[name], 1, cv2.LINE_AA)
    canvas.paste(Image.fromarray(overlay).resize((448, 448), Image.Resampling.NEAREST), (24, 81))
    draw.text((24, 548), "Native RGB 224x224 + ROI boundaries", fill="#455a64", font=font(17))
    if failure:
        draw.text((24, 580), failure, fill="#bd3246", font=font(17))
    for k, name in enumerate(NAMES):
        left, top = 502 + (k % 2) * 386, 78 + (k // 2) * 278
        draw.rectangle((left, top, left + 358, top + 260), fill="white", outline="#c8d1d8", width=1)
        draw.rectangle((left, top, left + 7, top + 260), fill=COLORS[name])
        draw.text((left + 18, top + 10), LABELS[name], fill="#263238", font=font(18))
        if name not in rois:
            draw.text((left + 18, top + 94), "Not extracted", fill="#bd3246", font=font(19))
            continue
        roi = rois[name]
        ys, xs = np.where(roi.mask)
        if not len(xs):
            draw.text((left + 18, top + 94), "Empty native ROI", fill="#bd3246", font=font(19))
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        pixels, mask = rgb[y0:y1, x0:x1], roi.mask[y0:y1, x0:x1].astype(bool)
        height, width = mask.shape
        yy, xx = np.indices((height, width))
        grid = (yy // 4 + xx // 4) % 2
        checker = np.where(grid[..., None] == 0, 222, 239).astype(np.uint8)
        checker = np.repeat(checker, 3, axis=2)
        display = np.where(mask[..., None], pixels, checker)
        scale = min(310 / width, 160 / height)
        image = Image.fromarray(display).resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST)
        canvas.paste(image, (left + 20 + (310 - image.width) // 2, top + 45 + (160 - image.height) // 2))
        state = "Geometry OK" if roi.accepted else "Rejected"
        draw.text((left + 18, top + 212), f"Native {width}x{height} | pixels {roi.native_pixels} | {state}",
                  fill="#278245" if roi.accepted else "#bd3246", font=font(11))
        if not roi.accepted:
            text = roi.reason.replace(";", " / ")
            for line, start in enumerate(range(0, min(len(text), 96), 48)):
                draw.text((left + 18, top + 230 + line * 13), text[start:start + 48], fill="#bd3246", font=font(11))
    draw.text((24, 617), "Native ROI pixels; display scaling only. Features use original pixels, never these preview images.",
              fill="#455a64", font=font(15))
    return canvas, Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    reference = ROOT / protocol["reference"]
    reference_protocol = json.loads((reference / "main_protocol.json").read_text())
    index_path = Path(reference_protocol["frame_index"])
    index = FrameOffsetIndex.load(index_path)
    if sha256(index_path) != reference_protocol["frame_index_sha256"] or not _index_is_reusable(index_path.parent, index.video_ids, "20frame"):
        raise RuntimeError("Current native224 reference frame index changed")
    selected = sample_frames(reference, index, protocol["targets"], args.count, args.seed)
    figures = HERE / f"outputs/figures/roi_preview_{args.count}"
    tables = HERE / f"outputs/tables/roi_preview_{args.count}"
    if figures.exists() and list(figures.glob("*.png")):
        raise FileExistsError(f"Preview results already exist: {figures}")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    selected.to_csv(tables / "selected_frames.csv", index=False)
    (HERE / "roi_schema.json").write_text(json.dumps(landmark_schema(), indent=2) + "\n")
    shutil.copy2(HERE / "protocol.json", tables / "protocol_snapshot.json")
    shutil.copy2(HERE / "roi_schema.json", tables / "roi_schema_snapshot.json")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=protocol["landmarker"]["min_face_detection_confidence"],
        min_face_presence_confidence=protocol["landmarker"]["min_face_presence_confidence"],
        output_face_blendshapes=False, output_facial_transformation_matrixes=False)
    records, overlays, handles, decoders = [], [], {}, OrderedDict()
    points_all = []
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            for row in selected.itertuples(index=False):
                video = index.video_lookup[row.video_id]
                path = str(index.video_paths[video])
                if path not in handles:
                    handles[path] = open(path, "rb")
                rgb = np.ascontiguousarray(decode_indexed_frame(index, video, row.global_frame_index, handles[path], decoders).permute(1, 2, 0).numpy())
                detection = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                points = np.full((478, 2), np.nan, np.float32)
                rois, failure = {}, ""
                if not detection.face_landmarks:
                    failure = "No face landmarks"
                else:
                    points = np.asarray([(p.x * 224, p.y * 224) for p in detection.face_landmarks[0]], np.float32)
                    try:
                        rois = extract_rois(rgb, points, protocol["roi"])
                    except ValueError as error:
                        failure = str(error)
                points_all.append(points)
                composite, overlay = render(rgb, rois, row, failure)
                filename = f"face_{row.preview_id:03d}_{row.video_id}_frame_{row.source_frame_index}.png"
                composite.save(figures / filename)
                overlays.append(overlay)
                for name in NAMES:
                    roi = rois.get(name)
                    records.append({**row._asdict(), "roi": name, "accepted": roi.accepted if roi else False,
                                    "reason": roi.reason if roi else failure, "native_pixels": roi.native_pixels if roi else 0,
                                    "outside_fraction": roi.outside_fraction if roi else np.nan, "preview_file": filename})
                if row.preview_id % 10 == 0:
                    print(f"[roi-preview] {row.preview_id}/{args.count}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    audit = pd.DataFrame(records)
    audit.to_csv(tables / "roi_audit.csv", index=False)
    np.savez_compressed(tables / "landmarks.npz", xy=np.stack(points_all), preview_id=selected.preview_id.to_numpy())
    summary = audit.groupby("roi").agg(frames=("accepted", "size"), accepted=("accepted", "sum"),
                                      native_pixels_median=("native_pixels", "median"))
    summary.to_csv(tables / "summary.csv")
    manifest = {"stage": "roi_preview_only_no_training", "seed": args.seed, "frames": args.count,
                "distinct_patients": selected.hospital_id.nunique(), "distinct_videos": selected.video_id.nunique(),
                "mirror_counts": dict(Counter(selected.mirror)), "sampling": "patient-distinct; round-robin mirrors; one seeded frame per video",
                "model_sha256": sha256(MODEL), "frame_index_sha256": sha256(index_path),
                "roi_schema_sha256": sha256(tables / "roi_schema_snapshot.json"),
                "protocol_sha256": sha256(tables / "protocol_snapshot.json"),
                "roi_schema_snapshot": "roi_schema_snapshot.json", "protocol_snapshot": "protocol_snapshot.json",
                "all_four_rois_accepted": int(audit.groupby("preview_id").accepted.all().sum()),
                "figures": str(figures), "tables": str(tables), "training_started": False}
    (tables / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = min(10, args.count)
    rows = int(np.ceil(args.count / columns))
    overview = Image.new("RGB", (columns * 174, rows * 192), "#f0f3f6")
    draw = ImageDraw.Draw(overview)
    for i, image in enumerate(overlays):
        x, y = (i % columns) * 174, (i // columns) * 192
        overview.paste(image.resize((164, 164)), (x + 5, y + 23))
        accepted = bool(audit.loc[audit.preview_id.eq(i + 1), "accepted"].all())
        draw.text((x + 6, y + 3), f"{i + 1:03d} {'OK' if accepted else 'CHECK'}", font=font(13),
                  fill="#278245" if accepted else "#bd3246")
    overview.save(figures.parent / f"roi_preview_{args.count}_overview.png")
    print(summary.to_string(), flush=True)
    print(f"[roi-preview-complete] {figures} | all four accepted={manifest['all_four_rois_accepted']}/{args.count}", flush=True)


if __name__ == "__main__":
    main()
