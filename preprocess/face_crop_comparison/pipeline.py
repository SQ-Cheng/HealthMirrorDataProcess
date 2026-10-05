"""Shared detection, temporal association, square crops and lossless video IO."""

from dataclasses import dataclass
from fractions import Fraction
import mmap
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class CropConfig:
    size: int = 224
    confidence: float = 0.5
    forehead_margin: float = 0.20
    side_margin: float = 0.08
    bottom_margin: float = 0.05
    smoothing_seconds: float = 0.08
    hold_seconds: float = 0.20
    reassociate_seconds: float = 0.50


class MediaPipeDetector:
    def __init__(self, config):
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_DIR / "blaze_face_short_range.tflite")
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=config.confidence,
            min_suppression_threshold=0.3,
        )
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, bgr):
        boxes, _ = self.detect_with_keypoints(bgr)
        return boxes

    def detect_with_keypoints(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        )
        detections = []
        landmarks = []
        for face in result.detections:
            box = face.bounding_box
            detections.append([
                box.origin_x, box.origin_y, box.origin_x + box.width,
                box.origin_y + box.height, face.categories[0].score,
            ])
            landmarks.append([[point.x * bgr.shape[1], point.y * bgr.shape[0]]
                              for point in face.keypoints])
        return (np.asarray(detections, dtype=float).reshape(-1, 5),
                np.asarray(landmarks, dtype=float).reshape(-1, 6, 2))

    def close(self):
        self.detector.close()


class InsightFaceDetector:
    def __init__(self, config, gpu=0):
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD

        # Reuse the installed CUDA libraries; do not install another Torch stack.
        ort.preload_dlls()
        providers = [
            ("CUDAExecutionProvider", {"device_id": gpu}), "CPUExecutionProvider"
        ] if gpu >= 0 else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        model_path = str(MODEL_DIR / "det_10g.onnx")
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        self.detector = SCRFD(model_file=model_path, session=session)
        self.detector.prepare(
            ctx_id=gpu, input_size=(640, 640), det_thresh=config.confidence
        )
        self.providers = self.detector.session.get_providers()
        self.model_class = type(self.detector).__name__
        if gpu >= 0 and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError("InsightFace CUDA provider failed to initialize")

    def detect(self, bgr):
        boxes, _ = self.detector.detect(bgr, max_num=0)
        return np.asarray(boxes, dtype=float).reshape(-1, 5)

    def close(self):
        pass


def intersection_over_union(first, second):
    low = np.maximum(first[:2], second[:2])
    high = np.minimum(first[2:4], second[2:4])
    overlap = np.prod(np.maximum(high - low, 0))
    area_first = np.prod(first[2:4] - first[:2])
    area_second = np.prod(second[2:4] - second[:2])
    return float(overlap / max(area_first + area_second - overlap, 1e-6))


class FaceCropper:
    def __init__(self, config):
        self.config = config
        self.last_box = None
        self.last_time = None
        self.smooth = None
        self.previous_time = None

    def select(self, boxes, timestamp, width, height):
        boxes = boxes[np.isfinite(boxes).all(axis=1)]
        boxes = boxes[(boxes[:, 2] - boxes[:, 0] >= 24)
                      & (boxes[:, 3] - boxes[:, 1] >= 24)]
        if not len(boxes):
            return None, "no_detection"
        if self.last_time is not None and timestamp - self.last_time <= self.config.reassociate_seconds:
            previous_center = (self.last_box[:2] + self.last_box[2:4]) / 2
            previous_size = max(self.last_box[2:4] - self.last_box[:2])
            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
            distances = np.linalg.norm(centers - previous_center, axis=1) / previous_size
            ious = np.array([intersection_over_union(box, self.last_box) for box in boxes])
            allowed = (ious >= 0.15) | (distances <= 0.5)
            if not allowed.any():
                return None, "association_rejected"
            scores = np.where(allowed, ious + 0.15 * boxes[:, 4] - 0.1 * distances, -np.inf)
            order = np.argsort(scores)[::-1]
            if len(order) > 1 and scores[order[0]] - scores[order[1]] < 0.10:
                return None, "ambiguous_faces"
            return boxes[order[0]], "detected"
        areas = np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
        distances = np.linalg.norm((centers - [width / 2, height / 2]) / [width, height], axis=1)
        scores = areas * boxes[:, 4] / (1 + 2 * distances)
        order = np.argsort(scores)[::-1]
        if len(order) > 1 and scores[order[0]] < 1.5 * scores[order[1]]:
            return None, "ambiguous_faces"
        self.smooth = None
        return boxes[order[0]], "detected"

    def crop(self, frame, boxes, timestamp):
        height, width = frame.shape[:2]
        chosen, status = self.select(boxes, timestamp, width, height)
        row = {"status": status, "candidate_count": len(boxes), "train_eligible": False,
               "confidence": np.nan, "upsampled": False, "boundary_shifted": False}
        if chosen is not None:
            x1, y1, x2, y2, score = chosen
            face_width, face_height = x2 - x1, y2 - y1
            top = y1 - self.config.forehead_margin * face_height
            bottom = y2 + self.config.bottom_margin * face_height
            side = max(face_width * (1 + 2 * self.config.side_margin), bottom - top)
            measured = np.array([(x1 + x2) / 2, (top + bottom) / 2, np.log(side)])
            if self.smooth is None:
                self.smooth = measured
            else:
                dt = timestamp - self.previous_time
                alpha = 1 - np.exp(-dt / self.config.smoothing_seconds)
                self.smooth = self.smooth + alpha * (measured - self.smooth)
            # Keep the current measurement inside the stabilized crop during motion.
            side = max(np.exp(self.smooth[2]),
                       2 * abs(measured[0] - self.smooth[0]) + side,
                       2 * abs(measured[1] - self.smooth[1]) + side)
            center = self.smooth[:2]
            self.last_box = chosen.copy()
            self.last_time = timestamp
            self.previous_time = timestamp
            row.update({"train_eligible": True, "confidence": float(score),
                        "box_x1": float(x1), "box_y1": float(y1),
                        "box_x2": float(x2), "box_y2": float(y2)})
        elif self.last_time is not None and timestamp - self.last_time <= self.config.hold_seconds:
            side = np.exp(self.smooth[2])
            center = self.smooth[:2]
            row["status"] = "held_short_gap"
        else:
            return np.zeros((self.config.size, self.config.size, 3), np.uint8), row
        side = int(np.ceil(min(side, width, height)))
        left = int(round(center[0] - side / 2))
        top = int(round(center[1] - side / 2))
        shifted_left = int(np.clip(left, 0, width - side))
        shifted_top = int(np.clip(top, 0, height - side))
        row.update({"crop_x": shifted_left, "crop_y": shifted_top, "crop_side": side,
                    "boundary_shifted": left != shifted_left or top != shifted_top,
                    "upsampled": side < self.config.size})
        roi = frame[shifted_top:shifted_top + side, shifted_left:shifted_left + side]
        interpolation = cv2.INTER_AREA if side >= self.config.size else cv2.INTER_LINEAR
        result = cv2.resize(roi, (self.config.size, self.config.size), interpolation=interpolation)
        row["interpolation"] = "area" if side >= self.config.size else "linear"
        return result, row


class RawRecording:
    """Index MJPEG payloads so a corrupt frame never shifts timestamp alignment."""

    def __init__(self, path):
        self.path = Path(path)
        timestamps = pd.read_csv(str(path) + ".ts", skipinitialspace=True)
        if list(timestamps.columns) != ["frame", "ts"]:
            raise ValueError(f"Unexpected timestamp schema: {path}")
        indices = pd.to_numeric(timestamps["frame"], errors="raise").to_numpy()
        self.timestamps = pd.to_numeric(timestamps["ts"], errors="raise").to_numpy(float)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError(f"Non-contiguous timestamp frame IDs: {path}")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError(f"Invalid or nonmonotonic recorder timestamps: {path}")
        self.handle = self.path.open("rb")
        self.mapped = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.ranges = []
        position = 0
        while True:
            start = self.mapped.find(b"\xff\xd8", position)
            if start < 0:
                break
            end_marker = self.mapped.find(b"\xff\xd9", start + 2)
            end = len(self.mapped) if end_marker < 0 else end_marker + 2
            self.ranges.append((start, end))
            position = end
        if len(self.ranges) != len(self.timestamps):
            self.close()
            raise ValueError(f"JPEG/timestamp count mismatch: {path}: "
                             f"{len(self.ranges)} vs {len(self.timestamps)}")
        if len(self.ranges) < 2:
            self.close()
            raise ValueError(f"Too few raw frames: {path}")

    def frame(self, index):
        start, end = self.ranges[index]
        return cv2.imdecode(np.frombuffer(self.mapped[start:end], dtype=np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self.mapped.close()
        self.handle.close()


class VideoWriter:
    def __init__(self, path, width, height, lossless=True):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("ffv1" if lossless else "libx264", rate=30)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "bgr0" if lossless else "yuv420p"
        self.stream.time_base = Fraction(1, 1000)
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.codec_context.thread_count = 2
        self.stream.options = ({"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
                               if lossless else {"crf": "18", "preset": "fast"})
        self.last_pts = -1

    def write(self, bgr, elapsed_seconds):
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        pts = int(round(elapsed_seconds * 1000))
        if pts <= self.last_pts:
            raise ValueError("Frame timestamps cannot be represented at millisecond resolution")
        frame.pts = pts
        frame.time_base = self.stream.codec_context.time_base
        self.last_pts = pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
