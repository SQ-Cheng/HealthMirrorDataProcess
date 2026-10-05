"""One MediaPipe detection/Kalman box, with and without eye-level alignment."""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .pipeline import CropConfig, FaceCropper


@dataclass
class ComparisonConfig:
    size: int = 224
    enable_alignment: bool = True
    forehead_margin: float = 0.20
    bbox_process_noise: float = 0.01
    bbox_measurement_noise: float = 0.5
    bbox_initial_error: float = 1.0
    angle_process_noise: float = 0.5
    angle_measurement_noise: float = 4.0
    reference_interval: float = 1 / 30
    hold_seconds: float = 0.20
    reset_seconds: float = 0.50
    max_alignment_angle_degrees: float = 60.0
    max_padding_fraction: float = 0.05


class KalmanFilter1D:
    def __init__(self, process_noise, measurement_noise, state, error, reference_interval):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = float(state)
        self.error = float(error)
        self.reference_interval = reference_interval

    def update(self, measurement, dt):
        prediction_error = self.error + self.process_noise * (dt / self.reference_interval) ** 2
        gain = prediction_error / (prediction_error + self.measurement_noise)
        self.state += gain * (measurement - self.state)
        self.error = (1 - gain) * prediction_error
        return self.state


class KalmanComparison:
    def __init__(self, config=None, quality_gate=None):
        self.config = config or ComparisonConfig()
        self.quality_gate = quality_gate
        self.selector = FaceCropper(CropConfig())
        self.box_filters = None
        self.angle_filter = None
        self.last_detection_time = None
        self.last_angle_time = None
        self.last_box = None

    def process(self, frame, boxes, landmarks, timestamp):
        size = self.config.size
        empty = np.zeros((size, size, 3), np.uint8)
        outputs = {"direct": empty.copy(), "aligned": empty.copy()}
        row = {"status": "no_detection", "candidate_count": len(boxes),
               "direct_eligible": False, "aligned_eligible": False,
               "alignment_status": "no_detection", "padding_fraction": np.nan}
        height, width = frame.shape[:2]
        chosen, status = self.selector.select(boxes, timestamp, width, height)
        row["status"] = status
        fresh = chosen is not None
        if fresh:
            index = next(i for i, candidate in enumerate(boxes) if np.array_equal(candidate, chosen))
            if self.quality_gate is not None:
                quality = self.quality_gate.assess(frame, chosen, landmarks[index])
                row.update(quality)
                row["raw_quality_accepted"] = quality["quality_accepted"]
                if not quality["quality_accepted"]:
                    row["status"] = "quality_rejected"
                    row["alignment_status"] = "not_run_quality_rejected"
                    return outputs, row
            x1, y1, x2, y2, confidence = chosen
            expanded = np.array([x1 / width, (y1 - self.config.forehead_margin * (y2 - y1)) / height,
                                 x2 / width, y2 / height])
            reset = self.last_detection_time is None or timestamp - self.last_detection_time > self.config.reset_seconds
            if reset:
                self.box_filters = [KalmanFilter1D(
                    self.config.bbox_process_noise, self.config.bbox_measurement_noise,
                    value, self.config.bbox_initial_error, self.config.reference_interval
                ) for value in expanded]
                self.angle_filter = None
                self.last_angle_time = None
            else:
                dt = timestamp - self.last_detection_time
                for filter_, value in zip(self.box_filters, expanded):
                    filter_.update(value, dt)
            normalized = np.clip([filter_.state for filter_ in self.box_filters], 0, 1)
            self.last_box = normalized * [width, height, width, height]
            self.last_detection_time = timestamp
            self.selector.last_box = chosen.copy()
            self.selector.last_time = timestamp
            row.update({"confidence": float(confidence),
                        **{f"raw_box_{key}": float(value) for key, value in zip(("x1", "y1", "x2", "y2"), chosen[:4])}})
            if self.config.enable_alignment:
                eyes = landmarks[index, :2]
                eyes = eyes[np.argsort(eyes[:, 0])]
                separation = np.linalg.norm(eyes[1] - eyes[0])
                angle = math.degrees(math.atan2(*(eyes[1] - eyes[0])[::-1]))
                angle_valid = (np.isfinite(eyes).all() and separation >= 0.10 * (x2 - x1)
                               and abs(angle) <= self.config.max_alignment_angle_degrees)
                row["raw_eye_angle_degrees"] = angle
                if angle_valid:
                    if self.angle_filter is None or self.last_angle_time is None or timestamp - self.last_angle_time > self.config.reset_seconds:
                        self.angle_filter = KalmanFilter1D(
                            self.config.angle_process_noise, self.config.angle_measurement_noise,
                            angle, self.config.angle_measurement_noise, self.config.reference_interval
                        )
                    else:
                        self.angle_filter.update(angle, timestamp - self.last_angle_time)
                    self.last_angle_time = timestamp
                    row["alignment_status"] = "aligned"
                else:
                    row["alignment_status"] = "invalid_eye_geometry"
        elif self.last_detection_time is not None and timestamp - self.last_detection_time <= self.config.hold_seconds:
            row["status"] = "held_short_gap"
            row["alignment_status"] = "held_short_gap"
        else:
            return outputs, row
        x1, y1 = np.floor(self.last_box[:2]).astype(int)
        x2, y2 = np.ceil(self.last_box[2:]).astype(int)
        if x2 <= x1 or y2 <= y1:
            row["status"] = "invalid_box"
            return outputs, row
        row.update({"crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                    "crop_width": x2 - x1, "crop_height": y2 - y1,
                    "upsampled": min(x2 - x1, y2 - y1) < size})
        if self.quality_gate is not None:
            if not fresh:
                row.update({"quality_accepted": False, "quality_reason": "held_short_gap"})
        outputs["direct"] = cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)
        row["direct_eligible"] = fresh
        if not self.config.enable_alignment:
            row["alignment_status"] = "disabled"
            return outputs, row
        angle_available = self.last_angle_time is not None and timestamp - self.last_angle_time <= self.config.hold_seconds
        if not angle_available or row["alignment_status"] == "invalid_eye_geometry":
            return outputs, row
        angle = self.angle_filter.state
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        rotation = np.vstack([cv2.getRotationMatrix2D(center, angle, 1), [0, 0, 1]])
        sx, sy = size / (x2 - x1), size / (y2 - y1)
        # Match resize's pixel-center convention while combining rotation/crop/resize.
        resize = np.array([[sx, 0, sx * (0.5 - x1) - 0.5],
                           [0, sy, sy * (0.5 - y1) - 0.5], [0, 0, 1]])
        matrix = (resize @ rotation)[:2]
        outputs["aligned"] = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        corners = np.array([[-0.5, -0.5], [size - 0.5, -0.5],
                            [size - 0.5, size - 0.5], [-0.5, size - 0.5]], np.float32)
        source_corners = cv2.transform(corners[None], cv2.invertAffineTransform(matrix))[0]
        row["alignment_angle_degrees"] = angle
        if ((source_corners[:, 0] >= 0).all() and (source_corners[:, 0] <= width - 1).all()
                and (source_corners[:, 1] >= 0).all() and (source_corners[:, 1] <= height - 1).all()):
            padding = 0.0
        else:
            source_mask = np.full((height, width), 255, np.uint8)
            valid = cv2.warpAffine(source_mask, matrix, (size, size), flags=cv2.INTER_LINEAR)
            padding = float((valid < 255).mean())
        row["padding_fraction"] = padding
        row["aligned_eligible"] = fresh and padding <= self.config.max_padding_fraction
        row.update({f"alignment_m{r}{c}": float(matrix[r, c]) for r in range(2) for c in range(3)})
        return outputs, row

    def manifest(self):
        return {"parameters": asdict(self.config),
                "common_geometry": "raw MediaPipe box; extend top by 20% original box height; no side/bottom expansion; rectangular ROI",
                "bbox_filter": "four scalar Kalman filters on normalized expanded bbox coordinates; source elapsed-time dt",
                "alignment": "eye-line roll correction around the shared bbox center; Kalman-filtered angle in degrees",
                "resize": "both paths use bilinear interpolation and the same rectangular bbox extent",
                "aligned_sampling": "one composed rotation/crop/resize warp from original RGB; source-border padding audited",
                "evaluation": "paired frames eligible for both methods; no fallback from alignment to direct resize"}
