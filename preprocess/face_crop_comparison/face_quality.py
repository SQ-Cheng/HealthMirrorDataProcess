"""Gate detection confidence and expanded-bbox containment in the source."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class QualityConfig:
    confidence_threshold: float = 0.80
    forehead_margin: float = 0.20
    maximum_outside_area_fraction: float = 0.10


def clipped_crop_bounds(frame, expanded):
    height, width = frame.shape[:2]
    x1, y1 = np.floor(expanded[:2]).astype(int)
    x2, y2 = np.ceil(expanded[2:]).astype(int)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


class FaceQualityGate:
    def __init__(self, config=None):
        self.config = config or QualityConfig()

    def assess(self, frame, box, keypoints):
        height, width = frame.shape[:2]
        x1, y1, x2, y2, confidence = box
        expanded = np.array([x1, y1 - self.config.forehead_margin * (y2 - y1), x2, y2])
        reasons = []
        valid_box = bool(np.isfinite(box).all() and x2 > x1 and y2 > y1)
        if not valid_box:
            reasons.append("invalid_detector_box")
        if not np.isfinite(confidence) or confidence < self.config.confidence_threshold:
            reasons.append("low_confidence")
        outside_fraction = np.nan
        if valid_box:
            area = (expanded[2] - expanded[0]) * (expanded[3] - expanded[1])
            intersection = (max(0, min(width, expanded[2]) - max(0, expanded[0]))
                            * max(0, min(height, expanded[3]) - max(0, expanded[1])))
            outside_area = max(0, area - intersection)
            outside_fraction = outside_area / area
            # Compare areas so exactly 10% remains accepted without subtraction error.
            if outside_area > self.config.maximum_outside_area_fraction * area:
                reasons.append("expanded_box_outside_source_over_10_percent")
        return {"confidence": float(confidence), "quality_accepted": not reasons,
                "expanded_box_outside_area_fraction": float(outside_fraction),
                "quality_reason": ";".join(reasons) if reasons else "accepted",
                **{f"expanded_{name}": float(value) for name, value in
                   zip(("x1", "y1", "x2", "y2"), expanded)}}

    def close(self):
        pass

    def manifest(self):
        return {"parameters": asdict(self.config),
                "rule": "Reject only when outside area / expanded bbox area > 0.10; exactly 10% is allowed",
                "accepted_crop": "Intersect accepted bbox with original image; no synthetic padding",
                "ordering": "confidence and expanded-box checks before Kalman update and alignment",
                "contour_based_rejection": False,
                "detector_confidence": "uncalibrated BlazeFace detection score, not a correctness probability"}
