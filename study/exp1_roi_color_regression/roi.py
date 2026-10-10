"""User-approved ROI polygons and native masks on original 224 RGB pixels."""

from dataclasses import dataclass

import cv2
import numpy as np


CHEEK_A = (117, 118, 119, 100, 36, 205, 187, 123)
CHEEK_B = (346, 347, 348, 329, 266, 425, 411, 352)
BROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
TEMPLES = (127, 356)
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
INNER_MOUTH = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
NAMES = ("forehead", "image_left_cheek", "image_right_cheek", "lips")


@dataclass
class ROI:
    name: str
    polygon: np.ndarray
    hole: np.ndarray | None
    mask: np.ndarray
    accepted: bool
    reason: str
    native_pixels: int
    outside_fraction: float


def polygons(points, options):
    if points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid_landmark_coordinates")
    brow = points[list(BROWS)]
    brow = brow[np.argsort(brow[:, 0])]
    temple_x = sorted(points[list(TEMPLES), 0])
    center, span = np.mean(temple_x), np.ptp(temple_x)
    half = .5 * options["forehead_width_fraction_of_temple_span"] * span
    left, right = center - half, center + half
    interior = brow[(brow[:, 0] > left) & (brow[:, 0] < right)].copy()
    lower = np.vstack(([left, np.interp(left, brow[:, 0], brow[:, 1])], interior,
                       [right, np.interp(right, brow[:, 0], brow[:, 1])]))
    lower[:, 1] -= options["forehead_lower_margin_above_eyebrows_pixels"]
    forehead = np.vstack(([left, 0.], [right, 0.], lower[::-1]))
    cheeks = [points[list(CHEEK_A)], points[list(CHEEK_B)]]
    cheeks.sort(key=lambda polygon: polygon[:, 0].mean())
    return {"forehead": (forehead, None), "image_left_cheek": (cheeks[0], None),
            "image_right_cheek": (cheeks[1], None),
            "lips": (points[list(OUTER_LIPS)], points[list(INNER_MOUTH)])}


def polygon_mask(polygon, hole, shape):
    height, width = shape
    coordinates = polygon if hole is None else np.vstack((polygon, hole))
    low = np.floor(np.minimum(coordinates.min(axis=0), [0, 0])).astype(int)
    high = np.ceil(np.maximum(coordinates.max(axis=0) + 1, [width, height])).astype(int)
    if np.any(high - low > 4 * max(height, width)):
        raise ValueError("implausible_roi_geometry")
    complete = np.zeros((int(high[1] - low[1]), int(high[0] - low[0])), np.uint8)
    cv2.fillPoly(complete, [np.rint(polygon - low).astype(np.int32)], 1)
    if hole is not None:
        cv2.fillPoly(complete, [np.rint(hole - low).astype(np.int32)], 0)
    total = int(complete.sum())
    in_image = complete[-low[1]:height - low[1], -low[0]:width - low[0]].copy()
    retained = int(in_image.sum())
    outside = (total - retained) / total if total else 1.
    return in_image, outside


def extract_rois(rgb, points, options):
    if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
        raise ValueError("expected_uint8_native224_RGB")
    result = {}
    for name, (polygon, hole) in polygons(points, options).items():
        mask, outside = polygon_mask(polygon, hole, rgb.shape[:2])
        if "cheek" in name:
            inset = options["cheek_inset_pixels"]
            mask = cv2.erode(mask, np.ones((2 * inset + 1, 2 * inset + 1), np.uint8))
        native = int(mask.sum())
        minimum = options["minimum_native_valid_pixels_lips" if name == "lips" else "minimum_native_valid_pixels_skin"]
        reasons = []
        if abs(cv2.contourArea(polygon.astype(np.float32))) < 1:
            reasons.append("degenerate_polygon")
        if outside > options["maximum_outside_polygon_fraction"]:
            reasons.append("roi_outside_image_over_10_percent")
        if native < minimum:
            reasons.append("too_few_native_valid_pixels")
        result[name] = ROI(name, polygon, hole, mask, not reasons,
                           ";".join(reasons) if reasons else "accepted", native, outside)
    return result


def landmark_schema():
    return {"schema_version": 1, "stage": "roi_geometry_user_approved", "user_review_date": "2026-10-10",
            "cheek_a": list(CHEEK_A), "cheek_b": list(CHEEK_B), "upper_eyebrows": list(BROWS),
            "temples": list(TEMPLES), "outer_lips": list(OUTER_LIPS), "inner_mouth": list(INNER_MOUTH),
            "left_right": "sort cheek polygon mean x; names refer to image coordinates",
            "forehead_top_y": 0, "coordinates": "MediaPipe normalized xy multiplied by original image width/height"}
