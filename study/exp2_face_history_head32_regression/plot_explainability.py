"""Generate input-gradient and occlusion maps for deterministic test examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torchvision.io import ImageReadMode, decode_jpeg

from .config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUTPUT_DIR,
    REFERENCE_INDEX_DIR,
    WEIGHTS_DIR,
)
from .frame_index import FrameOffsetIndex
from .history_data import HistoryFeatureStore
from .models import build_pretrained_model
from .scaling import RobustTargetScaler


ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
INTERPOLATION = {
    "mobilenet_v3_small": "bilinear",
    "efficientnet_b0": "bicubic",
}
DEFAULT_TARGET = "po2_low"
TARGET_LABELS = {
    "hemoglobin_low": "Hemoglobin",
    "po2_low": "PO2",
}
TARGET_UNITS = {
    "hemoglobin_low": "g/L",
    "po2_low": "mmHg",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_memory = [
        torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
    ]
    return torch.device(f"cuda:{int(np.argmax(free_memory))}")


def _select_examples(records: pd.DataFrame, count: int) -> pd.DataFrame:
    test = records.loc[records["split"].eq("test")].copy()
    if len(test) < count:
        raise ValueError(f"Only {len(test)} test videos are available for {count} examples")
    normal_count = count // 2
    abnormal_count = count - normal_count
    groups = (
        (test.loc[test["abnormal_score"].lt(0)].copy(), normal_count),
        (test.loc[test["abnormal_score"].gt(0)].copy(), abnormal_count),
    )
    selected = []
    for group, group_count in groups:
        if len(group) < group_count:
            raise ValueError(
                f"Need {group_count} examples on each score side, found {len(group)}"
            )
        scores = group["raw_value"].to_numpy(np.float64)
        selected_indices: list[int] = []
        for quantile in np.linspace(0.0, 1.0, group_count):
            target_value = float(np.quantile(scores, quantile))
            candidates = np.argsort(np.abs(scores - target_value), kind="stable")
            selected_indices.append(
                next(
                    int(index)
                    for index in candidates
                    if int(index) not in selected_indices
                )
            )
        selected.append(group.iloc[selected_indices])
    result = pd.concat(selected, ignore_index=True)
    result.insert(0, "case", np.arange(1, len(result) + 1, dtype=np.int64))
    return result


def _decode_frame(index: FrameOffsetIndex, video_id: str):
    start, end = index.frame_range(video_id)
    global_frame_index = start + (end - start) // 2
    index_video = index.video_lookup[str(video_id)]
    path = str(index.video_paths[index_video])
    byte_start = int(index.starts[global_frame_index])
    byte_end = int(index.ends[global_frame_index])
    with open(path, "rb") as handle:
        handle.seek(byte_start)
        payload = handle.read(byte_end - byte_start)
    encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    image = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
    return image, global_frame_index, int(index.source_indices[global_frame_index]), path


def _prepare_image(image: torch.Tensor, architecture: str, device: torch.device):
    image = image.unsqueeze(0).to(device).float().div_(255.0)
    resized = functional.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode=INTERPOLATION[architecture],
        align_corners=False,
        antialias=True,
    )
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    normalized = ((resized - mean) / std).contiguous(
        memory_format=torch.channels_last
    )
    display = resized[0].permute(1, 2, 0).detach().cpu().numpy()
    return normalized, display


def _history_tensors(store: HistoryFeatureStore, video_id: str, device: torch.device):
    history_row = store.lookup().get(str(video_id))
    if history_row is None:
        raise KeyError(f"Missing history features for {video_id}")
    start = int(store.offsets[history_row])
    end = int(store.offsets[history_row + 1])
    count = end - start
    sequence_length = max(1, count)
    features = torch.zeros((1, sequence_length, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((1, sequence_length), dtype=torch.bool, device=device)
    if count:
        features[0, :count] = torch.from_numpy(store.features[start:end]).to(device)
        mask[0, :count] = True
    return features, mask, count


def _input_gradient(model, image, history, history_mask, target_scaler):
    """Return |d raw prediction / d pre-normalization RGB pixel|."""
    model.zero_grad(set_to_none=True)
    differentiable_image = image.detach().clone().requires_grad_(True)
    prediction_scaled = model(
        differentiable_image, history, history_mask
    ).reshape(())
    prediction_raw = prediction_scaled * target_scaler.iqr + target_scaler.median
    prediction_raw.backward()
    gradient_normalized = differentiable_image.grad.detach()[0]
    std = gradient_normalized.new_tensor(IMAGENET_STD).view(3, 1, 1)
    gradient_rgb = gradient_normalized / std
    magnitude = torch.linalg.vector_norm(gradient_rgb, ord=2, dim=0)
    magnitude_raw = magnitude.cpu().numpy().astype(np.float32)
    display_scale = max(float(np.quantile(magnitude_raw, 0.99)), 1e-12)
    magnitude_display = np.clip(magnitude_raw / display_scale, 0.0, 1.0)
    model.zero_grad(set_to_none=True)
    return (
        float(prediction_raw.detach().cpu()),
        magnitude_raw,
        magnitude_display,
        display_scale,
    )


@torch.no_grad()
def _occlusion(
    model,
    image,
    history,
    history_mask,
    baseline_prediction,
    target_scaler,
    patch_size,
    stride,
    batch_size,
):
    last = IMAGE_SIZE - patch_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    coordinates = [(top, left) for top in positions for left in positions]
    deltas: list[np.ndarray] = []
    for batch_start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[batch_start : batch_start + batch_size]
        occluded = image.repeat(len(batch_coordinates), 1, 1, 1)
        for row, (top, left) in enumerate(batch_coordinates):
            # Zero in normalized space is the per-channel ImageNet mean.
            occluded[row, :, top : top + patch_size, left : left + patch_size] = 0.0
        predictions_scaled = model(
            occluded,
            history.repeat(len(batch_coordinates), 1, 1),
            history_mask.repeat(len(batch_coordinates), 1),
        ).flatten()
        predictions = predictions_scaled * target_scaler.iqr + target_scaler.median
        deltas.append((baseline_prediction - predictions).cpu().numpy())
    delta = np.concatenate(deltas)
    accumulated = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    coverage = np.zeros_like(accumulated)
    for value, (top, left) in zip(delta, coordinates):
        accumulated[top : top + patch_size, left : left + patch_size] += float(value)
        coverage[top : top + patch_size, left : left + patch_size] += 1.0
    return (accumulated / np.maximum(coverage, 1.0)).astype(np.float32)


def _input_gradient_overlay(image, gradient_display):
    heatmap = plt.get_cmap("inferno")(gradient_display)[..., :3]
    return np.clip(0.54 * image + 0.46 * heatmap, 0.0, 1.0)


def _occlusion_overlay(image, sensitivity, limit):
    scaled = np.clip(sensitivity / max(limit, 1e-12), -1.0, 1.0)
    heatmap = plt.get_cmap("coolwarm")((scaled + 1.0) / 2.0)[..., :3]
    alpha = (0.08 + 0.52 * np.abs(scaled))[..., None]
    return np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)


def _plot(results, output_path: Path, occlusion_limit: float, target: str):
    cases = sorted({int(item["case"]) for item in results})
    by_key = {(int(item["case"]), item["architecture"]): item for item in results}
    figure, axes = plt.subplots(len(cases), 5, figsize=(16.5, 3.1 * len(cases)))
    headers = (
        "Original test frame",
        "MobileNetV3-Small\nInput-gradient saliency",
        "MobileNetV3-Small\nOcclusion",
        "EfficientNet-B0\nInput-gradient saliency",
        "EfficientNet-B0\nOcclusion",
    )
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=11, fontweight="bold")
    for row, case in enumerate(cases):
        mobile = by_key[(case, "mobilenet_v3_small")]
        efficient = by_key[(case, "efficientnet_b0")]
        axes[row, 0].imshow(mobile["display"])
        axes[row, 0].set_ylabel(
            f"Case {case}\n{TARGET_LABELS[target]} {mobile['raw_value']:.1f} "
            f"{TARGET_UNITS[target]}\n"
            f"true value {mobile['y_true']:.2f}\nhistory n={mobile['history_count']}",
            fontsize=9,
        )
        axes[row, 1].imshow(
            _input_gradient_overlay(
                mobile["display"], mobile["input_gradient_display"]
            )
        )
        axes[row, 2].imshow(
            _occlusion_overlay(mobile["display"], mobile["occlusion"], occlusion_limit)
        )
        axes[row, 3].imshow(
            _input_gradient_overlay(
                efficient["display"], efficient["input_gradient_display"]
            )
        )
        axes[row, 4].imshow(
            _occlusion_overlay(
                efficient["display"], efficient["occlusion"], occlusion_limit
            )
        )
        for column, item in ((1, mobile), (2, mobile), (3, efficient), (4, efficient)):
            axes[row, column].text(
                0.02,
                0.04,
                f"frame pred {item['frame_prediction']:.2f}",
                transform=axes[row, column].transAxes,
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.62, "pad": 2, "edgecolor": "none"},
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.suptitle(
        f"{TARGET_LABELS[target]} raw-value regression: image attribution "
        "on fixed test examples",
        fontsize=15,
        fontweight="bold",
        y=0.998,
    )
    figure.text(
        0.5,
        0.008,
        "Input gradient: pixel sensitivity magnitude |d raw prediction / d RGB|.  "
        "Occlusion: red supports a higher raw prediction; blue suppresses it.",
        ha="center",
        fontsize=10,
    )
    scalar = ScalarMappable(
        norm=Normalize(vmin=-occlusion_limit, vmax=occlusion_limit), cmap="coolwarm"
    )
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=axes[:, 2::2].ravel().tolist(),
        fraction=0.012,
        pad=0.012,
        aspect=40,
    )
    colorbar.set_label(
        f"Prediction drop after occlusion ({TARGET_UNITS[target]})"
    )
    figure.subplots_adjust(left=0.09, right=0.93, top=0.955, bottom=0.035, wspace=0.025, hspace=0.07)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--target",
        choices=tuple(TARGET_LABELS),
        default=DEFAULT_TARGET,
    )
    args = parser.parse_args()
    if not 1 <= args.patch_size <= IMAGE_SIZE:
        raise ValueError("patch-size must be between 1 and the model input size")
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("stride and batch-size must be positive")

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    device = _device(args.device)
    output_root = Path(args.output_dir or OUTPUT_DIR)
    data_dir = output_root / "explainability"
    figure_dir = output_root / "figures" / "explainability"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    target = args.target
    records_path = output_root / "task_records" / f"{target}.csv"
    history_path = output_root / "history_records" / f"{target}.npz"
    index_path = Path(REFERENCE_INDEX_DIR) / "frame_offsets.npz"
    records = pd.read_csv(records_path, dtype={"hospital_id": str, "video_id": str})
    with (output_root / "target_scalers.json").open(encoding="utf-8") as handle:
        scaler_payload = json.load(handle)["targets"][target]
    target_scaler = RobustTargetScaler(**scaler_payload)
    examples = _select_examples(records, args.examples)
    frame_index = FrameOffsetIndex.load(index_path)
    history_store = HistoryFeatureStore.load(history_path)

    decoded = {}
    for row in examples.itertuples(index=False):
        decoded[str(row.video_id)] = _decode_frame(frame_index, str(row.video_id))

    results = []
    checkpoint_manifest = {}
    for architecture in ARCHITECTURES:
        checkpoint_path = (
            output_root / "runs" / architecture / target / "model.pt"
        )
        prediction_path = (
            output_root / "runs" / architecture / target / "video_predictions.csv"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model, _, _ = build_pretrained_model(architecture, WEIGHTS_DIR)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device, memory_format=torch.channels_last).eval()
        video_predictions = pd.read_csv(prediction_path, dtype={"video_id": str})
        video_prediction_lookup = video_predictions.set_index("video_id")["y_pred"].to_dict()
        checkpoint_manifest[architecture] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "selected_stage": checkpoint.get("selected_stage"),
            "gradient_target": "inverse-transformed scalar raw laboratory prediction",
            "interpolation": INTERPOLATION[architecture],
        }
        for row in examples.itertuples(index=False):
            video_id = str(row.video_id)
            source_image, global_frame_index, source_frame_index, video_path = decoded[
                video_id
            ]
            image, display = _prepare_image(source_image, architecture, device)
            history, history_mask, history_count = _history_tensors(
                history_store, video_id, device
            )
            (
                prediction,
                input_gradient_raw,
                input_gradient_display,
                input_gradient_display_scale,
            ) = _input_gradient(
                model, image, history, history_mask, target_scaler
            )
            sensitivity = _occlusion(
                model,
                image,
                history,
                history_mask,
                prediction,
                target_scaler,
                args.patch_size,
                args.stride,
                args.batch_size,
            )
            results.append(
                {
                    "case": int(row.case),
                    "architecture": architecture,
                    "hospital_id": str(row.hospital_id),
                    "video_id": video_id,
                    "video_path": video_path,
                    "global_frame_index": global_frame_index,
                    "source_frame_index": source_frame_index,
                    "raw_value": float(row.raw_value),
                    "y_true": float(row.raw_value),
                    "frame_prediction": prediction,
                    "video_prediction": float(video_prediction_lookup[video_id]),
                    "history_count": history_count,
                    "input_gradient_raw_max": float(np.max(input_gradient_raw)),
                    "input_gradient_raw_mean": float(np.mean(input_gradient_raw)),
                    "input_gradient_display_scale_p99": input_gradient_display_scale,
                    "occlusion_min": float(np.min(sensitivity)),
                    "occlusion_max": float(np.max(sensitivity)),
                    "occlusion_max_abs": float(np.max(np.abs(sensitivity))),
                    "display": display,
                    "input_gradient_raw": input_gradient_raw,
                    "input_gradient_display": input_gradient_display,
                    "occlusion": sensitivity,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    absolute_values = np.concatenate(
        [np.abs(item["occlusion"]).ravel() for item in results]
    )
    occlusion_limit = max(float(np.quantile(absolute_values, 0.99)), 1e-6)
    figure_path = figure_dir / f"{target}_input_gradient_occlusion_test_examples.png"
    _plot(results, figure_path, occlusion_limit, target)

    table_columns = [
        key
        for key in results[0]
        if key
        not in {
            "display",
            "input_gradient_raw",
            "input_gradient_display",
            "occlusion",
        }
    ]
    pd.DataFrame([{key: item[key] for key in table_columns} for item in results]).to_csv(
        data_dir / f"{target}_input_gradient_attribution_summary.csv", index=False
    )
    np.savez_compressed(
        data_dir / f"{target}_input_gradient_attribution_maps.npz",
        cases=np.asarray([item["case"] for item in results], dtype=np.int16),
        architectures=np.asarray([item["architecture"] for item in results], dtype=str),
        video_ids=np.asarray([item["video_id"] for item in results], dtype=str),
        source_frame_indices=np.asarray(
            [item["source_frame_index"] for item in results], dtype=np.int32
        ),
        input_gradient_raw=np.stack(
            [item["input_gradient_raw"] for item in results]
        ).astype(np.float32),
        input_gradient_display=np.stack(
            [item["input_gradient_display"] for item in results]
        ).astype(np.float32),
        occlusion=np.stack([item["occlusion"] for item in results]).astype(np.float32),
    )
    with (
        data_dir / f"{target}_input_gradient_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "target": target,
                "task": "robust_scaled_raw_value_regression",
                "split": "test",
                "example_selection": (
                    f"{args.examples // 2} normal-side and "
                    f"{args.examples - args.examples // 2} abnormal-side videos; nearest "
                    "unique videos to evenly spaced within-side raw-value quantiles"
                ),
                "frame_selection": "middle entry among the 20 deterministic indexed frames",
                "input_gradient": {
                    "method": "vanilla input-gradient saliency",
                    "quantity": (
                        "L2 norm across RGB channels of the gradient of the inverse-"
                        "transformed raw laboratory prediction with respect to resized "
                        "pre-normalization pixels"
                    ),
                    "sign_policy": "absolute sensitivity magnitude; gradient sign omitted",
                    "display_normalization": (
                        "divide by per-example 99th percentile and clip to [0,1]"
                    ),
                    "stored_maps": "raw magnitude and display-normalized magnitude",
                },
                "occlusion": {
                    "patch_size": [args.patch_size, args.patch_size],
                    "stride": [args.stride, args.stride],
                    "fill": "ImageNet channel mean (zero in normalized input space)",
                    "sensitivity": (
                        "baseline_raw_prediction - occluded_raw_prediction"
                    ),
                    "unit": target_scaler.unit,
                    "figure_shared_absolute_limit": occlusion_limit,
                },
                "input": {
                    "source_size": [int(decoded[next(iter(decoded))][0].shape[-2]), int(decoded[next(iter(decoded))][0].shape[-1])],
                    "model_size": [IMAGE_SIZE, IMAGE_SIZE],
                    "normalization": "ImageNet mean/std",
                    "view": "original",
                    "history": "the exact per-video prior-lab sequence used for training",
                },
                "device": str(device),
                "sources": {
                    "task_records": str(records_path.resolve()),
                    "history_features": str(history_path.resolve()),
                    "frame_index": str(index_path.resolve()),
                },
                "checkpoints": checkpoint_manifest,
                "figure": str(figure_path.resolve()),
            },
            handle,
            indent=2,
        )
    print(f"Saved figure: {figure_path}")
    print(f"Saved attribution data: {data_dir}")


if __name__ == "__main__":
    main()
