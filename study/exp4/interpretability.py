"""Generate standard Grad-CAM and occlusion maps for Exp4 test examples."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import (
    CACHE_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUTPUT_DIR,
)
from .data import RecoveryFrameDataset
from .frame_index import FrameOffsetIndex
from .models import build_model


DEFAULT_EXAMPLES = 5
OCCLUSION_PATCH_SIZE = 32
OCCLUSION_STRIDE = 16
OCCLUSION_BATCH_SIZE = 64


def _best_seed(output_dir):
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    test = metrics.loc[metrics["split"].eq("test")].copy()
    if test.empty or test["pearson_r"].isna().all():
        raise RuntimeError("No finite test Pearson correlation is available")
    return int(test.loc[test["pearson_r"].idxmax(), "seed"])


def _select_examples(predictions, count):
    test = predictions.loc[predictions["split"].eq("test")].copy()
    test = test.sort_values(["recovery_score", "hospital_id", "video_id"])
    targets = np.linspace(0.1, 0.9, count)
    selected, used_patients, used_videos = [], set(), set()
    for target in targets:
        ranked = test.assign(
            distance=(test["recovery_score"] - target).abs()
        ).sort_values(["distance", "recovery_score", "video_id"])
        choice = None
        for row in ranked.itertuples(index=False):
            if row.hospital_id in used_patients or row.video_id in used_videos:
                continue
            choice = row
            break
        if choice is None:
            raise RuntimeError(f"Could not select {count} distinct-patient examples")
        selected.append(choice._asdict())
        used_patients.add(choice.hospital_id)
        used_videos.add(choice.video_id)
    return pd.DataFrame(selected).reset_index(drop=True)


def _preprocess(images):
    images = images.float().div(255.0)
    images = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _display_image(image):
    resized = F.interpolate(
        image[None].float().div(255.0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]
    return resized.permute(1, 2, 0).clamp(0, 1).numpy()


def _grad_cam(model, normalized_image, target_layer):
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        prediction = model(normalized_image).squeeze()
        prediction.backward()
        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        cam -= cam.min()
        if float(cam.max()) > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), float(prediction.detach())
    finally:
        handle.remove()


@torch.no_grad()
def _occlusion_sensitivity(
    model,
    normalized_image,
    patch_size=OCCLUSION_PATCH_SIZE,
    stride=OCCLUSION_STRIDE,
    batch_size=OCCLUSION_BATCH_SIZE,
):
    baseline = float(model(normalized_image).squeeze())
    height, width = normalized_image.shape[-2:]
    tops = list(range(0, height - patch_size + 1, stride))
    lefts = list(range(0, width - patch_size + 1, stride))
    if tops[-1] != height - patch_size:
        tops.append(height - patch_size)
    if lefts[-1] != width - patch_size:
        lefts.append(width - patch_size)
    positions = [(top, left) for top in tops for left in lefts]
    sensitivity = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for batch_start in range(0, len(positions), batch_size):
        batch_positions = positions[batch_start:batch_start + batch_size]
        occluded = normalized_image.repeat(len(batch_positions), 1, 1, 1)
        for batch_row, (top, left) in enumerate(batch_positions):
            occluded[
                batch_row, :, top:top + patch_size, left:left + patch_size
            ] = 0.0
        predictions = model(occluded).squeeze(1).cpu().numpy()
        for (top, left), prediction in zip(batch_positions, predictions):
            delta = baseline - float(prediction)
            sensitivity[top:top + patch_size, left:left + patch_size] += delta
            coverage[top:top + patch_size, left:left + patch_size] += 1.0
    sensitivity /= np.maximum(coverage, 1.0)
    return sensitivity, baseline, len(positions)


@torch.no_grad()
def _representative_frame(model, dataset, video_row):
    frame_rows = np.flatnonzero(dataset.frame_video_rows == video_row)
    images = torch.stack([dataset[int(frame_row)][0] for frame_row in frame_rows])
    normalized = _preprocess(images)
    predictions = model(normalized).squeeze(1).cpu().numpy()
    video_prediction = float(predictions.mean())
    representative = int(np.argmin(np.abs(predictions - video_prediction)))
    return (
        images[representative],
        normalized[representative:representative + 1],
        float(predictions[representative]),
        video_prediction,
        int(frame_rows[representative]),
    )


def _draw_combined(examples, images, gradcams, occlusions, output_path):
    figure, axes = plt.subplots(len(examples), 3, figsize=(11.5, 3.25 * len(examples)))
    for row, example in examples.iterrows():
        title = (
            f"Example {row + 1} | true={example.recovery_score:.3f} | "
            f"video pred={example.recomputed_video_prediction:.3f}"
        )
        axes[row, 0].imshow(images[row]); axes[row, 0].set_title(title)
        axes[row, 1].imshow(images[row])
        axes[row, 1].imshow(gradcams[row], cmap="jet", alpha=0.50, vmin=0, vmax=1)
        axes[row, 1].set_title("Standard Grad-CAM")
        scale = max(float(np.abs(occlusions[row]).max()), 1e-8)
        axes[row, 2].imshow(images[row])
        axes[row, 2].imshow(
            occlusions[row], cmap="coolwarm", alpha=0.55, vmin=-scale, vmax=scale
        )
        axes[row, 2].set_title("Signed occlusion sensitivity")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(
        "Exp4 postoperative recovery interpretability\n"
        "Occlusion: red supports higher output; blue supports lower output",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _draw_method(examples, images, maps, method, output_path):
    figure, axes = plt.subplots(len(examples), 2, figsize=(8.2, 3.25 * len(examples)))
    for row, example in examples.iterrows():
        axes[row, 0].imshow(images[row])
        axes[row, 0].set_title(
            f"Example {row + 1} | true={example.recovery_score:.3f} | "
            f"pred={example.recomputed_video_prediction:.3f}"
        )
        axes[row, 1].imshow(images[row])
        if method == "gradcam":
            axes[row, 1].imshow(maps[row], cmap="jet", alpha=0.50, vmin=0, vmax=1)
            axes[row, 1].set_title("Standard Grad-CAM")
        else:
            scale = max(float(np.abs(maps[row]).max()), 1e-8)
            axes[row, 1].imshow(
                maps[row], cmap="coolwarm", alpha=0.55, vmin=-scale, vmax=scale
            )
            axes[row, 1].set_title("Signed occlusion: red higher / blue lower")
        axes[row, 0].axis("off"); axes[row, 1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    args = parser.parse_args()

    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    output_dir = args.output_dir.resolve()
    seed = args.seed if args.seed is not None else _best_seed(output_dir)
    run_dir = output_dir
    checkpoint_path = run_dir / "model.pt"
    predictions_path = run_dir / "video_predictions.csv"
    records_path = output_dir / "records.csv"
    frame_index_path = CACHE_DIR / "frames20" / "frame_offsets.npz"
    for path in (checkpoint_path, predictions_path, records_path, frame_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(
        predictions_path, dtype={"hospital_id": str, "video_id": str}
    )
    selected = _select_examples(predictions, args.examples)
    records = pd.read_csv(
        records_path, dtype={"hospital_id": str, "video_id": str}
    )
    selected_records = selected[["hospital_id", "video_id"]].merge(
        records, on=["hospital_id", "video_id"], how="left", validate="one_to_one"
    )
    if selected_records["recovery_score"].isna().any():
        raise RuntimeError("Selected test videos are missing from Exp4 records")

    model, _, _ = build_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    frame_index = FrameOffsetIndex.load(frame_index_path)
    dataset = RecoveryFrameDataset(frame_index, selected_records)
    target_layer = model.features[-1][0]

    display_images, gradcams, occlusions, rows = [], [], [], []
    for video_row, example in selected.iterrows():
        raw, normalized, frame_prediction, video_prediction, frame_row = (
            _representative_frame(model, dataset, video_row)
        )
        gradcam, gradcam_prediction = _grad_cam(model, normalized, target_layer)
        occlusion, occlusion_prediction, position_count = _occlusion_sensitivity(
            model, normalized
        )
        if not np.isclose(frame_prediction, gradcam_prediction, atol=1e-6):
            raise AssertionError("Grad-CAM forward prediction changed")
        if not np.isclose(frame_prediction, occlusion_prediction, atol=1e-6):
            raise AssertionError("Occlusion baseline prediction changed")
        global_index = int(dataset.frame_indices[frame_row])
        display_images.append(_display_image(raw))
        gradcams.append(gradcam)
        occlusions.append(occlusion)
        rows.append({
            "example": video_row + 1,
            "seed": seed,
            "hospital_id": example.hospital_id,
            "video_id": example.video_id,
            "split": example.split,
            "recovery_score": float(example.recovery_score),
            "saved_video_prediction": float(example.y_pred),
            "recomputed_video_prediction": video_prediction,
            "representative_frame_prediction": frame_prediction,
            "representative_frame_position_within_selected_20": int(frame_row % 20),
            "representative_source_frame_index": int(
                frame_index.source_indices[global_index]
            ),
            "gradcam_target_layer": "features[-1][0]",
            "occlusion_patch_size": OCCLUSION_PATCH_SIZE,
            "occlusion_stride": OCCLUSION_STRIDE,
            "occlusion_positions": position_count,
        })
        print(
            f"[example] {video_row + 1}/{len(selected)} "
            f"true={example.recovery_score:.3f} video_pred={video_prediction:.3f} "
            f"frame_pred={frame_prediction:.3f}",
            flush=True,
        )
    dataset.close()

    result = pd.DataFrame(rows)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "interpretability_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "interpretability_maps.npz",
        gradcam=np.stack(gradcams),
        occlusion_sensitivity=np.stack(occlusions),
    )
    manifest = {
        "schema_version": 1,
        "model_seed": seed,
        "checkpoint": str(checkpoint_path),
        "selection": "distinct test patients nearest recovery-score quantiles 0.1 to 0.9",
        "representative_frame": "frame prediction nearest the mean prediction across the video's 20 frames",
        "gradcam": {
            "method": "standard Grad-CAM",
            "target_layer": "EfficientNet-B0 final Conv2d: features[-1][0]",
            "target": "scalar postoperative recovery output",
            "channel_weights": "global spatial mean of output gradients",
            "combination": "ReLU(sum(channel_weight * activation))",
        },
        "occlusion": {
            "patch_size": OCCLUSION_PATCH_SIZE,
            "stride": OCCLUSION_STRIDE,
            "fill": "ImageNet channel mean (zero after normalization)",
            "sensitivity": "baseline recovery output - occluded recovery output",
            "positive_interpretation": "region supports a higher recovery output",
        },
    }
    (output_dir / "interpretability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _draw_combined(
        result, display_images, gradcams, occlusions,
        figures_dir / "interpretability_gradcam_occlusion.png",
    )
    _draw_method(
        result, display_images, gradcams, "gradcam",
        figures_dir / "interpretability_gradcam.png",
    )
    _draw_method(
        result, display_images, occlusions, "occlusion",
        figures_dir / "interpretability_occlusion.png",
    )
    print(f"[complete] figures={figures_dir} examples={len(result)} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
