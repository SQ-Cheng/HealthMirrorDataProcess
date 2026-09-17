"""Generate paired-face Grad-CAM and occlusion visualizations for Exp5."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from study.exp4.frame_index import FrameOffsetIndex

from .config import CACHE_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, OUTPUT_DIR
from .data import PairedFrameDataset
from .models import build_model


DEFAULT_EXAMPLES = 10
# These replace quantile-selected examples whose source face ROI is truncated.
DEFAULT_REPLACEMENT_PATIENTS = {
    2: "61396556",
    3: "61190454",
    10: "61492258",
}
OCCLUSION_PATCH_SIZE = 32
OCCLUSION_STRIDE = 16
OCCLUSION_BATCH_SIZE = 64


def _select_examples(predictions, count):
    test = predictions[predictions.split.eq("test")].copy()
    if test.hospital_id.nunique() < count:
        raise RuntimeError(f"Only {test.hospital_id.nunique()} distinct test patients")
    targets = np.quantile(test.y_true, np.linspace(0.05, 0.95, count))
    selected, used_patients = [], set()
    for target in targets:
        ranked = test.assign(distance=(test.y_true - target).abs()).sort_values(
            ["distance", "y_true", "video_id"], kind="stable"
        )
        choice = next(
            row for row in ranked.itertuples(index=False)
            if str(row.hospital_id) not in used_patients
        )
        selected.append(choice._asdict())
        used_patients.add(str(choice.hospital_id))
    result = pd.DataFrame(selected).reset_index(drop=True)
    if count == DEFAULT_EXAMPLES:
        for example_number, hospital_id in DEFAULT_REPLACEMENT_PATIENTS.items():
            candidates = test[test.hospital_id.astype(str).eq(hospital_id)].sort_values(
                ["y_true", "video_id"], kind="stable"
            )
            if candidates.empty:
                raise RuntimeError(f"Replacement patient {hospital_id} is unavailable")
            replacement = candidates.iloc[0]
            row = example_number - 1
            for column in result.columns:
                if column in replacement.index:
                    result.at[row, column] = replacement[column]
            result.at[row, "distance"] = abs(
                float(replacement.y_true) - float(targets[row])
            )
    if result.hospital_id.nunique() != count:
        raise AssertionError("Interpretability examples are not distinct patients")
    return result


def _preprocess(images, device):
    images = images.to(device).float().div_(255.0)
    images = F.interpolate(
        images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
        align_corners=False, antialias=True,
    )
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((images - mean) / std).contiguous(memory_format=torch.channels_last)


def _display_image(image):
    resized = F.interpolate(
        image[None].float().div(255.0), size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic", align_corners=False, antialias=True,
    )[0]
    return resized.permute(1, 2, 0).clamp(0, 1).numpy()


def _predict_from_pre_embedding(model, pre_embedding, post_images):
    post_embedding = model.encode_post(post_images)
    pre = pre_embedding.expand(len(post_embedding), -1)
    fused = torch.cat(
        (pre, post_embedding, post_embedding - pre, torch.abs(post_embedding - pre)),
        dim=1,
    )
    return model.head(fused)


@torch.no_grad()
def _representative_pair(model, dataset, video_row, device):
    frame_rows = np.flatnonzero(dataset.frame_video_rows == video_row)
    pairs = [dataset[int(frame_row)] for frame_row in frame_rows]
    raw_pre = torch.stack([pair[0] for pair in pairs])
    raw_post = torch.stack([pair[1] for pair in pairs])
    pre = _preprocess(raw_pre, device)
    post = _preprocess(raw_post, device)
    predictions = model(pre, post).squeeze(1).float().cpu().numpy()
    video_prediction = float(predictions.mean())
    position = int(np.argmin(np.abs(predictions - video_prediction)))
    return {
        "raw_pre": raw_pre[position],
        "raw_post": raw_post[position],
        "pre": pre[position:position + 1],
        "post": post[position:position + 1],
        "frame_prediction": float(predictions[position]),
        "video_prediction": video_prediction,
        "frame_row": int(frame_rows[position]),
        "position": position,
    }


def _grad_cam(model, pre_image, post_image, target_layer):
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__("gradient", gradient)
        )

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        pre_embedding = model.encode_pre(pre_image).detach()
    handle = target_layer.register_forward_hook(hook)
    try:
        prediction = _predict_from_pre_embedding(
            model, pre_embedding, post_image
        ).squeeze()
        prediction.backward()
        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.abs((channel_weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        if float(cam.max()) > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), float(prediction.detach())
    finally:
        handle.remove()


@torch.no_grad()
def _occlusion_sensitivity(model, pre_image, post_image):
    pre_embedding = model.encode_pre(pre_image)
    baseline = float(
        _predict_from_pre_embedding(model, pre_embedding, post_image).squeeze()
    )
    height, width = post_image.shape[-2:]
    tops = list(range(0, height - OCCLUSION_PATCH_SIZE + 1, OCCLUSION_STRIDE))
    lefts = list(range(0, width - OCCLUSION_PATCH_SIZE + 1, OCCLUSION_STRIDE))
    if tops[-1] != height - OCCLUSION_PATCH_SIZE:
        tops.append(height - OCCLUSION_PATCH_SIZE)
    if lefts[-1] != width - OCCLUSION_PATCH_SIZE:
        lefts.append(width - OCCLUSION_PATCH_SIZE)
    positions = [(top, left) for top in tops for left in lefts]
    sensitivity = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for start in range(0, len(positions), OCCLUSION_BATCH_SIZE):
        batch_positions = positions[start:start + OCCLUSION_BATCH_SIZE]
        occluded = post_image.repeat(len(batch_positions), 1, 1, 1)
        for row, (top, left) in enumerate(batch_positions):
            occluded[
                row, :, top:top + OCCLUSION_PATCH_SIZE,
                left:left + OCCLUSION_PATCH_SIZE,
            ] = 0.0
        values = _predict_from_pre_embedding(
            model, pre_embedding, occluded
        ).squeeze(1).float().cpu().numpy()
        for (top, left), value in zip(batch_positions, values):
            delta = baseline - float(value)
            sensitivity[
                top:top + OCCLUSION_PATCH_SIZE,
                left:left + OCCLUSION_PATCH_SIZE,
            ] += delta
            coverage[
                top:top + OCCLUSION_PATCH_SIZE,
                left:left + OCCLUSION_PATCH_SIZE,
            ] += 1
    sensitivity /= np.maximum(coverage, 1)
    return sensitivity, baseline, len(positions)


def _draw(examples, pre_images, post_images, gradcams, occlusions, output_path):
    figure, axes = plt.subplots(
        len(examples), 4, figsize=(12.5, 2.8 * len(examples)), squeeze=False
    )
    column_titles = (
        "Preoperative reference", "Postoperative face",
        "Postoperative Grad-CAM", "Postoperative occlusion",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, pad=8)
    for row, example in examples.iterrows():
        axes[row, 0].imshow(pre_images[row])
        axes[row, 1].imshow(post_images[row])
        axes[row, 2].imshow(post_images[row])
        axes[row, 2].imshow(
            gradcams[row], cmap="jet", alpha=0.50, vmin=0, vmax=1
        )
        scale = max(float(np.abs(occlusions[row]).max()), 1e-8)
        axes[row, 3].imshow(post_images[row])
        axes[row, 3].imshow(
            occlusions[row], cmap="coolwarm", alpha=0.58,
            vmin=-scale, vmax=scale,
        )
        axes[row, 0].set_ylabel(
            f"Face {row + 1}\ntrue {example.y_true:.3f}\npred {example.video_prediction:.3f}",
            fontsize=9,
        )
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.suptitle(
        "Exp5 paired-face recovery: attention on the postoperative face\n"
        "Occlusion red supports a higher score; blue supports a lower score",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.65, w_pad=0.4)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "model.pt"
    predictions_path = output_dir / "video_predictions.csv"
    records_path = output_dir / "records.csv"
    frame_index_path = CACHE_DIR / "frame_offsets.npz"
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
    if selected_records.recovery_score.isna().any():
        raise RuntimeError("Selected test examples are absent from records.csv")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)
    model, _ = build_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device, memory_format=torch.channels_last).eval()
    frame_index = FrameOffsetIndex.load(frame_index_path)
    dataset = PairedFrameDataset(frame_index, selected_records)
    target_layer = model.post_backbone.features[-1][0]

    pre_images, post_images, gradcams, occlusions, rows = [], [], [], [], []
    for video_row, example in selected.iterrows():
        pair = _representative_pair(model, dataset, video_row, device)
        gradcam, gradcam_prediction = _grad_cam(
            model, pair["pre"], pair["post"], target_layer
        )
        occlusion, occlusion_prediction, positions = _occlusion_sensitivity(
            model, pair["pre"], pair["post"]
        )
        if not np.isclose(pair["frame_prediction"], gradcam_prediction, atol=1e-5):
            raise AssertionError("Grad-CAM changed the representative prediction")
        if not np.isclose(pair["frame_prediction"], occlusion_prediction, atol=1e-5):
            raise AssertionError("Occlusion baseline changed the prediction")
        global_post_index = int(dataset.post_indices[pair["frame_row"]])
        pre_images.append(_display_image(pair["raw_pre"]))
        post_images.append(_display_image(pair["raw_post"]))
        gradcams.append(gradcam)
        occlusions.append(occlusion)
        rows.append({
            "example": video_row + 1,
            "hospital_id": str(example.hospital_id),
            "pre_video_id": str(example.pre_video_id),
            "post_video_id": str(example.video_id),
            "split": str(example.split),
            "y_true": float(example.y_true),
            "saved_video_prediction": float(example.y_pred),
            "video_prediction": pair["video_prediction"],
            "representative_frame_prediction": pair["frame_prediction"],
            "representative_pair_position": pair["position"],
            "post_source_frame_index": int(
                frame_index.source_indices[global_post_index]
            ),
            "gradcam_target_layer": "post_backbone.features[-1][0]",
            "occlusion_patch_size": OCCLUSION_PATCH_SIZE,
            "occlusion_stride": OCCLUSION_STRIDE,
            "occlusion_positions": positions,
            "gradcam_max": float(np.max(gradcam)),
            "occlusion_min": float(np.min(occlusion)),
            "occlusion_max": float(np.max(occlusion)),
        })
        print(
            f"[example] {video_row + 1}/{len(selected)} "
            f"true={example.y_true:.3f} video_pred={pair['video_prediction']:.3f} "
            f"frame_pred={pair['frame_prediction']:.3f}",
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
        occlusion=np.stack(occlusions),
    )
    output_path = figures_dir / "interpretability_gradcam_occlusion_10faces.png"
    _draw(result, pre_images, post_images, gradcams, occlusions, output_path)
    print(f"[complete] examples={len(result)} figure={output_path}", flush=True)


if __name__ == "__main__":
    main()
