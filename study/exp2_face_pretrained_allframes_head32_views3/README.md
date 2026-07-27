# Exp2 Pretrained All Frames: Head-32, Three Views

Canonical ID: `exp2_binary_allframes_head32_views3`.

This isolated experiment uses MobileNetV3-Small and EfficientNet-B0 with an independent
32-wide binary head for each of these targets:

- `hemoglobin_low`
- `pco2_low`
- `po2_low`
- `high_blood_pressure`
- `lactate_high`

Every decodable RGB frame from each matched MJPEG video is used. Each training frame
has exactly three deterministic views: original, horizontal flip, and 90% center crop.
Brightness and contrast views are disabled. Validation and test use only the original
view. All model, split, optimization, early-stopping, scheduling, and I/O settings
otherwise match `../exp2_face_pretrained_allframes_head32`.

## Storage and I/O

Decoded frames are never written to disk. `frame_index/frame_offsets.npz` stores only
JPEG byte offsets. DataLoader workers keep bounded file-handle and decoded-frame LRU
caches, seek directly to JPEG payloads, and decode them with the pixel-equivalent
`torchvision.io.decode_jpeg` CPU path. Training shuffles 256-frame contiguous
chunks rather than individual frames, then reads each chunk sequentially and decodes
each source frame once. The raw RGB tensor crosses the worker queue and PCIe bus once;
it is then expanded into all three views on the GPU. View transforms, 224x224
resizing, ImageNet normalization, and channels-last conversion run on the assigned
GPU. Per-frame predictions use compressed numeric NPZ instead of CSV.

Training uses one persistent training slot per visible GPU. Each completed slot
immediately takes the next architecture/target job from the dynamic queue. A job uses
six persistent training decoders and two decoders per evaluation split. The measured
source/effective training batches are 128/384 for MobileNetV3-Small and 48/144 for
EfficientNet-B0; evaluation batches are 1024 and 512 respectively.
Training execution uses `torch.compile(mode="reduce-overhead")`; optimizer state,
early-stopping snapshots, and checkpoints always use the unwrapped model state dict.

## Start

```bash
bash study/exp2_face_pretrained_allframes_head32_views3/launch_screen.sh
```

```bash
screen -r exp2_face_pretrained_allframes_head32_views3
```

## Result Figures

Regenerate the video-level result figures with:

```bash
python study/exp2_face_pretrained_allframes_head32_views3/plot_results.py
```

The script writes training curves, test metric bars, split ROC-AUC comparisons,
and test ROC/precision-recall curves to `outputs/figures/`.
