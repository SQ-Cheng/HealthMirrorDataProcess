# Exp2 Binary 20 Frames: Head-64, Aligned Split, Five Views

Canonical ID: `exp2_face_pretrained_head64`.

This experiment is the binary counterpart of the Head64 single-task abnormal-score
regression experiment. It uses the same corrected source records, closest-event
selection, patient-level distribution-balanced split, exact video records, exact
20-frame index, model input, five views, seed, and two-stage optimization settings.

This isolated experiment uses MobileNetV3-Small and EfficientNet-B0 with an independent
64-wide binary head for each of these targets:

- `hemoglobin_low`
- `po2_low`
- `lactate_high`

Twenty deterministic non-adjacent RGB frames from each matched raw MJPEG video are
used. Hemoglobin and PO2 reuse the exact task records and splits from
`exp2_face_pretrained_head32_regression/outputs/20frame`. Lactate records are filtered
against that same 20-frame index and assigned with the same distribution-balanced
patient split algorithm and seed. Runtime validation rejects any video without exactly
20 indexed frames or any patient assigned to multiple splits. Each training frame has
five deterministic views; validation and test use the original view.

The only intentional task-level differences from regression are the binary target,
`BCEWithLogitsLoss` with training-frame-derived `pos_weight`, video-level probability
aggregation, and ROC-AUC-based model selection.

After all jobs finish successfully, `run_all.py` automatically validates the result
set and writes the result figures and plotting tables to `outputs/figures`. A failed
job stops the run before plotting, so an incomplete experiment is not presented as a
complete result.

## Storage and I/O

Decoded frames are never written to disk. The regression experiment's compact frame
index stores only JPEG byte offsets. DataLoader workers keep bounded file-handle and
decoded-frame LRU
caches, seek directly to JPEG payloads, and decode them with the pixel-equivalent
`torchvision.io.decode_jpeg` CPU path. Training shuffles 256-frame contiguous
chunks rather than individual frames and decodes each selected frame once. The raw
RGB tensor crosses the worker queue and PCIe bus once;
it is then expanded into all five views on the GPU. This preserves the exact five-view
sample set while eliminating fivefold duplicate transfer. View transforms, 224x224
resizing, ImageNet normalization, and channels-last conversion run on the assigned
GPU. Per-frame predictions use compressed numeric NPZ instead of CSV.

Training uses one persistent training slot per visible GPU. Each completed slot
immediately takes the next architecture/target job from the dynamic queue. A job uses
six persistent training decoders and two decoders per evaluation split. The measured
source/effective training batches are 128/640 for MobileNetV3-Small and 48/240 for
EfficientNet-B0; evaluation batches are 1024 and 512 respectively.
Training execution uses `torch.compile(mode="reduce-overhead")`; optimizer state,
early-stopping snapshots, and checkpoints always use the unwrapped model state dict.

## Start

```bash
bash study/exp2_face_pretrained_head64/launch_screen.sh --overwrite
```

```bash
screen -r exp2_face_pretrained_head64
```
