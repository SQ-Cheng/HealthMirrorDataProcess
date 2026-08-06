# Exp2 Face + Prior Lab History: Head-32 Binary Classification

This experiment is the history-aware counterpart of
`study/exp2_face_pretrained_head32`.

It trains independent MobileNetV3-Small and EfficientNet-B0 models for:

- `hemoglobin_low`
- `po2_low`

The image path, 20 deterministic non-adjacent frames, five training views,
ImageNet preprocessing, Head-32 width, binary `BCEWithLogitsLoss`, frame-count
`pos_weight`, two-stage learning rates, epoch limits, early stopping, seed, video
set, and patient split are unchanged from the aligned binary baseline.

Each model additionally receives all same-analyte measurements from the same
admission episode that are strictly earlier than the current label. A small
per-measurement MLP encodes abnormal-distance and elapsed-time features, followed
by masked mean pooling. The 16-dimensional history representation is concatenated
with the image backbone output before the unchanged 32-dimensional classification
head. Missing history maps to a zero pooled vector.

PO2 uses only the exact item name `氧分压`. Patient-temperature-corrected PO2
is excluded from labels and history construction.

The GPU scheduler admits devices with at most 1024 MiB in use. It starts on GPUs
that are currently idle and polls busy GPUs every 15 seconds, adding them when the
preceding experiment releases them. After all four jobs complete successfully,
six result figures and the dataset-balance table are generated automatically.

## Start

```bash
bash study/exp2_face_history_head32/launch_screen.sh --overwrite
```

```bash
screen -r exp2_face_history_head32
```
