# Exp3: continuous face-video clips to a single laboratory value

Exp3 predicts one matched laboratory measurement from a patient's face video.
It is a video regression experiment, not a classification or paired-face task.
No training starts when this directory is created or when `--prepare-only` is
used.

## Cohort and label

- Eight independent tasks: O2Hb fraction, lactate, urea, total bilirubin,
  platelet count, hemoglobin, A/a PO2 ratio, and creatinine.
- Reuse the validated Exp2 face-only 20-frame task records: one video is
  matched to its nearest laboratory measurement within 24 hours, with the
  existing patient-disjoint train/validation/test split. The 20 isolated frames
  are **not** used as an R3D input.
- The target is the laboratory value in its original unit, transformed by the
  median and IQR of usable **training videos only**. Validation/test labels do
  not contribute to the scaler. Predictions are inverse-transformed for
  video-level MAE, RMSE, R2, and Pearson r.
- A source video is excluded only when it cannot yield any fully decodable
  continuous 16-frame clip. `outputs/data_summary.csv` reports exclusions.

## Video indexing and model input

- Scan each raw MJPEG AVI once; store only JPEG byte offsets for up to three
  disjoint continuous 16-frame windows centered near the early, late, and
  middle portions of the recording. Corrupt or wrong-size frames are not used.
  `cache/clip16_index/` contains a compact offset index and per-video summary,
  never decoded image files.
- Existing face frames are 128 x 128 RGB. A fixed center crop produces 112 x
  112; no temporal resampling, frame interpolation, or geometric stretching is
  applied. Normalize with Kinetics-400 R3D-18 mean/std. The input shape is
  `(batch, 3, 16, 112, 112)`.
- Each training epoch samples one of the indexed clips per video, then
  shuffles videos. A horizontal flip, if applied, is identical across all
  frames of that clip. Validation/test use all indexed clips and average their
  scaled predictions **per video** before inverse scaling and scoring. Thus a
  multi-clip video still contributes one evaluation sample.

## Architecture and optimization

The encoder is torchvision R3D-18 initialized from the official Kinetics-400
checkpoint at `study/common/pretrained_weights/r3d_18-b3b3357e.pth`. Its 400
class output is removed. Global pooled 512-dimensional features feed a
`Linear(512,32) -> LayerNorm -> SiLU -> Dropout(0.25) -> Linear(32,1)` head.
Each laboratory task has its own encoder and head.

Stage 1 freezes the encoder and trains only the head for up to 10 epochs
(`AdamW`, LR `1e-3`, cosine floor `1e-5`, patience 4). Stage 2 unfreezes the
whole network for up to 20 epochs (LR `1e-5`, floor `1e-7`, patience 6).
Both use weight decay `1e-4`, scaled SmoothL1 loss (`beta=0.5`), mixed
precision, batch size 4, and gradient clipping at 1. The best checkpoint in
each stage is selected by **validation video-level MAE**; the better stage is
used for final train/validation/test evaluation. Four GPUs run independent
tasks via a dynamic worker queue.

## Commands and outputs

Prepare and audit the complete dataset without training:

```bash
/root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp3_video_lab_regression.run_all --prepare-only
```

When ready to train later, launch in a detached screen:

```bash
bash study/exp3_video_lab_regression/launch_screen.sh
```

After all eight tasks succeed, the runner writes checkpoints, per-epoch
histories, video-level predictions, a held-out face-only Exp2 comparison on
the **same videos**, and result figures under `outputs/`. The official R3D
weights and derived indexes are local prerequisites and are ignored by Git.

The pretrained weight metadata and expected input conventions are documented
in [torchvision's R3D-18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html).

## Scope of the inference

The video clip is not synchronized to the blood draw. The existing video to
laboratory match is within 24 hours; a model may learn appearance correlates
of state, treatment, patient identity, or recording conditions rather than a
direct physiological measurement. Patient-disjoint testing prevents identity
leakage across splits, but it does not remove all such confounding.
