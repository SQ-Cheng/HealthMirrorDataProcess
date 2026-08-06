# Exp2 Biosensors 2025 Residual 3D CNN Reproduction

This experiment reproduces the best residual video-regression architecture from
Bal et al., *Estimation of Total Hemoglobin (SpHb) from Facial Videos Using 3D
Convolutional Neural Network-Based Regression*, Biosensors 2025, 15, 485,
DOI: 10.3390/bios15080485.

## Controlled data

- The target is raw Hemoglobin converted from g/L to g/dL.
- `task_records.csv` exactly reuses all 906 videos and the patient-disjoint
  train/validation/test assignment from
  `exp2_face_pretrained_head32_regression/outputs/20frame`.
- Each model input spans the complete source video with 224 deterministic,
  uniformly positioned frames. No temporal interpolation is used. For the 67
  videos with fewer than 224 decodable frames, nearest positions are repeated;
  this is recorded per video and preserves the exact split and sample set.
- Frames are streamed from the existing all-frame JPEG byte-offset index. No
  decoded video cache is written. The entry point validates and reuses that
  index, or rebuilds the compact byte offsets automatically if it is absent or
  stale.

## Paper model

```text
Video (3,224,224,224)
  -> Conv3D(3->64, kernel=7, stride=2, padding=3) + BN + ReLU
  -> ResidualBlock3D(64->64, stride=1)
  -> ResidualBlock3D(64->64, stride=1)
  -> ResidualBlock3D(64->128, stride=2, projection shortcut)
  -> ResidualBlock3D(128->128, stride=1)
  -> ResidualBlock3D(128->256, stride=2, projection shortcut)
  -> ResidualBlock3D(256->256, stride=1)
  -> AdaptiveAvgPool3D(1)
  -> Linear(256->1)
  -> Hemoglobin (g/dL)
```

Every residual block contains two `3x3x3` convolutions with batch normalization
and ReLU. The model is trained from scratch, as in the paper.

## Training protocol

- MSE loss in g/dL, Adam at `1e-3`, up to 100 epochs.
- Effective batch size 4. Formal training uses four-process distributed data
  parallelism with one complete video per GPU, so the global batch is exactly 4.
- The paper's early-stop threshold `validation MSE < 0.3` is retained.
- No augmentation or learning-rate schedule is introduced because neither is
  specified for the residual model.

The machine-readable manifest explicitly records paper ambiguities and all local
adaptations. In particular, the paper inconsistently mentions both 224 and 30
frames, and both 100 and 50 epochs; this implementation follows its methods and
Table 2: 224 frames and 100 epochs.

## Commands

Prepare and validate without training:

```bash
python -m study.exp2_video_3dresnet_sphb_reproduction.run_all --prepare-only
```

Formal detached training entry point (do not run until approved):

```bash
bash study/exp2_video_3dresnet_sphb_reproduction/launch_screen.sh
```
