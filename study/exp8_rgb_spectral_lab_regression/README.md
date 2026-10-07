# Exp8: RGB-Estimated Spectra to Laboratory Values

## Current Protocol

Only native 224x224 FFV1 face crops and a 12-hour nearest-lab window are
supported. The window is distance from the original video capture interval,
not the duration of filtered face crops. Clinical matching remains scoped
to the same hospitalization, with canonical Session Timestamp validation.

The three independent tasks are hemoglobin, total bilirubin, and lactate.
Records, patient-disjoint splits, train-only median/IQR scalers, and paired
RGB control predictions come from:

`../exp2_face_pretrained_head32_regression/outputs/ablations/lab_match_12h_face224/`

Each video contributes 20 deterministic nonadjacent frames from the shared
native-224 index. Exp8 averages their spectral features into one video-level
input; its existing protocol does not expand frames into five augmented views.

## Model And Training

The frozen official MST++ NTIRE 2022 checkpoint estimates 31 visible bands
(400-700 nm in 10-nm steps). Each RGB frame uses per-frame min-max normalization,
not ImageNet normalization. Estimated values are clipped to [0,1]. The central
168x168 region of the 224x224 cube is spatially averaged into 31x4x4 or 31x16x16
features. Full spectral cubes are not saved.

The per-task head is Flatten -> LayerNorm -> Linear(N,64) -> SiLU ->
Dropout(0.2) -> Linear(64,32) -> SiLU -> Linear(32,1), with N=496 or 7936.
It predicts robust-scaled raw values using SmoothL1(beta=0.5), AdamW,
lr=1e-3, weight decay=1e-2, batch size=32, cosine decay to 1e-5, up to
160 epochs, and patience=20 on raw-unit validation MAE. Seeds are 42,43,44
in task order. No model or training hyperparameters changed with the window.

This is estimated spectral response, not calibrated hyperspectral measurement.
There is no local paired RGB/spectral ground truth. The currently available
checkpoint is NTIRE 2022, not a Hyper-Skin-retrained checkpoint. The verified
weight is `../common/pretrained_weights/mst_plus_plus_ntire2022.pth`;
its provenance is recorded in `weights_manifest.json`.

## Outputs And Cache Reuse

- 4x4: `outputs_face224/`, `logs_face224/`, `cache_face224/`.
- 16x16: `outputs_grid16_face224/`, `logs_grid16_face224/`, `cache_grid16_face224/`.

Old 24h and 128 results/logs, and 128 feature caches, have been removed.
Native-224 feature caches are label-independent and reusable for 12h;
checkpoint/index hashes, dimensions, and finite values are checked before reuse.
The cache includes index videos outside the 12h cohort, but only the saved
12h task records enter training or evaluation.

Each run saves checkpoints, train/validation histories, per-video predictions,
MAE/RMSE/R2/Pearson, and a machine-readable experiment manifest. Figures are
generated automatically, including paired 12h RGB-control comparisons and,
after the 16x16 run, the 4x4-versus-16x16 comparison. The RGB control uses a
different training unit and augmentation policy, so this is a method comparison.

## Launch

```bash
# CPU-only input/cache/paired-control validation, without training:
EXP8_GRID_SIZE=4 python -m study.exp8_rgb_spectral_lab_regression.train --check-only
EXP8_GRID_SIZE=16 python -m study.exp8_rgb_spectral_lab_regression.train --check-only

# Wait for the current 224/12h five-fold queue, then run both grids sequentially:
bash study/exp8_rgb_spectral_lab_regression/launch_after_current_screen.sh
screen -r exp8_face224_12h_autostart

# Direct launch of a single grid when GPUs are available:
bash study/exp8_rgb_spectral_lab_regression/launch_screen.sh ntire2022 4
```

The waiting screen consumes no GPU. An interrupted predecessor is reported
as an error rather than silently launching concurrent jobs. Completed 12h
grid runs are skipped when restarting the monitor; partial runs are retrained.
