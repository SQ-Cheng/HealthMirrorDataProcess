# Native224 RGB + IR Face Regression

Independent four-channel EfficientNet-B0/head32 model for each of the current
ten raw laboratory regression targets. The existing RGB experiment is untouched.

## Model

`[R,G,B,IR](4,224,224) -> ImageNet EfficientNet-B0 -> head32 -> scalar`.
The stem becomes `Conv2d(4,32,3,stride=2,padding=1)`: pretrained RGB filters
are copied exactly and the fourth-channel filters are initialized to zero.
All other pretrained parameters and the original head design are unchanged.
This adds 288 parameters (total 4,048,925; head 41,089).

RGB uses ImageNet normalization. The single genuine grayscale IR channel uses
fixed mean .5/std .5, with no dataset-fitted or held-out intensity statistics.
Both modalities receive the same original/hflip/center-crop/brightness/contrast
view; contrast means are computed independently per channel, as in the RGB main
experiment. Independent approved face crops are reused without adding spatial
registration, alignment, enhancement or false color.

## Data And Training

- Reuse current ten-task main labels, original per-target patient splits and
  actual job seeds. Source hashes and the merged lab table are validated.
- Keep only videos with twenty verified joint frames. Labels and nearest24h
  matching are not recomputed. Raw-frame gaps >=2 and mutual-nearest matching
  within 33.333 ms remain unchanged. Full cohort: 1,311 videos / 549 patients.
- Refit target median/IQR using surviving training videos only, following the
  primary scaling algorithm. Clinical thresholds remain auxiliary metrics.
- Native FFV1 videos and selected20 mappings are reused. Two small packet-offset
  indexes are built; no decoded-frame or duplicated-video cache is created.
- Batch: 12 distinct clinical events, 20 frames, one view per video = 240
  four-channel inputs and frame-level SmoothL1 losses (beta .5). All five views
  are covered per epoch; train/val/test reporting uses original-view video means.
- Stage one: backbone frozen, head-only AdamW lr `2e-4`, max 40 epochs,
  patience 10. The frozen zero IR stem cannot contribute during this stage.
- Stage two: all parameters unfrozen, including IR filters, lr `1e-5`, max 60,
  patience 12. Weight decay `1e-4`, gradient clip 1, cosine floor `1e-6`, AMP;
  BatchNorm/stochastic-depth behavior matches the primary trainer.
- Four GPUs take independent jobs dynamically. Inductor compilation uses
  dynamic batch sizes with CUDA Graphs disabled; evaluation is eager and caches
  are cleared per task. The training schedule/loss/batch is otherwise unchanged.

## Launch

```bash
bash study/exp2_face_rgb_ir_head32_regression/launch_screen.sh
screen -r exp2_rgb_ir_head32_regression
```

`python -m study.exp2_face_rgb_ir_head32_regression.run --prepare-only` prepares
all data/indexes. Then launch with `--train-prepared`. An existing training run
is never silently overwritten. A bounded end-to-end check uses
`--smoke-test --targets hemoglobin_low --devices 0` and writes `outputs_smoke/`.
It uses 1+1 epochs, one training batch and no compile, without touching main
outputs. Unit tests: `python -m unittest study.exp2_face_rgb_ir_head32_regression.test_experiment`.

## Automatic Results

`logs/run.log` records continuous epoch/job progress. Checkpoints, per-frame
predictions, video predictions, all train/val losses/r/MAE/RMSE/R2 histories and
data contracts are saved in `outputs/`, with per-target runs under
`runs/efficientnet_b0/<target>/`.

After each model, the existing RGB checkpoint is re-evaluated on the exact new
joint-selected RGB test frames; this is inference only. After all jobs, PNG
charts compare RGB main/original frames, RGB main/joint frames and RGB+IR/joint
frames on identical test videos and labels. Training histories use separate
correlation scales within the same panels, and ten-target figures use 5x2.
Raw metrics and comparison cohort audits are retained in CSVs.

The training cohorts differ because IR eligibility removes samples. Matched
test/frame evaluation reduces evaluation confounding but does not turn these
results into a same-training-cohort architecture-only ablation. The original
RGB results, checkpoints, data and split assignments are never modified.
