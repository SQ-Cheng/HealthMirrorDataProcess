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
Each run also saves `ir_stem_audit.json` with the head-stage, fine-tune and
selected-model IR filter norms. The main validation selection rule is unchanged;
if the head-stage checkpoint wins, the selected model still ignores IR.

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

## RGB-Trained Initialization Variant

```bash
bash study/exp2_face_rgb_ir_head32_regression/launch_rgb_main_screen.sh
screen -r exp2_rgb_ir_rgb_main_init
```

The default run above starts from ImageNet. This additional variant starts from
each analyte's selected current RGB main `model.pt`, including both backbone and
head32. It then expands the stem and zeros the IR slice in exactly the same way.
The pretrained head's final linear weights/bias are converted to the paired
training cohort's existing median/IQR, preserving the initial raw prediction.
This does not alter the head design, paired target scaler or training schedule.

Patient splits, job seeds, task records, twenty synchronous pairs, all five views,
loss, batch composition and 40/60-epoch two-stage settings are copied from the
completed ImageNet RGB+IR run; no rematching or split search is performed.
Pretrained RGB checkpoint training-set/scaler provenance is checked against the
current source records, with checkpoint SHA256 recorded for every task.

Results and checkpoints: `outputs/rgb_main_init/`. Log:
`logs/rgb_main_init/run.log`. The original `outputs/` results remain untouched.
All new PNGs are generated automatically, including
`outputs/rgb_main_init/figures/initialization_comparison.png`: RGB main,
ImageNet-initialized RGB+IR and RGB-trained-initialized RGB+IR on the identical
joint-selected frames and test cohort. CSV audits accompany the chart.
The supervised RGB initialization saw the larger original training cohort but
the same held-out patients; this difference is explicitly reported.

Preparation: `python -m study.exp2_face_rgb_ir_head32_regression.run
--initialization rgb_main --prepare-only`, followed by the launch script with
`--train-prepared`. A bounded check uses `--initialization rgb_main --smoke-test
--targets hemoglobin_low --devices 0` and writes `outputs_smoke_rgb_main_init/`;
it retains the full shared frame index without rebuilding or altering it.

## Frozen RGB / Joint IR And Head

```bash
bash study/exp2_face_rgb_ir_head32_regression/launch_frozen_rgb_ir_head_screen.sh
screen -r exp2_rgb_ir_frozen_rgb_ir_head
```

This additional single-stage run loads the same per-task RGB main best model
and equivalent head-scaling conversion. Only the new IR stem slice (288 weights)
and head32 (41,089 weights) train: 41,377 parameters in total. RGB stem filters
are separate immutable parameters, concatenated with trainable IR filters in the
forward pass; no gradient mask or post-update restoration is needed. All other
backbone parameters and BatchNorm buffers are frozen. Backbone stochastic depth
is disabled while head dropout remains .25. Autograd still propagates through
the frozen backbone to the IR slice.

AdamW uses two simultaneous parameter groups: IR lr 1e-4 and head lr 5e-5,
weight decay 1e-4, cosine floor 1e-6 for both, maximum 60 epochs, patience 12.
The best checkpoint is selected by validation video MAE. AMP, clip 1, 24h,
ten targets, patient splits/seeds, twenty synchronous pairs, five views,
12 distinct lab events per batch and frame-level SmoothL1(.5) are unchanged.
This variant's launcher disables `torch.compile` (`--no-compile`); training and
evaluation run eagerly. This does not change AMP, batching or optimization.

Outputs: `outputs/rgb_main_frozen_ir_head/`; log:
`logs/rgb_main_frozen_ir_head/run.log`. Automatic PNG figures include
`figures/frozen_rgb_ir_head_comparison.png` with RGB main, both earlier RGB+IR
variants and the new joint model, using common test videos and identical RGB
frames. The comparison explicitly distinguishes the one-stage and two-stage
schedules. Every task writes `frozen_rgb_audit.json`, proving that ALL original
backbone parameters/buffers remained bitwise identical and both IR/head updated.
Checkpoints require `build_joint_model` because the stem saves separate
`features.0.0.weight` (RGB3) and `features.0.0.ir_weight` (IR1) tensors.

Use `--initialization rgb_main --training-protocol joint_ir_head --prepare-only`
with the run module to prepare; pass `--train-prepared` to the launch script.
The bounded `--smoke-test --targets hemoglobin_low --devices 0` mode performs
one joint batch and saves to `outputs_smoke_rgb_main_frozen_ir_head/` without
changing the full cached frame index or existing results.
