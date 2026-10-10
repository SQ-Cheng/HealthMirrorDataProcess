# Exp9: observed preoperative and interpolated postoperative face regression

## Target construction

For each patient, analyte, and hospitalization, preoperative videos use the
nearest valid observed report strictly before CABG start. There is **no time
limit**, interpolation, or bilateral-coverage requirement for these videos.
The report may precede or follow the video but must belong to the same admission
and remain preoperative. Intra/postoperative reports are never substituted.

A postoperative video uses the unchanged separate piecewise-linear curve of
reports at or after CABG end. Do not interpolate across surgery or admissions,
or extrapolate. A video must lie wholly before CABG start or wholly after CABG end. Exclude
recordings overlapping any part of that interval or lacking valid CABG
metadata. Its scalar target is the relevant curve at the original capture
interval midpoint, using `numpy.interp` with explicit coverage validation for postoperative data only.
For the **entire postoperative video**, require a same-phase report strictly before capture
start and another strictly after capture end. Both distances must be <=24h.
An observation only at the midpoint or at a video boundary is not sufficient
without this before/after coverage. At an actual curve node, preserve its
observed value. Linear segments do not overshoot their endpoints.

Admissions with multiple valid CABG events are excluded rather than treating
the second operation's reports as part of a curve following the first.

Report timestamps, not independently documented blood-draw times, define
the horizontal axis. These are retrospective pseudo-labels, not direct
measurements at every video time. Future reports define targets but are not
input features. Patient-disjoint splitting prevents another patient's lab
curve from entering model fitting or target scaling.

## Alignment with current Exp2 regression

- Same eight analytes, source aliases, units, specimen policy, censoring and
  physical-range filters. Canonical patient/timestamp duplicates are already
  collapsed. Missing analytes do not exclude otherwise valid analyte tasks.
- Same native224 FFV1 crops, twenty nonadjacent source frames, and five views:
  original, horizontal flip, center crop, brightness and contrast.
- Same ImageNet-pretrained EfficientNet-B0 and scalar head:
  1280 -> 32 -> LayerNorm -> SiLU -> Dropout(0.25) -> 1, no output activation.
  One independent model per analyte. This is the existing frame-based video
  prediction protocol, not a new 3D/temporal network.
- Same frame-level unweighted SmoothL1(beta=0.5), train-only median/IQR raw
  target scaling, and video-level average of twenty original-view predictions.
- Same two stages: frozen encoder/head lr=2e-4, 40 epochs, patience 10;
  all parameters unfrozen/lr=1e-5, 60 epochs, patience 12.
- Same AdamW, weight decay=1e-4, cosine floor=1e-6, no warmup, gradient clip=1,
  mixed precision, channels-last and `torch.compile(reduce-overhead)`.
- Four-GPU dynamic scheduling; six train decode workers and two workers per
  evaluation split. The existing compact frame index is reused read-only.

By user choice, reuse the corresponding Exp2 patient split for every existing
patient. Newly eligible patients without that indicator's assignment are added
using deterministic fixed-seed 60/20/20 allocation; old patients never move.
No seed search occurs. Refit scalers on mixed **training labels only**. Distribution audits are saved;
the preparation also offers `--split-policy balanced_search` for a separately
requested protocol, never selected silently.

## Batch interpretation

Every full batch has twelve different preoperative assay events or postoperative interpolation support intervals,
twenty frames per video, and one view per video: 240 inputs and 240 frame
losses. All five views occur once per source frame across the epoch; the
partial final batch is retained. The original distinct-lab sampler is reused,
but its `clinical_event_id` identifies a real preoperative assay or a postoperative
support segment. Points sharing the same preoperative assay or the same two
support assays cannot occur in one batch. Adjacent segments can still share
one endpoint; synthetic targets are not independent extra lab observations.

## Outputs and checks

`outputs/task_records/` stores interpolated targets, both support values/times,
strict before/after coverage, interpolation weight, clinical threshold,
original nearest label, and the reused split. Eligibility audits retain every
video/analyte exclusion reason. Curves, bracketing distances, split
distributions, batch coverage and scaler fingerprints are inspectable.
Recorded CABG durations above 24h are flagged, not silently corrected.

Before formal training, the worker validates source hashes, exact phase
membership, strict bracketing, <=24h distance on each side, interpolation
values, patient separation, scaling and twenty-frame coverage. Checkpoints,
all histories and standard Exp2-style results figures are saved automatically.
If the original Exp2 main predictions are complete, both models are re-scored
on common held-out videos **against the same interpolated truth**. Original
nearest-label metrics are not mixed with interpolation-label metrics.

## Entry points

Preparation is CPU-only and is the safe default:

```bash
python -m study.exp9_face_interpolated_lab_regression.run_all
python -m study.exp9_face_interpolated_lab_regression.run_all --check-only
python -m unittest study.exp9_face_interpolated_lab_regression.test_interpolation
```

Standalone formal training requires an explicit command; monitored autostart is described below:

```bash
bash study/exp9_face_interpolated_lab_regression/launch_screen.sh
screen -r exp9_face_interpolated_lab_regression
```

Results: `study/exp9_face_interpolated_lab_regression/outputs/`.
Formal log: `study/exp9_face_interpolated_lab_regression/logs/run.log`.
Existing Exp2 results and the active training queue are not modified.

## Current autostart

```bash
bash study/exp9_face_interpolated_lab_regression/launch_after_current_screen.sh
screen -r exp9_after_exp2_main
```

The detached monitor waits for both current Exp2 main experiments and their
figures to finish successfully. It allocates no GPU while waiting, revalidates
the prepared data, and starts eight Exp9 models on four GPUs. Histories,
checkpoints, result figures and common-test comparisons are automatic.

Expanded preoperative sessions come from the full validated video inventory,
not the original 24h-labelled subset. Native-only sessions absent from the
legacy inventory are independently validated. Original packet offsets are
reused and only additional videos indexed; the active Exp2 cache stays intact.
