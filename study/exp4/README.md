# Exp4: Postoperative Recovery from Facial Videos

## Task

For surgical hospitalizations, recovery is defined as 0 at the end of the
final valid surgery and 1 at discharge. A video receives the linearly
interpolated score at its capture-interval midpoint. The complete video must
fall between surgery end and discharge. Patients without a valid surgery are
excluded.

Hospitalization and surgery metadata come from `merged_lab_tests.csv`; lab
result values are not used. Video time comes from `video.avi.ts` and is checked
against `patient_info.txt` Session Timestamp after conversion to
`Asia/Shanghai`. Videos with an absolute start-time disagreement over five
minutes are excluded.

## Model and evaluation

- Patient-disjoint, recovery-distribution-balanced 60/20/20 split.
- 20 nonadjacent color face frames per video, streamed through a compact byte
  offset index without a decoded image cache.
- ImageNet-pretrained EfficientNet-B0 and a 32-dimensional scalar sigmoid head.
- Training views: original, horizontal flip, and center crop. Color intensity
  is not altered because pallor may be task-relevant.
- Stage 1 freezes the backbone and trains the head at `1e-3`.
- Stage 2 unfreezes only the last EfficientNet stage; backbone/head learning
  rates are `1e-5`/`1e-4`.
- Patient-balanced SmoothL1 training; evaluation averages the 20 original-frame
  predictions for each video.
- Four model seeds use the same split and run concurrently on four GPUs.

## Commands

Prepare and audit only:

```bash
python -m study.exp4.run_all --prepare-only
```

One-batch smoke test:

```bash
python -m study.exp4.run_all --smoke
```

Formal detached run:

```bash
bash study/exp4/launch_screen.sh
```

Training completion automatically produces per-seed histories, predictions,
checkpoints, aggregate metrics, and `outputs/figures/results_summary.png`.
