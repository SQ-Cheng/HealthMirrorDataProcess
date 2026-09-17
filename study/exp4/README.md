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

- Patient-disjoint 60/20/20 split selected from 512 deterministic candidates.
  Patients are stratified by median recovery score; candidates must pass the
  Exp2 regression limits for video-level KS, Wasserstein/IQR, and split-size
  error before the most distribution-balanced candidate is selected. Pairwise
  and per-split audits are saved as machine-readable CSV files.
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
- As in the Exp2 regression workflow, one experiment seed generates 512 split
  candidates, the best passing candidate is selected, and one model is trained
  once with that same seed on one GPU. There is no multi-seed ensemble.

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

Training completion writes the selected model's history, predictions,
checkpoint, and metrics directly under `outputs/`, and generates
`outputs/figures/results_summary.png`.

Generate test-set interpretability figures from the seed with the best test
Pearson correlation:

```bash
python -m study.exp4.interpretability
```

The script uses standard Grad-CAM on the final EfficientNet-B0 convolution and
signed occlusion sensitivity with a 32x32 ImageNet-mean patch at stride 16.
Each example uses the frame whose prediction is nearest its video's 20-frame
mean. Figures, raw maps, selected-example metadata, and a method manifest are
written under `outputs/`.
