# Frozen RGB+IR DINOv3-B Regression

Six structures, each with ten independent single-analyte regression heads:

| Family | RGB representation | IR representation | Fusion |
| --- | --- | --- | --- |
| DINOv3-B | Frozen DINOv3-B/16, 768 | Independent frozen DINOv3-B/16, 768 | Concat / vector gate / scalar gate |
| DINOv3-B + FaRL | Frozen DINO768 + FaRL512 concat, 1280 | Independent frozen DINO768 + FaRL512 concat, 1280 | Concat / vector gate / scalar gate |

All RGB and IR encoders are separate instances initialized from the same official
pretrained checkpoints; no IR-specific pretraining or encoder updates are claimed.
The native IR gray channel is copied to three channels, not pseudo-colored.
Each branch uses its own correct pretrained normalization (ImageNet for DINO,
CLIP for FaRL), after synchronized deterministic views. Original independently
cropped RGB/IR faces are not additionally spatially registered.

## Fusion

- Concat: `[RGB, IR] -> Linear(2d,32) -> LN -> SiLU -> Dropout(.25) -> Linear(32,1)`.
- Vector gate: independently project RGB and IR to64 with Linear/LN/SiLU;
  `g=sigmoid(Linear([RGB64,IR64],64))`; `z=g*RGB64+(1-g)*IR64`;
  `z -> Linear(64,32) -> LN -> SiLU -> Dropout(.25) -> Linear(32,1)`.
- Scalar gate: the same construction but one input-dependent scalar is broadcast
  across all64 projected coordinates. Both gates start at .5.

The DINO+FaRL family concatenates the two pretrained representations **within
each modality first**, then applies the same RGB-versus-IR fusion above.

| Family | Concat parameters | Vector-gate parameters | Scalar-gate parameters |
| --- | ---: | ---: | ---: |
| DINOv3-B | 49,281 | 109,121 | 100,994 |
| DINOv3-B + FaRL | 82,049 | 174,657 | 166,530 |

Counts include all trainable projections/gates/head, not frozen encoders. The
task head hidden width is always32; gated fusion uses a64-dimensional latent.
Each frozen DINOv3-B has85,669,632 parameters; each FaRL image encoder86,192,640.

## Data and Training

Reuse the verified RGB+IR experiment's exact current ten-task records, patient
splits, paired train-only median/IQR scalers, and task seeds. No rematching,
patient reassignment, or split search is performed. All six models have identical
training/test data for a given analyte. Per-task splits differ between analytes,
which is safe because only independent heads are trained; frozen representations
are label independent and never fitted to any clinical data.

Tasks: oxyhemoglobin fraction, lactate, urea, total bilirubin, platelet count,
hemoglobin, A/a PO2 ratio, creatinine, hematocrit and direct-field eGFR. Historical
`*_high`/`*_low` target IDs are inherited identifiers; these are **raw-value
regressions**, not binary classifiers.

- Nearest lab within +/-24h, inherited exactly from the RGB main experiment.
- Twenty mutual-nearest RGB/IR frame pairs/video, unique IR use, <=33.333ms
  pair separation, source index gap>=2 in both modalities.
- Five train views: original, horizontal flip, center crop(.9), brightness(+.06),
  contrast(+.08). All five appear each epoch; one view per video in a batch.
- Batch:12 distinct clinical events x20 frame pairs x1 synchronized view =240
  paired inputs; short final batches retained without duplicating/dropping data.
- One frozen-backbone stage, AdamW lr2e-4, weight decay1e-3; cosine to1e-6,
  max80 epochs, patience12; select by lowest validation video-level raw MAE.
- Unweighted per-frame SmoothL1(beta=.5) on robust-scaled raw values;
  gradient clip1, FP16 AMP, no compilation.
- Validation/test:20 original frame pairs/video; mean head predictions then
  inverse scaling, report video-level MAE/RMSE/Pearson r/R2/Explained Variance
  and the existing secondary abnormality-derived metrics.

Frozen FaRL/DINO representations use four read-only float32 memmaps (~1.25GiB
total). No new decoded images/videos are saved. Existing RGB feature rows are
reused only for exactly identical FFV1 packets/source frames, weights, views,
and normalization. Both groups share these deterministic frozen caches.
Cache manifests include weight/source/index hashes; changed inputs are rejected.

## Run and Outputs

```bash
bash study/exp2_face_rgb_ir_dinov3b_regression/launch_screen.sh
screen -r exp2_rgb_ir_dinov3b
tail -f study/exp2_face_rgb_ir_dinov3b_regression/logs/run.log
```

The script uses GPUs0-7 for bounded asynchronous feature extraction, then an
eight-worker dynamic head queue. A bounded real-data one-epoch smoke checks all
six structures before the full run. Frozen-head fitting is much lighter than
backbone training; eight allocated GPUs need not show100% arithmetic utilization.

Artifacts are separate from all existing results:

- `outputs/experiment_manifest.json`, `cohort_counts.csv`, `source_records/`:
  machine-readable full protocol and per-split video/patient/event counts.
- `outputs/regression/<family>_<fusion>/runs/<analyte>/`: checkpoint, full
  train/val history, metrics, video predictions, and individual `train.log`.
- `outputs/regression/<family>_<fusion>/figures/`: training curves and fitted
  predicted-versus-measured scatter plots.
- `outputs/figures/`: five six-structure grouped-bar comparison PNGs;
  `test_comparison.csv` and `paired_test_audit.csv` verify exact test equality.
- `outputs/COMPLETE` exists only after all60 heads and all figures succeed.

### Matched RGB-Only Comparisons

`outputs/figures/rgb_only_vs_rgb_ir_pearson_r_overview.png` compares both backbone
families and all three fusion options in a2x3 overview. The ten additional
`rgb_only_vs_rgb_ir_<dino|dino_farl>_<metric>.png` files provide per-analyte
MAE/RMSE/Pearson r/R2/Explained Variance grouped bars. They are generated
automatically at completion and by the normal plots command, or independently:

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_face_rgb_ir_dinov3b_regression.compare_rgb
```

No model is retrained or changed. Both RGB-only and RGB/IR checkpoints are
reevaluated using CPU float32 heads, the same paired test videos, exact paired20
RGB cached frames, and each checkpoint's own train-only target scaler. This
avoids raw-unit bias from using the paired model's scaler for an RGB checkpoint.
Original predictions are preserved; the new predictions, metrics, audit and
checkpoint/cache hashes are under `outputs/tables/rgb_only_vs_rgb_ir_*`.

These are backbone-family matched comparisons, not a strict same-training-cohort
IR ablation. RGB-only models used more training videos. The same single-DINO
RGB baseline is repeated against all three RGB/IR heads. Also, RGB-only FaRL
vector/scalar gates fuse DINO versus FaRL, whereas RGB/IR FaRL gates fuse the
two modalities after within-modality DINO/FaRL concatenation. Figure footnotes
and machine-readable audits explicitly identify these differences. The six
new RGB/IR structures remain same-data comparisons with one another.

```bash
/root/miniconda3/envs/healthmirrorenv/bin/python -m unittest study.exp2_face_rgb_ir_dinov3b_regression.test_experiment
/root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_face_rgb_ir_dinov3b_regression.run --check-only
```
