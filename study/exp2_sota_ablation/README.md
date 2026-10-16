# RGB/IR DINOv3-B + FaRL Improvement Ablations

This directory defines seven exploratory improvement protocols for the completed
`exp2_face_rgb_ir_dinov3b_regression/dino_farl_scalar_gate` baseline. "sota" is
the requested directory name, **not a claim of state-of-the-art performance**.
No baseline is retrained, no old output is overwritten, and training requires an
explicit `--run` or manual invocation of the screen launcher.

## Baseline and Evidence

The current baseline has four independent, permanently frozen encoders: RGB and
IR each use DINOv3-B/16 (CLS768) and FaRL ViT-B/16 (512). Per-modality concat1280
is projected to64, followed by an input-dependent scalar RGB/IR gate and a
head32. There are166,530 trainable parameters and343,724,544 frozen encoder
parameters per analyte. It is a **one-stage frozen-feature experiment**, not the
two-stage EfficientNet experiment.

The read-only audit produces:

- `outputs/audit/BASELINE_REPORT.md`: architecture/hyperparameter table and
  selected-checkpoint train/validation/test results.
- `outputs/audit/tables/baseline_architecture_hyperparameters.csv`.
- `outputs/audit/tables/baseline_results.csv`, `cohort_counts.csv`.
- `outputs/audit/tables/baseline_gate_distribution.csv`.
- `outputs/audit/tables/train_feature_scales.csv` and `ablation_plan.json`.
- `outputs/audit/provenance.json`: clinical, model, cache and source hashes.

Seven of ten analytes select epoch1. Selected-checkpoint train/validation
Pearson r is .985/.434 for lactate, .981/.107 for platelets and .999/.601 for
hematocrit. There are only270-327 training patients per task, not tens of
thousands of independent observations. Twenty frames/five views repeat labels.
Original-view training feature RMS is approximately .572/.170 for RGB
DINO/FaRL and .559/.159 for IR DINO/FaRL. This motivates post-encoder calibration;
it is not proof that different scales are a bug. The observed gates are not
generally saturated: mean RGB gate values span approximately .492-.560, and no
evaluated original-view frames have gates below .05 or above .95. Gate values
are mixing coefficients, not identifiable percentages of predictive information.

See [DESIGN.md](DESIGN.md) for the full hypotheses and caveats.

## Fixed Data Contract

- Ten **independent** single-analyte regressions: oxyhemoglobin fraction, lactate,
  urea, total bilirubin, platelet count, hemoglobin, A/a PO2 ratio, creatinine,
  hematocrit and direct-field eGFR. Old `_high`/`_low` IDs do not make these binary.
- The union is1,311 paired videos from549 patients. Reuse exact baseline records,
  labels and per-analyte patient assignments; no re-search or rematching.
- Nearest lab within +/-24h;20 mutual-nearest, unique IR pairs/video, <=33.333ms
  separation; source-frame index gaps>=2 in both modalities.
- Same20 original pairs in validation/test. No modality-specific sample dropping.
- Same target-specific seeds. No cross-analyte learned encoder sharing.
- Raw lab values reported in original units, even for the log-target protocol.
- Only training records fit target/feature transformations, density weights,
  patient weights and time weights. Validation selects the checkpoint by raw MAE.
  Test is evaluated after checkpoint selection, not used for early stopping.

## Seven Protocols

| ID | Primary change | Head LR | Max epochs / patience | WD | Views |
| --- | --- | ---: | --- | ---: | --- |
| a01_regularized_rgb_prior | Train-only branch z-score; head dropout .5; projection dropout .15; IR modality dropout .2; initial RGB gate .8 | 1e-4 | 40 /8 | .01 | 5 |
| a02_video_patient_balance | Video/view SmoothL1; up to12 distinct patients/events per batch; equal patient and within-patient event base weight | 1e-4 | 40 /8 | .003 | 5 |
| a03_hierarchical_rgb_warmstart | Scalar DINO/FaRL gate within each modality, scalar RGB/IR gate; init from existing supervised RGB scalar-gate model; RGB prior .95 | 5e-5 | 30 /8 | .003 | 5 |
| a04_log_density_rank | Video loss; train-only log1p on4 right-tailed analytes; capped inverse-density weighting; pair ranking .05 | 1e-4 | 40 /8 | .003 | 5 |
| a05_color_safe_views3 | Video/patient balance; remove brightness/contrast; train-only feature calibration; dropout .4, IR dropout .1, RGB prior .8 | 1e-4 | 40 /8 | .003 | 3 |
| a06_time_weighted | Video/patient balance; multiply train weights by .25+.75 exp(-abs(match_h)/6); keep all records | 1e-4 | 40 /8 | .003 | 5 |
| a07_ir_tail_adaptation | a03 warm stage, then train last IR DINO/FaRL block, terminal norms/FaRL projection and fusion; RGB stays frozen | 5e-5 then1e-5 | 30 /8 then20 /6 | .003 head; .01 encoder in stage2 | 5 |

Common: head hidden32; latent64; AdamW; clip1; SmoothL1 beta .5; FP16 AMP on
V100; two warmup epochs followed by cosine to1e-6. The unmodified baseline used
no warmup, lr2e-4, max80, patience12 and wd.001. a01 is frame loss; a02-a07
use one loss on the mean of20 predictions from one video/view. Original-frame
train/val losses remain separately recorded for readable like-unit curves.

a04 uses log1p only for lactate, urea, bilirubin and creatinine; the other six
remain raw-value targets. Quantiles and median/IQR are fitted in the chosen
target space on train only. Density bins use raw train1st-99th percentile
fixed-width edges, not equal-frequency bins. All raw targets, including tails,
remain in the dataset. Ranking excludes pairs closer than .25 training IQR,
uses temperature .2, and is auxiliary to regression.

a07 stage2: IR encoder LR1e-6 ->1e-7, fusion/head LR1e-5 ->1e-6, two warmup
epochs, max20/patience6. Train only IR final Transformer blocks, DINO final
LayerNorm, FaRL final LayerNorm and visual projection:14,573,568 encoder parameters.
Hierarchical fusion adds167,172 trainable parameters (166,530 for the other
fusion model), keeping head32. Total stage2 trainable parameters:14,740,740.
Encode four IR images/microbatch; accumulate each complete20-frame video graph,
backpropagate its video loss, then update once per logical batch. Do not compute
video loss separately per microbatch. RGB features remain cached; adapted IR
features are recomputed and **never replaced with stale frozen IR cache values**.
IR encoders use Inductor compilation without CUDA Graphs by default; compilation
is now static/fullgraph (`dynamic=False`, `fullgraph=True`), and FaRL attention
uses a local mask rather than mutating an attribute during forward. Compilation
failures raise. See [A07_COMPILE_FIX.md](A07_COMPILE_FIX.md). Frozen-feature MLP experiments stay eager because their cached
heads are small and the baseline is eager.

## Batch and Weight Semantics

Every training video retains20 frames x5 views (=100 inputs/epoch), except the
intentional views3 protocol (=60). A logical batch contains up to12 distinct
clinical measurements x20 frames xone common view per video. a01 keeps the old
distinct-event sampler. a02-a07 additionally require distinct patients, and keep
all tail batches rather than duplicating/discarding data. Patient/event base
weight is proportional to1/(events_in_patient xvideos_for_event), so five
augmentations and multiple videos do not inflate one patient's clinical mass.
Additional time/density factors in a04/a06 intentionally modify that base weight.
No patient ID, lab value or time interval is supplied as a model input.

## Automation and Outputs

**Do not run the following training command until authorized.** The files exist;
the launcher has not been invoked as part of this design task.

```bash
bash study/exp2_sota_ablation/launch_screen.sh
screen -r exp2_sota_ablation
```

It runs a01 through a07 sequentially. Within one protocol, up to eight GPUs
dynamically execute the ten analytes. After each complete protocol, own history
and scatter figures are generated, then comparisons against the original
baseline and every completed protocol are refreshed. Any failure stops the
chain before the next protocol. Restarting the same command resumes only
unfinished target jobs with identical protocol hashes. It does not restart
or stop other experiments; choose free GPUs via the direct runner if needed.

```bash
# Print the full machine-readable plan; default CLI does NOT train.
/root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_sota_ablation.run
# CPU-only source/diagnostic audit, no clinical optimization.
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_sota_ablation.run --audit
# Explicit opt-in, selected protocols/devices:
bash study/exp2_sota_ablation/run_all.sh --variants a01_regularized_rgb_prior a02_video_patient_balance --devices 0,1,2,3
```

- `outputs/<protocol>/runs/<target>/`: all train/val history, selected checkpoint,
  phase-best checkpoints, raw train/val/test predictions and per-target log.
- Online checkpoints contain only fusion and **trainable IR tail tensors**,
  referring to shared original pretrained weights; no four-backbone duplicates.
- `outputs/<protocol>/figures/`: history, predicted/true with linear fit, paired
  improvement confidence intervals versus baseline.
- `outputs/figures/`: incremental MAE/RMSE/r/R2/Explained Variance comparisons
  and macro overview. PNG only, no PDFs.
- `outputs/tables/`: same-test cohort audits, metrics,500-resample paired
  patient-cluster bootstrap intervals, and validation-only winners.
- `logs/run.log`: continuous suite log; each target also has its own log.

Warm starts deliberately use additional original RGB-only training videos,
although all paired supervised stages/evaluation cohorts stay identical. Before
loading each warm start, the runner verifies the exact old patient roles and
that no paired validation/test patient occurred in RGB training. Final-layer
scaling is converted between the RGB and paired training scalers. Initial warm
model epoch0 is eligible for validation-based selection, so a good initializer
is not lost if every update hurts. These are performance-oriented bundles, not
strict isolated-variable causal ablations. No improvement is guaranteed; test
plots must not be used to adapt the pre-specified protocols.

## Verification Scope

CPU source audits and synthetic unit tests were run at design time. During the
2026-10-16 compiler repair, the real dual-IR encoder/fusion path passed a GPU
20-frame forward/backward and repeated evaluation smoke test with random inputs
and no optimizer updates. Old A07 remains stopped after its compiler failure.
Source code and dependency imports, tail selection/checkpoint round trips,
warm-start raw-unit conversion, losses, sampling, transforms and scheduling
are covered by tests. The repaired smoke peak allocated memory was1350.8MiB;
full-epoch data loading and throughput still depend on the real run.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /root/miniconda3/envs/healthmirrorenv/bin/python -m unittest study.exp2_sota_ablation.test_design study.exp2_sota_ablation.test_plots
```
