# Five-Fold Cumulative RGB/IR Regression Ablations

Training was explicit opt-in after design. The 2026-10-16 IR compiler repair
preserves75 completed frozen-only fold jobs and all patient splits; incomplete
jobs restart, and the eight-GPU queue resumes. The old single-split A07 stays
stopped. `sota` is a directory name, not a performance claim.

## Evidence and Design

Completed single-split baseline: macro Pearson r=.330, R2=.082. The regularized
A01 gives r=.321, R2=.103. Pure video loss/patient balance (A02), the stronger
log/density/rank bundle (A04), and strong time weighting (A06) did not consistently
improve performance. A07 was unfinished when this design was prepared. These
findings motivate softer combinations, not a claim that every added component
must improve performance. `outputs/design_evidence/` snapshots actual completed
parent results and hashes; pending protocols are explicitly excluded.

This is exploratory evaluation on an already-studied cohort. Earlier results
informed the design, so the new folds are not an untouched external confirmatory
test. The fixed protocols below are all reported; no automatic test-based winner,
hyperparameter adaptation, or best-of-five checkpoint selection is allowed.

## Twelve Fixed Protocols

Each row trains all ten independent analyte models in all five folds. C01-C08
inherit the preceding row's settings, not its trained weights. X01-X03 are
deliberately non-cumulative bundles. No dependent-model barrier is needed.

| ID | Change relative to previous cumulative row | LR / maximum epochs / patience |
| --- | --- | --- |
| B00 | Original frozen four-encoder scalar fusion reference, frame loss, 5 views | 2e-4 /80 /12 |
| C01 | Train-only feature z-score; head/projection dropout .5/.15; IR dropout .2; RGB prior .8; WD .01; two warmup epochs | 1e-4 /40 /8 |
| C02 | Hybrid loss: .5 frame SmoothL1 +.5 SmoothL1 of mean20 frame predictions | inherited |
| C03 | Patient-diverse sampler; soft patient/event weights, exponent .5 instead of1 | inherited |
| C04 | Keep original/hflip/center crop; remove brightness and contrast | inherited |
| C05 | Hierarchical scalar DINO/FaRL and RGB/IR gates; fresh fold-local RGB warm start; RGB prior .95 | 5e-5 /30 /8 |
| C06 | Multiply training weight by .5+.5 exp(-abs(match_h)/12); keep all24h records | inherited |
| C07 | Add mild inverse-density weighting, exponent .25, cap2 | inherited |
| C08 | Add IR-only final-block adaptation; RGB stays frozen | head1e-5, IR1e-6 /20 /6 in added stage |
| X01 | C07 +weak auxiliary ranking weight .01 | same as C07 |
| X02 | Independent hybrid/log/rank bundle: raw-feature-normalized scalar model, log1p for4 analytes, soft patient/density weights, rank .01 | 1e-4 /40 /8 |
| X03 | C07 but frame-only loss; checks whether video supervision hurts | same as C07 |

Head hidden dimension32, projected dimension64. Fusion has166,530 trainable
parameters, or167,172 for hierarchical fusion. Four pretrained encoders remain
frozen except the C08 IR adaptation phase (14,573,568 trainable encoder parameters).
Common: AdamW; SmoothL1 beta .5; gradient clipping1; FP16 AMP; cosine floor1e-6;
two warmup epochs except B00. C08 IR floor1e-7, encoder WD .01, IR microbatch4;
complete20-frame graphs produce a single hybrid video objective before backward.
The head/fusion WD is inherited (.01 for cumulative rows). Frozen cached heads
are eager; C08 online IR encoders use strict Inductor without CUDA Graphs.
They use static full graphs and side-effect-free FaRL attention to avoid the
PyTorch2.4 dynamic-guard/CLIP graph-break bugs. Full20-frame train/eval GPU smoke
checks verified graph reuse, finite gradients, and per-video modality dropout.

Soft patient weights are proportional to
`1/sqrt(number_of_events_in_patient * number_of_videos_for_event)`.
They reduce dominance without forcing every patient's total weight to be equal.
Density uses five fixed-width bins between training P1 and P99, retains tails,
and caps the normalized factor. X02 log1p applies only to lactate, urea,
bilirubin and creatinine, before training-only robust scaling. No label clipping.
Ranking uses distinct video means in the batch, target gap>=.25 training IQR,
temperature .2; C08 does not include ranking, avoiding cross-video graph retention.

## Data and Leakage Protection

The same ten target cohorts as the parent experiment are retained: O2Hb fraction,
lactate, urea, bilirubin, platelets, hemoglobin, A/a PO2 ratio, creatinine,
hematocrit and direct-field eGFR. Union:1,311 paired videos /549 patients. Nearest
lab within+/-24h,20 unique mutual-nearest RGB/IR pairs/video, <=33.333ms, native224.
All schemes share each target's new patient folds; labels are not rematched.
Folds differ across targets, and no supervised encoder is shared across targets.

1. **Outer five-fold:** ~80% patients train /~20% patients test; all videos and
   lab events of a patient stay together. Every video has exactly one OOF test
   prediction. Video-count fractions need not equal patient-count fractions.
2. **Inner holdout:** split outer80% again80/20 (~64%/16% of all patients).
   Select epochs and, for C08, whether to use IR adaptation by minimum inner
   validation video-level raw MAE. The outer test records are absent from this
   trainer. Warm initialization can be selected at epoch0.
3. **Full outer refit:** discard selection models; reinitialize and train on
   **all outer80%** for the already selected number of epochs per stage. Refit
   repeats that prefix of the original LR schedule, not a newly shortened cosine.
   No validation/test early stopping or checkpoint choice happens during refit.
4. **Final test:** save the fixed-budget final checkpoint first, then evaluate
   outer20% once. Original20 frames only, no test-time augmented views. Average
   predictions in target space, then invert to raw units. Train metrics may also
   be saved; test values never enter training histories or selection.

Outer balance search:256 seeded candidates with patient-level median-value
quantile strata, scored by raw-value Wasserstein/IQR, KS, quantile and video-size
discrepancy. This is label-based distribution stratification, not model seed
selection. Inner search128 candidates uses only outer-training records. The same
saved splits are used by every protocol. No model performance enters split search.

Old single-split supervised RGB warm starts **must not be reused**: they can
contain new test patients. For each target/fold, a fresh RGB expert is selected
on the inner holdout, then separately refitted on outer80%. Inner downstream
models use only the inner expert; full80% refits use only the outer expert.
No extra RGB-only videos are added. These two fold-local initializers are
shared read-only across compatible variants, protected by a per-fold file lock.
Feature z-score changes are absorbed exactly into pretrained projection weights
and biases, preserving the RGB initializer function in raw feature coordinates.

Targets/scalers/feature normalization/weights are independently fitted on inner
train for selection, and on outer80% for refit. Original split/scaled labels are
not training inputs. Label-independent frozen feature caches are mmap-reused;
IR adaptation recomputes IR features, never reads stale adapted IR from cache.

## GPU Queue, Resumption and Figures

The global queue contains600 final fold jobs (12x10x5), each including inner
selection and refit. RGB initialization adds50 inner+50 outer fits, shared across
warm-start variants. C08 may add a second phase. Each free GPU takes the next
job immediately: **there is no ten-analyte or five-fold barrier between variants**.
Up to eight GPU workers run concurrently. CPU plotting is asynchronous and
coalesced, so dispatch does not wait for matplotlib. Config accumulation does
not require a previous variant's model checkpoint.

Errors stop new dispatch; already active jobs finish. A restart skips verified
completed fold jobs and reruns incomplete folds, not arbitrary mid-epoch resume.
Source/protocol/code and saved split fingerprints prevent mixed-result reuse.
Other experiments are never stopped or changed; choose free device IDs explicitly.

- `outputs/splits/`: assignments, inner/refit records, hashes, distribution tables.
- `outputs/<variant>/<target>/foldN/`: selected budget, final model, train/test
  predictions, metric table, job log; `selection/` and `refit/` stage histories
  and compact checkpoints. Final `complete.json` fingerprints all key outputs.
- `outputs/rgb_initialization/`: audited per-fold RGB initializers, no old weights.
- `outputs/<variant>/oof/`: every patient's held-out predictions, one row/video.
- `outputs/<variant>/figures/`: OOF predicted/true scatter with identity/fit lines.
- `outputs/figures/`: incremental metric bars with five-fold mean+/-SD and split
  distribution. Complete-five-fold targets only; partial progress stays in tables.
- `outputs/tables/`: fold metrics, fold mean/SD, pooled OOF metrics and paired
  patient-cluster bootstrap intervals against B00. Pooled r is not mean fold r.
- Per-fold history figures use a separate, correctly scaled r axis within each
  panel. PNG only. Bootstrap resamples patient clusters, not frames; it describes
  uncertainty conditional on fitted models, not training randomness.

## Commands

The screen launcher respects `outputs/paused_variants.json` when no explicit
`--variants` arguments are given. C08 was paused on2026-10-16 at the user's
request; its partial histories/checkpoints remain, while X01-X03 run next.
It will not automatically resume C08 afterward. Explicit `--variants` overrides
this launcher scheduling hold; training settings/splits/protocol are unchanged.

```bash
# Print plan only, no preparation or training.
/root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_sota_5fold_ablation.run
# CPU source audit, fold search and distribution figure only.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_sota_5fold_ablation.run --prepare
# Explicit manual opt-in: detached screen on all eight GPUs.
bash study/exp2_sota_5fold_ablation/launch_screen.sh
# Or selected free devices, e.g. 0-3:
bash study/exp2_sota_5fold_ablation/launch_screen.sh 0,1,2,3
# Explicit selection (overrides the launcher's scheduling hold):
bash study/exp2_sota_5fold_ablation/launch_screen.sh 0,1,2,3 --variants x01_rank_bundle x02_log_rank_bundle x03_frame_only_bundle
# Attach or inspect continuous suite log:
screen -r exp2_sota_5fold_ablation
tail -f study/exp2_sota_5fold_ablation/logs/run.log
```

CPU synthetic tests cover split integrity, independent folds, full80% refit,
no-test selection, cumulative configuration, calibrated warm starts, hybrid
losses/weights, failure-stop dispatch and crossing variant boundaries. Actual
online GPU20-frame compilation/memory was validated during the compiler repair.
Training/evaluation each need an initial graph pair; subsequent views/calls reuse
those graphs. The explicit migration record is under `outputs/protocol_migrations/`.
Preserved checkpoint bytes retain their original contract; completion markers
record that origin and why the frozen-only result is compatible with the repair.
