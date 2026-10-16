# Design Rationale and Interpretation

## What Is Being Tested

The baseline is not end-to-end facial physiology adaptation: four general
pretrained encoders are frozen and only166,530 parameters fit several hundred
training patients per task. The training frames/views are highly correlated.
This motivates a combination of improved statistical units, less aggressive
head fitting, branch calibration and a narrowly constrained IR adaptation stage.
The proposed gains are hypotheses, not established SOTA claims.

## Baseline Findings

1. Seven analytes select epoch1. Later training correlations become very high
   while validation does not improve. This supports lower learning rates,
   short warmup, stronger regularization and validation-preserved warm starts.
2. Selected train/validation r for lactate is .985/.434; platelets .981/.107;
   HCT .999/.601. These refer to the selected checkpoint, not only a late,
   discarded overfit epoch. Merely increasing epochs or head width is not the
   first intervention.
3. DINO and FaRL pretrained output scales differ roughly3.4x. A shared Linear
   projection can learn rescaling, so this is not an implementation error, but
   train-only coordinate standardization may improve conditioning and make
   regularization less dependent on the pretrained branch scale.
4. Gate mean RGB coefficients are about .492-.560 with no observed original
   gate below .05 or above .95. General gate collapse is not demonstrated.
   Equal coefficients also do not imply equal modality importance, because
   projections and the nonlinear head can rescale/rotate information.
5. Supervision is per-frame, while deployment/selection averages20 frames.
   One label cannot explain every instantaneous exposure/pose equally. A
   video/view objective is aligned with final prediction and avoids enforcing
   each individual noisy frame to reproduce exactly the same label.
6. The five views include brightness and contrast changes. Those can improve
   lighting robustness but may also alter clinically useful color cues. The
   views3 experiment tests this trade-off rather than declaring photometric
   augmentation inherently wrong.
7. Matching is +/-24h. Time mismatch may be more harmful in rapidly varying
   postoperative labs. Smooth train weighting tests label confidence without
   changing the cohort or excluding the difficult distant test samples.
8. Both IR encoders start from RGB-image pretraining and remain frozen in the
   baseline. Repeating gray IR into3 channels is input compatibility, not
   domain adaptation. The IR tail protocol tests whether a small amount of
   supervised adaptation provides useful complementary representations.

## Protocol Details

### A01: Regularize and Calibrate

Keep the original scalar modality architecture and frame objective. Fit one
mean/std per feature coordinate on original training frames only, with std
floor .05; do not normalize each image independently. Increase head dropout
from .25 to .5, add .15 projected-feature dropout and .2 IR modality dropout.
The IR mask is constant across all20 frames of a video/view and disabled at
evaluation. Start the scalar RGB coefficient at .8. AdamW lr1e-4/wd.01,
40 epochs/patience8, two warmup epochs and cosine floor1e-6.

Rationale: directly target fast head overfit and unstable feature conditioning.
Risk: stronger regularization can underfit genuinely useful IR information.

### A02: Use the Clinical/Video Unit

Use `SmoothL1(mean20(prediction), label)` once per video/view. A batch contains
up to12 distinct patients/clinical events, not merely12 different lab IDs.
Retain all original frame/view groups, including short final batches. Weight
each video proportional to1/(number of clinical events in its patient xnumber
of videos assigned to that event), normalized to train mean1. No resampling
increases data size. LR1e-4/wd.003,40 epochs/patience8, two warmup epochs.

Rationale: reduce correlated gradients, patient dominance and pseudo-replication.
Risk: video-mean loss allows individual-frame predictions to vary more; all
interpretation and main metrics should remain video-level.

### A03: Hierarchical Scalar Fusion and a Strong RGB Anchor

Within each modality, independently project DINO768 and FaRL512 to64 and use
one scalar gate to blend them. A second scalar gate blends the resulting
RGB64 and IR64. Head hidden width stays32. Total trainable fusion/head count
is167,172, only642 above the original, so this does not rely on a large head.

Initialize RGB expert/projections/head from the existing RGB-only DINOv3-B +
FaRL scalar-gate checkpoint for the same analyte. Copy the expert initialization
to a separate IR expert; no shared trainable tensors. Set the modality RGB
prior to .95, not a claim of exactly reproducing RGB-only output. For an
RGB-only expert output, the final affine conversion preserves raw prediction:

```text
weight_new = weight_old * IQR_old / IQR_paired
bias_new   = bias_old * IQR_old / IQR_paired
             + (median_old - median_paired) / IQR_paired
```

No feature standardization is applied here, so copied expert projections still
receive the representation on which they were trained. The initial warm model
is evaluated on train/val as epoch0 and can be retained if all updates worsen
validation raw MAE. Then train video loss/patient balance with dropout .35,
lr5e-5/wd.003,30 epochs/patience8 and two warmup epochs.

Rationale: preserve a useful supervised RGB predictor, make encoder fusion and
modality fusion separate choices, and avoid immediately averaging IR into a
strong RGB representation with weight .5.

Important confound: the old RGB initializer saw more RGB-only training videos.
All paired patient split roles are verified unchanged; extra data are train
only. This is a legitimate supervised warm start, not a strict architecture-only
or equal-training-exposure ablation.

### A04: Tail- and Ordering-Aware Regression

Apply log1p followed by train median/IQR scaling to lactate, urea, bilirubin
and creatinine. Leave the other six targets in raw robust space. Pool predictions
in target space, then inverse-transform for raw-unit MAE/RMSE/r/R2. No target
or prediction clipping is performed. This changes the optimization geometry,
not the reported target.

Use five fixed-width bins between train raw1st/99th percentiles, assigning all
outside values to the end bins. Multiply base patient/event weights by capped
inverse-square-root bin-frequency factors (relative factor cap3). Unlike equal
frequency bins, these actually distinguish sparse and dense ranges. Add
`0.05 * mean softplus(-sign(y_i-y_j)*(pred_i-pred_j)/.2)` over unordered
within-batch video pairs at least .25 transformed training IQR apart. Start
RGB gate at .7, use feature calibration/dropout .4,lr1e-4,40 epochs/patience8.

Rationale: improve long tails and ordering rather than regress most patients
toward a common central value. Risks: more tail influence can worsen average
MAE, and ranking can improve r without calibration; report both.

### A05: Preserve Photometric Information

Retain original/hflip/center-crop views only. Use video loss/patient balance,
coordinate calibration, head dropout .4, IR dropout .1 and RGB prior .8.
LR1e-4/wd.003,40 epochs/patience8. Keep all20 selected frames and all videos.

This reduces train inputs from100 to60 per video/epoch, so it changes both the
photometric invariance and the update budget. It is deliberately a performance
bundle, not an isolated causal claim about brightness augmentation. Test remains
the original20 frames, unchanged.

### A06: Confidence in the Lab/Video Match

Use A02 with additional train factor
`w_time = .25 + .75*exp(-abs(match_delta_h)/6)`.
No observations get zero weight, no label is rematched and no test subgroup is
removed. The time interval is a training weight only, never an input feature.

Rationale: emphasize more temporally reliable supervision without losing data.
Risk: temporally distant examples may be clinically different rather than merely
noisier. The full, unchanged test set and match-time subgroups must still be
reported to detect that trade-off.

### A07: IR-Only Last-Block Adaptation

First run the same frozen warm stage as A03. Load its best validation model,
then unfreeze only the last IR DINO block/final norm and last IR FaRL
block/final norm/visual projection. All RGB parameters and IR earlier blocks
remain frozen. The14,573,568 trainable IR parameters plus167,172 fusion/head
parameters are trained for max20 epochs/patience6. Head LR1e-5 ->1e-6;
IR LR1e-6 ->1e-7; two warmup epochs; head WD.003,IR WD.01.

No SSL pretraining is shared across analytes: their original patient splits
differ, and a common learned encoder could leak another task's held-out patients.
This task-specific supervised adaptation avoids that cross-task problem.

RGB representations are reused from the exact paired cache. Native FFV1 IR
frames are decoded only for this phase; use four images/microbatch. Retain the
gradient graph for all20 predictions of one video, compute one pooled loss,
backpropagate one video at a time weighted by its share of the logical batch,
then clip/update once per up-to12-video batch. Computing separate four-frame
losses would be a different objective and is not used. LayerNorm is trainable;
encoders stay eval to disable backbone stochastic regularization. All IR
features are recomputed with updated parameters, including validation/test.

Use strict Inductor/no-CUDA-Graphs compilation for the online IR encoders;
there is no silent eager fallback. This has not been GPU-tested during the
design-only task. It is the most expensive/highest-risk protocol; frozen heads
are much cheaper. Checkpoints store only selected trainable IR tensors plus
fusion state, not duplicate copies of all four foundation models.

## Comparability and Statistical Reporting

The supervised paired cohort,20 frames, labels, split assignments and seeds
are fixed. Test-time frames/view/pooling/raw units are fixed. Some training
settings intentionally differ: A05 has3 views, A04 changes target space/weights,
A03/A07 use a larger-cohort RGB warm start, and A07 updates the IR representation.
Changes are recorded rather than mislabeled as isolated one-factor ablations.

After each protocol completes all ten analytes, update same-test metric bars,
training histories, fitted predicted/true plots, macro raw-MAE normalized by
the same baseline train IQR, and paired improvements. Confidence intervals
resample hospital patient IDs (all that patient's videos together),500 replicates,
not frames. They are exploratory percentile intervals, not multiplicity-adjusted
significance claims. Selection of winners uses validation raw MAE only; readers
have already seen baseline test results, so a future untouched cohort/fold is
needed for a confirmatory claim.

No new preprocessing-derived massive cache is created: six protocols reuse
the existing~1.25GiB feature banks; online IR streams original indexed frames.
No raw data, existing outputs or shared caches are modified by this suite.

## Sources and Limits

- [Official DINOv3 implementation](https://github.com/facebookresearch/dinov3):
  pretrained general-purpose visual representations; this repo supplies the
  exact pinned implementation/weights used in the baseline.
- [Official FaRL implementation](https://github.com/FacePerceiver/FaRL) and
  [CVPR2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.html):
  facial representation learning, not proof of grayscale-IR laboratory regression.

All intervention rationales above are inferences from these local results and
the current implementation. None of the sources guarantees large improvements
on these analytes. Weak visual signal, temporal label noise and limited patients
can cap performance even after correct implementation and careful optimization.
