# Exp2 Face Architecture Ablations

This independent directory tests lightweight alternatives to pretrained
EfficientNet-B0. It is not stored inside a head32 experiment. Three architectures
each train eight independent binary classifiers and eight raw-value regressors:
48 models in total. The targets are O2Hb fraction, lactate, urea, total bilirubin,
platelets, hemoglobin, A/a PO2 ratio, and creatinine.

## Models

All methods use the same full 224x224 RGB face frames and existing five views.
No new ROI, skin mask, alignment, temporal input, or lab-history input is added.

| Architecture | Input feature / structure | Trainable parameters |
| --- | --- | ---: |
| Color histogram + MLP | RGB/HSV/Lab; 16 normalized bins per channel; 144 -> 64 -> 32 -> 1 | 11,585 |
| Color statistics + MLP | 43 features; 43 -> 64 -> 32 -> 1 | 5,121 |
| Small CNN | Four stride-2 3x3 Conv/GroupNorm/SiLU blocks: 3 -> 16 -> 32 -> 64 -> 120; global average pool; 120 -> 32 -> 1 | 97,025 |

The two MLPs use LayerNorm, SiLU, and dropout=0.25 after their hidden layers.
The CNN uses eight-group GroupNorm and dropout=0.25 in its scalar head. Every
model is randomly initialized; there is no ImageNet backbone.

Statistics are mean, population SD, and 10/50/90 percentiles for each RGB and
Lab channel and HSV saturation/value. Hue uses saturation-weighted mean sine,
mean cosine, and resultant length, avoiding a discontinuity at red's hue wrap.
OpenCV's explicit float32 RGB-to-HSV/Lab conversions are used, never BGR or
8-bit Lab encodings. RGB is [0,1], hue is normalized from degrees, and Lab is
scaled as L/100, a/128, b/128. Histogram a/b values are mapped to [0,1].
RGB is clipped to [0,1] before color conversion. Histogram/statistical features
are spatially flip-invariant, but both original and flipped views remain in the
training schedule to retain the control's sample/view frequencies.
Feature means and SDs are fitted to training frames/views only, with an SD
floor of 1e-4; they are saved as model buffers and reused for validation/test.

## Matched Data And Objective

The experiment reuses the corrected native224, 12h, single patient-disjoint
split and train-only target median/IQR scalers. Its paired EfficientNet control
is `view_loss_12_distinct_labs/` in each face-only classification/regression
study, not the earlier grouped-view or five-fold experiments.

Each complete training batch is 12 different lab events x 20 frames x 1 view
= 240 images. The exact same distinct-lab sampler is reused. All five views
of each video are used once per epoch in different batches; the final partial
batch is retained. Original-only evaluation averages twenty frame predictions.

- Classification: weighted BCE on the mean of twenty frame probabilities;
  pos_weight = training negative/positive video count. Hb thresholds remain
  male <130 g/L, female <120 g/L. Decisions use probability >=0.5.
- Regression: SmoothL1(beta=0.5) on the mean of twenty robust-scaled predictions;
  results are inverse-transformed to raw laboratory units. No density weights.

The supervised loss unit is one video x one view, not one frame and not five
views pooled together. Models remain independent per target and task family.
Data, splits, frames, views, seeds, batch composition, and loss formulas match
the control. Architecture, pretraining, normalization, and optimization differ;
results cannot isolate architecture independently of all those differences.

## Training

Without a pretrained encoder to freeze, all parameters are trained jointly in
one stage. Head-only freezing would not be meaningful for the fixed-feature
MLPs or the randomly initialized CNN.

| Setting | Histogram / statistics MLP | Small CNN |
| --- | --- | --- |
| Initial learning rate | 1e-3 | 3e-4 |
| Cosine minimum | 1e-5 | 3e-6 |
| Epoch limit / patience | 80 / 12 | 80 / 12 |
| Optimizer / weight decay | AdamW / 1e-3 | AdamW / 1e-3 |
| Gradient clip / dropout | 1 / 0.25 | 1 / 0.25 |

Early stopping and checkpoints retain the control's criteria: validation bACC
for classification and raw-unit validation MAE for regression. Test data are
evaluated only after checkpoint selection. AMP is enabled; compilation is
disabled for these small models. Four GPU workers dynamically schedule jobs.

## Efficient Storage And Queue

The native FFV1 index is reused. A CPU-only, four-worker extraction runs once
after the control completes, caching histogram and statistical features for
all selected frames/views in float32. For the current 26,520-frame index the
two reusable, label-independent arrays total about 95 MiB. No decoded images,
new videos, or full spectral cubes are saved. Cache source/augmentation/shape
and OpenCV version are verified. CNN inputs stream the original FFV1 frames.

The monitor waits for the current grouped-view experiments and the already
queued twelve-distinct-lab experiments to finish through the existing dependency
chain. It allocates no GPU and extracts no features while waiting. An interrupted
predecessor is an error, not permission to start concurrent experiments.

```bash
python -m unittest study.exp2_face_architecture_ablation.test_experiment
python -m study.exp2_face_architecture_ablation.run --check-only
bash study/exp2_face_architecture_ablation/launch_screen.sh
screen -r exp2_face_architecture_ablation
```

Results: `outputs/<classification|regression>/<architecture>/`, each with
`runs/<target>/model.pt`, history, video predictions, metrics, and automatic
figures. Paired four-method comparisons are in `outputs/figures/`; CSV tables
are outside figure directories. Classification uses metric bars/confusion
matrices, regression uses metric bars and true-versus-predicted scatter plots
with fitted lines. All eight-target panels use four columns and two rows.

Global log: `logs/run.log`; each model has its own `runs/<target>/train.log`.
Only code/configuration/documentation are committed, not caches or results.
