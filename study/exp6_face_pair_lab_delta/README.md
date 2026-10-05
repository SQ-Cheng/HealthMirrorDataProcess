# Native-224 Paired-Face Laboratory Change Regression

Nine independent models predict the difference between two distinct matched
laboratory measurements of the same patient: oxyhemoglobin fraction, lactate,
urea, troponin, platelet count, hemoglobin, A/a PO2 ratio, creatinine and total bilirubin.

The two faces share an ImageNet-pretrained EfficientNet-B0 encoder. The head
combines feature differences and has 32 hidden features. Twenty selected frames
per video form paired inputs, expanded to the same five synchronized views:
original, flip, center crop, brightness and contrast. Evaluation averages the
twenty original-view pair predictions.

Clinical pairing, canonical timestamps, original capture intervals, patient
splits and nearest-assay labels remain fixed. Target differences are robust-scaled
using training rows only. Training uses SmoothL1(beta=0.5), AdamW (wd=1e-4),
head lr=2e-4 / 40 epochs / patience 10, then full-backbone fine-tuning
lr=1e-5 / 60 epochs / patience 12, cosine floor=1e-6.

## Current Outputs

- Main 24h: `outputs/face224/`.
- Windows: `outputs/ablations/lab_match_6h_face224/` and
  `outputs/ablations/lab_match_12h_face224/`.
- Matching-window comparisons use the native-224 main, not deleted 128 results.

Each run saves model.pt, histories, metrics and pair_predictions.csv.
Figures are produced automatically.

```bash
bash study/common/launch_face224_reruns_screen.sh
# A single prepared protocol:
bash study/exp6_face_pair_lab_delta/launch_screen.sh --hours 24
```

The queue validates saved input hashes, skips complete protocols, and validates
individually completed tasks before reusing them. Native regression rejects 128
inputs. Previous 128 regression models, ablations and their launchers are removed.

## Protected 128-Only Classification Inputs

Root `outputs/task_records/`, scalers, run_index, source_data and the nine
root `runs/*/pair_predictions.csv` files are immutable clinical reference inputs
for the separate, protected 128-only direction-classification experiment.
Root `cache/frames20/` is its required shared index. No old regression model
weights or rendered frame caches remain in these reference locations.
