# Exp2 History-Only Head32 Regression Ablation

This controlled ablation removes the face input and pretrained image backbone
from `exp2_face_history_head32_regression`. It retains the exact robust-scaled
raw-value labels, patient-disjoint split, prior-lab sequences, HistoryEncoder,
width-32 head, SmoothL1 objective, and optimizer schedule.

## Model

```text
Prior measurements: (value feature, time feature), shape (N, L, 2)
  -> Linear(2,16) -> SiLU
  -> Linear(16,16) -> LayerNorm(16) -> SiLU
  -> masked mean over history length
  -> Linear(16,32) -> LayerNorm(32) -> SiLU -> Dropout(0.25)
  -> Linear(32,1)
  -> predicted train-only robust-scaled raw lab value
```

Parameter counts are reported by the runner. There is no image tensor, frame
expansion, augmentation, pretrained weight, backbone, or fine-tuning stage.
Each labelled video contributes exactly one history sequence and one loss term.
Targets are oxyhemoglobin fraction, lactate, urea, total bilirubin, platelet count,
hemoglobin, A/a PO2 ratio, and creatinine. Labels, histories,
target scalers, and split assignments are byte-identical copies from the latest
face-plus-history experiment. Predictions are inverse-transformed and reported
in the original laboratory units.

Training preserves the reference schedule: stage 1 uses `2e-4` for up to 40
epochs and stage 2 uses `1e-5` for up to 60 epochs. There is no backbone to
unfreeze, so all 993 history/head parameters remain trainable in both stages.

## Run

```bash
bash study/exp2_history_only_head32_regression/launch_screen.sh --overwrite
```

The eight independent tasks are scheduled across four GPUs. Figures, including
a joint comparison with the EfficientNet-B0 face+history and face-only controls,
are generated after training finishes.
