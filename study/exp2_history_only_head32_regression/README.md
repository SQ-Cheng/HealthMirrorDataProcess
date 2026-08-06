# Exp2 History-Only Head32 Regression Ablation

This controlled ablation removes the face input and pretrained image backbone
from `exp2_face_history_head32_regression`. It retains the exact abnormal-score
labels, patient-disjoint split, prior-lab sequences, HistoryEncoder, width-32
head, SmoothL1 objective, and head-stage optimizer settings.

## Model

```text
Prior measurements: (value feature, time feature), shape (N, L, 2)
  -> Linear(2,16) -> SiLU
  -> Linear(16,16) -> LayerNorm(16) -> SiLU
  -> masked mean over history length
  -> Linear(16,32) -> LayerNorm(32) -> SiLU -> Dropout(0.25)
  -> Linear(32,1)
  -> predicted abnormal score
```

Parameter counts are reported by the runner. There is no image tensor, frame
expansion, augmentation, pretrained weight, backbone, or fine-tuning stage.
Each labelled video contributes exactly one history sequence and one loss term.

## Run

```bash
bash study/exp2_history_only_head32_regression/launch_screen.sh --overwrite
```

The two independent tasks run concurrently on two GPUs. Figures, including the
controlled comparison with both face+history backbones, are generated after
training finishes.
