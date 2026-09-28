# Selected Exp2/Exp3 five-fold validation

Launch four specified configurations sequentially with four dynamically
scheduled GPUs:

```bash
bash study/common/launch_selected_5fold_screen.sh
```

Only these protocols run: face-only EfficientNet-B0 regression, its
patient-diverse 30/40 ablation, video R3D-18 regression, and face-only
EfficientNet-B0 classification. All eight existing targets are retained.
Other ablations and architectures are excluded.

For each target, the same 5-fold patient assignment is used by every protocol.
Patient classes are stratified first; 256 seeded assignments are assessed for
video/patient counts, positive rates, raw-value distribution and abnormal-score
distribution. Each outer fold is test once, the next fold is validation, and
the other three folds train. Original records are preserved; regression
median/IQR is fitted on that fold's training videos only. Classification
positive weight is recalculated from that fold's training videos.

The fold records, balancing audit, scalers and raw-distribution figure are at
`study/exp2_face_pretrained_head32_regression/outputs/5fold/splits/`.
Each selected experiment stores `fold_0` through `fold_4`, task checkpoints,
histories, per-fold figures, pooled out-of-fold predictions, mean/SD metrics,
and a `figures/` summary below its own `outputs/5fold/` directory. The
patient-diverse variant uses
`outputs/ablations/patient_diverse_schedule_30_40/5fold/`.
The three regression protocols additionally receive a common comparison at
`study/exp2_face_pretrained_head32_regression/outputs/5fold/figures/comparison/`.
No original experiment outputs are overwritten.
