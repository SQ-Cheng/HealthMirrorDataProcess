# Exp2 face + history Head32 classification

True binary counterpart of `exp2_face_history_head32_regression`. It reuses the
same corrected video/lab matching, patient-disjoint split, 20 nonadjacent RGB
frames, five deterministic training views, EfficientNet-B0 backbone, historical
lab encoder, Head32, and two-stage optimization.

The model emits one logit and is optimized with `BCEWithLogitsLoss`. Training
`pos_weight` is computed as negative/positive training videos. The target set is
the regression target set with Troponin I replaced by total bilirubin >21 umol/L.

The three controlled classification experiments are launched together:

```bash
bash study/exp2_face_history_head32_classification/launch_screen.sh
```
