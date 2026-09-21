# Exp2 face-only Head32 classification

True binary counterpart of `exp2_face_pretrained_head32_regression`. Data,
patient splits, 20-frame policy, five views, EfficientNet-B0 Head32 model, and
two-stage optimization match the regression experiment. The only learning-task
change is one-logit binary classification with weighted BCE.

Troponin I is replaced by total bilirubin >21 umol/L.
