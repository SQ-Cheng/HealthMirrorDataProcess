# Current Exp2 Experiments

All six predictive experiments use EfficientNet-B0 where an image is present,
20 nonadjacent frames per video, five training views, patient-disjoint splits,
and independent width-32 heads for each lab target. The classification trio
uses the shared `exp2_binary_classification_common` engine. The regression trio
uses train-only robust scaling of raw lab values.

| Input | Regression | Binary classification |
| --- | --- | --- |
| Face + prior labs | `exp2_face_history_head32_regression` | `exp2_face_history_head32_classification` |
| Face only | `exp2_face_pretrained_head32_regression` | `exp2_face_pretrained_head32_classification` |
| Prior labs only | `exp2_history_only_head32_regression` | `exp2_history_only_head32_classification` |

`exp2_lab_longitudinal_statistics` is the retained lab-only descriptive study.
`exp2_face_shared_backbone_head32_regression` is a separate ten-analyte regression
experiment using one shared EfficientNet-B0 and ten independent head32 modules.
Its common patient split prevents cross-task encoder leakage; original main
per-task splits are not reused. All code, logs, outputs and checkpoints are owned
by that directory, while common frame indexes and main source records are reused.
`exp2_face_ssl_head32_regression` pretrains one EfficientNet-B0 using train-only
all-frame BYOL and the five existing views, then trains ten independently parameterized
head32 regressors. It uses a common distribution-searched patient split across all
stages to prevent SSL/other-task leakage into validation or test patients.
`exp2_face_architecture_ablation` is an independent color-histogram MLP,
color-statistics MLP, and approximately 100k-parameter CNN comparison, using
native224/12h, the matched single split, and twenty-frame view-level losses.
It is separate from the six primary predictive experiments above.
`exp2_face_dinov3_frozen` retains its original 12h regression/classification
view-loss runs and current ten-task 24h frame-loss regression head32/head64
runs. Frozen EN-B0 head32/head64 controls use the latter identical clinical
protocol and frozen-DINO optimizer settings; paired five-model plots are automatic.
`exp2_face_dinov3_farl_regression` freezes DINOv3-S and local FaRL ViT-B/16,
then fits direct-concat and projected-vector-gate heads on the exact current
24h/20frame/five-view cohort. Each analyte/variant has an independent head32;
both encoders have separate correct input normalization and compact reusable caches.
`exp2_lab_multimodal` remains solely as a compatibility module for source-data
parsing; it is not a runnable experiment. Shared timestamp and plotting utilities
live in `study/common`.

Local ImageNet checkpoints belong in `study/common/pretrained_weights`. Download
and validate them with `python -m study.common.download_weights`. Raw videos,
`merged_lab_tests.csv`, and `merged_patient_info_*.csv` are external inputs;
checkpoints, result figures, splits, and source caches under experiment outputs
are not committed to Git.

The replaced Exp1 is `study/exp1_roi_color_regression`: native-pixel 41-dimensional
color features for forehead, cheeks and lips, video-level averaging, MLP and
patient-group-CV ridge with common ROI cohorts. Preserved ECG human annotations
under `study/common/annotations` are local data, not source-code repository inputs.
