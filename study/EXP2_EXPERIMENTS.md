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
`exp2_lab_multimodal` remains solely as a compatibility module for source-data
parsing; it is not a runnable experiment. Shared timestamp and plotting utilities
live in `study/common`.

Local ImageNet checkpoints belong in `study/common/pretrained_weights`. Download
and validate them with `python -m study.common.download_weights`. Raw videos,
`merged_lab_tests.csv`, and `merged_patient_info_*.csv` are external inputs;
checkpoints, result figures, splits, and source caches under experiment outputs
are not committed to Git.
