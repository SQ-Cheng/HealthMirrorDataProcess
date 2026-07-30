# Exp2 Experiment Registry

The directory names below are stable compatibility paths. Canonical IDs distinguish
task type and augmentation without moving paths used by the active training process.
New experiments must automatically generate their validated result figures after all
training jobs finish successfully.

| Canonical ID | Compatibility path | Status |
| --- | --- | --- |
| `exp2_binary_20frame_head32_views5` | `exp2_face_pretrained_head32` | Active five-view binary experiment |
| `exp2_binary_allframes_head32_views3` | `exp2_face_pretrained_allframes_head32_views3` | Completed primary binary experiment |
| `exp2_regression_20frame_head32_single_task` | `exp2_face_pretrained_head32_regression` | Completed distribution-balanced single-task regression |
| `exp2_regression_allframes_head32_multitask` | `exp2_face_pretrained_allframes_head32_multitask_regression` | Active multi-output regression |

Shared retained assets:

- `exp2_face_only/outputs_aug20_24h`: corrected 24-hour label/video source
- `exp2_face_pretrained/pretrained_weights`: verified ImageNet checkpoints
- `exp2_face_pretrained_head32_regression/outputs/frame_index`: current compact
  deterministic 20-frame MJPEG byte-offset index

The original multimodal experiment, initial grayscale face models, 96-pixel Aug20
models, obsolete monitors, and their generated results have been deleted.

The active multi-output experiment currently has a known split limitation: its seven
PCO2-low positive videos are allocated train/validation/test as 2/4/1. It remains
registered as active rather than completed; its running process was not changed by
the cleanup.
