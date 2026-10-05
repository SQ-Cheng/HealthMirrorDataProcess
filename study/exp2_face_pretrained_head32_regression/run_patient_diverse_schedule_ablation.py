"""Shared schedule for the retained native-224 patient-diverse protocols."""

STAGE_CONFIG = {
    "head_learning_rate": 1e-4,
    "head_max_epochs": 30,
    "head_patience": 8,
    "finetune_learning_rate": 3e-6,
    "finetune_min_learning_rate": 1e-7,
    "finetune_max_epochs": 40,
    "finetune_patience": 8,
}
