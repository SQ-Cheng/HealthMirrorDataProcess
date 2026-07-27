# Exp2 Data-Building Compatibility Module

The original multimodal model experiment was invalidated by the historical
eight-hour lab timestamp error and has been removed together with its outputs and
checkpoints.

`config.py` and `build_dataset.py` remain only because the corrected 24-hour RGB data
builder imports their patient-information, lab parsing, timezone, and label helpers.
This directory is not a supported experiment entry point.
