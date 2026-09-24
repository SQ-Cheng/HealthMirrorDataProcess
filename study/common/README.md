# Shared Study Utilities

`time_alignment.py` and `plot_layout.py` are used by the current experiments.
`pretrained_weights/` holds local torchvision ImageNet checkpoints shared by
Exp2 and Exp4-6. Weight binaries are deliberately excluded from Git; the tracked
manifest records their source, size, and SHA-256 digest.

From the repository root, install the checkpoints with:

```bash
python -m study.common.download_weights
```
