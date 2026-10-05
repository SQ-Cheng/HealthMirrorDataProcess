# Shared Study Utilities

## Native 224 Face Videos

Exp2, Exp6 and Exp8 default to validated lossless `face224.mkv` crops when the
raw-data production protocol exists. Explicit legacy reproduction remains
available with `HEALTHMIRROR_FACE_SOURCE=legacy128`.
See [FACE224_PROTOCOL.md](FACE224_PROTOCOL.md) for decoding, exclusions, cache
compatibility, retained splits, and the automatic screen rerun queue.

`time_alignment.py` and `plot_layout.py` are used by the current experiments.
`pretrained_weights/` holds local torchvision ImageNet checkpoints shared by
Exp2 and Exp4-6. Weight binaries are deliberately excluded from Git; the tracked
manifest records their source, size, and SHA-256 digest.

From the repository root, install the checkpoints with:

```bash
python -m study.common.download_weights
```
