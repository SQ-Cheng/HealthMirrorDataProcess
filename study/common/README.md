# Shared Study Utilities

## Native 224 Face Videos

Migrated face-only Exp2 and Exp6 regression accept native `face224.mkv` only.
Legacy reproduction remains only for protected studies with no native-224 main.
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
