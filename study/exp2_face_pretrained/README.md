# Shared Exp2 ImageNet Weights

This compatibility directory is not a runnable experiment. The obsolete 20-frame
pretrained experiment and its results were removed.

`pretrained_weights/` is retained at its original path because the active all-frame
experiments reference it. `manifest.json` records the official torchvision source,
size, and SHA-256 digest for each checkpoint. The three local weight files have been
verified against that manifest.

Regenerate or revalidate them with:

```bash
python -m study.exp2_face_pretrained.download_weights
```
