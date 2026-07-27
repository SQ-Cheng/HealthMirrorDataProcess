# Exp2 24-Hour RGB Data Source

This directory is now a data-preparation compatibility package, not a retained model
experiment.

`outputs_aug20_24h/` is the canonical corrected source for the retained all-frame
experiments. It contains the 24-hour lab/video matches, Asia/Shanghai time correction,
per-target hemoglobin conflict audit, and video identity table. Decoded PNG frames,
the duplicate 20-frame NPZ, old predictions, and old checkpoints were removed because
the retained experiments stream source MJPEG frames through the compact byte-offset
index.
