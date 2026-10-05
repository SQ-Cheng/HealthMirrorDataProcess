# Native-224 Face Protocol

Raw sources in `/root/shared/HealthMirrorRawData` remain untouched.
The production pipeline detects with MediaPipe (confidence >=0.75), expands
the upper box by 20%, rejects boxes with more than 10% outside the original
image, applies Kalman filtering without alignment, and resizes the accepted
RGB face once to 224x224.

Each original directory contains `face224.mkv` (FFV1 level 3, BGR0, GOP 1),
`face224_frames.csv`, frame-quality audit and metadata. Rejected frames are
not encoded. The sidecar retains original source frame indices, recorder
elapsed times, canonical session times and encoded PTS. Training timestamps
must never be reconstructed from the shortened processed video.

## Efficient Reading

`face_video.py` validates crop protocol, canonical session identity, source
signatures and lossless verification. Shared compact byte-offset indexes live
in `common/cache/face224_20frame/`. Independently coded packets are decoded
on demand with bounded decoder/file caches; decoded frames are not persisted.
Twenty-frame selection uses nonadjacent original source indices.
Original/hflip/brightness/contrast views do not upsample already-224 inputs;
the center-crop augmentation alone resizes the crop back to input size.

## Experiments

Migrated face-only Exp2 regression/classification and Exp6 regression accept
native FFV1 inputs only. Their old launchers, models, results, ablations and
image indexes are retired. Saved native clinical records/scalers/splits are
the queue inputs; no legacy baseline is read by the controller or waiting
native ablation/five-fold monitors.

```bash
bash study/common/launch_face224_reruns_screen.sh
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_224_screen.sh
bash study/common/launch_face224_12h_5fold_screen.sh
```

Completed tasks are reused after identity/scaler validation. Interrupted
unfinished tasks restart without changing split or settings. All screens detach.
See [FACE224_12H_5FOLD.md](FACE224_12H_5FOLD.md).

## Protected Exceptions

Studies whose main experiment exists only at 128 resolution are left intact,
including face+history, video Exp3, Exp4/Exp5 and Exp6 direction classification.
Exp8 also remains protected until its native main results actually exist.
The minimal shared MJPEG decoder/index reader and original external 128 data
are necessary for those studies, not fallback support for migrated entry points.

Clinical reference tables under face regression `outputs/20frame/` and
`outputs/5fold/splits/`, plus Exp6 root pair records/predictions and its
`cache/frames20/` index, are retained to avoid changing the protected studies'
clinical labels, split contracts or comparisons. They contain no old face
regression weights or decoded pixels.

Deletion audits are in `common/outputs/retire_128/`; see
[RETIRE_128_POLICY.md](RETIRE_128_POLICY.md).
