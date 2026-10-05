# MediaPipe: Direct Resize Versus Alignment

The active comparison processes the same 20 complete original recordings from
`/root/shared/HealthMirrorRawData`. Original files are not modified. InsightFace
is not used. Earlier detector comparison results remain in `outputs/pilot20/`
as historical references only.

## Shared Pipeline

1. Decode original MJPEG frames, preserving source indices and timestamps.
   Require contiguous IDs, increasing timestamps and exact frame counts.
2. Detect with MediaPipe BlazeFace short-range, confidence 0.5 and NMS 0.3.
   Associate the primary face across frames and reject ambiguous candidates.
3. Extend the box upward by **20% of its original height**. Do not add lateral
   or bottom margins or convert the rectangular box to a square.
4. Apply four shared 1D Kalman filters to normalized box coordinates: process
   noise 0.01, measurement noise 0.5, initial error 1, reference interval 1/30 s.
   Process noise scales with squared elapsed source-time ratio, not processing
   wall-clock time. Both branches receive exactly the same stabilized box.
5. Clamp to the original image. Direct crop uses bilinear resize to 224 x 224.

## Alignment

MediaPipe's two eye keypoints define the in-plane roll angle. An additional
Kalman filter stabilizes this angle: process noise 0.5 degrees squared,
measurement noise 4 degrees squared and initial error 4. Rotate about the
shared box center, then crop and resize through one composed affine transform
of the original image, using bilinear interpolation.

This is eye-line roll alignment, not template-based face scaling, recognition
or 3D frontalization. Crop extent and output size are shared. Rotation can
sample beyond source boundaries: padding is recorded and aligned frames with
over 5% padding are ineligible. Invalid eye geometry is also ineligible.

Detection holds up to 200 ms are for viewing only, never training. Longer
failures retain black placeholders at the original indices. Filter state resets
after a 500 ms detection gap. Downstream users must apply each branch's
eligibility mask. Successful detection does not certify patient identity.

## Storage And Verification

Both outputs use FFV1 level 3, `bgr0` Matroska, without lossy recompression or
chroma subsampling. Full timestamp precision remains in sidecars; video PTS has
millisecond precision. Every lossless video is decoded and checked against its
ordered pixel digest, frame count and source elapsed timestamps. Source video
and timestamp hashes are checked before and after processing. Clinical matching
continues to use validated **Session Timestamp**, not recorder absolute time.

## Running

Dependencies are installed in the existing `healthmirrorenv`, retaining its
NumPy and OpenCV. The active dependency list and downloader do not use
InsightFace. No separate environment is created.

```bash
/root/miniconda3/envs/healthmirrorenv/bin/python -m \
  preprocess.face_crop_comparison.download_models
bash preprocess/face_crop_comparison/run_pilot20.sh
```

Add `--resume` to reuse verified outputs after source/configuration checks.

## Outputs

Generated files are Git-ignored in `outputs/mediapipe_kalman_alignment20/`:

- `index.html`: 20 playable source/direct/aligned comparisons.
- `figures/contact_sheet_20.png`: overview of all paired crops.
- `figures/*_frame_*.png`: snapshots at 10%, 50% and 90% of each recording.
- `videos/<video_id>/direct_224.mkv`, `aligned_224.mkv`: lossless face sequences.
- `videos/<video_id>/*.mkv.ts`: unchanged source timestamps.
- `videos/<video_id>/frame_records.csv`: boxes, angles, affine matrices,
  eligibility masks, padding and enlargement flags.
- `videos/<video_id>/comparison.mp4`: viewing preview only, not training data.
- `videos/<video_id>/metadata.json`: provenance and lossless checks.
- `tables/`: selection, summaries and quality-review flags.
- `manifest.json`, `REPORT.md`: protocol and verification results.

The difficult recording `mirror6_patient_000454` needs manual review: the face
is largely outside the source image and MediaPipe can detect the torso instead.
Alignment does not repair incorrect or incomplete face detection.

## 100-Video Quality Audit

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
  preprocess.face_crop_comparison.run_quality_audit --workers 8
```

The audit uses 20 distinct patients per mirror (100 videos/patients total), with
seed 20261005, and does not require lab availability. All frames are processed.
Selection failures and frame rejection reasons are exported rather than hidden.

The acceptance score threshold is 0.80. Detector reporting is lowered to 0.10
to expose low-scoring candidates for statistics. Scores below that reporting
floor are unavailable; no detection/unselected candidates have missing scores,
not zeros. Confidence means the uncalibrated BlazeFace detection score, not
probability that the entire face is visible.

Before updating Kalman or aligning, expand the bbox upward by 20% of its original
height and reject it only when more than 10% of the expanded rectangle's area
lies outside the image. Compute `(expanded area - image intersection area) /
expanded area`; exactly 10% is allowed. Accepted boxes are clipped to the image
before cropping, without synthetic padding. Bbox maxima are exclusive boundaries.
There is no additional 2 px margin and no rejection based on contour/keypoint
coverage, mesh matching or Kalman lag. No FaceLandmarker inference is required.
Low scores and ambiguous/failed primary-face associations remain rejected.
The aligned branch still checks eye geometry and source-border padding.

`outputs/quality100/` separates human-facing `figures/`, `index.html`, `REPORT.md`
from machine-readable `tables/` and `manifest.json`. Per-frame tables record
confidence, quality flags, raw and filtered crop geometry and branch eligibility.
Plots show score histograms/CDFs, final outcomes and per-video retention. Scores
are frame-weighted; unequal recording lengths are also reported per video.

`videos/*.mp4` are viewing-only synchronized three-column comparisons: left is
the original video with red detector boxes, orange top-expanded boxes and cyan
Kalman-filtered boxes; middle is quality-gated Kalman + alignment; right is
**no Kalman, no alignment**, with the same detector,
raw quality gate, top margin and bilinear 224 x 224 resize. Both retain every
source index; rejected frames are black. The unfiltered branch has no
contour-containment check in either branch. These previews
are lossy, not new training caches. Original timestamp sidecars are copied.

To redraw these previews from saved frame geometry, without detection or
landmark inference:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
  preprocess.face_crop_comparison.render_quality_previews --workers 8
```

To rerun detection with the current rule on exactly the same selected videos
and overwrite the previous statistics and previews, use
`run_quality_audit --overwrite --workers 8` with the module command above.

To summarize retained-frame confidence, source-boundary clipping, actual ROI
dimensions and resize factors from saved CSVs only (no video decoding):

```bash
/root/miniconda3/envs/healthmirrorenv/bin/python -m \
  preprocess.face_crop_comparison.analyze_retained_frames
```

This writes `RETAINED_FRAME_QUALITY.md`, `figures/retained_frame_distributions.*`
and `tables/retained_frame_{quantiles,bins,per_video}.csv`. Distributions are
frame-weighted. Outside-area fraction is not a measure of missing facial anatomy;
ROI/source area describes framing, and any-axis enlargement uses the larger of
the horizontal and vertical resize factors.

## Full-Resolution Production Videos

```bash
bash preprocess/face_crop_comparison/launch_full_dataset_screen.sh
screen -r face224_full_c075
```

For a full replacement of existing generated outputs, pass `--overwrite` to
the launcher. This only deletes the explicitly named generated `face224` files,
not original videos, timestamp files or patient metadata.

The production run processes every `raw_video.avi` in
`/root/shared/HealthMirrorRawData/mirror*_data/patient_*`, independently of lab
availability. Each original session directory receives:

- `face224.mkv`: 224 x 224 RGB, FFV1 level 3 lossless encoding, GOP 1.
- `face224_frames.csv`: retained output index -> original frame index, precise
  source elapsed/recorder time, validated session-canonical time and encoded PTS.
- `face224_audit.csv`: every original frame, geometry and exclusion reason.
- `face224_metadata.json`: source hashes, protocol, counts and pixel-exact check.

Confidence acceptance is 0.75; the detector reporting floor remains 0.10 to
preserve the reviewed face-association behavior. All other geometry/quality
settings match the current audit: top expansion 20%, raw-box outside area <=10%,
Kalman bbox filtering and **no alignment** in production. Eye geometry and
alignment padding are not exclusion criteria for this non-aligned protocol.
Only fresh eligible direct crops are encoded. No black placeholders,
duplicate/held frames or temporal resampling are used. Crop and spatial resize
are applied directly to the original source, followed by one lossless
encoding; each completed video is decoded to verify exact pixel preservation.

FFV1 intra-only MKV supports independent-frame decoding using PyAV/FFmpeg/OpenCV,
without storing millions of separate images or uncompressed tensors. Training
must select **output indices from `face224_frames.csv`**; do not infer original
indices or times from nominal FPS. PTS preserve source elapsed-time gaps, but
container time resolution is approximately 1 ms; CSV timestamps retain full
precision. Use session-canonical timestamps for clinical matching, not recorder
wall-clock values. Missing/invalid session metadata does not discard the face
video, but canonical timestamps are left missing and the metadata error is
explicit, so that clinical matching cannot silently fall back to recorder time.

Aggregate progress/status live in `HealthMirrorRawData/_face224_processing/`.
Sessions with no accepted frames have CSVs and `no_valid_frames` status, but no
fake playable video. Missing/empty/unreadable sources and failures are accounted
for in `index.csv`; failed sessions do not stop processing other videos. Original
videos and timestamps are never rewritten. The launcher resumes completed
matching outputs rather than encoding them again; interrupted partial files
are replaced on the next run. Log:
`preprocess/face_crop_comparison/logs/full_dataset/run.log`.
