# Laboratory variability around videos

CPU-only descriptive analysis of the eight current Exp2 laboratory targets.
It reads the latest main experiment's immutable clinical preparation and
does not train models, run inference, or change any ongoing experiment.

## Scope

The video pool is the union of eligible native224 twenty-frame main task
records: 1,391 videos from 563 patients at creation time. An eight-analyte
complete panel is not required. All cleaned canonical assay events for each
analyte are considered, not just the one selected nearest label.

For each original capture interval `[start,end]`, use the contiguous inclusive
window `[start-H,end+H]`, for H=6,12,24 hours. Thus the widest window spans
approximately 48 hours, not 24. Clip the window to the same hospitalization.
Use the current Session Timestamp/time-zone, alias, unit, specimen, censoring,
physical-range and duplicate-patient-timestamp policies without alteration.
Nothing is interpolated or resampled.

At least two independent report times are necessary to quantify variation.
Windows with zero or one event remain in the coverage table but have missing
change/range/SD values; they are never treated as stable zero-change windows.

## Measurements

- Signed net change: last value minus first value in chronological order.
- Absolute net change: absolute value of that difference.
- Total observed excursion: maximum minus minimum, which also captures rises
  and falls that can cancel in net change.
- Sample SD and within-window IQR.
- Number of independent lab events and actual first-to-last observed time span.
- Maximum deviation from the nearest laboratory value chosen by the existing
  matching rule.
- Threshold crossing: both normal and abnormal measurements are present.
  This does not imply that widening the window changes the selected nearest
  label. Sex-dependent hemoglobin thresholds reuse the current source policy.

Percentage-unit assays report absolute differences in percentage points, not
relative percent changes. Cross-analyte magnitudes divide by each model's
existing training-set IQR; no scaler is refitted.

## Patient weights and matched cohorts

Each patient contributes one median of their eligible video-window magnitude
per analyte/window. Threshold-crossing rates first average windows within
patient and then average patients. Confidence intervals resample patients,
not correlated video windows: 2,000 percentile bootstraps with seed 20261008.

Three cohorts are exported:

- `all_eligible`: all windows with at least two events, separately at each H.
- `common_6h_videos`: exactly the same analyte/video pairs across all three
  windows, requiring at least two events already at +/-6h. This avoids
  confounding the paired comparison with changing sample composition.
- `bracketing_video`: at least one event before capture start and another
  after capture end. Ordinary continuous windows may contain only one-sided
  observations; this subgroup is saved separately rather than silently
  excluding such windows.

The eight-panel plots use four columns and two rows. Boxplots show per-patient
medians, IQR boxes, and fifth-to-ninety-fifth-percentile whiskers; full extrema
remain in the machine-readable summary. Range/SD depend on sampling count and
time span. Overlapping video windows may reuse laboratory events; they are
not independent extra clinical observations.

## Run

```bash
python -m study.exp2_lab_video_window_variability.analyze
python -m study.exp2_lab_video_window_variability.analyze --plots-only
python -m unittest study.exp2_lab_video_window_variability.test_analysis
```

CSV tables and the source-fingerprint manifest go in `outputs/`. PNG/PDF
charts go in `outputs/figures/`. Generated patient data are not Git-tracked.
