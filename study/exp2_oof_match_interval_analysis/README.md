# Exp2: OOF performance by lab/video matching interval

This post-hoc analysis uses the completed 24-hour, five-fold EfficientNet-B0
face-only regression experiment at
`../exp2_face_pretrained_head32_regression/outputs/5fold/`. It **does not train
or run inference**. Each held-out video prediction is joined one-to-one with
its fold-specific test split to recover the absolute video/lab time gap.

Run from the repository root:

```bash
python -m study.exp2_oof_match_interval_analysis.analyze
```

The bins are `[0,3)`, `[3,6)`, `[6,12)`, `[12,24]` hours. Within each target
and bin, the script pools held-out predictions from all five folds, then
computes Pearson r, MAE, R2, and sample target SD (`ddof=1`) in original lab
units. Undefined statistics for tiny or constant bins are reported as missing.
`n_videos` counts video/lab predictions, not independent patients; the CSV also
reports `n_patients`. One patient may contribute multiple videos. Results are
in `outputs/metrics_by_time_bin.csv`; `outputs/oof_with_match_intervals.csv`
provides the per-video audit, and `outputs/figures/` contains one 4-by-2 panel
chart per statistic plus bin counts. Source file hashes and conventions are in
`outputs/analysis_manifest.json`.

## Within-bin-centered correlation

Run `python -m study.exp2_oof_match_interval_analysis.within_bin_centering`
to evaluate 6-hour and 12-hour subsets of the **same 24-hour five-fold OOF
predictions**, without training or additional inference. The 6-hour subset
uses `[0,3)` and `[3,6)` hours; the 12-hour subset additionally uses `[6,12)`.
For each analyte and time bin, true and predicted values are centered by
their respective bin means. The centered samples are then pooled across bins
and folds, and Pearson r is computed once per analyte/window. Means are
estimated from all held-out predictions in each bin, not separately by fold.

`outputs/within_bin_centered_pearson_r.csv` gives centered and uncentered r,
video/patient counts, and their difference. `outputs/within_bin_centered_oof.csv`
contains the per-video centered values and bin means. The paired-bar figure is
`outputs/figures/within_bin_centered_pearson_r.png`. This removes differences
in the *mean* between time bins; it does not remove patient effects or
establish that the model detects individual-level physiology independently of
other confounders.
