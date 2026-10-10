# Attrition when tightening laboratory matching from 24h to 12h

CPU-only descriptive analysis of the current eight-target native224 face
classification/regression main cohort. No model training, inference, re-split,
or modification of the active training queue is performed.

## Definitions and verification

The unit of model-data loss is a **video-analyte matched pair**. Lost pairs
have `12 < match_delta_h <= 24`, where distance is to the original capture
interval, not its midpoint. Retained pairs have distance <=12h. Every 24h
nearest assay is independently reconstructed from the canonical same-admission
time series. Retained pairs must choose exactly the same assay at 12h; lost
pairs must have no eligible 12h candidate. Source-table and record hashes are
checked before and after analysis.

The canonical laboratory filtering, aliases, unit conversion, timestamp
normalization, duplicate-event handling, clinical thresholds, and sex-dependent
hemoglobin thresholds are the same as the current model data. Normality means
the saved clinical label, not a model output.

Independent assay events are deduplicated by target, hospital ID, and report
timestamp. Some events lose one video match while retaining another. These
`shared` events are not called entirely lost assays. The CSV explicitly
separates `retained_only`, `shared`, and `lost_only`. The original lab table
is never changed by tightening a model's matching criterion.

## Timeline and bias audit

Admission/discharge bounds come from the same validated hospitalization as
each video. Surgery is the first valid recorded **CABG** in that hospitalization,
using the existing recovery metadata pipeline. Unrelated procedures are not
substituted. Latest-table CABG metadata is reused only after fingerprint checks.
Missing CABG times remain explicit; they do not mean the patient had no surgery.

At both lab-report and video time, export:

- Days from admission, days to discharge, and normalized hospitalization position.
- Hours relative to CABG end and distance to the CABG start/end interval.
- Nearest landmark: admission, CABG, or discharge; unknown and ties are explicit.
- Preoperative, intraoperative, and postoperative 0-1d / 1-3d / 3-7d / >7d phases.
- Lab-before-video versus lab-after-video matching direction.

Lab timing is the supplied report timestamp, not an independently documented
blood-collection time. An intraoperative report classification is not proof
of the exact blood-draw time.

Other tables summarize loss counts, patients with any loss versus patients
lost entirely, per-mirror denominators, fixed train/validation/test attrition,
sex distribution, raw-value quartiles, length of stay, and same-analyte assay
cadence within the admission. Normality is also patient-weighted, with one
average per patient/subset and patient-bootstrap confidence intervals. A
within-patient difference is exported for patients represented in both subsets.
No p-value or causal device/clinical conclusion is implied.

Potential timing inconsistencies are exported for review: video midpoints
inside the recorded CABG interval and recorded CABG durations above 24h.
These flags do not automatically alter eligibility or the running experiments.

## Run and output

```bash
python -m study.exp2_match_window_attrition_analysis.analyze
python -m study.exp2_match_window_attrition_analysis.analyze --plots-only
python -m unittest study.exp2_match_window_attrition_analysis.test_analysis
```

Machine-readable CSVs and a source/definition manifest are under `outputs/`.
Human-readable PNG/PDF charts are under `outputs/figures/`. Eight-analyte
panels use four columns by two rows. The primary charts cover overall
attrition, normality, raw-value distributions, surgical/hospitalization phase,
mirror effects, fixed split effects, matching direction, and assay cadence.

Pooled counts across analytes count assay-video pairs, not independent patients.
Unique video and patient totals are separately provided in the manifest.
Generated patient data and results are not Git-tracked.
