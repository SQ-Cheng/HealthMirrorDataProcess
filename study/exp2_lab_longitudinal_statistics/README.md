# Exp2 Inpatient Lab Longitudinal Statistics

The primary trajectory analysis is independent of video data. It reads only
`merged_lab_tests.csv` and describes how numeric laboratory measurements change
during hospitalization.

## Full Laboratory / Video24h Availability Audit

An independent incremental audit ranks every field in the current merged table,
without rerunning the trajectory or surgery statistics:

```bash
HEALTHMIRROR_FACE_SOURCE=face224 python -m study.exp2_lab_longitudinal_statistics.video24h_matching
```

- Human-readable full descending list: `outputs/reports/VIDEO24H_MATCHING.md`.
- Canonical analyte/specimen ranking, exact source-field ranking, same-family
  membership, conversion audit and per-video matches: `outputs/tables/video24h_matching/`.
- PNG ranking pages: `outputs/figures/video24h_matching/`.
- Provenance and freshness: `outputs/metadata/video24h_matching_manifest.json`.

Matching uses validated Session Timestamp, the original capture interval,
same-patient/same-admission reports and nearest <=24h selection. Both raw-session
matches and native224/twenty-nonadjacent-frame eligibility are reported, without
new pixel/video caches. Exact numeric counts exclude censored, ordinal and range
strings; additional any-result coverage includes qualitative reports. Existing
unit/alias rules are reused, but blood-gas arterial/venous/unspecified specimens
remain distinct. Family-level union counts do not pool physiologically different
targets. Component and backup tables are inventoried, not added to the merged
table again. This broad descriptive audit does not invent per-analyte physical
ranges and is not a replacement for each training task's stricter policies.
The prior trajectory outputs retain their original source/version; this audit
explicitly records whether those outputs use the current laboratory table.

Read-only frame-budget sensitivity (same matches, no crop or training changes):

```bash
HEALTHMIRROR_FACE_SOURCE=face224 python -m study.exp2_lab_longitudinal_statistics.frame_budget_audit
```

`outputs/reports/FRAME_BUDGET_AUDIT.md` distinguishes fixed-N sampling,
minimum-N all-frame admission, and adaptive sampling capped at twenty frames.
It tests the existing uniform policy, a gap-constrained fallback, and adjacent
source frames separately. Detailed tables and decode checks live in
`outputs/tables/video24h_matching/frame_budget/`; the PNG tradeoff chart is
`outputs/figures/frame_budget_tradeoff.png`. Main ten-task baselines are checked
exactly; malformed timestamps, missing sources and zero-quality crops are not
silently recovered by lowering the frame threshold.

The analysis unit is a hospital episode, defined by hospital patient ID,
admission time, and discharge time. Test variables are separated by normalized
test name and unit. Rows outside the admission-discharge interval, nonnumeric
results, and censored values such as `<0.01` are audited and excluded from
inferential calculations.

Clinically equivalent aliases are harmonized before variable IDs are assigned.
This includes laboratory and blood-gas hemoglobin, glucose, lactate, troponin I,
creatinine, hematocrit, co-oximetry fractions, electrolytes, oxygen-derived
quantities, and exact source-system aliases. Explicit unit conversions are
applied before merging. Temperature-corrected values, patient-condition versus
standard-condition P50, total versus ionized magnesium, and blood versus urine
measurements remain distinct. The applied rules, source labels, affected row
counts, conversion formulas, and validation evidence are written to
`variable_harmonization_audit.csv` and `field_equivalence_evidence.csv`.

For each eligible variable:

- same-timestamp duplicates are collapsed by their median;
- first-to-last change is calculated once per episode;
- repeated admissions are collapsed to one median change per patient;
- a two-sided Wilcoxon signed-rank test evaluates change from zero;
- Benjamini-Hochberg correction controls the false discovery rate;
- bootstrap confidence intervals quantify the median patient-level change;
- Theil-Sen slopes summarize robust within-episode change per hospital day;
- normalized-stay trajectories use patient-balanced median and IQR summaries.

The same run also performs a surgery-aligned analysis. Multi-valued surgery
names, start times, and end times are split on `^` and paired by position. CABG
episodes form the primary cohort; all episodes with a valid principal surgery
form a sensitivity cohort. Non-overlapping perioperative phases are summarized
with patient-balanced medians, and prespecified paired contrasts use Wilcoxon
tests, patient bootstrap confidence intervals, and within-cohort BH-FDR
correction. Invalid surgery timestamps are retained in machine-readable audit
tables and are excluded only from surgery-aligned calculations.

Run the complete experiment with:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv
python -m study.exp2_lab_longitudinal_statistics.run_analysis
```

Generated files are separated by audience and format:

- `outputs/figures/`: PNG figures, `per_analyte/` for stay trajectories,
  and `per_analyte_surgery/` for surgery-aligned trajectories; no PDF output;
- `outputs/tables/`: machine-readable CSV tables and audit records;
- `outputs/reports/`: human-readable Markdown reports;
- `outputs/metadata/`: machine-readable JSON manifests and quality metadata.

### Current Ten-Task Field Audit

Audit the current Exp2 source aliases, unit/specimen exclusions, duplicate
events, questionable result text, cache equivalence, and matched-video impact:

```bash
python -m study.exp2_lab_longitudinal_statistics.audit_ten_task_fields
python -m unittest study.exp2_lab_longitudinal_statistics.test_ten_task_field_audit
```

This reads the current merged lab table and current native224 main-experiment
records. It does not rerun the longitudinal analysis or change training data.
The real lab loader is checked in an automatically removed temporary directory.
Results go to `outputs/tables/ten_task_field_audit/`,
`outputs/reports/TEN_TASK_FIELD_AUDIT.md`, and
`outputs/metadata/ten_task_field_audit_manifest.json`.
