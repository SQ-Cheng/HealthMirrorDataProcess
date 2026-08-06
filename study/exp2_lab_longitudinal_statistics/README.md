# Exp2 Inpatient Lab Longitudinal Statistics

This experiment is independent of video data. It reads only
`merged_lab_tests.csv` and describes how numeric laboratory measurements change
during hospitalization.

The analysis unit is a hospital episode, defined by hospital patient ID,
admission time, and discharge time. Test variables are separated by normalized
test name and unit. Rows outside the admission-discharge interval, nonnumeric
results, and censored values such as `<0.01` are audited and excluded from
inferential calculations.

Verified aliases are harmonized before variable IDs are assigned: blood-gas
glucose is converted to mmol/L, fractional arterial/alveolar PO2 ratios are
converted to percent, and duplicate analyzer labels for standard-condition or
patient-condition P50 are collapsed. Standard-condition P50 remains distinct
from patient-condition P50. The applied rules and row-level validation evidence
are written to `variable_harmonization_audit.csv` and
`field_equivalence_evidence.csv`.

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

- `outputs/figures/`: PNG figures, per-analyte PNGs, and PDF figure books;
- `outputs/tables/`: machine-readable CSV tables and audit records;
- `outputs/reports/`: human-readable Markdown reports;
- `outputs/metadata/`: machine-readable JSON manifests and quality metadata.
