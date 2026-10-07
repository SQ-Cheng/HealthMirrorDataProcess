# Four-laboratory partial-CASUS feasibility audit

This CPU-only analysis does not alter Exp4's recovery target or start training.
It asks how much score variation would be available if postoperative faces
were instead used to predict a four-laboratory CASUS subtotal (0-16).

## Cohort and matching

Use the current Exp4 cohort: 1,084 videos from 473 CABG postoperative patients,
each with twenty eligible native224 frames. Clinical time uses validated
Session Timestamp and the original recording duration. No preoperative face
is required. For each video, independently select the nearest eligible result
for each of four analytes within twelve hours of the original capture interval.
Candidates must fall after CABG completion and before discharge in the same
hospitalization. Interval distance is zero inside the recording interval;
ties use midpoint distance and then report time. Missing components exclude
the total, rather than being assigned zero points. Components need not come
from the same draw, so this is not a simultaneous four-assay measurement.

The current `merged_lab_tests.csv` has SHA256
`a830705e85c0e66b969ce223b7c3e7a183ad432a43e30b6916e29ffe418a78c5`.

## Score

Each component contributes 0-4 points, using the existing CASUS implementation
and the published additive-CASUS table:
https://pmc.ncbi.nlm.nih.gov/articles/PMC4559007/

| Component | Unit | Next-grade lower bounds | Four-point condition |
| --- | --- | --- | --- |
| Serum creatinine | mg/dL | 1.2, 2.3, 4.1 | >5.5 |
| Serum total bilirubin | mg/dL | 1.2, 3.6, 7.1 | >14 |
| Lactate | mmol/L | 2.1, 4.1, 8.1 | >12 |
| Platelets | 10^3/uL | 81, 51, 21 descending | <21 |

Platelets above 120 receive zero points; 81-120 receive one. Creatinine in
umol/L is divided by 88.4; bilirubin by 17.104. Platelets in 10^9/L have the
same numerical value as 10^3/uL. The published rounded ranges are implemented
as continuous bins beginning at the next-grade lower bounds. Unknown units,
censored values, unsupported specimens, and invalid physical ranges are
excluded. Explicit serum aliases exclude urinary creatinine and direct or
indirect bilirubin. Identical patient/timestamp/analyte events are collapsed
by the median after filtering.

This is a **partial-CASUS nearest-lab target**, not the full score or its
daily-worst-value protocol. Higher values denote greater physiological
abnormality, not greater recovery.

## Result on the updated table

Complete scores are available for **595 videos from 338 patients**.

| Score | Videos | Fraction |
| --- | ---: | ---: |
| 0 | 182 | 30.6% |
| 1 | 239 | 40.2% |
| 2 | 128 | 21.5% |
| 3 | 39 | 6.6% |
| 4 | 6 | 1.0% |
| 5 | 1 | 0.2% |
| 6-16 | 0 | 0% |

Mean: 1.077; SD: 0.947; median: 1; IQR: 0-2. Scores 0-2 account for
92.3% of videos. The patient-level plot gives each patient one median of
their eligible video scores, avoiding extra weight for patients with many
videos. Patient median scores range from 0 to 4, with median 1 and IQR 0-2.

Among the original 1,084 candidate videos, eligible 12h values are available
for creatinine in 937 videos, bilirubin in 921, lactate in 681, and platelets
in 1,016. Missing-component counts overlap. Four selected reports span a
median of 5.48 hours and at most 23.73 hours; this spread is saved per video.

## Reproduce

```bash
python -m study.exp4.analyze_casus_distribution
python -m unittest study.exp4.test_casus_distribution
```

Outputs remain local under `study/exp4/outputs/casus_12h_distribution/`:
CSV scores and audits, an analysis manifest with hashes and thresholds, and
PNG/PDF charts under `figures/`. Raw patient data and generated results are
not committed to Git.
