# Exp5 paired-face partial-CASUS regression

This experiment keeps the Exp5 independent pre/post EfficientNet-B0 encoders,
64-dimensional projectors, and 32-dimensional fusion head. It predicts the
additive subtotal from the four CASUS laboratory descriptors available in the
merged dataset: serum creatinine, serum total bilirubin, lactic acid, and
platelet count.

The target ranges from 0 to 16 points. It is a **partial CASUS**, not the full
10-descriptor additive CASUS. Training uses target/16 with the unchanged
sigmoid output and reports all errors and predictions in CASUS points.

Data rules:

- Same-patient preoperative and postoperative videos from one CABG admission.
- Each postoperative video uses the nearest value of every component within
  24 hours; all four components are required.
- Exact item aliases, specimen types, and units are audited in
  `outputs/lab_source_audit.csv`.
- Creatinine and bilirubin are converted from umol/L to mg/dL before scoring.
- Splits are patient-disjoint and selected from 512 candidates to balance the
  score, raw component values, postoperative progress, and split sizes.
- Twenty deterministic nonadjacent frames and the original three Exp5 views
  are used. Frame bytes are indexed, not materialized as image files.

Run on all four visible GPUs:

```bash
bash study/exp5_face_pair_casus/launch_screen.sh
```

Training automatically writes histories, the selected checkpoint, video-level
predictions, metrics, and figures under `outputs/`.

## Post-face-only protocol

The postoperative-only protocol uses every eligible postoperative CABG video;
it does not require a preoperative video. It has its own patient-disjoint,
distribution-balanced split and compact 20-frame offset index. Label matching,
three views, optimizer, and two-stage schedule remain unchanged. It executes
only one postoperative EfficientNet-B0 encoder. As in the recovery experiment's
single-input implementation, the missing 64-dimensional preoperative embedding
is fixed to zero and the unchanged 256-to-32 fusion head is retained.

```bash
bash study/exp5_face_pair_casus/launch_post_face_only_screen.sh
```

Its independent data audit, checkpoint, history, predictions, metrics, and
figures are written under
`outputs/protocols/post_face_only/`; the paired results are not overwritten.
