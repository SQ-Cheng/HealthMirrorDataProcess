# Refreshed laboratory-data queue

The `lab_update_20261007` run rebuilds the eight-target 12h source from the
current `merged_lab_tests.csv`, repeats the existing 512-candidate balanced
patient-disjoint split search, and refits robust scalers on training records
only. All classification, regression, and architecture-control models share
the new task-specific records, splits, twenty native224 frames, and five views.
The existing training configurations and loss/batch policies are unchanged.

Result order, with four-GPU scheduling inside each Exp2 group:

1. Chunked video/view-level EfficientNet-B0 loss: sixteen models.
2. Twelve-distinct-lab view-level EfficientNet-B0 loss: sixteen models.
3. Histogram MLP, RGB/HSV/Lab-statistics MLP, and small CNN: forty-eight models.
4. Exp4 postoperative recovery: one model, with its existing training protocol.

DINO is not included. No new model is substituted for its missing weights.
The rerun overwrites the previous results in the original result directories.
Only small cohort-difference audits are retained before replacement. The new data-only
reference is **not** represented as a retrained frame-loss baseline, and
comparisons between the old and new cohorts are not treated as controlled
model comparisons. Matched comparisons among the refreshed experiments and
all ordinary result/history figures remain automatic.

The literal Chinese starred total-bilirubin alias is added to the shared
analyte definitions. Unspecified units and different physiological quantities
are not inferred. Existing unit conversions, duplicate-event collapsing,
hospitalization constraints, session timestamps, and nearest-lab matching
remain in force. All available videos are considered; only those with twenty
valid nonadjacent native224 frames enter the models. Small packet-offset
indices are reused or rebuilt; no duplicated RGB video/image caches are made.

```bash
export HEALTHMIRROR_LAB_RUN_TAG=lab_update_20261007
export HEALTHMIRROR_FACE_SOURCE=face224
# Initial preparation uses temporary result paths; the launch script installs
# the prepared Exp4 cohort and replaces the original result trees once.
python -m study.common.run_latest_lab_queue --prepare-only
bash study/common/launch_latest_lab_screen.sh
screen -r exp2_exp4_lab_update_20261007
```

Audit tables and source fingerprints:
`study/common/outputs/lab_update_20261007/`.

Exp2 EfficientNet result roots, below each classification/regression experiment:
`outputs/ablations/lab_match_12h_face224/`, in `video_loss/` and
`view_loss_12_distinct_labs/`.

Architecture controls:
`study/exp2_face_architecture_ablation/outputs/`.

Exp4: `study/exp4/outputs/`.

Queue log: `study/common/logs/lab_update_20261007/run.log`.
Successful stages generate completion markers; an error stops the chain
instead of silently launching the next stage. Per-task markers allow completed
Exp2 jobs to be reused on restart under the same data/training contract.
