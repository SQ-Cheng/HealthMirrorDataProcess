# Native-224 laboratory change tracking

CPU-only post hoc evaluation of the completed 24h EfficientNet-B0 / head32
face-only regression. No weights, splits, labels or training settings are changed.

```bash
/root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_face224_lab_change_tracking.analyze
```

The eight tasks use held-out test patients only. Videos sharing one lab timestamp
are collapsed using a prediction-blind nearest-video rule. Adjacent distinct lab
events yield observed and predicted changes; reversed video chronology is audited
and excluded. Metrics include Pearson/Spearman correlation, MAE/RMSE/R2,
direction accuracy/balanced accuracy, patient-macro direction accuracy,
within-patient centered correlation, and MAE improvement over no change.
Confidence intervals resample patients, not independent pairs.

Machine-readable results and exclusions: `outputs/tables/` and
`outputs/protocol.json`. Human-readable report: `outputs/REPORT.md`.
Figures: `outputs/figures/`; eight-panel figures use four columns and two rows.
