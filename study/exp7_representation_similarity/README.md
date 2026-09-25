# Exp7: held-out feature similarity versus laboratory similarity

No retraining. Uses the existing five-view shared-backbone Exp6 pair-delta
checkpoints and the 20-frame Exp2 face-only and face+history regression
checkpoints. Each model is evaluated only on its original test patients.
Within each target, one observation is selected per patient by a fixed hash
of its ID, without looking at the target value. The two Exp2 models use the
same intersection of test videos and identical labels. Exp6 has a different
split and is **not** a paired cohort comparison with Exp2.

For every selected observation, all 20 indexed video frames are passed through
the trained EfficientNet-B0 encoder using the original (unaugmented) inference
view. Their 1280-dimensional pooled features are averaged per video. Exp6 uses
the late-minus-early feature difference; Exp2 uses the face feature alone,
including for the face+history model (history features are deliberately
excluded from the representation being tested). The analysis compares cosine
similarity between different patients with negative absolute distance between
their true raw lab values or raw lab deltas. It reports Spearman rho, a
patient-label permutation p-value, and FDR-adjusted q-value. These are
associations, not evidence that the encoder causally captures physiology.
For Exp6, an additional permutation shuffles targets only within quartiles of
the elapsed time between the two laboratory events; its p/q columns provide
a sensitivity check for lab-interval confounding.

Run from the repository root:

```bash
conda run -n healthmirrorenv python -m study.exp7_representation_similarity.analyze --gpus 0,1,2,3
```

Outputs are under `outputs/`: `association_metrics.csv`,
`gap_quantile_curves.csv`, `feature_manifest.csv`, compact per-job feature
arrays, `method.json`, and overview/per-model comparison PNGs in `figures/`.
To rerun only statistics and plots from saved features, use `--skip-extraction`.
