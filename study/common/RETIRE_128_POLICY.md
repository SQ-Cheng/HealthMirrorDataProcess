# Retirement of superseded 128 face artifacts

The pre-cleanup implementation and longitudinal tracking were committed and
pushed as `d03283d`. Cleanup is a separate subsequent commit.

Removed from migrated face-only Exp2 regression/classification and Exp6 regression:
128 model checkpoints, predictions, histories, figures, image indexes, all-frame
and ablation results, and obsolete launch/monitor/analysis scripts. Native results
and trained 24h models are untouched. Old-resolution comparison artifacts are
removed with their deleted baselines.

The 224 controller and pending 24h patient-diverse / 12h five-fold monitors now
depend only on saved native clinical records, native scalers and the shared FFV1
index. Parent-confirmed completed jobs are validated and reused; unfinished jobs
continue with exactly the original settings and patient splits. Switching the
controller required stopping the old legacy-dependent queue and restarting the
four unfinished 6h tasks, not retraining completed models.

## Exceptions

Whole studies with only a 128 main remain intact: face+history, history-only,
Exp1, Exp3, Exp4, Exp5, Exp6 direction classification, and existing analysis-only
studies. Laboratory-only statistics and clinical patient tables are independent
of pixel resolution and are not deleted. Exp8 currently has only 128 main results
and is retained while its native runs are queued.

Shared clinical inputs are protected because removing them would break Exp3
or Exp6 classification. These are face regression `outputs/20frame` task tables,
source tables, scalers and run index; its `outputs/5fold/splits`; and Exp6 root
pair tables/scalers/source metadata, pair_predictions.csv, and `cache/frames20`.
There are no old face-only or delta regression weights in those locations.
The protected Exp3 five-fold clinical split can be reconstructed deterministically
and must agree with its saved split-manifest hash and fold test identities.

Raw external sources, the 128 corpus needed by protected studies, and ImageNet
weights are not removed. Minimal shared MJPEG utilities remain for the exceptions;
migrated training entry points explicitly reject legacy indexes.

Machine-readable deletion lists and byte counts are generated under
`common/outputs/retire_128/`. `retire_128_artifacts.py` supports a dry run and
explicit `--apply`, refuses an active legacy-dependent queue, and does not follow
symlinks or delete outside `study/`.
