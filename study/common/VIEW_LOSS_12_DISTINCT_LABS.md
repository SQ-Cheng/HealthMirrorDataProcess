# Twelve-Distinct-Lab View-Level Loss Ablations

These classification and regression ablations change only batch composition
relative to the current single-split 12h `video_loss/` experiments. They retain
EfficientNet-B0/head32, head lr=2e-4 / 40 epochs / patience 10, full fine-tuning
lr=1e-5 / 60 epochs / patience 12, AdamW weight decay=1e-4, cosine floor=1e-6,
and compilation. There is no patient-diverse sampling or 30/40 schedule.

## Batch And Objective

Each full batch contains twelve different clinical measurement events, one
matched video per event, twenty selected frames per video, and one shared view
for those frames: 12 x 20 x 1 = 240 images and twelve losses.

The event ID is canonical hospital ID plus the matched target-specific lab
report timestamp in the corrected source manifest. Equal numerical values
from different events remain distinct. Videos matched to the same measurement
cannot occur together in a batch. Patients are not forced to be distinct;
one patient's different measurements can occur together.

The loss unit is **video x view**. Classification retains weighted BCE on the
mean of twenty frame probabilities; regression retains SmoothL1(beta=0.5)
on the mean of twenty robust-scaled predictions. Training class weights are
unchanged. This does not pool all five views into one loss.

All five views of every video's twenty frames still occur once per epoch,
in different batches. Remaining event-queue sizes balance scheduling so full
batches can be filled without oversampling. Only the final batch is partial
in each current cohort and remains included. No videos, frames, or views are
dropped. This is batch diversity, not event reweighting over an epoch.

Splits, labels, scalers, seeds, model, loss formulas, total inputs and optimizer
update count are unchanged. Only `clinical_event_id` is appended to task CSVs;
original numeric strings are preserved. Original-view evaluation is unchanged.
The existing FFV1 index/cache is reused; no new frame cache is created.

## Queue And Results

The waiting screen consumes no GPU. After both current `video_loss/` experiments
complete and release their queue lock, sixteen new jobs run dynamically on four
GPUs. An interrupted predecessor is reported as an error.

New results in each classification/regression study:
`outputs/ablations/lab_match_12h_face224/view_loss_12_distinct_labs/`.
Previous results remain intact. Standard figures, classification confusion
matrices, and paired comparisons with the current `video_loss/` results are
automatic. Regression additionally retains the frame-loss comparison.
Manifests/audits specify event IDs, source hashes, and complete-epoch coverage.

```bash
python -m unittest study.common.test_video_loss
python -m study.common.run_distinct_lab_views_12h --check-only
bash study/common/launch_distinct_lab_views_12h_screen.sh
screen -r exp2_view_loss_12labs_12h
```

Log: `study/common/logs/view_loss_12_distinct_labs_12h/run.log`.
Audit: `study/common/outputs/view_loss_12_distinct_labs_12h/batch_audit.csv`.
Tests are CPU-only: models and objectives are unchanged from the prior real-video
two-stage GPU smoke test.
