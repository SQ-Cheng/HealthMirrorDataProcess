# Single-Split Native224 Video-Loss Ablations

These two eight-task ablations retain the existing 12h regression task records,
patient-disjoint single split, train-only robust scalers, native224 FFV1 index,
twenty nonadjacent frames, and five training views. No new matching or split
search is performed. The input model remains a single-frame ImageNet
EfficientNet-B0 with the same 32-dimensional hidden head.

## Objective

For each video and each training view, all twenty frames are forwarded through
the unchanged model before calculating one supervised loss:

- Classification: p_video = mean(sigmoid(frame logits)); weighted binary cross
  entropy on this probability, with the unchanged train negative/positive
  video-count ratio. Log-space averaging avoids numerical clipping. This is
  not BCE on mean logits and not mean frame BCE.
- Regression: prediction_video = mean(robust-scaled frame predictions);
  SmoothL1(beta=0.5) against the saved robust-scaled label. No density weighting.

The five views remain five separate video-level augmented examples, rather
than one prediction pooled over all hundred images. Validation/test use the
original view only. Their losses and performance metrics are both video-level.
All twenty frame predictions are retained for the existing evaluation artifacts.

## Unchanged Schedule

| Setting | Value |
| --- | --- |
| Head | lr 2e-4; maximum 40 epochs; patience 10 |
| Full-backbone fine-tuning | lr 1e-5; maximum 60 epochs; patience 12 |
| Scheduler | Cosine; floor 1e-6; no warmup |
| Optimizer | AdamW; weight decay 1e-4; gradient clip 1 |
| Head dropout | 0.25 |
| BatchNorm | Frozen during head stage; updated during fine-tuning |
| Compilation | Enabled; reduce-overhead |
| Checkpoint selection | Classification: val bACC; regression: raw-unit val MAE |

There is no patient-diverse sampler and no 30/40 schedule. Complete video/view
groups necessarily change frame order, but each full training batch retains
240 images: twelve groups of twenty frames, usually three and occasionally four videos across
their views. No samples or views are discarded. Total model inputs and optimizer
step count per epoch match the original 240-image configuration. Evaluation
uses up to 500 images to keep complete groups within its original 512 budget.
Only the new grouped loaders enlarge their bounded decode cache from 16 to 20
frames so the five views can reuse a video's decoded frames.

## Outputs And Comparison

Both experiments use `outputs/ablations/lab_match_12h_face224/video_loss/`
under their respective face-only classification/regression study directories.
Original frame-loss and five-fold results are preserved. Manifests record source
hashes, grouping, seeds, objective, and schedule. Complete task markers support
reuse; partial tasks restart. Sixteen jobs share a four-GPU dynamic queue.

Regression automatically produces a paired comparison with the existing single
12h frame-loss result. Classification has no existing single-split 12h frame-loss
baseline: its five-fold experiments are not claimed as a controlled comparison.
Both produce standard results/history figures, with classification also producing
video-level confusion matrices. CSV outputs remain outside `figures/`.

```bash
python -m unittest study.common.test_video_loss
python -m study.common.run_video_loss_12h --check-only
python -m study.common.run_video_loss_12h --smoke
bash study/common/launch_video_loss_12h_screen.sh
screen -r exp2_video_loss_12h
```

Global log: `study/common/logs/video_loss_12h/run.log`; each model also has its
own `runs/efficientnet_b0/<target>/train.log`. Smoke runs use temporary files,
one head epoch and one fine-tuning epoch on two real videos per split, without
compilation; they do not overwrite any experiment results.

The loss is more precisely view-level: one loss per video's twenty frames under
one view. A later batch-composition ablation places twelve distinct lab events
in each full batch; see [VIEW_LOSS_12_DISTINCT_LABS.md](VIEW_LOSS_12_DISTINCT_LABS.md).
