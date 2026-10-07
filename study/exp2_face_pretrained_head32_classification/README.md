# Native-224 Face-Only Binary Classification

Eight independent EfficientNet-B0 / head32 classifiers use exactly the retained
native-224 regression clinical records: oxyhemoglobin fraction, lactate, urea,
total bilirubin, platelets, hemoglobin, A/a PO2 ratio and creatinine.

Twenty nonadjacent RGB frames per video and five training views are used.
Evaluation averages twenty original-view frame probabilities per video.
Saved patient-disjoint splits, labels and original-video matching intervals are
preserved. The classifier trains BCEWithLogitsLoss, not thresholded regression;
positive weights are fitted to training class counts only.

Hemoglobin positivity means raw Hb <130 g/L for males and <120 g/L for
females. The source implementation uses 120 g/L for any non-male metadata;
the current native cohorts contain only explicit male/female entries.
The same source labels are retained in every ablation and fold.

Loss is BCEWithLogitsLoss with pos_weight = training negative videos /
training positive videos. Each video contributes the same 20 frames and five
training views, so this is also the frame/input class-count ratio. The positive
loss term is multiplied by this weight; it may be less than one when abnormal
Hb is the majority class. There is no focal loss, label smoothing, or additional
class-penalty term. Decisions use probability >=0.5 after averaging the twenty
original-frame probabilities for each video.

Main output: `outputs/face224/`; logs: `logs/face224/`.
Head lr=2e-4, 40 epochs, patience 10; full-backbone fine-tuning lr=1e-5,
60 epochs, patience 12. AdamW weight decay=1e-4; cosine floor=1e-6.

Native 12h five-fold results are organized under:
- `outputs/ablations/lab_match_12h_face224/5fold/`;
- `outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold/`.

The patient-diverse schedule uses head 1e-4 / 30 epochs / patience 8 and
fine-tuning 3e-6 / 40 epochs / patience 8, floors 1e-6 / 1e-7.
All three queued native 12h five-fold protocols share the same saved folds.

```bash
bash study/common/launch_face224_reruns_screen.sh
bash study/common/launch_face224_12h_5fold_screen.sh
```

Checkpoints, train/val/test metrics, predictions and history are saved; plots and
the five-fold comparisons are automatic. Legacy face-only results and launchers
were retired. The face-only engine loads native clinical records directly and
does not require history features or fall back to a 128 frame index.

Video-level confusion matrices are automatic in `figures/test_confusion_matrices.png`
and `.pdf`; five-fold roots also contain `figures/oof_confusion_matrices.png`
and `.pdf`. Cells display counts and percentages normalized within the true
class; machine-readable counts are saved outside `figures/`. OOF pooling uses
exactly one held-out prediction per video, never concatenated train/val predictions.
To add these figures to all already-completed results without training:

```bash
python -m study.exp2_face_pretrained_head32_classification.plot_confusion_matrices
```

The single-split 12h video-loss ablation lives in
`outputs/ablations/lab_match_12h_face224/video_loss/`. It retains the main 40/60
schedule, not patient-diverse 30/40. See [VIDEO_LOSS_12H.md](../common/VIDEO_LOSS_12H.md).
