# Exp6 paired-face laboratory direction classification

This standalone experiment uses the completed shared-backbone Exp6 regression
experiment's nine ordered laboratory-event pairs, patient-disjoint split,
20-frame video index, ImageNet EfficientNet-B0 weights, 32-unit difference
head, five synchronized views, chunk-shuffled 24-source-frame-pair batches,
four-GPU dynamic queue, and two-stage schedule without rebuilding the data.

The label is `1` when the later laboratory value is higher and `0` when it is
lower. Pairs with numerically zero change (`abs(delta) <= 1e-12`) have no
binary direction and are excluded; `outputs/label_summary.csv` reports this
per task and split. The original patient split assignment is never changed.

The regression output becomes an up-direction logit. Training uses
patient-pair-count weights from the original dataset and BCEWithLogits with a
train-only negative/positive-pair `pos_weight`; the latter is necessary for
imbalanced targets such as troponin. The frozen-head and full-backbone stages
retain the main experiment's AdamW weight decay, learning rates, cosine floor,
epoch limits, patience, and five-view augmentation. The checkpoint is chosen
by validation pair-level weighted BCE. Validation and test aggregate the
20 original-frame logits per pair, then apply sigmoid and a fixed 0.5 cutoff.
No test-derived threshold is fitted.

Run with `bash study/exp6_face_pair_lab_delta_classifacation/launch_screen.sh`.
This starts a detached screen session. Checkpoints, histories, pair-level
predictions, label counts, metrics, and figures are written to `outputs/`.
Figures are generated automatically only after all nine tasks succeed.
