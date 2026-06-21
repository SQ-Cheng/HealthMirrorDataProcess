# ECG SQA human annotation — round 01

## Queue design

The queue contains 100 blinded tasks: 95 unique patient-disjoint windows and 5
blind repeats. Unique candidates combine TCN–ResNet disagreement, Window–Reference
disagreement, cross-task disagreement, suspicious consensus-high signals, obvious
artifacts, cleaned anchors, and random raw controls. `queue.csv` records the selection
reason and diagnostic SQIs; the GUI hides them by default.

## Three-state label

Assign one mandatory overall label to each 10-second window:

| Label | Meaning |
|---|---|
| `bad` | Unusable for reliable ECG analysis because signal content is obscured or invalid. |
| `uncertain` | Borderline, or physiology cannot confidently be separated from artifact. |
| `good` | Usable: dominant QRS timing and beat morphology are sufficiently reliable. |

Real arrhythmia or genuine morphology variation is not automatically poor quality.

The overall label applies to both outputs. Only when the two tasks clearly differ, set
an optional QRS or Morph override. For training, `good=1`, `bad=0`, and `uncertain` is
masked. A task override replaces the overall label for that output.

## Start or resume annotation on a headless server

The annotation interface is a local Streamlit web app. It binds to loopback only, so
it is not exposed directly to the network.

On the server:

```bash
conda activate healthmirrorenv
streamlit run study/exp1_sqa_pretrain/sqa/annotate.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

If Streamlit is missing from a recreated environment:

```bash
python -m pip install "streamlit>=1.35,<2"
```

On your local machine, create the SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@SERVER
```

Then open `http://127.0.0.1:8501` in a local browser.

For each window, optionally select QRS/Morph overrides and then click one of
`Save BAD`, `Save UNCERTAIN`, or `Save GOOD`. Saving advances to the next pending
sample. Previous/next navigation, skip, clear, and “first pending” are available.
The app is blinded by default.

Every completed action is saved atomically to
`study/exp1_sqa_pretrain/sqa_annotations/round01/labels.csv`. Browser refreshes,
SSH disconnects, and server restarts resume from the first unfinished task.

To use a different annotator name, append application arguments after `--`:

```bash
streamlit run study/exp1_sqa_pretrain/sqa/annotate.py \
  --server.address 127.0.0.1 --server.port 8501 -- \
  --annotator your_name
```

## Fine-tune the frozen Window heads

TCN:

```bash
python study/exp1_sqa_pretrain/sqa/train_human.py \
  --encoder tcn \
  --checkpoint study/exp1_sqa_pretrain/checkpoints/exp1_sqa_ecg_tcn_window_L1024_run01_best.pt
```

ResNet:

```bash
python study/exp1_sqa_pretrain/sqa/train_human.py \
  --encoder resnet \
  --checkpoint study/exp1_sqa_pretrain/checkpoints/exp1_sqa_ecg_resnet_window_L1024_run03_best.pt
```

Only the SQA head is optimized; the encoder remains frozen. Original checkpoints are
never overwritten. Outputs include before/after metrics, per-category score changes,
a comparison plot, consolidated labels, conflicts, and separate best/final checkpoints.
