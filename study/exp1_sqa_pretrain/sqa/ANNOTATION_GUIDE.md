# ECG SQA human annotation

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

## Round 02

Round 02 contains 280 blinded tasks: 265 unique patient-disjoint samples and 15
blind repeats.

| Category | Unique samples |
|---|---:|
| Clean and raw positive anchors | 50 |
| Gaussian / high-frequency noise | 35 |
| Clipping / saturation | 35 |
| Baseline drift | 25 |
| Impulse / dropout | 25 |
| TCN–ResNet disagreement | 35 |
| Window–Reference disagreement | 20 |
| Score-stratified random raw | 40 |

All four models used for candidate selection are the original pre-human-label SQA
models (TCN/ResNet × Window/Reference). The selector rejects human-fine-tuned
checkpoints.

Raw, uncleaned samples are multiplied by `-1` before display, corruption, and model
input construction. Cleaned samples retain their original polarity. Controlled
corruptions are deterministic and recorded in `queue.csv`; the annotation display and
later supervised loader reconstruct exactly the same transformed signal.

## Start or resume annotation on a headless server

The annotation interface is a local Streamlit web app. It binds to loopback only, so
it is not exposed directly to the network.

On the server:

```bash
conda activate healthmirrorenv
streamlit run study/exp1_sqa_pretrain/sqa/annotate.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true -- \
  --round-id round02
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
`study/exp1_sqa_pretrain/sqa_annotations/<round-id>/labels.csv`. Browser refreshes,
SSH disconnects, and server restarts resume from the first unfinished task.

To use a different annotator name, append application arguments after `--`:

```bash
streamlit run study/exp1_sqa_pretrain/sqa/annotate.py \
  --server.address 127.0.0.1 --server.port 8501 -- \
  --round-id round02 --annotator your_name
```

## Fine-tune the frozen Window heads

The commands below document the round-01 runs. Round-02 cumulative fine-tuning should
be launched only after all round-02 labels are complete.

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

Round-02 cumulative fine-tuning uses both annotation sources and starts from the
original weakly supervised Window checkpoint (not the old round-01 human checkpoint):

```bash
python study/exp1_sqa_pretrain/sqa/train_human.py \
  --encoder ENCODER \
  --checkpoint ORIGINAL_WINDOW_CHECKPOINT \
  --annotation-source study/exp1_sqa_pretrain/sqa_annotations/round01/queue.csv study/exp1_sqa_pretrain/sqa_annotations/round01/labels.csv \
  --annotation-source study/exp1_sqa_pretrain/sqa_annotations/round02/queue.csv study/exp1_sqa_pretrain/sqa_annotations/round02/labels.csv \
  --checkpoint-tag cumulative_round02_polarityfix_v1 \
  --epochs 100 --patience 20 --batch-size 32
```

The cumulative loader merges consistently labeled cross-round duplicates, rejects
conflicting duplicates, and verifies that patients do not cross train/validation/test
boundaries.
