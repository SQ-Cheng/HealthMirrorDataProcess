# Independent RGB/IR EfficientNet-B0 Regression

Each of the current ten analytes has an independent model:

```text
RGB224 -> ImageNet EN-B0 -> global average pool -> RGB1280
IR224  -> independent ImageNet EN-B0 -> global average pool -> IR1280
Concat(2560) -> Linear(2560,32) -> LayerNorm -> SiLU
             -> Dropout(.25) -> Linear(32,1) -> robust-scaled raw lab value
```

The IR gray plane is copied to three channels. Each branch uses the original
ImageNet mean/std after the same deterministic view; this is not IR-specific
pretraining. The branches start from identical pretrained values but do not
share parameter storage. Total parameters:8,097,145; head parameters:82,049.

Reuse exact native224 RGB+IR paired records, per-analyte patient splits, seeds
and train-only median/IQR scalers from `exp2_face_rgb_ir_head32_regression`.
No rematching or split search. The cohort contains1,311 videos/549 patients
across ten tasks; each task retains its own available records. Mutual-nearest
unique IR frames are within33.333ms; each modality has source frame gaps>=2.
The existing compact FFV1 indexes and original files are reused without storing
new decoded frames, copies of videos, or feature caches.

| Setting | Frozen-head stage | Full-finetune stage |
| --- | --- | --- |
| Trainable | Head only; both encoders eval/frozen | Both complete encoders and head |
| LR | 2e-4 | 1e-5 |
| Maximum epochs | 40 | 60 |
| Patience | 10 | 12 |
| Scheduler | Cosine to1e-6 | Cosine to1e-6 |

Other primary settings:AdamW, weight decay1e-4, SmoothL1(beta=.5), no target
clipping or density weighting, gradient clip1, FP16 AMP, no warmup. Five views
are original, horizontal flip, center crop(.9), brightness(+.06), contrast(+.08).
Every epoch retains all100 frame/view inputs per training video. Early stopping
uses validation video-level raw MAE; the best head/full model is selected by that
same metric. Test is not used for selection. Standard per-video20-original-frame
averaging and raw inverse scaling are preserved.

Logical batches retain12 distinct clinical lab events x20 paired frames x1 view
=240 frame pairs, one optimizer update. To control dual-backbone memory, process
microbatches of60 frame pairs, sum loss divided by actual logical-batch size,
accumulate gradients, then clip and update once. Partial batches are not dropped
or reweighted incorrectly. In finetuning, BatchNorm updates per60-pair microbatch;
this is not mathematically identical to a240-pair BatchNorm forward. Both training
stages use Inductor compilation with dynamic batch shapes and CUDA Graphs disabled.
Evaluation uses the original eager module to avoid extra eval-graph compilations.
Compilation errors raise rather than silently falling back to eager; actual
compiled forward/backward execution is checked and logged. Compile threads are
limited to1 per worker; existing shared compiler caches are retained and reused.
This execution policy and the BN consequence are recorded in the protocol.

## Launch

```bash
bash study/exp2_face_rgb_ir_dual_enb0_regression/launch_screen.sh
screen -r exp2_rgb_ir_dual_enb0
tail -f study/exp2_face_rgb_ir_dual_enb0_regression/logs/run.log
```

An initial bounded smoke trains one full240-pair batch in each stage, verifies
frozen parameters/buffers and both branches' finetune updates, checks saved stage
checkpoints and20-frame evaluation. It then starts an eight-GPU dynamic queue
of ten models. Existing experiments and outputs are not changed.

- `outputs/protocol.json`: source hashes, architecture, all hyperparameters.
- `outputs/dataset_counts.csv`, `task_records/`, `target_scalers.json`: cohort.
- `outputs/regression/dual_enb0/runs/<target>/`: checkpoint, both stage best
  checkpoints, complete train/val loss and metrics, per-frame/per-video predictions,
  individual training log and history PNG.
- `outputs/regression/dual_enb0/figures/`: summary histories and fitted scatter.
- `outputs/figures/`: matched test metric bars against the4-channel EN-B0 and
  RGB main re-evaluated on the exact paired RGB frames.
- `outputs/tables/`: machine-readable comparisons and cohort audits.
- `outputs/COMPLETE`: written only after all models and PNG plots succeed.

The single- and dual-backbone RGB/IR experiments use identical clinical data,
but differ in fine-tuning BN batch size. RGB-only main training uses a larger
cohort; its comparison is matched-test, not a strict same-training-cohort ablation.

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/healthmirrorenv/bin/python -m unittest study.exp2_face_rgb_ir_dual_enb0_regression.test_experiment
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -m study.exp2_face_rgb_ir_dual_enb0_regression.run --check-only
```
