# Shared-Backbone Head32 Regression

`study/exp2_face_shared_backbone_head32_regression` trains one ImageNet EfficientNet-B0
encoder with ten separately parameterized `SingleTaskHead(1280,32,1)` modules.
Each head retains the main FC/LayerNorm/SiLU/Dropout(0.25)/FC design.
Encoder parameters: 4,007,548. Each head: 41,089. Total: 4,418,438.

All 1,391 main native224 videos / 563 patients and the original ten per-task
24h labels are reused, including direct HCT/eGFR. The old per-task splits cannot
be reused together: 546 patients have different assignments across tasks and
would leak through the encoder. A new common patient split is searched with 512
candidates using the inherited per-analyte distribution audits. The selected split
has 339/112/112 train/validation/test patients and passes all sixty distribution
pairs. Per-task median/IQR scalers are refit on this common training split only.
No matched-split independent-backbone control is trained, as requested.

Each frame is encoded once. Missing analytes are masked, not imputed. For task
`t` with `N_t` training videos out of `N` union training videos, its per-frame
SmoothL1(beta=0.5) contribution receives fixed weight `N/(10*N_t)`.
Complete epoch coverage therefore estimates the equal average of ten observed-target
losses rather than letting larger cohorts dominate. Each full batch contains twelve
video/view groups, twenty frames each. No target-specific assay event occurs twice
within a batch. Conflict-constrained tail batches may be smaller; no frames, views
or target labels are discarded or oversampled. All five main views are retained.

Stage 1 freezes the entire encoder and its BatchNorm statistics; all heads train
at lr 2e-4 for at most forty epochs, patience ten. Stage 2 trains encoder and heads
at lr 1e-5 for at most sixty epochs, patience twelve. AdamW, weight decay 1e-4,
cosine floor 1e-6, compilation, and gradient clipping remain unchanged.
The joint early-stop/checkpoint criterion is the equal average of each task's
video-level validation MAE divided by its training IQR. This retains the main MAE
selection principle without mixing raw units. One shared checkpoint is selected;
heads never combine independently selected encoder states.

This independent experiment owns `outputs/` and `logs/`. The main experiment is read-only. `model.pt` contains the one shared encoder and
all ten heads; per-target `head.pt` files and run manifests point to it.
Histories, raw-unit metrics, original-view video averages, frame predictions, and
figures use the normal Exp2 layout. Comparisons include separate full-cohort and
common held-out test-video tables/plots; changed training splits are stated explicitly.
The source main results are preserved.

```bash
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp2_face_shared_backbone_head32_regression.run_all --prepare-only
bash study/exp2_face_shared_backbone_head32_regression/launch_screen.sh
```

The single joint model runs on GPU0 in detached screen `exp2_shared_backbone_24h`.
Logs: `logs/run.log`.

## CUDA Graph Memory Fix

The first fine-tuning epoch failed with 14.4 GiB allocated in CUDA Graph private
pools. This experiment now retains Inductor compilation but explicitly disables
CUDA Graphs and enables dynamic batch shapes. Evaluation runs eagerly, so its
different batch sizes and modes do not create additional compiled captures.
Transient prediction/loss tensors are released after each update; compiled wrappers,
optimizer/scaler state and allocator caches are released at stage boundaries.
Data, global split, masks, loss, 240-frame logical batches and BatchNorm policy are
unchanged.

Recover directly from the completed best head checkpoint:

```bash
bash study/exp2_face_shared_backbone_head32_regression/launch_screen.sh --resume-finetune
```

Existing head histories are preserved and fine-tuning continues at global epoch12.
The pre-crash CUDA RNG state was not saved, so recovery uses the fixed seed rather
than claiming an exact continuation of its stochastic stream. Both allocated and
reserved GPU memory peaks are recorded in the training history.
Use `run_all --reuse-prepared --compiled-smoke` to exercise the actual compiled
training backend on real 224-frame input, rather than an eager-only smoke check.
