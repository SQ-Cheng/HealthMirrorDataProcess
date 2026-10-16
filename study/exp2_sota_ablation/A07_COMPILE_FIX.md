# A07 / Five-Fold C08 Compiler Repair

Date:2026-10-16 (Asia/Shanghai). Old A07 was explicitly stopped. A01-A06 results
were not changed. The active five-fold queue was stopped only for the controlled
repair and resumed after validation; the old A07 is not automatically restarted.

## Reproduced Causes

Environment:PyTorch2.4.1+cu121, V100-SXM2-16GB. A07 entered IR adaptation around
14:19 and produced no completed IR epoch or IR-stage checkpoint before stopping.
Eight CPU workers kept running. Its log had1056 symbolic-range warnings with
compile IDs0/0 through0/5; there was no OOM/nonfinite/fatal exception.

1. **DINO dynamic-guard failure.** With a counting backend (no Inductor codegen),
   identical4x3x224x224 inputs still recompiled on the second/third call. The
   logged guard failure was `name 'ToFloat' is not defined`. Initial two CPU
   forward/backward calls took40.30s and38.95s, generating another graph each
   time; the bounded diagnostic timed out during the third compile. Static
   shapes produced one graph, first call4.70s, subsequent calls2.17/2.09s.
2. **FaRL graph fragmentation.** CLIP attention writes `self.attn_mask` inside
   forward, even when it remainsNone. A static counting probe generated24 graphs
   on the first call and44 by the second, with block-object identity and
   requires-grad guard failures and cache-limit warnings. Replacing only this
   side effect with a local mask produced one full graph; CPU outputs were
   exactly identical and parameter gradients matched in the small-model test.

These are compiler-path problems, not evidence of incorrect timestamps, labels,
or corrupted IR images. Raising the recompilation limit or suppressing warnings
would not address either cause.

## Production Changes

- `study/common/ir_compile.py`: static `torch.compile`, fullgraph, Inductor,
  CUDA Graphs disabled, one compilation thread, no silent eager fallback.
- FaRL attention is overridden on the local encoder instance only. No installed
  library, upstream source, pretrained tensor, parameter name or checkpoint key
  is changed. The attention computation is otherwise identical.
- Shared A07 IR forward batches encoders in groups of4 but invokes fusion once
  on all20 frames, so modality dropout remains constant within a video.
- Five-fold C08 inherits the repaired initializer; its complete-video forward
  and hybrid loss are retained. No split, label, LR, weighting, view or stage
  budget is changed by this repair.
- Explicit compiler-policy fields and source hash cover the repair. A controlled
  migration preserves75 completed B00/C01 folds after verifying artifact hashes;
  no C08 job had started. Models/predictions/splits remain byte-identical. Old
  and new protocols and original completion markers are archived under
  `exp2_sota_5fold_ablation/outputs/protocol_migrations/`.

## Validation

29 CPU tests passed: original suites plus exact FaRL output/gradient equivalence,
single-graph reuse, production compiler flags and migration rejection guards.
GPU probes use random inputs, no clinical data and **no optimizer updates**.

Individual compiled4-frame forward/backward: DINO first28.95s then7.04/6.64ms;
FaRL first25.16s then7.99/6.32ms. Compared with original eager FP16 outputs,
maximum absolute deviations were .00232 and .000977 respectively, not CPU-level
bitwise equivalence across compiled/eager mixed-precision kernels.

The joint production path retains all20 frame graphs before hybrid-loss
backward, includes the actual fusion/head and IR dropout, and changes original,
flip and center-crop views. The first train call took33.44s, then75.8/76.1ms.
All outputs/gradients were finite; frozen prefixes had no gradients. Peak
allocated memory was1350.8MiB. Training used exactly two encoder graphs despite
view changes. Switching to `no_grad` evaluation legitimately creates another
pair; repeated evaluation reused four total graphs (first24.54s, then54.5ms).
These microbenchmarks exclude video
decoding, data loading, a full optimizer epoch, and eight-process contention.

Diagnostic entry:

```bash
timeout --kill-after=5s 180s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.common.a07_compile_probe --encoder joint --device cuda:0 --steps 3
```

This does not launch a real experiment. The migration helper requires explicit `--apply` and refuses
changed data/splits/unrelated training settings or an already-started C08.

Reference: [PyTorch compiler troubleshooting](https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/torch.compiler_troubleshooting.md).
