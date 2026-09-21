# Exp5: paired pre/postoperative face recovery

This CABG experiment predicts postoperative deviation from the population's
average laboratory trajectories. It compares paired pre/postoperative RGB
faces, a preoperative face alone, and a postoperative face alone.

The score contains equally weighted Lactate, Troponin I, Creatinine, Total
bilirubin, Platelet count, Hemoglobin, C-reactive protein, Albumin, and PaO2
components. Each component is the absolute residual from the training-patient
median trajectory at the postoperative video time, divided by the local IQR.
Only reports strictly after surgery are eligible. This removes preoperative and
intraoperative measurements, including intraoperative Troponin outliers, before
both trajectory fitting and video-label matching. Validation and test patients
never contribute to trajectory fitting.

Patients are split before trajectory fitting. Every retained target requires all
nine lab values within 24 hours. Each protocol uses its maximum eligible cohort:
paired requires both usable face videos, pre-only requires only a usable
preoperative face, and post-only requires only a usable postoperative face.
Consequently the three test cohorts need not be identical. Twenty deterministic
nonadjacent frames are streamed from source MJPEG files without decoded-image
cache files.

The model uses two independently trainable ImageNet-pretrained EfficientNet-B0
backbones and separate 64-dimensional projectors for the preoperative and
postoperative faces. A 32-dimensional fusion head receives pre, post,
post-minus-pre, and absolute-difference features. Both backbones are frozen
first; both final EfficientNet stages are then fine-tuned at a lower learning
rate. The model runs in eager mode.

```bash
# Build protocol-specific records and the shared byte-offset cache
python -m study.exp5_face_pair_recovery.run_all --prepare-only

# Data/model smoke tests
python -m study.exp5_face_pair_recovery.run_all --smoke --device 0
python -m study.exp5_face_pair_recovery.ablation_train --mode pre_only --smoke --device 1
python -m study.exp5_face_pair_recovery.ablation_train --mode post_only --smoke --device 2

# Detached formal run
bash study/exp5_face_pair_recovery/launch_screen.sh
```

Training completion automatically writes each protocol's checkpoint, complete
history, video-level predictions, metrics, training curve, target-definition
figure, test-result figure, and the three-protocol comparison under `outputs/`.
