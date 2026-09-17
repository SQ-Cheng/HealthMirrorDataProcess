# Exp5: paired pre/postoperative face recovery

This CABG experiment predicts a postoperative laboratory-derived recovery score
from one preoperative RGB face and one postoperative RGB face from the same
patient.

The score contains equally weighted Lactate, Blood-gas Hb, high-sensitivity
Troponin I, O2Hb fraction, and laboratory Glucose components. Each component is
the position obtained by projecting the observed value onto its training-patient
average postoperative trajectory. Projection includes the postoperative time as
a local prior, so non-monotonic trajectories such as Troponin can rise and later
fall without assigning one global good/bad direction. Validation and test
patients never contribute to trajectory fitting.

Patients are split before trajectory fitting. Every retained postoperative
video requires all five lab values within 24 hours and at least one valid
preoperative video in the same CABG hospitalization. The nearest preoperative
video is used; 20 deterministic nonadjacent frame pairs are streamed from the
source MJPEG files without writing decoded images.

The model uses two independently trainable ImageNet-pretrained EfficientNet-B0
backbones and separate 64-dimensional projectors for the preoperative and
postoperative faces. A 32-dimensional fusion head receives pre, post,
post-minus-pre, and absolute-difference features. Both backbones are frozen
first; both final EfficientNet stages are then fine-tuned at a lower learning
rate. The model runs in eager mode.

```bash
# Data/model smoke test
python -m study.exp5_face_pair_recovery.run_all --smoke

# Detached formal run
bash study/exp5_face_pair_recovery/launch_screen.sh
```

Training completion automatically writes the checkpoint, complete history,
video-level predictions, metrics, training curve, score-definition figure, and
test-result figure under `outputs/`.
