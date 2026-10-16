"""Fixed protocols; no experiment or checkpoint is chosen using outer test data."""

from copy import deepcopy
from pathlib import Path

from study.exp2_sota_ablation import config as parent

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
SPLITS = OUTPUT / "splits"
TARGETS = parent.TARGETS
FOLDS = 5
SEED = 20261016
OUTER_CANDIDATES = 256
INNER_CANDIDATES = 128
INNER_VAL_FRACTION = .2
VARIANTS = {"b00_reference": deepcopy(parent.DEFAULT)}


def extend(name, previous, **changes):
    VARIANTS[name] = {**deepcopy(VARIANTS[previous]), **changes}


extend("c01_regularized", "b00_reference", normalize=True, dropout=.5,
       projection_dropout=.15, ir_dropout=.2, rgb_prior=.8, lr=1e-4,
       weight_decay=.01, epochs=40, patience=8, warmup=2)
extend("c02_hybrid_video_loss", "c01_regularized", loss="hybrid", frame_fraction=.5)
extend("c03_soft_patient_balance", "c02_hybrid_video_loss", balance=True, balance_power=.5)
extend("c04_color_safe", "c03_soft_patient_balance", views=["original", "hflip", "center_crop"])
extend("c05_fold_rgb_warmstart", "c04_color_safe", model="hierarchical", warm_start=True,
       rgb_prior=.95, lr=5e-5, epochs=30)
extend("c06_soft_time_weight", "c05_fold_rgb_warmstart", time_tau=12., time_floor=.5)
extend("c07_soft_density", "c06_soft_time_weight", density=True, density_power=.25, density_cap=2.)
extend("c08_ir_tail", "c07_soft_density", online_epochs=20, online_patience=6,
       online_head_lr=1e-5, online_encoder_lr=1e-6, online_head_min_lr=1e-6,
       online_encoder_min_lr=1e-7, online_encoder_weight_decay=.01, online_warmup=2,
       online_microbatch=4, compile_online=True, compile_dynamic=False,
       compile_fullgraph=True, farl_attention="local_mask_no_attribute_assignment")
# Non-cumulative bundles probe risks seen in the completed single-split runs.
extend("x01_rank_bundle", "c07_soft_density", rank=.01)
VARIANTS["x02_log_rank_bundle"] = {**deepcopy(parent.VARIANTS["a04_log_density_rank"]),
    "loss": "hybrid", "frame_fraction": .5, "balance_power": .5,
    "density_power": .25, "density_cap": 2., "rank": .01}
extend("x03_frame_only_bundle", "c07_soft_density", loss="frame")

PARENTS = dict(zip(list(VARIANTS)[1:9], list(VARIANTS)[:8]))
PARENTS.update(x01_rank_bundle="c07_soft_density", x02_log_rank_bundle="independent_bundle",
               x03_frame_only_bundle="c07_soft_density")
WARM_SETTINGS = {**deepcopy(parent.DEFAULT), "model": "rgb", "warm_start": False}


def settings(name):
    return deepcopy(VARIANTS[name])
