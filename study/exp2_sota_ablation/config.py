"""Seven fixed hypotheses; never select a configuration using test results."""

from copy import deepcopy
from pathlib import Path

from study.exp2_face_rgb_ir_dinov3b_regression import config as reference

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
BASELINE = reference.OUTPUT
TARGETS = tuple(reference.TARGETS)
ARCHITECTURE = "dino_farl_scalar_gate"
BRANCHES = reference.branches_for(ARCHITECTURE)
LOG_TARGETS = ("lactate_high", "urea_high", "total_bilirubin_high", "creatinine_high")
DEFAULT = dict(model="scalar", normalize=False, dropout=.25, projection_dropout=0.,
    ir_dropout=0., rgb_prior=.5, views=list(reference.VIEWS), loss="frame", balance=False,
    density=False, rank=0., time_tau=None, log_targets=[], lr=2e-4, min_lr=1e-6,
    weight_decay=.001, epochs=80, patience=12, warmup=0, warm_start=False, online_epochs=0)


def specification(**changes):
    return {**deepcopy(DEFAULT), **changes}


VARIANTS = {
    "a01_regularized_rgb_prior": specification(normalize=True, dropout=.5,
        projection_dropout=.15, ir_dropout=.2, rgb_prior=.8, lr=1e-4,
        weight_decay=.01, epochs=40, patience=8, warmup=2),
    "a02_video_patient_balance": specification(loss="video_view", balance=True,
        lr=1e-4, weight_decay=.003, epochs=40, patience=8, warmup=2),
    "a03_hierarchical_rgb_warmstart": specification(model="hierarchical", warm_start=True,
        rgb_prior=.95, loss="video_view", balance=True, dropout=.35, lr=5e-5,
        weight_decay=.003, epochs=30, patience=8, warmup=2),
    "a04_log_density_rank": specification(normalize=True, dropout=.4, rgb_prior=.7,
        loss="video_view", balance=True, density=True, rank=.05, log_targets=list(LOG_TARGETS),
        lr=1e-4, weight_decay=.003, epochs=40, patience=8, warmup=2),
    "a05_color_safe_views3": specification(normalize=True, dropout=.4, ir_dropout=.1,
        rgb_prior=.8, views=["original", "hflip", "center_crop"], loss="video_view",
        balance=True, lr=1e-4, weight_decay=.003, epochs=40, patience=8, warmup=2),
    "a06_time_weighted": specification(loss="video_view", balance=True, time_tau=6.,
        lr=1e-4, weight_decay=.003, epochs=40, patience=8, warmup=2),
    "a07_ir_tail_adaptation": specification(model="hierarchical", warm_start=True,
        rgb_prior=.95, loss="video_view", balance=True, dropout=.35, lr=5e-5,
        weight_decay=.003, epochs=30, patience=8, warmup=2, online_epochs=20,
        online_patience=6, online_head_lr=1e-5, online_encoder_lr=1e-6,
        online_head_min_lr=1e-6, online_encoder_min_lr=1e-7,
        online_encoder_weight_decay=.01, online_warmup=2, online_microbatch=4,
        compile_online=True, compile_dynamic=False, compile_fullgraph=True,
        farl_attention="local_mask_no_attribute_assignment"),
}


def get_variant(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown ablation: {name}")
    return deepcopy(VARIANTS[name])
