"""Exp2-aligned training; only the retrospective target construction changes."""

from pathlib import Path

from study.exp2_face_pretrained_head32_regression import config as reference


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = Path(reference.OUTPUT_DIRS["20frame"])
SOURCE_PROTOCOL = STUDY / "common/outputs/face_main_24h_frame_loss/protocol.json"
OUTPUT = HERE / "outputs"
LOGS = HERE / "logs"
CACHE = HERE / "cache"
PREOPERATIVE_POLICY = "nearest_preoperative_report_same_admission_no_distance_limit"
TARGETS = reference.TARGETS
MAX_ENDPOINT_DISTANCE_H = reference.LAB_MATCH_MAX_DELTA_HOURS
DEFAULT_SPLIT_POLICY = "reuse_exp2"
INTERPOLATION = "piecewise_linear_numpy_interp_no_extrapolation"
VIDEO_LABEL_TIME = "original capture-interval midpoint"
LOSS_LEVEL = "frame"
BATCH_POLICY = "distinct_lab_views"


def training_settings():
    return {
        "architecture":"efficientnet_b0","pretraining":"ImageNet","head_hidden_features":reference.HEAD_HIDDEN_FEATURES,
        "source_resolution":reference.SOURCE_IMAGE_SIZE,"frames_per_video":reference.FRAMES_PER_VIDEO,
        "training_views":list(reference.VIEW_NAMES),"loss_level":LOSS_LEVEL,
        "loss":"unweighted SmoothL1 on train-only median/IQR scaled raw targets; observed preoperative / interpolated postoperative",
        "smooth_l1_beta":reference.SMOOTH_L1_BETA,"frame_batch_size":reference.TRAIN_DISTINCT_LAB_EVENTS*reference.FRAMES_PER_VIDEO,
        "distinct_support_segments_per_batch":reference.TRAIN_DISTINCT_LAB_EVENTS,"base_seed":reference.SEED,
        "imagenet_mean":list(reference.IMAGENET_MEAN),"imagenet_std":list(reference.IMAGENET_STD),
        "crop_scale":reference.CROP_SCALE,"brightness_delta":reference.BRIGHTNESS_DELTA,"contrast_delta":reference.CONTRAST_DELTA,
        "views_per_video_per_batch":1,"head_lr":reference.HEAD_LEARNING_RATE,
        "head_epochs":reference.HEAD_MAX_EPOCHS,"head_patience":reference.HEAD_PATIENCE,
        "finetune_lr":reference.FINETUNE_LEARNING_RATE,"finetune_epochs":reference.FINETUNE_MAX_EPOCHS,
        "finetune_patience":reference.FINETUNE_PATIENCE,"min_lr":reference.MIN_LEARNING_RATE,
        "optimizer":"AdamW","weight_decay":reference.WEIGHT_DECAY,"scheduler":"cosine, no warmup",
        "compile":reference.TORCH_COMPILE_ENABLED,"compile_mode":reference.TORCH_COMPILE_MODE,
        "dropout":.25,"gradient_clip":reference.GRAD_CLIP_NORM,
        "train_decode_workers":reference.TRAIN_NUM_WORKERS,"eval_decode_workers":reference.EVAL_NUM_WORKERS,
        "evaluation":"twenty original frames averaged per video; raw-unit MAE, RMSE, median AE, Pearson/Spearman r, R2; clinical-sign metrics are secondary",
        "independent_models":"one separate encoder/head for each of the eight analytes",
    }
