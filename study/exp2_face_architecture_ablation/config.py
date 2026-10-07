"""Independent architecture controls on the shared 12h view-level protocol."""

from pathlib import Path

from study.common import run_video_loss_12h as reference
from study.common.run_distinct_lab_views_12h import STATE as PREDECESSOR, OUTPUTS as BASELINES


HERE = Path(__file__).resolve().parent
SOURCE = reference.SOURCE
INDEX_PATH = reference.INDEX_PATH
TARGETS = reference.config.TARGETS
VIEWS = reference.config.VIEW_NAMES
SEED = reference.config.SEED
ARCHITECTURES = ("color_histogram_mlp", "color_statistics_mlp", "small_cnn")
HISTOGRAM_BINS = 16
HISTOGRAM_FEATURES = 9 * HISTOGRAM_BINS
STATISTICS_FEATURES = 43
GROUPS_PER_BATCH = 12
FRAME_BATCH_SIZE = 240
MAX_EPOCHS = 80
PATIENCE = 12
WEIGHT_DECAY = 1e-3
DROPOUT = 0.25
LEARNING_RATES = {"color_histogram_mlp": 1e-3, "color_statistics_mlp": 1e-3, "small_cnn": 3e-4}
MIN_LEARNING_RATES = {"color_histogram_mlp": 1e-5, "color_statistics_mlp": 1e-5, "small_cnn": 3e-6}
