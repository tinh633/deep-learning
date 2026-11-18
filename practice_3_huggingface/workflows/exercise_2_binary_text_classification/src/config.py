"""Default baseline. Change one setting per controlled experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "data" / "imdb_baseline"
RUNS = ROOT / "outputs" / "runs"
VISUALIZATION = ROOT / "visualization"

DATASET_ID = "stanfordnlp/imdb"
MODEL_ID = "distilbert/distilbert-base-uncased"
ID_TO_LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
SEED = 42
TRAIN_SAMPLES = 2000
VALIDATION_SAMPLES = 400
TEST_SAMPLES = 500
MAX_LENGTH = 128

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
