"""Single source of truth for Practice 3. Edit and record settings before each run."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
RUNS = OUTPUTS / 'runs'
DATA_DIR = OUTPUTS / 'data'
SEED = 42
# Zero-shot reference baseline only (an already fine-tuned SST-2 model).
ZERO_SHOT_MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
FINETUNE_MODEL = 'distilbert/distilbert-base-uncased'
DATASET = 'stanfordnlp/imdb'   # full Hub repository identifier; not 'imdb'
MAX_LENGTH = 128
TRAIN_SAMPLES = 2000
VALIDATION_SAMPLES = 400
TEST_SAMPLES = 500
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EVAL_RUN = os.environ.get('EVAL_RUN', 'lan_1')   # env var lets run_exercise_2.bat pick the run
RUN_TEST = os.environ.get('RUN_TEST', '1') == '1'   # final test only after validation selection; set RUN_TEST=0 for validation
