"""Single source of truth for Practice 3. Edit and record settings before each run."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
RUNS = OUTPUTS / 'runs'
DATA_DIR = OUTPUTS / 'data'
SEED = 42
SENTIMENT_MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
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
EVAL_RUN = 'lan_1'            # change to a completed run only
RUN_TEST = True             # final test only after locking validation-selected checkpoint
