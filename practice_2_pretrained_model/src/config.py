import hashlib
from pathlib import Path


# Paths
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / 'data'
OUTPUT_FOLDER = PROJECT_FOLDER / 'outputs'
SPLIT_FILE = OUTPUT_FOLDER / 'cifar10_split.json'

# 1. Chọn model và thí nghiệm.
MODEL_NAME = 'vgg16'
PRETRAINED = True
EXPERIMENT = 'A'

# 2. Tham số dùng chung cho các thí nghiệm.
SEED = 42
EPOCHS = 20
WEIGHT_DECAY = 0.0
IMAGE_SIZE = 224
VAL_SIZE = 5000
NUM_WORKERS = 0  # Workers above zero previously caused Windows error 1455.
PIN_MEMORY = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 3. Chọn lượt train/evaluate.
RESUME_RUN = 'lan_2'  # None starts a new run.
EVAL_RUN = 'lan_1'
RUN_TEST = False

# Chỉ đổi EXPERIMENT ở trên để chọn một hàng trong bảng A-F.
MODEL_NAMES = ('vgg16', 'resnet18', 'densenet121')
EXPERIMENTS = {
    'A': dict(head='H1', fine_tune=False, lr=0.001, batch_size=64),
    'B': dict(head='H1', fine_tune=False, lr=0.0001, batch_size=64),
    'C': dict(head='H1', fine_tune=False, lr=0.001, batch_size=32),
    'D': dict(head='H2', fine_tune=False, lr=0.001, batch_size=64),
    'E': dict(head='H1', fine_tune=True, lr=0.001, batch_size=64),
    'F': dict(head='H1', fine_tune=True, lr=0.0001, batch_size=64),
}
HEAD = EXPERIMENTS[EXPERIMENT]['head']
FINE_TUNE = EXPERIMENTS[EXPERIMENT]['fine_tune']
LEARNING_RATE = EXPERIMENTS[EXPERIMENT]['lr']
BATCH_SIZE = EXPERIMENTS[EXPERIMENT]['batch_size']

# These dictionaries retain the format used by existing data/checkpoint files.
DATA = dict(seed=SEED, val_size=VAL_SIZE, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
TRAIN = dict(seed=SEED, epochs=EPOCHS, weight_decay=WEIGHT_DECAY)


def get_settings():
    """Describe this run for logging and checkpoint compatibility checks."""
    with SPLIT_FILE.open('rb') as source:
        split_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    return {
        'model_name': MODEL_NAME,
        'experiment': EXPERIMENT,
        'model_config': dict(head=HEAD, fine_tune=FINE_TUNE, lr=LEARNING_RATE, batch_size=BATCH_SIZE),
        'pretrained': PRETRAINED,
        'train': dict(seed=SEED, epochs=EPOCHS, weight_decay=WEIGHT_DECAY),
        'data': dict(DATA),
        'split_file': str(SPLIT_FILE),
        'split_sha256': split_hash,
        'source': 'torchvision.datasets.CIFAR10',
        'input_shape': [3, IMAGE_SIZE, IMAGE_SIZE],
    }
