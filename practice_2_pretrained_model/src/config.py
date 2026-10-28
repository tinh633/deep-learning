from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / 'data'
OUTPUT_FOLDER = PROJECT_FOLDER / 'outputs'
SPLIT_FILE = OUTPUT_FOLDER / 'cifar10_split.json'

# Worker processes hit Windows shared-memory error 1455 during the longer run.
DATA = dict(seed=42, val_size=5000, image_size=224, batch_size=64,
            num_workers=0, pin_memory=False)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAME = 'vgg16'
PRETRAINED = True
EXPERIMENT = 'A'
MODEL_NAMES = ('vgg16', 'resnet18', 'densenet121')
EXPERIMENTS = {
    'A': dict(head='H1', fine_tune=False, lr=0.001, batch_size=64),
    'B': dict(head='H1', fine_tune=False, lr=0.0001, batch_size=64),
    'C': dict(head='H1', fine_tune=False, lr=0.001, batch_size=32),
    'D': dict(head='H2', fine_tune=False, lr=0.001, batch_size=64),
    'E': dict(head='H1', fine_tune=True, lr=0.001, batch_size=64),
    'F': dict(head='H1', fine_tune=True, lr=0.0001, batch_size=64),
}
TRAIN = dict(seed=42, epochs=20, weight_decay=0.0)

RESUME_RUN = 'lan_2'  # None creates a new run; a run name resumes its last completed epoch.

EVAL_RUN = 'lan_1'
RUN_TEST = False  # Enable only after validation review and locking checkpoints.
