"""What: Đường dẫn và split CIFAR-100. Why: Dùng chung cấu hình dữ liệu.
Who: data/EDA và pipeline sau này. When: Khi import.
Where: data/, outputs/ của coursework. How: pathlib và hằng số tường minh.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / 'data'
OUTPUT_DIR = PROJECT / 'outputs'
SPLIT_FILE = OUTPUT_DIR / 'cifar100_split.json'
SEED = 42
VALIDATION_SIZE = 5000

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPERIMENTS = {
    'A': {'model': 'resnet50', 'strategy': 'freeze'},
    'B': {'model': 'resnet50', 'strategy': 'fine_tune'},
    'C': {'model': 'vit_b_16', 'strategy': 'freeze'},
    'D': {'model': 'vit_b_16', 'strategy': 'fine_tune'},
}

# Chỉ kiểm tra kỹ thuật; đây chưa phải thông số tuning/final đã chốt.
PHASE = 'smoke'
RUN_PURPOSE = 'experiment'  # 'resource_probe' chỉ đo khả năng chạy, không dùng chọn LR final.
SMOKE_EXPERIMENTS = ('A', 'B', 'C', 'D')
SMOKE_BATCH_SIZE = 2
SMOKE_TRAIN_IMAGES = 20
SMOKE_VAL_IMAGES = 4
SMOKE_LR_FREEZE = 1e-3
SMOKE_LR_FINE_TUNE = 1e-4
SMOKE_WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
USE_AMP = True

# Chưa chốt protocol: khai báo rõ trước khi bật tuning/final, không tự dùng LR smoke.
TRAINING_APPROVED = False
DUPLICATE_POLICY = None  # Hỗ trợ 'keep_reported'; lọc duplicate cần một split version mới.
EXPERIMENT = 'A'
CANDIDATE = None  # 'LR1'/'LR2' khi PHASE='tuning'.
SELECTED_FROM = None  # Tên run pilot được chọn khi PHASE='final'.
BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
EPOCHS = None
RESUME_RUN = None

EVAL_RUN = None
EVAL_CHECKPOINT = 'best.pt'
EVAL_SPLIT = 'validation'
RUN_TEST = False
ALLOW_SMOKE_CHECKPOINT = False
MAX_EVAL_IMAGES = None
DEMO_IMAGE = None
