"""What: Đường dẫn và split CIFAR-100. Why: Dùng chung cấu hình dữ liệu.
Which: CIFAR-100, ConvNeXt-Base/ViT-B16, A-D. When: Khi import.
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
    'A': {'model': 'convnext_base', 'strategy': 'freeze'},
    'B': {'model': 'convnext_base', 'strategy': 'fine_tune'},
    'C': {'model': 'vit_b_16', 'strategy': 'freeze'},
    'D': {'model': 'vit_b_16', 'strategy': 'fine_tune'},
}

# A đã hoàn tất pilot/final; smoke có cấu hình riêng bên dưới.
PHASE = 'final'
RUN_PURPOSE = 'experiment'  # 'resource_probe' chỉ đo khả năng chạy, không dùng chọn LR final.
SMOKE_EXPERIMENTS = ('A',)
SMOKE_BATCH_SIZE = 2
SMOKE_TRAIN_IMAGES = 20
SMOKE_VAL_IMAGES = 4
SMOKE_LR_FREEZE = 1e-3
SMOKE_LR_FINE_TUNE = 1e-4
SMOKE_WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
USE_AMP = True

# A được duyệt 18/11: smoke trước, LR1/LR2 mỗi 5 epoch, final 20 epoch khởi tạo mới.
TRAINING_APPROVED = False  # A đã xong 20/20; chặn vô tình chạy lặp. Bật lại khi chủ động resume/train.
DUPLICATE_POLICY = 'keep_reported'  # Giữ split và công bố các nhóm trùng; không đổi giữa pilot/final.
EXPERIMENT = 'A'
CANDIDATE = None  # 'LR1'/'LR2' khi PHASE='tuning'.
SELECTED_FROM = 'lan_13'  # LR2 được chọn theo validation.
BATCH_SIZE = 16  # Chỉ bắt đầu sau khi kiểm tra tài nguyên đạt.
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
EPOCHS = 20
RESUME_RUN = 'lan_14'

EVAL_RUN = 'lan_14'
EVAL_CHECKPOINT = 'best.pt'
EVAL_SPLIT = 'validation'
RUN_TEST = False
ALLOW_SMOKE_CHECKPOINT = False
MAX_EVAL_IMAGES = None
DEMO_IMAGE = None
