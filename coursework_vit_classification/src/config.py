"""What: Đường dẫn và split CIFAR-100. Why: Dùng chung cấu hình dữ liệu.
Which: CIFAR-100, ConvNeXt-Base/ResNet-50/ViT-B16, A-D. When: Khi import.
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
    'B': {'model': 'resnet50', 'strategy': 'fine_tune_layer4'},  # layer4 + fc = 15.169.636 tham số học
    'C': {'model': 'vit_b_16', 'strategy': 'freeze'},
    'D': {'model': 'vit_b_16', 'strategy': 'fine_tune'},
}

# A đã hoàn tất pilot/final; B đang ở pilot; smoke có cấu hình riêng bên dưới.
PHASE = 'final'  # 'tuning' cho pilot LR; 'final' cho lượt chính thức.
RUN_PURPOSE = 'experiment'  # 'resource_probe' chỉ đo khả năng chạy, không dùng chọn LR final.
SMOKE_EXPERIMENTS = ('B',)
SMOKE_BATCH_SIZE = 2
SMOKE_TRAIN_IMAGES = 20
SMOKE_VAL_IMAGES = 4
SMOKE_LR_FREEZE = 1e-3
SMOKE_LR_FINE_TUNE = 1e-4
SMOKE_WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
USE_AMP = True

# B (ResNet-50, chỉ học layer4 + fc): pilot LR1/LR2 mỗi 5 epoch, final 20 epoch khởi tạo mới, cùng protocol A.
# Số epoch/LR chưa được nhóm chốt riêng cho B: giữ ngân sách của A, LR fine-tune thấp hơn freeze.
TRAINING_APPROVED = True  # Đặt False sau khi xong để chặn chạy lặp.
DUPLICATE_POLICY = 'keep_reported'  # Giữ split và công bố các nhóm trùng; không đổi giữa pilot/final.
EXPERIMENT = 'B'
CANDIDATE = None  # Pilot: LR1=1e-4 rồi LR2=3e-5. Final: PHASE='final', CANDIDATE=None, SELECTED_FROM='lan_N' của pilot thắng.
SELECTED_FROM = 'lan_19'
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.0
EPOCHS = 20  # Final đổi thành 20.
RESUME_RUN = None

EVAL_RUN = 'lan_14'
EVAL_CHECKPOINT = 'best.pt'
EVAL_SPLIT = 'validation'
RUN_TEST = False
ALLOW_SMOKE_CHECKPOINT = False
MAX_EVAL_IMAGES = None
DEMO_IMAGE = None
