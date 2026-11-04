"""What: Benchmark configuration for six ImageNet-1K classifiers.
Why: Compare the same labeled images and fixed measurement settings.
Who: The benchmark script reads these values.
When: Before an explicitly started inference run.
Where: Local data/manifest.csv and outputs/.
How: Explicit pretrained variants, FP32, fixed batches and repeated timing.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'imagenet1k' / 'manifest.csv'
OUTPUT = ROOT / 'outputs'
MODELS = {
    'vgg16': 'VGG16_Weights.IMAGENET1K_V1',
    'googlenet': 'GoogLeNet_Weights.IMAGENET1K_V1',
    'resnet50': 'ResNet50_Weights.IMAGENET1K_V1',
    'densenet121': 'DenseNet121_Weights.IMAGENET1K_V1',
    'efficientnet_b0': 'EfficientNet_B0_Weights.IMAGENET1K_V1',
    'mobilenet_v2': 'MobileNet_V2_Weights.IMAGENET1K_V1',
}
# Manifest labels must be torchvision ImageNet-1K indices 0..999.
DATASET_NAME = 'ImageNet-1K'
SPLIT = 'validation'
SEED = 42
MAX_IMAGES = 500
BATCH_SIZE = 16
WARMUP = 20
REPEATS = 3
TIMING_REQUESTS = 100

HF_DATASET = 'ILSVRC/imagenet-1k'
IMAGES_PER_CLASS = 50
SELECTED_CLASSES = {
    207: 'golden retriever', 281: 'tabby', 340: 'zebra',
    404: 'airliner', 779: 'school bus', 671: 'mountain bike',
    504: 'coffee mug', 620: 'laptop', 949: 'strawberry', 980: 'volcano',
}
