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
