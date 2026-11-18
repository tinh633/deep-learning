"""Exercise 1 settings (self-contained; shares nothing with Exercise 2)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
SENTIMENT_MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
