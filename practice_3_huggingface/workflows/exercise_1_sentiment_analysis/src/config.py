"""Model and workflow-local paths for Exercise 1."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SENTENCE = "The film started slowly, but I absolutely loved its ending."
RESULT_FILE = ROOT / "outputs" / "sentiment_result.json"
MAX_LENGTH = 512
