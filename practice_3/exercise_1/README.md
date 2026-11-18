# Exercise 1 — Sentiment Analysis with Hugging Face

Self-contained: runs without `exercise_2/`. **No training and no dataset** — a pretrained
Hub model classifies sentences you provide.

Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

## Requirement → evidence

| Assignment step | Where | Evidence |
|---|---|---|
| 1. Install `transformers` | `requirements.txt` | `pip show transformers` |
| 2. Pretrained model from the Hub | `src/sentiment.py` | `model_id` in the JSON |
| 3. Tokenize a sentence | `AutoTokenizer` | `tokens`, `token_ids`, `attention_mask` |
| 4. Sentiment analysis | `pipeline('sentiment-analysis')` | `predicted_label`, `model_score` |

## Run (from this folder)

**Windows shortcut:** double-click `run_exercise_1.bat` (venv + install + tests + run).

Manual commands:

```bash
cd exercise_1
python -m pip install -r requirements.txt
python src/sentiment.py                       # 4 default sentences
python src/sentiment.py --sentence "This film is surprisingly good."
python -m pytest tests -q
```

Output: `outputs/sentiment_result.json` (overwritten on every run — copy it if you need
to keep an earlier one). Notebook: `Exercise_1_Report.ipynb` (read-only, run from this folder).

## Layout

```text
exercise_1/
├── README.md
├── requirements.txt
├── Exercise_1_Report.ipynb
├── src/{config.py, utils.py, sentiment.py}
├── docs/sentiment.md
├── tests/test_utils.py
└── outputs/sentiment_result.json
```

The model score is not a calibrated probability of being correct.
