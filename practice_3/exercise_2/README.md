# Exercise 2 — Fine-tuning a Pretrained Model for Binary Text Classification

Self-contained: runs without `exercise_1/`. Fine-tunes `distilbert/distilbert-base-uncased` on a
small IMDb subset (train 2000 / validation 400 / test 500, seed 42). **No data to prepare** —
`stanfordnlp/imdb` is downloaded from the Hub automatically (internet needed on first run).

## Requirement → evidence

| Assignment step | Where | Evidence |
|---|---|---|
| 1. Install `transformers`, `datasets`, `evaluate` | `requirements.txt` | `pip show` |
| 2. Load a binary dataset | `src/2_preparedata.py` | `outputs/data/split_info.json` |
| 3. Pretrained model + tokenizer | `src/3_train.py` | `outputs/runs/lan_N/config.json` |
| 4. Preprocess | `src/2_preparedata.py` | truncation at 128 tokens, dynamic padding |
| 5. Training arguments | `src/3_train.py`, `src/config.py` | `config.json` |
| 6. `Trainer` fine-tuning | `src/3_train.py` | `history.json`, `train_metrics.json`, `TRAINING_COMPLETE` |
| 7. Evaluate | `src/4_evaluate.py`, `src/7_baseline.py` | `validation_metrics.json`, `test_metrics.json`, `outputs/baseline/` |

## Run (from this folder)

**Windows shortcut:** double-click `run_exercise_2.bat` — it sets up a venv, runs tests, prepares data, trains a new `lan_N`, evaluates on validation, writes the report, then asks before using the test set. (`EVAL_RUN` / `RUN_TEST` can also be set as environment variables.)

Manual commands:

```bash
cd exercise_2
python -m pip install -r requirements.txt
python -m pytest tests -q
python src/2_preparedata.py          # download + split + tokenize
python src/0_eda.py                  # optional EDA
python src/3_train.py                # new outputs/runs/lan_N, never overwrites
python src/7_baseline.py             # majority + zero-shot reference (same split)
python src/4_evaluate.py             # needs RUN_TEST=False in src/config.py
python src/5_report.py               # figures + visualization/lan_N/REPORT.md
python src/8_log_mlflow.py           # optional: log the run to MLflow (pip install mlflow)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://127.0.0.1:5000
```

Final test, once, after the checkpoint is frozen by validation: set `EVAL_RUN='lan_N'` and
`RUN_TEST=True` in `src/config.py`, then `python src/4_evaluate.py --confirm-test` and
`python src/5_report.py`. `test_metrics.json` is never overwritten.

Extra reading: `Experiment_Plan.md`, `Training.md`, `docs/`. Notebook:
`Exercise_2_Report.ipynb` (read-only, run from this folder).

## Layout

```text
exercise_2/
├── README.md, requirements.txt, Experiment_Plan.md, Training.md
├── Exercise_2_Report.ipynb
├── src/{config,utils,metrics}.py, 0_eda.py, 2_preparedata.py, 3_train.py,
│       4_evaluate.py, 5_report.py, 6_infer_finetuned.py, 7_baseline.py, 8_log_mlflow.py
├── docs/01_problem … 07_results_interpretation
├── tests/
├── outputs/baseline/            # data/ and runs/ are generated locally (large, git-ignored)
└── visualization/lan_1/         # REPORT.md + 3 figures
```

Included result (`lan_1`): test accuracy 84.00%, F1 84.85% (500 reviews); zero-shot SST-2
81.80% / 81.39%; majority class 50.80%. Single seed, small subset — not a full IMDb benchmark.
