# Practice 3 — Operating runbook (CachyOS / Fish)

## Prerequisites

- Work from repository root. Python 3.10+ recommended. The project depends on PyTorch, Transformers, Datasets, Evaluate, Accelerate, NumPy, scikit-learn, Matplotlib and TensorBoard.
- The first run downloads model checkpoints (~hundreds MB) and IMDb (~hundreds MB downloaded/cache after expansion); check disk and network.
- For GPUs with around 4 GB VRAM, batch 8 at max length 128 is only an initial estimate. If OOM, reduce `BATCH_SIZE=4` or `MAX_LENGTH=96`, **start a new run**, and document the change.
- On CPU the same code works, but full fine-tuning can be slow.

## 1. Setup and validation

```fish
cd ~/Documents/deep-learning-Th-i
source .venv/bin/activate.fish  # only if not already active
python -m pip install -r practice_3_huggingface/requirements.txt
python -m pytest practice_3_huggingface/tests -q
python -m pip show transformers datasets evaluate accelerate
```

`HF_TOKEN` is optional for public repos. Do not commit tokens. You can set one temporarily in Fish using `set -gx HF_TOKEN ...` after authenticating; never paste it in README or screenshots.

## 2. Exercise 1: inference

```fish
python practice_3_huggingface/src/1_sentiment.py
python practice_3_huggingface/src/1_sentiment.py --sentence 'The ending was disappointing.'
```

Both calls write `outputs/exercise1/sentiment_result.json`; **the second replaces the first**. Save an independent copy if a submission requires both. Tokenizer/pipeline outputs are saved with the exact input sentence.

## 3. Exercise 2: prepare, train, evaluate

```fish
python practice_3_huggingface/src/2_preparedata.py
python practice_3_huggingface/src/0_eda.py  # optional
python practice_3_huggingface/src/3_train.py
```

`2_preparedata.py` overwrites prepared split directories on rerun; **do not change split settings midway through an experiment**. Inspect `outputs/data/split_info.json` first. `3_train.py` automatically chooses a new `outputs/runs/lan_N` and never overwrites a previous run. Verify `TRAINING_COMPLETE` before evaluating.

Validation recheck:

```fish
python practice_3_huggingface/src/4_evaluate.py
python practice_3_huggingface/src/5_report.py
```

Check `src/config.py`: `EVAL_RUN` must match the completed run and `RUN_TEST=False`. If `lan_1` does not match, change it before evaluating.

Once the model is selected using validation only, set `RUN_TEST=True` in `src/config.py`:

```fish
python practice_3_huggingface/src/4_evaluate.py --confirm-test
python practice_3_huggingface/src/5_report.py
```

This creates final `test_metrics.json` but refuses to overwrite an existing one. Use a new run for a new evaluation protocol.

## 4. Evidence map

| Location | Meaning |
|---|---|
| `outputs/exercise1/sentiment_result.json` | Input, token IDs, label, score, Hub model ID |
| `outputs/data/split_info.json` | Split sizes, counts per label, hashes, seed |
| `outputs/runs/lan_N/config.json` | Model, parameters, environment, data protocol |
| `outputs/runs/lan_N/history.json` | Actual Trainer log history |
| `outputs/runs/lan_N/checkpoints/` | Trainer checkpoints/optimizer state |
| `outputs/runs/lan_N/model/` | Final selected model and tokenizer |
| `outputs/runs/lan_N/validation_metrics.json` | Automatic evaluation at end of training |
| `outputs/runs/lan_N/validation_recheck.json` | Validation predictions & confusion |
| `outputs/runs/lan_N/test_metrics.json` | Final held-out test predictions & metrics |
| `visualization/lan_N/` | Loss, score, confusion figures and REPORT.md |

## 5. Failure modes and recovery

- **`HfUriError ... got 'imdb'`**: use full dataset ID `stanfordnlp/imdb` in `src/config.py` (already fixed here).
- **`ModuleNotFoundError`**: use `python -m pip install -r ...` from active `.venv`; confirm VS Code interpreter.
- **`CUDA out of memory`**: reduce batch or max length, close competing GPU processes, preserve failed run as evidence and train new `lan_N`.
- **Hugging Face warning about authentication**: public requests usually continue; rate limits may be lower.
- **Interrupted fine-tuning**: partial history saved as `history_partial.json`; do not report as completed or copy final checkpoints blindly.
- **`No module named tensorboard`**: install from requirements; TensorBoard logs are enabled for training.
- **`Invalid version / API argument`**: record `python -m pip freeze`, check currently installed Transformers and `TrainingArguments` API. Code targets recent `eval_strategy` + `processing_class` APIs.
- **`TEST locked`**: a deliberate safeguard. Freeze selection before turning on RUN_TEST and adding `--confirm-test`.
- **Untracked output files in Git**: generated data/checkpoints are intentionally ignored; use `git status` to review. Avoid `git add -A` for large datasets.

## 6. Submission checklist

Source and Markdown files are ready without a training run; **training result artifacts must come from your actual machine**. Provide a screenshot/JSON for Exercise 1; split_info, config/history, model evidence and validation + final test metrics for Exercise 2. Present charts via `Practice_3_Report.ipynb`; note model, seed, split and hardware.
