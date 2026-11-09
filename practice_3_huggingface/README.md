# Practice 3 — Get Started with Hugging Face

> **Coursework:** sentiment analysis (inference) and binary text classification (fine-tuning).  
> **Status:** team starter; source and documentation retained, all generated outputs removed. No completed run is included.  
> **Design:** preserve Practice 1/2 conventions (`src`, `docs`, `outputs`, `visualization`, `progress`, experiment report); make runs auditable and avoid data leakage.


## Team workflow (read first)

This is the **unrun team starter**. Previously generated dataset files, training checkpoints, metrics, plots, and reports are intentionally not bundled. The original experiment artifacts should be backed up separately; this package does not delete anything on your computer.

- **Exercise 1:** [`exercise_1/WORKFLOW.md`](exercise_1/WORKFLOW.md) — sentiment inference, independent of Exercise 2.
- **Exercise 2:** [`exercise_2/WORKFLOW.md`](exercise_2/WORKFLOW.md) — data preparation, fine-tuning, validation, test, reporting.
- **File map:** [`FOLDER_TREE.md`](FOLDER_TREE.md) — original runnable Python files remain under `src/` to avoid breaking imports and paths.

**Do not merge generated artifacts** from multiple members into the same run. Each member uses a separate branch and records their actual configuration and evidence.

## 1. Requirements-to-deliverables matrix

| Assignment requirement | Implementation | Evidence |
|---|---|---|
| Ex1.1 Install `transformers` | `requirements.txt` | `python -m pip show transformers` |
| Ex1.2 Pretrained sentiment model from Hub | `src/1_sentiment.py` | `outputs/exercise1/sentiment_result.json`: `model_id` |
| Ex1.3 Tokenize example | `AutoTokenizer` in `1_sentiment.py` | tokens, IDs and attention mask in JSON |
| Ex1.4 Perform sentiment analysis | `pipeline('sentiment-analysis')` | label and model score in JSON |
| Ex2.1 Install transformers/datasets/evaluate | `requirements.txt` | dependency inspection; `evaluate` is installed (metrics implemented with sklearn, avoiding downloads) |
| Ex2.2 Load binary dataset | `src/2_preparedata.py` | `outputs/data/split_info.json`, labels/hashes |
| Ex2.3 Pretrained model/tokenizer | `src/3_train.py` | `config.json`, saved model/tokenizer |
| Ex2.4 Preprocess data | tokenize, truncate, dynamic padding | saved Arrow dataset splits |
| Ex2.5 Define training arguments | `TrainingArguments` in `3_train.py` | logged run config; checkpoints |
| Ex2.6 Trainer + fine-tune | `Trainer.train` | `history.json`, `train_metrics.json`, `TRAINING_COMPLETE` |
| Ex2.7 Evaluate fine-tuned model | `src/4_evaluate.py` | validation/test JSON with confusion and per-class metrics |

## 2. Complete structure

```text
practice_3_huggingface/
├── Practice_3_Report.ipynb       # read-only lab narrative; no automatic training
├── README.md                     # scope, commands, artifact contract
├── Practice_3_Requirements.md   # verbatim checklist, evidence map
├── Experiment_Plan.md            # data, baseline, controls, evaluation protocol
├── Training.md                   # runbook, hardware, recovery, troubleshooting
├── AI_usage.md                   # transparent AI assistance statement
├── AI_Usage_Step.md, docs/AI_Usage_Full.md
├── progress.md                   # milestone ledger (unverified ≠ completed)
├── requirements.txt
├── docs/
│   ├── 01_problem.md
│   ├── 02_data_prepare.md
│   ├── 03_sentiment.md
│   ├── 04_training.md
│   ├── 05_evaluation.md
│   ├── 06_reproducibility.md
│   └── 07_results_interpretation.md
├── src/
│   ├── config.py, utils.py, metrics.py
│   ├── 0_eda.py (optional EDA)
│   ├── 1_sentiment.py
│   ├── 2_preparedata.py
│   ├── 3_train.py
│   ├── 4_evaluate.py
│   ├── 5_report.py
│   └── 6_infer_finetuned.py (optional demonstration)
├── tests/                         # synthetic local unit tests
├── outputs/                       # generated; ignored from Git by default
│   ├── exercise1/sentiment_result.json
│   ├── data/{train,validation,test,split_info.json}
│   └── runs/lan_N/{config,history,model,checkpoints,metrics,...}
└── visualization/                 # generated figures and REPORT.md
```

## 3. Setup — run from repository root

```fish
cd <repository-root>   # the folder that contains practice_3_huggingface/
python -m pip install -r practice_3_huggingface/requirements.txt
python -m pytest practice_3_huggingface/tests -q
```

First download requires Internet. HF_TOKEN is optional for public models/datasets; an unauthenticated-rate-limit warning does not itself indicate failure. For reliable reproduction record exact dependency versions via `python -m pip freeze`.

## 4. Run steps

```fish
python practice_3_huggingface/src/1_sentiment.py
python practice_3_huggingface/src/2_preparedata.py
python practice_3_huggingface/src/0_eda.py  # optional EDA; label and token length distributions
python practice_3_huggingface/src/3_train.py
python practice_3_huggingface/src/4_evaluate.py
python practice_3_huggingface/src/7_baseline.py   # zero-shot SST-2 + majority-class reference, same split
python practice_3_huggingface/src/5_report.py
```

**Important:** `RUN_TEST=False` by default, so the first evaluation is **validation recheck**, not a final test. After reviewing/locking the model by validation, set `EVAL_RUN='lan_N'` and `RUN_TEST=True` in `src/config.py`, then run:

```fish
python practice_3_huggingface/src/4_evaluate.py --confirm-test
python practice_3_huggingface/src/5_report.py
```

The test script refuses to overwrite an existing test report. Set `EVAL_RUN` to the actual completed run, not simply `lan_1` if you already trained multiple times.

## 5. Experiment design

- **Exercise 1** uses `distilbert/distilbert-base-uncased-finetuned-sst-2-english`: a sentiment classifier already fine-tuned for English positive/negative labels. No backward pass.
- **Exercise 2** starts from `distilbert/distilbert-base-uncased`: pretrained language encoder + randomly initialized binary classification head, then updates model weights for IMDb movie reviews.
- **Data**: `stanfordnlp/imdb` (official 25,000 train, 25,000 test, plus unlabeled). This practice selects 2,000 train + 400 validation (from official training) and reserves 500 official test reviews; seed 42. This is a **small teaching subset**, not a benchmark claim.
- **Training**: learning rate 2e-5, batch 8, 3 epochs, weight decay .01, max length 128, select by highest validation F1. Only adjust settings *before* the final test.
- **Preprocessing**: tokenizer uses truncation; batches use dynamic padding. Binary labels: 0 = NEGATIVE, 1 = POSITIVE.
- **Metrics**: accuracy, positive-class precision/recall/F1, per-class report, 2×2 confusion matrix, and ROC-AUC when applicable.

## 6. Artifact policy and reporting

- `outputs/data/`: saved tokenized splits and original-split hashes, not raw review dumps.
- `outputs/runs/lan_N/config.json`: true settings/environment for that run; `history.json`: actual Trainer log history; `TRAINING_COMPLETE`: written only if training completed.
- `model/`: fine-tuned weights + tokenizer. `checkpoints/`: Trainer checkpoints; may occupy hundreds of MB. **Never store fake metrics**.
- `test_metrics.json`: only created by the explicit final test command; includes predictions, label ordering and confusion matrix.
- `visualization/lan_N`: plots and evidence-based `REPORT.md` after running `5_report.py` (none included in this starter).
- Git ignores generated data/checkpoints by default. Only force-add selected, verified metrics if the course requires them, after checking size and privacy.

## 7. Read in this order

1. [Requirements](Practice_3_Requirements.md) → [Experiment plan](Experiment_Plan.md)
2. [Problem framing](docs/01_problem.md) → [Data preparation](docs/02_data_prepare.md)
3. [Exercise 1](docs/03_sentiment.md) → [Training](docs/04_training.md)
4. [Evaluation](docs/05_evaluation.md) → [Reproducibility](docs/06_reproducibility.md)
5. [Interpretation](docs/07_results_interpretation.md) → [Notebook report](Practice_3_Report.ipynb)
6. [Operation runbook](Training.md), [progress](progress.md)

## 8. Primary sources

- [Hugging Face fine-tuning guide](https://huggingface.co/docs/transformers/en/training)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [IMDb dataset card](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Sentiment analysis tutorial](https://huggingface.co/blog/sentiment-analysis-python)
- [Original IMDb paper (Maas et al., 2011)](https://aclanthology.org/P11-1015/)
- [Kaggle sentiment example](https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face)
- [Kaggle BERT fine-tuning example](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)

## 5. Additions in this revision

- `src/metrics.py` now uses the Hugging Face `evaluate` library (accuracy, precision, recall, F1) as in the course tutorial, with an automatic scikit-learn fallback when metric modules cannot be downloaded.
- `src/1_sentiment.py` analyses several sentences (positive, negative, lukewarm, negated) and truncates long input.
- `src/7_baseline.py` reports a majority-class floor and a zero-shot SST-2 baseline on the same split and truncation length as the fine-tuned model.
- `src/5_report.py` derives the error-asymmetry sentence from the confusion matrix and adapts plot limits.
- Logging uses `TENSORBOARD_LOGGING_DIR` (the transformers v5 mechanism); `logging_dir` and `warmup_ratio` are not used. Record `pip freeze` after your first successful run.
