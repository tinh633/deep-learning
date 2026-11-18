# Practice 3 — Alignment with Chapter 4 (Fine-Tune Models and Training Algorithms)

**Course source:** PhD Nguyễn Thị Khánh Tiên, *Chapter 4 — Fine-Tune Models and Training Algorithms*, pp. 2–25. This document distinguishes the required Practice 3 exercise (p. 25) from optional teaching concepts in the chapter.

## Required deliverables (p. 25)

| Requirement | Code/evidence | Outcome |
|---|---|---|
| Ex1: install Transformers | `requirements.txt` | available for installation |
| Ex1: use a pretrained sentiment classifier | `src/1_sentiment.py` | DistilBERT SST-2 |
| Ex1: tokenize sentence | `src/1_sentiment.py` | outputs generated when run |
| Ex1: infer sentiment | Exercise 1 console or `outputs/exercise1/` | generate prediction evidence |
| Ex2: install Transformers, Datasets, Evaluate | `requirements.txt` | available for installation |
| Ex2: load labeled binary dataset | `src/2_preparedata.py` | IMDb |
| Ex2: pretrained model and tokenizer | `src/3_train.py` | DistilBERT encoder; new 2-class head |
| Ex2: preprocessing | `src/2_preparedata.py` | tokenize, truncate at 128, dynamic padding |
| Ex2: training arguments | `src/config.py`, `src/3_train.py` | batch 8; LR 2e-5; 3 epochs; weight decay 0.01 |
| Ex2: Trainer fine-tuning | `src/3_train.py` | run and verify `TRAINING_COMPLETE` |
| Ex2: evaluate | `src/4_evaluate.py` | generate validation and final test evidence |

## Connection to lecture concepts

| Lecture section | Relevance to the actual experiment |
|---|---|
| Fine-tuning, pp. 2–4 | Start from pretrained language-model weights, initialize a new classification head, then update weights on IMDb. Unlike an optional freeze/unfreeze experiment, this run did not separately compare frozen and unfrozen encoders. |
| Loss and optimizer, pp. 5–10 | Binary sentiment is implemented using a **2-logit classification head and categorical cross entropy**, not a single-logit BCE loss. Hugging Face Trainer normally uses AdamW; verify run-specific optimizer details before claiming defaults. |
| Learning rate scheduling, pp. 11–14 | The training script can log learning-rate changes. The Trainer controls scheduling; code does not manually call `scheduler.step()`. |
| Data handling, pp. 16–18 | Hugging Face Dataset + Trainer batching is analogous to PyTorch Dataset/DataLoader. Image transforms shown in the lecture are not appropriate to copy directly to text. |
| Regularization, pp. 20–21 | Weight decay 0.01 and model-internal dropout regularize training. **Best checkpoint selection is not early stopping**: the script uses `load_best_model_at_end=True` and selects by validation F1, without necessarily stopping early. |
| Gradient clipping, pp. 22–23 | Do not claim a tested clipping experiment: the starter does not define a separately configured clipping comparison. |
| Logging, p. 24 | `history.json`, `train.log`, checkpoint records and final metrics provide traceability. |

## Results to complete after each team run

No generated results are bundled in the team starter. For each run, fill out:

| Run | Learning rate | Max tokens | Epochs | Best validation F1 | Final test F1 |
|---|---|---|---|---|---|
| `lan_N` | TBD | TBD | TBD | TBD | Only after selection |

Include training/validation loss curves, classification report, confusion matrix, data visuals, model-selection rationale, and a discussion of limitations. Do not claim early stopping unless training actually stops early.

## Required EDA and reporting

1. Execute `src/2_preparedata.py`, then `src/0_eda.py`.
2. For pre-truncation token lengths, execute `src/0_eda.py --raw`.
3. Complete the report with actual metrics; report validation and final test separately.

All numerical findings must come from generated artifacts of the named run.
