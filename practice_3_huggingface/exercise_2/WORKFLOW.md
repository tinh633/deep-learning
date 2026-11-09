# Exercise 2 — Fine-tuning a Pretrained Model for Binary Text Classification

## Goal (Chapter 4, Practice 3, page 25)
Install Hugging Face libraries, load a labeled text dataset, load a pretrained model/tokenizer, preprocess, define training arguments, fine-tune using `Trainer`, and evaluate the model.

## Owner & status
- Owner(s): TBD
- Status: source ready; training and final-test artifacts **not included** in this starter.

## Agreed baseline (change only before running)
- Model: `distilbert/distilbert-base-uncased` with two-class head.
- Dataset: `stanfordnlp/imdb` (subset: train 2000, val 400, test 500).
- Seed 42, batch 8, max length 128, epochs 3, LR `2e-5`, weight decay `0.01`.
- Read/write settings in `src/config.py`; set `RUN_TEST=False` for development and validation.

## Workflow from repository root
1. Install dependencies and run smoke tests:

   ```fish
   python -m pip install -r practice_3_huggingface/requirements.txt
   python -m pytest practice_3_huggingface/tests -q
   ```

2. **Prepare data** (creates tokenized splits):

   ```fish
   python practice_3_huggingface/src/2_preparedata.py
   ```

3. **Explore data** (plots saved under `visualization/eda/`):

   ```fish
   python practice_3_huggingface/src/0_eda.py
   python practice_3_huggingface/src/0_eda.py --raw
   ```

   `--raw` may download data. Stored 128-token lengths cannot establish how many inputs were truncated; `--raw` measures lengths *before* truncation.

4. **Fine-tune** (creates a fresh `outputs/runs/lan_N/`, saves configuration and model):

   ```fish
   python practice_3_huggingface/src/3_train.py
   ```

5. **Validate**: set `EVAL_RUN` to your completed run, keep `RUN_TEST=False`, then run:

   ```fish
   python practice_3_huggingface/src/4_evaluate.py
   python practice_3_huggingface/src/5_report.py
   ```

6. **Final test ONCE, after freezing configuration/checkpoint**: change `RUN_TEST=True`, then:

   ```fish
   python practice_3_huggingface/src/4_evaluate.py --confirm-test
   python practice_3_huggingface/src/5_report.py
   ```

7. **Optional demo** (requires saved model):

   ```fish
   python practice_3_huggingface/src/6_infer_finetuned.py --sentence 'This movie was fantastic.'
   ```

## Inputs and outputs
- Scripts: `src/2_preparedata.py`, `src/0_eda.py`, `src/3_train.py`, `src/4_evaluate.py`, `src/5_report.py`, `src/6_infer_finetuned.py`.
- Prepared dataset: `outputs/data/`, with split provenance in `split_info.json`.
- Training runs: `outputs/runs/lan_N/` (`config.json`, `history.json`, metrics and model).
- Plots/report: `visualization/eda/`, `visualization/lan_N/`.
- **These generated files are NOT present in the starter ZIP.**

## Checklist before submitting
- [ ] True data counts/labels and split seed documented.
- [ ] Preprocessing and model head described.
- [ ] TrainingArguments/optimizer settings recorded.
- [ ] `TRAINING_COMPLETE`, `history.json`, validation metrics exist.
- [ ] Training/validation graphs and EDA generated.
- [ ] Checkpoint selected using validation only.
- [ ] Held-out test evaluated only after selecting model.
- [ ] Report separates validation and test and describes overfitting/limitations.

## Coordination rules
Everyone uses their own Git branch and experiment run name; do not overwrite colleagues' results. If comparing hyperparameters, change one variable at a time, use the same split/seed, select by validation and avoid repeatedly tuning on test. Do not push heavy model checkpoints to GitHub.
