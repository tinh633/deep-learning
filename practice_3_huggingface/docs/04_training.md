# 04 — Exercise 2: model fine-tuning through Trainer

## Architecture

- **Base:** `distilbert/distilbert-base-uncased` (a general pretrained English encoder).
- **Head:** `AutoModelForSequenceClassification(...,num_labels=2)` creates task-specific logits for NEGATIVE vs POSITIVE.
- **Output:** float tensor `(batch_size,2)` before softmax; the model's supervised classification loss is cross entropy when labels are supplied.
- **Weights:** the baseline allows ordinary fine-tuning of trainable parameters; unlike Practice 2's frozen-backbone experiment, it does not implement head-only freeze.

## Training computation

For each batch, token IDs and attention mask enter the encoder; the classifier outputs logits. Cross-entropy penalizes low predicted probability of the true class. `Trainer` performs forward pass → loss → backward gradients → optimizer step; over epoch boundaries it runs validation and logs metrics. The optimizer adjusts weights and the best checkpoint is selected by validation F1 (`load_best_model_at_end=True`).

## Configuration choices

`src/config.py`: seed=42, batch 8, epoch 1, learning rate 2e-5, weight decay 0.01, max tokens 128. `TrainingArguments` uses evaluation/saving each epoch, logging each epoch, and TensorBoard reports. `Trainer` saves `history.json` (actual log_history), validation summary, train summary and final tokenizer/model. Run directory names increment (`lan_1`, `lan_2`, ...); code never overwrites an existing run folder.

## Why training and validation differ

Train loss measures fit to training batches under training mode; validation loss measures fit on unseen examples from a disjoint validation subset under evaluation mode. A decreasing train loss and increasing validation loss may indicate overfitting. Validation is also finite and noisy for 400 examples. **Do not interpret a single epoch as convergence or statistically sound superiority.**

## Checkpoint selection caveat

With `EPOCHS=1` the baseline evaluates at one end-of-epoch checkpoint. `load_best_model_at_end` becomes meaningful with multiple epochs/checkpoints; if you increase epochs, do so *before* final test and note the changed experiment conditions.

## Run procedures

```fish
python practice_3_huggingface/src/2_preparedata.py
python practice_3_huggingface/src/3_train.py
```

Inspect generated run directory for `TRAINING_COMPLETE`, `model/`, `config.json`, `train_metrics.json`, `validation_metrics.json`, `history.json`. If interrupted, a partial history is saved under `history_partial.json`; do not represent it as completed.

## GPU constraints

DistilBERT fine-tuning uses much more memory than pretrained inference; input length/batch size are primary levers. If an NVIDIA Quadro T2000 4GB runs OOM, try 4 batch and/or 96 max tokens. Document settings and start a new `lan_N`; do not silently rewrite config of an old experiment.
