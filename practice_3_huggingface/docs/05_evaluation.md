# 05 — Validation, held-out test and metrics

## Holdout protocol

The training script may consult only `train` and `validation`. Validation chooses the model checkpoint via F1. The official IMDb test subset is held back until the run and all decisions are finalized. `src/4_evaluate.py` uses `RUN_TEST=False` by default. To run the final test, change config to `RUN_TEST=True` **and** pass `--confirm-test`.

Repeatedly looking at final test accuracy and changing max length/LR/epochs biases the final reported estimate; this is test leakage. A separate run is useful for comparisons, but test should remain final, not a tuning dashboard.

## Accuracy, precision, recall, F1

For POSITIVE as the positive class:

- `accuracy = (TP+TN)/(TP+TN+FP+FN)`.
- `precision = TP/(TP+FP)` — fraction of positive predictions that are correct.
- `recall = TP/(TP+FN)` — fraction of actual positives found.
- `F1 = 2PR/(P+R)` — harmonic mean of precision and recall.

Confusion matrix order (rows actual, columns predicted):

| | Pred NEGATIVE | Pred POSITIVE |
|---|---:|---:|
| Actual NEGATIVE | TN | FP |
| Actual POSITIVE | FN | TP |

`classification_report` contains per-class precision/recall/F1/support. **ROC-AUC** is computed from positive-class softmax probabilities and true labels, provided both classes appear. This differs from F1, which uses argmax predictions.

## Evaluation artifact design

`validation_recheck.json` or `test_metrics.json` includes run, split, sample count, model path, label order, aggregate metrics, confusion matrix, per-class report and indexed predictions with positive-class probability. Predictions do not include review text; indices are relative to the saved split. This supports later error analysis without pretending to have reloaded original review strings.

If there is already a final `test_metrics.json`, the script refuses to overwrite it. Never edit a result file to fit expectations; if a report needs regeneration use the same saved artifact.

## How to report results

Example *format only* (placeholders are not results):

> `lan_N`; DistilBERT; seed 42; test n=500; accuracy = **[from JSON]**, positive F1 = **[from JSON]**, ROC-AUC = **[from JSON]**. Most common error type = **[compute from confusion]**. Inference trained on selected IMDb subset, not full benchmark.

Never fill placeholders with assumed values. Keep exact source file and run date beside every summary number.
