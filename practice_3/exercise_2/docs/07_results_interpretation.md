# 07 — Results interpretation and reporting guide

## Essential evidence table

Create this table **after running** the experiment:

| Split | n | Accuracy | Precision (positive) | Recall (positive) | F1 (positive) | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2000 | Not measured here | — | — | — | see trainer history |
| Validation | 400 | from JSON | from JSON | from JSON | from JSON | from JSON |
| Final test | 500 | from JSON | from JSON | from JSON | from JSON | from JSON |

The numbers above are counts in the baseline configuration, **not verified counts from your latest run**. The report notebook loads actual files and displays their recorded metrics.

## Diagnosing learning curves

- Both train and val loss declining: learning may be progressing, but one epoch cannot show the later trend.
- Train loss declining, val loss rising: possible overfitting or sampling variation.
- Both losses high: insufficient training, learning-rate issue, wrong labels, severe truncation or misconfiguration.
- Accuracy stable while loss fluctuates: confidence calibration affects loss even without changing class decisions.

## Interpreting per-class errors

Compare FPs (negative reviews misclassified positive) against FNs (positive reviews misclassified negative). A large imbalance may reflect learned language tendencies, threshold effects, dataset sampling, or class imbalance. Do not infer a universal causal explanation from one confusion matrix.

## Critical limitations

1. Base model pretrained outside the lab, so model knowledge is not learned only from 2,000 IMDb examples.
2. Small fixed subset + single seed yields noisy results and weak statistical inference.
3. Long reviews are truncated to max 128 tokens, possibly omitting key sentiment.
4. IMDb review domain differs from social-media feedback and multilingual sentiment.
5. The zero-shot SST-2 baseline and the fine-tuned model are comparable only because both are scored on the same held-out reviews with the same truncation.

## Suggested conclusion template

> A pretrained DistilBERT encoder was fine-tuned on selected labeled IMDb reviews with a separate validation split. The finalized model achieved **[read test metrics]** on **[confirmed test count]** examples. The confusion matrix shows **[grounded error pattern]**. Results are restricted by limited samples, truncation and a single seed; they are not a full IMDb benchmark.

Do not replace bracketed prompts with invented numbers.
