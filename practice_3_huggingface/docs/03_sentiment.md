# 03 — Exercise 1: pretrained sentiment inference

## How it works

The Hub model `distilbert/distilbert-base-uncased-finetuned-sst-2-english` combines a DistilBERT encoder and an already-trained SST-2 sentiment head. The provided exercise does **not train** it. `AutoTokenizer` loads its vocabulary/tokenizer configuration. `pipeline('sentiment-analysis')` handles preprocessing, forward pass and label formatting.

## Step mapping

| Assignment | Code | Observable evidence |
|---|---|---|
| Install transformers | `requirements.txt` | pip show/version |
| Use pretrained sentiment model | `pipeline(...,model=SENTIMENT_MODEL)` | exact `model_id` in output |
| Tokenize sample | `tokenizer.tokenize`; `tokenizer(sentence)` | tokens, `input_ids`, `attention_mask` |
| Perform sentiment analysis | `classifier(sentence)` | label and `model_score` |

## Execution

```fish
python practice_3_huggingface/src/1_sentiment.py
python practice_3_huggingface/src/1_sentiment.py --sentence 'I disliked the acting, but the ending was great.'
```

The run saves `outputs/exercise1/sentiment_result.json`. A second run replaces the JSON; copy/rename it if you need both examples as evidence.

## Interpreting sample text

The default sentence contrasts a slow start with liking the ending. The user previously observed a `POSITIVE` prediction with a high score (~0.9998) on their own machine. That observation is specific to that model/input and should not be generalized to all mixed sentences. The script stores the actual current response rather than hardcoding it.

## Why tokens differ from ordinary words

Tokenizers may lowercase, split punctuation and break rare words into multiple subword pieces. Model IDs include special tokens. For DistilBERT-style input, a single-sentence token sequence normally has special boundary markers; the exact IDs depend on vocabulary. Inspect IDs from the program, not by assuming one ID per English word.

## Limits and risks

- Binary predictions cannot represent neutral/mixed/ironic sentiment faithfully.
- SST-2 training domain differs from long IMDb reviews; do not directly compare Ex1 confidence to Ex2 accuracy.
- Uncalibrated softmax confidence can be high for incorrect predictions; model responses should be interpreted as statistical outputs, not ground truth.
- This model is English-focused; multilingual text requires another suitable model and evaluation.
