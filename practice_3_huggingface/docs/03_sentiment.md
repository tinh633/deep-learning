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

The run saves `outputs/exercise1/sentiment_result.json`. A second run replaces the JSON; copy/rename it if you need to keep earlier evidence. Pass `--sentence "..."` (repeatable) to analyse your own sentences.

## Interpreting sample text

The default sentence contrasts a slow start with liking the ending. A mixed-sentiment sentence is a useful probe: the model returns one label with a score, which is not a calibrated probability of being right. Read the label and score from your own `sentiment_result.json`; they are specific to this model and input and should not be generalized. The script runs several sentences (positive, negative, lukewarm, negated) and stores the actual responses rather than hardcoding them.

## Why tokens differ from ordinary words

Tokenizers may lowercase, split punctuation and break rare words into multiple subword pieces. Model IDs include special tokens. For DistilBERT-style input, a single-sentence token sequence normally has special boundary markers; the exact IDs depend on vocabulary. Inspect IDs from the program, not by assuming one ID per English word.

## Limits and risks

- Binary predictions cannot represent neutral/mixed/ironic sentiment faithfully.
- SST-2 training domain differs from long IMDb reviews; do not directly compare Ex1 confidence to Ex2 accuracy.
- Uncalibrated softmax confidence can be high for incorrect predictions; model responses should be interpreted as statistical outputs, not ground truth.
- This model is English-focused; multilingual text requires another suitable model and evaluation.

### Actual inference results

The pretrained DistilBERT model was tested on four English sentences. The results were saved in `outputs/exercise1/sentiment_result.json`.

| Sentence | Predicted label | Model score |
|---|---|---|
| The film started slowly, but I absolutely loved its ending. | POSITIVE | 0.9998 |
| This was a complete waste of two hours; the plot made no sense. | NEGATIVE | 0.9998 |
| The acting was fine, I guess, though nothing special. | POSITIVE | 0.9867 |
| I did not hate it. | POSITIVE | 0.9965 |

The model correctly identifies clearly positive and negative expressions in the first two sentences. The third sentence contains mild or mixed sentiment but is classified as POSITIVE, demonstrating the limitation of binary sentiment classification. The fourth sentence contains negation ("did not hate"), which the model interprets as POSITIVE.

Although all predictions have high model scores, these scores do not guarantee that the predictions are correct. The model cannot explicitly represent neutral or mixed sentiment because it only predicts POSITIVE or NEGATIVE.