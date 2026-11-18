# Practice 3 — Assignment requirements and acceptance criteria

## Exercise 1: Sentiment Analysis with Hugging Face

1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

**Proof of completion:** `outputs/exercise1/sentiment_result.json` must contain a nonempty sentence, tokenizer output, token IDs, exact Hub model identifier, predicted label and score. Verify generated output—not just source code.

## Exercise 2: Fine-tuning a Pretrained Model for Binary Text Classification

1. Install the necessary Hugging Face libraries (`transformers`, `datasets`, `evaluate`).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and fine-tune the model.
7. Evaluate the fine-tuned model.

**Proof of completion:** `split_info.json`, `lan_N/config.json`, `history.json`, `train_metrics.json`, `validation_metrics.json`, final saved model, and (after model freeze) `test_metrics.json`; the notebook presents actual values. The unit tests alone do not satisfy training/evaluation requirements.

## Course references

- https://huggingface.co/docs/transformers/en/training
- https://huggingface.co/blog/sentiment-analysis-python
- https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face
- https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification

## Scope boundary

The assignment asks for a working baseline, not a leaderboard optimization, broad hyperparameter search, or proof of statistical superiority. Distinguish **completed implementation** from **measured experiment**.
