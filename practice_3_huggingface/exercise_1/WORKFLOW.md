# Exercise 1 — Sentiment Analysis with Hugging Face

## Goal (Chapter 4, Practice 3, page 25)
Install `transformers`, load a pretrained sentiment classifier from Hugging Face Hub, tokenize one sentence, and predict sentiment. **No training is required for Exercise 1.**

## Owner & status
- Owner: TBD
- Status: source ready; inference evidence not included in this starter.

## Workflow
1. From repository root, install dependencies: `python -m pip install -r practice_3_huggingface/requirements.txt`.
2. Read `practice_3_huggingface/docs/03_sentiment.md` and inspect `src/1_sentiment.py`.
3. Execute:

   ```fish
   python practice_3_huggingface/src/1_sentiment.py
   python practice_3_huggingface/src/1_sentiment.py --sentence 'This film is surprisingly good.'
   ```

4. Inspect tokens, token IDs, attention mask, label and score. Save examples or copy the JSON before rerunning: the output file is overwritten.
5. Explain the difference between tokenization and inference; pretrained model confidence is not guaranteed accuracy.

## Inputs and outputs
- Input: sample English review string (CLI `--sentence`).
- Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
- Implementation: `src/1_sentiment.py`, `src/config.py`.
- Runtime output: `outputs/exercise1/sentiment_result.json` (**generated only after execution**).

## Checklist before submitting
- [ ] Script runs successfully.
- [ ] Tokenization output recorded.
- [ ] Predicted label and model score recorded.
- [ ] Model identifier and input sentence included.
- [ ] Result interpreted; no fine-tuning claimed for this exercise.

## Git contribution
Commit source changes, your workflow updates and small, verified evidence as permitted by leader. Do not commit model caches or personal tokens.
