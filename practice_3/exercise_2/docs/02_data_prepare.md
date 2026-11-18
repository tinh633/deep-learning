# 02 — Dataset, EDA, tokenization and leakage controls

## Source and classes

The [stanfordnlp/imdb dataset](https://huggingface.co/datasets/stanfordnlp/imdb) supplies 25,000 labeled train and 25,000 labeled test English movie reviews, plus 50,000 unlabeled reviews not used here. ClassLabel encodes negative as 0 and positive as 1. The lab uses a small seeded subset: 2,000 train, 400 validation and 500 test.

## Split design

1. Load the official train and test splits through `load_dataset('stanfordnlp/imdb')`.
2. Shuffle the official train with seed 42, take 2,400 examples.
3. Create a stratified 2,000 / 400 train-validation split; this prevents gross label imbalance but not all possible sampling bias.
4. Independently choose 500 examples from official test using the same fixed seed, **without using their labels to choose settings**.
5. Assert no exact text duplicates across selected splits. Create order-sensitive SHA256 per split over `(text,label)` so changing subsets can be detected.
6. Save summary information with class counts and hashes as `split_info.json`; save tokenized dataset on disk.

A stratified split does **not** guarantee absence of near-duplicate paraphrases or duplicated source documents. For this coursework, exact-text checks suffice as a simple integrity test; production-grade leakage detection requires more.

## Tokenization and padding

```python
from transformers import AutoTokenizer, DataCollatorWithPadding
model_id = 'distilbert/distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(model_id)
def preprocess(batch):
    return tok(batch['text'], truncation=True, max_length=128)
collator = DataCollatorWithPadding(tok)
```

- WordPiece-style subword IDs capture vocabulary that does not need whole-word coverage.
- `attention_mask` denotes valid tokens (1) versus padding (0).
- `max_length=128` bounds memory but discards late tokens in long reviews.
- Dynamic padding occurs at batch creation, not by padding all text to 128 in preprocessing.
- Preserve original `label` field (0/1); discard `text` from tokenized on-disk arrays to reduce output size and accidental review disclosure.

## Data sanity checks

- `len(train)==2000`, `len(validation)==400`, `len(test)==500`.
- Labels each in `{0,1}`; distributions logged per split.
- `input_ids` and `attention_mask` lengths match and do not exceed 128.
- Train/validation/test original subsets do not share exact text.
- One stable split shared by all later experiments; changing `SEED`, samples or max_length requires re-preparation and a fresh run.

## Failure mode: IMDb ID

Some HF dependency combinations reject a one-component Hub URI (`imdb`). Use full ID `stanfordnlp/imdb`; this was the specific issue in the user's environment. Public dataset access can show authentication warnings but should still work if rate limits permit.

## Potential EDA

Read `split_info.json` for label distribution and count. Token lengths may be inspected on the already prepared split (the saved sequence lengths represent truncated length, not raw review length). When discussing representativeness, distinguish this small subset from official full-dataset statistics.
