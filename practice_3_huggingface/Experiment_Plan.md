# Practice 3 — Experiment design and control plan

## 1. Research questions

- **Ex1:** Can an existing Hub sentiment classifier tokenize and classify a previously unseen sentence without additional training?
- **Ex2:** Can a pretrained English Transformer be fine-tuned using `Trainer` for binary IMDb classification, and what does its held-out performance show?
- **Secondary:** Does train loss diverge from validation loss? Which sentiment class has lower recall? Are conclusions limited by 2,000 training examples and one seed?

## 2. Units, labels, data partitions

IMDb is composed of review text and a binary class: 0 negative, 1 positive. The repository `stanfordnlp/imdb` includes 25k official train + 25k test; the unlabeled split is excluded. Do not use the official test set to choose learning rates, checkpoints or maximum token length. The scripts select 2,400 items from official train, subdivide stratified into 2,000 train and 400 validation; then reserve 500 items from official test. Check text overlap across partitions, hash each original subset and record class counts.

**Known caveats:** Hashing review strings helps verify selected content but is not cryptographic evidence of true underlying semantic independence; near-duplicate reviews may persist. Selecting small subsets introduces variance. Models may be biased toward English movie-review language.

## 3. Model selection

**Exercise 1:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`; inference only. Its sentiment classification head was trained before this coursework. Tokenization yields tokens and integer IDs; pipeline yields `label` and model score. Score is not guaranteed to be calibrated.

**Exercise 2:** `distilbert/distilbert-base-uncased`. Load with `AutoModelForSequenceClassification(...,num_labels=2)`. Its encoder comes from pretraining; classification head is initialized for the new two-label task. All trainable parameters are updated by Trainer unless explicitly frozen (this baseline does not freeze them). No claim of SOTA.

## 4. Fixed baseline

| Setting | Baseline | Reason / trade-off |
|---|---|---|
| Seed | 42 | Repeat selection & random initialization; GPU nondeterminism still possible |
| Train/validation/test | 2000 / 400 / 500 | Fast educational run, not full IMDb |
| Max tokens | 128 | Lower VRAM/time; long reviews may lose relevant endings |
| Batch size | 8 | Conservative starting point; check VRAM |
| Epochs | 1 | Smoke-to-baseline; may underfit |
| Learning rate | 2e-5 | Common conservative fine-tuning starting value |
| Optimizer | Trainer default AdamW | Avoid custom training logic |
| Weight decay | 0.01 | Mild regularization |
| Checkpoint selection | Max validation F1 | Do not select on test |
| Precision | FP32 | Simpler CPU/CUDA compatibility; slower than supported AMP |

## 5. Execution and exit criteria

1. Run Exercise 1; confirm token IDs and predicted label saved.
2. Prepare splits and validate counts, labels, hashes and text overlaps.
3. Train a new `lan_N`; successful exit produces `TRAINING_COMPLETE` and saved model/tokenizer.
4. Review validation F1/accuracy and learning curves; record chosen run and why.
5. Only then unlock final test (`RUN_TEST=True`, `--confirm-test`). Do not repeatedly consult test while tuning.
6. Create confusion, loss/score figures, a Markdown summary, and notebook narrative.

## 6. Evidence quality

- Report metrics as values from JSON, not hand-typed numbers. Identify split, run, seed, model and training configuration.
- Class-wise metrics reveal if accuracy conceals asymmetric performance.
- Confusion matrix order: actual labels in rows, predicted labels in columns, negative first.
- ROC-AUC uses positive-class score, not argmax; F1 uses threshold induced by argmax.
- One run/seed and a small subset do **not** show statistical significance or production readiness.

## 7. Optional extension (clearly outside required baseline)

After reporting the baseline, compare max lengths 128/256 or epochs 2/3 **using validation only**, with constant split and seed. Larger max length increases compute. Do not tune after inspecting final test.
