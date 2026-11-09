
"""Reference baselines on the SAME prepared split as the fine-tuned model.

1. Majority-class baseline (floor).
2. Zero-shot baseline: the Exercise 1 SST-2 model applied to IMDb reviews
   with the same MAX_LENGTH truncation used for fine-tuning.

Default split is validation. Test requires RUN_TEST=True and --confirm-test.
Test evidence is never overwritten.
"""

import argparse
import json

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, pipeline

from config import (
    DATASET,
    DATA_DIR,
    MAX_LENGTH,
    OUTPUTS,
    RUN_TEST,
    SEED,
    SENTIMENT_MODEL,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
)
from metrics import majority_baseline, sklearn_metrics
from utils import sha256_records, write_json


def restore_original_split(split):
    raw = load_dataset(DATASET)

    if split == "validation":
        pool = raw["train"].shuffle(seed=SEED).select(
            range(TRAIN_SAMPLES + VALIDATION_SAMPLES)
        )
        parts = pool.train_test_split(
            test_size=VALIDATION_SAMPLES,
            seed=SEED,
            stratify_by_column="label",
        )
        return parts["test"]

    if split == "test":
        return raw["test"].shuffle(seed=SEED).select(
            range(TEST_SAMPLES)
        )

    raise ValueError(f"Unsupported split: {split}")


def verify_split(dataset, split):
    info_path = DATA_DIR / "split_info.json"

    if not info_path.exists():
        raise SystemExit(
            f"Missing split metadata: {info_path}. "
            "Run 2_preparedata.py first."
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    expected = info["splits"][split]
    actual_hash = sha256_records(dataset)

    if len(dataset) != expected["count"]:
        raise RuntimeError(
            f"Sample count mismatch for {split}: "
            f"{len(dataset)} != {expected['count']}"
        )

    if actual_hash != expected["text_label_sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {split}. "
            "Restored data does not match the prepared split."
        )

    prepared_path = DATA_DIR / split

    if not prepared_path.exists():
        raise SystemExit(
            f"Prepared split absent: {prepared_path}. "
            "Run 2_preparedata.py first."
        )

    prepared = load_from_disk(str(prepared_path))

    if len(prepared) != len(dataset):
        raise RuntimeError(
            f"Prepared split length mismatch for {split}."
        )

    prepared_labels = np.asarray(prepared["label"]).astype(int)
    original_labels = np.asarray(dataset["label"]).astype(int)

    if not np.array_equal(prepared_labels, original_labels):
        raise RuntimeError(
            f"Label order mismatch for {split}."
        )

    print(f"Verified {split}: {len(dataset)} samples, SHA256 matched.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-test", action="store_true")
    opts = parser.parse_args()

    split = "test" if RUN_TEST else "validation"

    if RUN_TEST and not opts.confirm_test:
        raise SystemExit(
            "Test locked: set RUN_TEST=True AND pass --confirm-test."
        )

    destination = OUTPUTS / "baseline" / f"baseline_{split}.json"

    if split == "test" and destination.exists():
        raise SystemExit(
            f"{destination} already exists; refusing to overwrite test evidence."
        )

    dataset = restore_original_split(split)
    verify_split(dataset, split)

    texts = list(dataset["text"])
    labels = np.asarray(dataset["label"]).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)

    classifier = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=16,
    )

    outputs = classifier(texts)

    preds = np.asarray([
        1 if output["label"].upper().startswith("POS") else 0
        for output in outputs
    ])

    evidence = {
        "split": split,
        "sample_count": len(texts),
        "max_length": MAX_LENGTH,
        "zero_shot_model": SENTIMENT_MODEL,
        "zero_shot_metrics": sklearn_metrics(labels, preds),
        "zero_shot_confusion_matrix": [
            [
                int(((labels == t) & (preds == p)).sum())
                for p in (0, 1)
            ]
            for t in (0, 1)
        ],
        "majority_baseline": majority_baseline(labels),
        "note": (
            "Zero-shot = no fine-tuning on IMDb; "
            "SST-2 sentences differ from IMDb reviews."
        ),
    }

    write_json(destination, evidence)

    print("Split:", split, "| n =", len(texts))
    print("Zero-shot:", evidence["zero_shot_metrics"])
    print("Majority :", evidence["majority_baseline"])
    print("Saved:", destination)


if __name__ == "__main__":
    main()
