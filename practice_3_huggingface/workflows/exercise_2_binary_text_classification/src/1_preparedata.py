"""Build reproducible IMDb splits, tokenize reviews, and save split provenance."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from config import (
    DATASET_ID, ID_TO_LABEL, MAX_LENGTH, MODEL_ID, SEED, TEST_SAMPLES,
    TRAIN_SAMPLES, VALIDATION_SAMPLES,
)
from utils import add_data_argument, sha256_records, write_json


def take_stratified(dataset, size, seed):
    if size == len(dataset):
        return dataset.shuffle(seed=seed)
    if size > len(dataset):
        raise ValueError(f"Requested {size} samples from {len(dataset)} rows.")
    return dataset.train_test_split(test_size=size, seed=seed, stratify_by_column="label")["test"]


def indexed(dataset):
    return dataset.add_column("source_index", list(range(len(dataset))))


def text_sets(splits):
    return {name: {" ".join(text.split()).casefold() for text in ds["text"]} for name, ds in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_argument(parser)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    dest = args.data_dir.resolve()
    sizes = (args.train_samples, args.validation_samples, args.test_samples)
    if min(sizes) < 2 or not 8 <= args.max_length <= 512 or args.seed < 0:
        parser.error("Each split needs at least 2 samples, max length must be 8–512, seed must be nonnegative.")
    if dest.exists():
        raise SystemExit(f"{dest} already exists. Reuse it or give the new snapshot a different --data-dir.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    raw = load_dataset(DATASET_ID)
    pool = take_stratified(indexed(raw["train"]), args.train_samples + args.validation_samples, args.seed)
    parts = pool.train_test_split(
        test_size=args.validation_samples, seed=args.seed, stratify_by_column="label"
    )
    splits = {
        "train": parts["train"],
        "validation": parts["test"],
        "test": take_stratified(indexed(raw["test"]), args.test_samples, args.seed),
    }
    texts = text_sets(splits)
    overlap = {
        f"{left}/{right}": len(texts[left] & texts[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlap.values()):
        raise SystemExit(f"Duplicate normalized texts across splits: {overlap}. Choose a new snapshot seed before training.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        full = tokenizer(batch["text"], add_special_tokens=True, truncation=False, verbose=False)
        encoded = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        encoded["original_token_count"] = [len(ids) for ids in full["input_ids"]]
        return encoded

    meta = {
        "dataset_id": DATASET_ID, "tokenizer_id": MODEL_ID, "seed": args.seed,
        "max_length": args.max_length, "labels": ID_TO_LABEL, "cross_split_duplicates": overlap,
        "splits": {},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        snapshot = Path(tmp) / "snapshot"
        for name, dataset in splits.items():
            labels = Counter(dataset["label"])
            meta["splits"][name] = {
                "count": len(dataset), "source_split": "test" if name == "test" else "train",
                "label_counts": {ID_TO_LABEL[key]: labels[key] for key in ID_TO_LABEL},
                "sha256": sha256_records(dataset),
            }
            prepared = dataset.map(tokenize, batched=True)
            prepared.save_to_disk(str(snapshot / name))
            print(f"{name}: {len(dataset)} rows, {meta['splits'][name]['label_counts']}")
        write_json(snapshot / "split_info.json", meta)
        snapshot.rename(dest)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
