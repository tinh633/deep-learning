"""Practice 3 IMDb EDA. Local tokenized split analysis by default.

Pass --raw to reconstruct the original seeded IMDb train/validation split and
show non-sensitive review snippets and truncation statistics. This downloads
IMDb again if the Hugging Face cache has been cleared. Never changes saved data.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from config import (DATASET, DATA_DIR, FINETUNE_MODEL, MAX_LENGTH, ROOT,
                    SEED, TRAIN_SAMPLES, VALIDATION_SAMPLES, TEST_SAMPLES)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def local_eda(out):
    splits = {}
    for name in ("train", "validation", "test"):
        path = DATA_DIR / name
        if not path.exists():
            raise SystemExit(f"Missing {path}; run 2_preparedata.py first")
        splits[name] = load_from_disk(str(path))
    print("Prepared data fields:", splits["train"].column_names)

    names = list(splits)
    counts = {name: Counter(map(int, ds["label"])) for name, ds in splits.items()}
    x = np.arange(len(names))
    for idx, (label, title) in enumerate([(0, "Negative"), (1, "Positive")]):
        y = [counts[n][label] for n in names]
        plt.bar(x + (idx - 0.5) * 0.36, y, width=0.36, label=title)
    plt.xticks(x, names)
    plt.xlabel("Split")
    plt.ylabel("Review count")
    plt.title("IMDb — Label distribution")
    plt.legend()
    save_plot(out / "label_distribution.png")

    for name, ds in splits.items():
        lengths = [len(ids) for ids in ds["input_ids"]]
        plt.hist(lengths, bins=np.arange(0, MAX_LENGTH + 9, 8), alpha=0.45,
                 label=f"{name} (n={len(ds)})")
        print(name, "samples:", len(ds), "label counts:", dict(counts[name]),
              "mean stored tokens:", round(float(np.mean(lengths)), 2),
              "at MAX_LENGTH:", sum(z == MAX_LENGTH for z in lengths))
    plt.xlabel("Tokens stored AFTER truncation")
    plt.ylabel("Review count")
    plt.title(f"Stored token lengths (capped at {MAX_LENGTH})")
    plt.legend()
    save_plot(out / "token_lengths_after_truncation.png")
    print("NOTE: Stored length = MAX_LENGTH does NOT prove truncation occurred.")


def raw_eda(out):
    raw = load_dataset(DATASET)
    pool = raw["train"].shuffle(seed=SEED).select(
        range(TRAIN_SAMPLES + VALIDATION_SAMPLES))
    parts = pool.train_test_split(test_size=VALIDATION_SAMPLES, seed=SEED,
                                  stratify_by_column="label")
    splits = {"train": parts["train"], "validation": parts["test"]}
    # Never use test labels or content for model choice; analyze training only.
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    summary = {"dataset": DATASET, "seed": SEED, "max_length": MAX_LENGTH,
               "notes": "Raw-token counts from reproducible train/validation split only",
               "splits": {}}
    for name, ds in splits.items():
        lengths = [len(tokenizer(text, truncation=False)["input_ids"])
                   for text in ds["text"]]
        truncated = sum(n > MAX_LENGTH for n in lengths)
        summary["splits"][name] = {
            "samples": len(ds), "mean_raw_tokens": round(float(np.mean(lengths)), 2),
            "median_raw_tokens": float(np.median(lengths)),
            "would_be_truncated": truncated,
            "truncation_fraction": round(truncated / len(ds), 4)}
        plt.hist(lengths, bins=35, alpha=0.55, label=name)
        print(name, summary["splits"][name])
    plt.axvline(MAX_LENGTH, color="black", linestyle="--",
                label=f"MAX_LENGTH = {MAX_LENGTH}")
    plt.xlabel("Tokens BEFORE truncation")
    plt.ylabel("Review count")
    plt.title("IMDb — Original token lengths, train and validation")
    plt.legend()
    save_plot(out / "token_lengths_before_truncation.png")
    (out / "truncation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Only training examples; saved snippets for instructional inspection, not a data dump.
    sample_path = out / "train_review_examples.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "label", "review_excerpt"])
        for i in range(min(6, len(splits["train"]))):
            row = splits["train"][i]
            writer.writerow([i, "POSITIVE" if row["label"] == 1 else "NEGATIVE",
                             " ".join(row["text"].split())[:260]])
    print("Saved:", sample_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="Also inspect original training text before truncation (Hub access may be needed)")
    args = parser.parse_args()
    out = ROOT / "visualization" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    local_eda(out)
    if args.raw:
        raw_eda(out)


if __name__ == "__main__":
    main()
