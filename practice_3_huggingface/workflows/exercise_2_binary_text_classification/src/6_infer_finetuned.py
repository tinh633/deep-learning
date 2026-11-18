"""Try a new review with the saved model of a completed run."""

import argparse

from utils import completed_run, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sentence", required=True)
    args = parser.parse_args()
    if not args.sentence.strip():
        parser.error("--sentence must contain text")
    run = completed_run(args.run)
    max_length = read_json(run / "split_info.json")["max_length"]

    from transformers import pipeline

    model_path = str(run / "model")
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=-1)
    scores = classifier(args.sentence, truncation=True, max_length=max_length, top_k=None)
    print(f"Run: {args.run}")
    for item in sorted(scores, key=lambda item: item["score"], reverse=True):
        print(f"{item['label']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()
