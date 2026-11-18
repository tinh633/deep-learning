"""Evaluate an explicitly selected run on validation or on the held-out test."""

import argparse

from config import ID_TO_LABEL
from utils import add_data_argument, completed_run, data_metadata, read_json, sha256_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    add_data_argument(parser)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run = completed_run(args.run)
    split = args.split
    data_dir = args.data_dir.resolve()
    meta = data_metadata(data_dir)
    if meta != read_json(run / "split_info.json"):
        raise SystemExit("Data snapshot differs from this run. Pass its original --data-dir.")
    destination = run / f"{split}_evaluation.json"
    if split == "test" and destination.exists():
        raise SystemExit(f"Test evidence already exists at {destination}.")

    from datasets import load_from_disk
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments,
    )
    from metrics import build_metrics, logits_array, positive_probabilities

    data = load_from_disk(str(data_dir / split))
    if sha256_records(data) != meta["splits"][split]["sha256"]:
        raise SystemExit(f"{split} rows differ from the recorded data snapshot.")
    model_path = run / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    batch_size = read_json(run / "config.json")["training_arguments"]["per_device_eval_batch_size"]
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(run / "evaluation"), use_cpu=args.cpu,
            per_device_eval_batch_size=batch_size, report_to=[],
        ),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=build_metrics(),
    )
    prediction = trainer.predict(data, metric_key_prefix=split)
    actual = prediction.label_ids
    predicted = logits_array(prediction.predictions).argmax(axis=-1)
    probabilities = positive_probabilities(prediction.predictions)
    labels = list(ID_TO_LABEL.values())
    errors = []
    for row, target, guess, probability in zip(data, actual, predicted, probabilities):
        if target != guess:
            errors.append({
                "source_index": row["source_index"], "actual": ID_TO_LABEL[target],
                "predicted": ID_TO_LABEL[guess], "probability_positive": float(probability),
                "review_excerpt": row["text"][:400], "original_token_count": row["original_token_count"],
            })
    errors.sort(key=lambda item: abs(item["probability_positive"] - 0.5), reverse=True)
    evidence = {
        "run": args.run, "split": split, "sample_count": len(data), "label_order": labels,
        "metrics": prediction.metrics, "error_count": len(errors), "error_examples": errors[:10],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=[0, 1], target_names=labels, output_dict=True, zero_division=0
        ),
    }
    write_json(destination, evidence)
    print(f"{args.run} {split} | accuracy: {prediction.metrics[f'{split}_accuracy']:.2%}"
          f" | F1: {prediction.metrics[f'{split}_f1']:.4f} | saved: {destination}")


if __name__ == "__main__":
    main()
