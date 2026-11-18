"""Fine-tune DistilBERT with Trainer and save the best validation checkpoint."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from config import BATCH_SIZE, EPOCHS, ID_TO_LABEL, LEARNING_RATE, MODEL_ID, WEIGHT_DECAY
from utils import add_data_argument, data_metadata, next_run, sha256_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_argument(parser)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate <= 1:
        parser.error("Use positive epochs, batch size and a learning rate in (0, 1].")
    data_dir = args.data_dir.resolve()
    split_info = data_metadata(data_dir)
    if split_info["tokenizer_id"] != MODEL_ID:
        raise SystemExit("Prepared tokenizer and base model differ. Prepare a compatible snapshot.")

    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )
    from metrics import build_metrics

    class RunLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                with (run / "train.log").open("a", encoding="utf-8") as file:
                    file.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    train = load_from_disk(str(data_dir / "train"))
    validation = load_from_disk(str(data_dir / "validation"))
    for name, dataset in (("train", train), ("validation", validation)):
        if sha256_records(dataset) != split_info["splits"][name]["sha256"]:
            raise SystemExit(f"{name} rows differ from the prepared snapshot metadata.")
    seed = split_info["seed"]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    )
    compute_metrics = build_metrics()
    run = next_run()
    training_args = TrainingArguments(
        output_dir=str(run / "checkpoints"),
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        weight_decay=WEIGHT_DECAY, optim="adamw_torch", lr_scheduler_type="linear",
        logging_dir=str(run / "tensorboard"),
        max_grad_norm=1.0, eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_f1", greater_is_better=True, report_to=["tensorboard"],
        seed=seed, data_seed=seed, use_cpu=args.cpu,
    )
    configuration = {
        "run": run.name, "base_model": MODEL_ID, "python": sys.version.split()[0],
        "device": str(training_args.device),
        "versions": {name: version(name) for name in ("torch", "transformers", "datasets", "evaluate", "accelerate")},
        "training_arguments": training_args.to_dict(),
        "data_dir": str(data_dir),
    }
    write_json(run / "config.json", configuration)
    write_json(run / "split_info.json", split_info)
    print(f"Run: {run.name} | Device: {training_args.device}", flush=True)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=validation,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics, callbacks=[RunLogger()],
    )
    try:
        result = trainer.train()
        validation_metrics = trainer.evaluate()
        trainer.save_model(str(run / "model"))
        tokenizer.save_pretrained(run / "model")
        configuration["best_checkpoint"] = Path(trainer.state.best_model_checkpoint).name
        configuration["best_step"] = int(configuration["best_checkpoint"].rsplit("-", 1)[1])
        configuration["best_validation_f1"] = trainer.state.best_metric
        write_json(run / "config.json", configuration)
        write_json(run / "history.json", trainer.state.log_history)
        write_json(run / "train_metrics.json", result.metrics)
        write_json(run / "validation_metrics.json", validation_metrics)
        (run / "TRAINING_COMPLETE").touch()
    except BaseException:
        write_json(run / "history_partial.json", trainer.state.log_history)
        raise
    print(f"Completed {run.name} | Validation F1: {validation_metrics['eval_f1']:.4f}")


if __name__ == "__main__":
    main()
