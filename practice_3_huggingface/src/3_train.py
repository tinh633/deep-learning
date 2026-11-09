"""Exercise 2: reproducible fine-tuning. Create a new run, never overwrite old runs."""
import os
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed)
from config import (BATCH_SIZE, DATA_DIR, EPOCHS, FINETUNE_MODEL,
                    LEARNING_RATE, RUNS, SEED, WEIGHT_DECAY, MAX_LENGTH, DATASET)
from metrics import compute_metrics
from utils import environment, next_run, write_json
import json
from datetime import datetime, timezone

class RunLogCallback(TrainerCallback):
    """Write actual Trainer log events incrementally; survives interruption."""
    def __init__(self, filename):
        self.filename = filename
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        record = {'utc': datetime.now(timezone.utc).isoformat(),
                  'epoch': state.epoch, 'step': state.global_step,
                  'metrics': logs}
        with self.filename.open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

def main():
    if not all((DATA_DIR / name).exists() for name in ('train','validation')):
        raise SystemExit('Run src/2_preparedata.py first.')
    if not (DATA_DIR / 'split_info.json').exists():
        raise SystemExit('Missing split_info.json; do not train without provenance.')
    set_seed(SEED)
    run_dir = next_run(RUNS)
    train = load_from_disk(str(DATA_DIR / 'train'))
    valid = load_from_disk(str(DATA_DIR / 'validation'))
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNE_MODEL, num_labels=2,
        id2label={0:'NEGATIVE',1:'POSITIVE'},
        label2id={'NEGATIVE':0,'POSITIVE':1})
    configuration = {
        'base_model': FINETUNE_MODEL, 'dataset_id': DATASET, 'seed': SEED,
        'train_samples': len(train), 'validation_samples': len(valid),
        'max_length': MAX_LENGTH, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'optimizer': 'Trainer default AdamW', 'loss': 'CrossEntropyLoss',
        'model_selection': 'highest validation F1; tie handling by Trainer',
        'test_used_during_training': False, 'environment': environment(),
        'split_metadata': str(DATA_DIR / 'split_info.json'),
    }
    write_json(run_dir / 'config.json', configuration)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tensorboard")
    args = TrainingArguments(
        output_dir=str(run_dir/'checkpoints'), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        logging_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='eval_f1', greater_is_better=True,
        seed=SEED, data_seed=SEED,
        report_to="tensorboard",
        fp16=False,
    )
    with (run_dir/'train.log').open('w',encoding='utf-8') as log:
        log.write('# Trainer log events as JSONL; not a raw terminal transcript.\n')
    trainer = Trainer(model=model, args=args,
        train_dataset=train, eval_dataset=valid, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics, callbacks=[RunLogCallback(run_dir/'train.log')])
    try:
        result = trainer.train()
        validation = trainer.evaluate(eval_dataset=valid)
        trainer.save_model(str(run_dir/'model'))
        tokenizer.save_pretrained(str(run_dir/'model'))
        configuration['best_checkpoint'] = trainer.state.best_model_checkpoint
        configuration['best_metric'] = trainer.state.best_metric
        configuration['global_step'] = trainer.state.global_step
        write_json(run_dir/'config.json', configuration)
        write_json(run_dir/'history.json', trainer.state.log_history)
        write_json(run_dir/'train_metrics.json', result.metrics)
        write_json(run_dir/'validation_metrics.json', validation)
        (run_dir/'TRAINING_COMPLETE').write_text('Trainer completed successfully.\n', encoding='utf-8')
        print('RUN:', run_dir)
        print('VALIDATION:', validation)
        print('Final test NOT run. Freeze model before test.')
    except BaseException:
        write_json(run_dir/'history_partial.json', trainer.state.log_history)
        print('Interrupted run:', run_dir, '— retained for audit; not marked complete.')
        raise

if __name__=='__main__':
    main()
