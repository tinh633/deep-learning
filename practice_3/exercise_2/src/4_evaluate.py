"""Validation recheck by default. Test requires both config switch and --confirm-test."""
import argparse
import json
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from config import BATCH_SIZE, DATA_DIR, EVAL_RUN, RUNS, RUN_TEST
from metrics import compute_metrics
from utils import write_json

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-test', action='store_true',
        help='Explicit acknowledgement of final locked test evaluation')
    opts = parser.parse_args()
    is_test = bool(RUN_TEST)
    if is_test and not opts.confirm_test:
        raise SystemExit('Test locked: set RUN_TEST=True AND pass --confirm-test after selection.')
    run_dir = RUNS / EVAL_RUN
    model_dir = run_dir / 'model'
    if not model_dir.exists():
        raise SystemExit(f'Fine-tuned model missing: {model_dir}')
    if not (run_dir/'TRAINING_COMPLETE').exists():
        if not (run_dir/'history.json').exists() or not (run_dir/'config.json').exists():
            raise SystemExit(f'Run appears incomplete: {run_dir}')
        print('NOTICE: legacy run without TRAINING_COMPLETE; inspect history/config before reporting.')
    split = 'test' if is_test else 'validation'
    path = DATA_DIR / split
    if not path.exists():
        raise SystemExit(f'Prepared split absent: {path}')
    destination = run_dir/('test_metrics.json' if is_test else 'validation_recheck.json')
    if is_test and destination.exists():
        raise SystemExit('test_metrics.json already exists. Refusing to overwrite held-out test evidence.')
    dataset = load_from_disk(str(path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    args = TrainingArguments(output_dir=str(run_dir/'evaluation_tmp'),
        per_device_eval_batch_size=BATCH_SIZE, report_to='none')
    trainer = Trainer(model=model, args=args, eval_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics)
    output = trainer.predict(dataset)
    logits = output.predictions
    if isinstance(logits, tuple):logits=logits[0]
    probabilities = softmax(logits)
    labels = np.asarray(output.label_ids).astype(int)
    preds = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, preds, labels=[0,1])
    report = classification_report(labels,preds,labels=[0,1],
        target_names=['NEGATIVE','POSITIVE'],output_dict=True,zero_division=0)
    metrics = {str(k):float(v) if isinstance(v,(int,float,np.number)) else v
               for k,v in output.metrics.items()}
    auc = float(roc_auc_score(labels,probabilities[:,1])) if len(set(labels.tolist()))==2 else None
    evidence = {'run':EVAL_RUN, 'split':split, 'sample_count':len(dataset),
        'model_path':str(model_dir), 'label_order':['NEGATIVE','POSITIVE'],
        'metrics':metrics,'roc_auc':auc,'confusion_matrix':matrix.tolist(),
        'classification_report':report,
        'predictions':[{'index':int(i),'true':int(labels[i]),'predicted':int(preds[i]),
                         'probability_positive':float(probabilities[i,1])}
                       for i in range(len(labels))]}
    write_json(destination,evidence)
    print('RUN:',EVAL_RUN, 'SPLIT:',split)
    print('Accuracy:',metrics.get('test_accuracy'))
    print('F1:',metrics.get('test_f1'))
    print('Confusion matrix [true rows, predicted columns]:',matrix.tolist())
    print('Saved:',destination)

if __name__=='__main__':
    main()
