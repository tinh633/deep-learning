"""Optional: log a finished run's saved artifacts to MLflow (experiment tracking).

Reads files that 3_train.py / 4_evaluate.py / 7_baseline.py already wrote; it does not train,
evaluate or change any of them, and invents no numbers (a missing file is skipped with a note).

    python src/8_log_mlflow.py                 # logs EVAL_RUN from src/config.py (e.g. lan_1)
    python src/8_log_mlflow.py --run lan_2
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000

Logged: parameters (config.json), per-epoch train/validation curves (history.json), final
validation / test metrics, zero-shot and majority baselines, and the JSON + PNG + REPORT.md files.
Model weights are NOT uploaded (large); the path of the local model folder is stored as a tag.
Running it twice for the same run is refused unless --force is given.
"""
import argparse
import json
import numbers
from pathlib import Path

import mlflow

from config import EVAL_RUN, OUTPUTS, ROOT, RUNS

EXPERIMENT = 'practice3-imdb-distilbert'


def read_json(path):
    path = Path(path)
    if not path.is_file():
        print(f'  [skip] missing: {path.relative_to(ROOT)}')
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def numeric(d, prefix=''):
    """Keep only real numbers (drops strings, lists, bools)."""
    return {f'{prefix}{k}': float(v) for k, v in (d or {}).items()
            if isinstance(v, numbers.Real) and not isinstance(v, bool)}


def log_epoch_curves(history):
    """history.json is Trainer's log_history: one train row and one eval row per epoch."""
    seen = set()
    for row in history or []:
        epoch = row.get('epoch')
        if epoch is None or float(epoch) != int(float(epoch)):
            continue
        epoch = int(epoch)
        if 'eval_loss' in row:
            if epoch in seen:   # later duplicate = post-training recheck of the best model
                continue
            seen.add(epoch)
            mlflow.log_metrics({k.replace('eval_', 'val_', 1): v for k, v in numeric(row).items()
                                if k.startswith('eval_')
                                and not k.endswith(('runtime', 'per_second', 'steps_per_second'))},
                               step=epoch)
        elif 'loss' in row:
            mlflow.log_metric('train_loss', float(row['loss']), step=epoch)


def main():
    parser = argparse.ArgumentParser(description='Log a finished run to MLflow')
    parser.add_argument('--run', default=EVAL_RUN, help='run folder name, e.g. lan_1')
    parser.add_argument('--tracking-uri', default=f'sqlite:///{ROOT / "mlflow.db"}')
    parser.add_argument('--force', action='store_true', help='log again even if this run was logged')
    opts = parser.parse_args()

    run_dir = RUNS / opts.run
    if not run_dir.is_dir():
        raise SystemExit(f'Run folder not found: {run_dir}')
    if not (run_dir / 'TRAINING_COMPLETE').exists():
        print('NOTICE: no TRAINING_COMPLETE marker; logging whatever artifacts exist.')

    mlflow.set_tracking_uri(opts.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    existing = client.search_runs([experiment.experiment_id],
                                 filter_string=f"tags.source_run = '{opts.run}'")
    if existing and not opts.force:
        raise SystemExit(f'{opts.run} is already in MLflow (run {existing[0].info.run_id}). Use --force to log again.')

    config = read_json(run_dir / 'config.json')
    history = read_json(run_dir / 'history.json')
    train_m = read_json(run_dir / 'train_metrics.json')
    val_m = read_json(run_dir / 'validation_metrics.json')
    test_m = read_json(run_dir / 'test_metrics.json')
    split = read_json(OUTPUTS / 'data' / 'split_info.json')
    base_val = read_json(OUTPUTS / 'baseline' / 'baseline_validation.json')
    base_test = read_json(OUTPUTS / 'baseline' / 'baseline_test.json')

    with mlflow.start_run(run_name=opts.run):
        mlflow.set_tags({'source_run': opts.run, 'task': 'binary text classification (IMDb)',
                         'local_model_dir': str(run_dir / 'model'),
                         'test_evaluated': str(test_m is not None)})
        if config:
            keep = ('base_model', 'dataset_id', 'seed', 'train_samples', 'validation_samples', 'max_length',
                    'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer', 'loss',
                    'model_selection', 'global_step')
            mlflow.log_params({k: config[k] for k in keep if k in config})
            if config.get('best_checkpoint'):
                mlflow.log_param('best_checkpoint', Path(str(config['best_checkpoint'])).name)
            if config.get('best_metric') is not None:
                mlflow.log_metric('best_val_f1', float(config['best_metric']))
        if split and 'test' in split.get('splits', {}):
            mlflow.log_param('test_samples', split['splits']['test']['count'])

        log_epoch_curves(history)

        if train_m and isinstance(train_m.get('train_loss'), numbers.Real):
            mlflow.log_metric('final_train_loss', float(train_m['train_loss']))   # mean over all steps
            if isinstance(train_m.get('train_runtime'), numbers.Real):
                mlflow.log_metric('train_runtime_sec', float(train_m['train_runtime']))
        if val_m:   # eval_accuracy -> final_val_accuracy
            mlflow.log_metrics({k.replace('final_val_eval_', 'final_val_'): v
                                for k, v in numeric(val_m, 'final_val_').items()
                                if not k.endswith(('runtime', 'per_second'))})
        if test_m:  # test_accuracy etc. are already prefixed by Trainer.predict
            metrics = {k: v for k, v in numeric(test_m.get('metrics', {})).items()
                       if k.startswith('test_') and not k.endswith(('runtime', 'per_second', 'preparation_time'))}
            mlflow.log_metrics(metrics)
            if test_m.get('roc_auc') is not None:
                mlflow.log_metric('test_roc_auc', float(test_m['roc_auc']))
        for name, base in (('val', base_val), ('test', base_test)):
            if not base:
                continue
            for key, short in (('zero_shot_metrics', 'zero_shot'), ('majority_baseline', 'majority')):
                for m in ('accuracy', 'f1'):
                    if base.get(key, {}).get(m) is not None:
                        mlflow.log_metric(f'baseline_{name}_{short}_{m}', float(base[key][m]))

        files = [run_dir / f for f in ('config.json', 'history.json', 'train_metrics.json',
                                       'validation_metrics.json', 'test_metrics.json')]
        files += [OUTPUTS / 'data' / 'split_info.json']
        files += [OUTPUTS / 'baseline' / f for f in ('baseline_validation.json', 'baseline_test.json')]
        for f in files:
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path=f.parent.name)
        figures = ROOT / 'visualization' / opts.run
        if figures.is_dir():
            mlflow.log_artifacts(str(figures), artifact_path='report')
        print('Logged run:', mlflow.active_run().info.run_id)

    print('View with:  mlflow ui --backend-store-uri', opts.tracking_uri)


if __name__ == '__main__':
    main()
