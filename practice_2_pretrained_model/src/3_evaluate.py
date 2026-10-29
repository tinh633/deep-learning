"""What: Đánh giá checkpoint và phân tích lớp/ảnh dự đoán sai.
Why: Kiểm tra chất lượng model ngoài loss/accuracy trong lúc train.
Who: Người học chọn EVAL_RUN; chỉ bật test sau khi khóa checkpoint.
When: Sau train hoặc khi review checkpoint, không tự chạy từ notebook.
Where: Đọc best.pt/split; ghi metrics/log và ảnh sai của đúng run.
How: Khôi phục config thực, eval/no_grad, metrics sklearn và gallery lỗi.
"""

import os

# Windows workers import this file; avoid multiplying BLAS thread pools.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import math
import time
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from config import EVAL_RUN, OUTPUT_FOLDER, PROJECT_FOLDER, RUN_TEST, SPLIT_FILE
from data import build_loaders
from logger import close_log, file_sha256, get_environment, open_log, save_json
from model import PretrainedCNN


def save_error_gallery(dataset, examples, classes, split, target):
    """Show up to 24 confident mistakes using original dataset images."""
    if not examples:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (900, math.ceil(len(examples) / 6) * 155), 'white')
    draw = ImageDraw.Draw(canvas)
    for position, example in enumerate(examples):
        x, y = (position % 6) * 150, (position // 6) * 155
        image = Image.fromarray(dataset.data[example['id']])
        image = image.resize((96, 96), resample=Image.Resampling.NEAREST)
        canvas.paste(image, (x + 25, y + 3))
        lines = [f"{split} #{example['id']}",
                 f"True: {classes[example['true']]}",
                 f"Pred: {classes[example['predicted']]}",
                 f"Confidence: {example['confidence']:.3f}"]
        for line, text in enumerate(lines):
            draw.text((x + 4, y + 101 + line * 13), text, fill='black')
    canvas.save(target)


# Windows DataLoader workers import this file in child processes.
if __name__ == '__main__':
    # 1. Select the trained checkpoint.
    folder = OUTPUT_FOLDER / 'runs' / EVAL_RUN
    checkpoint_path = folder / 'best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'No best checkpoint at {checkpoint_path}; select EVAL_RUN in config.py.')
    logger = open_log(folder, 'evaluate.log')
    started = time.perf_counter()

    try:
        # 2. Restore model and verify the saved split.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Only load checkpoints created locally by this project: RNG includes NumPy state.
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        settings = saved['settings']
        checkpoint_hash = file_sha256(checkpoint_path)
        split_hash = file_sha256(SPLIT_FILE)
        if split_hash != settings['split_sha256']:
            raise ValueError('Current split does not match checkpoint split SHA-256.')
        model_config = settings['model_config']
        model = PretrainedCNN(settings['model_name'],
                              fine_tune=model_config['fine_tune'], pretrained=False)
        model.load_state_dict(saved['model_state'], strict=True)
        model = model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # 3. Evaluation data has no augmentation or shuffling.
        train_loader, val_loader, test_loader = build_loaders(settings['data'], training=False)
        classes = saved['classes']
        if classes != val_loader.dataset.dataset.classes:
            raise ValueError('Dataset class order differs from checkpoint.')
        loaders = dict(train=train_loader, validation=val_loader)
        if RUN_TEST:
            loaders['test'] = test_loader
        environment = get_environment(device)
        logger.info('Checkpoint: %s | SHA256: %s | best_epoch=%d',
                    checkpoint_path, checkpoint_hash, saved['epoch'])
        logger.info('Saved configuration: %s | pretrained weights: %s', settings, saved['weights'])
        logger.info('Environment: %s | Loss: %s', environment, criterion)
        logger.info('Architecture:\n%s', model)
        logger.info('Splits: %s | test enabled=%s | train eval has no augmentation', list(loaders), RUN_TEST)

        image_folder = PROJECT_FOLDER / 'visualization' / 'runs' / EVAL_RUN
        # 4. Predict, calculate metrics and save one report per split.
        for split, loader in loaders.items():
            split_started = time.perf_counter()
            targets, predictions, confidences = [], [], []
            probabilities = []
            loss_sum = 0.0
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    scores = outputs.softmax(dim=1)
                    confidence, predicted = scores.max(dim=1)
                    if split == 'test':
                        probabilities.extend(scores.cpu().tolist())
                    loss_sum += loss.item() * len(labels)
                    targets.extend(labels.cpu().tolist())
                    predictions.extend(predicted.cpu().tolist())
                    confidences.extend(confidence.cpu().tolist())

            # Confusion matrix: rows are true classes; columns are predictions.
            targets = np.asarray(targets)
            predictions = np.asarray(predictions)
            matrix = confusion_matrix(targets, predictions, labels=range(len(classes)))
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, labels=range(len(classes)), zero_division=0)
            tp = matrix.diagonal()
            fp = matrix.sum(axis=0) - tp
            fn = matrix.sum(axis=1) - tp
            tn = matrix.sum() - tp - fp - fn
            normalized = np.divide(matrix, support[:, None],
                                   out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0)
            ids = list(loader.dataset.indices) if split != 'test' else list(range(len(loader.dataset)))
            records = [dict(id=int(index), true=int(actual), predicted=int(predicted),
                            confidence=float(confidence))
                       for index, actual, predicted, confidence in zip(ids, targets, predictions, confidences)]
            wrong = [row for row in records if row['true'] != row['predicted']]
            per_class = []
            for i, name in enumerate(classes):
                per_class.append(dict(
                    label=i, name=name, precision=float(precision[i]), recall=float(recall[i]),
                    f1=float(f1[i]), support=int(support[i]), tp=int(tp[i]), tn=int(tn[i]),
                    fp=int(fp[i]), fn=int(fn[i]),
                    tpr=float(tp[i] / support[i]) if support[i] else None,
                    fpr=float(fp[i] / (fp[i] + tn[i])) if fp[i] + tn[i] else None,
                    specificity=float(tn[i] / (fp[i] + tn[i])) if fp[i] + tn[i] else None))

            result = dict(
                checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                epoch=saved['epoch'], settings=settings, pretrained_weights=saved['weights'],
                environment=environment, evaluated_at=datetime.now(timezone.utc).isoformat(),
                split=split, count=len(targets), loss=loss_sum / len(targets),
                accuracy=float((targets == predictions).mean()), macro_f1=float(f1.mean()),
                classes=classes, confusion_axes='rows=true, columns=predicted',
                confusion=matrix.tolist(), confusion_normalized=normalized.tolist(),
                per_class=per_class,
                predictions=records, wrong_ids=[row['id'] for row in wrong],
                error_selection='Up to 24 highest-confidence mistakes; tie: original image ID',
                error_gallery=f'visualization/runs/{EVAL_RUN}/{split}_errors.png' if wrong else None,
                seconds=time.perf_counter() - split_started)
            if len(records) != len(loader.dataset) or int(matrix.sum()) != len(records):
                raise ValueError('Prediction count does not match dataset size.')
            if not math.isfinite(result['loss']) or not np.isfinite(confidences).all():
                raise ValueError('Non-finite evaluation result.')
            # ROC needs every class score, in the same row order as predictions.
            if split == 'test':
                scores = np.asarray(probabilities)
                if (scores.shape != (len(records), len(classes)) or
                        not np.isfinite(scores).all() or
                        not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
                    raise ValueError('Invalid test probabilities.')
                result['probabilities'] = probabilities
                result['probability_class_order'] = classes
            save_json(folder / f'{split}_metrics.json', result)

            examples = sorted(wrong, key=lambda row: (-row['confidence'], row['id']))[:24]
            original = loader.dataset if split == 'test' else loader.dataset.dataset
            save_error_gallery(original, examples, classes, split, image_folder / f'{split}_errors.png')
            logger.info('%s | n=%d | loss=%.6f | accuracy=%.2f%% | macro_f1=%.6f | errors=%d',
                        split, result['count'], result['loss'], result['accuracy'] * 100,
                        result['macro_f1'], len(wrong))
            for metrics in result['per_class']:
                logger.info('Class metrics: %s', metrics)
        logger.info('Finished evaluation | splits=%s | seconds=%.2f', list(loaders), time.perf_counter() - started)
    except KeyboardInterrupt:
        logger.warning('Evaluation interrupted; completed split reports preserved')
        raise
    except Exception:
        logger.exception('Evaluation failed')
        raise
    finally:
        close_log(logger)
