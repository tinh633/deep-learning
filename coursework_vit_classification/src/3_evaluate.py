"""What: Đánh giá checkpoint trên một split. Why: Metric và lỗi theo lớp có thể tái tính.
Who: Nhóm chạy sau khi chọn checkpoint bằng validation. When: Test chỉ sau khóa cấu hình.
Where: outputs/runs/lan_N/evaluation. How: eval/inference_mode, sklearn, lưu predictions không pixel.
"""
import hashlib
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

import config
from data import build_evaluation_loader, preprocessing_settings
from logger import guard_memory, read_checkpoint, resolve_run, save_json
from model import model_from_checkpoint


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if config.EVAL_CHECKPOINT not in ('best.pt', 'last.pt'):
        raise ValueError('EVAL_CHECKPOINT phải là best.pt hoặc last.pt')
    if config.EVAL_SPLIT == 'test' and not config.RUN_TEST:
        raise SystemExit('Test cần RUN_TEST=True, không chạy test ngầm.')
    folder = resolve_run(config.EVAL_RUN)
    checkpoint_path = folder / config.EVAL_CHECKPOINT
    guard_memory()
    state = read_checkpoint(checkpoint_path)
    settings = state['settings']
    if settings['phase'] == 'smoke' and not config.ALLOW_SMOKE_CHECKPOINT:
        raise SystemExit('Checkpoint smoke không phải model train đầy đủ; từ chối đánh giá mặc định.')
    if config.EVAL_SPLIT == 'test' and settings['phase'] != 'final':
        raise ValueError('Chỉ checkpoint phase=final được đánh giá official test')
    if settings.get('preprocessing') != preprocessing_settings():
        raise ValueError('Checkpoint thiếu/khác preprocessing; không suy đoán transform.')
    split_hash = hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest()
    if settings['split_sha256'] != split_hash:
        raise ValueError('Split hiện tại khác split checkpoint')
    batch_size = config.BATCH_SIZE or settings['batch_size']
    loader, classes, ids = build_evaluation_loader(config.EVAL_SPLIT, batch_size,
                                                  config.NUM_WORKERS, config.MAX_EVAL_IMAGES)
    if classes != settings['classes']:
        raise ValueError('Class order không khớp checkpoint')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_from_checkpoint(state, device)
    del state
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    true, scores, loss_sum = [], [], 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            if outputs.shape != (len(labels), 100) or not torch.isfinite(outputs).all():
                raise ValueError('Logits sai shape hoặc NaN/Inf')
            loss_sum += float(criterion(outputs, labels.to(device)))
            true.extend(labels.tolist())
            scores.extend(outputs.softmax(1).cpu().tolist())
            guard_memory()
    seconds = time.perf_counter() - start
    y = np.asarray(true)
    probabilities = np.asarray(scores)
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(y, predicted, labels=range(100))
    classification = classification_report(y, predicted, labels=range(100),
                                            target_names=classes, output_dict=True, zero_division=0)
    per_class, curves = [], {}
    for label, name in enumerate(classes):
        tp = int(matrix[label, label])
        fn = int(matrix[label].sum()) - tp
        fp = int(matrix[:, label].sum()) - tp
        tn = len(y) - tp - fn - fp
        binary = y == label
        auc = ap = None
        if binary.any():
            ap = float(average_precision_score(binary, probabilities[:, label]))
        if binary.any() and not binary.all():
            auc = float(roc_auc_score(binary, probabilities[:, label]))
            fpr, tpr, _ = roc_curve(binary, probabilities[:, label])
            curves[name] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
        per_class.append(dict(class_name=name, label=label, tp=tp, fn=fn, fp=fp, tn=tn,
                              tpr=tp/(tp+fn) if tp+fn else None,
                              fpr=fp/(fp+tn) if fp+tn else None,
                              specificity=tn/(fp+tn) if fp+tn else None,
                              auc=auc, ap=ap, **classification[name]))
    aucs = [row['auc'] for row in per_class if row['auc'] is not None]
    aps = [row['ap'] for row in per_class if row['ap'] is not None]
    normalized = np.divide(matrix, matrix.sum(1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(1, keepdims=True)!=0)
    with checkpoint_path.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(run=folder.name, split=config.EVAL_SPLIT, phase=settings['phase'],
                  model=settings['model'], experiment=settings['experiment'],
                  checkpoint=checkpoint_path.name, checkpoint_sha256=checkpoint_hash,
                  epoch=read_checkpoint(checkpoint_path)['epoch'], split_sha256=split_hash,
                  preprocessing=settings['preprocessing'], classes=classes, n=len(y),
                  limited_to=config.MAX_EVAL_IMAGES, loss=loss_sum/len(y),
                  accuracy=float((y==predicted).mean()), correct=int((y==predicted).sum()),
                  top5_accuracy=float((np.argsort(probabilities, axis=1)[:, -5:]==y[:, None]).any(1).mean()),
                  macro_f1=classification['macro avg']['f1-score'],
                  weighted_f1=classification['weighted avg']['f1-score'],
                  macro_auc=float(np.mean(aucs)) if aucs else None, auc_classes=len(aucs),
                  macro_ap=float(np.mean(aps)) if aps else None, ap_classes=len(aps),
                  averaging='OvR AUC/AP over defined classes; macro-F1 over 100 labels, zero_division=0',
                  confusion_matrix=matrix.tolist(), normalized_confusion=normalized.tolist(),
                  per_class=per_class, roc_curves=curves, evaluation_seconds=seconds,
                  timing_scope='loader/transform/transfer/forward/guard; not a forward-only benchmark',
                  precision='FP32', device=str(device),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)
    parent = folder / 'evaluation'
    parent.mkdir(exist_ok=True)
    number = 1
    while (parent / f'{config.EVAL_SPLIT}_{checkpoint_path.stem}_{number}').exists():
        number += 1
    output = parent / f'{config.EVAL_SPLIT}_{checkpoint_path.stem}_{number}'
    output.mkdir()
    predictions = [dict(id=source_id, source='test' if config.EVAL_SPLIT=='test' else 'train',
                        true=int(y[i]), pred=int(predicted[i]), probabilities=scores[i])
                   for i, source_id in enumerate(ids)]
    save_json(output / 'metrics.json', report)
    save_json(output / 'predictions.json', predictions)
    (output / 'evaluate.log').write_text(
        f'{folder.name} {checkpoint_path.name} epoch={report["epoch"]} {config.EVAL_SPLIT} n={len(y)}\n'
        f'accuracy={report["accuracy"]:.4f} macro-F1={report["macro_f1"]:.4f}\n', encoding='utf-8')
    guard_memory()
    print(f'Đã lưu {output} | accuracy {report["accuracy"]:.2%} | macro-F1 {report["macro_f1"]:.4f}')
