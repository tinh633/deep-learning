"""Binary metrics with positive class 1 (POSITIVE).

Uses the Hugging Face `evaluate` library when its metric modules can be loaded
(the route taken by the course tutorial); otherwise falls back to scikit-learn so
training never fails just because the Hub is unreachable. Both give identical
values for binary averaging with pos_label=1.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

_HF_METRICS = None   # None = not tried yet, False = unavailable

def _hf_metrics():
    global _HF_METRICS
    if _HF_METRICS is None:
        try:
            import evaluate
            _HF_METRICS = evaluate.combine(['accuracy', 'precision', 'recall', 'f1'])
        except Exception:
            _HF_METRICS = False
    return _HF_METRICS

def sklearn_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0)
    return {'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    hf = _hf_metrics()
    if hf:
        result = hf.compute(predictions=preds.tolist(), references=labels.tolist())
        return {k: float(result[k]) for k in ('accuracy', 'precision', 'recall', 'f1')}
    return sklearn_metrics(labels, preds)

def majority_baseline(labels):
    """Accuracy/F1 of always predicting the most frequent label (floor reference)."""
    labels = np.asarray(labels).astype(int)
    majority = int(np.bincount(labels).argmax())
    preds = np.full_like(labels, majority)
    return {'majority_label': majority, **sklearn_metrics(labels, preds)}
