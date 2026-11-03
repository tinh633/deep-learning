"""Binary metrics with positive class 1 (POSITIVE). No external metric download."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.asarray(logits).argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0)
    return {'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
