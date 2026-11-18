"""Metric computation shared by Trainer and the evaluation script."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def logits_array(predictions):
    return np.asarray(predictions[0] if isinstance(predictions, tuple) else predictions)


def positive_probabilities(predictions):
    logits = logits_array(predictions).astype(np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def build_metrics():
    import evaluate

    accuracy = evaluate.load("accuracy")

    def compute_metrics(result):
        predictions = logits_array(result.predictions).argmax(axis=-1)
        labels = result.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1, zero_division=0
        )
        measured = accuracy.compute(predictions=predictions, references=labels)
        return {
            "accuracy": float(measured["accuracy"]),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
        }

    return compute_metrics
