"""Tests for baselines, report helpers and the evaluate/sklearn metric paths."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import metrics  # noqa: E402

def load_report():
    spec = importlib.util.spec_from_file_location('report5', SRC / '5_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_majority_baseline_predicts_most_frequent_label():
    result = metrics.majority_baseline([1, 1, 1, 0])
    assert result['majority_label'] == 1
    assert result['accuracy'] == 0.75
    assert result['recall'] == 1.0

def test_sklearn_path_matches_manual_values():
    labels = np.array([1, 1, 0, 0]); preds = np.array([1, 0, 0, 1])
    scores = metrics.sklearn_metrics(labels, preds)
    assert scores['accuracy'] == 0.5 and scores['precision'] == 0.5 and scores['recall'] == 0.5

def test_evaluate_and_sklearn_agree_when_evaluate_available(monkeypatch):
    pytest.importorskip('evaluate')
    if not metrics._hf_metrics():
        pytest.skip('evaluate metric modules not loadable (offline)')
    logits = np.array([[0.1, 2.0], [2.0, 0.1], [0.3, 0.9], [1.0, 0.2]])
    labels = np.array([1, 0, 0, 0])
    with_hf = metrics.compute_metrics((logits, labels))
    monkeypatch.setattr(metrics, '_HF_METRICS', False)
    assert with_hf == pytest.approx(metrics.compute_metrics((logits, labels)))

def test_compute_metrics_falls_back_without_evaluate(monkeypatch):
    monkeypatch.setattr(metrics, '_HF_METRICS', False)
    scores = metrics.compute_metrics((np.array([[0, 1], [1, 0]]), np.array([1, 0])))
    assert scores['accuracy'] == 1.0

def test_error_direction_follows_data_not_a_fixed_sentence():
    report = load_report()
    positive_better = np.array([[40, 10], [2, 48]])
    negative_better = np.array([[48, 2], [10, 40]])
    assert 'POSITIVE reviews' in report.error_direction(positive_better).split('more reliably')[0]
    assert 'NEGATIVE reviews' in report.error_direction(negative_better).split('more reliably')[0]
    assert 'equally' in report.error_direction(np.array([[5, 3], [3, 5]]))

def test_fmt_support_tolerates_missing():
    report = load_report()
    assert report.fmt_support(None) == 'N/A'
    assert report.fmt_support(250) == '250'
