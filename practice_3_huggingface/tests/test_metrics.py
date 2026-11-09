"""Run with: python -m pytest practice_3_huggingface/tests -q"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import compute_metrics
from utils import sha256_records

def test_perfect_binary_classification():
    scores=compute_metrics((np.array([[3,0],[0,3],[4,0],[0,4]]),np.array([0,1,0,1])))
    assert scores == dict(accuracy=1.,precision=1.,recall=1.,f1=1.)

def test_one_class_prediction_is_safe():
    scores=compute_metrics((np.array([[1,0],[1,0],[1,0]]),np.array([1,1,0])))
    assert scores['f1']==0
    assert scores['precision']==0
    assert 0<=scores['accuracy']<=1

def test_hash_is_order_sensitive():
    a=[{'text':'one','label':0},{'text':'two','label':1}]
    assert sha256_records(a)!=sha256_records(a[::-1])
