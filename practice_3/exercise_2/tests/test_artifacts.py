import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from utils import write_json, next_run

def test_writes_utf8_json(tmp_path):
    p=tmp_path/'run'/'sample.json'
    write_json(p, {'text':'tích cực', 'score':0.9})
    assert json.loads(p.read_text(encoding='utf-8'))['text']=='tích cực'

def test_new_runs_do_not_overwrite(tmp_path):
    assert next_run(tmp_path).name=='lan_1'
    assert next_run(tmp_path).name=='lan_2'
