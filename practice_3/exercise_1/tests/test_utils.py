import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils import write_json

def test_writes_utf8_json(tmp_path):
    p = tmp_path / 'run' / 'sample.json'
    write_json(p, {'text': 'tích cực', 'score': 0.9})
    assert json.loads(p.read_text(encoding='utf-8'))['text'] == 'tích cực'
