"""Run allocation and JSON artifacts shared by Exercise 2 entry points."""

import hashlib
import json
import re
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import DATA_DIR, RUNS


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_records(dataset):
    digest = hashlib.sha256()
    for sample in dataset:
        row = [sample["source_index"], sample["label"], sample["text"]]
        digest.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def next_run():
    RUNS.mkdir(parents=True, exist_ok=True)
    for index in count(1):
        run = RUNS / f"lan_{index}"
        try:
            run.mkdir()
        except FileExistsError:
            continue
        return run


def completed_run(name):
    if not re.fullmatch(r"lan_[1-9][0-9]*", name):
        raise SystemExit("Use an explicit run name such as --run lan_1.")
    run = RUNS / name
    if not (run / "TRAINING_COMPLETE").is_file() or not (run / "model" / "config.json").is_file():
        raise SystemExit(f"Training artifacts are incomplete in {run}. Select a completed run.")
    return run


def data_metadata(path):
    path = path.resolve()
    if not (path / "split_info.json").is_file():
        raise SystemExit(f"Prepared data missing in {path}. Run src/1_preparedata.py first.")
    return read_json(path / "split_info.json")


def add_data_argument(parser):
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Prepared snapshot directory")
