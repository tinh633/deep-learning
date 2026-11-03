# 06 — Reproducibility, artifact integrity and Git hygiene

## Minimum reproducibility record

A fully reproducible lab record needs: dataset Hub ID and revision (record manually if strict pinning is required), sample selection seed, split sizes and stable hashes, model Hub ID/revision, tokenizer max length, actual dependency versions, device, trainer arguments and the saved checkpoint. A seed improves repeatability but floating-point operations and GPU kernels can introduce nondeterministic differences.

## Generated artifacts are not source code

`outputs/data/` contains Arrow datasets, `outputs/runs/` contains models, optimizer checkpoints and TensorBoard logs. These can be very large. Root `.gitignore` excludes them by default. Prefer committing source, documentation, notebook and selected compact metrics or a release attachment, subject to course rules. Do not force-add model files to GitHub without checking repo file limits and your team's agreement.

## Data integrity and source trace

`split_info.json` records review count, class count and a SHA256 hash of ordered `(text,label)` entries. If data changes, hashes change and the old model may no longer correspond to current `outputs/data/`. Before using an older `lan_N` verify its model ID, split provenance and actual data directory. **Do not rerun prepare-data with changed settings and then evaluate an older model as if nothing changed.**

## Output naming and non-overwrite rules

- Each training attempt creates a new `lan_N` directory.
- Training writes `TRAINING_COMPLETE` only after final model, history and metrics are persisted.
- Test evaluation requires a second explicit flag and refuses overwriting `test_metrics.json`.
- Reports only read artifacts and can be regenerated; missing metrics remain absent, not invented.

## Limitations of implementation

- This lab does not pin remote Hub commit revisions or exact dependency patch versions. For strict archival runs add explicit revisions and record `pip freeze`.
- Text overlap checks detect exact strings, not near-duplicates.
- Validation and test are small subsets, not full IMDb results.
- PyTorch/CUDA nondeterminism and environment drift may still influence scores.
- Pretrained models inherit upstream data limitations and possible bias; this lab does not audit them.

## Review checklist

1. `python -m pytest practice_3_huggingface/tests -q` passes.
2. Correct .venv interpreter visible in VS Code.
3. Check `outputs/data/split_info.json` has expected label counts.
4. Check `config.json` is run-specific and `TRAINING_COMPLETE` exists.
5. Check metrics contain no fabricated values and test is only executed after lock.
6. Review tracked files with `git diff --cached --stat` before commit.
