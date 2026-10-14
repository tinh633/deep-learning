# Project Progress

> Record verified project facts. Do not mark tasks complete without evidence.

## Current State

- Last updated: 2026-10-11
- Current phase: Result report written; lan_10/best.pt selected for the assignment at user request
- Current task: Consistency fixes after review (selection wording, stale README, checklist). Earlier: created docs/04_result.md with checkpoint selection, verified scores, error analysis and plots; synchronized README and AI usage. No new training.
- Git state / last commit: f193a6a committed and pushed to origin/Thái with user authorization; current documentation changes are not committed.

## Task Checklist

- [x] Define project problem and acceptance criteria
- [x] Analyze available data and code
- [x] Agree on architecture
- [x] Implement scoped tasks
- [x] Run and verify tests (train/evaluate/test logs for lan_1–lan_10; no automated test suite)
- [x] Present results for user review (docs/04_result.md)
- [ ] Obtain explicit approval before commit

## Status Definitions

- Pending: Not started.
- In progress: Work begun, not yet validated.
- Validated: Acceptance checks passed with evidence.
- Approved: User explicitly accepted the reviewed result.
- Committed: Approved changes were committed; record hash.

## Last Session Handoff

- Session date: 2026-10-11
- User decision: Explicitly requested 04_result.md and model finalization for the assignment.
- Selected artifact: practice_1_fashionmnist/outputs/runs/lan_10/best.pt, epoch 12 by minimum validation loss.
- Verified report scores: train loss 0.170986 / accuracy 93.50%; validation 0.275095 / 90.57%; test 0.320097 / 89.41%.
- Evidence: model_report.json, train.log and existing plots; log records 23 of 30 planned epochs.
- Documentation: docs/04_result.md, project README and AI_usage updated. No code or training changes.
- Limitations: Test was viewed in earlier experiments; no multi-seed stability study or global optimality claim.
- Git authorization: User explicitly requested committing/pushing current work to Thái only; no merge into main. Syntax/docs and diff checks passed before upload. Verify final commit and remote state with Git.

## Review fixes (2026-10-12)

- lan_9 has lower validation loss (0.2646) than lan_10 (0.2751); lan_10 kept as user-selected, wording in 04_result.md corrected. lan_10 stopped at epoch 23/30 (no model.pt), recorded.
- Re-verified lan_10 train/validation/test loss, accuracy and confusion matrices with NumPy and with PyTorch: all match reports (also lan_9 val/test). Added lan_9 model_report.json and plots for comparison.
- Scripts accept `[lan_N]` on the command line (test, evaluate, model_report, visualize_metrics); explorer.html accepts `?run=lan_N`. Added src/show_predictions.py (predictions.png, confident_mistakes.png).
- Added docs/05_experiments.md (10-run table from logs, assignment checklist); fixed stale docs (03_model, model.md, train.md, test.md, 01_problem, README rewritten).
- Verified by running: PyTorch (CPU) re-ran data_pipeline, model, evaluate, test, model_report (lan_10 report identical in confusion/mistakes; lan_9 chain also run). explorer.html tested in jsdom for default, ?run=lan_10, ?run=lan_9, invalid run. Not run: train.py (no retraining), real browser, Live Server extension.

## Next Session

1. Review docs/04_result.md with the user.
2. Keep selected checkpoint and historical runs intact; do not start new experiments without a new request.
3. Commit only after explicit approval; push only with separate authorization.

## Archive

Move older session notes to linked files under `docs_for_agent/progress_archive/` at the workspace root when needed.
