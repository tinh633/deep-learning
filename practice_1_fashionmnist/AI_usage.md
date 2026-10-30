# AI collaboration record

- Handoff upload explicitly authorized: commit message asks readers to read progress for remaining tasks and review the entire project. Checked current branch Thái and documentation diff; upload contains handoff Markdown and related links/records only.

- User requested handoff Markdown for others: created project progress.md covering MLP 1–9 exploration, confirmed baseline10/wider11/deeper12 comparison, sequential CNN architecture/hyperparameter experiments, best-family comparison, detailed error-analysis acceptance checks and HTML axes/ticks/test-curve constraints. Verified actual incomplete CNN log; reused existing notebook. Added README/agent-progress links. Documentation only, no new model run or final selection.

- User requested faster data loading, then stopped work and authorized uploading all changes. Added optional 2-worker train loader, persistent/prefetch workers, pinned CUDA memory/non-blocking transfers and GPU-side metric accumulation to notebook. Windows multi-worker benchmark stalled with system resource errors; no performance gain established. Earlier successful synthetic test predates these speed edits. Preserve current implementation for upload; num_workers=0 remains fallback. CNN user run log has 17 epochs, not a completed experiment. No further training or optimization under this stop request.

- CNN notebook explicitly requested: added cnn_basic.ipynb outside src with 2-block CNN backbone and classification head, matched data/training settings against MLP 10–12, isolated CNN run logging/checkpoints, comparison, validation confusion/class metrics/mistakes, feature maps and opt-in test. Added pandas/ipykernel requirements and usage docs. Checked every code cell syntax and executed all cells with actual dataset initialization plus a temporary synthetic 16-sample one-epoch training/evaluation/plot test; old runs untouched. No full CNN training or claim of improved accuracy, no commit/push. Notebook schema library unavailable; JSON structure checked directly.

- Dedicated run report requested: wrote docs/06_run_report.md summarizing actual architecture/hyperparameters, all 12 best-checkpoint results, parameter trials and baseline/wider/deeper comparisons, error patterns and limitations. Used verified run_review artifacts; no new experiment or source edits. Linked from README; sixth document explicitly requested by user.

- User corrected report scope: updated existing explorer.html to compare baseline 10 / wide 11 / deep 12 explicitly and select any of 12 runs for confusion/classes/mistake gallery, using previously verified run_review.json. Added score/accuracy comparison, paired error changes and all-run metrics. Missing test is — and validation defaults. Inline JS syntax and all 12 run renderings checked with mocked DOM/local files; no actual browser retest or new training/commit/push.

- All-run review requested: audited 12 best checkpoints/configs/history/logs; actual widths inferred from weights (old lan_5/6 omit hidden3). One-off saved-linear-tensor/ReLU eval with original ToTensor/split reproduced every validation score; measured train/val and already-tested 9/11/12 only. Exported run_review.json/csv with SHA256, per-class TP/TN/FP/FN/F1, wrong IDs/confidences, fixed/regressed IDs; plotted val confusion/mistake montage. Updated existing evaluation docs/README and regenerated current lan_12 report. Inspected montage: similar silhouettes are a hypothesis, not proof of bad labels. No source edits, training, final selection, commit or push.

- lan_12 requested and completed: deeper MLP [512,256,128,64], keeping lan_11 dropout/optimizer/scheduler/seed/split/epochs fixed. Added optional fourth hidden layer with explicit forward; verified old checkpoint reload and finite gradients. Trained 50 CUDA epochs, best epoch 48. Reloaded best.pt: train 96.1889%/loss 0.100915, val 91.7333%/loss 0.299211, matching saved validation. Gain vs lan_11 is 0.25 pp (15 images), not proof of general superiority from one seed. Saved eval log, updated training docs/progress. No test, final model selection, report regeneration, commit or push; HTML remains historical.

- Full repository upload: user explicitly authorized committing and pushing all remaining changes to branch Thái, extending the prior src-only scope. Checked repository status, ignored datasets/secrets, artifact sizes and completed lan_11 log. Includes current documentation, visualization, EDA, run artifacts and recorded deletions; no new training, main merge or force push.

- Source refresh upload requested: user authorized committing/pushing all current FashionMNIST src and recording deleted legacy modules on branch Thai. Verified 10 Python files parse and imports no longer reference removed local modules; fetched remote state. Commit scope is src only; documentation/outputs are not included in this upload. No force push, main merge or training.

- New comparison plan: user is redoing MLP with baseline, wider lan_11 and deeper lan_12, then MLP error analysis and one additional basic CNN. Read current logs: lan_10 baseline [256,128,64]; lan_11 currently records [512,256,128] with the same training settings; old lan_12 train.log is absent. Documented a proposed deeper [512,256,128,64] and later basic CNN, keeping split/validation protocol. Did not edit source, interrupt/start training or implement CNN; flagged that exported old lan_12 report must not be treated as new-run evidence.

- External LAB1 comparison requested: browsed zombieTDV/Deeplearning_Course_UTH branch LAB1-FashionMNIST-Classification (data/train/eval/model/vis utilities, MLP changelog and audit). Used reporting ideas with attribution, not source/weights/results copying. Kept official test split because reference utilities can repartition the combined 70k dataset; their scores are not directly comparable. Strengthened existing report step with per-class precision/recall/F1, correct/wrong examples, classification CSV and checkpoint-derived experiment history. Regenerated user's selected lan_12 report and HTML; train 96.1296%, val 91.4%, test 90.05%, matching prior log. Added lab checklist/brief report to existing docs. Validated metrics against sklearn including empty classes, counts/CSV/history, doc links and actual Edge rendering/filters. No architecture/config change, training, final-model selection, commit or push; no guarantee of a grade without the instructor's rubric.

- Train readability: user found continuous code and inline logging hard to follow. Added blank-line-separated setup/train/validation/save blocks, expanded settings/metric/checkpoint fields vertically and moved epoch string formatting into existing logger.log_epoch. Kept batch learning steps visible, direct script execution and existing checkpoint/log semantics. Syntax and exact one-line log-format checks passed; no training, model/config edits, new helper file, commit or push.

- 2026-10-18 requested implementation: numbered runnable scripts 1_preparedata through 5_report, removed main/__main__ guards from those scripts, kept side-effect-free data.py and metrics.py for shared imports. Implemented deep train/validation EDA and single-HTML display: full split feature distributions/class summaries, mean/variation images, train-fit PCA, traceable extreme images and sampled nearest pairs. Exported JSON and four PNG figures. Verified split/image IDs/RMSE, browser filters and PNG loading in Edge, plus top-level train/evaluate/report on 8 synthetic samples in a temporary directory. No actual training run, model/config changes, test scoring, commit or push.

- 2026-10-18 continuity review: user requested rereading src and agent/project docs before further tasks. Inspected all 8 Python files, 5 workflow docs, README, AI usage, rules, HTML report paths and run metrics. Verified lan_9 completed 50 epochs with dropout 0.2 and StepLR(25,0.3); best validation 91.05% at epoch 41. Current evaluation target remains lan_1, and narrative docs lag the latest run. Updated this record/progress only; no source edits, training, evaluation, commit or push.

- Continued tuning authorized: lan_8 changes only dropout to 0.1 versus lan_6 baseline, restoring weight_decay=0; keeps [256,128,64], seed/split/batch, Adam lr, StepLR and 30 epochs. Completed 30/30 epochs; best epoch 22 val 90.9667%, loss 0.278345. Reloaded best checkpoint and evaluated train (94.4611%) and validation with dropout disabled, saved comparison with verified lan_6/7 scores. Gap reduced to 3.4944 pp but accuracy did not surpass baseline (91.0333%); no test, architecture expansion, final selection, commit or push.

- Regularization trial authorized: kept user's MLP [256,128,64], seed/split 42, batch 64, Adam lr=0.001 and StepLR(15,0.3), 30 epochs; changed only weight_decay 0 to 0.0001. Recorded hidden3 explicitly in config without changing architecture. lan_7 completed; best epoch 26 validation 90.85% vs lan_6 91.03%. Reloaded both best checkpoints and evaluated train/validation only, saving comparison JSON/evaluate logs. Val loss improved 0.312959 to 0.300252, but accuracy declined 0.1833 percentage points; no new final-model decision, test, commit or push.

- PDF review: Read the user-provided 25-page Chapter 3 training document and visually inspected code examples on pages 5, 6, 7, 8, 11 and 12. Recorded its direct PyTorch coding style in training docs and reusable coding rules. Kept user requirements over simplified slide choices (separate val/test, epoch-only logs, dataset-specific preprocessing). Used a temporary PDF reader only; no project dependency, source-code or training changes.

## Current cleanup/reset - 2026-10-14

- User contribution: requested docs by workflow step only, fewer main source files and deletion of all runs to restart.
- Agent actions: consolidated src into 8 files, docs into 5 steps; removed obsolete per-script docs and historical report/plan duplication.
- Deletion: verified D:/newtask/DL/practice_1_fashionmnist/outputs/runs exactly, then removed it at explicit user request. No archived copy or selected checkpoint remains.
- Retained: raw FashionMNIST, EDA numeric report, single HTML viewer and unrelated cifar10_benchmark files.
- Updated HTML model report path and no-model state; dataset gallery remains separate.
- Code: no dynamic MLP builder, historical registry or duplicate training pipeline; model/config/logger stay explicit.
- Verification: source import/parse, temporary 1-epoch train on 8 synthetic images, checkpoint reload, same split/no overlap, JavaScript syntax and documentation checks passed.
- Limits: no actual model-quality training or test evaluation; browser rendering not retested.
- Git: no new commit/push. Deletions and cleanup remain reviewable locally.

## Historical context

Earlier collaboration produced data preparation, EDA, HTML and MLP experiments, including lan_10 and E5.
Those run artifacts were intentionally removed in this reset; their historical scores are not current results.
The user repeatedly requested short direct code, explicit forward operations and step-by-step review.
Current requirements and running instructions are in README.md and the five docs files.

- Source upload result: c5dc80b successfully pushed to origin/Thai. Only src included; source status clean and local/remote commit hashes match. Other working changes remain uncommitted locally.

## 2026-10-30 ? Branch synchronization

- Synced all 152 tracked Practice 1 files from lab-1-hoan-chinh commit 080a41e; 102 files added/updated including byte-format differences. Local data and Practice 2 changes preserved.
- Imported CNN runs 2/3, cnn_new.ipynb and MLP error analysis. Verified every imported file against source Git bytes before adding this journal entry; JSON parsing and notebook Python compilation passed.
- No train/evaluate/notebook execution, commit or push. Next task: presentation report notebook using synchronized artifacts; existing training notebooks remain separate.

## 2026-10-30 - CNN new post-test analysis

- Scope: cnn_new.ipynb only; MLP stays on existing HTML. Added historical-result interpretation, standalone test inference/cache, per-class TP/TN/FP/FN/TPR/FPR, confusion, ROC/AUC, PR/AP and rule-based error/confidence analysis.
- Historical output: best epoch35, val92.20%, test91.47%, 853/10000 errors. Checkpoint cnn_new_best.pt absent locally and from imported branch; CNN runs 1-3 have different architectures and were not substituted.
- New cells compile and missing-checkpoint path passes. Synthetic perfect/chance ROC/AP, rates/confusion/gallery and invalid probabilities checks passed; synthetic outputs not saved. Real per-class inference awaits checkpoint. No train, commit or push.

## 2026-10-30 - MLP lan_10 test

- Evaluated original best.pt epoch39 on CUDA, no training. Test n=10000: accuracy 89.65%, loss 0.32805356, macro F1 0.89640104, 1035 mistakes. Train/validation remain 95.15%/91.3333%.
- Existing evaluate/report scripts produced test.log, model_report.json, class CSV and plots. Updated run_review JSON/CSV for HTML; verified checkpoint SHA, confusion totals and error count.
- Corrected config DATA_FOLDER to existing data/FashionMNIST_raw after imported branch path failed dataset lookup. No data download, commit or push.

- 2026-10-30: Publishing Practice 1 CNN analysis and lan_10 test reports to lab-1-hoan-chinh; Practice 2 excluded. Local cnn_basic.ipynb deletion is excluded to preserve the existing remote notebook.
