# Project Progress

- User requested stopping optimization and uploading all current changes to branch Thái. Notebook currently includes num_workers=2, persistent workers, pinned CUDA memory/non-blocking transfers and epoch-level GPU metric accumulation. Worker benchmark on Windows stalled/resource failure; no speedup verified and no full optimized smoke test completed. Recommend num_workers=0 if startup fails. Changes are preserved as requested, not claimed validated. User CNN log currently reaches epoch 17; it is not a completed 50-epoch result. Git upload includes current artifacts under existing ignore rules; no main merge or force push.

- Latest task: user explicitly requested a basic CNN notebook. Created project-root cnn_basic.ipynb (15 cells): shared data split/ToTensor, 2 Conv-ReLU-Pool blocks -> 32x7x7 -> Linear64 -> 10, 105866 parameters, matched 50-epoch MLP optimizer/schedule, best-val checkpoint, per-epoch log, separate outputs/cnn_runs/lan_N, MLP 10/11/12 comparison, confusion/mistakes and backbone feature maps. RUN_TEST defaults False. All code cells passed on 16 synthetic samples/1 epoch in a removed temporary directory; real split/forward verified. No full CNN training, result claim, commit or push. Added pandas/ipykernel notebook dependencies.

- User explicitly requested a dedicated Markdown run report. Added docs/06_run_report.md with verified configs, 12-run scores, trial interpretation, baseline 10 vs wider 11 vs deeper 12, error/test limitations and artifact links. This user request adds a sixth document beyond the previous five-step layout; no source/training changes or final selection.

- Latest UI correction: user clarified that HTML must compare runs, especially baseline lan_10, wider lan_11 and deeper lan_12. Added three-way scores/accuracy chart, paired fixed/regressed error table, full 12-run scores and run selector feeding existing confusion/class/error gallery from run_review.json. Validation is default, unmeasured test stays absent. JS syntax and mocked DOM rendering for all 12 runs passed; actual browser was not retested. No training, model/config changes, commit or push.

## Current state

- Date: 2026-10-18
- Phase: User is redoing controlled MLP baseline/wider/deeper comparison; basic CNN requested as a later addition.
- Latest task: User requested reviewing/error analysis for all runs. Re-evaluated 12 best.pt on train/validation; test only for already-tested lan_9/11/12. Wrote outputs/run_review.json/csv, visualization/run_confusion.png/run_errors.png and existing docs/05_evaluation.md with per-run comparisons/errors.
- Current config/run: [512,256,128,64], dropout=0.2, Adam lr=0.001, weight_decay=1e-5, 50 epochs, StepLR(15,0.3), batch 64, seed 42. lan_11 completed with best epoch 50, val 91.4833%, loss 0.308205.
- New deeper lan_12 completed 50 epochs. Best epoch 48: val 91.7333%, loss 0.299211; train eval 96.1889%, loss 0.100915. User subsequently ran test: 90.17%, loss 0.357034; re-evaluation matches. No final model selection.
- Regenerated existing 5_report.py for new lan_12: HTML reads current model_report.json and 12-run experiments.json. Removed obsolete old-lan_12 results from evaluation docs/README. No source/HTML edits during analysis.
- Scope: practice_1_fashionmnist only; cifar10_benchmark untouched.
- User explicitly authorized deleting outputs/runs. Exact absolute path verified before recursive removal.
- Earlier outputs/runs reset was completed; user subsequently retrained lan_1 through lan_6. Current baseline lan_6: MLP [256,128,64], best val 91.03% at epoch 22.
- Dataset, outputs/eda.json and single visualization/explorer.html retained.
- src: 5 numbered direct scripts plus config/model/logger/data/metrics modules. data.py and metrics.py prevent numbered task imports from triggering downloads/train/evaluation.
- docs has exactly 5 workflow files; no per-script docs, obsolete experiments or duplicate plans.
- Shared dataset loading belongs to data.py; shared metrics/checkpoint loading belongs to metrics.py. Runnable scripts execute immediately and must not be imported.
- Legacy checkpoint mapping removed; fresh runs carry model/data/train config.
- HTML reads outputs/model_report.json, handles 404 as no model, and lists only evaluated splits.

## Verification

- All numbered scripts parse without main/entry guards; shared modules import without task execution. End-to-end top-level 3_train/4_evaluate/5_report passed with 8 synthetic samples and temporary output only.
- Temporary 1-epoch smoke run on 8 synthetic images: train/val, checkpoint reload and evaluation passed.
- Earlier temporary smoke-test artifacts were removed; actual user/agent runs lan_1 through lan_9 now exist. The synthetic smoke test was not a model-quality experiment.
- Original FashionMNIST split still 54,000/6,000/10,000; no train/validation overlap.
- Deep EDA verified in actual Edge over local HTTP: 10 label rows, 15 extreme examples/split, 8 pairs, all 4 PNG loaded, split/label filters correct, no JS exceptions. Live Server extension not automated.
- Markdown links/line limits and git diff whitespace checked.

## Next step

- Review completed wider/deeper MLP comparison, then analyze MLP errors on request. lan_12 val 91.7333% vs lan_11 91.4833% (+0.25 pp, 15/6000 images) and lan_10 91.3333%. No final selection; CNN remains a later separate step.
- Targets >96% train and >95% val/test remain unverified goals.
- Use validation to choose configuration; test after locking it.
- User authorized uploading all remaining project changes to branch Thái after the src-only push c5dc80b. Preserve ignore rules, Git history and main.

- Verification for latest trial: 30 metrics rows and completed log; best checkpoint validation matches re-evaluation exactly. Baseline values reused from the verified lan_7 comparison; no earlier artifacts overwritten.

- Deep EDA checks: full 54k/6k split match and disjointness; 5000 train-only PCA fit / 1000 validation projection; 2000 sampled train nearest neighbors; finite ordered quantiles, original image labels/IDs and RMSE verified.

- Latest checks: class metrics match sklearn (including zero-support classes); report confusion totals/errors and CSV count verified; actual Edge shows lan_12, 11 experiment rows, 10 class rows, predictions PNG and working split filter, no JS exceptions.
- Reference branch includes custom train/test repartitioning, additional CNN/ROC/PR/Optuna work; do not treat those as mandatory grading requirements or compare its accuracy directly with our official test split.

- Source upload authorization: user explicitly requested replacing the old src tree on branch Thai with all current FashionMNIST src files. Scope is src only; keep Git history and main untouched. Syntax/local-import checks passed before staging; docs/outputs remain local.

- Source upload completed: c5dc80b committed/pushed to origin/Thai with explicit user authorization. Commit includes only practice_1_fashionmnist/src (10 current Python files and deleted legacy paths); source is clean/synced. Other docs/UI/output changes remain local. No main merge or force push.

- Full upload preparation: user approved all remaining tracked/unignored changes, including docs, HTML, EDA and run artifacts. lan_11 has finished 50 epochs (best epoch 50, val 91.48% in log); no training started or interrupted for upload. Dataset, environments and caches remain ignored.

- Review evidence: 12 val scores matched checkpoints; old two-hidden-layer runs evaluated from saved linear tensors with verified ReLU, ToTensor and split settings, without changing production loader. lan_3/5 incomplete. 217 validation IDs wrong in all runs; 11->12 fixes 114, regresses 99. JSON includes per-class metrics, confusion, wrong IDs/confidences and checkpoint SHA256. No training, final choice, commit or push.
