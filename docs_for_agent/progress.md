# Project Progress

## Current state

- Date: 2026-10-18
- Phase: User is redoing controlled MLP baseline/wider/deeper comparison; basic CNN requested as a later addition.
- Latest task: Checked actual logs/config and documented sequence: lan_10 [256,128,64] baseline, user-running lan_11 [512,256,128] wider, proposed new lan_12 [512,256,128,64] deeper; then MLP errors and basic CNN. No source edits or new execution.
- Current config/run: [512,256,128], dropout=0.2, Adam lr=0.001, weight_decay=1e-5, 50 epochs, StepLR(15,0.3), batch 64, seed 42. lan_11 log exists and has not finished at inspection; do not interfere.
- Old lan_12 folder/train.log no longer exists. Its earlier 96.13/91.40/90.05 report is historical only; new deeper lan_12 is not implemented/run by agent. EVALUATE still points to lan_12 and should only be used after intended checkpoint exists.
- HTML now reads the refreshed lan_12 report rather than stale lan_9; class CSV and predictions PNG under lan_12, all-run metadata in outputs/experiments.json. Historical run configurations are read from their own checkpoints.
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

- Finish user-run wider MLP, implement deeper variant when requested, compare validation then analyze MLP errors. User has requested a later basic CNN; handle it as a separate step after MLP review, not simultaneously.
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
