# AI collaboration record

- User authorized uploading current work to branch Th?i only, without merging main. Reviewed Git scope and checked Python syntax, paired script documentation and whitespace before committing/pushing. Dataset and secrets remain excluded by .gitignore.

- User explicitly requested 04_result.md and model selection. Documented lan_10/best.pt, epoch 12 by minimum validation loss, with verified train/validation/test scores, confusion analysis and plots. Updated README and progress; no additional training, commit or push. Selection is for this assignment, not a claim of global optimality.

- User requested a plotting script in src: added visualize_metrics.py to plot three confusion matrices, one-vs-rest TP/TN/FP/FN, checkpoint loss/accuracy comparisons and actual epoch curves. No fabricated per-epoch train accuracy/test metrics; no additional training.

- User requested stop at run 10 and model visualization. Added model_report.py; evaluated same lan_10/best.pt on train/validation/test, exported original-index mistakes and confusion matrices. Existing HTML now displays configuration, epoch loss, scores, matrix/pair filtering, class metrics and paged misclassified images. No further training or commit/push.

- Run completed: lan_9 trained 20 epochs; its best.pt is epoch 15 by minimum validation loss (0.264620, 90.67%). Evaluate/test were executed (test 89.21%). These are run results, not a final-model decision. No commit/push.

- Epoch validation requested: added best.pt selection by lowest validation loss and retained last-epoch model.pt. User clarified that no final model has been chosen; deleted premature 04_final_model.md and removed finalization claims. Logs and checkpoints preserved.

- Parameter tuning requested: inspected lan_6 baseline (train loss 0.1588; validation loss 0.3187, accuracy 90.05%). Added only StepLR decay after 10 epochs, gamma=0.3; preserved Adam lr=0.001, 20 epochs and current model. New configuration not fully trained yet; no accuracy improvement claim and no test-driven selection.

- User explicitly requested test evaluation and comparison with validation. Added short test.py with run/checkpoint selection, same-model eval/no_grad on both splits and append test.log. No training or model changes; test results should not guide further hyperparameter selection.

- Validation evaluation requested: added evaluate.py to load model.pt and evaluate validation only with eval/no_grad, reporting sample-weighted CrossEntropyLoss and accuracy, append-mode evaluate.log. Added final state_dict saving to train.py because prior runs did not save weights. Verification uses a two-batch training smoke run, not a model-quality experiment; no test evaluation or commit/push.

- Training log enrichment: user requested actual loss, optimizer and parameters recorded once before training. Added runtime introspection of criterion options and optimizer parameter groups plus model/data/device/seed metadata; epoch logs include current lr. Preserved user's SGD, lr=0.01, momentum=0.9 and 20 epochs. Old logs unchanged.

- User requested epoch-only output and readable run names. Removed batch logs; new runs use lan_1, lan_2, etc. Existing directories preserved. Updated agent rules and train documentation; no full training rerun required.

- Minimal training revision: user requested a short basic loop matching example. Reduced train.py to criterion/Adam and direct epoch/batch forward/backward/step, plus one persistent train.log per run. Deferred validation/CSV/checkpoints; old results preserved. New model weights remain in memory only until a separate saving step is added.

- Training simplification: user requested direct epoch/batch loops and code only in src/. Removed main wrapper, config output and source-copy behavior; retained validation, batch/epoch logging, CSV and best/last state_dict files. Deleted only the three .py source copies in the old run; retained its logs/metrics/config/checkpoints unchanged. New checkpoint format documented separately from old run format.

- Completed training run 20261011_151848_737820_9368156e: 10 epochs on RTX 3050 CUDA, 106.5 seconds; final train accuracy 90.29%, validation accuracy 89.33%, validation loss 0.2969. Best checkpoint selected by validation loss at epoch 10, not maximum accuracy. Test untouched.

- Training requested by user: added simple train.py with editable epochs/lr/seed, CrossEntropyLoss/Adam and validation each epoch. Unique timestamp/UUID folders preserve previous runs; log, CSV, config, source snapshots and best/last checkpoints stored together. First 10-epoch CUDA run started; no test evaluation or commit/push.

- MLP sample: user authorized a model template for later parameter edits; added src/model.py using nn.Sequential (784→128→10 with ReLU), plus paired script/model docs. Only forward smoke check; no training, commit or push.

- Example-style simplification: user supplied direct torchvision/transforms/DataLoader example; rewrote data_pipeline.py as a 41-line sequential script, removed CLI/build_loaders/report output, retained validation/test separation. Only path-adapter class remains for shallow data directory. Historical JSON/index outputs are not refreshed; no model training.

- Library-first revision: user requested code using ready-made library components. Replaced manual IDX reading/tensor conversion/splitting with torchvision FashionMNIST/ToTensor, sklearn train_test_split and PyTorch Subset/DataLoader. Small LocalFashionMNIST adapter only overrides raw_folder for the requested shallow layout. Split indices are regenerated by sklearn and therefore differ from the previous handwritten splitter despite the same seed; all default sizes remain unchanged. No training or commit/push.

- Simplification requested by user: replaced custom Dataset and long verification function with TensorDataset/Subset and concise inline checks. Preserved stratified split, CLI options, seeded train shuffle and existing outputs; full image tensors now use approximately 210 MiB RAM. No model training.

- Data pipeline: user requested the next isolated data-preparation step. Created src/data_pipeline.py plus design/script docs; default stratified 54,000/6,000 train/validation, seed 42, test unchanged. Float32 /255 image transform and int64 labels; deterministic train shuffle and sequential evaluation loaders. CLI preparation and split/batch/sample checks passed. No model code or training; no commit/push for these changes.

- Problem introduction: created docs/01_problem.md at user's request, covering FashionMNIST/MLP, input/output, assignment requirements and evaluation scope; linked README. No model code or training.

- Commit approval: user explicitly requested committing prepared changes on Thái. Dataset/cache/secrets remain excluded; push not authorized. Final commit hash is reported in chat and can be verified with git log.

- Git preparation: connected workspace root to tinh633/deep-learning, created local branch Thái from origin/main, added root .gitignore excluding downloaded datasets/cache/secrets and integrated root README. No new commit or push; awaits user approval.

- Source-link update: saved the actual four download URLs and checksums in docs/data_source.md; linked README/preparation docs and updated reusable documentation rules. No dataset or code changes.

## 2026-10-11 — Data download and explorer

- User contribution: supplied Practice 1 assignment; chose Python scripts rather than notebook; requested full-label image browsing and detailed EDA in HTML.
- User modifications: none observed in newly generated project files during this task.
- Agent actions: inspected local rules and dependencies, verified source information, wrote Python preparation script, HTML template, integration checks and documentation.
- Tools: shell/Python for download and statistics; Node for JavaScript verification; web for official source/checksums; Sites skill for local HTML workflow. No publication or messaging tools used.
- Generated components: all files in this project except original dataset files downloaded from the author.
- Request purpose: inspect data before model development.
- Validation: four MD5 checksums passed; 33 gallery filters and final-image access passed; three EDA views passed; torchvision independently loaded both splits and matched counts and pixel means.
- Failure resolved: torchvision initially could not find extracted IDX files; added extraction from verified compressed originals, then validation passed.
- Results: 60,000 train / 10,000 test; balanced classes; no exact duplicate pixel arrays found.
- Limitations: no visual browser QA; no near-duplicate or manual label audit; no model training yet.
- Approval state: implementation requested by user; completed result awaits review. No commit or push.

## Layout revision — Part 2

- User requested continuation after review of updated file-placement rules.
- Moved HTML template and generated page into `visualization/`; flattened data to `data/FashionMNIST_raw/`.
- Updated preparation/verification paths and project documentation. No HTML behavior changes.
- Verified SHA-256 unchanged for all eight moved data files; preparation passed four MD5 checks without downloading again; Node integration checks passed.
- Previous torchvision validation applied to the old cache layout. The new shallow directory is not its default cache layout.
- HTML viewing issue remains pending for Part 3. No commit or push.

## HTML viewing revision — Part 3

- Diagnosed a concrete issue: the open `explorer.html` was a template with unresolved placeholders, which cannot parse as dataset JSON.
- Renamed source to `visualization/explorer_template.html`; actual data viewer is now `visualization/explorer.html`. Updated generation and verification paths.
- Added explicit Windows/VS Code opening instructions; source template is not a directly viewable dataset page.
- Actual headless Edge test passed gallery/EDA rendering, nonblank pixel canvas, 11 label options, test/class filtering and final-image dialog, with no JavaScript exceptions.
- Inspected screenshot `outputs/browser_preview.png`; gallery and controls are visible. Node 33-filter regression check also passed.
- User desktop viewing confirmation remains pending. No model training, commit or push.

## Python-only src revision

- User clarified reusable layout preference: src contains only Python files; visual checks live with visualization assets.
- Moved verify_explorer.cjs and browser_check.cjs from src/ to visualization/; updated commands, script paths, design, and reusable agent rules.
- Validation: 33-filter Node integration checks passed from the new location; browser-check JavaScript syntax passed. src contains only prepare_data.py.
- Browser smoke test was not rerun for this relocation; its project-root calculation is unchanged. Prior Edge results remain historical evidence.
- User confirmed assignment uses MLP but explicitly deferred model work. No training, commit or push.

## Single HTML viewer revision

- User requested one HTML and reported no dataset in VS Code Preview.
- Removed redundant explorer_template.html; preparation now refreshes embedded data in explorer.html while preserving UI code and validating block counts.
- Updated reusable rules and active documentation; added visible guidance when the loading message persists in IDE Preview.
- Validation: preparation checksum checks passed; 33-filter integration checks passed; actual Edge test passed rendered gallery/EDA, nonblank canvas and detail/filter interactions without JS exceptions.
- VS Code Preview remains unverified; its specific restriction has not been established. Instructions use verified Edge/Chrome opening rather than claiming Preview is fixed.
- No dataset deletion, training, commit or push.

## Live Server revision

- User requested Live Server viewing first and removal of unnecessary .cjs files.
- Replaced embedded base64 dataset with async fetch of EDA JSON and original extracted IDX files; HTML reduced from approximately 70 MiB to 18 KB.
- Added loading/error messages, HTTP requirement, input validation and disabled controls during initialization.
- Python now prepares only dataset and EDA, preserving the HTML UI.
- Preparation passed four MD5 checks. Actual headless Edge over local HTTP passed gallery/EDA, nonblank canvas, class/test filter and final-image dialog with no JS exceptions; screenshot refreshed.
- Removed both project .cjs scripts, their paired docs and temporary verification tools. No Node required for project use.
- Updated reusable layout rules, design and Live Server instructions. Extension itself not automated; user viewing confirmation pending.
- No data deletion, training, commit or push.
