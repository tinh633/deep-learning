# Documentation & Project Layout

## General

- Maximum 150 lines per Markdown file. Split longer files and link them using relative paths.
- Keep documentation concise, factual, and synchronized with code.
- Do not create empty placeholder documents or folders.
- Adapt the structure to the real project; the following is a reference, not a mandate.

```text
Workspace/
├── AGENTS.md
├── README.md
├── docs_for_agent/
│   ├── workflow.md
│   ├── coding_rules.md
│   ├── documentation_rules.md
│   └── progress.md
└── <project>/
    ├── README.md
    ├── AI_usage.md
    ├── docs/
    │   ├── 01_problem.md
    │   ├── 02_analysis.md
    │   ├── 03_requirements.md
    │   ├── 04_architecture.md
    │   └── scripts/
    ├── src/
    ├── visualization/
    │   └── explorer.html
    ├── data/
    │   ├── <dataset>_raw/
    │   └── <dataset>_processed/
    └── outputs/
```

`<project>` and script names are examples; use actual names when creating the project. Paths under `docs/` below are relative to the project directory; `docs_for_agent/` is relative to the workspace root.

## File Placement

- Keep small projects organized around workflow steps. Prefer a few step documents over per-script documents when the user asks for fewer files; show execution order in README.
- When numbered scripts execute at top level by user request, document their order and keep shared data/metrics functions in plain modules; update all imports and commands together during renaming.
- User-requested EDA visual assets may be generated under visualization/eda/ for HTML; keep EDA numeric reports, logs and checkpoints in outputs/.


- For model visualization, extend the existing single HTML page and read script-generated reports from outputs/. Use real original image indices for prediction examples; distinguish fixed-checkpoint evaluation from online training metrics and label confusion-matrix axes explicitly.

- At training start, record the actual loss type/options and optimizer type/parameter groups, plus model, seed, data/batch/transform and device settings once in the run log. Read optimizer settings from the actual instance, not a fixed optimizer name. Epoch logs include loss and current learning rate.

- Use readable sequential training-run names such as outputs/runs/lan_1/, lan_2/, … instead of long timestamp names. Never overwrite existing runs; preserve older directory names unless renaming is requested.

- For repeated training runs, create a unique directory under outputs/runs/; preserve older results and flush logs during execution. Keep source code only in src/, without .py snapshots in runs/. Record configuration/model architecture in train.log. CSV, plots and checkpoints are optional outputs for separately agreed steps; do not require them for a minimal training example.

- `src/`: only project Python scripts and modules (`.py`). Do not place HTML, JavaScript, browser checks or other visualization tooling here. Interpreter-generated caches are not project source files.
- `visualization/`: directly viewable HTML pages, UI assets, and scripts for visualization verification or browser checks, including `.cjs` files.
- For each visualization, maintain one directly viewable HTML file, without a second template HTML. For Live Server, keep HTML lightweight and fetch prepared data through relative HTTP paths rather than embedding a large dataset.
- Keep visual tooling minimal; remove optional .cjs helpers when not needed by the user and record verification evidence in documentation.
- `outputs/`: results produced by project scripts, such as statistics JSON, plots, metrics, logs, predictions and model checkpoints. HTML viewer pages belong in `visualization/`.
- `data/`: use short, descriptive dataset directories such as `FashionMNIST_raw/` and `FashionMNIST_processed/`; avoid extra levels such as `FashionMNIST/raw/`.
- Only create directories when needed. Example processed-data or model folders are not required in advance.
- When changing placement, update script paths, usage instructions and paired documentation together; preserve downloaded data.
- Validate the actual page in a browser when possible. Generated HTML and DOM-emulation checks alone do not establish that the user can view the page.
- State the verified way to open the page. Do not assume IDE Preview supports JavaScript or large embedded datasets; distinguish browser verification from the user's preview environment and show helpful loading instructions.

## Keeping Reusable Rules Current

- Before Git upload, preserve existing repository history/content, confirm root placement and branch, exclude downloaded datasets/cache/local secrets, and retain dataset source links plus reproduction instructions. Commit/push approval gates still apply.

- Record dataset source pages, actual download URLs and available integrity checks in project documentation; link them from README and the preparation-script document.

- After each change, update affected script documentation, usage paths, design notes, AI usage records and progress with actual actions and check results.
- When the user clarifies a reusable working preference, update the relevant `docs_for_agent/` rules in the same change. Keep important entry-point rules synchronized in `AGENTS.md`.
- Keep reusable rules independent of the current assignment. Dataset and project names are examples; task facts and results belong in project documentation and progress.
- Preserve earlier log entries as history; update current instructions so obsolete paths are not presented as commands to run.

## README.md

Include, when applicable:

1. Project description and goals.
2. Problem statement.
3. Main features.
4. System architecture and data flow.
5. Project file structure.
6. Installation and usage.
7. How to run and test.
8. Results and evaluation evidence.
9. Limitations.
10. Future improvements.

Link to detailed documents instead of duplicating them in README.

## Script Documentation

- Document every script role, but group related scripts in workflow-step documents for small projects when requested; do not require docs/scripts in that case.
- Explain purpose, inputs, outputs, main logic, dependencies, testing, and known issues.
- Update the paired document whenever the script's behavior or interface changes.
- Do not document each trivial line of code.

## AI_usage.md

This file transparently records actual collaboration with AI agents:

- Tools and main purposes.
- User contribution.
- Important AI-generated components.
- Prompt purpose or a concise description of important requests.
- User modifications.
- Validation methods and outcomes.
- AI collaboration log: task, agent actions, affected files, user contributions, verification, and open issues.

Record only actions and contributions that actually occurred. If long, archive detailed logs in linked Markdown files under `docs/`.

## docs_for_agent/progress.md

- Serves as a cross-session project handoff, not just a checklist.
- Track current phase, task states, last session, blockers, changed files, test evidence, and next steps.
- Update after significant milestones and before ending a session.
- Archive older sessions when the file approaches 150 lines.
- Distinguish implemented, validated, user-approved, and committed states.
