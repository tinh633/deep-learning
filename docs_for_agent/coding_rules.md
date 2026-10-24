# Coding & Validation Rules

## Code Quality

- Separate readable training blocks with blank lines: setup, batch forward/loss, backward/step, validation and saving. Put epoch-log string formatting in the existing logger module; do not hide the learning loop in extra abstractions.

- For this user's introductory PyTorch projects, follow the Chapter 3 PDF example style: direct library calls, explicitly declared layers, one forward operation per line and a visible epoch/batch loop. Avoid replacing those steps with generic factories or dynamic builders.
- Adapt slide examples to the actual task: keep validation separate from test, use dataset-appropriate preprocessing, and retain the user's epoch-only logging preference. A reference document is not authorization to add its optional tools or change project scope.
- Avoid helper modules that only wrap a single call; put one-use preparation/EDA logic in its numbered script. Keep helpers only for shared responsibilities and keep logs separate from the training loop.

- For basic model examples, write one operation per forward line instead of nested layer calls; show each hidden layer, activation and dropout explicitly.

- Prefer explicit layer declarations and a direct forward for small MLPs; avoid dynamic architecture builders, legacy run registries and optimizer factories when the user asks for basic readable code.

- When requested, use model classes with separate __init__/forward, central Python config and a small logging helper to keep training loops readable.
- Follow the user's requested numbered workflow scripts and direct top-level execution without main/__main__ guards. Shared modules retain importable names and have no task side effects; never import an executable script to reuse helpers.
- Consolidate experiments into one configurable pipeline; preserve old run artifacts and load their actual architectures explicitly.

- Distinguish the best validation checkpoint within a run from final model selection across experiments. Do not declare a final model or create a final-model report unless the user confirms that decision.
- Keep validation evaluation separate from basic training when requested. Evaluate saved trained weights with eval/no_grad; append evaluation logs and reserve test evaluation until the user finalizes the model.
- For training progress, print and save one metrics line per epoch; do not log each batch unless explicitly requested. Initial configuration and final status messages may remain.
- When the user asks for a minimal training example, keep only the requested training loop and essential persistent logging. Implement evaluation, plots and checkpoint saving as separately reviewed steps rather than bundling them into train.py.
- For training scripts, prefer the user's direct criterion/optimizer/epoch/batch loop style. Keep output handling concise; do not add source snapshots or separate configuration files unless requested.
- For introductory practice scripts, follow the user's short library examples: explicit dataset/transform/DataLoader statements, minimal configuration, and no CLI, wrapper functions or generated reports unless needed or requested.
- Prefer established library APIs for dataset loading, transforms and dataset splitting. Avoid hand-written equivalents; keep any necessary adapter small and document why it exists.
- For small learning projects, prefer a short, sequential pipeline and framework-provided datasets over custom classes or elaborate validation scaffolding. Keep necessary reproducibility and data checks; explain only non-obvious steps.
- Favor simple, readable, maintainable implementations over clever abstractions.
- Use descriptive names and separate responsibilities into sensible modules.
- Add comments for non-obvious logic, assumptions, and important edge cases.
- Avoid unnecessary dependencies, speculative features, and unrelated refactoring.
- Preserve existing behavior outside the approved task.
- Do not delete or overwrite important data without authorization.
- Keep only `.py` project source files in `src/`; visualization assets and visual/browser verification scripts belong in `visualization/`.
- After moving files, update all active references and run relevant checks from the new locations; preserve existing data and behavior.

## Validation
- Identify success criteria before implementing.
- Run the most relevant available tests and inspect their output.
- For fixes, reproduce the failure when practical and add regression coverage when appropriate.
- Re-run affected tests after modifications.
- Record the exact outcome, including failures, skipped checks, and environment limitations.
- Do not mark work validated solely because code was written or imported successfully.

## Agent Communication
- Work autonomously on routine implementation, debugging, and testing within scope.
- Do not stream large source files or report every small edit in chat.
- Ask the user only when a decision, approval, or unavailable input is essential.
- Provide a concise review summary when acceptance criteria are met or progress is blocked.
