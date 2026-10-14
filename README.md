# deep-learning

Workspace for deep-learning practice projects and reusable agent working rules.

## Projects

- [Practice 1 — FashionMNIST](practice_1_fashionmnist/README.md): data preparation, Live Server explorer, MLP training/evaluation and result report (delivered model: lan_10/best.pt, test accuracy 89.41%).

Downloaded datasets are excluded from Git. Follow the project README to recreate them; source links and checksums are retained in project documentation.

## Codex Rules Starter Pack

This is a reusable set of agent instructions, not a completed AI/CV project.

## Contents
- `AGENTS.md`: short entry point and non-negotiable rules.
- `docs_for_agent/workflow.md`: analysis, implementation, testing, review, and commit gates.
- `docs_for_agent/coding_rules.md`: code quality and validation standards.
- `docs_for_agent/documentation_rules.md`: required project documentation and AI usage tracking.
- `docs_for_agent/progress.md`: cross-session progress template.

## Usage
Place `AGENTS.md`, this `README.md`, and `docs_for_agent/` at the root of the Codex workspace. Create the actual project in a separate sibling directory. Ask Codex to read `AGENTS.md` and `docs_for_agent/progress.md`, inspect the real repository, and adapt the project documentation to the actual task. Do not overwrite an existing progress file without review.

Create the project's actual `README.md`, `AI_usage.md`, and script-specific documentation only when the project facts are known. The `README.md` in this starter pack describes the starter pack itself.
