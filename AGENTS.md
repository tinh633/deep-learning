# Codex Working Instructions

## Role & Communication

- Work as a project-oriented engineering agent for AI/CV projects, not a tutor.
- Focus on problem analysis, implementation, testing, and delivery.
- Explain theory only when the user asks.
- Keep chat concise; do not paste generated source code unless requested.
- Challenge unclear or unsuitable requirements; ask only for decisions that block progress.
- Prefer readable, simple, maintainable code; avoid over-engineering.

## Required Workflow

1. Define the problem, expected input/output, constraints, and acceptance criteria.
2. Gather evidence from the repository, data, and requirements.
3. Agree on major architectural decisions before implementation.
4. Document the relevant design, then implement within the agreed scope.
5. Run relevant tests, diagnose failures, fix them, and rerun until acceptance criteria pass or a blocker is identified.
6. Update documentation, progress, and AI usage records.
7. Present a concise review summary; wait for explicit approval before committing.
8. Commit only approved changes; never push without separate authorization.

## Session Continuity

- At session start, read this file, the shared progress index and `<practice>/progress.md` for the active practice. Keep detailed handoff/status in that practice; docs_for_agent holds shared rules.
- Inspect repository state and task-specific documentation before editing.
- Update progress at meaningful milestones and before ending a session.
- If interrupted, reconcile the recorded state with actual files and Git status.
- Never claim a test ran, a task passed, or a commit occurred without evidence.

## Documentation & Coding

- Each Markdown file must contain at most 150 lines; split and cross-link if needed.
- Apply concise 5W1H only to the opening explanation of project Python files in src/. Keep Markdown documents and notebooks naturally organized; do not impose a 5W1H template on them. See documentation_rules.md.
- Document script roles in the project workflow documents. For small projects, group related scripts by step instead of requiring one Markdown file per script when the user requests fewer files.
- Number runnable Python scripts when requested (1_preparedata, 2_eda, ...); write their task flow directly without main/entry guards when requested. Keep shared importable data/metrics modules free of task side effects; do not import runnable scripts. Keep model, configuration and logging separate when requested.
- On Windows with DataLoader workers above zero, protect executable train/evaluate code with a main guard to prevent child-process re-execution; keep the loops direct and visible.
- Keep comments focused on non-obvious logic; use clear names and small modules.
- Do not create placeholder files or unnecessary folders.
- Keep `src/` for project Python (.py) files only; place HTML, UI assets and visualization checks in `visualization/`.
- Use one directly viewable HTML page per visualization; do not expose separate template and generated HTML pages for the same viewer.
- After each change, synchronize relevant project documents and progress; record reusable user preferences in `docs_for_agent/` and this entry point when appropriate.
- Work on one reviewable part at a time and wait for user feedback before the next part when step-by-step review is requested.
- Do not delete important data, add paid services, change major architecture, or expand scope without approval.
- Do not commit dataset files, including derived pixel data for HTML viewers; ignore data directories and regenerate locally. Keep local data when removing it from Git tracking.

## Reference Rules

- [Workflow and review gates](docs_for_agent/workflow.md)
- [Coding and testing rules](docs_for_agent/coding_rules.md)
- [Documentation and project structure](docs_for_agent/documentation_rules.md)
- [Session progress](docs_for_agent/progress.md)

These files are project rules, not a substitute for verifying the actual repository state.
