# Development Workflow

## 1. Analyze Before Coding
- Define the problem and desired outcome.
- Gather available facts: existing code, dataset, interfaces, constraints, and failure evidence.
- State assumptions and acceptance criteria.
- Discuss major design choices or unresolved product decisions with the user.
- Prefer the simplest solution that meets the criteria.

## 2. Implement
- Create a project directory alongside the workspace's `AGENTS.md`, `README.md`, and `docs_for_agent/`, with only necessary subdirectories; reuse it if it already exists.
- Document architecture and responsibilities before significant changes.
- Work in small, testable increments within the approved scope.
- Implement one user-reviewable part at a time. State its scope, complete it, check it and report the result before starting another part.
- When the user requests step-by-step review, stop after each part and wait for their response; do not batch implementation of multiple parts.
- Do not paste generated code into chat unless asked; edit files in the workspace.
- With each change, synchronize affected documentation and progress. Capture reusable user preferences in the corresponding `docs_for_agent/` rules; do not leave them only in chat.

## 3. Test, Diagnose, Iterate
- Run relevant checks, tests, or a reproducible smoke test.
- On failure: inspect error, find root cause, fix, and rerun.
- Check regressions after behavior-changing edits.
- Optimize only for a demonstrated issue; do not loop indefinitely.
- If tests cannot run or acceptance criteria cannot be met, report the blocker and evidence.
- Passing tests does not prove the absence of all defects.

## 4. Review Gate
Before requesting approval, provide a short report with:
- Implemented functionality and changed files.
- Commands/checks run and their actual results.
- Important design decisions and remaining limitations.
- Any untested aspects or open questions.

**Do not commit before explicit user approval.**
If revisions are requested, implement, test, and present them for review again.
After approval, commit only the approved changes and report the commit hash.
Never push to a remote without separate explicit authorization.

## 5. Approval Boundaries
Ask before:
- Changing project scope or important architecture.
- Destructive data operations or sensitive external actions.
- Incurring costs or introducing paid services.
- Committing or pushing changes.
- Making unresolved product/business decisions.
