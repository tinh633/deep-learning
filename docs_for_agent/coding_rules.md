# Coding & Validation Rules

## Code Quality
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
