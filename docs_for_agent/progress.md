# Project Progress

> Record verified project facts. Do not mark tasks complete without evidence.

## Current State

- Last updated: 2026-10-11
- Current phase: User approved commit; push remains unauthorized
- Current task: Commit prepared workspace rules and FashionMNIST explorer on branch Thái
- Git state / last commit: Local branch Thái based on origin/main at 47959f4; no new commit or push. Remote branch Thái not created yet.

## Task Checklist

- [ ] Define project problem and acceptance criteria
- [ ] Analyze available data and code
- [ ] Agree on architecture
- [ ] Implement scoped tasks
- [ ] Run and verify tests
- [ ] Present results for user review
- [ ] Obtain explicit approval before commit

## Status Definitions

- Pending: Not started.
- In progress: Work begun, not yet validated.
- Validated: Acceptance checks passed with evidence.
- Approved: User explicitly accepted the reviewed result.
- Committed: Approved changes were committed; record hash.

## Last Session Handoff

- Session date: 2026-10-11
- Completed and validated: Downloaded and checksum-verified dataset; generated HTML/EDA; verified gallery logic and torchvision cache compatibility.
- In progress: User review through Open with Live Server. One 18 KB HTML loads original IDX and EDA JSON over HTTP. MLP work deferred.
- Blockers / errors: No failure in actual Edge HTTP smoke test. Live Server extension itself not automated; user confirmation pending.
- Attempts and outcomes: Network download required escalation and succeeded. Added extracted IDX files after initial torchvision read failed; retry passed.
- Changed files: AGENTS.md, docs_for_agent/ rules and progress; moved two .cjs files into project visualization/; updated project README, design, script docs and AI_usage.md. No dataset or HTML-content changes in this revision.
- Tests run and results: Data preparation passed four MD5 checks; actual Edge over local HTTP passed gallery/EDA, nonblank canvas, class/test filtering and final-image detail without JS exceptions. Temporary checks removed after validation.
- Decisions / approvals: User chose DL contents at repository root and branch Thái; explicitly authorized commit. Push has not been authorized. origin is https://github.com/tinh633/deep-learning.git.

## Next Session

1. Inspect repository state and read AGENTS.md.
2. Review Git upload scope: rules, docs, project source, HTML and EDA JSON; downloaded data/cache/secrets excluded by .gitignore. Root README preserves repo title and links project/rules. Commit and push require explicit approval.
2. Open visualization/explorer.html with Live Server from workspace/project root. Actual Edge HTTP test passed; optional .cjs and temporary tools removed. Confirm user's viewing result before further work.
3. Model design and training remain outside the current revision until the user resumes them.

## Archive

Move older session notes to linked files under `docs_for_agent/progress_archive/` at the workspace root when needed.
