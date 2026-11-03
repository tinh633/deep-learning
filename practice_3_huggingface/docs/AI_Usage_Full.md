# AI assistance — Engineering audit

## Purpose and provenance

This project was authored with AI assistance and reviewed iteratively after direct feedback that the first package was not sufficiently professional. The AI acted as a code/documentation drafting assistant. The student owns the final code execution, verification, and interpretation.

## Requests and corrections

1. Deliver two Hugging Face exercises using Practice 1/2 style: pretrained sentiment inference and fine-tuned binary IMDb text classification.
2. First iteration produced basic Python scripts and minimal Markdown; direct user feedback documented missing depth.
3. User encountered `HfUriError` when using dataset ID `imdb`; corrected the dataset reference to `stanfordnlp/imdb`.
4. Revised architecture added clear run namespaces, separate prepared splits, duplicate checks, SHA256 provenance, callbacks for actual trainer log events, checkpoints, model and tokenizer saved locally, confusion/per-class metrics and final test guard.
5. Revised reporting uses notebook and Markdown derived only from generated artifacts, with explicit missing-data messages.

## Verification strategy

- Static checks: Python compile/AST validation; JSON notebook validation; synthetic unit tests.
- Functional checks still required by student: authenticated/unauthenticated Hub access, tokenization, actual Trainer fine-tuning, validation, final test, plotting and runtime compatibility.
- No hardcoded experimental performance numbers except user-reported Exercise 1 label/score from their previous terminal log.

## Known technical decisions

- `datasets` full Hub ID for compatibility and provenance.
- `sklearn` for local binary metrics so the experiment does not depend on downloading metric scripts; `evaluate` remains an installed course dependency.
- `RUN_TEST` flag plus `--confirm-test` blocks accidental final-test evaluation; not a security system.
- Best checkpoint by validation F1; if using 1 epoch, only one selection point.
- Tokenization max 128 and dynamic padding designed for modest hardware.

## Residual limitations

- Specific model/dataset Hub commit revisions not pinned; for high-stakes reproducibility, pin revisions and record exact dependency locks.
- Full end-to-end train was not executed in the artifact-authoring environment; student must validate locally.
- One seed and small sample counts are insufficient to claim robust generalization.
- Do not report reconstructed losses, confusion metrics or model checkpoints when they were not measured.
