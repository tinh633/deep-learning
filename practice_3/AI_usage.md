# AI assistance disclosure — Practice 3

## Scope and role

AI assistance was used to draft the initial project structure, Python scripts, explanatory Markdown and review/checklists. During user review, substantial quality gaps were identified (thin documentation, limited artifacts and plotting, ambiguous dataset identifier). A revised package expanded documentation, added auditable run metadata, tests, dataset integrity checks, Trainer logging, held-out evaluation safeguards and evidence-based reporting.

## Verification and ownership

- **Generated code is not a validated experimental result.** Running training and interpreting actual numbers remain the student's responsibility.
- Source references should be read directly; API compatibility depends on installed library versions.
- No fabricated loss, accuracy, F1 or checkpoint is included in this package.
- AI-written sections must be reviewed against actual assignment requirements before submission.
- Training/test outcomes must be cited to local artifacts, with run, split and configuration.

## Limitations

The code and documentation were produced in a tool environment that may not have the user's CUDA/Python 3.14 Hugging Face dependencies or model cache. Syntax and dependency-light tests can be checked here; end-to-end model downloads/fine-tuning must be verified on the student's environment. The user's observed Exercise 1 success is evidence for that run, not proof of later experiments.

## Reference integrity

Primary sources: [Transformers Training](https://huggingface.co/docs/transformers/en/training), [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [IMDb data card](https://huggingface.co/datasets/stanfordnlp/imdb), [Hugging Face sentiment tutorial](https://huggingface.co/blog/sentiment-analysis-python). All additional experimental claims require actual artifacts.
