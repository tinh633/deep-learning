# AI assistance — Step-by-step development record

This record is a disclosure of AI-assisted **implementation work**, not a claim of training results.

| Stage | Assistance | Required student verification |
|---|---|---|
| Requirements | Mapped assignment clauses to scripts and artifacts | Compare to instructor handout |
| Dataset | Suggested full Hub identifier `stanfordnlp/imdb`, partition checks and SHA256 provenance | Run prepare script and verify label counts |
| Exercise 1 | Implemented tokenizer/pipeline and evidence JSON | Run input and inspect actual predicted label |
| Exercise 2 | Built Trainer code with split isolation and logging | Inspect actual loss, F1 and saved weights |
| Evaluation | Added confusion, per-class scores and explicit test unlock | Confirm independent final test was not used for tuning |
| Visualization/report | Added figures and notebook reading disk artifacts | Interpret results without fabricating performance |
| QA | Added synthetic unit tests and project documentation | Run pytest, run Python training end-to-end on target GPU |

**Revision trigger:** User identified the original project as incomplete, especially README/docs and reporting. The redesigned version addresses those gaps. External sources must be checked before claiming a result or version-specific API behavior.
