# AI usage

## 2026-11-02

- Implemented pretrained-only benchmark for six torchvision ImageNet-1K models.
- Earlier ImageNetV2 smoke passed6/6; no full benchmark was run.
- Revised data design to ten predefined ImageNet-1K validation classes, fifty each.
- Removed separate audit/presentation documents and audit code; retained provenance in reports.
- Added authenticated validation streaming preparation, strict class mapping and subset checks.
- ImageNet-1K acquisition awaits user-local Hugging Face login and dataset access consent.
- No training, no fabricated benchmark results, no commit/push.

- Verification: Python syntax, balanced500-image manifest, missing-sample/wrong-split rejection, notebook empty state and documentation checks passed. No new dataset inference.

- Confirmed local browser-device authentication and real image access. Installed datasets5.1.0/huggingface_hub2.4.0; streaming the500-image subset from pinned source revision49e2ee26f3810fb5a7536bbf732a7b07389a47b5. No tokens recorded.

- Download completed500 ImageNet1K validation images; verified10 classes ×50, unique IDs and all image SHA. Smoke_2 passed6/6 on RTX3050; launched full lan_1 benchmark, pending final verification.

- Final: acquired500 ImageNet1K validation images, smoke_2 passed6/6 and full lan_1 passed6/6 on CUDA. Recomputed prediction accuracies; checked shared manifest/source/weight hashes,300 timing samples and GPU memory fields.
- Executed all4 notebook code cells in .venv kernel and saved numeric/metric plots. Updated README with measured table and progress with handoff. No training, image commit or Git publication.

- Added explicit Top1 correct/wrong and Top5 correct counts to reports/CSV and simplified notebook main table. Counts derived from saved predictions; no inference or timing rerun. Explained common ImageNet1K pretraining dataset versus differing recipes and identical evaluation images.

- README: added benchmark motivation after architecture presentation, practical evaluation questions and accuracy/latency/throughput/VRAM trade-offs. Documentation only; no benchmark rerun.

- 2026-11-04: Publication authorized: test_model_present source/notebook/results to Thai branch; dataset and weights cache excluded.
