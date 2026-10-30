# AI Usage Step - Finalization 2026-10-30

## Completed work

- Imported DenseNet A/B/C as lan_1-3 and VGG A/B/C as lan_4-6 without overwriting source runs.
- Normalized VGG best/last download filenames; file SHA256 unchanged by rename.
- Verified all 12 checkpoint metadata and six strict model loads; existing train/val/test reports match best checkpoint hashes.
- Updated report with six-run metrics, parameter counts, recorded training cost/hardware, separate curves/test bars, aligned confusion panels and paired A/B/C errors.
- Deep EDA covers brightness/contrast/saturation/adjacent pixel variation, split distributions, train quantiles/classes/correlation and extreme examples.
- HTML includes EDA, one 60000-image PCA map, dataset browser, preprocessing/quality, TensorBoard and paired test gallery.
- Dataset map is based on RGB8x8 block means, PCA fit on train then applied to all splits. No model features; removed 4_feature_space.py and feature_space.json.
- Model-independent PCA2 retains 48.89% training variance; colors represent class labels, not clusters inferred by the algorithm.

## Validation

- Real split IDs, class counts, pixel descriptors, report confusion/score consistency and checkpoint SHA checked.
- Node DOM/canvas checks cover filters/pagination/races, image decoding, EDA tables and all 60000 map points with hover/click.
- Python source compilation and Markdown limits checked; no new training or repeated test inference during presentation updates.
- Final notebook rerun passed all 17 code cells (six runs, six test analyses, three pairs), no repeated inference; actual browser/IDE layout still requires visual review.

## State

- Dataset and derived pixel assets remain local; existing Git ignore policy retained.
- Test has already been observed for six configurations; no tuning or final-model selection from these results.
- Pending review/commit/push. Full history: [AI_Usage_Full.md](docs/AI_Usage_Full.md).

- 2026-10-30: Final publication authorized on new branch practice2-pretrained-model. Git LFS tracks all12 best/last checkpoints, preserving binary SHA; actual TensorBoard events included. Dataset/derived pixel galleries remain ignored. Added clone/LFS pull instructions; no history rewrite or training.

- 2026-10-30: Corrected branch scope: practice2-pretrained-model publishes Practice 2 only. Untrack Practice 1 on this branch while preserving all local files and other branches; root README/index updated. No model or result changes.
