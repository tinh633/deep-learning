# Practice 1 — FashionMNIST

Current scope: data preparation and Live Server visualization; no model training.

## Open the page

1. Open workspace `D:\newtask\DL` in VS Code, not just the visualization folder.
2. Open `practice_1_fashionmnist/visualization/explorer.html`.
3. Right-click → **Open with Live Server**.
4. Wait until the loading message becomes `70.000 ảnh phù hợp`.

With default port 5500 and workspace root, the URL is `http://127.0.0.1:5500/practice_1_fashionmnist/visualization/explorer.html`. Use the URL opened by Live Server if its port/root differs. Refresh with Ctrl+F5 if already running.

The only HTML is approximately 18 KB. It loads EDA JSON and four extracted IDX files through relative HTTP paths. Serve the workspace or project root and keep folders together. file:// and editor source Preview are not supported. JavaScript is required.

- Filter all labels or one label, and train/test/all. Exact counts update immediately.
- Scroll or jump to any matching image; click to inspect details.
- EDA has an independent split selector.
- Approximately 55 MB data loads once per page load; filtering stays local and only nearby rows are rendered.
- Errors are displayed in the page with instructions.

## Prepare data

From workspace root: `python practice_1_fashionmnist/src/prepare_data.py`.

Requires Python and NumPy (`requirements.txt`). The script verifies originals, extracts IDX and computes EDA; it does not rewrite HTML.

## Files and verification

- `src/prepare_data.py`: Python preparation.
- `visualization/explorer.html`: one page containing UI, CSS and browser JavaScript.
- `data/FashionMNIST_raw/`: compressed originals and extracted files.
- `outputs/eda.json`: computed statistics.
- `outputs/browser_preview.png`: screenshot from actual Edge HTTP smoke test.
- No .cjs or Node dependency is needed to use the project.

Four MD5 checks passed. Dataset: 60,000 train and 10,000 test, ten balanced classes. Actual Edge over local HTTP passed image/EDA rendering, label/test filtering and final-image details without JavaScript exceptions. This verifies the static HTTP behavior used by Live Server; the extension itself was not automated.

Source: [Zalando Research FashionMNIST](https://github.com/zalandoresearch/fashion-mnist).

Link tải trực tiếp và checksum: [Nguồn dữ liệu](docs/data_source.md).

See [design](docs/design.md), [preparation](docs/scripts/prepare_data.md), [viewer](docs/scripts/explorer.md).
