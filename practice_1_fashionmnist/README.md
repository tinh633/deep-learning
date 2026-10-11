# Practice 1 — FashionMNIST

Xuất biểu đồ đánh giá: `python practice_1_fashionmnist/src/visualize_metrics.py`. Xem [visualize_metrics.md](docs/scripts/visualize_metrics.md); ảnh lưu trong outputs/runs/lan_10/plots/.

Review run 10: chạy `python practice_1_fashionmnist/src/model_report.py`, rồi mở explorer.html bằng Live Server và cuộn đến **Đánh giá model**. Có confusion matrix và ảnh sai nhãn; xem [model_report.md](docs/scripts/model_report.md).

Chưa chốt mô hình cuối. Mỗi run lưu best.pt theo validation loss thấp nhất trong quá trình train; model.pt là trọng số epoch cuối của run đó.

Current scope: data preparation, MLP training, validation/test evaluation and visualization. Selected checkpoint: lan_10/best.pt; see the result report above.

Huấn luyện MLP: `python practice_1_fashionmnist/src/train.py`. Validation chạy mỗi epoch; best.pt chọn theo validation loss, model.pt lưu epoch cuối, một dòng log mỗi epoch. Run cũ giữ nguyên. Xem [train.md](docs/scripts/train.md).

Đánh giá validation sau train: `python practice_1_fashionmnist/src/evaluate.py`. Mặc định best.pt của run mới nhất; có thể chỉnh run_name/checkpoint. Kết quả append vào evaluate.log. Xem [evaluate.md](docs/scripts/evaluate.md).

So sánh validation/test theo yêu cầu: `python practice_1_fashionmnist/src/test.py`. Chỉnh run_name/checkpoint trong script; kết quả append vào test.log. Xem [test.md](docs/scripts/test.md).

Giới thiệu bài toán: [01_problem.md](docs/01_problem.md).

MLP mẫu: [03_model.md](docs/03_model.md). Chạy kiểm tra forward bằng `python practice_1_fashionmnist/src/model.py`; chỉnh hidden_size trong model.py. Chưa huấn luyện.

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

Chuẩn bị train/validation/test cho MLP: `python practice_1_fashionmnist/src/data_pipeline.py`. Seed 42, validation 10%, batch size 64 đặt trực tiếp trong code; script chỉ in số mẫu và kích thước batch, không ghi báo cáo. Xem [thiết kế dữ liệu](docs/02_data_preparation.md) và [script](docs/scripts/data_pipeline.md). Chưa huấn luyện mô hình.

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
