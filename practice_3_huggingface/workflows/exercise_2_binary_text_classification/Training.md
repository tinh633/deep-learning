# Chạy và kiểm tra quá trình huấn luyện

## Dữ liệu và EDA

`1_preparedata.py` tạo snapshot mặc định tại `outputs/data/imdb_baseline/`. Mỗi split là một Dataset đã token hóa, kèm `split_info.json`. Dùng lại snapshot cho các thí nghiệm chung dữ liệu. Để đổi kích thước hoặc max length, tạo snapshot khác bằng `--data-dir` và truyền cùng đường dẫn cho EDA, train, evaluate. Ví dụ chạy nhanh để kiểm tra quy trình:

```bash
python src/1_preparedata.py --data-dir outputs/data/smoke --train-samples 64 --validation-samples 16 --test-samples 16
python src/2_eda.py --data-dir outputs/data/smoke
python src/3_train.py --data-dir outputs/data/smoke --epochs 1
```

Lượt smoke dùng để kiểm tra chương trình. Khi báo cáo baseline, dùng kích thước mặc định trong [Experiment_Plan.md](Experiment_Plan.md).

## Huấn luyện

`Trainer` gọi collator cho từng batch, forward để tạo logits và loss, backward để tính gradient rồi cập nhật trọng số bằng AdamW. Linear scheduler giảm learning rate theo optimizer step. Mỗi epoch có một lần ghi loss, đánh giá validation và lưu checkpoint. `load_best_model_at_end=True` nạp checkpoint có validation F1 cao nhất sau khi hoàn thành số epoch đã yêu cầu.

| Artifact trong `outputs/runs/lan_n/` | Ý nghĩa |
| --- | --- |
| `config.json`, `split_info.json` | Đối số Trainer, phiên bản thư viện, thiết bị, snapshot, checkpoint được chọn |
| `train.log`, `history.json` | Log từng sự kiện và lịch sử để vẽ biểu đồ |
| `checkpoints/`, `tensorboard/` | Checkpoint epoch và file sự kiện để theo dõi |
| `model/`, `TRAINING_COMPLETE` | Mô hình và tokenizer đã chọn, cùng marker hoàn thành |
| `validation_metrics.json` | Validation của checkpoint đã chọn |
| `validation_evaluation.json`, `test_evaluation.json` | Đánh giá theo từng split, gồm metrics, ma trận và ví dụ lỗi |

Mở TensorBoard bằng `tensorboard --logdir outputs/runs/lan_1/tensorboard`. Thay tên run theo kết quả thực tế.

## Đánh giá và demo

```bash
python src/4_evaluate.py --run lan_1 --split validation
python src/5_report.py --run lan_1
python src/6_infer_finetuned.py --run lan_1 --sentence "The performances kept me engaged until the end."
```

Sau khi chọn run bằng validation, tạo bằng chứng test và vẽ biểu đồ test.

```bash
python src/4_evaluate.py --run lan_1 --split test
python src/5_report.py --run lan_1 --split test
```

Mở notebook ở thư mục workflow và đặt biến `RUN` theo tên đã chọn. Đối với dữ liệu khác mặc định, thêm `--data-dir` vào lệnh evaluate.

## Xử lý các trường hợp thường gặp

- **Model báo head mới được khởi tạo.** Checkpoint cơ sở có encoder pretrained. Head hai lớp của bài IMDb cần được học trong lượt train này.
- **Thiếu RAM GPU.** Chạy lại với `--batch-size 4` hoặc `--cpu`, và ghi đúng batch size của run vào báo cáo.
- **Run bị dừng giữa chừng.** Kiểm tra `train.log` và `history_partial.json`. Các lệnh đánh giá yêu cầu run có marker hoàn thành. Lệnh train mới tự cấp tên run khác.
- **Lỗi tải dataset hoặc metric.** Kiểm tra kết nối tới Hugging Face Hub và repository `stanfordnlp/imdb`. Dùng phiên bản trong `requirements.txt` nếu môi trường đang có Transformers khác major.
- **Lỗi dữ liệu không khớp.** Chỉ định snapshot ban đầu với `--data-dir`. Mô hình và kết quả cần cùng snapshot để so sánh.
