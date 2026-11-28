# Audit log — Checklist smoke run
Hi?n tr?ng28/11: A/B/C ?? c? final; B/C th?nh vi?n kh?c train, hash/header ?? ki?m tra; [k?t qu?/gi?i h?n](Experiment_Results.md).

18/11: đổi CNN sang ConvNeXt-Base; các PASS và số đo ResNet dưới đây là lịch sử, không chuyển sang ConvNeXt.
Smoke ConvNeXt A: lan_11 PASS sau khi giải phóng bộ nhớ; B: NOT_RUN. Lỗi lan_10 giữ nguyên.

Mỗi dòng dưới đây ghi riêng một lần chạy; các bảng phía sau giải thích tiêu chí và bằng chứng chi tiết.
Chiến lược: [Training_Strategy.md](design/Training_Strategy.md).
Thay đổi code/pipeline ghi ở [CHANGELOG.md](CHANGELOG.md); việc tiếp theo ghi ở [progress.md](progress.md).

## Quy tắc ghi nhận

- Chỉ ghi cấu hình, kiểm tra đã chạy, bằng chứng và kết luận smoke; không kể quá trình sửa code.
- Trạng thái: PASS, FAIL hoặc NOT_RUN; không xem chạy không lỗi là mọi kiểm tra đều đạt.
- Mỗi lượt thực có ngày, run ID/thư mục, phiên bản source, model/A–D, seed, split và cấu hình.
- Ghi máy/GPU, precision, batch, số batch/epoch thực chạy và đường dẫn log/checkpoint nguồn.
- Khi kiểm tra thất bại, ghi FAIL và liên kết thay đổi/kiểm chứng lại trong changelog, không xóa kết quả cũ.
- “Ổn định” chỉ trong phạm vi số batch đo: có/không NaN, Inf, OOM, biến động thời gian/VRAM.
- Smoke ngắn không chứng minh hội tụ hoặc khả năng tổng quát hóa; không dùng test để smoke/tuning.
- Audit Markdown là bản tóm tắt; bằng chứng chi tiết đọc từ artifact thật, không bắt buộc thêm audit.py/JSON riêng.

## Kết quả từng lần chạy — lan_1 đến lan_14

Smoke lan_1–10 dùng batch 2, 20 ảnh train/4 val; lan_11 dùng batch 16, 32 train/16 val. Không mở test.
PASS chỉ áp dụng các check được lưu ở source của run đó; NOT_RUN là chưa hoàn tất kiểm tra, kể cả đã chạy một batch.

| Run / cấu hình | Cụ thể đã PASS hoặc hoàn tất | Lỗi, phần chưa kiểm tra và giới hạn |
|---|---|---|
| [lan_1](../outputs/runs/lan_1/summary.json) — ResNet-50 freeze | Đủ 12 check: đúng nguồn/split, chỉ mở head, optimizer đúng tập tham số; logits B×100 và loss hữu hạn; gradient head hữu hạn, backbone không gradient; head cập nhật, backbone/buffers giữ nguyên; validation eval/không gradient/buffers không đổi; save/load logits và resume một bước khớp; resources/stability PASS qua 10 batch train. | Source cũ; PASS bộ nhớ chỉ tại các điểm guard đã cài, chưa chứng minh train dài hoặc chất lượng phân loại. |
| [lan_2](../outputs/runs/lan_2/summary.json) — ResNet-50 fine-tune | Đủ 12 check: đúng nguồn/split, mở toàn model, optimizer đúng tham số; logits B×100, loss/gradient hữu hạn; head và tensor backbone được kiểm tra có cập nhật; validation eval/không gradient/buffers không đổi; save/load và resume một bước khớp; resources/stability PASS qua 10 batch train. | Source cũ; chưa chứng minh bộ nhớ ổn định toàn workflow hoặc hội tụ. |
| [lan_3](../outputs/runs/lan_3/summary.json) — ViT-B/16 freeze | Không có check PASS được lưu. | INTERRUPTED trong lúc nạp pretrained; cả 12 check NOT_RUN. |
| [lan_4](../outputs/runs/lan_4/summary.json) — ResNet-50 freeze | Đủ 12 check: nguồn/split đúng; chỉ head được học, optimizer đúng; logits B×100, loss/gradient hữu hạn; head đổi, backbone/buffers không đổi; validation đúng eval/không gradient; save/load và resume khớp; resources/stability đạt qua 10 batch train. | Kiểm chứng lại nhánh freeze trên source cập nhật; vẫn là smoke, guard chưa bao phủ các bước cuối workflow. |
| [lan_5](../outputs/runs/lan_5/summary.json) — ViT-B/16 freeze | Đủ 12 check: nguồn/split đúng; chỉ head được học, optimizer đúng; logits B×100, loss/gradient hữu hạn; head đổi, backbone/buffers không đổi; validation đúng eval/không gradient; save/load và resume khớp; resources/stability đạt qua 10 batch train. | Hoàn tất smoke ViT freeze sau lan_3 bị ngắt; chưa có kết quả train đầy đủ. |
| [lan_6](../outputs/runs/lan_6/summary.json) — ViT-B/16 fine-tune | 3 check PASS: nguồn/split/config đúng; toàn model được mở để fine-tune; optimizer chứa đúng tập tham số. | Guard dừng vì commit 2676,77 MiB <3072. Summary cũ ghi gradients=FAIL, resources=NOT_RUN; traceback cho thấy nguyên nhân là thiếu bộ nhớ, không phải bằng chứng gradient sai. Các check khác NOT_RUN. |
| [lan_7](../outputs/runs/lan_7/summary.json) — ViT-B/16 fine-tune | 3 check PASS: nguồn/split/config đúng; mở toàn model; optimizer đúng tham số. | Sau 1 update, resources FAIL vì commit 2553,90 MiB <3072; shape/loss/gradient toàn lượt, state update, validation, checkpoint, resume, stability đều NOT_RUN. |
| [lan_8](../outputs/runs/lan_8/summary.json) — ResNet-50 fine-tune | Đủ 12 check đã cài: nguồn/split đúng, mở toàn model, optimizer đúng; logits B×100, loss/gradient hữu hạn; head/backbone được kiểm tra có cập nhật; validation đúng; save/load và resume khớp; resources/stability PASS qua 10 batch train. | Commit cuối 2952,80 MiB <3072; guard cũ chưa bao phủ cuối save/load/replay. PASS trong summary không xác nhận bộ nhớ đạt ngưỡng toàn workflow. |
| [lan_9](../outputs/runs/lan_9/summary.json) — ConvNeXt-Base freeze hiện tại | Không có check PASS; 0 batch. | source_split FAIL vì tải pretrained lỗi socket WinError 10013; 11 check còn lại NOT_RUN. Không phải lan_9 ResNet lịch sử đã xóa. |
| [lan_10](../outputs/runs/lan_10/summary.json) — ConvNeXt-Base freeze | 3 check PASS: nguồn/split/config đúng; chỉ Linear(1024,100) được học, 102.500 tham số; optimizer đúng tập này. | Hoàn tất 1 batch rồi resources FAIL vì commit 2942,83 MiB <3072; các check shape/loss/gradient toàn lượt, cập nhật, validation, checkpoint, resume và stability NOT_RUN. |
| [lan_11](../outputs/runs/lan_11/summary.json) — ConvNeXt-Base freeze | Đủ 12 check: nguồn/split đúng; chỉ 102.500 tham số head được học, optimizer đúng; logits B×100, loss/gradient hữu hạn; head đổi, backbone/LayerNorm/buffers không đổi; validation đúng; save/load/resume khớp; guard qua cả cuối workflow, stability đạt sau 2 batch train và 1 batch val. | Commit cuối 3716,92 MiB; đủ điều kiện kỹ thuật trước pilot, chưa chứng minh hội tụ hoặc ổn định 20 epoch. |
| [lan_12](../outputs/runs/lan_12/summary.json) — ConvNeXt freeze, pilot LR=1e-3 | COMPLETED đủ 5 epoch trên 45.000 train/5.000 val; best val 80,64% ở epoch 5. Kiểm chứng best/last, history/config và tensor freeze đạt theo A_Results. | Không có bảng 12 check smoke trong summary; đây là pilot chọn LR, không phải test. |
| [lan_13](../outputs/runs/lan_13/summary.json) — ConvNeXt freeze, pilot LR=3e-4 | COMPLETED đủ 5 epoch; best val 80,78% ở epoch 5. Kiểm chứng best/last, history/config và tensor freeze đạt; được chọn vì hơn lan_12 0,14 điểm phần trăm. | Chỉ một seed, chênh lệch nhỏ; chưa phải đánh giá test. |
| [lan_14](../outputs/runs/lan_14/summary.json) — ConvNeXt freeze, final LR=3e-4 | COMPLETED đủ 20 epoch, khởi tạo mới; best val 81,46% ở epoch 9, last val 80,94%. Kiểm chứng checkpoint/history 1–20, frozen tensors, optimizer/RNG và resume đạt. | Guard ngắt giữa epoch 20; resume từ epoch 19 và chạy lại epoch 20. Chưa test; xem [A_Results](A_Results.md) và interruptions.json để đọc bằng chứng. |

## Mẫu ghi một lượt thực tế

1. Nhận diện: run ID, model/chiến lược, source revision/hash, weights và split hash.
2. Điều kiện: máy, GPU, batch/precision, seed và phạm vi thực thi.
3. Kết quả: từng check, trạng thái, số đo/tolerance và bằng chứng; ghi rõ check chưa chạy.
4. Ổn định: số batch quan sát, NaN/Inf/OOM, thống kê thời gian/VRAM nếu đã đo.
5. Kết luận: có đủ điều kiện tuning chưa, còn giới hạn nào; liên kết recheck nếu cần.

## 14/11/2026 — Kết quả smoke thực và khôi phục phiên

Máy: RTX3050 Laptop 4 GB, RAM 16 GB; torch 2.11+cu126, torchvision 0.26, AMP float16.
Batch 2, workers 0, seed 42; cùng 20 train/4 val từ split hiện có, không mở test.
LR smoke freeze=1e-3, fine-tune=1e-4; AdamW WD=0, không scheduler. Chưa tuning các giá trị này.
Mỗi lượt hoàn tất gồm 10 batch train, 2 batch val và hai nhánh so sánh một bước resume.
Mini-epoch không đại diện một epoch đầy đủ CIFAR-100; xem config từng run để đối chiếu source hashes.

| Cấu hình | Bằng chứng | Kết quả | Batch train hoàn tất | Peak train / workflow MiB |
|---|---|---|---:|---:|
| A ResNet freeze | [lan_4](../outputs/runs/lan_4/summary.json) | PASS | 10 | 121,85 / 139,54 |
| B ResNet fine-tune | [lan_8](../outputs/runs/lan_8/summary.json) | PASS | 10 | 460,17 / 460,17 |
| C ViT freeze | [lan_5](../outputs/runs/lan_5/summary.json) | PASS | 10 | 365,39 / 367,39 |
| D ViT fine-tune | [lan_7](../outputs/runs/lan_7/summary.json) | FAIL — memory guard | 1 | Chưa ghi đủ |

| Kiểm tra đã lưu | A / lan_4 | B / lan_8 | C / lan_5 | D / lan_7 |
|---|---|---|---|---|
| Nguồn/split/config: đúng weights, thứ tự lớp và split/config đã ghi | PASS | PASS | PASS | PASS |
| Trainable: freeze chỉ mở head; fine-tune mở toàn model | PASS | PASS | PASS | PASS |
| Optimizer: chứa đúng tập tham số được mở gradient | PASS | PASS | PASS | PASS |
| Shape/loss: logits B×100, logits và loss hữu hạn qua đủ 10 batch | PASS | PASS | PASS | NOT_RUN |
| Gradient: tham số được học có gradient hữu hạn; tham số freeze không có gradient | PASS | PASS | PASS | NOT_RUN |
| Cập nhật: head thay đổi; freeze giữ backbone/buffers; fine-tune cập nhật tensor backbone được kiểm tra | PASS | PASS | PASS | NOT_RUN |
| Validation: mọi module ở eval, không gradient, buffers không đổi và loss hữu hạn | PASS | PASS | PASS | NOT_RUN |
| Save/load: logits sau nạp khớp trước lưu (atol=1e-6, rtol=1e-5) | PASS | PASS | PASS | NOT_RUN |
| Resume: khôi phục epoch/optimizer/RNG; loss và state sau một bước khớp nhánh đối chứng | PASS | PASS | PASS | NOT_RUN |
| Bộ nhớ: qua guard ở các điểm đã kiểm tra; source cũ chưa bao phủ cuối workflow | PASS | PASS | PASS | FAIL |
| Ổn định: đủ 10 batch train, 2 batch val và replay, không NaN/Inf/OOM | PASS | PASS | PASS | NOT_RUN |

A/B/C: checkpoint reload và state sau replay khớp rtol=1e-5, atol=1e-6; chênh loss <1e-5.
Không NaN/Inf/OOM ở phạm vi hoàn tất của A/B/C; không kết luận về hội tụ hoặc train dài.
D dừng sau một update vì Windows commit còn 2553,90 MiB <3072 MiB, RAM trống 3934,05 MiB.
Guard vẫn giữ ngưỡng 2 GiB RAM/3 GiB commit sau thu hồi cache; chưa đủ điều kiện tiếp tục D.
NOT_RUN ở D nghĩa kiểm tra đầy đủ chưa hoàn tất/lưu PASS, không suy đoán từ một bước đã chạy.

Lịch sử giữ nguyên: [lan_1 A](../outputs/runs/lan_1/summary.json), [lan_2 B](../outputs/runs/lan_2/summary.json)
đã PASS ở source cũ; [lan_3 C](../outputs/runs/lan_3/summary.json) bị ngắt khi loading, không có kết quả.
[lan_6 D](../outputs/runs/lan_6/run.log) dừng do memory guard, không tính là một lượt đạt.
Review sau chạy: B/lan_8 kết thúc còn 2952,80 MiB Windows commit, thấp hơn ngưỡng 3072 MiB.
PASS trong summary chỉ phản ánh các check đã cài; guard chưa bao phủ cuối save/load/replay.
Thời gian batch có warm-up CUDA và kiểm tra gradient, không dùng để xếp hạng tốc độ model.
Kết luận: A/B/C hoàn tất smoke hiện có; D chưa đạt. Chưa xác nhận bộ nhớ ổn định toàn workflow.
Trước tuning/final cần bổ sung guard cuối workflow và đủ RAM/commit để recheck, nhất là D.

## 16/11/2026 — Kiểm tra hồi quy CPU tổng hợp

Nguồn: [source_pipeline_check.json](../outputs/validation/source_pipeline_check.json),
[test_pipeline.py](../tests/test_pipeline.py). Hai bài integration hoàn tất, exit code 0.
Chỉ dùng model nhỏ/dữ liệu tổng hợp trên CPU, memory guard được mock; không nạp pretrained/CIFAR/test thật.

| Kiểm tra | Kết quả trong phạm vi tổng hợp |
|---|---|
| Smoke A–D: gradient/freeze/save/load/replay | PASS |
| Guard sau save, load, replay và cuối workflow được gọi | PASS |
| Giả lập thiếu commit sau replay phải dừng và ghi resources FAIL | PASS |
| Ngắt/resume epoch 1→2, state khớp nhánh liên tục | PASS |
| Từ chối resume khi LR thay đổi | PASS |
| Final khởi tạo độc lập, nhận config từ pilot đã completed | PASS |
| Evaluate/predictions nhất quán, xử lý AUC thiếu lớp | PASS |
| RUN_TEST=False ngăn truy cập loader test | PASS |
| Demo Top-5 và JSON đầu ra | PASS |

Không chuyển PASS tổng hợp thành PASS GPU hoặc accuracy thật; D/GPU và train dài vẫn chưa được xác nhận.

## 16/11/2026 — Resource probe A, ngoài ma trận smoke

Lan_9 ResNet lịch sử đã xóa ngày 18/11 theo yêu cầu; bằng chứng cũ ở commit 6bf7dcc, không liên kết run mới.
Đây là thử năng lực chạy cũ, không phải chọn LR hoặc kết quả ConvNeXt.
ResNet freeze, 45.000 train/5.000 val, 1 epoch, batch 16, AMP train/FP32 val, workers 0.
Train acc 58,73%, val acc 64,16%, thời gian train+val 173,57 giây; 2813 update, AMP skip 0.
Đã kiểm chứng best/last cùng epoch/weights; backbone/BN khớp pretrained, counter BN legacy vẫn 0.
Quan sát 18 mẫu: GPU dùng 342–494 MiB, commit thấp nhất mẫu 4736 MiB; commit cuối 4672 MiB.
Số GPU là snapshot toàn GPU, không khẳng định peak allocator; không suy ra VRAM của fine-tune ViT.
Không có guard stop/NaN/Inf/OOM trong lượt hoàn tất. Đạt khả năng chạy A ở cấu hình này.
Giữ duplicate/split cũ chỉ cho probe, không mở test, không dùng điểm val để chọn LR final.

## 18/11/2026 — ConvNeXt A smoke

- Lan_9 ResNet đã xóa theo yêu cầu; lan_9 mới FAIL khi tải weights: sandbox socket WinError 10013, 0 batch.
- Tải riêng weights chính thức thành công, checksum prefix 6075fbad; không đổi model hoặc nguồn weights.
- [Lan_10 summary](../outputs/runs/lan_10/summary.json): batch 2, AMP, workers 0, 20 train/4 val, GPU RTX3050.
- Source/split PASS: đúng weights và split/config; trainable PASS: chỉ Linear(1024,100), 102.500 tham số; optimizer PASS: chứa đúng tập tham số được học. Hoàn tất 1 batch rồi guard dừng.
- Windows commit còn 2942,83 MiB (<3072); RAM 3319,13 MiB. Resources FAIL, chưa val/save/replay/stability.
- Tiến trình đã thoát; GPU trở về 0 MiB. Không giảm guard hoặc tự chạy lại liên tục, chưa pilot/final.
- Sau khi được phép đóng Zalo/Discord, commit trước khởi chạy mới tăng lên khoảng 6,3 GiB.
- [Lan_11 summary](../outputs/runs/lan_11/summary.json): batch 16, 32 train/16 val; 2 batch, đủ 12 checks PASS.
- Peak allocated train/workflow 511,22/607,22 MiB; guard cuối RAM 4028,70 MiB, commit 3716,92 MiB.
- Save/load/replay và backbone/LayerNorm bất biến đạt; không suy ra độ hội tụ hoặc ổn định 20 epoch.

Ý nghĩa 12 check PASS của lan_11, đối chiếu các khóa trong summary.json:

| Check | Nội dung đã đạt trong smoke lan_11 |
|---|---|
| source_split / trainable / optimizer | Đúng weights, lớp, split/config; chỉ học 102.500 tham số Linear(1024,100); optimizer chứa đúng tập này. |
| shape / finite_loss | Logits B×100; logits và loss train hữu hạn qua cả 2 batch, không NaN/Inf. |
| gradients / state_update | Tham số được học có gradient hữu hạn, head thay đổi; tham số freeze không gradient, fingerprint backbone/LayerNorm và buffers không đổi. |
| validation | Loss hữu hạn trên 16 ảnh; mọi module ở eval, không gradient, buffers không đổi. |
| checkpoint / resume_replay | Nạp lại tái tạo logits; khôi phục epoch 1, optimizer/scaler/RNG; một bước replay có chênh loss <1e-5, state khớp atol=1e-6, rtol=1e-5. |
| resources / stability | Guard qua cả save/load/replay và cuối workflow; đủ 2 batch train, 1 batch val và replay, không NaN/Inf/OOM. Chưa chứng minh ổn định train dài. |
