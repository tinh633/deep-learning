# 4_demo.py — Demo inference

Source: [4_demo.py](../../src/4_demo.py).

| 5W1H | Mô tả |
|---|---|
| What | Dự đoán một ảnh bằng checkpoint CIFAR-100 |
| Why | Minh họa model đã train, không nhân bản pipeline xử lý ảnh |
| Which | Checkpoint EVAL_RUN/EVAL_CHECKPOINT, ảnh DEMO_IMAGE và class order 100 lớp đã lưu |
| When | Sau khi có checkpoint tương thích và chọn DEMO_IMAGE |
| Where | Ảnh local, run nguồn; ghi run/demo/prediction_N.json |
| How | PIL RGB, transform chung, model eval và softmax Top-5 |

Đặt EVAL_RUN, EVAL_CHECKPOINT và DEMO_IMAGE trong config.py.
Kiểm tra file tồn tại, checkpoint/preprocessing/class order phù hợp trước inference.
Không tải pretrained hoặc tự train lại; không nạp official test dataset ngầm.
Checkpoint smoke bị chặn mặc định, không dùng như model đã train đầy đủ.

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/4_demo.py
```

In năm nhãn có điểm cao nhất, lưu JSON model/checkpoint/epoch/phase và đường dẫn ảnh.
Không sao chép pixel ảnh vào output Git; mỗi lần dự đoán tạo file số mới.
Chỉ là demo dòng lệnh, chưa có web/API upload ảnh.
Ảnh ngoài 100 lớp vẫn bị ép vào một nhãn; confidence không mặc định được hiệu chỉnh.
