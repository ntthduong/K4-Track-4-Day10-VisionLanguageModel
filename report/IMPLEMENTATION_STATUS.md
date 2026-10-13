# Trạng thái thực hiện — Nguyễn Thị Thùy Dương

## Đã triển khai

- Notebook Colab tự chứa source: E0, E1, E2, E3, E4 (10%/100% dữ liệu), E5 GRPO.
- Mã ở `solution/`; `scripts/`, `tests/`, `configs/` của đề giữ nguyên.
- Loss mask, mục tiêu LoRA, lưu/nạp projector, checkpoint và resume optimizer/RNG.
- Mỗi lần train/infer chạy trong process riêng để giải phóng bộ nhớ.
- Sinh báo cáo 10 câu từ evidence thật, 6 ảnh minh hoạ, loss/F1 plot,
  paired bootstrap theo phiên quay, phân tích reward hacking và checker rubric.
- Source đã commit theo các bước trên nhánh `codex/full-lab`.

## Đã kiểm chứng trên Colab T4

- Dataset thật: `d10-21e4070301`, 1.422 train / 276 val / 311 test.
- Smoke test E3 **một bước**, batch 1, accumulation 1 (chỉ kiểm tra kỹ thuật):
  348 module LoRA; 148.426.752 tham số học; loss 1.352759838;
  gradient norm 9.05917263; peak allocated VRAM 9.119306087 GiB.
- Một ảnh 640×360: 220 token ảnh, 250 token assistant; mask đúng JSON và EOS.
- Tổng model đo từ hình dạng tensor: 4.437.815.808 tham số;
  ViT không projector 306.242.560; projector 109.105.152;
  LLM + head 4.022.468.096.
- Đã sửa hai lỗi phát hiện khi chạy thật: prefix `model.visual` để tránh
  quantize ViT ở Transformers 5.19.0; gỡ torchao 0.10.0 không tương thích PEFT.

Smoke test không phải kết quả E3 nộp bài; không đưa số liệu này vào evidence chấm điểm.

## Phần chưa hoàn thành

### Chạy lại ngày 13/10/2026

Đã xác nhận Tesla T4 và mount Drive thành công. Pipeline `solution.finish`
chạy từ khoảng 20:43 (UTC+7), lưu trực tiếp tại
`/content/drive/MyDrive/Day10_ThuyDuong/checkpoints`, để chạy đủ E0–E5,
báo cáo và đóng gói kết quả. Chưa có F1 hoàn chỉnh và chưa xác nhận đạt điểm tối đa.

Audit dữ liệu phát hiện một nhãn crack cao bằng 0 ở ảnh train `vlcsnap-00058`.
Chỉ loại nhãn này khỏi target SFT/GRPO, giữ split, count gốc và GT chấm điểm.
Mọi setup dùng suy luận greedy batch 16, tối đa 512 tokens; SFT batch 1 × accumulation 8.
Output E0 chạy thử batch 4 được lưu riêng trong `diagnostic_batch4_*` trên Drive;
không dùng làm evidence chấm điểm. Bộ thí nghiệm batch 16 chạy mới thống nhất.
Bản sửa đã qua **91 tests, 1 skipped** trên máy local.

### Lần chạy trước

Colab mất runtime trong lần kiểm tra batch 2, rồi báo:
**“Failed to assign a backend — No backend with GPU available.”**
Chưa chạy đủ 300 bước/setup, chưa có predictions/F1 E0–E5, chưa có report thực nghiệm
hoàn chỉnh. `REPORT.md` còn mẫu để generator điền khi đủ kết quả thật.
Không có evidence giả trong `report/evidence/`. Không thể xác nhận điểm tối đa ở trạng thái này.

Notebook đầy đủ đã upload:
[Lab Day 10 trên Colab](https://colab.research.google.com/drive/1mUq6yGQf8BejJxS8uv3QuEvbSC9AtVLY).
Notebook kiểm tra GPU ban đầu:
[GPU smoke test](https://colab.research.google.com/drive/1EvNytCkV0R5Onc1KNhoYOCdywj4xsSNC).

Khi Colab có T4, chạy notebook đầy đủ từ đầu. Checkpoint/evidence lưu vào
`MyDrive/Day10_ThuyDuong/`. Lệnh tự chạy đủ thí nghiệm, báo cáo và đóng gói kết quả:

```bash
python -u -m solution.finish --work /content/drive/MyDrive/Day10_ThuyDuong/checkpoints
```

Đưa kết quả thật về repo local, chạy `scripts/check_submission.py` và
`solution.check_quality`, rồi kiểm tra repo GitHub công khai theo yêu cầu đề.
