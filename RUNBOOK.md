# Bài lab Day 10 — Nguyễn Thị Thùy Dương

Mã sinh viên nằm ở `solution/`; giữ nguyên công cụ và tests của đề.
`lab_day10.ipynb` là notebook chạy được trên Colab T4. Không có kết quả giả: report
chỉ được sinh sau khi có đủ bằng chứng thật E0–E5.

## Chạy

1. Mở notebook, chọn T4 GPU. Cell đầu giải nén mã nguồn đã đóng gói trong notebook.
2. Mount Drive để lưu checkpoint; dữ liệu và Hugging Face cache để trên ổ runtime.
3. Cài `requirements-student.txt`, chạy `python -m pytest -q tests student_tests`.
4. Tải dataset công khai qua kagglehub, chạy `scripts/prepare_data.py --source ...`.
5. Chạy `python -m solution.experiment suite --work <thư mục checkpoint trên Drive>`.
6. Chạy `python -m solution.reporting --work <cùng thư mục>`; kiểm tra bằng
   `python scripts/check_submission.py` và `python -m solution.check_quality`.

Pipeline: E0 → E1 → E2 → E3 → chọn setup bằng validation → E4 10%/100% → E5.
Chạy test một lần/setup; đọc validation để chọn baseline. E4 100% dùng lại run baseline
đúng như GUIDE cho phép. Tất cả SFT dùng batch 1, accumulation 8, 300 bước, cùng seed 42.
Mọi inference greedy, batch 4 và tối đa 512 tokens; E5 rollout tối đa 128 tokens theo đề.
Loss chỉ trên assistant answer và EOS. Quy ước nhãn/output là xyxy 0–1000.

## Khi bị ngắt

Giữ nguyên config, dataset và thư mục work. Chạy lại `suite`: nó tiếp tục checkpoint
SFT hoàn chỉnh gần nhất, optimizer/scheduler/scaler/RNG và cursor dữ liệu; output
JSONL tiếp tục những ảnh còn thiếu. Pointer checkpoint chỉ cập nhật sau khi lưu đủ file.
E5 dùng checkpoint chuẩn Trainer, lưu riêng projector kế thừa từ SFT.
Không thay siêu tham số giữa chừng trong cùng run; pipeline sẽ từ chối provenance lệch.
Mỗi lần train/infer chạy trong process riêng để giải phóng hoàn toàn bộ nhớ GPU/CPU
trước lần nạp model kế tiếp, tránh tích luỹ bộ nhớ trong kernel notebook.

Checkpoint cần Drive trống đủ chỗ: E2/E3 gồm projector float32 và optimizer state.
Số phút/VRAM trong evidence lấy từ GPU thật. Không suy luận điểm khi chưa chạy xong.

## Cổng kiểm tra điểm

- Đủ evidence test E0–E3, hai mức E4 và E5; không thiếu prediction, cùng dataset_version.
- E1 parse_valid_rate ≥ 0.95 và F1 E1−E0 ≥ 0.10 theo RUBRIC. Nếu không đạt,
  điều tra raw outputs/mask/learning curves trên validation trước một run mới.
- Có ≥6 ảnh GT/pred, log loss/tài nguyên, log bonus và test reward.
- Báo cáo trả lời 10 câu bằng số thật, thêm CI chênh lệch bootstrap theo phiên quay.
- Repo công khai của bạn có ≥5 commit; không commit data/work/secrets/trọng số.

Không đổi tool chấm để vượt kiểm tra. Test gốc `test_report_template_has_student_markers...`
kiểm tra **mẫu** REPORT có placeholder; chạy test đó trước khi sinh report hoàn chỉnh.
Sau khi có report, dùng checker bài nộp và tests sinh viên; template-test không áp dụng nữa.

Dataset `d10-21e4070301` có một nhãn crack cao bằng 0 trong ảnh train
`vlcsnap-00058`. Mã bỏ đúng nhãn suy biến này khỏi target SFT/GRPO, giữ hai
nhãn hợp lệ còn lại của ảnh. Không sửa split, count gốc hoặc GT của scorer.
`report/data_audit.json` và báo cáo cuối ghi rõ annotation bị loại khỏi target.

## Nội dung cần hiểu để vấn đáp

- ViT tạo đặc trưng, merger/DeepStack chiếu vào không gian LLM; mỗi setup mở khối nào?
- Vì sao ảnh liền kề phải cùng split và CI theo ảnh có thể quá lạc quan?
- YOLO tâm/kích thước đổi sang xyxy thế nào, thang 0–1000 đổi ngược về pixel ra sao?
- Vì sao phải mask prompt, ảnh và padding; kiểm tra tiền tố chat template có tác dụng gì?
- Vì sao adapter không chứa projector full-tune; lần inference đã nạp đủ chưa?
- Reward IoU ghép cùng lớp, phạt box thừa, xử lý rỗng/JSON lỗi ra sao?
- F1, CI, VRAM và thời gian thực tế hỗ trợ kết luận nào? Không khẳng định cải thiện
  nếu CI chênh lệch vẫn chứa 0; không tự suy ra nguyên nhân lỗi từ một metric.

## Tài liệu API

- [Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [TRL GRPO 1.9.2](https://huggingface.co/docs/trl/v1.9.2/en/grpo_trainer)

Đã kiểm tra source wheel đúng versions ghim. TRL 1.9.2 tự cast QLoRA trainables
sang bf16 khi khởi tạo trainer; mã phục hồi fp32 master weights sau khởi tạo cho T4/fp16.
Transformers 5.19.0 đối chiếu skip_modules theo prefix/regex từ đầu tên module:
phải có `model.visual` để giữ cả ViT fp16, không chỉ `visual`. Assertion kiểm tra
mọi tham số ViT/head sau khi nạp, trước khi bắt đầu bất kỳ thí nghiệm nào.
Colab hiện cài sẵn torchao 0.10.0, không tương thích PEFT 0.21.2. Bootstrap
gỡ torchao cũ (thư viện tùy chọn không dùng ở đây), giữ torch/torchvision của runtime.
