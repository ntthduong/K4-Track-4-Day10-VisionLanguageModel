# E5 continuation — 14/10/2026

E5 được chạy riêng trên [Kaggle E5 Stable](https://www.kaggle.com/code/ntthduong/day10-qwen3vl-thuyduong-e5-stable).
Input là output Version 2 của notebook Full Bonus (session `358072033`).
Adapter và projector E3 phải khớp run ID `700c3fb0abc0f7c454bf2e1c2b35f0451cfc798dd381aefb96184bfb56e35395`,
dataset `d10-21e4070301`, 300 bước. Runner không tự huấn luyện lại E0–E4 nếu thiếu checkpoint.

Lỗi trước: callback ghi log chứa NaN bị JSON strict từ chối ở bước 18/60.
Bản sửa lưu chỉ số phụ không hữu hạn thành null và ghi nguyên tên/giá trị trong
`non_finite_metrics`. Loss, grad_norm và reward không hữu hạn vẫn gây lỗi rõ ràng
sau khi lưu dòng chẩn đoán; không biến chúng thành 0 hoặc coi run là thành công.
E5 lưu checkpoint mỗi 5 bước, giữ hai checkpoint gần nhất.

Phiên chẩn đoán [E5 Continuation Version 1](https://www.kaggle.com/code/ntthduong/day10-qwen3vl-thuyduong-e5-continuation)
(session `358218040`) xác định lỗi `grad_norm=NaN` ở bước 9 và đã lưu checkpoint 5.
Runner mới nhận thêm output đó để resume. GRPO fp16 dùng GradScaler mặc định
init_scale=65536; callback `on_train_begin` giới hạn scale <=128 sau khi Trainer
nạp scaler checkpoint, như vòng SFT T4. Mỗi log ghi loss_scale và optimizer_step_skipped.
Đây là sửa precision/loss scaling; không thay LR, reward, số bước hay lượng dữ liệu.
E5 Stable Version 1 (session `358221873`) đã hoàn tất 60 bước và chấm validation/test.
Các bước resume 6–60 ghi scale=128, optimizer_step_skipped=false; giá trị mới nhất
của cả 60 bước có loss/grad_norm/reward hữu hạn. Dòng NaN lịch sử bước 9 được giữ nguyên.

Cấu hình thí nghiệm giữ nguyên: 60 bước, 4 generations, 128 completion tokens,
LR=5e-6, beta=0, projector SFT đóng băng. Sau train, runner chấm validation/test
bằng scorer gốc và sinh báo cáo so sánh SFT/GRPO, paired CI theo phiên và reward hacking.

Notebook local: `lab_day10_e5_kaggle.ipynb`.
Tạo lại: `python -m solution.make_e5_notebook`.
Chạy trên Kaggle: `python -u -m solution.finish_e5 --previous <thư mục output Full Bonus>`.
Archive kết quả: `/kaggle/working/lab_day10_e5/day10_e5_results.zip`.

Kết quả test: 311 ảnh, parse rate=1, F1=0.4367816092; TP=228, FP=233, FN=355.
E3 F1=0.4332061069; paired session CI95 của delta=[-0.0039464, 0.0560386], chứa 0.
Chưa chứng minh được GRPO cải thiện F1 rõ ràng. Validation F1=0.3397212544.
Mean test reward tăng 0.457672→0.465913; không có output rỗng/nhãn lạ/box >= nửa ảnh.

Archive đã tải và kiểm tra CRC/SHA256; cả sáu raw test output chấm lại bằng scorer gốc
khớp toàn bộ trường evidence. Chi tiết ở `report/e5_provenance.json`.
Tái kiểm chứng và sinh báo cáo: `python -m solution.import_e5_results <day10_e5_results.zip>`.
Kiểm tra local: 101 passed, 1 skipped, 1 deselected; test placeholder REPORT được
deselect vì báo cáo đã điền. Quality checks đã đủ E0–E5, sáu ảnh và paired CI theo phiên.
