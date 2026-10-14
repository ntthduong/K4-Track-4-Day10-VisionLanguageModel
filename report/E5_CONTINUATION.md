# E5 continuation — 14/10/2026

E5 đang được chạy riêng trên [Kaggle](https://www.kaggle.com/code/ntthduong/day10-qwen3vl-thuyduong-e5-continuation).
Input là output Version 2 của notebook Full Bonus (session `358072033`).
Adapter và projector E3 phải khớp run ID `700c3fb0abc0f7c454bf2e1c2b35f0451cfc798dd381aefb96184bfb56e35395`,
dataset `d10-21e4070301`, 300 bước. Runner không tự huấn luyện lại E0–E4 nếu thiếu checkpoint.

Lỗi trước: callback ghi log chứa NaN bị JSON strict từ chối ở bước 18/60.
Bản sửa lưu chỉ số phụ không hữu hạn thành null và ghi nguyên tên/giá trị trong
`non_finite_metrics`. Loss, grad_norm và reward không hữu hạn vẫn gây lỗi rõ ràng
sau khi lưu dòng chẩn đoán; không biến chúng thành 0 hoặc coi run là thành công.
E5 lưu checkpoint mỗi 5 bước, giữ hai checkpoint gần nhất.

Cấu hình thí nghiệm giữ nguyên: 60 bước, 4 generations, 128 completion tokens,
LR=5e-6, beta=0, projector SFT đóng băng. Sau train, runner chấm validation/test
bằng scorer gốc và sinh báo cáo so sánh SFT/GRPO, paired CI theo phiên và reward hacking.

Notebook local: `lab_day10_e5_kaggle.ipynb`.
Tạo lại: `python -m solution.make_e5_notebook`.
Chạy trên Kaggle: `python -u -m solution.finish_e5 --previous <thư mục output Full Bonus>`.
Archive kết quả: `/kaggle/working/lab_day10_e5/day10_e5_results.zip`.

Kiểm tra local: 101 passed, 1 skipped; test placeholder REPORT được deselect vì báo cáo
đã điền. Chưa có evidence E5 hoàn chỉnh khi ghi tài liệu này; không dùng log tiến trình
để khẳng định bonus đã hoàn thành.
