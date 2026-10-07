# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Lê Phan Việt Cường |
| MSSV | 2A202602641 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/cuonglpv/K4-L3L4-Track2-Day21-LePhanVietCuong-2A202602641-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Tôi so sánh bằng F1 của lớp thu nhập cao. Cấu hình 200 cây, learning rate 0,1 và độ sâu 5 đạt F1 cao nhất 0,7149, cao hơn mặc định 0,7109 và nông/chậm 0,6051, nên qua Quality Gate 0,65. Accuracy cao nhất lại thuộc cấu hình mặc định (0,8780), không trùng lần có F1 cao nhất (0,8740); accuracy bị lớp thu nhập thấp chiếm đa số chi phối. Với learning rate 0,05 và chỉ 50 cây, mô hình thiếu vòng boosting nên cả F1 lẫn accuracy giảm. Tăng lên 200 cây, giữ learning rate 0,1 và cây sâu hơn giúp nhận diện lớp dương tốt hơn.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult không cân bằng: lớp thu nhập cao (`target=1`) chiếm khoảng 24,8%. Mô hình luôn đoán thu nhập thấp vẫn có accuracy xấp xỉ 75,2% nhưng không tìm được ca dương nào, nên F1 bằng 0. F1 kết hợp precision và recall của đúng lớp quan tâm, phản ánh tốt hơn khả năng nhận diện thu nhập cao. Tôi dùng trực tiếp `f1_score(y_eval, preds)`, không dùng `average="weighted"` hay `average="macro"`, vì điểm trung bình có thể được lớp đa số kéo lên và che khuất hiệu quả trên lớp thiểu số. Do đó Quality Gate dùng F1 >= 0,65.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Xung đột phụ thuộc Python | DVC nâng protobuf ngoài phạm vi MLflow hỗ trợ | Dùng `.venv` và khóa phiên bản trong `requirements.txt`. |
| SSH máy cá nhân lỗi | Plink không dùng khóa GCP | Dùng SSH-in-browser và khóa ED25519 riêng cho Actions. |
| Release health check lỗi | API khởi động lâu hơn 5 giây | Thử lại `/healthz` tối đa 30 giây. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0,7149 | 0,8740 |
| Bước 3 (thêm `train_batch2`) | 0,7354 | 0,8820 |

**Nhận xét:** Thêm 22.361 mẫu làm F1 tăng 0,0205 và accuracy tăng 0,0080. Dù kết quả tốt hơn trong lần này, hai batch cùng nguồn và phân phối nên vẫn cần theo dõi F1 qua các lần tái huấn luyện tiếp theo.
