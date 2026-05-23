# PDF Extraction Benchmark

Bộ dữ liệu này dùng để đo chất lượng trích xuất PDF có bảng trong hệ thống RAG Cosmic.

## Cấu trúc

- `cases/*.json`: ground truth từng mẫu PDF.
- `scripts/benchmark_pdf_extraction.py`: chạy 6 phương pháp trích xuất và ghi log chi tiết.

## Nhóm tài liệu

- `schedule`: các trang thời khóa biểu và bảng khung giờ.
- `transcript`: bảng điểm mock dùng để kiểm thử trích xuất bảng điểm mà không dùng dữ liệu cá nhân thật.
- `english_mapping`: bảng tham chiếu quy đổi điểm tiếng Anh trong sổ tay học vụ.

## Ghi chú

- Mục `expected_rows` là tập hàng dùng để chấm benchmark.
- Mục `key_fields` là các trường bắt buộc phải khớp để tính độ chính xác trường khóa.
- Các PDF bảng điểm mock được materialize khi chạy benchmark; log chi tiết về file sinh ra được đặt trong `reports/`.
