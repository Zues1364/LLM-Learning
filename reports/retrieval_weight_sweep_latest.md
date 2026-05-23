# Báo cáo quét trọng số cho điểm tổng hợp truy hồi

- Nguồn số liệu: `D:\LLM\LLM Learning\reports\embedding_benchmark_compare_20260515.json`
- Số mô hình đem so: 3
- Bước quét trọng số: 0.05
- Trọng số tối thiểu mỗi chỉ số: 0.20
- Ràng buộc miền quét: trọng số độ phủ không nhỏ hơn hai trọng số MRR.
- Vùng ổn định: các cấu hình có khoảng cách giữa mô hình đứng đầu và mô hình thứ hai đạt ít nhất 85% so với cấu hình tách tốt nhất.

## Bộ trọng số đề xuất

- Đề xuất: `Coverage@5 = 0.35`, `MRR tài liệu = 0.30`, `MRR đoạn = 0.35`
- Mô hình đứng đầu dưới bộ trọng số này: `multilingual-e5-small`
- Khoảng cách với mô hình đứng thứ hai: `0.0613`
- Khoảng cách của cấu hình này tới phân bố cân bằng `(1/3, 1/3, 1/3)`: `0.001667`
- Xếp hạng theo khoảng cách nếu chỉ nhìn `gap_to_second`: `13/17`
- Bộ này được chọn vì là cấu hình cân bằng nhất trong vùng ổn định, không phải vì có khoảng cách lớn nhất.

## So với bộ trọng số đang dùng

- Hiện tại: `0.40 / 0.30 / 0.30`, mô hình đứng đầu `multilingual-e5-small`, khoảng cách `0.0625`
- Chênh lệch khoảng cách so với bộ đề xuất: `-0.0013`

## Các cấu hình đứng đầu theo khoảng cách

Bảng dưới được sắp theo `gap_to_second` giảm dần. Vì vậy, cấu hình được đề xuất có thể không xuất hiện trong nhóm đầu nếu nó được chọn theo tiêu chí cân bằng trong vùng ổn định.

| # | Coverage@5 | MRR tài liệu | MRR đoạn | Mô hình đứng đầu | Cách biệt |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.60 | 0.20 | 0.20 | `multilingual-e5-small` | 0.0718 |
| 2 | 0.55 | 0.20 | 0.25 | `multilingual-e5-small` | 0.0717 |
| 3 | 0.50 | 0.20 | 0.30 | `multilingual-e5-small` | 0.0704 |
| 4 | 0.45 | 0.20 | 0.35 | `multilingual-e5-small` | 0.0691 |
| 5 | 0.55 | 0.25 | 0.20 | `multilingual-e5-small` | 0.0691 |
| 6 | 0.40 | 0.20 | 0.40 | `multilingual-e5-small` | 0.0679 |
| 7 | 0.50 | 0.25 | 0.25 | `multilingual-e5-small` | 0.0678 |
| 8 | 0.45 | 0.25 | 0.30 | `multilingual-e5-small` | 0.0665 |
| 9 | 0.40 | 0.25 | 0.35 | `multilingual-e5-small` | 0.0652 |
| 10 | 0.50 | 0.30 | 0.20 | `multilingual-e5-small` | 0.0651 |
| 11 | 0.45 | 0.30 | 0.25 | `multilingual-e5-small` | 0.0638 |
| 12 | 0.40 | 0.30 | 0.30 | `multilingual-e5-small` | 0.0625 |
| 13 | 0.35 | 0.30 | 0.35 | `multilingual-e5-small` | 0.0613 |
| 14 | 0.45 | 0.35 | 0.20 | `multilingual-e5-small` | 0.0612 |
| 15 | 0.40 | 0.35 | 0.25 | `multilingual-e5-small` | 0.0599 |
| 16 | 0.35 | 0.35 | 0.30 | `multilingual-e5-small` | 0.0586 |
| 17 | 0.40 | 0.40 | 0.20 | `multilingual-e5-small` | 0.0572 |

## Tần suất mô hình đứng đầu trong toàn bộ miền quét

- `multilingual-e5-small`: 17 cấu hình
