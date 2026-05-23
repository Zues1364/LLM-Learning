# Báo cáo quét trọng số cho điểm tổng hợp trích xuất PDF

- Nguồn số liệu: `D:\LLM\LLM Learning\reports\pdf_extraction_benchmark_latest.json`
- Số phương pháp đem so: 6
- Bước quét trọng số: 0.05
- Trọng số tối thiểu mỗi chỉ số: 0.20
- Vùng ổn định: các cấu hình có khoảng cách giữa phương pháp đứng đầu và phương pháp thứ hai đạt ít nhất 85% so với cấu hình tách tốt nhất.

## Bộ trọng số đề xuất

- Đề xuất: `Cột chính = 0.35`, `Hàng = 0.35`, `F1 ô = 0.30`
- Phương pháp đứng đầu dưới bộ trọng số này: `hybrid_current`
- Khoảng cách với phương pháp đứng thứ hai: `0.1835`
- Khoảng cách của cấu hình này tới phân bố cân bằng `(1/3, 1/3, 1/3)`: `0.001667`
- Xếp hạng theo khoảng cách nếu chỉ nhìn `gap_to_second`: `6/45`
- Bộ này được chọn vì là cấu hình cân bằng nhất trong vùng ổn định, không phải vì có khoảng cách lớn nhất.

## So với bộ trọng số đang dùng

- Hiện tại: `0.40 / 0.35 / 0.25`, phương pháp đứng đầu `hybrid_current`, khoảng cách `0.1849`
- Chênh lệch khoảng cách so với bộ đề xuất: `-0.0014`

## Các cấu hình đứng đầu theo khoảng cách

Bảng dưới được sắp theo `gap_to_second` giảm dần. Vì vậy, cấu hình được đề xuất có thể không xuất hiện trong nhóm đầu nếu nó được chọn theo tiêu chí cân bằng trong vùng ổn định.

| # | Cột chính | Hàng | F1 ô | Phương pháp đứng đầu | Cách biệt |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.40 | 0.40 | 0.20 | `hybrid_current` | 0.1944 |
| 2 | 0.45 | 0.35 | 0.20 | `hybrid_current` | 0.1863 |
| 3 | 0.35 | 0.40 | 0.25 | `hybrid_current` | 0.1852 |
| 4 | 0.40 | 0.35 | 0.25 | `hybrid_current` | 0.1849 |
| 5 | 0.35 | 0.45 | 0.20 | `hybrid_current` | 0.1849 |
| 6 | 0.35 | 0.35 | 0.30 | `hybrid_current` | 0.1835 |
| 7 | 0.50 | 0.30 | 0.20 | `hybrid_current` | 0.1782 |
| 8 | 0.45 | 0.30 | 0.25 | `hybrid_current` | 0.1768 |
| 9 | 0.40 | 0.30 | 0.30 | `hybrid_current` | 0.1754 |
| 10 | 0.30 | 0.35 | 0.35 | `hybrid_current` | 0.1749 |
| 11 | 0.30 | 0.40 | 0.30 | `hybrid_current` | 0.1747 |
| 12 | 0.30 | 0.45 | 0.25 | `hybrid_current` | 0.1744 |
| 13 | 0.30 | 0.50 | 0.20 | `hybrid_current` | 0.1741 |
| 14 | 0.35 | 0.30 | 0.35 | `hybrid_current` | 0.1740 |
| 15 | 0.30 | 0.30 | 0.40 | `hybrid_current` | 0.1726 |
| 16 | 0.55 | 0.25 | 0.20 | `hybrid_current` | 0.1701 |
| 17 | 0.50 | 0.25 | 0.25 | `hybrid_current` | 0.1687 |
| 18 | 0.45 | 0.25 | 0.30 | `hybrid_current` | 0.1673 |
| 19 | 0.40 | 0.25 | 0.35 | `hybrid_current` | 0.1659 |
| 20 | 0.25 | 0.30 | 0.45 | `hybrid_current` | 0.1647 |
| 21 | 0.35 | 0.25 | 0.40 | `hybrid_current` | 0.1645 |
| 22 | 0.25 | 0.35 | 0.40 | `hybrid_current` | 0.1644 |
| 23 | 0.25 | 0.40 | 0.35 | `hybrid_current` | 0.1641 |
| 24 | 0.25 | 0.45 | 0.30 | `hybrid_current` | 0.1639 |
| 25 | 0.25 | 0.50 | 0.25 | `hybrid_current` | 0.1636 |

## Tần suất phương pháp đứng đầu trong toàn bộ miền quét

- `hybrid_current`: 45 cấu hình
