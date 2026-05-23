# Dữ liệu mock cho đánh giá

Thư mục này chứa dữ liệu giả lập để kiểm tra chatbot học vụ mà không dùng thông tin cá nhân thật.

## Hồ sơ giả lập chính

| Hồ sơ | CTĐT | Tín chỉ đã tích lũy | Tín chỉ còn thiếu | Học phần bắt buộc còn thiếu | Tín chỉ nhóm mở còn thiếu |
| --- | --- | ---: | ---: | --- | --- |
| `mock_ai2025_cross_program` | `ai_2025` | 28 | 108 | `AIT2001, AIT3001, AIT3002, AIT3003, INT4050` | `math_foundation: 9, ai_core: 24` |
| `mock_ce2022_mid_program` | `ce_2022` | 75 | 61 | `MAT1094, HIS1001, INT1003, INT1007, INT2208, INT3401, INT3402, INT3413, INT3414, INT3415, INT3416, INT3417, INT3131, INT4050` | `specialized_electives: 18` |
| `mock_cs2022_near_graduation` | `cs_2022` | 115 | 21 | `INT3131, INT3132, INT4050` | `general_knowledge: 6, specialized_electives: 5` |
| `mock_cyber2024_mid_program` | `cyber_2024` | 60 | 76 | `INT3505, INT3131, INT4050` | `security_electives: 10` |
| `mock_ds2025_mid_program` | `ds_2025` | 50 | 86 | `DSA3002, DSA3003, DSA3004, INT4050` | `data_electives: 12` |
| `mock_is2022_mid_program` | `is_2022` | 72 | 64 | `MAT1094, PEC1008, INT1003, INT1007, INT3220, INT3221, INT3222, INT3223, INT3224, INT3131, INT4050` | `specialized_electives: 20` |
| `mock_it2022_mid_program` | `it_2022` | 62 | 74 | `PEC1008, HIS1001, INT3110, INT3117, INT3131, INT3132, INT4050` | `general_knowledge: 8, specialized_electives: 18` |
| `mock_se2022_mid_program` | `se_2022` | 51 | 85 | `INT3124, INT3125, INT3131, INT3132, INT4050` | `specialized_electives: 18` |
## Tổ chức file

- `transcripts/*.json`: nguồn chính cho hồ sơ bảng điểm mock.
- `transcripts/*.csv`: bản bảng để đối chiếu nhanh từng học phần.
- `curricula/*.json`: CTĐT mock gồm các nhóm học phần, tổng số tín chỉ và mã môn cốt lõi.

## Ghi chú sử dụng

Script `scripts/evaluate_chatbot.py` có thể render PDF bảng điểm từ các file JSON vào `tmp/eval_mock_pdfs/` trước khi upload lên `/upload_pdfs`.
PDF sinh ra dùng văn bản an toàn cho parser; JSON và CSV trong thư mục này mới là nguồn đối chiếu chính khi kiểm tra thủ công.
