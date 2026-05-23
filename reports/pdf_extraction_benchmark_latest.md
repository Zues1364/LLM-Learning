# PDF Extraction Benchmark Report

- Generated at: `2026-05-18T18:06:30+00:00`
- Dataset dir: `D:\LLM\LLM Learning\evals\pdf_extraction`
- Cases: `15`
- Methods: `pdfplumber_raw_text_only, pdfplumber_text_plus_tables, page_ocr_tesseract_only, img2table_tesseract, table_first_strict, hybrid_current`
- Score weights: `key=0.35`, `row=0.35`, `cell_f1=0.30`

## Overall Summary

| Method | Cases | Pass | Pass rate | Key acc | Row acc | Cell F1 | Score PDF | p50 ms | p95 ms | Vision rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pdfplumber_raw_text_only` | 15 | 2 | 13.33% | 0.4865 | 0.2703 | 0.1279 | 0.3033 | 66.42 | 597.87 | 0.00% |
| `pdfplumber_text_plus_tables` | 15 | 3 | 20.00% | 0.6486 | 0.3243 | 0.1662 | 0.3904 | 60.31 | 1270.11 | 0.00% |
| `page_ocr_tesseract_only` | 15 | 0 | 0.00% | 0.1892 | 0.0000 | 0.0267 | 0.0742 | 1538.04 | 22418.03 | 0.00% |
| `img2table_tesseract` | 15 | 1 | 6.67% | 0.8378 | 0.1351 | 0.1727 | 0.3923 | 12838.05 | 134957.76 | 0.00% |
| `table_first_strict` | 15 | 1 | 6.67% | 0.8378 | 0.1351 | 0.1727 | 0.3923 | 10844.83 | 130537.31 | 0.00% |
| `hybrid_current` | 15 | 4 | 26.67% | 0.9730 | 0.4324 | 0.2799 | 0.5759 | 59.88 | 2134.3 | 6.67% |

## Summary by Document Type

### `pdfplumber_raw_text_only`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `schedule` | 8 | 12.50% | 0.5625 | 0.1875 | 0.0398 | 0.2744 |
| `transcript` | 6 | 16.67% | 0.5000 | 0.3889 | 0.2276 | 0.3794 |

### `pdfplumber_text_plus_tables`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `schedule` | 8 | 25.00% | 0.9375 | 0.3125 | 0.1282 | 0.4760 |
| `transcript` | 6 | 16.67% | 0.5000 | 0.3889 | 0.2276 | 0.3794 |

### `page_ocr_tesseract_only`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `schedule` | 8 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `transcript` | 6 | 0.00% | 0.3889 | 0.0000 | 0.0548 | 0.1526 |

### `img2table_tesseract`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `schedule` | 8 | 12.50% | 0.9375 | 0.1250 | 0.1380 | 0.4133 |
| `transcript` | 6 | 0.00% | 0.8889 | 0.1667 | 0.2324 | 0.4392 |

### `table_first_strict`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `schedule` | 8 | 12.50% | 0.9375 | 0.1250 | 0.1380 | 0.4133 |
| `transcript` | 6 | 0.00% | 0.8889 | 0.1667 | 0.2324 | 0.4392 |

### `hybrid_current`

| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `english_mapping` | 1 | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `schedule` | 8 | 25.00% | 0.9375 | 0.3125 | 0.1230 | 0.4744 |
| `transcript` | 6 | 16.67% | 1.0000 | 0.4445 | 0.2993 | 0.5954 |

## English Mapping Downstream Queries

| Method | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| `pdfplumber_raw_text_only` | 0 | 5 | 0.00% |
| `pdfplumber_text_plus_tables` | 0 | 5 | 0.00% |
| `page_ocr_tesseract_only` | 0 | 5 | 0.00% |
| `img2table_tesseract` | 0 | 5 | 0.00% |
| `table_first_strict` | 0 | 5 | 0.00% |
| `hybrid_current` | 5 | 5 | 100.00% |

## Notable Failures

- `pdfplumber_text_plus_tables` / `transcript_is2022_sparse_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_raw_text_only` / `transcript_is2022_sparse_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_text_plus_tables` / `transcript_is2022_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_text_plus_tables` / `transcript_ce2022_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_raw_text_only` / `transcript_is2022_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_raw_text_only` / `transcript_ce2022_image`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
- `pdfplumber_raw_text_only` / `official_time_slots_page3`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:2
  - preview: ĐẠI HỌC QUỐC GIA HÀ NỘI CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM | TRƯỜNG ĐẠI HỌC CÔNG NGHỆ Độc lập - Tự do - Hạnh phúc | THỜI GIAN HỌC TẬP VÀ GIẢNG DẠY NĂM HỌC 2025-2026 | Buổi Ca Tiết Thời gian học Ghi chú | 1 Tiết 1-3 07:00 – 09:40 Nghỉ 5 phút giữa các tiết | Sán
- `pdfplumber_text_plus_tables` / `english_mapping_handbook_page26`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
  - preview: ĐIỀU KIỆN ĐỂ ĐƯỢC MIỄN HỌC HỌC PHẦN TIẾNG ANH | Sinh viên được miễn học các học phần ngoại ngữ nếu thuộc một trong các đối tượng sau: | a) Đã tham gia kỳ thi đánh giá năng lực ngoại ngữ do Trường Đại học Ngoại ngữ , Đại học | Quốc gia Hà Nội tổ chức và đạt kết
- `pdfplumber_raw_text_only` / `english_mapping_handbook_page26`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
  - preview: ĐIỀU KIỆN ĐỂ ĐƯỢC MIỄN HỌC HỌC PHẦN TIẾNG ANH | Sinh viên được miễn học các học phần ngoại ngữ nếu thuộc một trong các đối tượng sau: | a) Đã tham gia kỳ thi đánh giá năng lực ngoại ngữ do Trường Đại học Ngoại ngữ , Đại học | Quốc gia Hà Nội tổ chức và đạt kết
- `pdfplumber_raw_text_only` / `schedule_appendix_p20_engineering`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:2
  - preview: K70E-CE7 U M E A T T . 1051 Giải tích 2 5 60 145 9 61 U 36 ET.MAT1051 CL TH 5 1 307-B Trần Mạnh Cường C t t h u a ờ ầ n i 2 v 1 ( ớ 1 h i ) ọ C c a đ ồ 1 n t g ừ H 11 ọ h c ọ 1 c c 2 a / c 1 a 0 /t t u u ầ ầ n n , t đ h ầ i u v , à từ o đ tu ợ ầ t n 2 | K70E-C
- `pdfplumber_raw_text_only` / `schedule_appendix_p21_math_logic`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:2
  - preview: K70M-MT3 U M E A T T . 1051 Giải tích 2 5 45 145 9 70 U 42 ET.MAT1051 CL LT 6 4 206-T Nguyễn Đình Kiên Học 1 ca/15 tuần, thi đợt 2 | K K 7 7 0 0 M M - - M M T T 3 3 U M U M E E A A T T T T . . 1 1 0 0 5 5 1 1 G G i i ả ả i i t t í í c c h h 2 2 5 5 6 6 0 0 1 1
- `page_ocr_tesseract_only` / `official_time_slots_page3`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:2
  - preview: ĐẠI HỌC QUỐC GIA HÀ NỘI CONG HOA XÃ HOI CHỦ NGHĨA VIỆT NAM | TRƯỜNG ĐẠI HỌC CÔNG NGHỆ Độc lập - Tự do - Hạnh phúc |  | THỜI GIAN HỌC TẬP VÀ GIẢNG DẠY NĂM HỌC 2025-2026 |  | Buổi | Ca Tiết Thời gian học Ghi chú |  | 1 | Tiết 1-3 07:00 — 09:40 | Nghỉ 5 phút giữa
- `page_ocr_tesseract_only` / `transcript_ai2025_text`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:3
  - preview: MOCE TRANSCR | STT | Ma HP | MAT1093 | MAT1094 |  | INT1007 | AIT1001 |  | DN OF WHY PR |  | | | | | | INT1003 | | | | | | MAT1101 |  | PT | bang diem mock ai2025 - text pdf (text table pdf) |  | Diem he 4 |  | Ten hoc phan | TC | đai so | 4 | 3.2 | giai tích 
- `page_ocr_tesseract_only` / `transcript_it2022_text`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:3
  - preview: MOCK TRANSCRIPT | bang diem mock it2022 - text pdf (text table pdf) | STT Ma HP Ten hoc phan | TC | Diem he 4 | Hoc ky | PHI1006 triet hoc mac - lenin | 3 2.8 | 2022-2023-1 | MAT1093 dai so | 4 | 3.0 | 2022-2023-1 |  | NT1003 tin hoc co so 1 | 4 | 3.1 2022-202
- `page_ocr_tesseract_only` / `english_mapping_handbook_page26`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:3
  - preview: DIEU KIEN DE ĐƯỢC MIEN HỌC HOC PHAN TIENG ANH |  | Sinh viên được miễn học các hoc phan ngoại ngữ nêu thuộc một trong các đối tượng sau: |  | a) Đã tham gia kỳ thi đánh giá năng lực ngoại ngữ do Trường Dai học Ngoại ngữ, Dai học | Quốc gia Hà Nội tô chức và đạ
- `page_ocr_tesseract_only` / `transcript_is2022_image`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:3
  - preview: bang diem mock is2022 - image pdf (image_table_ pdf) |  | Du lieu gia lap phục vu benchmark trích xuat PDF |  | 1 PHI1006 triet hoc mac - lenin 3 2.8 2022-2023-1 | 2 MAT1093 dai so 4 3.0 2022-2023-1 | 3 MAT 1041 giai tich 1 4 29 2022-2023-1 | 4 INT1008 tin hoc
- `page_ocr_tesseract_only` / `schedule_appendix_p22_vnu1001`: score=0.0000, key=0.0000, row=0.0000, errors=ocr_noise:2
  - preview: K?0C- |  | [net (Nhập môn công nghệ số và ứng dụng trí tuệ |  | (CEAtKroe.  [vNUlonl - [Nho mẻ 4s 202 [VNDI001 10 €L | ont ONL |Miện Khảo thi Tự học Học | ea/ IS tuk, thi đợt 2 | (CE1+K?0C-1D4. |  | 1 |  | KT0C- |  | Real YNUI001 ergo men công nghệ số và ung d
- `table_first_strict` / `english_mapping_handbook_page26`: score=0.0000, key=0.0000, row=0.0000, errors=missing_row:3
