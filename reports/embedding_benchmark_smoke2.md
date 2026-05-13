# Embedding Benchmark Report

- Generated at: `2026-05-12T14:20:01`
- Corpus resources: 5
- Query cases: 22

## Summary

| Model | Composite | Coverage@k | Source MRR | Evidence MRR | Hit@1 src | Hit@1 ev | Peak RSS (GB) | Artifact (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multilingual-e5-small` | 0.7500 | 0.6818 | 0.9470 | 0.6439 | 0.9091 | 0.5455 | 0.829 | n/a |

## Hard Cases

### `multilingual-e5-small`

- `time_slot_ca1_accented`: source_rank=1 evidence_rank=1 coverage=0.3333
  - #1 `schedule_appendix_pdf` p.20: ### TABLE (Page 20) | K70E-CE7 | UET. MAT1051 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | Giải tích 2 | Giải tích 2 | 5 | | 60 | 145 | 9 | 61 | UET.MAT1051 36 | CL | TH | 5 | 1 | 
  - #2 `schedule_appendix_pdf` p.20: ### TABLE (Page 20) | K70E-CE7 | UET. MAT1051 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | Giải tích 2 | Giải tích 2 | 5 | | 60 | 145 | 9 | 61 | UET.MAT1051 36 | CL | TH | 5 | 1 | 
  - #3 `schedule_appendix_pdf` p.20: ### TABLE (Page 20) | K70E-CE7 | UET. MAT1051 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | Giải tích 2 | Giải tích 2 | 5 | | 60 | 145 | 9 | 61 | UET.MAT1051 36 | CL | TH | 5 | 1 | 
- `time_slot_ca1_ascii`: source_rank=2 evidence_rank=4 coverage=0.3333
  - #1 `handbook_pdf` p.21: ### 3. Kiểm tra thông tin cá nhân a. Nhấn chuột vào mục "Cập nhật hồ sơ" b. Màn hình thông tin sinh viên xuất hiện ![PORTAL SINH VIÊN Interface](image_student_portal.png) *(Image s
  - #2 `schedule_appendix_pdf` p.5: ### TABLE (Page 5) | K67I-IT3 | INT3509 | INT3509 | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | 4 | 21 | | 1 |
  - #3 `schedule_appendix_pdf` p.5: ### TABLE (Page 5) | K67I-IT3 | INT3509 | INT3509 | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | Dự án (bắt buộc) | 4 | 21 | | 1 |
- `time_slot_ca2_query`: source_rank=1 evidence_rank=1 coverage=0.3333
  - #1 `schedule_appendix_pdf` p.20: ### TABLE (Page 20) | K70E-CE7 | UET. MAT1051 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | Giải tích 2 | Giải tích 2 | 5 | | 60 | 145 | 9 | 61 | UET.MAT1051 36 | CL | TH | 5 | 1 | 
  - #2 `handbook_pdf` p.31: công nhận tốt nghiệp và cấp bằng cử nhân chương trình đào tạo chuẩn tương ứng theo hình thức đào tạo chính quy. c. Sinh viên không đủ điều kiện tốt nghiệp được cấp giấy chứng nhận 
  - #3 `schedule_appendix_pdf` p.3: TC3+K69A-AI- 15 TC4 K69A-AI- TC1+K69A-AI- Lớp tiến trình TC2+K69A-AI- INT3229 Kỹ thuật và công nghệ dữ liệu lớn 3 30 1 30 INT3229 1 CL TH 5 1 306-A Ngô Minh Hương nhanh, học tuần 6
- `time_slot_ca4_query`: source_rank=1 evidence_rank=None coverage=0.0
  - #1 `schedule_appendix_pdf` p.4: ### TABLE (Page 4) | K68I-CS2 | INT3230E | INT3230E | Mật mã và an toàn thông tin | Mật mã và an toàn thông tin | Mật mã và an toàn thông tin | Mật mã và an toàn thông tin | 4 | 60
  - #2 `handbook_pdf` p.31: công nhận tốt nghiệp và cấp bằng cử nhân chương trình đào tạo chuẩn tương ứng theo hình thức đào tạo chính quy. c. Sinh viên không đủ điều kiện tốt nghiệp được cấp giấy chứng nhận 
  - #3 `schedule_appendix_pdf` p.3: TC3+K69A-AI- 15 TC4 K69A-AI- TC1+K69A-AI- Lớp tiến trình TC2+K69A-AI- INT3229 Kỹ thuật và công nghệ dữ liệu lớn 3 30 1 30 INT3229 1 CL TH 5 1 306-A Ngô Minh Hương nhanh, học tuần 6
- `ai_teacher_list_accented`: source_rank=1 evidence_rank=None coverage=0.0
  - #1 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
  - #2 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
  - #3 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
- `ai_teacher_list_ascii`: source_rank=1 evidence_rank=3 coverage=0.6667
  - #1 `schedule_appendix_pdf` p.21: ### TABLE (Page 21) | K70M-MT3 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | 5 | 45 | | 145 | 9 | 70 | UET.MAT1051 42 | CL | LT | 6 | 4 | 206-T | Nguyễn Đình Kiên | Nguyễn Đình Kiên
  - #2 `schedule_appendix_pdf` p.21: ### TABLE (Page 21) | K70M-MT3 | UET. MAT1051 | Giải tích 2 | Giải tích 2 | 5 | 45 | | 145 | 9 | 70 | UET.MAT1051 42 | CL | LT | 6 | 4 | 206-T | Nguyễn Đình Kiên | Nguyễn Đình Kiên
  - #3 `schedule_appendix_pdf` p.1: ### TABLE (Page 1) | Lớp | Mã HP | Mã HP | Mã HP | Môn | Môn | Môn | TC | LT | TH | Tự học | PCGD | SS lớp | Mã LHP | Mã LHP | Nhóm | LT/TH | Thứ | Ca | GĐ | GĐ | GV | GV | GV | Gh
- `ai_schedule_by_name`: source_rank=1 evidence_rank=None coverage=0.0
  - #1 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
  - #2 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
  - #3 `schedule_appendix_pdf` p.22: ### TABLE (Page 22) | K70C- CE1+K70C- CE2+K70C- CE3+K70C-ID4. | VNU1001 | Nhập môn công nghệ số và ứng dụng trí tuệ nhân tạo | 3 | 45 | 202 | VNU1001 10 | CL | ONL | ONL | Viện Khả
- `curriculum_hmi_group`: source_rank=1 evidence_rank=1 coverage=0.6667
  - #1 `cs2022_curriculum_html` p.None: | 60 | INT3420E | Học sâu và Ứng dụngDeep learning and Applications | 3 | 45 | | | INT3405E | | 61 | INT3137 | Phân tích dữ liệu trực quanVisual Data Analytics | 3 | 45 | | | INT34
  - #2 `schedule_appendix_pdf` p.4: K68I-CS3 INT3123 Các thuật toán đồ thị và ứng dụng 3 45 1 74 INT3123 1 CL LT 2 4 205-T Tạ Việt Cường Học 1 ca/15 tuần, thi đợt 2 K68I-CS3 INT3420E Học sâu và Ứng dụng 3 45 1 74 INT
  - #3 `cs2022_curriculum_html` p.None: | 39 | INT3131INT3132 | | 75 | | 3 tín chỉ từ danh sách các học phần tự chọn theo các định hướng mà sinh viên chưa học | 3 | | | | | | | Tổng cộng | 136 | | | | | Ghi chú: Học phần
