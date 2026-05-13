# Mock evaluation data

Thu muc nay chua du lieu mock de kiem tra chatbot hoc vu ma khong dung thong tin ca nhan that.

## Profiles

| Profile | CTDT | Tin chi da hoc | Tin chi con thieu ky vong | Muc dich kiem thu |
| --- | --- | ---: | ---: | --- |
| `mock_cs2022_near_graduation` | `cs_2022` | 115 | 21 | Sinh vien gan tot nghiep, con thieu do an/khoa luan va mot phan tu chon |
| `mock_it2022_mid_program` | `it_2022` | 62 | 74 | Sinh vien giua chuong trinh, dung de kiem tra he thong khong suy luan qua muc |
| `mock_ai2025_cross_program` | `ai_2025` | 28 | 108 | CTDT khac CS2022, dung de bat loi lay nham chuong trinh dao tao |

## File organization

- `transcripts/*.json`: nguon chinh cho bang diem mock.
- `transcripts/*.csv`: ban bang de doc/kiem tra nhanh.
- `curricula/*.json`: CTDT mock co nhom hoc phan va tin chi.
- `curricula/*.html`: ban HTML don gian de test luong tai nguyen/CTDT khi can.

## PDF generation

Script `scripts/evaluate_chatbot.py` co the render PDF transcript tu JSON vao `tmp/eval_mock_pdfs/` truoc khi upload len `/upload_pdfs`.
PDF sinh ra chi chua du lieu ASCII de parser PDF doc on dinh; JSON/CSV trong thu muc nay moi la nguon doi chieu chinh.

## Manual check checklist

1. Mo file JSON profile can kiem tra.
2. Doi chieu `summary.expected_missing_credits` va `summary.expected_required_missing_codes`.
3. Mo CSV de kiem tra tung hoc phan, tin chi va diem he 4.
4. Khi eval fail, xem lai report trong `reports/eval_academic_advisor_*.md` de biet case nao sai route, citation hay noi dung.
