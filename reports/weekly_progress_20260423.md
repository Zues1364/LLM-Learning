# Báo cáo Tiến độ Hàng tuần

**Chủ đề:** Mở rộng tài nguyên CTĐT, ổn định Advisor/Parser, và chuẩn bị Mail Agent (giai đoạn trước commit gần nhất)  
**Người thực hiện:** Vũ Ninh Giang  
**Ngày báo cáo:** 23/04/2026

## 1. Tổng quan
Báo cáo này tổng hợp **toàn bộ thay đổi code trước commit gần nhất** trong repository `D:\LLM\LLM Learning`, dựa trên lịch sử Git.

Mốc commit gần nhất hiện tại là:
- `29b14f2` (2026-03-09): `merge: deep advisor matrix and curriculum parser stabilization`

Phạm vi commit được tổng hợp trong báo cáo này (trước `29b14f2`):
- `a29e544` (2026-03-09)
- `94a4b5b` (2026-03-02)
- `8f33acd` (2026-02-24, commit date 2026-03-02)

## 2. Trạng thái Git tại thời điểm 23/04/2026
- Nhánh hiện tại: `develop/feat/gmail-user-auth-mail-polling`
- Commit HEAD: `29b14f2`
- Số commit mới kể từ `29b14f2`: **0**
- Kết luận: từ sau 09/03/2026 đến 23/04/2026 chưa có commit mới trên HEAD hiện tại; các thay đổi mới đang nằm ở trạng thái working tree (chưa commit).

## 3. Chi tiết thay đổi code trước commit gần nhất

### 3.1. Commit `8f33acd`
**Thông điệp:** `fix(app): normalize Gemini key usage and guard embedding cache dimensions`  
**Quy mô thay đổi:** 6 files changed, 90 insertions(+), 40 deletions(-)

Nội dung chính:
- Chuẩn hóa cách dùng API key Gemini trong app/agent.
- Tăng độ an toàn cho cơ chế cache embeddings bằng kiểm tra chiều vector.
- Giảm lỗi runtime do cấu hình env/model không đồng nhất.

File chính bị tác động:
- `src/agents.py`
- `src/app.py`
- `src/env_loader.py`
- `src/mcp_server/server.py`
- `src/utils.py`

### 3.2. Commit `94a4b5b`
**Thông điệp:** `feat(resources): add batch local uploads and improve curriculum metadata parsing`  
**Quy mô thay đổi:** 4 files changed, 269 insertions(+), 33 deletions(-)

Nội dung chính:
- Bổ sung upload local resources theo batch (PDF/HTML) thay cho luồng file đơn.
- Cải thiện parse metadata CTĐT phục vụ phân nhóm/hiển thị chương trình đào tạo.
- Đồng bộ logic backend/frontend cho refresh resources/programs.

File chính bị tác động:
- `frontend/src/App.jsx`
- `src/app.py`
- `src/mcp_server/server.py`
- `src/utils.py`

### 3.3. Commit `a29e544`
**Thông điệp:** `fix(advisor): stabilize curriculum parsing and deep matrix regression tooling`  
**Quy mô thay đổi:** 8 files changed, 1191 insertions(+), 117 deletions(-)

Nội dung chính:
- Ổn định parser curriculum, giảm lỗi parse nhóm học phần.
- Củng cố logic advisor cho bài toán thiếu tín chỉ/lịch học theo CTĐT.
- Bổ sung tooling matrix regression + deep I/O logging phục vụ kiểm thử diện rộng.
- Tăng độ chặt strict matching cho dò mã học phần/lớp học.

File chính bị tác động:
- `src/mcp_server/server.py`
- `src/utils.py`
- `test/run_ctdt_matrix_fake_transcript.py`
- `test/run_deep_advisor_io.py`
- `tests/integration/test_ctdt_matrix_smoke.py`
- `tests/integration/test_curriculum_group_parser_regression.py`
- `tests/integration/test_schedule_strict_code_match.py`

### 3.4. Commit `29b14f2` (mốc hợp nhất)
**Thông điệp:** `merge: deep advisor matrix and curriculum parser stabilization`  
Ý nghĩa:
- Đóng vai trò merge checkpoint để đưa phần ổn định parser/matrix lên nền `dev-main`.
- Tạo mốc ổn định để tiếp tục phát triển các nhánh feature mới.

## 4. Đánh giá tổng hợp giai đoạn trước commit gần nhất
- Hệ thống đã chuyển từ trạng thái chỉnh sửa rời rạc sang nền parse/kiểm thử có cấu trúc rõ ràng.
- Khối tài nguyên local + metadata CTĐT được nâng cấp theo hướng production-friendly hơn.
- Advisor pipeline được gia cố bằng matrix regression, giảm rủi ro hồi quy khi mở rộng thêm CTĐT.

## 5. Snapshot tham khảo sau commit gần nhất (chưa commit)
Phần này **không thuộc phạm vi “trước commit gần nhất”**, chỉ để theo dõi tiến độ hiện hành:
- Working tree đang có: `14 files modified`, nhiều file mới chưa track.
- Trọng tâm WIP nằm ở các cụm:
- `src/mail_agent.py`
- `src/mcp_server/structured_schedule_store.py`
- `src/app.py`, `src/mcp_server/server.py`, `src/resource_loader.py`
- `frontend/src/App.jsx`, `frontend/src/style.css`
- test tích hợp/đơn vị cho mail updates và structured schedule
- Đã hoàn thành chức năng hiển thị nguồn tham chiếu theo kiểu **NotebookLM**:
  - Gắn citation badge theo từng dòng nội dung trả lời (thay vì chỉ liệt kê nguồn ở cuối).
  - Mở popup xem trích đoạn nguồn khi bấm vào badge, kèm thông tin vị trí (page/chunk/line khi có).
  - Cải thiện lọc/match citation để đoạn trích sát ngữ cảnh câu trả lời hơn.

## 6. Kế hoạch tiếp theo
- Chốt commit theo nhóm logic rõ ràng: mail agent, resources scope, frontend review panel, test.
- Chạy lại full regression gate trước merge vào nhánh chính.
- Hoàn thiện tài liệu cấu hình OAuth + vận hành poll/review/apply theo session/user.

## 7. Phụ lục lệnh kiểm tra
- `git log --oneline --decorate -n 20`
- `git log --all --since="2026-03-12 00:00:00"`
- `git show --stat 8f33acd`
- `git show --stat 94a4b5b`
- `git show --stat a29e544`
