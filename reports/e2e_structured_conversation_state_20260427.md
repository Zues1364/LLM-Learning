# E2E Report - Structured Conversation State

- Date: 2026-04-27
- Branch: `develop/feat/structured-conversation-state`
- Scope: Triển khai lưu state hội thoại có cấu trúc + resolve truy vấn tham chiếu (`môn này`, `kỳ này`, `câu trước`) trong luồng `/ask`.

## 1. Thay đổi chính

### Backend
- Added: `src/conversation_state.py`
  - `default_conversation_state()`
  - `resolve_query_with_state()`
  - `update_state_after_turn()`
- Updated: `src/app.py`
  - Load structured state trước bước planner.
  - Resolve query tham chiếu theo state trước khi build planner prompt.
  - Persist state sau khi có answer (hoặc planner error branch).
- Updated: `src/persistent_memory.py`
  - New table: `conversation_state(session_id, state_json, updated_at)`
  - New APIs:
    - `get_structured_state(session_id)`
    - `save_structured_state(session_id, state)`
  - `clear_session()` now clears both `history` and `conversation_state`.
- Updated: `src/mcp_server/server.py`
  - New MCP tools:
    - `memory_state_get`
    - `memory_state_upsert`
    - `memory_state_clear`

### Tests
- Added: `tests/unit/test_conversation_state.py`
- Added: `tests/unit/test_persistent_memory_state.py`
- Added: `tests/integration/test_structured_state_e2e.py`
- Updated: `tests/integration/test_mcp_server_tools.py` (roundtrip test cho memory state tools)

## 2. E2E Scenario đã kiểm thử

### Scenario: Follow-up với tham chiếu `môn này`
1. User hỏi: `tôi cần lịch học môn thị giác máy ở kì này`
2. System trả về context có `INT3412E` và lưu vào structured state (`entities.course_codes`, `referents.last_subject_codes`).
3. User hỏi tiếp: `môn này có mở lớp không`
4. System resolve query sang dạng có mã môn (`INT3412E`) trước planner.
5. Planner prompt lượt 2 chứa `INT3412E`, answer vẫn đúng ngữ cảnh follow-up.

Kết quả: PASS.

## 3. Test execution

### Command 1
```bash
python -m pytest tests/unit/test_conversation_state.py tests/unit/test_persistent_memory_state.py -q
```
Result: `4 passed`.

### Command 2
```bash
python -m pytest tests/integration/test_structured_state_e2e.py tests/integration/test_app_ask.py -q
```
Result: `5 passed`.

### Command 3
```bash
python -m pytest tests/integration/test_mcp_server_tools.py -q
```
Result: `13 passed`.

### Tổng
- Total: `22 passed`
- Failures: `0`

## 4. Kết luận
- Structured conversation state đã được tích hợp end-to-end vào pipeline `/ask`.
- Truy vấn follow-up dạng tham chiếu ngắn (`môn này`) đã route đúng nhờ state thay vì phụ thuộc hoàn toàn vào memory text thô.
- Regression tests hiện không phát sinh lỗi.

## 5. Cảnh báo còn lại (không chặn chức năng)
- Một số warning dependency/deprecation vẫn xuất hiện trong test runtime:
  - `google.generativeai` deprecation warning
  - FastAPI `on_event` deprecation warning
- Đây là vấn đề kỹ thuật nền, không ảnh hưởng logic state mới.
