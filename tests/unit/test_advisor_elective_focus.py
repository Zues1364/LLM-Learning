from src.mcp_server.server import (
    _format_subject_name_vi_en,
    _postprocess_advisor_answer_text,
    _query_targets_elective_opened_not_taken,
    _render_elective_opened_not_taken_text,
)


def test_query_targets_elective_opened_not_taken_detects_intent():
    q = "ve cac mon tu chon trong chuong trinh dao tao co mon nao o ki nay mo lop ma toi chua hoc khong"
    assert _query_targets_elective_opened_not_taken(q) is True


def test_render_elective_opened_not_taken_filters_completed_courses():
    ctx = {
        "credit_summary": {"total_missing_credits": 21},
        "missing_subjects": {
            "credit_analysis": [
                {"block_name": "Khoi kien thuc nganh - hoc phan tu chon", "block_type": "elective", "missing_credits": 5},
            ]
        },
        "transcript_json": {
            "semesters": [
                {"subjects": [{"code": "INT3306"}]},
            ]
        },
        "elective_catalog": {
            "opened": [
                {"code": "INT3306", "name": "Phat trien ung dung Web", "credits": 3, "group": "V.2.1"},
                {"code": "INT3323", "name": "Phat trien IoT", "credits": 3, "group": "V.2.1"},
            ]
        },
    }

    text = _render_elective_opened_not_taken_text(ctx)
    assert "INT3323" in text
    assert "INT3306" not in text


def test_format_subject_name_vi_en_wraps_english_suffix():
    assert _format_subject_name_vi_en("Lập trình nâng cao Advanced Programming") == "Lập trình nâng cao (Advanced Programming)"
    assert _format_subject_name_vi_en("Các chuyên đề trong KHMT Special Problems in Computer Science") == "Các chuyên đề trong KHMT (Special Problems in Computer Science)"
    assert _format_subject_name_vi_en("Tối ưu hóa Optimization") == "Tối ưu hóa (Optimization)"
    assert _format_subject_name_vi_en("Mật mã KHMT") == "Mật mã KHMT"


def test_postprocess_keeps_space_before_credit_parentheses():
    raw = "- INT3306 - Ten mon (Web Application Development)(3 tin chi), nhom V.2.1"
    out = _postprocess_advisor_answer_text(raw)
    assert "(Web Application Development) (3 tin chi)" in out
