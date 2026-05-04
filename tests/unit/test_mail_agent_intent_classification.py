import sys
from pathlib import Path
import sqlite3

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mail_agent as mail_agent_mod
from mail_agent import MailAgentService


def _build_service(monkeypatch):
    monkeypatch.setenv("MAIL_INTENT_CLASSIFIER_MODE", "rule_only")
    monkeypatch.setenv(
        "MAIL_RELEVANCE_KEYWORDS",
        "mo lop,dang ky hoc phan,hoc ky,thoi khoa bieu,tkb",
    )
    monkeypatch.setenv(
        "MAIL_NEGATIVE_TOKENS",
        "job fair,hoi cho viec lam,tuyen dung,su kien,workshop",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return MailAgentService()


def test_schedule_email_is_classified_as_relevant(monkeypatch):
    svc = _build_service(monkeypatch)
    relevant, reasons, classification = svc._match_relevance(
        sender_email="daotao_dhcn@vnu.edu.vn",
        subject="Thời khóa biểu học kỳ II năm học 2025-2026",
        snippet="Phòng Đào tạo gửi phụ lục thời khóa biểu và lịch mở cổng đăng ký học phần.",
        body="Sinh viên xem phụ lục kèm công văn để đăng ký học phần đúng hạn.",
        whitelist=["vnu.edu.vn"],
        attachment_count=2,
        link_count=1,
    )
    assert relevant is True
    assert classification["is_relevant"] is True
    assert classification["intent"] in {"schedule_update", "registration_notice", "academic_notice"}
    assert reasons


def test_save_mail_connection_first_insert_sets_connected_at(monkeypatch, tmp_path):
    monkeypatch.setattr(mail_agent_mod, "MAIL_DB_PATH", tmp_path / "memory.db")
    monkeypatch.setenv("APP_SESSION_SECRET", "test-secret")
    svc = MailAgentService()

    svc._save_mail_connection(
        user_id="u1",
        gmail_email="student@vnu.edu.vn",
        refresh_token="refresh-token",
        access_token="access-token",
        access_expiry=123,
        scope="https://www.googleapis.com/auth/gmail.readonly",
    )

    connection = svc._get_mail_connection("u1")
    assert connection is not None
    assert connection["refresh_token"] == "refresh-token"
    assert connection["connected_at"]

    with sqlite3.connect(tmp_path / "memory.db") as conn:
        row = conn.execute("SELECT connected_at FROM mail_connections WHERE user_id = 'u1'").fetchone()
    assert row is not None
    assert row[0]


def test_job_fair_email_is_rejected_even_if_keyword_exists(monkeypatch):
    svc = _build_service(monkeypatch)
    relevant, _reasons, classification = svc._match_relevance(
        sender_email="ctsv_dhcn@vnu.edu.vn",
        subject="Triệu tập sinh viên tham dự UET JOB FAIR 2026",
        snippet="Thông báo sự kiện, có nhắc thời khóa biểu hoạt động sự kiện.",
        body="Đây là mail sự kiện tuyển dụng, không phải mở lớp học phần.",
        whitelist=["vnu.edu.vn"],
        attachment_count=0,
        link_count=1,
    )
    assert relevant is False
    assert classification["is_relevant"] is False
    assert classification["intent"] == "other"


def test_schedule_change_email_is_classified_as_relevant(monkeypatch):
    monkeypatch.setenv("MAIL_INTENT_CLASSIFIER_MODE", "rule_only")
    monkeypatch.setenv(
        "MAIL_RELEVANCE_KEYWORDS",
        "mo lop,dang ky hoc phan,hoc ky,thoi khoa bieu,tkb,thay doi lich hoc,tam ngung cap dien,dieu chinh ke hoach giang day",
    )
    monkeypatch.setenv(
        "MAIL_NEGATIVE_TOKENS",
        "job fair,hoi cho viec lam,tuyen dung,su kien,workshop",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    svc = MailAgentService()

    relevant, reasons, classification = svc._match_relevance(
        sender_email="daotao_dhcn@vnu.edu.vn",
        subject="THÔNG BÁO TẠM NGỪNG CẤP ĐIỆN NGÀY 18/03/2026 TẠI KHU GD KIỀU MAI",
        snippet="các lớp học tại giảng đường Kiều Mai theo dõi thông báo điều chỉnh kế hoạch giảng dạy",
        body="Khu giảng đường Kiều Mai tạm ngừng cấp điện. Các lớp học theo dõi thông báo điều chỉnh kế hoạch giảng dạy từ GV trong nhóm lớp.",
        whitelist=["vnu.edu.vn", "uet.edu.vn"],
        attachment_count=1,
        link_count=0,
    )

    assert relevant is True
    assert classification["is_relevant"] is True
    assert classification["intent"] in {"schedule_update", "academic_notice"}
    assert reasons
