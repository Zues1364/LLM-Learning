import copy
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = 1
MAX_TRACKED_CODES = 8
MAX_TRACKED_TEACHERS = 4
MAX_TRACKED_TOPICS = 16


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_conversation_state() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "turn_index": 0,
        "last_query": "",
        "last_answer": "",
        "active_intent": "",
        "entities": {
            "course_codes": [],
            "teacher_names": [],
            "semester_code": None,
            "topic_keywords": [],
        },
        "referents": {
            "last_subject_codes": [],
            "last_teacher_names": [],
            "last_topic": "",
            "last_raw_query": "",
        },
        "updated_at": _utc_now_iso(),
    }


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("đ", "d").replace("Đ", "d")
    replacements = {
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
    }
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    lowered = (
        unicodedata.normalize("NFKD", lowered)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", " ", lowered).strip()


def _merge_unique(items: List[str], limit: int) -> List[str]:
    merged: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        if value in merged:
            continue
        merged.append(value)
    return merged[:limit]


def _extract_course_codes(text: str) -> List[str]:
    codes = re.findall(r"\b[A-Z]{2,6}\d{4}[A-Z]?\b", str(text or "").upper())
    return _merge_unique(codes, MAX_TRACKED_CODES)


def _extract_semester_code(text: str) -> str | None:
    match = re.search(r"\b(2\d{2})\b", str(text or ""))
    return match.group(1) if match else None


def _infer_intent(query: str) -> str:
    norm = _normalize_text(query)
    if any(token in norm for token in ("thay ", "co ", "giang vien", "gv ")):
        return "teacher_lookup"
    if any(token in norm for token in ("gpa", "tin chi", "tot nghiep", "ra truong")):
        return "advising"
    if any(token in norm for token in ("lich", "ca ", "phong", "thu ", "mo lop", "tkb")):
        return "schedule"
    if any(token in norm for token in ("ielts", "toeic", "ngoai ngu")):
        return "language_policy"
    return "general"


def _extract_teacher_names(text: str) -> List[str]:
    candidates: List[str] = []
    patterns = (
        r"(?:gv|giang vien|thay|co|thầy|cô)\s+([A-ZÀ-Ỵ][A-Za-zÀ-Ỵà-ỵ]+(?:\s+[A-ZÀ-Ỵ][A-Za-zÀ-Ỵà-ỵ]+){1,4})",
    )
    raw_text = str(text or "")
    for pattern in patterns:
        for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
            name = (match.group(1) or "").strip()
            if len(name) < 4:
                continue
            candidates.append(name)
    return _merge_unique(candidates, MAX_TRACKED_TEACHERS)


def _extract_topics(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-Ỵà-ỵ0-9]+", str(query or ""))
    stopwords = {
        "toi",
        "ban",
        "la",
        "nay",
        "kia",
        "do",
        "co",
        "khong",
        "gi",
        "nhu",
        "the",
        "nao",
        "moi",
        "vay",
        "cua",
        "voi",
        "ve",
        "cho",
        "can",
        "muon",
    }
    keywords: List[str] = []
    for token in tokens:
        t = _normalize_text(token)
        if len(t) < 3 or t in stopwords or t.isdigit():
            continue
        keywords.append(t)
    return _merge_unique(keywords, MAX_TRACKED_TOPICS)


def resolve_query_with_state(query: str, state: Dict[str, Any]) -> Dict[str, Any]:
    raw_query = str(query or "").strip()
    resolved_query = raw_query
    applied: List[str] = []

    normalized = _normalize_text(raw_query)
    referents = (state or {}).get("referents") or {}
    entities = (state or {}).get("entities") or {}
    last_subject_codes = referents.get("last_subject_codes") or entities.get("course_codes") or []
    last_subject_codes = [str(code).upper() for code in last_subject_codes if code]
    last_teacher_names = referents.get("last_teacher_names") or entities.get("teacher_names") or []
    semester_code = entities.get("semester_code")
    previous_query = str((state or {}).get("last_query") or "").strip()

    subject_ref_markers = (
        "mon nay",
        "mon do",
        "mon kia",
        "mon ay",
        "cac mon nay",
        "mon nay ",
    )
    if any(marker in normalized for marker in subject_ref_markers) and last_subject_codes:
        primary_code = last_subject_codes[0]
        resolved_query = re.sub(
            r"(?i)\bmôn này\b|\bmôn đó\b|\bmôn kia\b|\bmôn ấy\b|\bcác môn này\b",
            f"môn {primary_code}",
            resolved_query,
        )
        if "môn " + primary_code not in resolved_query:
            resolved_query = f"{resolved_query} (mon {primary_code})"
        applied.append(f"subject:{primary_code}")

    teacher_ref_markers = ("thay nay", "co nay", "giang vien nay", "gv nay")
    if any(marker in normalized for marker in teacher_ref_markers) and last_teacher_names:
        teacher_name = str(last_teacher_names[0]).strip()
        resolved_query = re.sub(
            r"(?i)\bthầy này\b|\bcô này\b|\bgiảng viên này\b|\bgv này\b",
            teacher_name,
            resolved_query,
        )
        applied.append(f"teacher:{teacher_name}")

    semester_ref_markers = ("ky nay", "hoc ky nay")
    if any(marker in normalized for marker in semester_ref_markers) and semester_code:
        if str(semester_code) not in resolved_query:
            resolved_query = f"{resolved_query} (hoc ky {semester_code})"
            applied.append(f"semester:{semester_code}")

    if any(marker in normalized for marker in ("toi vua hoi gi", "cau truoc", "cau hoi truoc")) and previous_query:
        resolved_query = f"Câu hỏi trước của tôi là: {previous_query}"
        applied.append("previous_query")

    return {
        "raw_query": raw_query,
        "resolved_query": resolved_query.strip(),
        "applied_referents": applied,
    }


def update_state_after_turn(
    previous_state: Dict[str, Any] | None,
    raw_query: str,
    resolved_query: str,
    answer: str,
    planner_source: str = "",
    planner_context: str = "",
    selected_program_id: str | None = None,
) -> Dict[str, Any]:
    state = copy.deepcopy(previous_state or default_conversation_state())

    entities = state.setdefault("entities", {})
    referents = state.setdefault("referents", {})

    codes_from_turn = _extract_course_codes(" ".join([raw_query, resolved_query, answer, planner_context]))
    teacher_names = _extract_teacher_names(" ".join([raw_query, answer, planner_context]))
    semester_code = _extract_semester_code(" ".join([raw_query, resolved_query, planner_context]))
    topics = _extract_topics(raw_query)

    old_codes = entities.get("course_codes") or []
    old_teachers = entities.get("teacher_names") or []
    old_topics = entities.get("topic_keywords") or []

    merged_codes = _merge_unique(codes_from_turn + old_codes, MAX_TRACKED_CODES)
    merged_teachers = _merge_unique(teacher_names + old_teachers, MAX_TRACKED_TEACHERS)
    merged_topics = _merge_unique(topics + old_topics, MAX_TRACKED_TOPICS)

    entities["course_codes"] = merged_codes
    entities["teacher_names"] = merged_teachers
    entities["topic_keywords"] = merged_topics
    entities["semester_code"] = semester_code or entities.get("semester_code")

    referents["last_subject_codes"] = _merge_unique(codes_from_turn, MAX_TRACKED_CODES) or referents.get("last_subject_codes", [])
    referents["last_teacher_names"] = _merge_unique(teacher_names, MAX_TRACKED_TEACHERS) or referents.get("last_teacher_names", [])
    referents["last_topic"] = topics[0] if topics else (referents.get("last_topic") or "")
    referents["last_raw_query"] = str(raw_query or "").strip()

    state["schema_version"] = SCHEMA_VERSION
    state["turn_index"] = int(state.get("turn_index") or 0) + 1
    state["last_query"] = str(raw_query or "").strip()
    state["last_answer"] = str(answer or "").strip()
    state["active_intent"] = _infer_intent(raw_query)
    state["last_resolved_query"] = str(resolved_query or "").strip()
    state["last_planner_source"] = str(planner_source or "").strip()
    state["selected_program_id"] = str(selected_program_id or state.get("selected_program_id") or "").strip()
    state["updated_at"] = _utc_now_iso()

    return state
