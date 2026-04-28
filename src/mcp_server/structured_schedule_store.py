import hashlib
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import normalize_for_match, process_pdf


COURSE_CODE_RE = re.compile(r"\b(?:UET\.)?([A-Z]{2,4}\d{3,4}[A-Z]?)\b", re.IGNORECASE)
ROOM_RE = re.compile(r"\b(?:\d{1,4}-[A-Z0-9]{1,3}|[1-9]-G[1-9])\b", re.IGNORECASE)
TEACHER_NAME_ALLOWED_RE = re.compile(r"^[A-Za-zÀ-ỹ\s.'-]+$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_code(code: str) -> str:
    if not code:
        return ""
    return str(code).upper().replace("UET.", "").replace(" ", "").strip()


def _normalize_text(text: str) -> str:
    return normalize_for_match(str(text or ""))


def _compact_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize_text(text))


def _to_day_label(day_token: str) -> str:
    day = str(day_token or "").strip()
    if day in {"2", "3", "4", "5", "6", "7"}:
        return f"Thứ {day}"
    if day == "8":
        return "Chủ nhật"
    return ""


def _dedupe_word_pattern(words: List[str]) -> List[str]:
    if not words:
        return words
    n = len(words)
    for span in range(2, min(7, n // 2 + 1)):
        if n % span != 0:
            continue
        unit = words[:span]
        if unit * (n // span) == words:
            return unit
    return words


def _letters_only(token: str) -> str:
    return "".join(ch for ch in str(token or "") if ch.isalpha())


def _collapse_adjacent_duplicate_tokens(words: List[str]) -> List[str]:
    collapsed: List[str] = []
    for word in words:
        normalized = normalize_for_match(word)
        if collapsed and normalize_for_match(collapsed[-1]) == normalized:
            continue
        collapsed.append(word)
    return collapsed


def _collapse_repeated_prefix(words: List[str]) -> List[str]:
    if len(words) < 4:
        return words
    # Handle common OCR duplication at prefix, e.g. "Ngô Thái Ngô Thái Hà ..."
    for span in range(2, min(6, (len(words) // 2) + 1)):
        left = [normalize_for_match(w) for w in words[:span]]
        right = [normalize_for_match(w) for w in words[span : 2 * span]]
        if left == right:
            return words[:span] + words[2 * span :]
    return words


def _collapse_repeated_ngrams(words: List[str]) -> List[str]:
    if len(words) < 4:
        return words
    normalized_words = list(words)
    changed = True
    while changed:
        changed = False
        max_span = min(5, len(normalized_words) // 2)
        for span in range(max_span, 1, -1):
            i = 0
            while i + 2 * span <= len(normalized_words):
                left = [normalize_for_match(w) for w in normalized_words[i : i + span]]
                right = [normalize_for_match(w) for w in normalized_words[i + span : i + 2 * span]]
                if left and left == right:
                    del normalized_words[i + span : i + 2 * span]
                    changed = True
                    continue
                i += 1
    return normalized_words


def _sanitize_teacher_name(name: str) -> str:
    candidate = re.sub(r"\s+", " ", str(name or "")).strip(" -|")
    if not candidate:
        return ""
    if not TEACHER_NAME_ALLOWED_RE.fullmatch(candidate):
        return ""
    if re.search(r"\d", candidate):
        return ""
    if any(k in _normalize_text(candidate) for k in ("hoc 1 ca", "hoc 2 ca", "thi dot", "dot ")):
        return ""

    words = [w for w in candidate.split() if w]
    if len(words) < 2:
        return ""
    words = _dedupe_word_pattern(words)
    words = _collapse_adjacent_duplicate_tokens(words)
    prev_words: List[str] = []
    while words and words != prev_words:
        prev_words = list(words)
        words = _collapse_repeated_prefix(words)
        words = _collapse_repeated_ngrams(words)
        words = _collapse_adjacent_duplicate_tokens(words)
    if len(words) < 2:
        return ""

    letter_tokens = [_letters_only(word) for word in words if _letters_only(word)]
    if len(letter_tokens) < 2:
        return ""
    single_char_count = sum(1 for token in letter_tokens if len(token) <= 1)
    if single_char_count > 0 and single_char_count / max(1, len(letter_tokens)) >= 0.4:
        return ""
    if sum(1 for token in letter_tokens if len(token) >= 2) < 2:
        return ""

    normalized = " ".join(words).strip(" ,;")
    if not normalized:
        return ""
    return normalized


def _parse_teacher_list(text: str) -> List[str]:
    chunk = str(text or "").strip()
    if not chunk:
        return []
    chunk = re.sub(r"\s+", " ", chunk)
    chunk = (
        chunk.replace(" và ", "+")
        .replace("/", "+")
        .replace(";", "+")
        .replace(" , ", "+")
        .replace(",", "+")
    )
    teachers: List[str] = []
    for part in chunk.split("+"):
        candidate = part.strip(" -|")
        if not candidate:
            continue
        normalized = _sanitize_teacher_name(candidate)
        if normalized and normalized not in teachers:
            teachers.append(normalized)
    return teachers


def _infer_semester_label(source_name: str, text: str) -> str:
    norm = _normalize_text(f"{source_name} {text[:2000]}")
    year_match = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2})", norm)
    year_label = ""
    if year_match:
        year_label = f"{year_match.group(1)}-{year_match.group(2)}"
    if "hkiii" in norm or "hoc ky iii" in norm or "hoc ky 3" in norm:
        return f"HKIII {year_label}".strip()
    if "hkii" in norm or "hoc ky ii" in norm or "hoc ky 2" in norm:
        return f"HKII {year_label}".strip()
    if "hki" in norm or "hoc ky i" in norm or "hoc ky 1" in norm:
        return f"HKI {year_label}".strip()
    return year_label or "Unknown"


def _semester_query_patterns(semester_query: str) -> List[str]:
    raw = str(semester_query or "").strip()
    if not raw:
        return []

    code_match = re.search(r"\b(\d{3})\b", raw)
    if code_match:
        code = code_match.group(1)
        year_start = 2000 + int(code[:2])
        year_end = year_start + 1
        term_digit = code[2]
        if term_digit == "1":
            return [f"hki{year_start}{year_end}", "hki"]
        if term_digit == "2":
            # Policy mapping: x52 covers both semester 2 and summer semester.
            return [f"hkii{year_start}{year_end}", f"hkiii{year_start}{year_end}", "hkii", "hkiii"]
        return []

    norm = _normalize_text(raw)
    norm_compact = _compact_norm(norm)
    term_tokens: List[str] = []
    if any(tok in norm_compact for tok in ("hkiii", "hockyiii", "hocky3", "kyhe", "kihe", "hoche", "summer")):
        term_tokens = ["hkiii"]
    elif any(tok in norm_compact for tok in ("hkii", "hockyii", "hocky2", "ky2", "ki2", "semester2")):
        term_tokens = ["hkii"]
    elif any(tok in norm_compact for tok in ("hki", "hockyi", "hocky1", "ky1", "ki1", "semester1")):
        term_tokens = ["hki"]

    year_match = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", norm)
    patterns: List[str] = []
    if year_match:
        year_start = int(year_match.group(1))
        year_end = int(year_match.group(2))
        if year_end == year_start + 1:
            if term_tokens:
                patterns.extend(f"{term}{year_start}{year_end}" for term in term_tokens)
            else:
                patterns.append(f"{year_start}{year_end}")
    if term_tokens:
        patterns.extend(term_tokens)
    if not patterns and norm_compact:
        patterns.append(norm_compact)
    return list(dict.fromkeys([p for p in patterns if p]))


def _semester_matches_query(row_semester: str, semester_query: Optional[str]) -> bool:
    if not semester_query:
        return True
    row_token = _compact_norm(str(row_semester or ""))
    if not row_token:
        return False
    patterns = _semester_query_patterns(str(semester_query or ""))
    if not patterns:
        return _compact_norm(str(semester_query or "")) in row_token
    for pattern in patterns:
        compact_pattern = _compact_norm(pattern)
        if not compact_pattern:
            continue
        if compact_pattern in {"hki", "hkii", "hkiii"}:
            # Avoid accidental prefix matches: "hki" must not match "hkii/hkiii", etc.
            if re.search(rf"{compact_pattern}(?:20\d{{2}}|$)", row_token):
                return True
            continue
        if compact_pattern in row_token:
            return True
    return False


def _line_hash_payload(row: Dict[str, Any]) -> str:
    ordered = [
        str(row.get("semester") or "").strip(),
        _normalize_code(str(row.get("subject_code") or "")),
        _normalize_text(str(row.get("subject_name_vi") or "")),
        _normalize_text(str(row.get("subject_name_en") or "")),
        _normalize_text(str(row.get("class_code") or "")),
        _normalize_text(str(row.get("teacher_name") or "")),
        _normalize_text(str(row.get("day_of_week") or "")),
        _normalize_text(str(row.get("slot") or "")),
        _normalize_text(str(row.get("room") or "")),
        _normalize_text(str(row.get("week_note") or "")),
    ]
    raw = "||".join(ordered)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _to_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        number = int(str(value).strip())
    except Exception:
        return None
    if number <= 0:
        return None
    return number


class StructuredScheduleStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schedule_rows (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        semester TEXT,
                        subject_code TEXT,
                        subject_name_vi TEXT,
                        subject_name_en TEXT,
                        class_code TEXT,
                        teacher_name TEXT,
                        day_of_week TEXT,
                        slot TEXT,
                        room TEXT,
                        week_note TEXT,
                        source_file TEXT,
                        source_page INTEGER,
                        source_line INTEGER,
                        row_hash TEXT UNIQUE
                    );

                    CREATE TABLE IF NOT EXISTS teacher_alias (
                        alias_norm TEXT,
                        teacher_name_canonical TEXT,
                        UNIQUE(alias_norm, teacher_name_canonical)
                    );

                    CREATE TABLE IF NOT EXISTS course_alias (
                        alias_norm TEXT,
                        subject_code TEXT,
                        subject_name_vi TEXT,
                        UNIQUE(alias_norm, subject_code)
                    );

                    CREATE TABLE IF NOT EXISTS ingest_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_file TEXT,
                        checksum TEXT,
                        ingested_at TEXT,
                        row_count INTEGER
                    );

                    CREATE INDEX IF NOT EXISTS idx_schedule_rows_subject_code ON schedule_rows(subject_code);
                    CREATE INDEX IF NOT EXISTS idx_schedule_rows_teacher_name ON schedule_rows(teacher_name);
                    CREATE INDEX IF NOT EXISTS idx_schedule_rows_source_file ON schedule_rows(source_file);
                    CREATE INDEX IF NOT EXISTS idx_teacher_alias_alias_norm ON teacher_alias(alias_norm);
                    CREATE INDEX IF NOT EXISTS idx_course_alias_alias_norm ON course_alias(alias_norm);
                    """
                )
                existing_columns = {
                    str(row["name"] or "").strip()
                    for row in conn.execute("PRAGMA table_info(schedule_rows)").fetchall()
                }
                if "source_page" not in existing_columns:
                    conn.execute("ALTER TABLE schedule_rows ADD COLUMN source_page INTEGER")
                if "source_line" not in existing_columns:
                    conn.execute("ALTER TABLE schedule_rows ADD COLUMN source_line INTEGER")
                conn.commit()
            finally:
                conn.close()

    def ingest_schedule_files(self, file_paths: Sequence[Path], force: bool = False) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        changed = False
        with self._lock:
            conn = self._connect()
            try:
                existing_paths = sorted([Path(p) for p in file_paths if p and Path(p).exists()], key=lambda p: p.name)
                current_source_files = {path.name for path in existing_paths}

                # Prune rows/runs for schedule sources that no longer exist on disk.
                existing_sources = conn.execute(
                    "SELECT DISTINCT source_file FROM schedule_rows"
                ).fetchall()
                stale_sources = [
                    str(row["source_file"] or "").strip()
                    for row in existing_sources
                    if str(row["source_file"] or "").strip()
                    and str(row["source_file"] or "").strip() not in current_source_files
                ]
                if stale_sources:
                    for source_name in stale_sources:
                        conn.execute("DELETE FROM schedule_rows WHERE source_file = ?", (source_name,))
                        conn.execute("DELETE FROM ingest_runs WHERE source_file = ?", (source_name,))
                        summary.append(
                            {
                                "source_file": source_name,
                                "checksum": None,
                                "row_count": 0,
                                "skipped": False,
                                "removed": True,
                            }
                        )
                    changed = True

                for path in existing_paths:
                    checksum = self._compute_file_checksum(path)
                    prev = conn.execute(
                        "SELECT checksum, row_count FROM ingest_runs WHERE source_file = ? ORDER BY id DESC LIMIT 1",
                        (path.name,),
                    ).fetchone()
                    if prev and str(prev["checksum"]) == checksum and not force:
                        summary.append(
                            {
                                "source_file": path.name,
                                "checksum": checksum,
                                "row_count": int(prev["row_count"] or 0),
                                "skipped": True,
                            }
                        )
                        continue

                    parsed_rows = self._parse_schedule_pdf(path)
                    conn.execute("DELETE FROM schedule_rows WHERE source_file = ?", (path.name,))
                    for row in parsed_rows:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO schedule_rows (
                                semester, subject_code, subject_name_vi, subject_name_en,
                                class_code, teacher_name, day_of_week, slot, room,
                                week_note, source_file, source_page, source_line, row_hash
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                row.get("semester"),
                                row.get("subject_code"),
                                row.get("subject_name_vi"),
                                row.get("subject_name_en"),
                                row.get("class_code"),
                                row.get("teacher_name"),
                                row.get("day_of_week"),
                                row.get("slot"),
                                row.get("room"),
                                row.get("week_note"),
                                row.get("source_file"),
                                row.get("source_page"),
                                row.get("source_line"),
                                row.get("row_hash"),
                            ),
                        )

                    conn.execute(
                        """
                        INSERT INTO ingest_runs (source_file, checksum, ingested_at, row_count)
                        VALUES (?, ?, ?, ?)
                        """,
                        (path.name, checksum, _utc_now_iso(), len(parsed_rows)),
                    )
                    changed = True
                    summary.append(
                        {
                            "source_file": path.name,
                            "checksum": checksum,
                            "row_count": len(parsed_rows),
                            "skipped": False,
                        }
                    )

                if changed:
                    self._rebuild_aliases(conn)

                conn.commit()
            finally:
                conn.close()
        return summary

    def resolve_course_alias(self, query: str) -> Dict[str, Any]:
        normalized_query = _normalize_text(query)
        code_match = COURSE_CODE_RE.search(str(query or "").upper())
        direct_code = _normalize_code(code_match.group(1)) if code_match else ""

        with self._lock:
            conn = self._connect()
            try:
                if direct_code:
                    row = conn.execute(
                        """
                        SELECT subject_code, COALESCE(NULLIF(subject_name_vi, ''), MIN(subject_name_vi)) AS subject_name_vi
                        FROM schedule_rows
                        WHERE subject_code = ?
                        LIMIT 1
                        """,
                        (direct_code,),
                    ).fetchone()
                    if row:
                        return {
                            "matched_subject": {
                                "subject_code": row["subject_code"],
                                "subject_name_vi": row["subject_name_vi"] or "",
                            },
                            "confidence": 1.0,
                            "candidates": [
                                {
                                    "subject_code": row["subject_code"],
                                    "subject_name_vi": row["subject_name_vi"] or "",
                                    "score": 1.0,
                                }
                            ],
                        }

                rows = conn.execute(
                    """
                    SELECT
                        ca.subject_code AS subject_code,
                        COALESCE(NULLIF(ca.subject_name_vi, ''), MIN(sr.subject_name_vi)) AS subject_name_vi,
                        MAX(CASE WHEN ca.alias_norm = ? THEN 1 ELSE 0 END) AS exact_alias,
                        MAX(CASE WHEN ca.alias_norm LIKE ? THEN 1 ELSE 0 END) AS partial_alias,
                        COUNT(*) AS alias_hits
                    FROM course_alias ca
                    LEFT JOIN schedule_rows sr ON sr.subject_code = ca.subject_code
                    WHERE ca.alias_norm = ?
                       OR ca.alias_norm LIKE ?
                       OR ? LIKE '%' || ca.alias_norm || '%'
                    GROUP BY ca.subject_code
                    ORDER BY exact_alias DESC, partial_alias DESC, alias_hits DESC, ca.subject_code ASC
                    LIMIT 30
                    """,
                    (
                        normalized_query,
                        f"%{normalized_query}%",
                        normalized_query,
                        f"%{normalized_query}%",
                        normalized_query,
                    ),
                ).fetchall()
            finally:
                conn.close()

        candidates: List[Dict[str, Any]] = []
        stop_tokens = {
            "mon",
            "hoc",
            "phan",
            "ky",
            "ki",
            "nay",
            "co",
            "nhung",
            "ai",
            "day",
            "giang",
            "vien",
            "lop",
            "vao",
            "hom",
            "nao",
            "va",
            "voi",
            "cung",
        }
        query_tokens = {tok for tok in normalized_query.split() if len(tok) >= 3 and tok not in stop_tokens}
        for row in rows:
            score = 0.25
            subject_tokens = {
                tok
                for tok in _normalize_text(str(row["subject_name_vi"] or "")).split()
                if len(tok) >= 3 and tok not in stop_tokens
            }
            overlap = len(query_tokens & subject_tokens)
            precision = overlap / max(1, len(subject_tokens))
            recall = overlap / max(1, len(query_tokens)) if query_tokens else 0.0
            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)

            if int(row["exact_alias"] or 0):
                score = 0.95
            elif int(row["partial_alias"] or 0):
                score = max(score, 0.68)
            else:
                if overlap >= 3:
                    score = max(score, 0.84)
                elif overlap >= 2:
                    score = max(score, 0.72)
                elif overlap == 1:
                    score = max(score, 0.6)

            if f1 > 0:
                score = max(score, 0.5 + 0.45 * f1)

            candidates.append(
                {
                    "subject_code": row["subject_code"],
                    "subject_name_vi": row["subject_name_vi"] or "",
                    "score": round(score, 2),
                }
            )

        if not candidates:
            return {"matched_subject": None, "confidence": 0.0, "candidates": []}
        candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("subject_code") or "")))
        return {
            "matched_subject": {
                "subject_code": candidates[0]["subject_code"],
                "subject_name_vi": candidates[0]["subject_name_vi"],
            },
            "confidence": candidates[0]["score"],
            "candidates": candidates,
        }

    def get_teachers_by_subject(self, subject_code: str, semester: Optional[str] = None) -> Dict[str, Any]:
        code = _normalize_code(subject_code)
        if not code:
            return {"matched_subject": None, "confidence": 0.0, "teachers": [], "rows": [], "source_files": [], "coverage_note": "Thiếu mã môn hợp lệ."}

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT semester, subject_code, subject_name_vi, class_code, teacher_name, day_of_week, slot, room, week_note, source_file, source_page, source_line
                    FROM schedule_rows
                    WHERE subject_code = ?
                    ORDER BY class_code, day_of_week, slot, teacher_name
                    """,
                    (code,),
                ).fetchall()
            finally:
                conn.close()

        materialized = [self._row_to_dict(row) for row in rows]
        if semester:
            materialized = [
                row for row in materialized if _semester_matches_query(str(row.get("semester") or ""), semester)
            ]
        materialized = [row for row in materialized if str(row.get("teacher_name") or "").strip()]
        materialized = self._dedupe_rows(materialized)
        teachers = sorted({str(r.get("teacher_name") or "").strip() for r in materialized if str(r.get("teacher_name") or "").strip()})
        source_files = sorted({str(r.get("source_file") or "").strip() for r in materialized if str(r.get("source_file") or "").strip()})

        subject_name = ""
        if materialized:
            subject_name = str(materialized[0].get("subject_name_vi") or "")

        coverage_note = (
            f"Tìm thấy {len(teachers)} giảng viên từ {len(materialized)} dòng lịch."
            if materialized
            else "Không tìm thấy dòng lịch phù hợp cho môn này."
        )
        return {
            "matched_subject": {"subject_code": code, "subject_name_vi": subject_name},
            "confidence": 1.0 if materialized else 0.0,
            "teachers": teachers,
            "rows": materialized,
            "source_files": source_files,
            "coverage_note": coverage_note,
        }

    def get_classes_by_teacher(self, teacher_name: str, semester: Optional[str] = None) -> Dict[str, Any]:
        norm_teacher = _normalize_text(teacher_name)
        if not norm_teacher:
            return {"matched_teacher": None, "confidence": 0.0, "rows": [], "source_files": [], "coverage_note": "Thiếu tên giảng viên."}

        with self._lock:
            conn = self._connect()
            try:
                exact_alias_rows = conn.execute(
                    """
                    SELECT DISTINCT teacher_name_canonical
                    FROM teacher_alias
                    WHERE alias_norm = ?
                    LIMIT 25
                    """,
                    (norm_teacher,),
                ).fetchall()
                alias_rows = conn.execute(
                    """
                    SELECT DISTINCT teacher_name_canonical
                    FROM teacher_alias
                    WHERE alias_norm LIKE ? OR ? LIKE '%' || alias_norm || '%'
                    LIMIT 25
                    """,
                    (f"%{norm_teacher}%", norm_teacher),
                ).fetchall()
            finally:
                conn.close()

        canonical_names = [
            str(r["teacher_name_canonical"]).strip()
            for r in exact_alias_rows
            if str(r["teacher_name_canonical"] or "").strip()
        ]
        if not canonical_names:
            fuzzy_candidates = [
                str(r["teacher_name_canonical"]).strip()
                for r in alias_rows
                if str(r["teacher_name_canonical"] or "").strip()
            ]
            if fuzzy_candidates:
                query_tokens = {tok for tok in norm_teacher.split() if len(tok) >= 2}
                scored: List[Tuple[float, str]] = []
                for candidate in fuzzy_candidates:
                    candidate_norm = _normalize_text(candidate)
                    candidate_tokens = {tok for tok in candidate_norm.split() if len(tok) >= 2}
                    if not query_tokens or not candidate_tokens:
                        continue
                    overlap = len(query_tokens & candidate_tokens)
                    precision = overlap / max(1, len(candidate_tokens))
                    recall = overlap / max(1, len(query_tokens))
                    score = (precision + recall) / 2.0
                    scored.append((score, candidate))
                scored.sort(key=lambda x: (-x[0], x[1]))
                if scored:
                    top_score = scored[0][0]
                    canonical_names = [name for score, name in scored if score >= max(0.65, top_score - 0.05)]

        if not canonical_names:
            canonical_names = [teacher_name.strip()]

        rows = self.get_schedule_rows(subject_code=None, teacher_name="|".join(canonical_names), semester=semester).get("rows") or []
        source_files = sorted({str(r.get("source_file") or "").strip() for r in rows if str(r.get("source_file") or "").strip()})
        coverage_note = (
            f"Tìm thấy {len(rows)} dòng lịch cho {len(canonical_names)} tên giảng viên khớp."
            if rows
            else "Không tìm thấy lớp học cho giảng viên này."
        )
        confidence = 0.9 if rows else 0.0
        return {
            "matched_teacher": {
                "query": teacher_name,
                "canonical_names": canonical_names,
            },
            "confidence": confidence,
            "rows": rows,
            "source_files": source_files,
            "coverage_note": coverage_note,
        }

    def get_schedule_rows(
        self,
        subject_code: Optional[str] = None,
        teacher_name: Optional[str] = None,
        semester: Optional[str] = None,
    ) -> Dict[str, Any]:
        filters: List[str] = []
        params: List[Any] = []

        if subject_code:
            code = _normalize_code(subject_code)
            if code:
                filters.append("subject_code = ?")
                params.append(code)

        teacher_names: List[str] = []
        if teacher_name:
            raw_names = [n.strip() for n in str(teacher_name).split("|") if n.strip()]
            teacher_names.extend(raw_names)
        if teacher_names:
            placeholders = ",".join(["?"] * len(teacher_names))
            filters.append(f"teacher_name IN ({placeholders})")
            params.extend(teacher_names)

        where_clause = " AND ".join(filters) if filters else "1=1"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT semester, subject_code, subject_name_vi, class_code, teacher_name, day_of_week, slot, room, week_note, source_file, source_page, source_line
                    FROM schedule_rows
                    WHERE {where_clause}
                    ORDER BY subject_code, class_code, day_of_week, slot, teacher_name
                    LIMIT 500
                    """,
                    tuple(params),
                ).fetchall()
            finally:
                conn.close()

        materialized = self._dedupe_rows([self._row_to_dict(row) for row in rows])
        if semester:
            materialized = [
                row for row in materialized if _semester_matches_query(str(row.get("semester") or ""), semester)
            ]
        source_files = sorted({str(r.get("source_file") or "").strip() for r in materialized if str(r.get("source_file") or "").strip()})
        coverage_note = "Tìm thấy dữ liệu lịch học." if materialized else "Không tìm thấy dữ liệu lịch phù hợp."
        return {
            "rows": materialized,
            "source_files": source_files,
            "coverage_note": coverage_note,
        }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        row_keys = set(row.keys())
        teacher_name = _sanitize_teacher_name(row["teacher_name"])
        source_page = _to_positive_int(row["source_page"]) if "source_page" in row_keys else None
        source_line = _to_positive_int(row["source_line"]) if "source_line" in row_keys else None
        return {
            "semester": row["semester"],
            "subject_code": row["subject_code"],
            "subject_name_vi": row["subject_name_vi"],
            "subject_name_en": row["subject_name_en"] if "subject_name_en" in row_keys else "",
            "class_code": row["class_code"],
            "teacher_name": teacher_name,
            "day_of_week": row["day_of_week"],
            "slot": row["slot"],
            "room": row["room"],
            "week_note": row["week_note"],
            "source_file": row["source_file"],
            "source_page": source_page,
            "source_line": source_line,
        }

    def _dedupe_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for row in rows:
            hash_key = _line_hash_payload(row)
            if hash_key in seen:
                continue
            seen.add(hash_key)
            deduped.append(row)
        return deduped

    def _compute_file_checksum(self, path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _rebuild_aliases(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM teacher_alias")
        conn.execute("DELETE FROM course_alias")
        rows = conn.execute(
            """
            SELECT subject_code, subject_name_vi, teacher_name
            FROM schedule_rows
            """
        ).fetchall()
        for row in rows:
            code = _normalize_code(str(row["subject_code"] or ""))
            name_vi = str(row["subject_name_vi"] or "").strip()
            teacher = _sanitize_teacher_name(str(row["teacher_name"] or ""))
            if code:
                aliases = {
                    _normalize_text(code),
                    _normalize_text(name_vi),
                    _normalize_text(f"mon {name_vi}"),
                }
                name_norm = _normalize_text(name_vi)
                words = [w for w in name_norm.split() if w]
                if len(words) >= 2:
                    for width in range(2, min(7, len(words) + 1)):
                        head = " ".join(words[:width]).strip()
                        tail = " ".join(words[-width:]).strip()
                        if len(head) >= 8:
                            aliases.add(head)
                        if len(tail) >= 8:
                            aliases.add(tail)
                    for start in range(len(words)):
                        for width in range(2, min(5, len(words) - start + 1)):
                            phrase = " ".join(words[start : start + width]).strip()
                            if len(phrase) >= 8:
                                aliases.add(phrase)
                for alias in [a for a in aliases if a]:
                    conn.execute(
                        "INSERT OR IGNORE INTO course_alias(alias_norm, subject_code, subject_name_vi) VALUES (?, ?, ?)",
                        (alias, code, name_vi),
                    )
            if teacher:
                for alias in self._teacher_aliases(teacher):
                    conn.execute(
                        "INSERT OR IGNORE INTO teacher_alias(alias_norm, teacher_name_canonical) VALUES (?, ?)",
                        (alias, teacher),
                    )

    def _teacher_aliases(self, teacher_name: str) -> List[str]:
        aliases = set()
        cleaned = _sanitize_teacher_name(teacher_name)
        norm = _normalize_text(cleaned)
        if not norm:
            return []
        aliases.add(norm)
        words = [w for w in norm.split() if w]
        if len(words) >= 2:
            aliases.add(" ".join(words[-2:]))
        if len(words) >= 3:
            aliases.add(" ".join(words[-3:]))
        return [a for a in aliases if a]

    def _parse_schedule_pdf(self, path: Path) -> List[Dict[str, Any]]:
        docs = process_pdf(str(path))
        full_text = "\n".join(str(doc.page_content or "") for doc in docs)
        semester = _infer_semester_label(path.name, full_text)

        parsed_rows: List[Dict[str, Any]] = []
        for doc_idx, doc in enumerate(docs, start=1):
            page_text = str(doc.page_content or "")
            source_page = _to_positive_int(doc.metadata.get("page")) if isinstance(doc.metadata, dict) else None
            if source_page is None:
                source_page = doc_idx
            for line_idx, raw_line in enumerate(page_text.splitlines(), start=1):
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if not COURSE_CODE_RE.search(line.upper()):
                    continue
                rows = self._parse_schedule_line(
                    line,
                    semester=semester,
                    source_file=path.name,
                    source_page=source_page,
                    source_line=line_idx,
                )
                parsed_rows.extend(rows)

        # Local dedupe before DB write.
        unique_by_hash: Dict[str, Dict[str, Any]] = {}
        for row in parsed_rows:
            row_hash = _line_hash_payload(row)
            row["row_hash"] = row_hash
            unique_by_hash[row_hash] = row
        return list(unique_by_hash.values())

    def _parse_schedule_line(
        self,
        raw_line: str,
        semester: str,
        source_file: str,
        source_page: Optional[int] = None,
        source_line: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        line = " ".join(str(raw_line or "").split())
        code_match = COURSE_CODE_RE.search(line.upper())
        if not code_match:
            return []
        subject_code = _normalize_code(code_match.group(1))
        if not subject_code:
            return []

        class_code = ""
        class_match = re.search(
            rf"\b(?:UET\.)?{re.escape(subject_code)}\s*([1-9]\d{{0,2}})\b",
            line.upper(),
        )
        if class_match:
            class_code = f"{subject_code} {class_match.group(1)}"

        day_token = ""
        slot_token = ""
        day_slot_match = re.search(
            r"\|\s*(?:LT\+TH|LT|TH|ONL)\s*\|\s*([2-8])\s*\|\s*([1-9])\s*\|",
            raw_line,
            flags=re.IGNORECASE,
        )
        if day_slot_match:
            day_token = day_slot_match.group(1)
            slot_token = day_slot_match.group(2)
        else:
            day_slot_match = re.search(
                r"\b(?:LT\+TH|LT|TH|ONL)\b\s+([2-8])\s+([1-9])\b",
                line,
                flags=re.IGNORECASE,
            )
            if day_slot_match:
                day_token = day_slot_match.group(1)
                slot_token = day_slot_match.group(2)

        room = ""
        room_match = ROOM_RE.search(line.upper())
        if room_match:
            room = room_match.group(0).upper()

        week_note = ""
        note_match = re.search(r"(H[ọo]c\s+[^|]+?(?:tuần|tuan)[^|]*)", line, flags=re.IGNORECASE)
        if note_match:
            week_note = note_match.group(1).strip(" ,")

        subject_name_vi = self._extract_subject_name(line, subject_code)
        subject_name_en = ""
        teachers = self._extract_teachers(raw_line=raw_line, compact_line=line, room=room)
        if not teachers:
            teachers = [""]

        day_label = _to_day_label(day_token)
        rows: List[Dict[str, Any]] = []
        for teacher in teachers:
            rows.append(
                {
                    "semester": semester,
                    "subject_code": subject_code,
                    "subject_name_vi": subject_name_vi,
                    "subject_name_en": subject_name_en,
                    "class_code": class_code,
                    "teacher_name": teacher,
                    "day_of_week": day_label,
                    "slot": slot_token,
                    "room": room,
                    "week_note": week_note,
                    "source_file": source_file,
                    "source_page": source_page,
                    "source_line": source_line,
                }
            )
        return rows

    def _extract_subject_name(self, line: str, subject_code: str) -> str:
        m = re.search(
            rf"\b{re.escape(subject_code)}\b\s+(.+?)\s+\d{{1,2}}\s+\d{{1,3}}",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            return ""
        raw = m.group(1).strip(" |")
        raw = re.sub(r"\s+", " ", raw)
        raw = re.sub(rf"\b{re.escape(subject_code)}\b", "", raw, flags=re.IGNORECASE).strip()
        return raw[:240]

    def _extract_teachers(self, raw_line: str, compact_line: str, room: str) -> List[str]:
        tail = ""
        if room:
            idx = compact_line.upper().find(room.upper())
            if idx >= 0:
                tail = compact_line[idx + len(room):].strip()
        if not tail:
            tail = compact_line

        cut_match = re.search(r"\b(?:H[ọo]c|Hoc|thi|đợt|dot)\b", tail, flags=re.IGNORECASE)
        if cut_match:
            tail = tail[: cut_match.start()].strip()
        tail = tail.strip(" |")

        teachers = _parse_teacher_list(tail)
        if teachers:
            return teachers

        # Secondary pass for pipe-heavy rows.
        if "|" in raw_line:
            tokens = [t.strip() for t in str(raw_line).split("|") if t.strip()]
            if room:
                room_norm = room.upper()
                room_idx = -1
                for i, token in enumerate(tokens):
                    if token.upper() == room_norm:
                        room_idx = i
                        break
                if room_idx >= 0:
                    tail_tokens = tokens[room_idx + 1 : room_idx + 8]
                    tail_text = " ".join(tail_tokens)
                    teachers = _parse_teacher_list(tail_text)
                    if teachers:
                        return teachers

        return []
