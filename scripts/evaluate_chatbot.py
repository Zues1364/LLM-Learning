from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "golden_academic_advisor.jsonl"
DEFAULT_MOCK_DIR = ROOT / "evals" / "mock_data"
DEFAULT_REPORTS_DIR = ROOT / "reports"
DEFAULT_DEPLOY_URL = "https://backend-production-3cb2.up.railway.app"
DEFAULT_LOCAL_URL = "http://127.0.0.1:9000"
DEFAULT_FRONTEND_URL = "https://llm-learning.vercel.app"
DEFAULT_MCP_PUBLIC_URL = "https://mcp-production-95c4.up.railway.app"

CASE_CONTEXT_FIELDS = [
    "query",
    "program_id",
    "mock_profile_id",
    "turn_group",
    "execution",
    "expected_source_any",
    "expected_keywords",
    "forbidden_keywords",
    "expected_codes",
    "expected_numbers",
    "review_rubric",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def attach_case_context(row: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    row["category"] = case.get("category")
    for key in CASE_CONTEXT_FIELDS:
        if key in case and case.get(key) not in (None, "", [], {}):
            row[key] = case.get(key)
    return row


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def contains_text(haystack: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(haystack)


def p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.95)))
    return round(float(ordered[idx]), 2)


def p50(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(statistics.median(values)), 2)


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_value(key: str) -> str:
    return os.getenv(key) or read_env_file(ROOT / ".env").get(key, "")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        obj.setdefault("_line_no", line_no)
        cases.append(obj)
    return cases


def dataset_counts(path: Path, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    static_cases = [case for case in cases if case.get("execution") == "mock_static"]
    smoke_cases = [case for case in cases if case.get("execution") == "deploy_smoke"]
    live_cases = [
        case
        for case in cases
        if case.get("execution") not in {"mock_static", "deploy_smoke"}
    ]
    return {
        "path": str(path),
        "case_count": len(cases),
        "static_case_count": len(static_cases),
        "smoke_case_count": len(smoke_cases),
        "live_case_count": len(live_cases),
    }


def select_cases(cases: List[Dict[str, Any]], requested_ids: List[str]) -> List[Dict[str, Any]]:
    if not requested_ids:
        return cases
    by_id = {str(case.get("id")): case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown case id(s): {', '.join(missing)}")
    seen: set[str] = set()
    selected: List[Dict[str, Any]] = []
    for case_id in requested_ids:
        if case_id in seen:
            continue
        selected.append(by_id[case_id])
        seen.add(case_id)
    return selected


def load_mock_profiles(mock_dir: Path) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for path in sorted((mock_dir / "transcripts").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        profiles[str(data["profile_id"])] = data
    return profiles


def load_mock_curricula(mock_dir: Path) -> Dict[str, Dict[str, Any]]:
    curricula: Dict[str, Dict[str, Any]] = {}
    for path in sorted((mock_dir / "curricula").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        curricula[str(data["program_id"])] = data
    return curricula


def collect_taken_subject_codes(profile: Dict[str, Any]) -> List[str]:
    return [
        str(subject.get("code"))
        for semester in profile.get("semesters") or []
        for subject in (semester.get("subjects") or [])
        if str(subject.get("code") or "").strip()
    ]


def compute_required_missing_codes(
    profile: Dict[str, Any],
    curriculum: Optional[Dict[str, Any]],
) -> List[str]:
    if not curriculum:
        return []
    taken_codes = set(collect_taken_subject_codes(profile))
    missing_codes: List[str] = []
    for group in curriculum.get("groups") or []:
        for code in group.get("required_subjects") or []:
            normalized = str(code or "").strip()
            if normalized and normalized not in taken_codes:
                missing_codes.append(normalized)
    return missing_codes


def validate_mock_data(mock_dir: Path, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    profiles = load_mock_profiles(mock_dir)
    curricula = load_mock_curricula(mock_dir)
    checks: List[Dict[str, Any]] = []

    for profile_id, profile in profiles.items():
        program_id = profile.get("student", {}).get("program_id")
        summary = profile.get("summary") or {}
        curriculum = curricula.get(str(program_id))
        subjects = [
            subject
            for sem in profile.get("semesters") or []
            for subject in (sem.get("subjects") or [])
        ]
        stored_required_codes = summary.get("expected_required_missing_codes") or []
        computed_required_codes = compute_required_missing_codes(profile, curriculum)
        required_codes_match = stored_required_codes == computed_required_codes
        checks.append(
            {
                "name": f"profile:{profile_id}",
                "pass": bool(program_id in curricula and subjects and required_codes_match),
                "details": {
                    "program_id": program_id,
                    "subject_count": len(subjects),
                    "completed_credits": summary.get("completed_credits"),
                    "expected_missing_credits": summary.get("expected_missing_credits"),
                    "required_missing_codes": stored_required_codes,
                    "computed_required_missing_codes": computed_required_codes,
                    "required_missing_match": required_codes_match,
                    "open_group_missing_credits": summary.get("expected_open_group_missing_credits") or {},
                    "has_curriculum": program_id in curricula,
                },
            }
        )

    referenced_profiles = sorted(
        {
            str(case.get("mock_profile_id"))
            for case in cases
            if str(case.get("mock_profile_id") or "").strip()
        }
    )
    for profile_id in referenced_profiles:
        checks.append(
            {
                "name": f"case-reference:{profile_id}",
                "pass": profile_id in profiles,
                "details": {"profile_id": profile_id},
            }
        )

    return {
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
        "profile_count": len(profiles),
        "curriculum_count": len(curricula),
        "profiles": summarize_mock_profiles(profiles, curricula),
        "checks": checks,
    }


def summarize_mock_profiles(
    profiles: Dict[str, Dict[str, Any]],
    curricula: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for profile_id, profile in sorted(profiles.items()):
        summary = profile.get("summary") or {}
        student = profile.get("student") or {}
        curriculum = curricula.get(str(student.get("program_id")))
        required_missing_codes = compute_required_missing_codes(profile, curriculum)
        rows.append(
            {
                "profile_id": profile_id,
                "program_id": student.get("program_id"),
                "completed_credits": summary.get("completed_credits"),
                "recognized_credits": summary.get("recognized_credits"),
                "expected_missing_credits": summary.get("expected_missing_credits"),
                "expected_required_missing_codes": required_missing_codes,
                "expected_open_group_missing_credits": summary.get("expected_open_group_missing_credits") or {},
            }
        )
    return rows


def pdf_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def render_mock_transcript_pdf(profile: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_id = str(profile["profile_id"])
    out_path = out_dir / f"{profile_id}.pdf"
    student = profile.get("student") or {}
    summary = profile.get("summary") or {}
    lines = [
        "MOCK ACADEMIC TRANSCRIPT - FOR EVALUATION ONLY",
        f"Student ID: {student.get('student_id', '')}",
        f"Student name: {student.get('name', '')}",
        f"Program ID: {student.get('program_id', '')}",
        f"Program name: {student.get('program_name', '')}",
        f"Completed credits: {summary.get('completed_credits', '')}",
        f"Recognized credits: {summary.get('recognized_credits', '')}",
        f"Expected missing credits: {summary.get('expected_missing_credits', '')}",
        "Subjects:",
        "Semester | Code | Name | Credits | Grade4",
    ]
    for sem in profile.get("semesters") or []:
        for subject in sem.get("subjects") or []:
            lines.append(
                f"{sem.get('semester')} | {subject.get('code')} | {subject.get('name')} | "
                f"{subject.get('credits')} | {subject.get('grade_4')}"
            )

    content_lines = ["BT", "/F1 9 Tf", "42 790 Td"]
    for idx, line in enumerate(lines[:52]):
        if idx:
            content_lines.append("0 -13 Td")
        content_lines.append(f"({pdf_escape(line[:110])}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: List[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(sum(len(p) for p in parts))
        parts.append(f"{idx} 0 obj\n".encode("ascii"))
        parts.append(obj)
        parts.append(b"\nendobj\n")
    xref_offset = sum(len(p) for p in parts)
    parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets:
        parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    parts.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    out_path.write_bytes(b"".join(parts))
    return out_path


def safe_get(url: str, timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = requests.get(url, timeout=timeout)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "ok": resp.status_code < 500,
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "body": resp.text[:4000],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"ok": False, "status_code": None, "latency_ms": elapsed_ms, "body": "", "error": str(exc)}


def safe_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        data: Dict[str, Any] = {}
        if resp.headers.get("content-type", "").startswith("application/json"):
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                data = {}
        return {
            "ok": resp.status_code < 500,
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "json": data,
            "body": resp.text[:4000],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"ok": False, "status_code": None, "latency_ms": elapsed_ms, "json": {}, "body": "", "error": str(exc)}


def upload_mock_pdf(base_url: str, session_id: str, pdf_path: Path, timeout: int) -> List[str]:
    with pdf_path.open("rb") as fh:
        files = [("files", (pdf_path.name, fh, "application/pdf"))]
        resp = requests.post(
            f"{base_url.rstrip()}/upload_pdfs",
            files=files,
            data={"session_id": session_id},
            timeout=timeout,
        )
    resp.raise_for_status()
    payload = resp.json()
    return [str(item.get("file_id")) for item in payload.get("uploaded", []) if item.get("file_id")]


def score_observation(case: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        [
            str(observation.get("answer") or ""),
            str(observation.get("body") or ""),
            json.dumps(observation.get("json") or {}, ensure_ascii=False),
        ]
    )
    program_selection_prompt = (
        contains_text(text, "vui lòng chọn chương trình đào tạo")
        and contains_text(text, "khóa tuyển sinh")
    ) or contains_text(text, "requires_program_selection")
    checks: List[Dict[str, Any]] = []

    def add_check(
        name: str,
        passed: bool,
        details: Dict[str, Any],
        dimension: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "pass": passed,
                "details": details,
                "dimension": dimension,
            }
        )

    if case.get("accept_program_selection_guardrail") and program_selection_prompt:
        checks = [
            {
                "name": "program_selection_guardrail_source",
                "pass": True,
                "details": {"actual": str(observation.get("source") or "")},
                "dimension": "source",
            },
            {
                "name": "program_selection_guardrail_content",
                "pass": True,
                "details": {"matched": "select_program_before_answer"},
                "dimension": "content",
            },
        ]
        return {
            "status": "warn",
            "pass": True,
            "checks": checks,
            "dimensions": {
                "source": {"status": "warn", "pass": True, "check_count": 1, "fail": 0},
                "content": {"status": "warn", "pass": True, "check_count": 1, "fail": 0},
                "transport": {"status": "not_applicable", "pass": True, "check_count": 0, "fail": 0},
            },
            "source_status": "warn",
            "content_status": "warn",
        }

    if "expected_status_lt" in case:
        status_code = observation.get("status_code")
        expected_lt = int(case["expected_status_lt"])
        add_check(
            "status_code",
            isinstance(status_code, int) and status_code < expected_lt,
            {"actual": status_code, "expected_lt": expected_lt},
            "transport",
        )

    expected_sources = case.get("expected_source_any") or []
    source_optional = bool(case.get("source_optional"))
    if expected_sources and not source_optional:
        actual_source = str(observation.get("source") or "")
        add_check(
            "source",
            actual_source in set(expected_sources),
            {"actual": actual_source, "expected_any": expected_sources},
            "source",
        )
    elif expected_sources and source_optional:
        add_check(
            "source_optional",
            True,
            {"actual": str(observation.get("source") or ""), "expected_any": expected_sources},
            "source",
        )

    if case.get("expected_citation") is True:
        citations = observation.get("citations") or []
        add_check(
            "citation_present",
            isinstance(citations, list) and len(citations) > 0,
            {"citation_count": len(citations) if isinstance(citations, list) else 0},
            "source",
        )

    for field_name, check_name in [
        ("expected_keywords", "keyword_present"),
        ("expected_codes", "code_present"),
        ("expected_numbers", "number_present"),
    ]:
        for item in case.get(field_name) or []:
            add_check(
                f"{check_name}:{item}",
                contains_text(text, str(item)),
                {"needle": item},
                "content",
            )

    for item in case.get("forbidden_keywords") or []:
        add_check(
            f"forbidden_absent:{item}",
            not contains_text(text, str(item)),
            {"needle": item},
            "content",
        )

    failed = [item for item in checks if not item["pass"]]
    if failed and case.get("allow_known_failure"):
        status = "warn"
    else:
        status = "pass" if not failed else "fail"
    dimensions: Dict[str, Dict[str, Any]] = {}
    for dimension in ("source", "content", "transport"):
        dimension_checks = [item for item in checks if item.get("dimension") == dimension]
        if not dimension_checks:
            dimensions[dimension] = {"status": "not_applicable", "pass": True, "check_count": 0, "fail": 0}
            continue
        dimension_failures = [item for item in dimension_checks if not item.get("pass")]
        dimension_status = "pass" if not dimension_failures else ("warn" if case.get("allow_known_failure") else "fail")
        dimensions[dimension] = {
            "status": dimension_status,
            "pass": dimension_status in {"pass", "warn"},
            "check_count": len(dimension_checks),
            "fail": len(dimension_failures),
        }
    return {
        "status": status,
        "pass": status in {"pass", "warn"},
        "checks": checks,
        "dimensions": dimensions,
        "source_status": dimensions["source"]["status"],
        "content_status": dimensions["content"]["status"],
    }


def run_mock_static_case(case: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    profile_id = str(case.get("mock_profile_id") or "")
    profile = profiles.get(profile_id) or {}
    summary = profile.get("summary") or {}
    student = profile.get("student") or {}
    body = json.dumps({"student": student, "summary": summary}, ensure_ascii=False)
    observation = {
        "target": "mock_static",
        "case_id": case["id"],
        "status_code": 200 if profile else 404,
        "source": "mock_static",
        "answer": body,
        "body": body,
        "citations": [],
        "latency_ms": 0.0,
    }
    observation["score"] = score_observation(case, observation)
    return observation


def run_smoke_case(case: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = str(case["endpoint"])
    result = safe_get(url, timeout=timeout)
    observation = {
        "target": "deploy_smoke",
        "case_id": case["id"],
        "status_code": result.get("status_code"),
        "source": "deploy_smoke",
        "answer": result.get("body") or result.get("error") or "",
        "body": result.get("body") or result.get("error") or "",
        "citations": [],
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
    }
    observation["score"] = score_observation(case, observation)
    return observation


def target_preflight(name: str, base_url: str, timeout: int) -> Dict[str, Any]:
    health = safe_get(f"{base_url.rstrip('/')}/healthz", timeout=timeout)
    ready = safe_get(f"{base_url.rstrip('/')}/readyz", timeout=timeout)
    health_status = health.get("status_code")
    ready_status = ready.get("status_code")
    return {
        "target": name,
        "base_url": base_url,
        "healthz": {k: health.get(k) for k in ("ok", "status_code", "latency_ms", "error")},
        "readyz": {k: ready.get(k) for k in ("ok", "status_code", "latency_ms", "error")},
        "ready": bool(
            isinstance(health_status, int)
            and 200 <= health_status < 400
            and isinstance(ready_status, int)
            and 200 <= ready_status < 400
        ),
    }


def supabase_preflight(timeout: int) -> Dict[str, Any]:
    url = env_value("SUPABASE_URL").rstrip("/")
    key = env_value("SUPABASE_SERVICE_ROLE_KEY")
    bucket = env_value("SUPABASE_STORAGE_BUCKET") or "rag-files"
    if not url or not key:
        return {"enabled": False, "ready": False, "bucket": bucket}

    def req(path: str) -> Tuple[bool, Optional[int], str]:
        try:
            request = urllib.request.Request(url + path, method="GET")
            request.add_header("apikey", key)
            request.add_header("Authorization", "Bearer " + key)
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return True, int(resp.status), resp.read(4000).decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            return False, None, str(exc)

    rest_ok, rest_status, _ = req("/rest/v1/")
    storage_ok, storage_status, storage_body = req("/storage/v1/bucket")
    bucket_found = False
    try:
        bucket_found = bucket in [item.get("name") for item in json.loads(storage_body)]
    except Exception:  # noqa: BLE001
        bucket_found = False
    return {
        "enabled": True,
        "ready": bool(rest_ok and storage_ok and bucket_found),
        "postgrest_status": rest_status,
        "storage_status": storage_status,
        "bucket": bucket,
        "bucket_found": bucket_found,
    }


def run_live_case(
    case: Dict[str, Any],
    target_name: str,
    base_url: str,
    profiles: Dict[str, Dict[str, Any]],
    upload_cache: Dict[Tuple[str, str, str], List[str]],
    timeout: int,
    stamp: str,
) -> Dict[str, Any]:
    turn_group = str(case.get("turn_group") or case["id"])
    profile_id = str(case.get("mock_profile_id") or "")
    session_suffix = profile_id or re.sub(r"[^A-Za-z0-9_-]+", "_", turn_group)[:40]
    session_id = f"eval_{stamp}_{target_name}_{session_suffix}"
    file_ids = list(case.get("file_ids") or [])

    if profile_id:
        profile = profiles.get(profile_id)
        if not profile:
            observation = {
                "target": target_name,
                "case_id": case["id"],
                "status_code": None,
                "source": "error",
                "answer": f"missing mock profile {profile_id}",
                "body": "",
                "citations": [],
                "latency_ms": 0.0,
                "error": "missing_mock_profile",
            }
            observation["score"] = score_observation(case, observation)
            return observation
        cache_key = (target_name, session_id, profile_id)
        if cache_key not in upload_cache:
            pdf_path = render_mock_transcript_pdf(profile, ROOT / "tmp" / "eval_mock_pdfs")
            upload_cache[cache_key] = upload_mock_pdf(base_url, session_id, pdf_path, timeout=timeout)
        file_ids = upload_cache[cache_key]

    payload = {
        "query": case["query"],
        "session_id": session_id,
        "file_ids": file_ids,
        "program_id": case.get("program_id"),
        "allow_web_search": False,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    result = safe_post_json(f"{base_url.rstrip('/')}/ask", payload, timeout=timeout)
    data = result.get("json") or {}
    answer = str(data.get("answer") or result.get("body") or result.get("error") or "")
    observation = {
        "target": target_name,
        "case_id": case["id"],
        "session_id": session_id,
        "status_code": result.get("status_code"),
        "source": data.get("source"),
        "answer": answer,
        "body": result.get("body") or "",
        "citations": data.get("citations") or [],
        "selected_program_id": data.get("selected_program_id"),
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
        "file_ids": file_ids,
    }
    observation["score"] = score_observation(case, observation)
    return observation


def summarize_observations(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_target: Dict[str, Dict[str, Any]] = {}
    for target, rows_iter in group_by(observations, lambda row: str(row.get("target") or "unknown")).items():
        rows = list(rows_iter)
        latencies = [
            float(row.get("latency_ms"))
            for row in rows
            if isinstance(row.get("latency_ms"), (int, float)) and float(row.get("latency_ms")) > 0
        ]
        pass_count = sum(1 for row in rows if row.get("score", {}).get("status") == "pass")
        warn_count = sum(1 for row in rows if row.get("score", {}).get("status") == "warn")
        fail_count = sum(1 for row in rows if row.get("score", {}).get("status") == "fail")
        source_scored = [row for row in rows if dimension_status(row, "source") not in {None, "not_applicable"}]
        content_scored = [row for row in rows if dimension_status(row, "content") not in {None, "not_applicable"}]
        source_pass = sum(
            1
            for row in source_scored
            if dimension_status(row, "source") in {"pass", "warn"}
        )
        content_pass = sum(
            1
            for row in content_scored
            if dimension_status(row, "content") in {"pass", "warn"}
        )
        by_target[target] = {
            "case_count": len(rows),
            "pass": pass_count,
            "warn": warn_count,
            "fail": fail_count,
            "pass_rate": round((pass_count + warn_count) / len(rows), 4) if rows else None,
            "source_pass_rate": round(source_pass / len(source_scored), 4) if source_scored else None,
            "content_pass_rate": round(content_pass / len(content_scored), 4) if content_scored else None,
            "latency_p50_ms": p50(latencies),
            "latency_p95_ms": p95(latencies),
            "by_category": summarize_by_category(rows),
        }
    return by_target


def summarize_by_category(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    categories = group_by(rows, lambda row: str(row.get("category") or "unknown"))
    out: List[Dict[str, Any]] = []
    for category, items_iter in sorted(categories.items()):
        items = list(items_iter)
        pass_count = sum(1 for row in items if row.get("score", {}).get("status") in {"pass", "warn"})
        fail_count = sum(1 for row in items if row.get("score", {}).get("status") == "fail")
        source_scored = [row for row in items if dimension_status(row, "source") not in {None, "not_applicable"}]
        content_scored = [row for row in items if dimension_status(row, "content") not in {None, "not_applicable"}]
        source_pass = sum(
            1
            for row in source_scored
            if dimension_status(row, "source") in {"pass", "warn"}
        )
        content_pass = sum(
            1
            for row in content_scored
            if dimension_status(row, "content") in {"pass", "warn"}
        )
        latencies = [
            float(row.get("latency_ms"))
            for row in items
            if isinstance(row.get("latency_ms"), (int, float)) and float(row.get("latency_ms")) > 0
        ]
        out.append(
            {
                "category": category,
                "case_count": len(items),
                "pass": pass_count,
                "fail": fail_count,
                "pass_rate": round(pass_count / len(items), 4) if items else None,
                "source_pass_rate": round(source_pass / len(source_scored), 4) if source_scored else None,
                "content_pass_rate": round(content_pass / len(content_scored), 4) if content_scored else None,
                "latency_p50_ms": p50(latencies),
                "latency_p95_ms": p95(latencies),
            }
        )
    return out


def dimension_status(row: Dict[str, Any], dimension: str) -> Optional[str]:
    status = row.get("score", {}).get("dimensions", {}).get(dimension, {}).get("status")
    return str(status) if status is not None else None


def group_by(items: Iterable[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        out[key_fn(item)].append(item)
    return dict(out)


def enrich_observations_with_case_context(
    observations: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {str(case.get("id")): case for case in cases}
    enriched: List[Dict[str, Any]] = []
    for row in observations:
        copied = dict(row)
        case = by_id.get(str(copied.get("case_id") or ""))
        if case:
            attach_case_context(copied, case)
        enriched.append(copied)
    return enriched


def merge_with_latest_report(
    current_report: Dict[str, Any],
    latest_path: Path,
    all_cases: List[Dict[str, Any]],
    all_dataset: Dict[str, Any],
) -> Dict[str, Any]:
    if not latest_path.exists():
        current_report["dataset"] = all_dataset
        current_report["observations"] = enrich_observations_with_case_context(
            current_report.get("observations") or [],
            all_cases,
        )
        current_report["summary"] = summarize_observations(current_report["observations"])
        current_report["merge"] = {
            "mode": "created_latest",
            "latest_path": str(latest_path),
            "updated_observations": len(current_report["observations"]),
        }
        return current_report

    latest_report = json.loads(latest_path.read_text(encoding="utf-8"))
    current_rows = current_report.get("observations") or []
    current_by_key = {
        (str(row.get("target") or ""), str(row.get("case_id") or "")): row
        for row in current_rows
    }
    merged_rows: List[Dict[str, Any]] = []
    replaced = 0
    for row in latest_report.get("observations") or []:
        key = (str(row.get("target") or ""), str(row.get("case_id") or ""))
        replacement = current_by_key.pop(key, None)
        if replacement is not None:
            merged_rows.append(replacement)
            replaced += 1
        else:
            merged_rows.append(row)
    added = len(current_by_key)
    merged_rows.extend(current_by_key.values())
    merged_rows = enrich_observations_with_case_context(merged_rows, all_cases)

    merged_report = dict(latest_report)
    merged_report.update(
        {
            "generated_at_utc": current_report["generated_at_utc"],
            "dataset": all_dataset,
            "mock_validation": current_report["mock_validation"],
            "preflight": current_report["preflight"],
            "embedding_benchmark": current_report["embedding_benchmark"],
            "observations": merged_rows,
            "summary": summarize_observations(merged_rows),
            "merge": {
                "mode": "merge_latest",
                "latest_path": str(latest_path),
                "updated_observations": len(current_rows),
                "replaced_observations": replaced,
                "added_observations": added,
            },
        }
    )
    return merged_report


def latest_embedding_summary() -> Dict[str, Any]:
    candidates = sorted(DEFAULT_REPORTS_DIR.glob("embedding_benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {}
    path = candidates[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    model = (data.get("models") or [{}])[0]
    return {
        "path": str(path),
        "generated_at": data.get("generated_at"),
        "resource_count": len(data.get("resources") or []),
        "case_count": len(data.get("cases") or []),
        "summary": model.get("summary") or {},
    }


def clamp_text(value: Any, max_chars: int = 1800) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def status_label(row: Dict[str, Any]) -> str:
    status = str(row.get("score", {}).get("status") or "unknown").upper()
    return status


def failed_checks(row: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    return [check for check in row.get("score", {}).get("checks", []) if not check.get("pass")][:limit]


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Academic Advisor Chatbot Evaluation")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- dataset_cases: `{report['dataset']['case_count']}`")
    lines.append(f"- mock_profiles: `{report['mock_validation']['profile_count']}`")
    lines.append("")
    lines.append("## Mock data")
    lines.append("")
    lines.append("| Profile | Program | Completed credits | Missing credits | Required missing courses (full) | Open-group missing credits |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for row in report["mock_validation"].get("profiles") or []:
        required_missing = ", ".join(row["expected_required_missing_codes"]) or "—"
        open_group_missing = row.get("expected_open_group_missing_credits") or {}
        open_group_text = (
            ", ".join(f"{key}: {value}" for key, value in open_group_missing.items())
            if open_group_missing
            else "—"
        )
        lines.append(
            f"| `{row['profile_id']}` | `{row['program_id']}` | {row['completed_credits']} | "
            f"{row['expected_missing_credits']} | `{required_missing}` | `{open_group_text}` |"
        )
    lines.append("")
    lines.append("## Preflight")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.get("preflight"), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Target summary")
    lines.append("")
    lines.append("| Target | Cases | Pass | Warn | Fail | Accepted rate | Source | Content | p50 ms | p95 ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for target, summary in sorted((report.get("summary") or {}).items()):
        lines.append(
            f"| `{target}` | {summary['case_count']} | {summary['pass']} | {summary['warn']} | {summary['fail']} | "
            f"{summary['pass_rate']} | {summary.get('source_pass_rate')} | {summary.get('content_pass_rate')} | "
            f"{summary['latency_p50_ms']} | {summary['latency_p95_ms']} |"
        )
    lines.append("")
    lines.append("## Failed cases")
    lines.append("")
    failed = [row for row in report.get("observations", []) if row.get("score", {}).get("status") == "fail"]
    if not failed:
        lines.append("No deterministic failures.")
    for row in failed[:30]:
        lines.append(f"- `{row.get('target')}/{row.get('case_id')}` source=`{row.get('source')}` status=`{row.get('status_code')}`")
        if row.get("query"):
            lines.append(f"  - query: `{row.get('query')}`")
        checks = [c for c in row.get("score", {}).get("checks", []) if not c.get("pass")]
        for check in checks[:3]:
            lines.append(f"  - {check.get('name')}: {json.dumps(check.get('details'), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Case details")
    lines.append("")
    for row in sorted(
        report.get("observations", []),
        key=lambda item: (
            str(item.get("target") or ""),
            str(item.get("category") or ""),
            str(item.get("case_id") or ""),
        ),
    ):
        lines.append(
            f"### `{row.get('target')}/{row.get('case_id')}` - {status_label(row)}"
        )
        lines.append("")
        lines.append(f"- Category: `{row.get('category')}`")
        lines.append(f"- Query: `{row.get('query') or ''}`")
        lines.append(
            f"- Source: `{row.get('source')}`; HTTP status: `{row.get('status_code')}`; latency_ms: `{row.get('latency_ms')}`"
        )
        if row.get("program_id"):
            lines.append(f"- Program: `{row.get('program_id')}`")
        if row.get("mock_profile_id"):
            lines.append(f"- Mock profile: `{row.get('mock_profile_id')}`")
        checks = failed_checks(row)
        if checks:
            lines.append("- Failed checks:")
            for check in checks:
                lines.append(
                    f"  - `{check.get('name')}`: {json.dumps(check.get('details'), ensure_ascii=False)}"
                )
        lines.append("- Answer:")
        lines.append("")
        lines.append("```text")
        lines.append(clamp_text(row.get("answer")).replace("```", "'''"))
        lines.append("```")
        lines.append("")
    lines.append("## Embedding benchmark")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.get("embedding_benchmark"), ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the LLM Learning academic advisor chatbot.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--mock-data-dir", default=str(DEFAULT_MOCK_DIR))
    parser.add_argument("--target", choices=["local", "deploy", "both"], default="both")
    parser.add_argument("--local-url", default=DEFAULT_LOCAL_URL)
    parser.add_argument("--deploy-url", default=DEFAULT_DEPLOY_URL)
    parser.add_argument("--frontend-url", default=DEFAULT_FRONTEND_URL)
    parser.add_argument("--mcp-public-url", default=DEFAULT_MCP_PUBLIC_URL)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit live /ask cases per target; 0 means all.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the selected case id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--merge-latest",
        action="store_true",
        help="Merge this run into eval_academic_advisor_latest instead of replacing unrelated cases.",
    )
    parser.add_argument("--skip-live", action="store_true", help="Only validate mock/static and smoke checks.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = utc_stamp()
    dataset_path = Path(args.dataset)
    mock_dir = Path(args.mock_data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_jsonl(dataset_path)
    try:
        cases = select_cases(all_cases, args.case_id)
    except ValueError as exc:
        print(f"[eval] {exc}")
        return 2
    profiles = load_mock_profiles(mock_dir)
    mock_validation = validate_mock_data(mock_dir, all_cases)
    upload_cache: Dict[Tuple[str, str, str], List[str]] = {}

    preflight: Dict[str, Any] = {
        "supabase": supabase_preflight(timeout=min(args.timeout, 20)),
        "frontend": safe_get(args.frontend_url, timeout=min(args.timeout, 20)),
        "mcp_public_discover": safe_get(args.mcp_public_url.rstrip("/") + "/mcp/discover", timeout=min(args.timeout, 20)),
        "targets": {},
    }

    targets: Dict[str, str] = {}
    if args.target in {"local", "both"}:
        targets["local"] = args.local_url
    if args.target in {"deploy", "both"}:
        targets["deploy"] = args.deploy_url

    for name, base_url in targets.items():
        preflight["targets"][name] = target_preflight(name, base_url, timeout=min(args.timeout, 20))

    observations: List[Dict[str, Any]] = []
    static_cases = [case for case in cases if case.get("execution") == "mock_static"]
    smoke_cases = [case for case in cases if case.get("execution") == "deploy_smoke"]
    live_cases = [case for case in cases if case.get("execution") not in {"mock_static", "deploy_smoke"}]
    if args.max_cases > 0:
        live_cases = live_cases[: args.max_cases]

    if args.merge_latest and live_cases and not args.skip_live:
        not_ready_targets = [
            name
            for name in targets
            if not bool(preflight["targets"].get(name, {}).get("ready"))
        ]
        if not_ready_targets:
            for name in not_ready_targets:
                target_info = preflight["targets"].get(name, {})
                print(
                    "[eval] aborting merge because target is not ready: "
                    f"{name} healthz={target_info.get('healthz')} readyz={target_info.get('readyz')}"
                )
            return 3

    for case in static_cases:
        print(f"[eval] mock_static {case['id']}", flush=True)
        row = attach_case_context(run_mock_static_case(case, profiles), case)
        observations.append(row)

    for case in smoke_cases:
        print(f"[eval] deploy_smoke {case['id']}", flush=True)
        row = attach_case_context(run_smoke_case(case, timeout=min(args.timeout, 30)), case)
        observations.append(row)

    if not args.skip_live:
        for target_name, base_url in targets.items():
            target_ready = bool(preflight["targets"].get(target_name, {}).get("ready"))
            if not target_ready:
                for case in live_cases:
                    row = {
                            "target": target_name,
                            "case_id": case["id"],
                            "status_code": None,
                            "source": "blocked",
                            "answer": "target preflight not ready",
                            "body": "",
                            "citations": [],
                            "latency_ms": 0.0,
                            "error": "target_not_ready",
                            "score": {"status": "fail", "pass": False, "checks": [{"name": "target_ready", "pass": False, "details": {}}]},
                        }
                    observations.append(attach_case_context(row, case))
                continue
            for case in live_cases:
                print(f"[eval] {target_name} {case['id']}", flush=True)
                row = attach_case_context(
                    run_live_case(
                        case=case,
                        target_name=target_name,
                        base_url=base_url,
                        profiles=profiles,
                        upload_cache=upload_cache,
                        timeout=args.timeout,
                        stamp=stamp,
                    ),
                    case,
                )
                observations.append(row)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_counts(dataset_path, cases),
        "selected_case_ids": args.case_id,
        "mock_validation": mock_validation,
        "preflight": preflight,
        "embedding_benchmark": latest_embedding_summary(),
        "summary": summarize_observations(observations),
        "observations": observations,
    }

    json_path = reports_dir / f"eval_academic_advisor_{stamp}.json"
    md_path = reports_dir / f"eval_academic_advisor_{stamp}.md"
    latest_json = reports_dir / "eval_academic_advisor_latest.json"
    latest_md = reports_dir / "eval_academic_advisor_latest.md"
    if args.merge_latest:
        report = merge_with_latest_report(
            current_report=report,
            latest_path=latest_json,
            all_cases=all_cases,
            all_dataset=dataset_counts(dataset_path, all_cases),
        )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    latest_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")

    print(f"[eval] json={json_path}")
    print(f"[eval] md={md_path}")
    print(f"[eval] latest_json={latest_json}")
    print(f"[eval] latest={latest_md}")
    print(f"[eval] targets={','.join(sorted(report['summary'].keys()))}")
    return 0 if all(s.get("fail", 0) == 0 for s in report["summary"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
