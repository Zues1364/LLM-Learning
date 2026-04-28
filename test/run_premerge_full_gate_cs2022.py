from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_client.client import MCPClient  # noqa: E402
from utils import compute_curriculum_missing_credits  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    normalized = unicodedata.normalize("NFD", lowered)
    no_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(no_diacritics.split())


def _has_any(text: str, needles: List[str]) -> bool:
    hay = _normalize_text(text)
    return any(_normalize_text(item) in hay for item in needles)


def _extract_codes(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z]{2,4}\d{4}[A-Z]?\b", text or "")))


def _ensure_reports_dir() -> Path:
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_key_status() -> Dict[str, str]:
    env_path = ROOT / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}
    status: Dict[str, str] = {}
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = values.get(key) or os.getenv(key)
        status[key] = "set" if val else "missing"
    return status


def _http_ok(url: str, timeout: int = 30) -> Tuple[bool, str]:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500, f"status={resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _invoke_mcp(mcp: MCPClient, tool: str, args: Dict[str, Any]) -> Any:
    return mcp.invoke(tool, args)


def _json_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_completed_map(semesters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for sem in semesters or []:
        sem_code = sem.get("semester_code")
        for sub in sem.get("subjects") or []:
            code = str(sub.get("code") or "").strip().upper().replace(" ", "")
            if not code:
                continue
            grade = sub.get("grade_4")
            current = best.get(code)
            if current is None or (grade is not None and grade > (current.get("grade_4") or -1)):
                best[code] = {
                    "code": code,
                    "name": sub.get("name"),
                    "credits": int(sub.get("credits") or 0),
                    "grade_4": grade,
                    "semester": sem_code,
                }
    return best


def _build_structure_from_lookup(groups_lookup: Dict[str, Any]) -> List[Dict[str, Any]]:
    structure: List[Dict[str, Any]] = []
    main_blocks: Dict[str, Dict[str, Any]] = {}

    for group_code, group_data in groups_lookup.items():
        group_credits = int(group_data.get("credits_required") or 0)
        group_subjects = []
        for subj in group_data.get("subjects") or []:
            code = str(subj.get("code") or "").strip()
            if not code:
                continue
            group_subjects.append(
                {
                    "code": code,
                    "name": subj.get("name", ""),
                    "credits": int(subj.get("credits") or 0),
                }
            )
        notes = group_data.get("notes") or []

        if "." not in group_code:
            block = main_blocks.get(group_code)
            if block is None:
                block = {
                    "id": group_code,
                    "name": group_data.get("group_name", ""),
                    "required_credits": group_credits,
                    "type": "main",
                    "subjects": group_subjects,
                    "sub_blocks": [],
                    "notes": notes,
                }
                main_blocks[group_code] = block
                structure.append(block)
            else:
                block["required_credits"] = group_credits or block.get("required_credits", 0)
                if group_subjects:
                    block["subjects"] = group_subjects
                if notes:
                    block["notes"] = notes
            continue

        parent_code = group_code.split(".", 1)[0]
        parent_block = main_blocks.get(parent_code)
        if parent_block is None:
            parent_ref = groups_lookup.get(parent_code) or {}
            parent_block = {
                "id": parent_code,
                "name": parent_ref.get("group_name", ""),
                "required_credits": int(parent_ref.get("credits_required") or 0),
                "type": "main",
                "subjects": [],
                "sub_blocks": [],
                "notes": parent_ref.get("notes") or [],
            }
            main_blocks[parent_code] = parent_block
            structure.append(parent_block)

        parent_block["sub_blocks"].append(
            {
                "id": group_code,
                "name": group_data.get("group_name", ""),
                "required_credits": group_credits,
                "type": "sub",
                "subjects": group_subjects,
                "sub_blocks": [],
                "notes": notes,
            }
        )

    return structure


def _deterministic_checks(
    mcp: MCPClient,
    session_id: str,
    program_id: str,
    file_ids: List[str],
    expected_opened: List[str],
    expected_missing_credits: int = 21,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    transcript_raw = _invoke_mcp(
        mcp,
        "analyze_transcript",
        {"file_ids": file_ids},
    )
    transcript_data = _json_payload(transcript_raw)
    transcript_error = transcript_data.get("error")

    curriculum_raw = _invoke_mcp(
        mcp,
        "get_curriculum_lookup",
        {"program_id": program_id, "session_id": session_id},
    )
    curriculum = _json_payload(curriculum_raw)
    curriculum_error = curriculum.get("error")
    groups = curriculum.get("groups") or {}

    missing_credits_total = None
    if not transcript_error and not curriculum_error and groups:
        completed_map = _build_completed_map(transcript_data.get("semesters") or [])
        structure = _build_structure_from_lookup(groups)
        credit_analysis = compute_curriculum_missing_credits(structure, completed_map)
        missing_credits_total = int(
            sum(int(item.get("missing_credits") or 0) for item in (credit_analysis or []))
        )

    v24_codes = [
        str(item.get("code") or "").upper()
        for item in (groups.get("V.2.4", {}).get("subjects") or [])
    ]

    electives_raw = _invoke_mcp(
        mcp,
        "get_electives_with_schedule",
        {"check_schedule": True, "program_id": program_id, "session_id": session_id},
    )
    electives = json.loads(electives_raw)
    selected_groups = set(electives.get("selected_group_codes") or [])
    opened_codes = {
        str(item.get("code") or "").upper() for item in (electives.get("opened") or [])
    }
    not_opened_codes = {
        str(item.get("code") or "").upper() for item in (electives.get("not_opened") or [])
    }

    schedule_raw = _invoke_mcp(
        mcp,
        "get_schedule",
        {"subject_codes": ["INT3412E"], "session_id": session_id},
    )
    schedule_payload = json.loads(schedule_raw)
    has_schedule_lines = bool(
        isinstance(schedule_payload, list)
        and schedule_payload
        and (schedule_payload[0].get("schedule_lines") or [])
    )
    sample_line = ""
    if has_schedule_lines:
        sample_line = str((schedule_payload[0].get("schedule_lines") or [""])[0])
    has_teacher_or_class = _has_any(
        sample_line,
        ["Le Thanh Ha", "INT3412E 1", "Lop", "Thi giac may"],
    )

    checks.append(
        {
            "name": "missing_credits_expected_21",
            "pass": missing_credits_total == expected_missing_credits,
            "details": {
                "missing_credits_total": missing_credits_total,
                "expected": expected_missing_credits,
                "transcript_error": transcript_error,
                "curriculum_error": curriculum_error,
            },
        }
    )
    checks.append(
        {
            "name": "selected_group_codes_include_specialized_blocks",
            "pass": {"V.2.1", "V.2.2", "V.2.3", "V.2.4", "V.3"}.issubset(selected_groups),
            "details": sorted(selected_groups),
        }
    )
    checks.append(
        {
            "name": "opened_contains_expected_minimum_codes",
            "pass": set(expected_opened).issubset(opened_codes),
            "details": sorted(opened_codes),
        }
    )
    checks.append(
        {
            "name": "int3412e_schedule_has_lines_and_teacher_or_class",
            "pass": has_schedule_lines and has_teacher_or_class,
            "details": {
                "has_schedule_lines": has_schedule_lines,
                "sample_line": sample_line[:400],
            },
        }
    )
    checks.append(
        {
            "name": "int3404e_in_curriculum_v24",
            "pass": "INT3404E" in set(v24_codes),
            "details": v24_codes,
        }
    )
    checks.append(
        {
            "name": "int3404e_offered_false_current_term",
            "pass": ("INT3404E" in not_opened_codes) and ("INT3404E" not in opened_codes),
            "details": {
                "opened_has_INT3404E": "INT3404E" in opened_codes,
                "not_opened_has_INT3404E": "INT3404E" in not_opened_codes,
            },
        }
    )

    all_pass = all(item["pass"] for item in checks)
    return {
        "status": "pass" if all_pass else "fail",
        "checks": checks,
        "opened_count": electives.get("opened_count"),
        "not_opened_count": electives.get("not_opened_count"),
        "selection_mode": electives.get("selection_mode"),
        "selected_group_codes": sorted(selected_groups),
        "missing_credits_total": missing_credits_total,
    }


def _is_retryable_model_error(status_code: int, body_text: str, answer_text: str) -> bool:
    probes = [
        "400 bad request",
        "503 service unavailable",
        "<response [400 bad request]>",
        "<response [503 service unavailable]>",
    ]
    blob = f"{status_code} {body_text} {answer_text}".lower()
    return any(p in blob for p in probes)


def _assert_turn(turn_id: str, answer: str) -> Dict[str, Any]:
    answer_codes = set(_extract_codes(answer))
    checks: List[Dict[str, Any]] = []

    if turn_id == "specialized_opened":
        checks.append(
            {
                "name": "contains_broad_specialized_opened_list_not_just_three",
                "pass": len(
                    answer_codes
                    & {
                        "INT3117",
                        "INT3120",
                        "INT3306",
                        "INT3323",
                        "INT3230E",
                        "INT3229E",
                        "INT3121",
                        "INT3406",
                        "INT3123",
                        "INT3419",
                        "INT3420E",
                        "INT3403",
                        "INT3412E",
                        "INT2041",
                    }
                )
                >= 8,
                "details": sorted(answer_codes),
            }
        )
    elif turn_id == "full_list":
        checks.append(
            {
                "name": "mentions_all_groups_v21_v22_v23_v24_v3",
                "pass": _has_any(answer, ["V.2.1", "V.2.2", "V.2.3", "V.2.4", "V.3"]),
                "details": answer[:500],
            }
        )
        checks.append(
            {
                "name": "includes_mobile_and_vision_related_codes",
                "pass": {"INT3120", "INT3412E"}.issubset(answer_codes),
                "details": sorted(answer_codes),
            }
        )
    elif turn_id == "vision_teacher_schedule":
        checks.append(
            {
                "name": "contains_teacher_and_schedule_markers",
                "pass": _has_any(answer, ["le thanh ha", "giang vien", "day"])
                and _has_any(answer, ["thu", "ca", "tiet", "phong"]),
                "details": answer[:500],
            }
        )
    elif turn_id == "image_opened_and_in_curriculum":
        checks.append(
            {
                "name": "int3404e_in_answer",
                "pass": "INT3404E" in answer_codes or _has_any(answer, ["INT3404E"]),
                "details": sorted(answer_codes),
            }
        )
        checks.append(
            {
                "name": "states_not_opened_this_term",
                "pass": _has_any(answer, ["khong mo lop", "chua mo", "khong mo"]),
                "details": answer[:500],
            }
        )
        checks.append(
            {
                "name": "states_belongs_to_curriculum_v24_or_hci_group",
                "pass": _has_any(answer, ["V.2.4", "tuong tac nguoi-may", "tuong tac nguoi may"]),
                "details": answer[:500],
            }
        )
    else:
        checks.append({"name": "generic_non_empty_answer", "pass": bool(answer.strip()), "details": ""})

    passed = all(c["pass"] for c in checks)
    return {"pass": passed, "checks": checks}
def _ask_with_retry(
    base_url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    timeout: int = 240,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for idx in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            resp = requests.post(f"{base_url.rstrip('/')}/ask", json=payload, timeout=timeout)
            elapsed = round(time.perf_counter() - started, 2)
            body_text = resp.text or ""
            answer_text = ""
            selected_program_id = None
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    obj = resp.json()
                    answer_text = str(obj.get("answer") or "")
                    selected_program_id = obj.get("selected_program_id")
                except Exception:  # noqa: BLE001
                    pass

            retryable = _is_retryable_model_error(resp.status_code, body_text, answer_text)
            attempts.append(
                {
                    "attempt": idx,
                    "status_code": resp.status_code,
                    "elapsed_seconds": elapsed,
                    "retryable_model_error": retryable,
                    "body_preview": body_text[:600],
                    "answer_preview": answer_text[:600],
                }
            )
            if resp.status_code == 200 and not retryable:
                return {
                    "ok": True,
                    "attempts": attempts,
                    "answer": answer_text,
                    "selected_program_id": selected_program_id,
                }
            if idx >= max_retries or not retryable:
                return {
                    "ok": False,
                    "attempts": attempts,
                    "error": f"http_{resp.status_code}",
                    "answer": answer_text,
                }
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.perf_counter() - started, 2)
            attempts.append(
                {
                    "attempt": idx,
                    "status_code": None,
                    "elapsed_seconds": elapsed,
                    "retryable_model_error": False,
                    "body_preview": str(exc)[:600],
                    "answer_preview": "",
                }
            )
            if idx >= max_retries:
                return {"ok": False, "attempts": attempts, "error": str(exc), "answer": ""}

    return {"ok": False, "attempts": attempts, "error": "unknown_retry_state", "answer": ""}


def _replay_conversation(
    app_url: str,
    session_id: str,
    program_id: str,
    file_ids: List[str],
) -> Dict[str, Any]:
    turns = [
        {
            "id": "missing_subjects",
            "query": "toi can ban kiem tra giup toi xem toi con thieu nhung mon nao theo chuong trinh dao tao voi",
            "assertion": "generic",
        },
        {
            "id": "schedule_plan",
            "query": "toi can ban giup toi lap lich cho khoa luan va du cac mon kien thuc chung con lai mon tu chon tu dang ky giup toi sao cho du tin chi la duoc roi gui toi lich hoc",
            "assertion": "generic",
        },
        {
            "id": "his_teachers",
            "query": "ve mon Lich su Dang ki nay co nhung ai day",
            "assertion": "generic",
        },
        {
            "id": "teacher_class_schedule",
            "query": "co Vu Thi Thu Ha day nhung lop nao lich hoc ra sao",
            "assertion": "generic",
        },
        {
            "id": "specialized_opened",
            "query": "y toi la cac mon tu chon theo chuyen nganh chuong trinh dao tao cua toi ma",
            "assertion": "specialized_opened",
        },
        {
            "id": "full_list",
            "query": "day la tat ca cac mon a toi tuong con thi giac may voi phat trien ung dung mobile cac thu ma liet ke toan bo giup toi",
            "assertion": "full_list",
        },
        {
            "id": "opened_in_specialized_list",
            "query": "the co tat ca nhung mon nao trong cho nay mo lop",
            "assertion": "specialized_opened",
        },
        {
            "id": "vision_teacher_schedule",
            "query": "mon thi giac may do ai day va duoc day vao nhung hom nao",
            "assertion": "vision_teacher_schedule",
        },
        {
            "id": "image_opened_and_in_curriculum",
            "query": "the mon xu li anh ki nay co mo khong va ma mon nay co nam trong khung ctdt cua toi khong",
            "assertion": "image_opened_and_in_curriculum",
        },
    ]

    results: List[Dict[str, Any]] = []
    blockers: List[str] = []
    for idx, turn in enumerate(turns, start=1):
        payload = {
            "query": turn["query"],
            "session_id": session_id,
            "file_ids": file_ids,
            "program_id": program_id,
            "allow_web_search": False,
        }
        resp = _ask_with_retry(app_url, payload, max_retries=3)
        turn_row: Dict[str, Any] = {
            "index": idx,
            "id": turn["id"],
            "query": turn["query"],
            "ok": bool(resp.get("ok")),
            "attempts": resp.get("attempts") or [],
            "answer": resp.get("answer") or "",
            "error": resp.get("error"),
        }
        if not resp.get("ok"):
            blockers.append(f"{turn['id']}: request_failed_after_retry ({resp.get('error')})")
            turn_row["assertion"] = {"pass": False, "checks": []}
            results.append(turn_row)
            continue

        if turn["assertion"] == "generic":
            assertion = {"pass": bool((resp.get("answer") or "").strip()), "checks": []}
        else:
            assertion = _assert_turn(turn["assertion"], str(resp.get("answer") or ""))
            if not assertion["pass"]:
                blockers.append(f"{turn['id']}: semantic_assertion_failed")
        turn_row["assertion"] = assertion
        results.append(turn_row)

    return {
        "status": "pass" if not blockers else "fail",
        "turns": results,
        "blockers": blockers,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Pre-merge Full Gate Report (CS_2022 + 2 transcripts)")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report.get('generated_at_utc')}`")
    lines.append(f"- verdict: `{report.get('verdict')}`")
    lines.append(f"- session_id: `{report.get('config', {}).get('session_id')}`")
    lines.append("")

    lines.append("## Preflight")
    lines.append("```json")
    lines.append(json.dumps(report.get("preflight"), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Deterministic")
    lines.append("```json")
    lines.append(json.dumps(report.get("deterministic"), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Replay")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "status": report.get("replay", {}).get("status"),
                "blockers": report.get("replay", {}).get("blockers"),
                "turn_count": len(report.get("replay", {}).get("turns") or []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")

    for turn in report.get("replay", {}).get("turns") or []:
        lines.append(f"### Turn {turn.get('index')}: {turn.get('id')}")
        lines.append(f"- query: `{turn.get('query')}`")
        lines.append(f"- ok: `{turn.get('ok')}`")
        if turn.get("error"):
            lines.append(f"- error: `{turn.get('error')}`")
        lines.append("- attempts:")
        lines.append("```json")
        lines.append(json.dumps(turn.get("attempts"), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("- assertion:")
        lines.append("```json")
        lines.append(json.dumps(turn.get("assertion"), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("- answer:")
        lines.append("```text")
        lines.append(str(turn.get("answer") or ""))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pre-merge full gate for cs_2022 with two transcript PDFs and replayed conversation."
    )
    parser.add_argument("--app-url", default="http://127.0.0.1:9000")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8000")
    parser.add_argument("--program-id", default="cs_2022")
    parser.add_argument("--file-id-1", default="1_9e2842aa.pdf")
    parser.add_argument("--file-id-2", default="2_416655f7.pdf")
    parser.add_argument("--session-id", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stamp = _utc_stamp()
    session_id = args.session_id or f"premerge_cs2022_{stamp}_{uuid4().hex[:8]}"
    file_ids = [args.file_id_1, args.file_id_2]

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "app_url": args.app_url,
            "mcp_url": args.mcp_url,
            "program_id": args.program_id,
            "file_ids": file_ids,
            "session_id": session_id,
            "retry_policy": "max_3_for_400_503_then_block",
        },
    }

    preflight = {
        "env_keys": _load_key_status(),
        "app_health": {},
        "mcp_health": {},
    }
    app_ok, app_note = _http_ok(f"{args.app_url.rstrip('/')}/files", timeout=20)
    preflight["app_health"] = {"ok": app_ok, "note": app_note}

    mcp_ok, mcp_note = _http_ok(f"{args.mcp_url.rstrip('/')}/mcp/discover")
    preflight["mcp_health"] = {"ok": mcp_ok, "note": mcp_note}
    report["preflight"] = preflight

    if not (app_ok and mcp_ok):
        report["deterministic"] = {"status": "blocked", "checks": []}
        report["replay"] = {"status": "blocked", "turns": [], "blockers": ["service_not_ready"]}
        report["verdict"] = "NO-GO"
    else:
        mcp = MCPClient(args.mcp_url)
        deterministic = _deterministic_checks(
            mcp=mcp,
            session_id=session_id,
            program_id=args.program_id,
            file_ids=file_ids,
            expected_opened=[
                "INT3117",
                "INT3120",
                "INT3306",
                "INT3323",
                "INT3230E",
                "INT3229E",
                "INT3121",
                "INT3406",
                "INT3123",
                "INT3419",
                "INT3420E",
                "INT3403",
                "INT3412E",
                "INT2041",
                "INT3418",
                "INT3102",
                "INT3103",
            ],
        )
        report["deterministic"] = deterministic

        replay = _replay_conversation(
            app_url=args.app_url,
            session_id=session_id,
            program_id=args.program_id,
            file_ids=file_ids,
        )
        report["replay"] = replay

        report["verdict"] = (
            "GO"
            if deterministic.get("status") == "pass" and replay.get("status") == "pass"
            else "NO-GO"
        )

    out_dir = _ensure_reports_dir()
    json_path = out_dir / f"premerge_full_gate_cs2022_{stamp}.json"
    md_path = out_dir / f"premerge_full_gate_cs2022_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"[report] json={json_path}")
    print(f"[report] md={md_path}")
    print(f"[verdict] {report.get('verdict')}")
    return 0 if report.get("verdict") == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

