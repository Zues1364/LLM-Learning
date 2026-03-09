from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
REPORT_DIR = ROOT_DIR / "reports"


def _ensure_paths() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _extract_subjects(groups: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = set()
    subjects: List[Dict[str, Any]] = []
    for group_data in (groups or {}).values():
        for subj in group_data.get("subjects", []) or []:
            code = str(subj.get("code") or "").strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            credits = int(subj.get("credits") or 0)
            subjects.append(
                {
                    "code": code,
                    "name": str(subj.get("name") or code),
                    "credits": credits if credits > 0 else 3,
                }
            )
    subjects.sort(key=lambda x: x["code"])
    return subjects


def _split_semesters(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mid = max(1, len(subjects) // 2)
    sem1 = subjects[:mid]
    sem2 = subjects[mid:]
    return [
        {"semester_code": "251", "subjects": sem1},
        {"semester_code": "252", "subjects": sem2},
    ]


def _build_fake_transcript(program_id: str, subjects: List[Dict[str, Any]], ratio: float = 0.65) -> Dict[str, Any]:
    rng = random.Random(program_id)
    pool = subjects[:]
    rng.shuffle(pool)
    if not pool:
        pool = [{"code": "DUMMY1001", "name": "Dummy", "credits": 3}]

    target = max(1, int(len(pool) * ratio))
    completed = pool[:target]

    e_subjects = [s for s in pool if str(s.get("code", "")).upper().endswith("E")]
    if e_subjects:
        keep_missing = e_subjects[0]
        completed = [s for s in completed if s["code"] != keep_missing["code"]]
        if not completed:
            completed = pool[:1]

    # Ensure at least one missing item.
    if len(completed) >= len(pool):
        completed = completed[:-1] or completed

    grade_cycle = [3.7, 3.3, 3.0, 2.6, 2.2, 1.8, 1.2]
    completed_rows: List[Dict[str, Any]] = []
    low_grade_forced = 0
    for idx, subj in enumerate(completed):
        grade = grade_cycle[idx % len(grade_cycle)]
        if low_grade_forced < 2 and idx % 7 == 0:
            grade = 1.2
            low_grade_forced += 1
        completed_rows.append(
            {
                "code": subj["code"],
                "name": subj["name"],
                "credits": int(subj["credits"]),
                "grade_4": float(grade),
            }
        )

    semesters = _split_semesters(completed_rows)
    total_credits = sum(int(s.get("credits") or 0) for s in completed_rows)

    return {
        "student_info": {
            "name": f"Fake Student {program_id}",
            "id": f"FAKE-{program_id.upper()}",
            "class": f"FAKE-{program_id.upper()}",
            "major": program_id,
            "program_hint": program_id,
        },
        "semesters": semesters,
        "completed_subjects": completed_rows,
        "overview": {
            "total_credits_accumulated": total_credits,
            "raw_gpa_4": round(sum(s["grade_4"] for s in completed_rows) / max(1, len(completed_rows)), 3),
        },
    }


def _read_programs(server_module: Any, refresh: bool = True) -> List[Dict[str, Any]]:
    raw = server_module.get_available_programs(refresh=refresh)
    data = json.loads(raw)
    return list(data.get("programs") or [])


def _get_git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _classify_error(err: Exception) -> str:
    txt = str(err or "")
    lower = txt.lower()
    if "api key" in lower or "invalid_argument" in lower or "400" in lower:
        return "advisor_skipped"
    if "timeout" in lower or "timed out" in lower:
        return "advisor_skipped"
    return "advisor_error"


def _validate_advisor_content(answer: str) -> Tuple[bool, Dict[str, bool]]:
    norm = _norm(answer)
    checks = {
        "has_missing": any(k in norm for k in ["thiếu", "thieu", "tín chỉ", "tin chi"]),
        "has_courses": any(k in norm for k in ["môn", "mon", "học phần", "hoc phan"]),
        "has_opened": any(k in norm for k in ["mở", "mo", "lớp", "lop", "học kỳ", "hoc ky"]),
        "has_schedule": any(k in norm for k in ["lịch", "lich", "thứ", "ca ", "tiết", "tiet"]),
    }
    return all(checks.values()), checks


@dataclass
class ProgramResult:
    program_id: str
    display_name: str
    smoke_status: str
    missing_credits: int
    opened_electives_count: int
    schedule_probe_count: int
    selection_mode: str
    selected_group_count: int
    advisor_status: str
    advisor_checks: Dict[str, bool]
    errors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "display_name": self.display_name,
            "smoke_status": self.smoke_status,
            "missing_credits": self.missing_credits,
            "opened_electives_count": self.opened_electives_count,
            "schedule_probe_count": self.schedule_probe_count,
            "selection_mode": self.selection_mode,
            "selected_group_count": self.selected_group_count,
            "advisor_status": self.advisor_status,
            "advisor_checks": self.advisor_checks,
            "errors": self.errors,
        }


def run_matrix(limit: int | None, run_deep: bool, deep_programs: List[str], ratio: float) -> Dict[str, Any]:
    _ensure_paths()
    import mcp_server.server as server  # pylint: disable=import-outside-toplevel

    programs = _read_programs(server, refresh=True)
    if limit:
        programs = programs[:limit]

    registry = server._scan_curriculum_programs(force_refresh=False)  # noqa: SLF001
    deep_set = set(deep_programs)
    query = "Mình còn thiếu bao nhiêu tín chỉ, thiếu môn nào theo chương trình này, môn nào đang mở kỳ này và gợi ý lịch học phù hợp."

    results: List[ProgramResult] = []
    errors_summary: Dict[str, int] = {}

    for idx, item in enumerate(programs, start=1):
        pid = str(item.get("id") or "").strip()
        display = str(item.get("display_name") or pid)
        if not pid:
            continue

        print(f"[{idx}/{len(programs)}] {pid} ...")
        local_errors: List[str] = []
        missing_credits = -1
        opened_count = 0
        schedule_probe_count = 0
        selection_mode = ""
        selected_group_count = 0
        advisor_status = "not_run"
        advisor_checks: Dict[str, bool] = {}
        smoke_status = "pass"

        try:
            lookup_raw = server.get_curriculum_lookup(program_id=pid)
            lookup = json.loads(lookup_raw)
            if "error" in lookup:
                raise RuntimeError(f"curriculum_parse_error:{lookup['error']}")
            if "total_credits_required" not in lookup or "groups" not in lookup:
                raise RuntimeError("curriculum_parse_error:lookup_shape_invalid")
            groups = lookup.get("groups") or {}
            subjects = _extract_subjects(groups)

            curriculum = server.analyze_curriculum(program_hint=pid)
            source_path = str(curriculum.get("source_path") or "")
            expected_entry = registry.get(pid)
            if expected_entry and source_path:
                if Path(source_path).name != Path(expected_entry["file_path"]).name:
                    raise RuntimeError(
                        f"program_mapping_mismatch:{Path(source_path).name}!={Path(expected_entry['file_path']).name}"
                    )

            fake_transcript = _build_fake_transcript(pid, subjects, ratio=ratio)
            missing_info = server.compute_missing_subjects(fake_transcript, curriculum)
            credit_summary = missing_info.get("credit_summary") or {}
            missing_credits = int(credit_summary.get("total_missing_credits") or 0)
            if missing_credits < 0:
                raise RuntimeError(f"missing_credit_negative:{missing_credits}")

            electives = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id=pid))
            if "error" in electives:
                raise RuntimeError(f"group_selection_error:{electives['error']}")
            if "opened" not in electives or "not_opened" not in electives:
                raise RuntimeError("schedule_match_error:elective_shape_invalid")
            opened_count = int(electives.get("opened_count") or len(electives.get("opened") or []))
            schedule_probe_count = len(electives.get("opened") or []) + len(electives.get("not_opened") or [])
            selection_mode = str(electives.get("selection_mode") or "")
            selected_group_codes = electives.get("selected_group_codes") or []
            if isinstance(selected_group_codes, list):
                selected_group_count = len(selected_group_codes)

            if run_deep and pid in deep_set:
                fake1 = f"{pid}_1_ff18aead.pdf"
                fake2 = f"{pid}_2_0a2fefa1.pdf"
                original_analyze = server.analyze_transcript
                try:
                    server.analyze_transcript = lambda _ids, payload=fake_transcript: json.dumps(payload, ensure_ascii=False)
                    answer = server.consult_advisor(
                        query=query,
                        file_ids=[fake1, fake2],
                        session_id=f"matrix_{pid}",
                        program_id=pid,
                    )
                    ok, checks = _validate_advisor_content(str(answer or ""))
                    advisor_checks = checks
                    advisor_status = "pass" if ok else "fail_intent"
                except Exception as e:  # pragma: no cover - external API/runtime dependent
                    advisor_status = _classify_error(e)
                    local_errors.append(f"advisor:{e}")
                finally:
                    server.analyze_transcript = original_analyze

        except Exception as e:
            smoke_status = "fail"
            local_errors.append(str(e))

        for err in local_errors:
            key = err.split(":", 1)[0]
            errors_summary[key] = errors_summary.get(key, 0) + 1

        results.append(
            ProgramResult(
                program_id=pid,
                display_name=display,
                smoke_status=smoke_status,
                missing_credits=missing_credits,
                opened_electives_count=opened_count,
                schedule_probe_count=schedule_probe_count,
                selection_mode=selection_mode,
                selected_group_count=selected_group_count,
                advisor_status=advisor_status,
                advisor_checks=advisor_checks,
                errors=local_errors,
            )
        )

    return {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "git_head": _get_git_head(),
        "program_count": len(results),
        "deep_programs_requested": deep_programs,
        "results": [r.as_dict() for r in results],
        "errors_summary": errors_summary,
    }


def _to_markdown(report: Dict[str, Any]) -> str:
    results = report.get("results") or []
    smoke_fail = [r for r in results if r.get("smoke_status") != "pass"]
    advisor_fail = [
        r
        for r in results
        if r.get("advisor_status") not in {"not_run", "pass", "advisor_skipped"}
    ]
    advisor_skipped = [r for r in results if r.get("advisor_status") == "advisor_skipped"]

    lines = [
        "# CTDT Matrix Fake Transcript Report",
        "",
        f"- generated_at_utc: `{report.get('generated_at_utc')}`",
        f"- git_head: `{report.get('git_head')}`",
        f"- program_count: `{report.get('program_count')}`",
        f"- smoke_fail_count: `{len(smoke_fail)}`",
        f"- advisor_fail_count: `{len(advisor_fail)}`",
        f"- advisor_skipped_count: `{len(advisor_skipped)}`",
        "",
        "## Error Summary",
    ]

    err_summary = report.get("errors_summary") or {}
    if err_summary:
        for key, count in sorted(err_summary.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")

    lines += ["", "## Failed Programs (Smoke)"]
    if smoke_fail:
        for r in smoke_fail:
            lines.append(f"- `{r['program_id']}`: {', '.join(r.get('errors') or [])}")
    else:
        lines.append("- none")

    lines += ["", "## Failed Programs (Advisor Intent)"]
    if advisor_fail:
        for r in advisor_fail:
            lines.append(f"- `{r['program_id']}`: status={r.get('advisor_status')} checks={r.get('advisor_checks')}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTDT matrix test with fake in-memory transcript.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of programs for quick run (0 = all).")
    parser.add_argument("--skip-deep", action="store_true", help="Skip deep advisor checks.")
    parser.add_argument(
        "--deep-programs",
        default="cs_2016,cs_2019,cs_2022,cs_2025,it_2015,it_2019,it_2022,it_2025,is_2022,network_2025,ai_2025,ce_2022",
        help="Comma-separated deep-check program IDs.",
    )
    parser.add_argument("--ratio", type=float, default=0.65, help="Completed ratio for fake transcript (0..1).")
    parser.add_argument("--output-dir", default=str(REPORT_DIR), help="Report output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deep_programs = [x.strip() for x in str(args.deep_programs or "").split(",") if x.strip()]
    limit = args.limit if args.limit > 0 else None

    report = run_matrix(
        limit=limit,
        run_deep=not args.skip_deep,
        deep_programs=deep_programs,
        ratio=max(0.1, min(0.95, args.ratio)),
    )
    stamp = _utc_ts()
    json_path = out_dir / f"ctdt_matrix_fake_transcript_{stamp}.json"
    md_path = out_dir / f"ctdt_matrix_fake_transcript_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"MD report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
