"""
Debug utility script.

Primary mode:
- Debug why elective subjects "opened_count" may be low in get_electives_with_schedule.

Legacy mode:
- Keep old missing-subject consistency check for transcript/curriculum.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"


def _ensure_import_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def _normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def _build_variants(code: str) -> List[str]:
    code = (code or "").strip().upper()
    if not code:
        return []
    base = code.split(".")[-1]
    variants = [code, base]
    for item in [code, base]:
        if item.endswith("E"):
            variants.append(item[:-1])
        else:
            variants.append(item + "E")
    # Keep order but unique
    seen = set()
    result = []
    for v in variants:
        if v not in seen and v:
            seen.add(v)
            result.append(v)
    return result


def _load_schedule_texts() -> List[Dict[str, Any]]:
    _ensure_import_path()
    from utils import process_pdf  # pylint: disable=import-outside-toplevel

    schedule_dir = ROOT_DIR / "data" / "resources" / "pdfs"
    patterns = ["*TKB*.pdf", "*THỜI KHÓA BIỂU*.pdf", "*thoi khoa bieu*.pdf", "*PHỤ LỤC*.pdf"]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(schedule_dir.glob(pattern))
    # unique preserving order
    uniq: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(path)

    rows: List[Dict[str, Any]] = []
    for path in uniq:
        try:
            docs = process_pdf(str(path))
            text = "\n".join(d.page_content for d in docs)
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "text_len": len(text),
                    "text": text,
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "text_len": 0,
                    "text": "",
                    "error": str(exc),
                }
            )
    return rows


def debug_electives(program_id: str, sample: int = 20) -> Dict[str, Any]:
    _ensure_import_path()
    import mcp_server.server as server  # pylint: disable=import-outside-toplevel

    raw_no_schedule = json.loads(server.get_electives_with_schedule(check_schedule=False, program_id=program_id))
    all_electives = raw_no_schedule.get("all_electives", [])

    raw_with_schedule = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id=program_id))
    opened = raw_with_schedule.get("opened", [])
    not_opened = raw_with_schedule.get("not_opened", [])

    schedule_rows = _load_schedule_texts()
    best = max(schedule_rows, key=lambda x: x.get("text_len", 0), default=None)
    best_text = (best or {}).get("text", "")
    best_upper = best_text.upper()
    best_norm = _normalize_code(best_text)

    regex_codes = sorted(set(re.findall(r"\b[A-Z]{2,6}\d{4}[A-Z]?\b", best_upper)))

    elective_codes = [str(item.get("code") or "").strip() for item in all_electives]
    elective_codes = [x for x in elective_codes if x]

    strict_hits = []
    soft_hits = []
    missed = []
    for code in elective_codes:
        variants = _build_variants(code)
        strict = any(v in best_upper for v in variants)
        soft = any(_normalize_code(v) in best_norm for v in variants)
        if strict:
            strict_hits.append(code)
        if soft:
            soft_hits.append(code)
        if not strict:
            missed.append(code)

    opened_codes = [x.get("code") for x in opened if x.get("code")]
    not_opened_codes = [x.get("code") for x in not_opened if x.get("code")]

    # Prefix stats to quickly see what group of codes is missing
    prefix_counter = Counter()
    for code in not_opened_codes:
        base = code.split(".")[-1].upper()
        m = re.match(r"([A-Z]+)", base)
        if m:
            prefix_counter[m.group(1)] += 1

    possible_logic_issue = len(strict_hits) > len(opened_codes)

    return {
        "program_id": program_id,
        "summary": {
            "electives_total": len(all_electives),
            "opened_count_reported": len(opened),
            "not_opened_count_reported": len(not_opened),
            "strict_hits_in_best_tkb": len(strict_hits),
            "soft_hits_in_best_tkb": len(soft_hits),
            "possible_logic_issue": possible_logic_issue,
        },
        "opened_codes": opened_codes,
        "not_opened_codes_sample": not_opened_codes[:sample],
        "prefix_distribution_not_opened": prefix_counter,
        "schedule_files": [
            {k: row.get(k) for k in ["name", "text_len", "error"]}
            for row in sorted(schedule_rows, key=lambda x: x.get("text_len", 0), reverse=True)
        ],
        "best_schedule_file": {
            "name": (best or {}).get("name"),
            "text_len": (best or {}).get("text_len"),
        },
        "codes_detected_in_best_tkb_count": len(regex_codes),
        "codes_detected_in_best_tkb_sample": regex_codes[:sample],
        "notes": [
            "If strict_hits_in_best_tkb ~= opened_count_reported, low opened_count is likely from data (few codes in current TKB), not matching logic.",
            "possible_logic_issue=True means code matching may be missing valid subjects.",
        ],
    }


def debug_missing_legacy() -> Dict[str, Any]:
    _ensure_import_path()
    from mcp_server.server import (  # pylint: disable=import-outside-toplevel
        _build_completed_subjects,
        analyze_curriculum,
        analyze_transcript,
        compute_missing_subjects,
    )

    pdf1 = r"D:\LLM\LLM Learning\data\pdfs\ĐIỂM_1_ef61b158.pdf"
    pdf2 = r"D:\LLM\LLM Learning\data\pdfs\ĐIỂM_2_68657c33.pdf"

    transcript_json = analyze_transcript([pdf1, pdf2])
    transcript_data = json.loads(transcript_json)

    completed_map = _build_completed_subjects(transcript_data.get("semesters") or [])
    curriculum = analyze_curriculum("Khoa học máy tính")
    missing_info = compute_missing_subjects(transcript_data, curriculum)
    missing_list = missing_info.get("missing") or []

    completed_codes = list(completed_map.keys())
    missing_codes = [m.get("code") for m in missing_list]

    check_codes = ["MAT1041", "EPN1095", "INT1008", "INT1009"]
    bug_results = {}
    for code in check_codes:
        norm_code = code.upper().replace(" ", "")
        in_completed = any(k.upper().replace(" ", "") == norm_code for k in completed_codes)
        in_missing = code in missing_codes
        bug_results[code] = {
            "in_completed": in_completed,
            "in_missing": in_missing,
            "bug": in_completed and in_missing,
        }

    return {
        "completed_count": len(completed_codes),
        "missing_count": len(missing_codes),
        "missing_codes_sample": missing_codes[:15],
        "bug_check": bug_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug helper for core checks.")
    parser.add_argument(
        "--mode",
        choices=["electives", "missing"],
        default="electives",
        help="Debug mode. Default: electives",
    )
    parser.add_argument("--program-id", default="it_2025", help="Program ID for electives mode.")
    parser.add_argument("--sample", type=int, default=20, help="Sample size in output.")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.mode == "electives":
        result = debug_electives(program_id=args.program_id, sample=args.sample)
    else:
        result = debug_missing_legacy()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: dict(x)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
