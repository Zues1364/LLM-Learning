from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import queue as std_queue
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_server.server as server  # noqa: E402

_MATRIX_PATH = ROOT / "test" / "run_ctdt_matrix_fake_transcript.py"
_spec = importlib.util.spec_from_file_location("matrix_runner", _MATRIX_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load matrix helper module from {_MATRIX_PATH}")
_matrix_runner = importlib.util.module_from_spec(_spec)
sys.modules["matrix_runner"] = _matrix_runner
_spec.loader.exec_module(_matrix_runner)

_build_fake_transcript = _matrix_runner._build_fake_transcript
_extract_subjects = _matrix_runner._extract_subjects
_validate_advisor_content = _matrix_runner._validate_advisor_content

DEEP_PROGRAMS = [
    "cs_2016",
    "cs_2019",
    "cs_2022",
    "cs_2025",
    "it_2015",
    "it_2019",
    "it_2022",
    "it_2025",
    "is_2022",
    "network_2025",
    "ai_2025",
    "ce_2022",
]

DEFAULT_QUERY = (
    "Mình còn thiếu bao nhiêu tín chỉ, thiếu môn nào theo chương trình này, "
    "môn nào đang mở kỳ này và gợi ý lịch học phù hợp."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep advisor IO checks with fake in-memory transcripts."
    )
    parser.add_argument(
        "--timeout-per-program",
        type=int,
        default=900,
        help="Advisor timeout per program in seconds (default: 900).",
    )
    parser.add_argument(
        "--programs",
        type=str,
        default="",
        help="Comma-separated program IDs. Empty means default deep set.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Composite advisor query used for all programs.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["in-process", "subprocess"],
        default="in-process",
        help="Advisor execution mode. in-process reuses cache and is more stable.",
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _consult_worker(
    query: str,
    file_ids: List[str],
    session_id: str,
    program_id: str,
    fake_transcript: Dict[str, Any],
    queue: mp.Queue,
) -> None:
    try:
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        import mcp_server.server as child_server  # noqa: WPS433

        original_analyze = child_server.analyze_transcript
        try:
            child_server.analyze_transcript = (
                lambda _ids, payload=fake_transcript: json.dumps(payload, ensure_ascii=False)
            )
            answer = child_server.consult_advisor(
                query=query,
                file_ids=file_ids,
                session_id=session_id,
                program_id=program_id,
            )
        finally:
            child_server.analyze_transcript = original_analyze
        queue.put({"ok": True, "answer": str(answer or "")})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def _consult_in_process(
    query: str,
    file_ids: List[str],
    session_id: str,
    program_id: str,
    fake_transcript: Dict[str, Any],
) -> Dict[str, Any]:
    original_analyze = server.analyze_transcript
    try:
        server.analyze_transcript = (
            lambda _ids, payload=fake_transcript: json.dumps(payload, ensure_ascii=False)
        )
        answer = server.consult_advisor(
            query=query,
            file_ids=file_ids,
            session_id=session_id,
            program_id=program_id,
        )
        return {"ok": True, "answer": str(answer or "")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        server.analyze_transcript = original_analyze


def _render_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Deep Advisor IO Log")
    lines.append("")
    lines.append(f"- generated_at_utc: `{summary['generated_at_utc']}`")
    lines.append(f"- program_count: `{summary['program_count']}`")
    lines.append(f"- execution_mode: `{summary['execution_mode']}`")
    lines.append(f"- timeout_per_program: `{summary['timeout_per_program']}`")
    lines.append(f"- status_breakdown: `{json.dumps(summary['status_breakdown'], ensure_ascii=False)}`")
    lines.append("")

    for row in summary.get("results") or []:
        lines.append(f"## {row.get('program_id')} [{row.get('status')}]")
        if row.get("errors"):
            lines.append(f"- errors: {row['errors']}")
        if row.get("input"):
            lines.append("- input:")
            lines.append("```json")
            lines.append(json.dumps(row["input"], ensure_ascii=False, indent=2))
            lines.append("```")
        if row.get("output"):
            lines.append("- output_meta:")
            lines.append("```json")
            lines.append(
                json.dumps(
                    {
                        "advisor_status": (row["output"] or {}).get("status"),
                        "advisor_checks": (row["output"] or {}).get("advisor_checks"),
                        "answer_length": (row["output"] or {}).get("answer_length"),
                        "advisor_seconds": (row["output"] or {}).get("advisor_seconds"),
                        "error": (row["output"] or {}).get("error"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            lines.append("```")
            lines.append("- answer_text:")
            lines.append("```text")
            lines.append((row["output"] or {}).get("answer_text") or "")
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()
    selected_programs = [
        prog.strip() for prog in (args.programs or "").split(",") if prog.strip()
    ] or DEEP_PROGRAMS

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    timeout_seconds = max(1, int(args.timeout_per_program))

    for idx, pid in enumerate(selected_programs, start=1):
        print(f"[{idx}/{len(selected_programs)}] {pid} ...")
        item: Dict[str, Any] = {
            "program_id": pid,
            "query": args.query,
            "status": "unknown",
            "errors": [],
            "input": {},
            "output": {},
        }
        try:
            lookup = json.loads(server.get_curriculum_lookup(program_id=pid))
            if "error" in lookup:
                raise RuntimeError(f"lookup_error:{lookup['error']}")

            groups = lookup.get("groups") or {}
            subjects = _extract_subjects(groups)
            fake_transcript = _build_fake_transcript(pid, subjects, ratio=0.65)

            curriculum = server.analyze_curriculum(program_hint=pid)
            missing_info = server.compute_missing_subjects(fake_transcript, curriculum)
            credit_summary = missing_info.get("credit_summary") or {}

            electives = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id=pid))
            file_ids = [f"{pid}_1_ff18aead.pdf", f"{pid}_2_0a2fefa1.pdf"]

            item["input"] = {
                "file_ids": file_ids,
                "transcript_total_credits": fake_transcript.get("overview", {}).get("total_credits_accumulated"),
                "completed_subject_count": len(fake_transcript.get("completed_subjects") or []),
                "curriculum_source": curriculum.get("source_path"),
                "curriculum_total_credits": curriculum.get("total_credits"),
                "missing_credits": credit_summary.get("total_missing_credits"),
                "opened_electives_count": electives.get("opened_count"),
                "selection_mode": electives.get("selection_mode"),
                "selected_group_codes": electives.get("selected_group_codes"),
            }

            advisor_start = time.perf_counter()
            if args.execution_mode == "in-process":
                worker_result = _consult_in_process(
                    query=args.query,
                    file_ids=file_ids,
                    session_id=f"deep_log_{pid}",
                    program_id=pid,
                    fake_transcript=fake_transcript,
                )
            else:
                queue: mp.Queue = mp.Queue()
                proc = mp.Process(
                    target=_consult_worker,
                    args=(
                        args.query,
                        file_ids,
                        f"deep_log_{pid}",
                        pid,
                        fake_transcript,
                        queue,
                    ),
                )
                proc.start()
                proc.join(timeout=timeout_seconds)

                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    elapsed = round(time.perf_counter() - advisor_start, 2)
                    item["status"] = "error"
                    item["errors"].append(f"advisor_timeout_per_program_{timeout_seconds}s")
                    item["output"] = {
                        "status": "timeout",
                        "advisor_checks": None,
                        "answer_text": "",
                        "answer_length": 0,
                        "advisor_seconds": elapsed,
                        "error": f"timeout_after_{elapsed}s",
                    }
                    results.append(item)
                    continue

                try:
                    worker_result = queue.get_nowait()
                except std_queue.Empty:
                    worker_result = {
                        "ok": False,
                        "error": f"advisor_no_worker_result(exitcode={proc.exitcode})",
                    }

            if not worker_result.get("ok"):
                raise RuntimeError(str(worker_result.get("error") or "advisor_unknown_error"))

            answer = str(worker_result.get("answer") or "")
            elapsed = round(time.perf_counter() - advisor_start, 2)
            ok, checks = _validate_advisor_content(answer)

            item["status"] = "pass" if ok else "fail_intent"
            item["output"] = {
                "status": item["status"],
                "advisor_checks": checks,
                "answer_text": answer,
                "answer_length": len(answer),
                "advisor_seconds": elapsed,
                "error": None,
            }
        except Exception as exc:
            item["status"] = "error"
            item["errors"].append(str(exc))
            if not item.get("output"):
                item["output"] = {
                    "status": "error",
                    "advisor_checks": None,
                    "answer_text": "",
                    "answer_length": 0,
                    "advisor_seconds": None,
                    "error": str(exc),
                }
        results.append(item)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "program_count": len(results),
        "execution_mode": args.execution_mode,
        "timeout_per_program": timeout_seconds,
        "status_breakdown": {},
        "results": results,
    }
    for row in results:
        status = row.get("status", "unknown")
        summary["status_breakdown"][status] = summary["status_breakdown"].get(status, 0) + 1

    stamp = _utc_stamp()
    json_path = reports_dir / f"deep_advisor_io_{stamp}.json"
    md_path = reports_dir / f"deep_advisor_io_{stamp}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(summary), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"MD report: {md_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
