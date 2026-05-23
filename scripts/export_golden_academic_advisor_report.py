from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "golden_academic_advisor.jsonl"
REPORTS_DIR = ROOT / "reports"
LATEST_EVAL_REPORT = REPORTS_DIR / "eval_academic_advisor_latest.json"


DETAIL_FIELDS = [
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


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        item["_line_no"] = line_no
        cases.append(item)
    return cases


def load_latest_eval_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def group_eval_results(eval_report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_report.get("observations") or []:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        grouped[case_id].append(
            {
                "target": row.get("target"),
                "status": row.get("score", {}).get("status"),
                "source": row.get("source"),
                "status_code": row.get("status_code"),
                "latency_ms": row.get("latency_ms"),
                "answer": row.get("answer"),
                "failed_checks": [
                    {
                        "name": check.get("name"),
                        "details": check.get("details"),
                    }
                    for check in row.get("score", {}).get("checks", [])
                    if not check.get("pass")
                ],
            }
        )
    return grouped


def build_report(cases: list[dict[str, Any]], eval_report: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(case.get("category") or "uncategorized") for case in cases)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_by_case = group_eval_results(eval_report)
    for idx, case in enumerate(cases, start=1):
        category = str(case.get("category") or "uncategorized")
        case_id = str(case.get("id") or "")
        grouped[category].append(
            {
                "index": idx,
                "id": case_id,
                "line_no": case.get("_line_no"),
                "query": case.get("query"),
                "details": {
                    key: case[key]
                    for key in DETAIL_FIELDS
                    if key in case and case.get(key) not in (None, "", [], {})
                },
                "results": eval_by_case.get(case_id, []),
            }
        )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(DATASET),
        "latest_eval_report": str(LATEST_EVAL_REPORT) if eval_report else None,
        "latest_eval_generated_at_utc": eval_report.get("generated_at_utc") if eval_report else None,
        "total_cases": len(cases),
        "category_counts": dict(counts),
        "categories": grouped,
    }


def clamp_text(value: Any, max_chars: int = 1800) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def status_mark(status: Any) -> str:
    value = str(status or "not_run").lower()
    if value == "pass":
        return "[PASS]"
    if value == "warn":
        return "[WARN]"
    if value == "fail":
        return "[FAIL]"
    return "[NOT RUN]"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Golden Academic Advisor Question Set")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Dataset: `{report['dataset']}`")
    if report.get("latest_eval_report"):
        lines.append(f"- Latest eval report: `{report['latest_eval_report']}`")
        lines.append(f"- Latest eval generated at UTC: `{report.get('latest_eval_generated_at_utc')}`")
    lines.append(f"- Total cases: `{report['total_cases']}`")
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Cases |")
    lines.append("| --- | ---: |")
    for category, count in report["category_counts"].items():
        lines.append(f"| `{category}` | {count} |")
    lines.append("")

    for category, items in report["categories"].items():
        lines.append(f"## {category}")
        lines.append("")
        for item in items:
            lines.append(
                f"### {item['index']}. `{item['id']}` (line {item['line_no']})"
            )
            lines.append("")
            lines.append(f"- Query: `{item['query']}`")
            details = item["details"]
            if details:
                for key, value in details.items():
                    rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                    lines.append(f"- {key}: `{rendered}`")
            results = item.get("results") or []
            if not results:
                lines.append("- Result: `[NOT RUN]`")
            for result in results:
                lines.append(
                    f"- Result: {status_mark(result.get('status'))} target=`{result.get('target')}` "
                    f"source=`{result.get('source')}` status=`{result.get('status_code')}` "
                    f"latency_ms=`{result.get('latency_ms')}`"
                )
                failed_checks = result.get("failed_checks") or []
                if failed_checks:
                    lines.append("  - Failed checks:")
                    for check in failed_checks[:5]:
                        lines.append(
                            f"    - `{check.get('name')}`: {json.dumps(check.get('details'), ensure_ascii=False)}"
                        )
                lines.append("  - Answer:")
                lines.append("")
                lines.append("```text")
                lines.append(clamp_text(result.get("answer")).replace("```", "'''"))
                lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"golden_academic_advisor_cases_{stamp}.json"
    md_path = REPORTS_DIR / f"golden_academic_advisor_cases_{stamp}.md"
    latest_json = REPORTS_DIR / "golden_academic_advisor_cases_latest.json"
    latest_md = REPORTS_DIR / "golden_academic_advisor_cases_latest.md"

    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    md_text = render_markdown(report)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def main() -> None:
    cases = load_cases(DATASET)
    eval_report = load_latest_eval_report(LATEST_EVAL_REPORT)
    report = build_report(cases, eval_report)
    json_path, md_path = write_report(report)
    print(f"cases={report['total_cases']}")
    print(f"json={json_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
