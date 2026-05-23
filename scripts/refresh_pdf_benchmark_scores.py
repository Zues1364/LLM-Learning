from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from benchmark_pdf_extraction import PDF_SCORE_WEIGHTS, compute_pdf_score, render_markdown_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh score_pdf fields in an existing PDF benchmark report without rerunning OCR."
    )
    parser.add_argument(
        "--input-json",
        default=str(REPO_ROOT / "reports" / "pdf_extraction_benchmark_latest.json"),
        help="Existing PDF extraction benchmark JSON report.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "reports" / "pdf_extraction_benchmark_latest.json"),
        help="Output JSON path. Default overwrites the latest report.",
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "pdf_extraction_benchmark_latest.md"),
        help="Output Markdown path. Default overwrites the latest report.",
    )
    return parser.parse_args()


def _refresh_summary_row(row: Dict[str, Any]) -> None:
    row["score_pdf"] = round(
        compute_pdf_score(
            row.get("key_field_accuracy") or 0.0,
            row.get("row_exact_accuracy") or 0.0,
            row.get("cell_f1") or 0.0,
        ),
        4,
    )


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    payload["score_weights"] = dict(PDF_SCORE_WEIGHTS)
    payload["score_weights_refreshed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for row in payload.get("summary_by_method") or []:
        _refresh_summary_row(row)

    for block in payload.get("summary_by_type") or []:
        for row in block.get("rows") or []:
            _refresh_summary_row(row)

    for item in payload.get("case_results") or []:
        score = item.get("score") or {}
        score["score_pdf"] = round(
            compute_pdf_score(
                score.get("key_field_accuracy") or 0.0,
                score.get("row_exact_accuracy") or 0.0,
                score.get("cell_f1") or 0.0,
            ),
            4,
        )

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown_report(payload), encoding="utf-8")

    print(f"[pdf-refresh] json_report={output_json}")
    print(f"[pdf-refresh] md_report={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
