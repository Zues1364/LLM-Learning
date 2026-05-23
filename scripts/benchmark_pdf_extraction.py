from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from mcp_server.structured_schedule_store import StructuredScheduleStore  # noqa: E402
from utils import normalize_for_match, process_pdf  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "evals" / "pdf_extraction"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_METHODS = [
    "pdfplumber_raw_text_only",
    "pdfplumber_text_plus_tables",
    "page_ocr_tesseract_only",
    "img2table_tesseract",
    "table_first_strict",
    "hybrid_current",
]

PDF_SCORE_WEIGHTS = {
    "key_field_accuracy": 0.35,
    "row_exact_accuracy": 0.35,
    "cell_f1": 0.30,
}

TRANSCRIPT_CODE_RE = re.compile(r"\b([A-Z]{2,4}\d{4}[A-Z]?)\b", re.IGNORECASE)
SEMESTER_RE = re.compile(r"\b(20\d{2}-20\d{2}-[12])\b")
FLOAT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TIME_RE = re.compile(r"\b\d{1,2}[:h]\d{2}(?:[’']?)\b", re.IGNORECASE)


@dataclass
class ExtractionResult:
    method: str
    pipeline: str
    text: str
    tables: List[List[List[str]]]
    latency_ms: float
    ocr_used: bool
    vision_used: bool
    notes: List[str]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def norm_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value))


def ascii_fold(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = normalize_for_match(text)
    return text


def try_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return None


def p50(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(statistics.median(values)), 2)


def p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.95)))
    return round(ordered[idx], 2)


def round4(value: float) -> float:
    return round(float(value), 4)


def compute_pdf_score(key_field_accuracy: float, row_exact_accuracy: float, cell_f1: float) -> float:
    return (
        PDF_SCORE_WEIGHTS["key_field_accuracy"] * float(key_field_accuracy)
        + PDF_SCORE_WEIGHTS["row_exact_accuracy"] * float(row_exact_accuracy)
        + PDF_SCORE_WEIGHTS["cell_f1"] * float(cell_f1)
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(dataset_dir: Path) -> List[Dict[str, Any]]:
    case_dir = dataset_dir / "cases"
    cases: List[Dict[str, Any]] = []
    for path in sorted(case_dir.glob("*.json")):
        item = read_json(path)
        item["_case_path"] = str(path)
        cases.append(item)
    if not cases:
        raise FileNotFoundError(f"Khong tim thay case JSON nao trong {case_dir}")
    return cases


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "artifact"


def flatten_tables_to_text(tables: List[List[List[str]]]) -> str:
    chunks: List[str] = []
    for table in tables:
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            if any(cells):
                chunks.append(" | ".join(cells))
    return "\n".join(chunks)


def markdown_tables_from_text(text: str) -> List[List[List[str]]]:
    lines = [line.rstrip() for line in text.splitlines()]
    tables: List[List[List[str]]] = []
    current: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if any(cell for cell in cells):
                current.append(cells)
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def open_pdf_page(pdf_path: Path, page_number: int) -> pdfplumber.page.Page:
    pdf = pdfplumber.open(str(pdf_path))
    try:
        return pdf.pages[page_number - 1]
    except Exception:
        pdf.close()
        raise


def close_pdf_page(page: pdfplumber.page.Page) -> None:
    try:
        page.pdf.close()
    except Exception:
        pass


def extract_pdfplumber_raw_text_only(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    page = open_pdf_page(pdf_path, page_number)
    try:
        text = page.extract_text() or ""
    finally:
        close_pdf_page(page)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="pdfplumber_raw_text_only",
        pipeline="pdfplumber:extract_text",
        text=text,
        tables=[],
        latency_ms=latency_ms,
        ocr_used=False,
        vision_used=False,
        notes=[],
    )


def extract_pdfplumber_text_plus_tables(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    page = open_pdf_page(pdf_path, page_number)
    try:
        text = page.extract_text() or ""
        tables = page.extract_tables() or []
    finally:
        close_pdf_page(page)
    normalized_tables: List[List[List[str]]] = []
    for table in tables:
        rows: List[List[str]] = []
        for row in table or []:
            rows.append([str(cell or "").strip() for cell in row])
        if rows:
            normalized_tables.append(rows)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="pdfplumber_text_plus_tables",
        pipeline="pdfplumber:extract_text+extract_tables",
        text=text,
        tables=normalized_tables,
        latency_ms=latency_ms,
        ocr_used=False,
        vision_used=False,
        notes=[],
    )


def extract_page_ocr_tesseract_only(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    page = open_pdf_page(pdf_path, page_number)
    notes: List[str] = []
    try:
        pil_image = page.to_image(resolution=250).original
    finally:
        close_pdf_page(page)
    try:
        text = pytesseract.image_to_string(pil_image, lang="vie+eng")
    except Exception as exc:
        text = ""
        notes.append(f"tesseract_error:{exc}")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="page_ocr_tesseract_only",
        pipeline="page_image->pytesseract",
        text=text,
        tables=[],
        latency_ms=latency_ms,
        ocr_used=True,
        vision_used=False,
        notes=notes,
    )


def _extract_img2table_tables(pdf_path: Path, page_number: int) -> Tuple[List[List[List[str]]], List[str]]:
    notes: List[str] = []
    try:
        from img2table.document import PDF as Img2TablePDF
        from img2table.ocr import TesseractOCR
    except Exception as exc:
        return [], [f"img2table_import_error:{exc}"]

    try:
        ocr = TesseractOCR(lang="vie+eng")
        img_pdf = Img2TablePDF(str(pdf_path))
        extracted = img_pdf.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
        )
    except Exception as exc:
        return [], [f"img2table_error:{exc}"]

    page_tables = list(extracted.get(page_number - 1) or [])
    normalized_tables: List[List[List[str]]] = []
    for table_obj in page_tables:
        try:
            df = getattr(table_obj, "df", None)
            if df is not None:
                table_data = df.fillna("").values.tolist()
            else:
                content = getattr(table_obj, "content", None)
                table_data = [[str(c.value) for c in row] for row in content] if content else []
        except Exception:
            table_data = []
        table_rows = [[str(cell or "").strip() for cell in row] for row in table_data if row]
        if table_rows:
            normalized_tables.append(table_rows)
    return normalized_tables, notes


def extract_img2table_tesseract(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    tables, notes = _extract_img2table_tables(pdf_path, page_number)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="img2table_tesseract",
        pipeline="img2table+tesseract",
        text=flatten_tables_to_text(tables),
        tables=tables,
        latency_ms=latency_ms,
        ocr_used=True,
        vision_used=False,
        notes=notes,
    )


def extract_table_first_strict(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    tables, notes = _extract_img2table_tables(pdf_path, page_number)
    if not tables:
        page = open_pdf_page(pdf_path, page_number)
        try:
            fallback_tables = page.extract_tables() or []
        finally:
            close_pdf_page(page)
        for table in fallback_tables:
            rows = [[str(cell or "").strip() for cell in row] for row in (table or [])]
            if rows:
                tables.append(rows)
        if tables:
            notes.append("fallback:pdfplumber_tables")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="table_first_strict",
        pipeline="img2table->pdfplumber_tables(no prose)",
        text=flatten_tables_to_text(tables),
        tables=tables,
        latency_ms=latency_ms,
        ocr_used=True if tables else False,
        vision_used=False,
        notes=notes,
    )


def extract_hybrid_current(pdf_path: Path, page_number: int) -> ExtractionResult:
    started = time.perf_counter()
    notes: List[str] = []
    docs = process_pdf(str(pdf_path))
    page_docs = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        try:
            page = int(meta.get("page") or 0)
        except Exception:
            page = 0
        if page == page_number:
            page_docs.append(doc)
    text = "\n".join(str(doc.page_content or "") for doc in page_docs)
    tables = markdown_tables_from_text(text)
    parser_names = sorted(
        {
            str((getattr(doc, "metadata", {}) or {}).get("parser") or "")
            for doc in page_docs
            if isinstance((getattr(doc, "metadata", {}) or {}), dict)
        }
    )
    if parser_names:
        notes.append("parsers:" + ",".join(name for name in parser_names if name))
    vision_used = "AI EXTRACTED CONTENT FROM IMAGES" in text
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return ExtractionResult(
        method="hybrid_current",
        pipeline="process_pdf(current production path)",
        text=text,
        tables=tables,
        latency_ms=latency_ms,
        ocr_used=True,
        vision_used=vision_used,
        notes=notes,
    )


METHOD_DISPATCH = {
    "pdfplumber_raw_text_only": extract_pdfplumber_raw_text_only,
    "pdfplumber_text_plus_tables": extract_pdfplumber_text_plus_tables,
    "page_ocr_tesseract_only": extract_page_ocr_tesseract_only,
    "img2table_tesseract": extract_img2table_tesseract,
    "table_first_strict": extract_table_first_strict,
    "hybrid_current": extract_hybrid_current,
}


def make_schedule_parser() -> StructuredScheduleStore:
    temp_db = Path(tempfile.gettempdir()) / f"pdf_benchmark_schedule_{os.getpid()}.db"
    return StructuredScheduleStore(temp_db)


def parse_schedule_rows(extraction: ExtractionResult, page_label: str) -> List[Dict[str, Any]]:
    store = make_schedule_parser()
    lines: List[str] = []
    for table in extraction.tables:
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
    lines.extend([line.strip() for line in extraction.text.splitlines() if line.strip()])

    parsed: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        rows = store._parse_schedule_line(  # noqa: SLF001
            line,
            semester="BENCHMARK",
            source_file=page_label,
            source_page=1,
            source_line=idx,
        )
        parsed.extend(rows)

    dedup: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for row in parsed:
        simplified = {
            "subject_code": str(row.get("subject_code") or "").strip().upper(),
            "course_name": str(row.get("subject_name_vi") or "").strip(),
            "class_code": str(row.get("class_code") or "").strip().upper(),
            "teacher_name": str(row.get("teacher_name") or "").strip(),
            "day": str(row.get("day_of_week") or "").strip(),
            "slot": str(row.get("slot") or "").strip(),
            "room": str(row.get("room") or "").strip().upper(),
            "week_note": str(row.get("week_note") or "").strip(),
        }
        key = (
            compact_text(simplified["subject_code"]),
            compact_text(simplified["class_code"]),
            compact_text(simplified["teacher_name"]),
            compact_text(simplified["slot"]),
            compact_text(simplified["room"]),
        )
        dedup[key] = simplified
    return list(dedup.values())


def load_mock_transcript_rows(source_path: Path) -> List[Dict[str, Any]]:
    if source_path.suffix.lower() == ".json":
        payload = read_json(source_path)
        rows: List[Dict[str, Any]] = []
        for semester in payload.get("semesters") or []:
            semester_code = str(semester.get("semester") or "").strip()
            for subject in semester.get("subjects") or []:
                rows.append(
                    {
                        "semester": semester_code,
                        "course_code": str(subject.get("code") or "").strip().upper(),
                        "course_name": str(subject.get("name") or "").strip(),
                        "credits": int(subject.get("credits") or 0),
                        "grade_4": float(subject.get("grade_4") or 0),
                    }
                )
        return rows

    rows = []
    with source_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for item in reader:
            rows.append(
                {
                    "semester": str(item.get("semester") or "").strip(),
                    "course_code": str(item.get("code") or "").strip().upper(),
                    "course_name": str(item.get("name") or "").strip(),
                    "credits": int(float(str(item.get("credits") or 0).strip())),
                    "grade_4": float(str(item.get("grade_4") or "0").replace(",", ".")),
                }
            )
    return rows


def simple_pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(lines: Sequence[str], out_path: Path) -> Path:
    content_lines = ["BT", "/F1 9 Tf", "42 790 Td"]
    for idx, line in enumerate(lines[:56]):
        if idx:
            content_lines.append("0 -13 Td")
        content_lines.append(f"({simple_pdf_escape(line[:120])}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    objects.append(
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
    )

    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: List[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{idx} 0 obj\n".encode("ascii"))
        parts.append(obj)
        parts.append(b"\nendobj\n")
    xref_offset = sum(len(part) for part in parts)
    parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets:
        parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    parts.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    out_path.write_bytes(b"".join(parts))
    return out_path


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def write_image_table_pdf(
    title: str,
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Path:
    width, height = 1654, 2339
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32)
    header_font = load_font(20)
    cell_font = load_font(18)

    draw.text((70, 55), title, fill="black", font=title_font)
    subtitle = "Du lieu gia lap phuc vu benchmark trich xuat PDF"
    draw.text((70, 95), subtitle, fill="black", font=header_font)

    columns = [
        ("STT", 70),
        ("Ma HP", 170),
        ("Ten hoc phan", 600),
        ("TC", 80),
        ("Diem he 4", 120),
        ("Hoc ky", 180),
    ]
    start_x, start_y = 70, 150
    row_h = 32

    x_positions = [start_x]
    for _, col_w in columns:
        x_positions.append(x_positions[-1] + col_w)

    for idx, (label, col_w) in enumerate(columns):
        x0 = x_positions[idx]
        x1 = x_positions[idx + 1]
        draw.rectangle((x0, start_y, x1, start_y + row_h), outline="black", width=2)
        draw.text((x0 + 6, start_y + 5), label, fill="black", font=header_font)

    y = start_y + row_h
    for stt, row in enumerate(rows, start=1):
        values = [
            str(stt),
            str(row.get("course_code") or ""),
            str(row.get("course_name") or ""),
            str(row.get("credits") or ""),
            str(row.get("grade_4") or ""),
            str(row.get("semester") or ""),
        ]
        for idx, value in enumerate(values):
            x0 = x_positions[idx]
            x1 = x_positions[idx + 1]
            draw.rectangle((x0, y, x1, y + row_h), outline="black", width=1)
            trimmed = value[:52]
            draw.text((x0 + 6, y + 6), trimmed, fill="black", font=cell_font)
        y += row_h
        if y + row_h >= height - 70:
            break

    image.save(out_path, "PDF", resolution=150.0)
    return out_path


def materialize_case_source(case: Dict[str, Any], generated_dir: Path) -> Tuple[Path, int, Dict[str, Any]]:
    source = case.get("source") or {}
    kind = str(source.get("kind") or "").strip()
    if kind == "pdf_page":
        rel = Path(str(source.get("relative_path") or ""))
        page = int(source.get("page") or 1)
        return REPO_ROOT / rel, page, {"generated": False}

    if kind not in {"mock_transcript_json", "mock_transcript_csv"}:
        raise ValueError(f"Unsupported source kind: {kind}")

    rel = Path(str(source.get("relative_path") or ""))
    render_mode = str(source.get("render_mode") or "text_table_pdf").strip()
    rows = load_mock_transcript_rows(REPO_ROOT / rel)
    out_path = generated_dir / f"{safe_file_name(case['doc_id'])}.pdf"
    title = f"{case['title']} ({render_mode})"
    render_title = ascii_fold(title)
    render_rows = [
        {
            **row,
            "course_name": ascii_fold(row.get("course_name")),
        }
        for row in rows
    ]

    if render_mode == "text_table_pdf":
        lines = [
            f"MOCK TRANSCRIPT | {render_title}",
            "STT | Ma HP | Ten hoc phan | TC | Diem he 4 | Hoc ky",
        ]
        for idx, row in enumerate(render_rows, start=1):
            lines.append(
                f"{idx} | {row['course_code']} | {row['course_name']} | "
                f"{row['credits']} | {row['grade_4']} | {row['semester']}"
            )
        write_text_pdf(lines, out_path)
    elif render_mode == "image_table_pdf":
        write_image_table_pdf(render_title, render_rows, out_path)
    else:
        raise ValueError(f"Unsupported transcript render mode: {render_mode}")

    return out_path, 1, {"generated": True, "render_mode": render_mode, "row_count": len(rows)}


def parse_time_slot_rows(extraction: ExtractionResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table in extraction.tables:
        if not table:
            continue
        header_norm = " ".join(compact_text(cell) for cell in table[0] if cell)
        if "thoigianhoc" not in header_norm and "tiet" not in header_norm:
            continue
        for row in table[1:]:
            cells = [str(cell or "").strip() for cell in row]
            if len(cells) < 4:
                continue
            if compact_text(cells[1]) in {"nghi", ""}:
                continue
            if "tiết" in norm_text(cells[2]) or re.fullmatch(r"\d+", cells[1]):
                rows.append(
                    {
                        "session": cells[0],
                        "ca": cells[1],
                        "period": cells[2],
                        "time_range": cells[3],
                    }
                )
    if rows:
        return rows

    for line in extraction.text.splitlines():
        compact = norm_text(line)
        if "tiet" not in compact and "ca" not in compact:
            continue
        match = re.search(r"\b([1-9])\b.*?(Tiet\s*[0-9-]+).*?(\d{2}:\d{2}\s*[–-]\s*\d{2}:\d{2})", line, re.IGNORECASE)
        if match:
            rows.append(
                {
                    "session": "",
                    "ca": match.group(1),
                    "period": match.group(2),
                    "time_range": match.group(3),
                }
            )
    return rows


def parse_transcript_line(line: str) -> Optional[Dict[str, Any]]:
    original = " ".join(str(line or "").split())
    if not original:
        return None
    parts = [part.strip() for part in original.split("|") if part.strip()]
    if len(parts) >= 6:
        code_idx = next((idx for idx, part in enumerate(parts) if TRANSCRIPT_CODE_RE.fullmatch(part.upper())), -1)
        if code_idx >= 0 and code_idx + 4 < len(parts):
            code = parts[code_idx].upper()
            semester = ""
            for part in reversed(parts):
                sem = SEMESTER_RE.search(part)
                if sem:
                    semester = sem.group(1)
                    break
            grade = try_float(parts[-2])
            credits = try_float(parts[-3])
            name = " ".join(parts[code_idx + 1 : -3]).strip()
            if credits is not None and grade is not None:
                return {
                    "semester": semester,
                    "course_code": code,
                    "course_name": name,
                    "credits": int(round(credits)),
                    "grade_4": round(float(grade), 1),
                }

    code_match = TRANSCRIPT_CODE_RE.search(original.upper())
    if not code_match:
        return None
    code = code_match.group(1).upper()
    semester_match = SEMESTER_RE.search(original)
    semester = semester_match.group(1) if semester_match else ""
    tail = original[code_match.end() :].strip()
    number_tokens = list(FLOAT_RE.finditer(tail))
    if len(number_tokens) < 2:
        return None
    grade_token = number_tokens[-1]
    grade = try_float(grade_token.group(0))
    credits_token = number_tokens[-2]
    credits = try_float(credits_token.group(0))
    if grade is None or credits is None:
        return None
    name = tail[: credits_token.start()].strip(" |")
    if semester:
        name = name.replace(semester, "").strip(" |")
    name = re.sub(r"^\d+\s*", "", name).strip()
    if not name:
        name = code
    return {
        "semester": semester,
        "course_code": code,
        "course_name": name,
        "credits": int(round(credits)),
        "grade_4": round(float(grade), 1),
    }


def parse_transcript_rows(extraction: ExtractionResult) -> List[Dict[str, Any]]:
    lines: List[str] = []
    for table in extraction.tables:
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            if any(cells):
                lines.append(" | ".join(cells))
    lines.extend([line.strip() for line in extraction.text.splitlines() if line.strip()])

    parsed: List[Dict[str, Any]] = []
    for line in lines:
        row = parse_transcript_line(line)
        if row:
            parsed.append(row)

    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in parsed:
        key = (compact_text(row.get("course_code")), compact_text(row.get("semester")))
        dedup[key] = row
    return list(dedup.values())


def parse_english_mapping_rows(extraction: ExtractionResult) -> List[Dict[str, Any]]:
    lines: List[str] = []
    for table in extraction.tables:
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
    lines.extend([line.strip() for line in extraction.text.splitlines() if line.strip()])

    parsed: List[Dict[str, Any]] = []
    for line in lines:
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 6 and "bac" in norm_text(cells[0]):
                parsed.append(
                    {
                        "level": cells[0],
                        "ielts": cells[1],
                        "toefl_ibt": cells[2],
                        "aptis": cells[3],
                        "cambridge": cells[4],
                        "vstep": cells[5],
                    }
                )
                continue
        if "bac" not in norm_text(line) or "ielts" not in norm_text(line):
            continue
        level_match = re.search(r"(B[ậa]c\s*[345])", line, re.IGNORECASE)
        if not level_match:
            continue
        numbers = FLOAT_RE.findall(line)
        if len(numbers) < 2:
            continue
        parsed.append(
            {
                "level": level_match.group(1),
                "ielts": numbers[0],
                "toefl_ibt": numbers[1],
                "aptis": "B2" if "b2" in norm_text(line) else ("B1" if "b1" in norm_text(line) else ("C1" if "c1" in norm_text(line) else "")),
                "cambridge": line,
                "vstep": line,
            }
        )

    dedup: Dict[str, Dict[str, Any]] = {}
    for row in parsed:
        key = compact_text(row.get("level"))
        if key:
            dedup[key] = row
    return list(dedup.values())


def parse_case_rows(case: Dict[str, Any], extraction: ExtractionResult) -> List[Dict[str, Any]]:
    doc_type = str(case.get("doc_type") or "")
    if doc_type == "schedule":
        subtype = str(case.get("subtype") or "")
        if subtype == "time_slot":
            return parse_time_slot_rows(extraction)
        return parse_schedule_rows(extraction, page_label=case["doc_id"])
    if doc_type == "transcript":
        return parse_transcript_rows(extraction)
    if doc_type == "english_mapping":
        return parse_english_mapping_rows(extraction)
    raise ValueError(f"Unsupported doc_type: {doc_type}")


def english_mapping_field_match(field: str, expected: Any, observed: Any) -> bool:
    expected_text = str(expected or "").strip()
    observed_text = str(observed or "").strip()
    if not expected_text:
        return True
    if not observed_text:
        return False

    if field == "cambridge":
        expected_norm = normalize_for_match(expected_text)
        observed_norm = normalize_for_match(observed_text)
        expected_score_match = re.search(r"(\d+)", expected_text)
        expected_score = expected_score_match.group(1) if expected_score_match else ""
        label_only = re.sub(r"\(.*?\)", "", expected_text).strip()
        label_norm = normalize_for_match(label_only)
        has_label = bool(label_norm) and label_norm in observed_norm
        has_score = bool(expected_score) and expected_score in observed_norm
        return has_label and has_score

    return compact_text(expected_text) == compact_text(observed_text)


def field_match(expected: Any, observed: Any, *, case: Optional[Dict[str, Any]] = None, field: str = "") -> bool:
    if expected in (None, ""):
        return True
    if case and str(case.get("doc_type") or "") == "english_mapping":
        return english_mapping_field_match(field, expected, observed)
    return compact_text(expected) == compact_text(observed)


def build_row_key(row: Dict[str, Any], key_fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(compact_text(row.get(field)) for field in key_fields)


def classify_error(case: Dict[str, Any], expected: Dict[str, Any], observed: Optional[Dict[str, Any]], extraction: ExtractionResult) -> str:
    if observed is None:
        if extraction.method.startswith("page_ocr") and extraction.text.strip():
            return "ocr_noise"
        if extraction.tables:
            return "column_shift"
        return "missing_row"
    doc_type = str(case.get("doc_type") or "")
    if doc_type == "schedule":
        if not field_match(expected.get("subject_code"), observed.get("subject_code"), case=case, field="subject_code"):
            return "wrong_code"
        if not field_match(expected.get("slot"), observed.get("slot"), case=case, field="slot"):
            return "wrong_slot"
        if not field_match(expected.get("room"), observed.get("room"), case=case, field="room"):
            return "merged_cells"
        return "column_shift"
    if doc_type == "english_mapping":
        return "wrong_equivalence"
    return "column_shift"


def score_case(case: Dict[str, Any], extraction: ExtractionResult, observed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_rows = case.get("expected_rows") or []
    expected_fields = list(case.get("expected_fields") or [])
    key_fields = list(case.get("key_fields") or expected_fields)

    observed_by_key = {build_row_key(row, key_fields): row for row in observed_rows}
    row_results: List[Dict[str, Any]] = []
    matched_cells = 0
    expected_cells = 0
    extracted_cells = 0
    key_matches = 0
    key_total = len(expected_rows) * len(key_fields)
    exact_rows = 0

    for row in observed_rows:
        for field in expected_fields:
            if str(row.get(field) or "").strip():
                extracted_cells += 1

    error_counts: Dict[str, int] = {}
    for expected in expected_rows:
        key = build_row_key(expected, key_fields)
        observed = observed_by_key.get(key)
        field_details: Dict[str, bool] = {}
        if observed is not None:
            for field in key_fields:
                if field_match(expected.get(field), observed.get(field), case=case, field=field):
                    key_matches += 1
        else:
            for field in key_fields:
                if compact_text(expected.get(field)):
                    key_matches += 0

        correct_fields = 0
        for field in expected_fields:
            if str(expected.get(field) or "").strip():
                expected_cells += 1
            passed = field_match(expected.get(field), (observed or {}).get(field), case=case, field=field)
            field_details[field] = passed
            if passed and str(expected.get(field) or "").strip():
                correct_fields += 1
                matched_cells += 1

        exact = observed is not None and all(field_details.values())
        if exact:
            exact_rows += 1
        error_label = "" if exact else classify_error(case, expected, observed, extraction)
        if error_label:
            error_counts[error_label] = error_counts.get(error_label, 0) + 1

        row_results.append(
            {
                "expected": expected,
                "observed": observed,
                "field_matches": field_details,
                "exact_match": exact,
                "error_label": error_label,
            }
        )

    key_field_accuracy = (key_matches / key_total) if key_total else 0.0
    row_exact_accuracy = (exact_rows / len(expected_rows)) if expected_rows else 0.0
    cell_precision = (matched_cells / extracted_cells) if extracted_cells else 0.0
    cell_recall = (matched_cells / expected_cells) if expected_cells else 0.0
    cell_f1 = (2 * cell_precision * cell_recall / (cell_precision + cell_recall)) if (cell_precision + cell_recall) else 0.0
    score_pdf = compute_pdf_score(key_field_accuracy, row_exact_accuracy, cell_f1)
    pass_doc = key_field_accuracy >= 0.90 and row_exact_accuracy >= 0.80

    return {
        "doc_id": case["doc_id"],
        "doc_type": case["doc_type"],
        "subtype": case.get("subtype"),
        "key_field_accuracy": round4(key_field_accuracy),
        "row_exact_accuracy": round4(row_exact_accuracy),
        "cell_precision": round4(cell_precision),
        "cell_recall": round4(cell_recall),
        "cell_f1": round4(cell_f1),
        "score_pdf": round4(score_pdf),
        "pass_doc": pass_doc,
        "expected_row_count": len(expected_rows),
        "observed_row_count": len(observed_rows),
        "error_counts": error_counts,
        "row_results": row_results,
    }


def run_english_queries(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_level = {compact_text(row.get("level")): row for row in rows}
    questions = [
        {"id": "level4_ielts", "expected": "5.5", "actual": str((by_level.get("bac4") or {}).get("ielts") or "")},
        {"id": "level4_toefl", "expected": "72", "actual": str((by_level.get("bac4") or {}).get("toefl_ibt") or "")},
        {"id": "level3_aptis", "expected": "B1", "actual": str((by_level.get("bac3") or {}).get("aptis") or "")},
        {"id": "level5_vstep", "expected": "8.5", "actual": str((by_level.get("bac5") or {}).get("vstep") or "")},
        {"id": "level4_cambridge", "expected": "160", "actual": str((by_level.get("bac4") or {}).get("cambridge") or "")},
    ]
    passed = 0
    for question in questions:
        ok = compact_text(question["expected"]) in compact_text(question["actual"])
        question["pass"] = ok
        if ok:
            passed += 1
    accuracy = passed / len(questions) if questions else 0.0
    return {
        "total": len(questions),
        "passed": passed,
        "accuracy": round4(accuracy),
        "questions": questions,
    }


def micro_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(results)
    passed = sum(1 for item in results if (item.get("score") or {}).get("pass_doc"))
    latencies = [float(item.get("latency_ms") or 0.0) for item in results]
    key_weighted = sum(
        float((item.get("score") or {}).get("key_field_accuracy") or 0.0)
        * int((item.get("score") or {}).get("expected_row_count") or 0)
        for item in results
    )
    row_weighted = sum(
        float((item.get("score") or {}).get("row_exact_accuracy") or 0.0)
        * int((item.get("score") or {}).get("expected_row_count") or 0)
        for item in results
    )
    f1_weighted = sum(
        float((item.get("score") or {}).get("cell_f1") or 0.0)
        * int((item.get("score") or {}).get("expected_row_count") or 0)
        for item in results
    )
    total_rows = sum(int((item.get("score") or {}).get("expected_row_count") or 0) for item in results) or 1

    overall = {
        "cases": total_cases,
        "passed_docs": passed,
        "pass_rate": round4(passed / total_cases) if total_cases else 0.0,
        "key_field_accuracy": round4(key_weighted / total_rows),
        "row_exact_accuracy": round4(row_weighted / total_rows),
        "cell_f1": round4(f1_weighted / total_rows),
        "score_pdf": round4(
            compute_pdf_score(
                key_weighted / total_rows,
                row_weighted / total_rows,
                f1_weighted / total_rows,
            )
        ),
        "latency_p50_ms": p50(latencies),
        "latency_p95_ms": p95(latencies),
        "vision_fallback_rate": round4(sum(1 for item in results if item.get("vision_used")) / total_cases) if total_cases else 0.0,
        "ocr_usage_rate": round4(sum(1 for item in results if item.get("ocr_used")) / total_cases) if total_cases else 0.0,
    }
    return overall


def summarize_by_type(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item.get("doc_type") or ""), []).append(item)
    rows: List[Dict[str, Any]] = []
    for doc_type, items in sorted(grouped.items()):
        summary = micro_summary(items)
        summary["doc_type"] = doc_type
        rows.append(summary)
    return rows


def method_table_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    header = "| Method | Cases | Pass | Pass rate | Key acc | Row acc | Cell F1 | Score PDF | p50 ms | p95 ms | Vision rate |\n"
    header += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    body = []
    for row in rows:
        body.append(
            "| `{method}` | {cases} | {passed_docs} | {pass_rate:.2%} | {key_field_accuracy:.4f} | "
            "{row_exact_accuracy:.4f} | {cell_f1:.4f} | {score_pdf:.4f} | {latency_p50_ms} | {latency_p95_ms} | {vision_fallback_rate:.2%} |".format(
                **row
            )
        )
    return header + "\n".join(body)


def type_table_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    header = "| Doc type | Cases | Pass rate | Key acc | Row acc | Cell F1 | Score PDF |\n"
    header += "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    body = []
    for row in rows:
        body.append(
            "| `{doc_type}` | {cases} | {pass_rate:.2%} | {key_field_accuracy:.4f} | {row_exact_accuracy:.4f} | {cell_f1:.4f} | {score_pdf:.4f} |".format(
                **row
            )
        )
    return header + "\n".join(body)


def render_markdown_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# PDF Extraction Benchmark Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Dataset dir: `{payload['dataset_dir']}`",
        f"- Cases: `{payload['case_count']}`",
        f"- Methods: `{', '.join(payload['methods'])}`",
        (
            "- Score weights: "
            f"`key={payload.get('score_weights', {}).get('key_field_accuracy', PDF_SCORE_WEIGHTS['key_field_accuracy']):.2f}`, "
            f"`row={payload.get('score_weights', {}).get('row_exact_accuracy', PDF_SCORE_WEIGHTS['row_exact_accuracy']):.2f}`, "
            f"`cell_f1={payload.get('score_weights', {}).get('cell_f1', PDF_SCORE_WEIGHTS['cell_f1']):.2f}`"
        ),
        "",
        "## Overall Summary",
        "",
        method_table_markdown(payload["summary_by_method"]),
        "",
        "## Summary by Document Type",
        "",
    ]

    for method_block in payload["summary_by_type"]:
        lines.extend(
            [
                f"### `{method_block['method']}`",
                "",
                type_table_markdown(method_block["rows"]),
                "",
            ]
        )

    lines.extend(
        [
            "## English Mapping Downstream Queries",
            "",
            "| Method | Passed | Total | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["english_query_summary"]:
        lines.append(
            f"| `{row['method']}` | {row['passed']} | {row['total']} | {row['accuracy']:.2%} |"
        )

    lines.extend(["", "## Notable Failures", ""])
    notable = sorted(
        (item for item in payload["case_results"] if not item["score"]["pass_doc"]),
        key=lambda item: (item["score"]["score_pdf"], item["latency_ms"]),
    )
    for item in notable[:18]:
        if item["score"]["pass_doc"]:
            continue
        error_labels = ", ".join(f"{k}:{v}" for k, v in sorted(item["score"]["error_counts"].items()))
        lines.append(
            f"- `{item['method']}` / `{item['doc_id']}`: score={item['score']['score_pdf']:.4f}, "
            f"key={item['score']['key_field_accuracy']:.4f}, row={item['score']['row_exact_accuracy']:.4f}, "
            f"errors={error_labels or 'n/a'}"
        )
        preview = str(item["extraction"]["text"] or "").strip().replace("\n", " | ")
        if preview:
            lines.append(f"  - preview: {preview[:260]}")
    return "\n".join(lines).strip() + "\n"


def run_benchmark(dataset_dir: Path, methods: Sequence[str], reports_dir: Path) -> Dict[str, Any]:
    cases = load_cases(dataset_dir)
    stamp = utc_stamp()
    generated_inputs_dir = ensure_dir(reports_dir / f"pdf_benchmark_inputs_{stamp}")
    case_results: List[Dict[str, Any]] = []

    for case in cases:
        pdf_path, page_number, materialized_meta = materialize_case_source(case, generated_inputs_dir)
        for method in methods:
            extractor = METHOD_DISPATCH[method]
            extraction = extractor(pdf_path, page_number)
            observed_rows = parse_case_rows(case, extraction)
            score = score_case(case, extraction, observed_rows)
            language_queries = (
                run_english_queries(observed_rows)
                if str(case.get("doc_type") or "") == "english_mapping"
                else None
            )
            case_results.append(
                {
                    "doc_id": case["doc_id"],
                    "title": case["title"],
                    "doc_type": case["doc_type"],
                    "subtype": case.get("subtype"),
                    "method": method,
                    "source": {
                        "pdf_path": str(pdf_path),
                        "page": page_number,
                        **materialized_meta,
                    },
                    "expected_fields": case.get("expected_fields") or [],
                    "key_fields": case.get("key_fields") or [],
                    "expected_rows": case.get("expected_rows") or [],
                    "observed_rows": observed_rows,
                    "score": score,
                    "extraction": asdict(extraction),
                    "language_queries": language_queries,
                    "latency_ms": extraction.latency_ms,
                    "ocr_used": extraction.ocr_used,
                    "vision_used": extraction.vision_used,
                }
            )

    summary_by_method: List[Dict[str, Any]] = []
    summary_by_type: List[Dict[str, Any]] = []
    english_query_summary: List[Dict[str, Any]] = []
    for method in methods:
        method_results = [item for item in case_results if item["method"] == method]
        overall = micro_summary(method_results)
        overall["method"] = method
        summary_by_method.append(overall)
        summary_by_type.append({"method": method, "rows": summarize_by_type(method_results)})

        english_items = [
            item["language_queries"]
            for item in method_results
            if item.get("language_queries") is not None
        ]
        if english_items:
            total = sum(int(item.get("total") or 0) for item in english_items)
            passed = sum(int(item.get("passed") or 0) for item in english_items)
            english_query_summary.append(
                {
                    "method": method,
                    "total": total,
                    "passed": passed,
                    "accuracy": round4(passed / total) if total else 0.0,
                }
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "case_count": len(cases),
        "methods": list(methods),
        "score_weights": dict(PDF_SCORE_WEIGHTS),
        "summary_by_method": summary_by_method,
        "summary_by_type": summary_by_type,
        "english_query_summary": english_query_summary,
        "case_results": case_results,
        "generated_inputs_dir": str(generated_inputs_dir),
    }
    return payload


def write_reports(payload: Dict[str, Any], reports_dir: Path) -> Tuple[Path, Path]:
    stamp = utc_stamp()
    json_path = reports_dir / f"pdf_extraction_benchmark_{stamp}.json"
    md_path = reports_dir / f"pdf_extraction_benchmark_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")

    latest_json = reports_dir / "pdf_extraction_benchmark_latest.json"
    latest_md = reports_dir / "pdf_extraction_benchmark_latest.md"
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PDF extraction pipelines for academic advisor documents.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    invalid = [method for method in methods if method not in METHOD_DISPATCH]
    if invalid:
        raise SystemExit(f"Unsupported methods: {', '.join(invalid)}")
    payload = run_benchmark(args.dataset, methods, args.output_dir)
    json_path, md_path = write_reports(payload, args.output_dir)
    print(f"Saved JSON report to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
