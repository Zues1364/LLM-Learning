import logging
import os
from typing import List, Tuple

import pdfplumber
from img2table.document import PDF as Img2TablePDF
from img2table.ocr import TesseractOCR
from langchain_core.documents import Document
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TABLE_CHUNK_SIZE = 1000
TEXT_CHUNK_SIZE = 1024
TEXT_CHUNK_OVERLAP = 256


# Helper for Vision
def describe_image_with_gemini(image) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("[VISION] No API Key found, skipping Vision.")
        return ""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")
        
        prompt = (
            "This image is a page from a document. "
            "Please transcribe the content into Markdown. "
            "If there are tables, represent them as Markdown tables. "
            "If there is text, preserve headers and structure. "
            "Focus on accuracy for numbers."
        )
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        print(f"!!! GEMINI ERROR: {e}")
        return ""


def cell_to_text(cell) -> str:
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell
    for attr in ("value", "text", "content"):
        if hasattr(cell, attr):
            val = getattr(cell, attr)
            if val is not None:
                return str(val)
    return str(cell)


def clean_cell(cell) -> str:
    return cell_to_text(cell).replace("\n", " ").strip()


def normalize_table(table: List[List[str]]) -> List[List[str]]:
    if not table:
        return []

    cleaned = [[clean_cell(cell) for cell in row] for row in table]
    max_cols = max(len(row) for row in cleaned)
    normalized = [row + [""] * (max_cols - len(row)) for row in cleaned]
    return normalized


def extract_table_data(table_obj) -> List[List[str]]:
    if table_obj is None:
        return []

    df = getattr(table_obj, "df", None)
    if df is not None:
        try:
            df = df.fillna("")
            return df.values.tolist()
        except Exception:
            pass

    content = getattr(table_obj, "content", None)
    if content:
        try:
            return [[clean_cell(cell) for cell in row] for row in content]
        except Exception:
            pass

    if hasattr(table_obj, "extract"):
        try:
            extracted = table_obj.extract()
            if extracted:
                return extracted
        except Exception:
            pass

    if isinstance(table_obj, list):
        return table_obj

    return []


def extract_table_bbox(table_obj):
    bbox = getattr(table_obj, "bbox", None)
    if bbox is None:
        return None

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (bbox[0], bbox[1], bbox[2], bbox[3])

    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
            return (bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
        if all(k in bbox for k in ("x0", "top", "x1", "bottom")):
            return (bbox["x0"], bbox["top"], bbox["x1"], bbox["bottom"])

    attr_sets = (
        ("x1", "y1", "x2", "y2"),
        ("x0", "y0", "x1", "y1"),
        ("x0", "top", "x1", "bottom"),
    )
    for attrs in attr_sets:
        if all(hasattr(bbox, attr) for attr in attrs):
            return tuple(getattr(bbox, attr) for attr in attrs)

    return None


def normalize_bbox_to_page(bbox: Tuple[float, float, float, float], page, img_size: Tuple[int, int] | None = None) -> Tuple[float, float, float, float]:
    x0, top, x1, bottom = bbox
    if not img_size or not isinstance(img_size, (tuple, list)) or len(img_size) < 2:
        return (x0, top, x1, bottom)

    img_w, img_h = img_size[0], img_size[1]
    if not img_w or not img_h:
        return (x0, top, x1, bottom)

    if x1 > page.width + 1 or bottom > page.height + 1:
        scale_x = page.width / img_w
        scale_y = page.height / img_h
        return (x0 * scale_x, top * scale_y, x1 * scale_x, bottom * scale_y)
    return (x0, top, x1, bottom)


def get_image_sizes(img_pdf) -> List[Tuple[int, int] | None]:
    images = getattr(img_pdf, "images", None)
    if not images:
        return []
    sizes: List[Tuple[int, int] | None] = []
    for img in images:
        size = None
        if hasattr(img, "size"):
            try:
                candidate = img.size
                if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
                    size = (candidate[0], candidate[1])
            except Exception:
                size = None

        if size is None and hasattr(img, "shape"):
            try:
                shape = img.shape
                if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                    size = (shape[1], shape[0])
            except Exception:
                size = None

        sizes.append(size)
    return sizes


def looks_fragmented_table(table: List[List[str]]) -> bool:
    if not table:
        return True
    cells = [cell for row in table for cell in row if cell]
    if not cells:
        return True
    single_char = sum(1 for cell in cells if len(cell.strip()) == 1)
    ratio = single_char / len(cells)
    max_cols = max(len(row) for row in table)
    return max_cols >= 10 and ratio >= 0.6


def validate_table(table: List[List[str]], min_chars: int = 10) -> bool:
    if not table:
        return False

    normalized = normalize_table(table)
    if not normalized:
        return False

    # 1. Kiểm tra tổng thể
    cell_texts = [cell.strip() for row in normalized for cell in row if cell and cell.strip()]
    if not cell_texts:
        return False
    
    total_chars = sum(len(cell) for cell in cell_texts)
    if total_chars < min_chars:
        return False

    # 2. Logic mới: Kiểm tra độ "vụn" của bảng (Fragmented check)
    # Tính trung bình số ký tự trên mỗi ô CÓ DỮ LIỆU
    avg_len = total_chars / len(cell_texts)
    
    # Lấy số lượng cột
    num_cols = len(normalized[0])
    num_rows = len(normalized)

    # RULE QUAN TRỌNG:
    # Nếu bảng có nhiều cột (>4) nhưng trung bình mỗi ô rất ngắn (<6 ký tự)
    # -> Đây chắc chắn là văn bản bị cắt dọc.
    # Ví dụ Chunk 21: 12 cột, mỗi ô là "Đầu", "mỗi", "học" (3-4 ký tự) -> avg_len ~ 3.5 -> REJECT
    if num_cols > 4 and avg_len < 6:
        logger.info(f"Rejected Fragmented Table: {num_cols} cols, avg_len {avg_len:.2f}")
        return False
        
    # Nếu bảng cực nhiều cột (>8) (trường hợp dòng kẻ text thẳng hàng)
    if num_cols > 8:
        return False

    # 3. Check Header rác (Dòng đầu tiên toàn chữ cái đơn lẻ hoặc ký tự lạ)
    header_cells = [c.strip() for c in normalized[0] if c.strip()]
    if len(header_cells) > 3:
        # Đếm số ô header chỉ có 1-2 ký tự (mà không phải số thứ tự hay mã)
        short_headers = sum(1 for c in header_cells if len(c) < 3 and not c.isdigit())
        if short_headers / len(header_cells) > 0.5:
            return False

    return True

def table_to_row_lines(table: List[List[str]]) -> Tuple[str, str, List[str]]:
    normalized = normalize_table(table)
    if not normalized:
        return "", "", []

    header = normalized[0]
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"

    data_lines = []
    for row in normalized[1:]:
        row_line = "| " + " | ".join(row) + " |"
        data_lines.append(row_line)

    return header_line, separator_line, data_lines


def chunk_table_rows(table: List[List[str]], chunk_size: int) -> List[str]:
    header_line, separator_line, data_lines = table_to_row_lines(table)
    if not header_line:
        return []

    header_block = header_line + "\n" + separator_line + "\n"
    chunks = []
    current = header_block

    for row_line in data_lines:
        addition = row_line + "\n"
        if len(current) + len(addition) > chunk_size and len(current) > len(header_block):
            chunks.append(current.rstrip())
            current = header_block + addition
        else:
            current += addition

    if current.strip():
        chunks.append(current.rstrip())

    return chunks


def find_tables(page) -> Tuple[List[object], bool]:
    table_settings_lines = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    tables = page.find_tables(table_settings_lines)
    if tables:
        return tables, False

    table_settings_text = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "intersection_x_tolerance": 15,
        "snap_tolerance": 4,
        "text_y_tolerance": 3,
    }
    return page.find_tables(table_settings_text), True


def is_inside_table(word_bbox: Tuple[float, float, float, float], table_bboxes: List[Tuple[float, float, float, float]], margin: float = 1.0) -> bool:
    x0, top, x1, bottom = word_bbox
    cx = (x0 + x1) / 2.0
    cy = (top + bottom) / 2.0

    for bx0, btop, bx1, bbottom in table_bboxes:
        if (bx0 - margin) <= cx <= (bx1 + margin) and (btop - margin) <= cy <= (bbottom + margin):
            return True
    return False


def words_to_text(words: List[dict], line_tol: float = 3.0) -> str:
    if not words:
        return ""

    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current_line = []
    current_top = words_sorted[0]["top"]

    for word in words_sorted:
        if abs(word["top"] - current_top) <= line_tol:
            current_line.append(word)
        else:
            line = " ".join(w["text"] for w in sorted(current_line, key=lambda w: w["x0"]))
            lines.append(line)
            current_line = [word]
            current_top = word["top"]

    if current_line:
        line = " ".join(w["text"] for w in sorted(current_line, key=lambda w: w["x0"]))
        lines.append(line)

    return "\n".join(lines).strip()


def extract_text_outside_tables(page, table_bboxes: List[Tuple[float, float, float, float]]) -> str:
    words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True) or []
    filtered = []

    for word in words:
        bbox = (word["x0"], word["top"], word["x1"], word["bottom"])
        if not is_inside_table(bbox, table_bboxes):
            filtered.append(word)

    return words_to_text(filtered)


def process_pdf_with_pdfplumber(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    source = os.path.basename(file_path)
    final_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        separators=["\n### ", "\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    try:
        ocr = TesseractOCR(lang="vie+eng")
        img_pdf = Img2TablePDF(file_path)
        tables_by_page = img_pdf.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
        )
        image_sizes = get_image_sizes(img_pdf)
    except Exception as e:
        logger.error(f"img2table extraction failed: {e}")
        tables_by_page = {}
        image_sizes = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            if isinstance(tables_by_page, dict):
                page_tables = tables_by_page.get(i) or tables_by_page.get(page_num) or []
            elif isinstance(tables_by_page, list):
                page_tables = tables_by_page[i] if i < len(tables_by_page) else []
            else:
                page_tables = []
            table_bboxes: List[Tuple[float, float, float, float]] = []

            for t_idx, table_obj in enumerate(page_tables, start=1):
                table_data = extract_table_data(table_obj)
                if not table_data:
                    continue
                if not validate_table(table_data):
                    logger.info(f"Page {page_num} table {t_idx} rejected by validate_table.")
                    continue

                bbox = extract_table_bbox(table_obj)
                if bbox:
                    img_size = image_sizes[i] if i < len(image_sizes) else None
                    table_bboxes.append(normalize_bbox_to_page(bbox, page, img_size))
                else:
                    logger.info(f"Page {page_num} table {t_idx} missing bbox; text may include table content.")
                table_chunks = chunk_table_rows(table_data, TABLE_CHUNK_SIZE)
                for c_idx, chunk_text in enumerate(table_chunks, start=1):
                    doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "page": page_num,
                            "source": source,
                            "type": "table",
                            "table_index": t_idx,
                            "table_chunk": c_idx,
                        },
                    )
                    final_chunks.append(doc)

            text_content = extract_text_outside_tables(page, table_bboxes)

            # --- VISION LOGIC PROTOTYPE ---
            # Check for large images only if text content is low OR just as augmentation
            images = page.images
            use_vision = False
            if images:
                for img in images:
                     # Check area. pdfplumber image dict has 'width', 'height'
                     w = img.get('width', 0)
                     h = img.get('height', 0)
                     if w * h > 20000:
                         use_vision = True
                         break
            
            vision_text = ""
            if use_vision:
                 logger.info(f"Page {page_num}: Significant image detected. Running Gemini Vision...")
                 try:
                     # resolution=300 is good for OCR
                     # page.to_image() returns PageImage, .original gives PIL Image
                     page_img_obj = page.to_image(resolution=300)
                     pil_image = page_img_obj.original
                     vision_desc = describe_image_with_gemini(pil_image)
                     if vision_desc:
                         vision_text = f"\n\n### AI EXTRACTED CONTENT FROM IMAGES:\n{vision_desc}"
                 except Exception as e:
                     logger.error(f"Vision failed on page {page_num}: {e}")

            if vision_text:
                text_content += vision_text
            # ------------------------------

            if text_content:
                text_doc = Document(
                    page_content=text_content,
                    metadata={
                        "page": page_num,
                        "source": source,
                        "type": "text",
                    },
                )
                text_chunks = text_splitter.split_documents([text_doc])
                for c_idx, chunk in enumerate(text_chunks, start=1):
                    chunk.metadata["text_chunk"] = c_idx
                    final_chunks.append(chunk)

    for idx, doc in enumerate(final_chunks, start=1):
        doc.metadata["chunk_index"] = idx

    logger.info(f"Generated {len(final_chunks)} chunks.")
    return final_chunks


# --- RUN TEST ---
if __name__ == "__main__":
    # Update this path to your PDF
    pdf_file = r"D:\LLM\LLM Learning\data\pdfs\SỔ TAY HỌC VỤ KỲ I NĂM 22-23.pdf"

    chunks = process_pdf_with_pdfplumber(pdf_file)

    target_page = 6
    page_chunks = [c for c in chunks if c.metadata.get("page") == target_page]
    if page_chunks:
        print(f"\n=== Preview Page {target_page} (Chunk 1) ===\n")
        print(page_chunks[0].page_content[:2000].encode('ascii', 'ignore').decode('ascii'))

    with open("debug_pdfplumber_chunks.txt", "w", encoding="utf-8") as f:
        for idx, doc in enumerate(chunks, start=1):
            page = doc.metadata.get("page")
            dtype = doc.metadata.get("type")
            f.write(f"=== Chunk {idx} | Page {page} | type {dtype} | chars {len(doc.page_content)} ===\n")
            f.write(doc.page_content)
            f.write("\n\n" + "=" * 50 + "\n\n")

    print("\nSaved chunk log to debug_pdfplumber_chunks.txt")
