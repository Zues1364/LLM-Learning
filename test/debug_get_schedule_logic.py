import sys
import os
import json
import logging
import re
from unittest.mock import MagicMock

# Define logger mock
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Stub dependencies
def process_pdf(pdf_path):
    print(f"Mock processing {pdf_path}")
    # Simulating the PDF content with Time Table info
    from langchain_core.documents import Document
    content = """
    ĐẠI HỌC QUỐC GIA HÀ NỘI
    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
    
    BẢNG GIỜ HỌC
    1 Tiết 1-3 07:00 – 09:40
    2 Tiết 4-6 10:00 – 12:40
    3 Tiết 7-9 13:00 – 15:40
    4 Tiết 10-12 16:20 – 19:00
    
    DANH SÁCH LỚP MÔN HỌC
    INT3306 1 Phát triển ứng dụng Web 3 Thứ 2 10-12 301-G2
    PEC1008 1 Kinh tế chính trị Mác - Lênin 2 Thứ 4 1-3 205-G+
    """
    return [Document(page_content=content)]

# --- REPLICATE THE LOGIC FROM server.py (Modified) ---
def get_schedule_test(subject_codes):
    full_text = " ".join([d.page_content for d in process_pdf("dummy.pdf")])
    
    # --- NEW: Extract Time Table Context ---
    time_lines = []
    for line in full_text.splitlines():
        # Matches strings like "Tiết 1-3 07:00" or "Ca 1 ... 07:00"
        if re.search(r"(Tiết|Ca)\s+\d+.*(\d{1,2}:\d{2})", line, re.IGNORECASE):
            time_lines.append(line.strip())
    
    time_table_context = ""
    if time_lines:
        unique_lines = list(set(time_lines))
        time_table_context = "\n[CONTEXT TIME TABLE DETECTED IN PDF]:\n" + "\n".join(unique_lines[:15])
    
    print("Detected Time Context:", time_table_context)
    
    results = []
    for code in subject_codes:
        norm_code = code.upper().strip()
        matches = []
        for line in full_text.splitlines():
            if norm_code in line.upper():
                matches.append(line.strip())
        
        if matches:
            item = {
                "subject_code": norm_code,
                "schedule_lines": matches
            }
            if time_table_context:
                item["time_definitions"] = time_table_context
            results.append(item)
    
    return results

# Test it
res = get_schedule_test(["INT3306"])
print("\nFinal Result:")
print(json.dumps(res, indent=2, ensure_ascii=False))
