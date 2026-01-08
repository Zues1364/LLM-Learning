
import sys
from unittest.mock import MagicMock

# Mock modules to avoid full server setup
sys.modules["fastapi"] = MagicMock()
sys.modules["agno"] = MagicMock()
sys.modules["agno.agent"] = MagicMock()
sys.modules["agno.models.google"] = MagicMock()
sys.modules["mcp_client"] = MagicMock()
sys.modules["mcp_client.client"] = MagicMock()
sys.modules["persistent_memory"] = MagicMock()
# sys.modules["utils"] = MagicMock() # We need utils for PDF processing if possible, or we can just impl simple pdfplumber here
sys.modules["agents"] = MagicMock()
sys.modules["env_loader"] = MagicMock()

import pdfplumber
import re

def extract_grade_table(pdf_path):
    # Set stdout to utf-8 for windows console
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"Reading PDF...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and ("thang điểm" in text.lower() or "quy đổi" in text.lower() or "điểm chữ" in text.lower()):
                    print(f"--- Page {i+1} ---")
                    print(text)
                    print("----------------")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    extract_grade_table("d:/LLM/LLM Learning/data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 22-23.pdf")
