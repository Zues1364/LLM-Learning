import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import process_pdf
import logging

logging.basicConfig(level=logging.INFO)

pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"

print(f"Processing {pdf_path}...")
try:
    chunks = process_pdf(pdf_path)
    print(f"Generated {len(chunks)} chunks.")
    found_time_info = False
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        # Look for keywords indicating the time table
        if "16:20" in content or "07:00" in content or "Tiết 1-3" in content:
            print(f"--- Chunk {i} ---")
            print(content)
            found_time_info = True

    if not found_time_info:
        print("WARNING: Could not find time table info in chunks.")

except Exception as e:
    print(e)
