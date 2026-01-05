
import sys
import os
import logging
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_transcript_parsing():
    file_path = r"D:\LLM\LLM Learning\data\pdfs\ĐIỂM_1.pdf"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    safe_path = file_path.encode('ascii', 'replace').decode('ascii')
    print(f"--- Processing {safe_path} ---")
    chunks = process_pdf(file_path)
    print(f"Generated {len(chunks)} chunks.")
    
    full_text = "\n".join([c.page_content for c in chunks])
    print("--- Extracted Text Preview (First 1000 chars) ---")
    print(full_text[:1000].encode('ascii', 'replace').decode('ascii'))
    
    print("\n--- Saving full text to debug_transcript.txt ---")
    with open("debug_transcript.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

if __name__ == "__main__":
    debug_transcript_parsing()
