
import sys
import os
import argparse
from typing import List
from langchain_core.documents import Document

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Force UTF-8 encoding for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Target the specific Schedule PDF
    pdf_path = r"D:\LLM\LLM Learning\data\resources\pdfs\Signed.[TKB] DỰ KIẾN TKB HKII NĂM HỌC 2025-2026 (SV).pdf"
    
    # Keyword: "Mạng máy tính" (Common subject usually in schedule) or "INT2213"
    keywords = ["INT2213", "Mạng máy tính"]
    
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        try:
            from utils import process_pdf
            pdf_chunks = process_pdf(pdf_path)
            
            print(f"\n--- Checking first 3 chunks containing keywords ---")
            found = 0
            for i, doc in enumerate(pdf_chunks):
                content = doc.page_content
                if any(k in content for k in keywords):
                    print(f"\n[CHUNK {i}]")
                    # Print full content of chunk to see structure
                    print(content.strip())
                    print("-" * 40)
                    found += 1
                    if found >= 3: break
            
            if found == 0:
                print("❌ No chunks found with keywords.")
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"PDF file not found: {pdf_path}")

if __name__ == "__main__":
    main()
