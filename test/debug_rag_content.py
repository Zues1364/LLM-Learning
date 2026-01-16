import sys
import os
import argparse
from typing import List
from langchain_core.documents import Document

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from crawler import crawl_url
    from utils import process_pdf
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Force UTF-8 encoding for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def check_keywords_in_chunks(chunks: List[Document], keywords: List[str], source_name: str):
    print(f"\n🔍 Checking {len(chunks)} chunks from: {source_name}")
    found_count = 0
    for i, doc in enumerate(chunks):
        content = doc.page_content.lower()
        for kw in keywords:
            if kw.lower() in content:
                print(f"  ✅ Found '{kw}' in Chunk {i}")
                print(f"     Context: ...{content.replace(kw.lower(), '>>>'+kw.upper()+'<<<')[max(0, content.find(kw.lower())-50):min(len(content), content.find(kw.lower())+100)]}...\n")
                found_count += 1
    
    if found_count == 0:
        print(f"  ❌ Keywords {keywords} NOT FOUND in any chunks from {source_name}")
    else:
        print(f"  ✨ Found total {found_count} matches in {source_name}")

def main():
    pdf_path = r"D:\LLM\LLM Learning\data\resources\pdfs\SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
    html_path = r"D:\LLM\LLM Learning\data\resources\html\Chương trình đào tạo ngành Khoa học máy tính (CLC TT23) - Trường Đại học Công nghệ, ĐHQGHN - Univeristy of Engineering and Technology.html"

    keywords = ["Lý thuyết thông tin", "INT2044E"]

    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        try:
            pdf_chunks = process_pdf(pdf_path)
            check_keywords_in_chunks(pdf_chunks, keywords, "PDF")
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"PDF file not found: {pdf_path}")

    if os.path.exists(html_path):
        print(f"Processing HTML: {html_path}")
        try:
            # crawl_url handles local files if prefixed with file:// or just path if code handles it
            # The code in crawler.py checks os.path.isfile(url)
            html_chunks = crawl_url(html_path) 
            check_keywords_in_chunks(html_chunks, keywords, "HTML")
        except Exception as e:
            print(f"Error processing HTML: {e}")
    else:
        print(f"HTML file not found: {html_path}")

if __name__ == "__main__":
    main()
