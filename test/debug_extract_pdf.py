import pdfplumber

pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages[:3]): # Check first 3 pages
        print(f"--- Page {i+1} ---")
        text = page.extract_text()
        print(text)
        print("\n")
