import pdfplumber

pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        print(f"--- Page {i+1} Tables ---")
        tables = page.extract_tables()
        for t in tables:
            for row in t:
                print(row)
        print("\n--- Page {i+1} Text ---")
        print(page.extract_text())
