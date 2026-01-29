import pdfplumber

pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            for line in text.split('\n'):
                if "Tiết" in line or "Ca" in line or ":" in line:
                    print(line)
