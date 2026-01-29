import pdfplumber

pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"

print("Scanning for PEC/HIS/PHI...")
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if not text: continue
        
        # Check simple substring
        if "PEC" in text or "HIS" in text or "PHI" in text:
            print(f"--- Page {i+1} Match ---")
            for line in text.splitlines():
                if any(x in line for x in ["PEC", "HIS", "PHI"]):
                    print(line)
