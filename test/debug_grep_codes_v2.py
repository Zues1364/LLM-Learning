import pdfplumber

# The other file
pdf_path = r"d:\LLM\LLM Learning\data\resources\pdfs\Signed.[TKB] DỰ KIẾN TKB HKII NĂM HỌC 2025-2026 (SV).pdf"

print(f"Scanning {pdf_path} for PEC/HIS/PHI...")
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if any(x in text for x in ["PEC", "HIS", "PHI"]):
            print(f"--- Page {i+1} Match ---")
            for line in text.splitlines():
                if any(x in line for x in ["PEC", "HIS", "PHI"]):
                    print(line)
                    # Print only first few matches to confirm
                    if i > 2: break
