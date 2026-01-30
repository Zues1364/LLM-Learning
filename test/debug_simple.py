
"""
Minimal debug script - output JSON only for verification.
"""
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mcp_server.server import analyze_transcript, analyze_curriculum, compute_missing_subjects, _build_completed_subjects

def main():
    pdf1 = r"D:\LLM\LLM Learning\data\pdfs\ĐIỂM_1_ef61b158.pdf"
    pdf2 = r"D:\LLM\LLM Learning\data\pdfs\ĐIỂM_2_68657c33.pdf"
    
    transcript_json = analyze_transcript([pdf1, pdf2])
    transcript_data = json.loads(transcript_json)
    
    completed_map = _build_completed_subjects(transcript_data.get("semesters") or [])
    
    curriculum = analyze_curriculum("Khoa học máy tính")
    
    missing_info = compute_missing_subjects(transcript_data, curriculum)
    missing_list = missing_info.get("missing") or []
    
    # Extract just codes for comparison
    completed_codes = list(completed_map.keys())
    missing_codes = [m.get("code") for m in missing_list]
    
    # Check specific codes
    check_codes = ["MAT1041", "EPN1095", "INT1008", "INT1009"]
    bug_results = {}
    for code in check_codes:
        norm_code = code.upper().replace(" ", "")
        in_completed = any(k.upper().replace(" ", "") == norm_code for k in completed_codes)
        in_missing = code in missing_codes
        bug_results[code] = {
            "in_completed": in_completed,
            "in_missing": in_missing,
            "bug": in_completed and in_missing  # BUG if both true
        }
    
    result = {
        "completed_count": len(completed_codes),
        "missing_count": len(missing_codes),
        "missing_codes_sample": missing_codes[:15],
        "bug_check": bug_results,
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
