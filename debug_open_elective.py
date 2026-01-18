
import sys
import os
import json
from bs4 import BeautifulSoup

# Mock utils for verification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import parse_curriculum_from_html_content, compute_curriculum_missing_credits

# 1. Mock HTML with the problematic section AND HEADER
mock_html = """
<html>
<body>
<table>
<tr><td>STT</td><td>Mã học phần</td><td>Tên học phần</td><td>Số tín chỉ</td></tr>
<tr><td>I</td><td>Khối kiến thức chung</td><td>10</td><td></td></tr>
<tr><td>II</td><td>Khối kiến thức ngành</td><td>5</td><td></td></tr>
<tr><td>V.3</td><td>Các học phần bổ trợ</td><td>5</td><td></td></tr>
<tr><td>71</td><td>INT3103</td><td>Tối ưu hóa</td><td>3</td></tr>
<tr><td>72</td><td colspan="7">Các học phần thuộc các nhóm ngành Điện tử-viễn thông, Kinh tế, Luật</td></tr>
</table>
</body>
</html>
"""

# 2. Mock Transcript with "Marketing"
mock_completed_map = {
    "MAT1093": {"code": "MAT1093", "name": "Đại số", "credits": 4, "grade_4": 4.0}, # General
    "INE1050": {"code": "INE1050", "name": "Nguyên lý Marketing", "credits": 3, "grade_4": 3.0}, # Should be caught by V.3
    "INT3103": {"code": "INT3103", "name": "Tối ưu hóa", "credits": 3, "grade_4": 4.0}, # Matches V.3 explicitly
}

# 3. Parse
print("--- Parsing ---")
structure = parse_curriculum_from_html_content(mock_html)
with open("debug_open_parse.json", "w", encoding="utf-8") as f:
    json.dump(structure, f, indent=2, ensure_ascii=False)

# 4. Compute Missing
print("\n--- Computing Missing Credits ---")
analysis = compute_curriculum_missing_credits(structure, mock_completed_map)
with open("debug_open_missing.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print("Done. Check debug_open_parse.json and debug_open_missing.json")
