from bs4 import BeautifulSoup
import re
import json

HTML_PATH = "d:/LLM/LLM Learning/data/resources/html/Chương trình đào tạo ngành Khoa học máy tính - Trường Đại học Công nghệ, ĐHQGHN - Univeristy of Engineering and Technology.html"

def extract_curriculum_groups():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    
    groups = {}
    current_group_code = None
    current_group_name = None
    
    for row in rows:
        cells = row.find_all("td")
        if not cells: continue
        
        # Check for Group Header (e.g., V.2.1)
        # Usually checking the first cell content
        first_cell_text = cells[0].get_text(strip=True)
        
        # Heuristic for group code like V.2.1 or V.2
        if re.match(r"^[IVX]+\.\d+(\.\d+)?$", first_cell_text):
            current_group_code = first_cell_text
            # Identify group name in the next cell (often colspan)
            if len(cells) > 1:
                current_group_name = cells[1].get_text(strip=True)
                # Cleanup "Nhóm các học phần về" prefix if present
                current_group_name = re.sub(r"Nhóm các học phần về\s*", "", current_group_name, flags=re.IGNORECASE).strip()
                
            groups[current_group_code] = {
                "name": current_group_name,
                "subjects": [],
                "credits_required": 0 # Logic to extract this needed
            }
            # Try to find credit requirement in this row?
            # looking for "x/y" pattern in any cell
            for cell in cells:
                txt = cell.get_text(strip=True)
                parts = txt.split("/")
                if len(parts) == 2 and parts[0].isdigit():
                     groups[current_group_code]["credits_required"] = int(parts[0])

        elif first_cell_text.isdigit() and current_group_code:
            # Subject Row
            # Expected cols: STT | Code | Name | Credits ...
            if len(cells) >= 4:
                code = cells[1].get_text(strip=True)
                name = cells[2].get_text(strip=True)
                try:
                    credits = int(cells[3].get_text(strip=True))
                except:
                    credits = 0
                
                if code and name:
                     groups[current_group_code]["subjects"].append({
                         "code": code,
                         "name": name,
                         "credits": credits
                     })

    return groups

if __name__ == "__main__":
    result = extract_curriculum_groups()
    print(json.dumps(result, indent=2, ensure_ascii=False))
