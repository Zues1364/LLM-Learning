
import sys
import os
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from utils import FAISSVectorStore, VietnameseEmbedder
    from resource_loader import ResourceLoader
    from agno.agent import Agent
    from agno.models.google import Gemini
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_scheduler_agent():
    instructions = (
        "Bạn là trợ lý lập lịch cá nhân. Nhiệm vụ của bạn là trích xuất thông tin lịch học từ văn bản được cung cấp.\n"
        "Đầu vào: Một đoạn văn bản chứa thông tin thời khóa biểu (có thể là bảng hoặc danh sách).\n"
        "Đầu ra: Trả về kết quả dưới dạng JSON với cấu trúc sau (nếu không tìm thấy thì để null/empty):\n"
        "[\n"
        "  {\n"
        "    \"subject_code\": \"Mã môn (ví dụ: INT3306)\",\n"
        "    \"subject_name\": \"Tên môn\",\n"
        "    \"class_code\": \"Mã lớp (ví dụ: INT3306 1)\",\n"
        "    \"credits\": Số tín chỉ,\n"
        "    \"schedule\": [\n"
        "       { \"day\": \"Thứ mấy (2,3,4,5,6,7,CN)\", \"period\": \"Tiết (ví dụ: 1-3)\", \"room\": \"Phòng học\" }\n"
        "    ],\n"
        "    \"lecturer\": \"Giảng viên (nếu có)\",\n"
        "    \"group\": \"Nhóm (CL, 1, 2...)\"\n"
        "  }\n"
        "]\n"
        "Chỉ trả về JSON thuần túy, không có Markdown block (```json)."
    )
    return Agent(
        name="Scheduler Agent",
        model=Gemini(id="gemini-2.5-flash"),
        instructions=instructions,
        markdown=False,
    )

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("--- RUNNING FULL SCHEDULER GENERATION ---")
    
    # 1. Setup Vector Store
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embedder = VietnameseEmbedder()
    store = FAISSVectorStore([], embedder)
    loader = ResourceLoader(store)
    loader.load_resources()
    
    # 2. Define Subjects
    subjects = [
        "PEC1008", # Kinh te chinh tri
        "PHI1002", # CNXH Khoa hoc
        "HIS1001", # Lich su Dang
        "INT3131", # Du an khoa hoc
        "INT3133", # Ky nghe yeu cau
        "INT3111", # Quan ly du an phan mem
        "INT3420E" # Hoc sau
    ]
    
    agent = get_scheduler_agent()
    final_schedule = []
    
    print(f"\nProcessing {len(subjects)} subjects...")
    
    for code in subjects:
        print(f"\n========== PROCESSING: {code} ==========")
        
        # A. Retrieval
        query = f"{code} hoc ky 252 HKII thoi khoa bieu TKB lịch học"
        
        # Scope restricted to schedule PDF usually (but we let it search global if needed)
        # In this specific case, we know the file name
        schedule_pdf_name = "Signed.[TKB] DỰ KIẾN TKB HKII NĂM HỌC 2025-2026 (SV).pdf"
        scope = [schedule_pdf_name]
        
        chunks = store.retrieve(query, top_k=3, file_ids=scope)
        
        if not chunks:
             print(f"❌ No chunks found for {code}")
             continue
             
        # Detect correct chunk
        best_chunk = chunks[0]
        print(f"✅ Found chunk from page {best_chunk.metadata.get('page')}")
        
        # B. Generation
        context = best_chunk.page_content
        prompt = f"Trích xuất lịch học cho môn {code} từ thông tin sau:\n\n{context}"
        
        try:
            response = agent.run(prompt)
            raw_content = response.content
            # Clean possible markdown format
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(raw_content)
            print(f"  📝 Extracted: {json.dumps(data, ensure_ascii=False)}")
            if isinstance(data, list):
                final_schedule.extend(data)
            elif isinstance(data, dict):
                final_schedule.append(data)
                
        except Exception as e:
            print(f"  ⚠️ Generation Failed: {e}")
            print(f"  Raw response: {response.content if 'response' in locals() else 'None'}")

    # 3. Final Summary
    print("\n\n================ FINAL SCHEDULE ================")
    print(json.dumps(final_schedule, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
