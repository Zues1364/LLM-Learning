
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Configure logging to show info but not spam
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    print("--- DEBUG: Personal Schedule Planner Retrieval ---")
    
    # 1. Setup Vector Store
    try:
        from utils import FAISSVectorStore, VietnameseEmbedder
        from resource_loader import ResourceLoader
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        resources_dir = os.path.join(base_dir, "data", "resources")
        
        print(f"Loading resources from: {resources_dir}")
        embedder = VietnameseEmbedder()
        store = FAISSVectorStore([], embedder)
        loader = ResourceLoader(store)
        loader.load_resources()
        
        # Ensure Schedule PDF is in scope (Mocking server.py logic)
        schedule_pdf_name = "Signed.[TKB] DỰ KIẾN TKB HKII NĂM HỌC 2025-2026 (SV).pdf"
        if schedule_pdf_name not in loader.loaded_resources:
            print(f"WARNING: Schedule PDF not loaded. Loaded resources: {len(loader.loaded_resources)}")
        
        # 2. Define Subjects to Check (from User's Case)
        subjects = [
            "PEC1008", # Kinh te chinh tri
            "PHI1002", # CNXH Khoa hoc
            "HIS1001", # Lich su Dang
            "INT3131", # Du an khoa hoc
            "INT3133", # Ky nghe yeu cau
            "INT3111", # Quan ly du an phan mem
            "INT3420E" # Hoc sau (Example elective)
        ]
        
        print(f"\n--- Checking {len(subjects)} Subjects ---")
        
        for code in subjects:
            print(f"\n[Checking: {code}]")
            
            # Construct Query (Same logic as server.py)
            query_parts = [code]
            target_semester = "252" # Assume next semester
            query_parts.append(f"hoc ky {target_semester}")
            if str(target_semester).endswith("2"):
                query_parts.append("HKII")
                query_parts.append("Học kỳ 2")
                query_parts.append("Học kỳ II")
                query_parts.append("Semester 2")
            
            query_parts.append("thoi khoa bieu TKB lich hoc")
            query = " ".join(query_parts)
            
            # Retrieve
            # We explicitly strictly filter? Or just search? 
            # In server.py we do: file_ids=search_scope
            # Here we simulate the scope including the schedule PDF
            scope = [schedule_pdf_name]
            
            chunks = store.retrieve(query, top_k=5, file_ids=scope)
            
            if not chunks:
                # Fallback global
                chunks = store.retrieve(query, top_k=5)
            
            if chunks:
                print(f"  ✅ Found {len(chunks)} chunks.")
                # Log only a small part of the first chunk
                best = chunks[0]
                content_preview = best.page_content.replace('\n', ' ')[:200]
                print(f"  snippet: {content_preview}...")
                
                # Check if specific schedule keywords exist in the chunk
                lower_content = best.page_content.lower()
                if "thứ" in lower_content or "ca" in lower_content:
                    print("  -> Contains Schedule Keywords (Thu/Ca)")
                else:
                    print("  ⚠️ Content might not be a schedule table.")
            else:
                print("  ❌ NOT FOUND.")
                
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
