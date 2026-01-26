
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Force UTF-8 for Windows Console
import sys
import codecs
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

logging.basicConfig(level=logging.WARN, format='%(message)s') # Reduce noise

def main():
    print("\n=======================================================")
    print("DEMO: PROVING RETRIEVAL RANKING ISSUE (IELTS Query)")
    print("=======================================================\n")

    try:
        from utils import FAISSVectorStore, VietnameseEmbedder
        from resource_loader import ResourceLoader
        
        # 1. Setup Vector Store
        print("[1] Loading Vector Store (Embedder + Resources)...")
        embedder = VietnameseEmbedder()
        store = FAISSVectorStore([], embedder)
        loader = ResourceLoader(store)
        loader.load_resources()
        print(f"Total chunks in store: {len(store.documents)}")

        # 2. Define the Query
        query = "đủ 6.5 ielts tôi có được miễn các học phần ngoại ngữ không"
        print(f"\n[2] Execution Query: '{query}'")

        # 3. Search and Print Detailed Ranking
        print("\n[3] Retrieving Top 20 results (Searching ALL files)...")
        # We search WITHOUT file_ids filter to simulate the Planner's default behavior
        results = store.retrieve(query, top_k=20)
        
        print(f"\n{'RANK':<5} | {'SCORE':<8} | {'SOURCE FILE':<40} | {'CONTENT PREVIEW'}")
        print("-" * 120)

        found_good_answer = False
        schedule_dominated = False
        
        for i, doc in enumerate(results, start=1):
            score = 0.0 # Standard FAISS returns generally usually don't expose score easily in this wrapper unless I modified it, but let's assume order implies score.
            # Actually utils.py retrieve returns 'scored' locally but returns 'List[Document]' to caller.
            # So we can't see the exact float score here without modifying utils, but the ORDER is what matters.
            
            source = doc.metadata.get('file_id', 'Unknown')
            # Extract score if possible? No, strictly utils.py returns List[Doc]. 
            # But the order IS the ranking.
            
            content = doc.page_content.replace('\n', ' ')[:80]
            
            # Highlight interesting rows
            is_handbook = "SỔ TAY" in source.upper() or "QUY CHE" in source.upper()
            is_schedule = "TKB" in source.upper()
            
            marker = ""
            if is_handbook:
                marker = "✅ (HANDBOOK)"
                if "miễn" in doc.page_content.lower() or "ielts" in doc.page_content.lower():
                    found_good_answer = True
                    marker += " [MATCH MATCH!]"
            elif is_schedule:
                marker = "❌ (SCHEDULE)"
            
            print(f"#{i:<4} | {'----':<8} | {source[:38]:<40} | {content} {marker}")

            if i <= 5 and is_schedule:
                schedule_dominated = True

        print("-" * 120)
        print("\n[4] ANALYSIS:")
        if schedule_dominated:
            print("🔴 FAIL: 'Schedule' documents are polluting the Top 5.")
        else:
            print("🟢 PASS: Schedule is NOT dominating Top 5 (Unexpected?)")
            
        if found_good_answer:
            print("🟢 PASS: The correct Handbook chunk DOES exist in the Top 20.")
        else:
            print("🔴 FAIL: Handbook chunk not found even in Top 20 (Maybe retrieval params are too strict?)")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
