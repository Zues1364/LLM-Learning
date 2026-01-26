 
import sys
import os
import logging
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from mcp_server.server import get_schedule
except ImportError:
    # If direct import fails due to path issues, try relative
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'mcp_server')))
    from server import get_schedule

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("--- TEST TOOL: get_schedule ---")
    
    subjects = ["PEC1008", "PHI1002", "HIS1001", "INT3306"]
    print(f"Querying for: {subjects}")
    
    result_json = get_schedule(subjects)
    
    try:
        data = json.loads(result_json)
        print("\nResult:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Validation
        if isinstance(data, dict) and "error" in data:
            print(f"\n❌ Error: {data['error']}")
            return

        found_count = sum(1 for item in data if item.get("schedule_lines"))
        print(f"\nFound {found_count}/{len(subjects)} subjects.")
        
        if found_count > 0:
            print("✅ Tool is working (found matches in TKB).")
        else:
            print("⚠️ Tool ran but found 0 matches. Check PDF content.")
            
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw output: {result_json}")

if __name__ == "__main__":
    main()
