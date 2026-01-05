
import json

def test_deduplication_logic():
    print("--- Testing Server Deduplication Logic ---")
    
    # Scenario: Agent already has Semester 231 parsed (maybe from File 1 or previous chunk)
    merged_data = {
        "student_info": {"name": "Test Student"},
        "semesters": [
            {
                "semester_code": "231",
                "semester_title": "Học kỳ 1 năm 2023",
                "subjects": [
                    {"code": "INT1000", "name": "Tin học cơ sở", "grade_4": 4.0}, # Duplicate target
                    {"code": "INT1001", "name": "Toán rời rạc", "grade_4": 3.0}   # Unique to first set
                ]
            }
        ],
        "overview": {}
    }

    # Scenario: Agent parses the SAME semester again (e.g. from duplicate text or File 2)
    # This contains INT1000 (Duplicate) and INT1002 (New subject)
    incoming_data = {
        "student_info": {"name": "Test Student"},
        "semesters": [
            {
                "semester_code": "231", # Matches existing code
                "subjects": [
                    {"code": "INT1000", "name": "Tin học cơ sở", "grade_4": 4.0}, # SHOULD BE IGNORED
                    {"code": "INT1002", "name": "Giải tích", "grade_4": 3.5}      # SHOULD BE ADDED
                ]
            },
            {
                "semester_code": "232", # New Semester
                "subjects": [
                    {"code": "PHY1001", "name": "Vật lý", "grade_4": 3.0}
                ]
            }
        ],
        "overview": {}
    }

    print(f"Initial Subject Count (Sem 231): {len(merged_data['semesters'][0]['subjects'])}")

    # --- SIMULATING SERVER LOGIC ---
    merged = merged_data
    sems = incoming_data.get("semesters")
    
    if isinstance(sems, list):
        # Smart Merge: Deduplicate semesters and subjects
        existing_sems = {s.get("semester_code"): s for s in merged["semesters"] if s.get("semester_code")}
        
        for incoming_sem in sems:
            sem_code = incoming_sem.get("semester_code")
            if not sem_code:
                merged["semesters"].append(incoming_sem)
                continue

            if sem_code in existing_sems:
                # Merge subjects into existing semester
                target_sem = existing_sems[sem_code]
                existing_subjects = {subj.get("code"): subj for subj in target_sem.get("subjects", []) if subj.get("code")}
                
                inc_subjects = incoming_sem.get("subjects", [])
                for sub in inc_subjects:
                    sub_code = sub.get("code")
                    if sub_code and sub_code in existing_subjects:
                        print(f"[LOGIC] Detected duplicate subject '{sub_code}' in sem '{sem_code}'. Skipping.")
                        continue # Skip duplicate subject
                    
                    # Add new subject
                    print(f"[LOGIC] Adding new subject '{sub_code}' to sem '{sem_code}'.")
                    if "subjects" not in target_sem: target_sem["subjects"] = []
                    target_sem["subjects"].append(sub)
            else:
                # New semester found
                print(f"[LOGIC] Adding new semester '{sem_code}'.")
                merged["semesters"].append(incoming_sem)
                existing_sems[sem_code] = incoming_sem
    # --- END LOGIC ---

    print("\n--- Final Results ---")
    for sem in merged["semesters"]:
        print(f"Semester {sem['semester_code']}:")
        for sub in sem['subjects']:
            print(f"  - {sub['code']}: {sub['grade_4']}")
            
    # Verification assertions
    sem_231 = next(s for s in merged["semesters"] if s["semester_code"] == "231")
    count_231 = len(sem_231["subjects"])
    if count_231 == 3:
        print("\n[SUCCESS] Semester 231 has 3 subjects (Merged correctly).")
    else:
        print(f"\n[FAILURE] Semester 231 has {count_231} subjects (Expected 3).")

if __name__ == "__main__":
    test_deduplication_logic()
