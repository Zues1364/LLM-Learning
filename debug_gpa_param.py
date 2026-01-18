
import sys
# Mock the function EXACTLY as it is in server.py now to verify logic

def calculate_gpa_feasibility_mock(
    transcript_data,
    curriculum_total_credits=None,
    target_gpa=None,
    # Policy flavors (Default: VNU UET 2023-2024)
    mandatory_retake_grade=0.0, # F must retake
    improve_threshold=1.5,      # Grades <= 1.5 (D+) can be improved
    improve_target_grade=4.0,   # Assume improvement leads to A
):
    
    # Mock _build_completed_subjects -> flattening transcript
    completed_map = {}
    for sem in transcript_data.get("semesters", []):
        for sub in sem.get("subjects", []):
            completed_map[sub["code"]] = sub
            
    # --- LOGIC START (Copied from server.py) ---
    
    # 1. Calculate strictly "Secure" credits (>= 2.0) vs "Retake-able" (<= 1.5)
    secure_points = 0.0
    secure_credits = 0
    
    retake_mandatory_credits = 0 # F
    retake_optional_credits = 0  # D, D+
    
    retake_candidates = []

    for s in completed_map.values():
        cr = s.get("credits") or 0
        g4 = s.get("grade_4")
        if g4 is None or cr == 0: continue
        
        # Policy Check:
        if g4 <= mandatory_retake_grade + 0.01: # F
             retake_mandatory_credits += cr
             retake_candidates.append(s)
        elif g4 <= improve_threshold: # D, D+
             retake_optional_credits += cr
             secure_points += (g4 * cr) 
             secure_credits += cr
             retake_candidates.append(s)
        else:
             secure_points += (g4 * cr)
             secure_credits += cr

    # Total Curriculum
    curriculum_total = curriculum_total_credits or 130
    if not curriculum_total:
         curriculum_total = max(secure_credits + retake_optional_credits + retake_mandatory_credits, 130)

    # Missing from curriculum (never taken)
    credits_attempted = secure_credits + retake_mandatory_credits 
    credits_never_taken = max(curriculum_total - credits_attempted, 0)
    
    # MAX GPA SCENARIO:
    real_secure_points = 0.0
    for s in completed_map.values():
        g4 = s.get("grade_4")
        # Secure means > improve_threshold
        if g4 is not None and g4 > improve_threshold: 
             real_secure_points += g4 * (s.get("credits") or 0)
             
    credits_to_ace = retake_mandatory_credits + retake_optional_credits + credits_never_taken
    max_total_points = real_secure_points + (credits_to_ace * improve_target_grade)
    
    max_possible_gpa = (max_total_points / curriculum_total) if curriculum_total > 0 else 0.0

    feasible = None
    if target_gpa is not None:
        feasible = target_gpa <= max_possible_gpa + 1e-6

    # Sort retake candidates
    retake_candidates.sort(key=lambda x: (x.get("grade_4") or 0))

    return {
        "max_possible_gpa": round(max_possible_gpa, 4),
        "retake_candidates": [s['code'] for s in retake_candidates],
        "policy_note": f"Policy: Retake <= {mandatory_retake_grade}, Improve <= {improve_threshold}, Target Grade: {improve_target_grade}"
    }

# --- TEST DATA ---
mock_transcript = {
    "semesters": [
        {
            "subjects": [
                {"code": "MATH", "branch": "MATH", "grade_4": 1.0, "credits": 4}, # D (1.0)
                {"code": "PHYS", "branch": "PHYS", "grade_4": 0.0, "credits": 3}, # F (0.0)
                {"code": "CODE", "branch": "CODE", "grade_4": 4.0, "credits": 3}, # A (4.0)
                {"code": "HIST", "branch": "HIST", "grade_4": 2.0, "credits": 2}, # C (2.0)
            ]
        }
    ]
}

print("1. Testing Default Policy (VNU)...")
# Default: F must retake, D/D+ can improve. C cannot.
res1 = calculate_gpa_feasibility_mock(mock_transcript, curriculum_total_credits=130, target_gpa=3.9)
print(f"Res1 Max GPA: {res1['max_possible_gpa']}")
# Expected: Same as before ~3.9692
if res1['max_possible_gpa'] == 3.9692:
     print("SUCCESS: Default VNU logic maintained.")
else:
     print(f"FAILURE: Expected 3.9692, got {res1['max_possible_gpa']}")

print("\n2. Testing Custom Policy (Allow C improvement)...")
# Custom: Allow improving C (2.0)
res2 = calculate_gpa_feasibility_mock(
    mock_transcript, 
    curriculum_total_credits=130, 
    improve_threshold=2.0
)
print(f"Res2 Max GPA: {res2['max_possible_gpa']}")
# Logic:
# Secure > 2.0: Only CODE (3*4=12).
# Improve: HIST (2.0) now eligible! MATH (1.0). F (0.0).
# Credits to Ace: HIST(2) + MATH(4) + PHYS(3) + Remaining(118).
# Total Ace: (2+4+3+118) * 4 = 127 * 4 = 508.
# Total Max Point = 12 + 508 = 520.
# Max GPA = 520 / 130 = 4.0.
if res2['max_possible_gpa'] == 4.0:
     print("SUCCESS: Custom policy correctly allows improving C.")
else:
     print(f"FAILURE: Expected 4.0, got {res2['max_possible_gpa']}")

print("\n3. Testing Candidates...")
if "HIST" in res2['retake_candidates']:
    print("SUCCESS: HIST is now a candidate.")
else:
    print("FAILURE: HIST missing from candidates.")
