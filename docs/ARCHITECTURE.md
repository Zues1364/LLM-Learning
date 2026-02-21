# System Architecture (Planner + MCP + Deterministic Core)

## Summary
This system is organized as an orchestrated multi-agent RAG stack with strict boundaries:

1. `app.py` is the orchestrator layer (`/ask`).
2. `agents.py` provides reasoning agents (planner + answer/advisor formatter).
3. `mcp_server/server.py` provides deterministic academic tools and retrieval tools.

The goal is to avoid overlap: planner decides tool, deterministic core computes facts, advisor formats one final response.

## Call Flow

`/ask` request flow:

1. Validate session + selected program.
2. Build planner prompt with `[SESSION]`, `[PROGRAM]`, `[FILES]`.
3. Run planner and parse JSON decision.
4. If planner output is invalid, fallback to deterministic MCP route.
5. Execute chosen context path.
6. Generate final answer from `AnswerGeneratorAgent`.
7. Persist memory and return response.

Text diagram:

`/ask (app.py)`  
-> `Planner Agent (agents.py)`  
-> `MCP tool selection`  
-> `consult_advisor / retrieve_chunks / ... (mcp_server/server.py)`  
-> `deterministic pipeline + evidence`  
-> `Academic advisor formatter (agents.py)`  
-> `/ask response`

## Allowed Dependencies

- `app.py` may call planner + MCP client + answer agent.
- Planner may call MCP tools.
- `consult_advisor` may call deterministic helpers (`analyze_transcript`, `analyze_curriculum`, `compute_missing_subjects`, `calculate_gpa_feasibility`, `check_course_schedule`).
- Academic advisor agent only formats provided context.

## Disallowed Overlap

- Advisor formatter should not re-run transcript/curriculum/schedule retrieval pipeline.
- Planner should not return prose; must return a single JSON object.
- `consult_advisor` should avoid duplicate schedule checks in one request.

## Reliability Guards

- Session-scoped lock in `/ask` prevents concurrent overlapping answers in same session.
- Planner parse fallback ensures user does not see raw JSON parse errors.
- Schedule text extraction is cached to reduce repeated heavy PDF parsing and noisy logs.

