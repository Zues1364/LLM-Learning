import importlib
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def test_parse_planner_output_handles_fenced_json_with_raw_newlines():
    app_mod = importlib.import_module("app")
    raw = """```json
{"source": "academic_advisor", "context": "Dong 1
Dong 2", "memory": "", "chunk_index": null}
```"""

    parsed = app_mod._parse_planner_output(raw)
    assert isinstance(parsed, dict)
    assert parsed["source"] == "academic_advisor"
    assert "Dong 1" in parsed["context"]
    assert "Dong 2" in parsed["context"]


def test_parse_planner_output_heuristic_recovers_source_context_when_json_broken():
    app_mod = importlib.import_module("app")
    raw = """```json
{"source": "academic_advisor", "context": "Lich hoc: INT4050 - mo lop
PHI1002 - mo lop", "memory": "m", "chunk_index": null
```"""

    parsed = app_mod._parse_planner_output(raw)
    assert isinstance(parsed, dict)
    assert parsed["source"] == "academic_advisor"
    assert "INT4050" in parsed["context"]
    assert "PHI1002" in parsed["context"]
    assert parsed["chunk_index"] is None


def test_parse_planner_output_returns_none_when_no_json_shape():
    app_mod = importlib.import_module("app")
    raw = "xin loi, toi khong co du lieu"

    parsed = app_mod._parse_planner_output(raw)
    assert parsed is None
