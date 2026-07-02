"""
Unit tests for Standup Summarizer Agent — no API calls, run in <1s.

End-to-end behaviour (real LLM calls) is covered by the eval suite:
    python -m tests.eval_golden_dataset

Run directly:  python -m tests.test_agent
Or via pytest: pytest tests/test_agent.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Dummy credentials so imports work on a fresh clone with no real .env.
# These tests never call the API.
load_dotenv(Path(__file__).parent.parent / ".env.test", override=True)

from langgraph.graph import END

from src.agent import TOOL_SCHEMAS, should_continue
from src.config import MAX_ITERATIONS
from src.tools import _ensure_json, format_output


# ── format_output (pure function) ──────────────────────────────────────────

def test_format_output_shape():
    result = format_output.invoke({
        "done": ["API endpoint"],
        "in_progress": ["Frontend"],
        "blockers": [{"description": "Staging env", "owner": "DevOps", "severity": "high"}],
        "actions": [],
    })
    data = json.loads(result)
    assert "standup_summary" in data
    assert data["standup_summary"]["done"] == ["API endpoint"]
    assert data["metadata"]["item_counts"] == {
        "done": 1, "in_progress": 1, "blockers": 1, "actions": 0,
    }


def test_format_output_empty_lists():
    result = format_output.invoke({
        "done": [], "in_progress": [], "blockers": [], "actions": [],
    })
    data = json.loads(result)
    assert data["standup_summary"] == {
        "done": [], "in_progress": [], "blockers": [], "actions": [],
    }


# ── _ensure_json (LLM output salvage) ──────────────────────────────────────

def test_ensure_json_passes_valid_json_through():
    raw = '{"blockers": [], "risks": []}'
    assert _ensure_json(raw, {}) == raw


def test_ensure_json_salvages_markdown_fences():
    raw = '```json\n{"blockers": [{"description": "x"}], "risks": []}\n```'
    data = json.loads(_ensure_json(raw, {"blockers": [], "risks": []}))
    assert data["blockers"][0]["description"] == "x"
    assert "_parse_error" not in data


def test_ensure_json_falls_back_on_garbage():
    data = json.loads(_ensure_json("no json here at all", {"done": []}))
    assert data["done"] == []
    assert "_parse_error" in data


def test_ensure_json_takes_fenced_json_after_reasoning_notes():
    # The prompts ask for brief notes first, then the JSON in a fence last
    raw = (
        "Alice finished the API - done. Bob is blocked on CI.\n\n"
        '```json\n{"done": ["Alice: finished the API"], "in_progress": []}\n```'
    )
    data = json.loads(_ensure_json(raw, {"done": [], "in_progress": []}))
    assert data["done"] == ["Alice: finished the API"]
    assert "_parse_error" not in data


def test_ensure_json_skips_stray_braces_in_notes():
    # A stray {token} in the notes must not derail extraction of the real JSON
    raw = 'Note: severity {high} for CI.\n{"blockers": [{"description": "CI down"}], "risks": []}'
    data = json.loads(_ensure_json(raw, {"blockers": [], "risks": []}))
    assert data["blockers"][0]["description"] == "CI down"
    assert "_parse_error" not in data


# ── Tool schemas (Anthropic format) ────────────────────────────────────────

def test_tool_schemas_cover_all_three_tools():
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {"extract_blockers", "summarize_progress", "format_output"}


def test_tool_schemas_are_valid_anthropic_shape():
    for schema in TOOL_SCHEMAS:
        assert schema["description"], f"{schema['name']} missing description"
        assert schema["input_schema"]["type"] == "object"
        for prop in schema["input_schema"]["properties"].values():
            if prop.get("type") == "array":
                assert "items" in prop  # Anthropic API requires items on arrays


# ── should_continue routing ────────────────────────────────────────────────

def _state(blocks, iterations=1):
    return {"messages": [{"role": "assistant", "content": blocks}], "iterations": iterations}


def test_should_continue_routes_tool_use_to_call_tools():
    state = _state([SimpleNamespace(type="tool_use")])
    assert should_continue(state) == "call_tools"


def test_should_continue_ends_on_text_only():
    state = _state([SimpleNamespace(type="text")])
    assert should_continue(state) == END


def test_should_continue_ends_at_max_iterations():
    state = _state([SimpleNamespace(type="tool_use")], iterations=MAX_ITERATIONS)
    assert should_continue(state) == END


# ── Plain runner (ASCII output for Windows cp1250 console) ─────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
