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
from unittest.mock import patch

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Dummy credentials so imports work on a fresh clone with no real .env.
# These tests never call the API.
load_dotenv(Path(__file__).parent.parent / ".env.test", override=True)

from langgraph.graph import END
from streamlit.testing.v1 import AppTest

from src import agent as agent_module
from src import tools as tools_module
from src.agent import TOOL_SCHEMAS, run_agent, should_continue
from src.config import MAX_ITERATIONS
from src.tools import _call_llm, _ensure_json, format_output


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


# ── Cut-off extraction replies ─────────────────────────────────────────────

def _reply(stop_reason, *blocks):
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), usage=usage)


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _fake_client(create):
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_call_llm_default_cap_is_8192():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _reply("end_turn", _text("ok"))

    with patch.object(tools_module, "_client", _fake_client(create)):
        assert _call_llm("prompt") == "ok"
    assert calls[0]["max_tokens"] == 8192


def _run_agent_with_extraction_reply(reply):
    # The orchestrator would carry on after a swallowed error, so the second
    # reply makes the run look successful unless the error stops it.
    tool_call = SimpleNamespace(
        type="tool_use", id="t1", name="extract_blockers",
        input={"transcript": "Alice: blocked on staging."},
    )
    orchestrator = iter([_reply("tool_use", tool_call), _reply("end_turn", _text("Summary"))])

    with patch.object(agent_module, "_client", _fake_client(lambda **kw: next(orchestrator))), \
            patch.object(tools_module, "_client", _fake_client(lambda **kw: reply)):
        return run_agent("Alice: blocked on staging.")


def test_run_agent_fails_when_extraction_reply_is_cut_off():
    result = _run_agent_with_extraction_reply(_reply("max_tokens", _text("Notes continue and then")))
    assert result["status"] == "error", f"status was {result['status']}"
    assert "max_tokens" in result["error"]


def test_run_agent_fails_when_extraction_reply_is_refused():
    result = _run_agent_with_extraction_reply(_reply("refusal"))
    assert result["status"] == "error", f"status was {result['status']}"
    assert "refusal" in result["error"]


def _run_agent_with_main_loop_reply(reply):
    # Returns the run's result and every extraction call a tool made.
    tool_calls = []

    def extraction(**kwargs):
        tool_calls.append(kwargs)
        return _reply("end_turn", _text("{}"))

    orchestrator = iter([reply, _reply("end_turn", _text("Summary"))])
    with patch.object(agent_module, "_client", _fake_client(lambda **kw: next(orchestrator))), \
            patch.object(tools_module, "_client", _fake_client(extraction)):
        return run_agent("Alice: blocked on staging."), tool_calls


def test_run_agent_fails_when_main_loop_reply_is_cut_off():
    result, _ = _run_agent_with_main_loop_reply(_reply("max_tokens", _text("Partial summary")))
    assert result["status"] == "error", f"status was {result['status']}"
    assert "max_tokens" in result["error"]


def test_run_agent_fails_when_main_loop_reply_is_refused():
    result, _ = _run_agent_with_main_loop_reply(_reply("refusal"))
    assert result["status"] == "error", f"status was {result['status']}"
    assert "refusal" in result["error"]


def test_run_agent_runs_no_tool_from_a_cut_off_tool_call():
    tool_call = SimpleNamespace(
        type="tool_use", id="t1", name="extract_blockers",
        input={"transcript": "Alice: blocked"},
    )
    result, tool_calls = _run_agent_with_main_loop_reply(_reply("max_tokens", tool_call))
    assert tool_calls == [], "a tool ran from a cut-off tool call"
    assert result["status"] == "error", f"status was {result['status']}"
    assert "max_tokens" in result["error"]


def test_run_agent_succeeds_on_a_normal_run():
    tool_call = SimpleNamespace(
        type="tool_use", id="t1", name="format_output",
        input={"done": ["API"], "in_progress": [], "blockers": [], "actions": []},
    )
    result, _ = _run_agent_with_main_loop_reply(_reply("tool_use", tool_call))
    assert result["status"] == "success", f"status was {result['status']}: {result.get('error')}"
    assert result["structured_output"]["done"] == ["API"]


# ── Runs without a summary, and a finished summary ─────────────────────────

def _format_call(**lists):
    fields = {"done": [], "in_progress": [], "blockers": [], "actions": [], **lists}
    return SimpleNamespace(type="tool_use", id="f1", name="format_output", input=fields)


def _run_agent_with_replies(*replies):
    orchestrator = iter(replies)
    with patch.object(agent_module, "_client", _fake_client(lambda **kw: next(orchestrator))):
        return run_agent("Alice: blocked on staging.")


def test_run_agent_fails_when_model_ends_without_summary():
    result = _run_agent_with_replies(_reply("end_turn", _text("Please paste a real standup.")))
    assert result["status"] == "error", f"status was {result['status']}"
    assert "without producing a summary" in result["error"]


def test_run_agent_fails_at_step_limit_without_summary():
    tool_call = SimpleNamespace(
        type="tool_use", id="t1", name="extract_blockers", input={"transcript": "Alice"},
    )
    orchestrator = _fake_client(lambda **kw: _reply("tool_use", tool_call))
    extraction = _fake_client(lambda **kw: _reply("end_turn", _text("{}")))
    with patch.object(agent_module, "_client", orchestrator), \
            patch.object(tools_module, "_client", extraction):
        result = run_agent("Alice: blocked on staging.")
    assert result["status"] == "error", f"status was {result['status']}"
    assert "without producing a summary" in result["error"]


def test_run_agent_keeps_summary_when_closing_reply_is_cut_off():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("max_tokens", _text("Here is the summ")),
    )
    assert result["status"] == "success", f"status was {result['status']}: {result.get('error')}"
    assert result["structured_output"]["done"] == ["API"]


def test_run_agent_keeps_summary_when_closing_reply_is_refused():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("refusal"),
    )
    assert result["status"] == "success", f"status was {result['status']}: {result.get('error')}"
    assert result["structured_output"]["done"] == ["API"]


def test_run_agent_fails_when_a_later_summary_call_is_cut_off():
    # The first summary may be partial, so it must not pass as the final one.
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("max_tokens", _format_call(done=["API", "Forms"])),
    )
    assert result["status"] == "error", f"status was {result['status']}"
    assert "max_tokens" in result["error"]


def test_run_agent_treats_an_empty_summary_as_a_summary():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call()),
        _reply("end_turn", _text("Nothing to report.")),
    )
    assert result["status"] == "success", f"status was {result['status']}: {result.get('error')}"
    assert result["structured_output"]["done"] == []


# ── Streamlit app: empty summaries ─────────────────────────────────────────

_EMPTY_SUMMARY = {"done": [], "in_progress": [], "blockers": [], "actions": []}


def _run_app(result):
    app = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30)
    with patch("src.agent.run_agent", return_value=result):
        app.run()
        app.text_area(key="transcript_input").set_value("Apple cake: mix flour and eggs.").run()
        next(b for b in app.button if b.label == "Summarize").click().run()
    assert not app.exception, app.exception
    return app


def _success_result(summary, output):
    return {"status": "success", "structured_output": summary, "output": output, "iterations": 4}


def test_app_shows_model_reply_for_an_empty_summary():
    app = _run_app(_success_result(_EMPTY_SUMMARY, "This is a recipe, not a standup."))
    assert len(app.warning) == 1 and "empty" in app.warning[0].value
    assert [t.value for t in app.text] == ["This is a recipe, not a standup."]
    assert len(app.success) == 0, "the empty summary was shown as a success"
    assert not any("great standup" in c.value for c in app.caption)
    assert any("Recent runs" in s.value for s in app.sidebar.subheader), "history disappeared"


def test_app_explains_an_empty_summary_without_a_model_reply():
    app = _run_app(_success_result(_EMPTY_SUMMARY, ""))
    assert len(app.warning) == 1 and "empty" in app.warning[0].value
    assert [t.value for t in app.text] == ["The model did not add an explanation."]


def test_app_treats_a_blank_model_reply_as_no_reply():
    app = _run_app(_success_result(_EMPTY_SUMMARY, "  \n "))
    assert [t.value for t in app.text] == ["The model did not add an explanation."]


def test_app_shows_an_error_for_a_run_without_a_summary():
    app = _run_app({"status": "error", "error": "No summary."})
    assert len(app.error) == 1 and "No summary." in app.error[0].value
    assert len(app.warning) == 0


def test_app_shows_model_reply_under_an_error():
    app = _run_app({"status": "error", "error": "No summary.", "output": "This is a recipe."})
    assert len(app.error) == 1
    assert [t.value for t in app.text] == ["This is a recipe."]


def test_run_agent_keeps_model_text_when_there_is_no_summary():
    result = _run_agent_with_replies(_reply("end_turn", _text("Please paste a real standup.")))
    assert result["status"] == "error", f"status was {result['status']}"
    assert result["output"] == "Please paste a real standup."


def test_run_agent_flags_an_unfinished_closing_message():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("max_tokens", _text("Here is the summ")),
    )
    assert result["closing_unfinished"] is True


def test_run_agent_does_not_flag_a_normal_run():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("end_turn", _text("Here is the summary.")),
    )
    assert result["closing_unfinished"] is False


def test_app_shows_a_note_with_a_summary_when_the_last_message_did_not_finish():
    summary = {**_EMPTY_SUMMARY, "done": ["Alice: API"]}
    app = _run_app({**_success_result(summary, "Here is the summ"), "closing_unfinished": True})
    assert len(app.warning) == 1 and "did not finish" in app.warning[0].value
    assert any("Done (1)" in s.value for s in app.subheader)


def test_app_shows_the_note_with_an_empty_summary_too():
    app = _run_app({**_success_result(_EMPTY_SUMMARY, "This is a rec"), "closing_unfinished": True})
    warnings = [w.value for w in app.warning]
    assert len(warnings) == 2
    assert any("did not finish" in w for w in warnings)
    assert any("empty" in w for w in warnings)


def test_run_agent_flags_a_refused_closing_message():
    result = _run_agent_with_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("refusal"),
    )
    assert result["closing_unfinished"] is True


def _run_app_with_model_replies(*replies):
    # Real run_agent and real app; only the model's replies are fake.
    orchestrator = iter(replies)
    app = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30)
    with patch.object(agent_module, "_client", _fake_client(lambda **kw: next(orchestrator))):
        app.run()
        app.text_area(key="transcript_input").set_value("Alice: finished the API.").run()
        next(b for b in app.button if b.label == "Summarize").click().run()
    assert not app.exception, app.exception
    return app


def test_app_shows_summary_and_note_after_a_cut_off_model_reply():
    app = _run_app_with_model_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("max_tokens", _text("Here is the summ")),
    )
    assert len(app.warning) == 1 and "did not finish" in app.warning[0].value
    assert any("Done (1)" in s.value for s in app.subheader)


def test_app_shows_no_note_after_a_normal_model_reply():
    app = _run_app_with_model_replies(
        _reply("tool_use", _format_call(done=["API"])),
        _reply("end_turn", _text("Here is the summary.")),
    )
    assert len(app.warning) == 0
    assert any("Done (1)" in s.value for s in app.subheader)


def test_app_shows_a_summary_with_items_in_any_one_section():
    sections = {
        "in_progress": ["Bob: tests"],
        "blockers": [{"description": "CI down", "owner": "Bob", "severity": "high"}],
        "actions": [{"task": "Fix CI", "owner": "Bob", "deadline": "today"}],
    }
    for name, items in sections.items():
        app = _run_app(_success_result({**_EMPTY_SUMMARY, name: items}, "Summary"))
        assert len(app.warning) == 0, f"warning shown with items in {name}"
        assert len(app.success) == 1, f"no success shown with items in {name}"


def test_app_shows_a_normal_summary_as_before():
    summary = {**_EMPTY_SUMMARY, "done": ["Alice: API"]}
    app = _run_app(_success_result(summary, "Summary"))
    assert len(app.warning) == 0
    assert len(app.success) == 1
    assert any("Done (1)" in s.value for s in app.subheader)
    assert any("Recent runs" in s.value for s in app.sidebar.subheader)


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
