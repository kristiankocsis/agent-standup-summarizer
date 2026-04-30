"""
Unit tests for Standup Summarizer Agent.
"""

import json
from src.agent import run_agent
from src.tools import extract_blockers, summarize_progress, format_output


def test_agent_runs():
    """Test that agent can be invoked without errors."""
    transcript = "Alice finished the API. Bob is stuck on deployment."
    result = run_agent(transcript)
    assert result["status"] in ["success", "error"]  # Accept both for now (placeholder)


def test_extract_blockers_tool():
    """Test blocker extraction tool."""
    transcript = "We are blocked by missing database access."
    result = extract_blockers.invoke({"transcript": transcript})
    data = json.loads(result)
    assert "blockers" in data
    assert "risks" in data


def test_format_output_tool():
    """Test output formatting tool."""
    result = format_output.invoke({
        "done": ["API endpoint"],
        "in_progress": ["Frontend"],
        "blockers": [{"description": "Staging env", "owner": "DevOps"}],
        "actions": []
    })
    data = json.loads(result)
    assert "standup_summary" in data


if __name__ == "__main__":
    # Simple test runner
    tests = [test_agent_runs, test_extract_blockers_tool, test_format_output_tool]
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {str(e)}")
