import json
from typing import Any
from langchain_core.tools import tool


@tool
def extract_blockers(transcript: str) -> str:
    """
    Extract blockers and risks from a standup transcript.

    This tool identifies obstacles, blockers, and impediments mentioned by team members.
    Look for keywords like: blocked, waiting, can't, stuck, need, missing, broken, issue.

    Returns JSON with:
    - blockers: list of {description, owner, severity}
    - risks: list of {description, owner, impact}

    Args:
        transcript: Full or partial standup transcript

    Returns:
        JSON string with extracted blockers and risks
    """
    # This is a placeholder for the tool logic
    # In production, this would call Claude or use rule-based extraction
    return json.dumps({
        "blockers": [
            {
                "description": "Placeholder: implement actual extraction logic",
                "owner": "Unknown",
                "severity": "medium"
            }
        ],
        "risks": [],
        "extraction_status": "placeholder"
    })


@tool
def summarize_progress(transcript: str) -> str:
    """
    Create concise summaries of completed and ongoing work.

    This tool extracts what team members have accomplished and what they're currently working on.
    Group by person and task, keep summaries to 1-2 sentences per person.

    Returns JSON with:
    - done: list of completed items with owner
    - in_progress: list of current work items with owner
    - person_summaries: detailed summary per person

    Args:
        transcript: Full or partial standup transcript

    Returns:
        JSON string with progress summaries
    """
    return json.dumps({
        "done": [],
        "in_progress": [],
        "person_summaries": {},
        "summary_status": "placeholder"
    })


@tool
def format_output(done: list, in_progress: list, blockers: list, actions: list) -> str:
    """
    Format extracted information as a structured summary.

    This tool takes categorized information and produces a clean, actionable summary
    suitable for sharing in Slack or Notion.

    Args:
        done: List of completed items (strings or dicts)
        in_progress: List of items in progress
        blockers: List of blockers with owners
        actions: List of follow-up actions

    Returns:
        Formatted JSON summary
    """
    return json.dumps({
        "standup_summary": {
            "done": done,
            "in_progress": in_progress,
            "blockers": blockers,
            "actions": actions
        },
        "timestamp": None,  # Will be set by agent
        "status": "placeholder"
    }, indent=2)


# Tool registry for easy access
TOOLS = [
    extract_blockers,
    summarize_progress,
    format_output,
]
