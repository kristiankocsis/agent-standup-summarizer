import json
from datetime import datetime, timezone
from langchain_core.tools import tool
from anthropic import Anthropic
from .config import ANTHROPIC_API_KEY, MODEL


_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Focused single-turn LLM call for extraction tasks."""
    response = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


@tool
def extract_blockers(transcript: str) -> str:
    """
    Extract blockers and risks from the standup transcript.
    Call this FIRST. Returns JSON with 'blockers' and 'risks' lists.
    Each blocker has: description, owner, severity (high/medium/low).
    """
    prompt = f"""Extract all blockers and risks from this standup transcript.

BLOCKER = anything preventing a team member from making progress.
Include EXPLICIT blockers: "blocked by X", "stuck on X", "can't proceed until Y".
Include IMPLICIT blockers: "waiting for X", "need Y from Z", "X is down/broken/missing", "pending approval/review".

RISK = something that might become a blocker soon (e.g., "looks like it could take longer").

Severity:
- high: stops all progress today
- medium: slows progress, workaround exists
- low: minor inconvenience

RULES:
- Normal in-progress work is NOT a blocker. Being busy ≠ blocked.
- "Design review ongoing" alone is NOT a blocker unless someone is waiting on it to proceed.
- Include the person's name as owner.

Transcript:
{transcript}

Respond with ONLY valid JSON, no markdown, no explanation:
{{"blockers": [{{"description": "...", "owner": "name", "severity": "high|medium|low"}}], "risks": [{{"description": "...", "owner": "name"}}]}}

If nothing found: {{"blockers": [], "risks": []}}"""

    raw = _call_llm(prompt)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return json.dumps({"blockers": [], "risks": [], "_parse_error": raw[:200]})


@tool
def summarize_progress(transcript: str) -> str:
    """
    Extract done items, in-progress work, and follow-up actions from the transcript.
    Call this SECOND, after extract_blockers.
    Returns JSON with 'done', 'in_progress', and 'actions' lists.
    """
    prompt = f"""Extract completed work, current work, and follow-up actions from this standup.

DONE — work completed before this standup:
- Past tense: finished, completed, wrapped up, deployed, merged, shipped, released, resolved, fixed
- "PR is up for review" = DONE (the PR was created)
- "tests are passing" = DONE
- NOT done: "almost done", "nearly finished", "one more day" — those are IN PROGRESS

IN PROGRESS — work happening right now:
- Present/future tense: working on, implementing, starting, continuing, building, reviewing
- "almost done with X" = IN PROGRESS
- "another day or two on X" = IN PROGRESS
- "design review is ongoing" = IN PROGRESS

ACTIONS — concrete follow-up tasks with a clear owner:
- "I'll fix X", "will do Y", "need to Z", "can you handle W"
- Only explicit commitments, not vague mentions

CRITICAL RULES:
- Parse each clause separately: "finished X and now working on Y" → X is DONE, Y is IN PROGRESS
- Each item should include the person's name, e.g. "Alice: finished auth API"
- One item per task (not one item per sentence)
- If a person mentions nothing done/in-progress, skip them for that category

Transcript:
{transcript}

Respond with ONLY valid JSON, no markdown, no explanation:
{{"done": ["Alice: ...", "Bob: ..."], "in_progress": ["Alice: ...", "Carol: ..."], "actions": [{{"task": "...", "owner": "name", "deadline": "timeframe or null"}}]}}

Use empty lists if nothing found for a category."""

    raw = _call_llm(prompt)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return json.dumps({"done": [], "in_progress": [], "actions": [], "_parse_error": raw[:200]})


@tool
def format_output(done: list, in_progress: list, blockers: list, actions: list) -> str:
    """
    Combine extracted data into the final structured standup summary.
    Call this LAST with data from extract_blockers and summarize_progress results.
    Returns the final JSON output ready to share.
    """
    return json.dumps({
        "standup_summary": {
            "done": done,
            "in_progress": in_progress,
            "blockers": blockers,
            "actions": actions,
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item_counts": {
                "done": len(done),
                "in_progress": len(in_progress),
                "blockers": len(blockers),
                "actions": len(actions),
            },
        },
    }, indent=2)


TOOLS = [extract_blockers, summarize_progress, format_output]
