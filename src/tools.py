import json
import re
from datetime import datetime, timezone
from langchain_core.tools import tool
from anthropic import Anthropic
from .config import ANTHROPIC_API_KEY, MODEL, LANGFUSE_ENABLED

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Langfuse (optional) ────────────────────────────────────────────────────
try:
    from langfuse import observe as _lf_observe, get_client as _lf_get_client
    _LANGFUSE_OK = LANGFUSE_ENABLED and _lf_get_client().auth_check()
except Exception:
    _LANGFUSE_OK = False


def _first_text(response) -> str:
    """First text block of a response. Newer models (e.g. Sonnet 5 with
    adaptive thinking) may emit thinking blocks before the text, so
    response.content[0] is not guaranteed to be text."""
    return next((b.text for b in response.content if b.type == "text"), "")


@(_lf_observe(as_type="generation", name="tool-llm-call") if _LANGFUSE_OK else lambda f: f)
def _call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Focused single-turn LLM call for extraction tasks."""
    response = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _first_text(response)
    if _LANGFUSE_OK:
        _lf_get_client().update_current_generation(
            model=MODEL,
            input=prompt,
            output=text,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )
    return text


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _ensure_json(raw: str, fallback: dict) -> str:
    """Return the JSON payload of an LLM reply.

    The extraction prompts ask the model to reason in brief notes first and
    emit the final JSON in a ```json fence last, so candidates are tried in
    order: the raw string itself, the LAST fenced block that parses, then
    successive brace spans (skipping stray braces in the notes). Falls back
    to `fallback` with the unparseable head attached for debugging."""
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    for candidate in reversed(_JSON_FENCE.findall(raw)):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    end = raw.rfind("}")
    start = raw.find("{")
    while 0 <= start < end:
        candidate = raw[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
    return json.dumps({**fallback, "_parse_error": raw[:200]})


@tool
def extract_blockers(transcript: str) -> str:
    """
    Extract blockers and risks from the standup transcript.
    Call this FIRST. Returns JSON with 'blockers' and 'risks' lists.
    Each blocker has: description, owner, severity (high/medium/low).
    """
    prompt = f"""Extract all blockers and risks from this standup transcript.

BLOCKER = something stopping or delaying a specific person's CURRENT work.
Explicit: "blocked by X", "stuck on X", "can't proceed until Y".
Implicit: "waiting for X", "need Y from Z before I can continue", "X is down and I can't work".

RISK = something that might become a blocker or endanger a goal soon
(e.g. "might take longer", "sprint goal at risk", "removing X could break Y").

NOT a blocker:
- Normal in-progress work. Being busy is not being blocked.
- "Design review ongoing" alone, unless someone is waiting on it to proceed.
- An issue or anomaly nobody is waiting on (e.g. "dashboard metrics are 5 minutes
  delayed") — that is work to do, not a blocker.
- Planned sequencing ("we need to do X before we can Y") — that is a plan,
  unless someone's current task is stopped waiting for it.
  BUT a defect stopping current work IS a blocker: "found a bug, need to fix
  it before we can move forward" blocks the very testing that found it.
- A blocker already resolved ("the issue that was blocking me is fixed now").

Severity: high = stops progress today | medium = slows progress or workaround exists | low = minor.

Owner = the blocked person's name. If one blocker affects several people, list it
ONCE with all affected names as owner (e.g. "Yuri, Zara, Adrian").
Description = short noun phrase naming what is broken/missing/pending, then what
it blocks — reuse the transcript's own nouns, and name the underlying problem,
never the awaited help:
  "Staging environment down — blocking frontend testing"
  "Team decision on JWT vs sessions pending — blocking auth integration"
  "Production database out of disk space — site down" (NOT "DBA help pending")

First, walk through the transcript speaker by speaker in brief notes (no curly
braces in the notes). For each candidate blocker, name WHO is stopped right now —
if you cannot name a stopped person, it belongs in risks or nowhere.

Transcript:
{transcript}

Then output the final JSON inside a ```json fence as the LAST part of your response:
{{"blockers": [{{"description": "...", "owner": "name", "severity": "high|medium|low"}}], "risks": [{{"description": "...", "owner": "name"}}]}}

If nothing found: {{"blockers": [], "risks": []}}"""

    raw = _call_llm(prompt)
    return _ensure_json(raw, {"blockers": [], "risks": []})


@tool
def summarize_progress(transcript: str) -> str:
    """
    Extract done items, in-progress work, and follow-up actions from the transcript.
    Call this SECOND, after extract_blockers.
    Returns JSON with 'done', 'in_progress', and 'actions' lists.
    """
    prompt = f"""Extract completed work, current work, and follow-up actions from this standup transcript.

DONE — work completed before this standup:
- Past-tense accomplishments: finished, completed, wrapped up, deployed, merged, shipped, fixed
- "PR is up for review" / "submitted X for review" = DONE (the artifact was created)
- "X is live / in production" = DONE even in present tense; "tests are passing" = DONE
- A completed review of someone else's work = DONE ("reviewed and approved X")
NOT done — these belong in IN PROGRESS:
- "almost done", "nearly finished", "should finish today", "one more day"
- Sending a DRAFT for feedback — the work continues:
  "sent draft design for review" → in_progress "X design (draft sent for review)"
- "X is ready for testing/QA" with no completion verb — readiness is not completion:
  "Backend API is ready for testing" → in_progress "Backend API (ready for testing)"

IN PROGRESS — work happening now or planned for today:
- working on, implementing, starting, continuing, building, reviewing, investigating, scoping
- "starting X next" / "X after that" = IN PROGRESS (planned for today)
- BLOCKED work stays IN PROGRESS with a short note:
  "Salesforce integration (blocked)", "End-to-end testing (pending staging)"
- A person firefighting or troubleshooting their own blocker — list that work too:
  "Mobile app build troubleshooting", "Database disk space permanent fix",
  "Dev environment setup (awaiting access)"
- "need to do X before we can Y" = X is upcoming work: list it ("X planning")
- An issue the reporter is expected to address = IN PROGRESS for them ("Dashboard metrics fix")
- Monitoring production, presenting work for feedback, helping a colleague = IN PROGRESS

ACTIONS — follow-up tasks that resolve a problem or were explicitly committed to:
- EVERY unresolved blocker gets exactly ONE action describing its resolution
- An explicit offer to help someone ELSE = action for the offerer:
  "I can set up those credentials this afternoon" → owner: that person, deadline: "this afternoon"
- An unresolved issue someone reported (e.g. delayed metrics) = action owned by the reporter
- A flagged risk to the sprint goal = action for "Team" ("Review sprint commitments and reassign tickets")
- A prerequisite that other work or a flagged risk depends on ("need to notify
  consumers before deprecating") = BOTH an in_progress item AND an action with owner
NO action for:
- Someone continuing or finishing their OWN listed work ("almost done, should finish today")
- A PR or review that nobody said they are blocked on ("PR is up for review" alone)

Owner heuristics when nobody claims the fix:
- staging / CI / build / infrastructure down → "DevOps"
- credentials / access / permissions → "IT" (or "Admin" for internal systems)
- pending architecture or team decision → "Tech Lead"
- production database emergency → "DBA Team"
- unresponsive external vendor → the affected person owns the escalation
- code review needed → the reviewer (e.g. "Senior Engineer")
- bug blocking someone → whoever built that component, if identifiable

WORDING — summaries must be scannable:
- done/in_progress items: "Name: task", task ≤ 8 words, reusing the transcript's key words
- action task: short imperative ≤ 6 words naming only the fix, built from the
  transcript's own key words — NEVER append the reason, and avoid generic verbs
  like "resolve/address the issue" when the transcript names the fix:
  "staging is down" → "Restore staging environment"
  "waiting for cloud credentials from IT" → "Provision cloud credentials"
  "vendor support unresponsive for days" → "Escalate [vendor] support ticket"
  "team must decide between X and Y" → "Make X vs Y decision"
  "we need a proper fix" → "Implement proper disk space fix"
  (BAD: "Provide cloud credentials to unblock infrastructure setup" — reason appended)
  (BAD: "Resolve database issue" — generic verb, drops the transcript's words)
- Name the artifact, not a possessive: "Review notification system design",
  NOT "Review Vera's design"
- Split compound sentences: "finished X and now working on Y" → X is DONE, Y is IN PROGRESS
- One item per task: "completed unit tests and integrated with Stripe" → TWO separate DONE items
- But split only DISTINCT tasks — keep a task and its outcome together:
  "updated HPA configuration — pods now auto-scale correctly" is ONE item
- Deadline: the stated timeframe; "ASAP" when a blocker stops work; null when none given
- If a person mentions nothing done/in-progress, skip them for that category

First, walk through the transcript speaker by speaker in brief notes (no curly
braces in the notes). Then verify against this checklist:
1. Every clause classified? Compound sentences split?
2. Every blocked or firefighting person's own task captured in in_progress?
3. Every unresolved blocker has exactly one action with an owner?
4. No action invented for someone's own ongoing work?
5. Every item within the word limits above?

Transcript:
{transcript}

Then output the final JSON inside a ```json fence as the LAST part of your response:
{{"done": ["Alice: ...", "Bob: ..."], "in_progress": ["Alice: ...", "Carol: ..."], "actions": [{{"task": "...", "owner": "name", "deadline": "timeframe or null"}}]}}

Use empty lists when a category has nothing."""

    raw = _call_llm(prompt)
    return _ensure_json(raw, {"done": [], "in_progress": [], "actions": []})


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
