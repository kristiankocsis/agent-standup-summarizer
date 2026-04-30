"""
Standup Summarizer Agent - LangGraph implementation.

Single-agent loop: Think → Act → Observe → Repeat
"""

from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from anthropic import Anthropic

from .config import ANTHROPIC_API_KEY, MODEL, VERBOSE
from .tools import TOOLS


def create_agent():
    """Create Anthropic client for the agent."""
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def run_agent(transcript: str) -> dict:
    """
    Run the agent on a standup transcript using Anthropic API directly.

    Args:
        transcript: Raw standup meeting transcript

    Returns:
        dict with agent output and metadata
    """

    client = create_agent()

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"STANDUP SUMMARIZER AGENT")
        print(f"{'='*60}")
        print(f"Input length: {len(transcript)} characters")
        print(f"Model: {MODEL}")
        print(f"{'='*60}\n")

    # Prepare the prompt
    system_prompt = """You are a Standup Meeting Summarizer Agent. Your goal is to extract and structure information from standup transcripts.

Process the standup transcript and provide a structured summary with:
1. Done - what the team completed
2. In Progress - current work
3. Blockers - obstacles with owners
4. Actions - follow-ups with assignees

Output as clean JSON."""

    user_prompt = f"""Analyze this standup transcript and provide a structured summary.

Transcript:
---
{transcript}
---

Respond with ONLY a valid JSON object (no markdown, no code blocks) with this structure:
{{
    "done": ["item1", "item2"],
    "in_progress": ["item1", "item2"],
    "blockers": [
        {{"description": "blocker", "owner": "name", "severity": "high|medium|low"}}
    ],
    "actions": [
        {{"task": "action", "owner": "name", "deadline": "when"}}
    ]
}}"""

    try:
        # Call Anthropic API
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        output = response.content[0].text

        if VERBOSE:
            print("\n[OK] Agent completed successfully")
            print(f"\nAgent output:\n{output}")

        return {
            "status": "success",
            "output": output,
            "transcript_length": len(transcript),
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        }

    except Exception as e:
        if VERBOSE:
            print(f"\n[ERROR] Agent failed: {str(e)}")

        return {
            "status": "error",
            "error": str(e),
            "transcript_length": len(transcript),
        }


if __name__ == "__main__":
    # Test example
    sample_transcript = """
    Alice: Hi team. Yesterday I finished the user authentication API endpoint.
    Today I'm starting on the frontend forms. But I'm blocked by the staging environment
    still being down.

    Bob: I wrapped up the database migration. Tests are passing. Starting on the
    performance optimization next.

    Carol: Design review is still ongoing. Looks like another day or two. Nothing blocking me yet.
    """

    result = run_agent(sample_transcript)
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    import json
    print(json.dumps(result, indent=2, default=str))
