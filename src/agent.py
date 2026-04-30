"""
Standup Summarizer Agent - LangGraph implementation.

Single-agent loop: Think → Act → Observe → Repeat
"""

from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, MODEL, SYSTEM_PROMPT, VERBOSE
from src.tools import TOOLS


def create_agent():
    """
    Create and return the Standup Summarizer agent.

    Uses LangGraph's ReAct pattern: Reasoning + Acting in a loop.
    """

    # Initialize Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Create the agent using LangGraph's prebuilt React agent
    # (This is a simplified version; full implementation would define custom state schema)
    agent = create_react_agent(
        model=client,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_agent(transcript: str) -> dict:
    """
    Run the agent on a standup transcript.

    Args:
        transcript: Raw standup meeting transcript

    Returns:
        dict with agent output and metadata
    """

    agent = create_agent()

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"STANDUP SUMMARIZER AGENT")
        print(f"{'='*60}")
        print(f"Input length: {len(transcript)} characters")
        print(f"Model: {MODEL}")
        print(f"{'='*60}\n")

    # Prepare the prompt
    prompt = f"""Analyze this standup transcript and provide a structured summary.

Transcript:
---
{transcript}
---

Use your tools to:
1. Extract all blockers and risks
2. Summarize what's done and in progress
3. Format the output as a clean summary

Provide the final output as a structured JSON."""

    try:
        # Invoke the agent
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        if VERBOSE:
            print("\n✓ Agent completed successfully")

        return {
            "status": "success",
            "output": result,
            "transcript_length": len(transcript),
        }

    except Exception as e:
        if VERBOSE:
            print(f"\n✗ Agent failed: {str(e)}")

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
    print("\nResult:")
    import json

    print(json.dumps(result, indent=2, default=str))
