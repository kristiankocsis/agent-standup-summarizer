"""
Standup Summarizer Agent - LangGraph implementation.

Real single-agent loop: Think → Act → Observe → Repeat

Graph structure:
  START → call_llm → [tool_use?] → call_tools → call_llm → ... → END
"""

import json
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from anthropic import Anthropic

from .config import (
    ANTHROPIC_API_KEY, MODEL, SYSTEM_PROMPT, VERBOSE, MAX_ITERATIONS,
    LANGFUSE_ENABLED,
)
from .tools import TOOLS


# ── Langfuse (optional) ────────────────────────────────────────────────────
try:
    from langfuse import observe as _lf_observe, get_client as _lf_get_client
    _LANGFUSE_OK = LANGFUSE_ENABLED and _lf_get_client().auth_check()
except Exception:
    _LANGFUSE_OK = False

def _observe(name: str = None, as_type: str = None):
    """Applies @observe when Langfuse is available, else returns the function unchanged."""
    def decorator(func):
        if not _LANGFUSE_OK:
            return func
        kwargs = {}
        if name:
            kwargs["name"] = name
        if as_type:
            kwargs["as_type"] = as_type
        return _lf_observe(**kwargs)(func)
    return decorator


# ── State ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # accumulates; each node appends
    iterations: int                           # replaced each call_llm step
    structured_output: dict | None            # populated when format_output is called


# ── Tool schemas (Anthropic format) ────────────────────────────────────────

def _to_anthropic_schema(tool) -> dict:
    """Convert a LangChain @tool to Anthropic tool schema dict."""
    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        if hasattr(tool.args_schema, "model_json_schema"):
            schema = tool.args_schema.model_json_schema()   # Pydantic v2
        else:
            schema = tool.args_schema.schema()              # Pydantic v1
    else:
        schema = {}

    props = schema.get("properties", {})
    required = schema.get("required", [])

    clean_props = {}
    for k, v in props.items():
        entry: dict = {"type": v.get("type", "string")}
        if "description" in v:
            entry["description"] = v["description"]
        # arrays need items field
        if v.get("type") == "array":
            entry["items"] = {}
        clean_props[k] = entry

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": clean_props,
            "required": required,
        },
    }


TOOL_SCHEMAS = [_to_anthropic_schema(t) for t in TOOLS]
TOOL_MAP = {t.name: t for t in TOOLS}

_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Nodes ──────────────────────────────────────────────────────────────────

@_observe(name="call_llm")
def call_llm(state: AgentState) -> dict:
    """Think: call the LLM with full message history and available tools."""
    if VERBOSE:
        print(f"\n[think] iteration {state['iterations'] + 1}")

    response = _client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=state["messages"],
    )

    if _LANGFUSE_OK:
        tool_calls = [b.name for b in response.content if b.type == "tool_use"]
        _lf_get_client().update_current_generation(
            model=MODEL,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
            metadata={"stop_reason": response.stop_reason, "tool_calls": tool_calls},
        )

    if VERBOSE:
        tool_calls = [b.name for b in response.content if b.type == "tool_use"]
        print(f"  stop_reason={response.stop_reason}" +
              (f"  tools={tool_calls}" if tool_calls else ""))

    return {
        "messages": [{"role": "assistant", "content": response.content}],
        "iterations": state["iterations"] + 1,
    }


_progress_cb = None  # set by run_agent, consumed by call_tools


def call_tools(state: AgentState) -> dict:
    """Act + Observe: execute every tool_use block from the last assistant message."""
    last_content = state["messages"][-1]["content"]

    results = []
    for block in last_content:
        if block.type != "tool_use":
            continue

        if _progress_cb:
            _progress_cb(block.name)

        if VERBOSE:
            print(f"  [act]     {block.name}({list(block.input.keys())})")

        try:
            output = TOOL_MAP[block.name].invoke(block.input)
        except Exception as e:
            output = json.dumps({"error": str(e)})

        if VERBOSE:
            preview = str(output)[:120]
            print(f"  [observe] {block.name} -> {preview}")

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })

    # Capture structured output when format_output is called
    state_update: dict = {"messages": [{"role": "user", "content": results}]}
    for block in last_content:
        if block.type == "tool_use" and block.name == "format_output":
            try:
                parsed = json.loads(results[-1]["content"])
                if "standup_summary" in parsed:
                    state_update["structured_output"] = parsed["standup_summary"]
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    return state_update


# ── Routing ────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """After call_llm: route to call_tools if there are tool_use blocks, else END."""
    if state["iterations"] >= MAX_ITERATIONS:
        return END

    last_content = state["messages"][-1]["content"]
    if isinstance(last_content, list):
        for block in last_content:
            if hasattr(block, "type") and block.type == "tool_use":
                return "call_tools"

    return END


# ── Graph ──────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("call_llm", call_llm)
    g.add_node("call_tools", call_tools)
    g.add_edge(START, "call_llm")
    g.add_conditional_edges(
        "call_llm",
        should_continue,
        {"call_tools": "call_tools", END: END},
    )
    g.add_edge("call_tools", "call_llm")
    return g.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────

@_observe(name="standup-summarizer")
def run_agent(transcript: str, progress_callback=None) -> dict:
    """Run the agent on a standup transcript and return structured output."""
    global _progress_cb
    _progress_cb = progress_callback

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"STANDUP SUMMARIZER AGENT  |  model: {MODEL}")
        print(f"{'='*60}")
        print(f"Input: {len(transcript)} chars")

    if _LANGFUSE_OK:
        _lf_get_client().set_current_trace_io(input=transcript)

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": f"Analyze this standup transcript:\n\n{transcript}"}],
        "iterations": 0,
        "structured_output": None,
    }

    try:
        final_state = _graph.invoke(initial_state)

        # Extract text from the last assistant message
        last_content = final_state["messages"][-1]["content"]
        output_text = ""
        if isinstance(last_content, list):
            for block in last_content:
                if hasattr(block, "text"):
                    output_text = block.text
                    break
        elif isinstance(last_content, str):
            output_text = last_content

        result = {
            "status": "success",
            "output": output_text,
            "structured_output": final_state.get("structured_output"),
            "iterations": final_state["iterations"],
            "transcript_length": len(transcript),
        }

        if _LANGFUSE_OK:
            _lf_get_client().set_current_trace_io(output=output_text)

        if VERBOSE:
            print(f"\n[done] {final_state['iterations']} iteration(s)")
            if output_text:
                safe = output_text.encode("ascii", errors="replace").decode("ascii")
                print(f"\nOutput:\n{safe}")

        return result

    except Exception as e:
        if _LANGFUSE_OK:
            _lf_get_client().update_current_span(metadata={"error": str(e)})
        if VERBOSE:
            print(f"\n[error] {e}")
        return {
            "status": "error",
            "error": str(e),
            "transcript_length": len(transcript),
        }


if __name__ == "__main__":
    sample = """
    Alice: Yesterday I finished the user authentication API endpoint.
    Today I'm starting on the frontend forms. But I'm blocked by the staging
    environment still being down.

    Bob: I wrapped up the database migration. Tests are passing. Starting on
    performance optimization next.

    Carol: Design review is still ongoing. Looks like another day or two.
    Nothing blocking me yet.
    """

    result = run_agent(sample)
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2, default=str))
