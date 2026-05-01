"""
Standup Summarizer — Streamlit UI
"""

import time
import streamlit as st

from src.agent import run_agent

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Standup Summarizer",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Standup Summarizer")
    st.caption("AI agent powered by Claude + LangGraph")
    st.divider()
    st.markdown("""
**How it works:**
1. Paste your standup transcript
2. Click **Summarize**
3. Agent runs 3 tools:
   - `extract_blockers`
   - `summarize_progress`
   - `format_output`
4. Get a structured summary
""")
    st.divider()
    st.markdown("Built with [Claude](https://anthropic.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [Langfuse](https://langfuse.com)")

# ── Main area ──────────────────────────────────────────────────────────────

st.header("Daily Standup Summarizer")

PLACEHOLDER = """\
Alice: Finished the user authentication API endpoint yesterday. Today I'm starting frontend forms. I'm blocked by the staging environment still being down.

Bob: Wrapped up the database migration. Tests are passing. Starting performance optimization next.

Carol: Design review is still ongoing. Looks like another day or two. Nothing blocking me yet."""

transcript = st.text_area(
    "Paste standup transcript here:",
    value="",
    placeholder=PLACEHOLDER,
    height=220,
)

run_btn = st.button("Summarize", type="primary", disabled=not transcript.strip())

# ── Run agent ──────────────────────────────────────────────────────────────

if run_btn and transcript.strip():
    with st.spinner("Agent is thinking..."):
        t0 = time.time()
        result = run_agent(transcript)
        elapsed = time.time() - t0

    if result["status"] == "error":
        st.error(f"Agent error: {result.get('error')}")
        st.stop()

    summary = result.get("structured_output")

    if summary is None:
        st.warning("Agent finished but did not produce structured output. Raw output below:")
        st.text(result.get("output", ""))
        st.stop()

    # ── Results ────────────────────────────────────────────────────────────

    st.success(f"Done in {elapsed:.1f}s · {result['iterations']} agent iterations")

    col1, col2 = st.columns(2)

    with col1:
        done = summary.get("done", [])
        st.subheader(f"✅ Done ({len(done)})")
        if done:
            for item in done:
                st.markdown(f"- {item}")
        else:
            st.caption("Nothing completed yet")

    with col2:
        in_progress = summary.get("in_progress", [])
        st.subheader(f"🔄 In Progress ({len(in_progress)})")
        if in_progress:
            for item in in_progress:
                st.markdown(f"- {item}")
        else:
            st.caption("Nothing in progress")

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        blockers = summary.get("blockers", [])
        st.subheader(f"🚧 Blockers ({len(blockers)})")
        if blockers:
            for b in blockers:
                severity = b.get("severity", "medium")
                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                owner = b.get("owner", "?")
                desc = b.get("description", str(b))
                st.markdown(f"{icon} **{owner}:** {desc}")
        else:
            st.caption("No blockers — great standup!")

    with col4:
        actions = summary.get("actions", [])
        st.subheader(f"📋 Actions ({len(actions)})")
        if actions:
            for a in actions:
                task = a.get("task", str(a))
                owner = a.get("owner", "?")
                deadline = a.get("deadline")
                deadline_str = f" _(by {deadline})_" if deadline and deadline != "null" else ""
                st.markdown(f"- **{owner}:** {task}{deadline_str}")
        else:
            st.caption("No action items")

    # ── Raw JSON expander ──────────────────────────────────────────────────
    with st.expander("Raw JSON output"):
        st.json(summary)
