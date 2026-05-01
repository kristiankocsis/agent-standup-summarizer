"""
Standup Summarizer — Streamlit UI
"""

import time
from datetime import datetime
import streamlit as st

from src.agent import run_agent


def _to_markdown(summary: dict, timestamp: str) -> str:
    lines = [f"## Daily Standup Summary", f"*{timestamp}*", ""]

    done = summary.get("done", [])
    lines.append(f"### ✅ Done ({len(done)})")
    lines += [f"- {item}" for item in done] if done else ["- *(nothing completed)*"]
    lines.append("")

    ip = summary.get("in_progress", [])
    lines.append(f"### 🔄 In Progress ({len(ip)})")
    lines += [f"- {item}" for item in ip] if ip else ["- *(nothing in progress)*"]
    lines.append("")

    blockers = summary.get("blockers", [])
    lines.append(f"### 🚧 Blockers ({len(blockers)})")
    if blockers:
        for b in blockers:
            sev = b.get("severity", "medium")
            icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "🟢"
            lines.append(f"- {icon} **{b.get('owner', '?')}:** {b.get('description', '')}")
    else:
        lines.append("- *(no blockers)*")
    lines.append("")

    actions = summary.get("actions", [])
    lines.append(f"### 📋 Actions ({len(actions)})")
    if actions:
        for a in actions:
            deadline = a.get("deadline")
            dl = f" *(by {deadline})*" if deadline and str(deadline) != "null" else ""
            lines.append(f"- **{a.get('owner', '?')}:** {a.get('task', '')}{dl}")
    else:
        lines.append("- *(no action items)*")

    return "\n".join(lines)

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Standup Summarizer",
    page_icon="🤖",
    layout="wide",
)

# ── Example transcripts ────────────────────────────────────────────────────

EXAMPLES = {
    "🚧 Basic blocker": (
        "Alice: Finished the user authentication API endpoint yesterday. "
        "Today I'm starting frontend forms. I'm blocked by the staging environment still being down.\n\n"
        "Bob: Wrapped up the database migration. Tests are passing. Starting performance optimization next.\n\n"
        "Carol: Design review is still ongoing. Looks like another day or two. Nothing blocking me yet."
    ),
    "🔥 Hotfix crisis": (
        "Beth: Production is down. Database ran out of disk space at 3am. "
        "I cleared old logs to buy us about 30 minutes.\n\n"
        "Chris: I paged the DBA team. Now drafting the incident report."
    ),
    "✅ Clean standup": (
        "Steve: Finished the login page styling yesterday. Today working on the profile settings page. "
        "No blockers.\n\n"
        "Tanya: Reviewed and approved Steve's login PR. Now working on the user preferences feature. "
        "All good."
    ),
    "⚠️ Sprint at risk": (
        "Oliver: We're 3 days from end of sprint. Only 8 of 21 story points done. "
        "I'm behind — picked up a production bug mid-sprint.\n\n"
        "Paula: I'm ahead of schedule on my tickets. Happy to take something off Oliver's plate "
        "if the Scrum Master wants to rebalance."
    ),
}

TOOL_LABELS = {
    "extract_blockers":  "🔍 Extracting blockers...",
    "summarize_progress": "📋 Summarizing progress...",
    "format_output":     "📦 Formatting output...",
}

# ── Session state init ─────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

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

# ── Example buttons ────────────────────────────────────────────────────────

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["transcript_input"] = text
            st.rerun()

# ── Transcript input ───────────────────────────────────────────────────────

transcript = st.text_area(
    "Paste standup transcript here:",
    placeholder="Alice: Yesterday I finished...",
    height=200,
    key="transcript_input",
)

run_btn = st.button("Summarize", type="primary", disabled=not (transcript or "").strip())

# ── Run agent ──────────────────────────────────────────────────────────────

if run_btn and (transcript or "").strip():
    t0 = time.time()

    with st.status("Agent is thinking...", expanded=True) as status:
        status.write("🧠 Sending transcript to agent...")

        def on_tool(tool_name: str):
            status.write(TOOL_LABELS.get(tool_name, f"⚙️ Running `{tool_name}`..."))

        result = run_agent(transcript, progress_callback=on_tool)
        elapsed = time.time() - t0

        if result["status"] == "error":
            status.update(label="Agent failed", state="error")
            st.error(f"Error: {result.get('error')}")
            st.stop()

        status.update(
            label=f"Done in {elapsed:.1f}s · {result['iterations']} iterations",
            state="complete",
            expanded=False,
        )

    summary = result.get("structured_output")

    if summary is None:
        st.warning("Agent finished but produced no structured output.")
        st.text(result.get("output", ""))
        st.stop()

    # ── Compute a rough quality score for history ──────────────────────────
    def _has(lst): return 1.0 if lst else 0.7
    scores = {
        "done":        _has(summary.get("done")),
        "in_progress": _has(summary.get("in_progress")),
        "blockers":    1.0,
        "actions":     1.0,
    }
    avg_score = sum(scores.values()) / len(scores)

    st.session_state.history.append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "transcript": transcript,
        "scores":     scores,
        "score":      avg_score,
        "iterations": result["iterations"],
        "elapsed":    elapsed,
    })

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
                deadline_str = f" _(by {deadline})_" if deadline and str(deadline) != "null" else ""
                st.markdown(f"- **{owner}:** {task}{deadline_str}")
        else:
            st.caption("No action items")

    with st.expander("Raw JSON output"):
        st.json(summary)

    with st.expander("📤 Export as Markdown"):
        md = _to_markdown(summary, datetime.now().strftime("%Y-%m-%d %H:%M"))
        st.code(md, language="markdown")
        st.download_button(
            label="⬇️ Download .md file",
            data=md,
            file_name=f"standup_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

# ── Sidebar history (rendered after history.append so current run is visible) ──

with st.sidebar:
    if st.session_state.history:
        st.divider()
        st.subheader("🕒 Recent runs")
        for i, run in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"{run['time']}  ·  {run['score']:.0%}", expanded=False):
                s = run["scores"]
                st.caption(
                    f"✅ done {s['done']:.0%}  "
                    f"🔄 prog {s['in_progress']:.0%}  "
                    f"🚧 blk {s['blockers']:.0%}  "
                    f"📋 act {s['actions']:.0%}"
                )
                st.caption(f"{run['iterations']} iterations · {run['elapsed']:.1f}s")
                if st.button("Restore transcript", key=f"restore_{i}"):
                    st.session_state["transcript_input"] = run["transcript"]
                    st.rerun()
