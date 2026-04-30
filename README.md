# Standup Summarizer Agent

An AI agent that automatically extracts and structures key information from daily standup meeting notes.

## Goal

Transform raw standup transcripts into organized, actionable summaries with:
- **Done** — what the team completed
- **In Progress** — current work
- **Blockers** — obstacles with owners and deadlines
- **Actions** — follow-ups with assignees

## Features

- **Single-agent loop** — thinks, acts, observes, repeats until done
- **Context-aware extraction** — understands implicit blockers and risks
- **Structured output** — JSON/Markdown, ready for Slack or Notion
- **Eval-driven** — 50 golden examples for quality assurance
- **Langfuse tracing** — full visibility into agent decisions

## Architecture

```
Input (standup transcript)
    ↓
Agent (LangGraph)
    ├─ Tool: extract_blockers()
    ├─ Tool: summarize_progress()
    └─ Tool: format_output()
    ↓
Output (structured JSON)
```

## Prerequisites

- Python 3.12+
- Anthropic API key (Claude Sonnet 4.6)
- (Optional) Langfuse account for tracing

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/kristiankocsis/agent-standup-summarizer.git
   cd agent-standup-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # ANTHROPIC_API_KEY=sk-ant-...
   # LANGFUSE_PUBLIC_KEY=... (optional)
   # LANGFUSE_SECRET_KEY=... (optional)
   ```

4. **Run the agent**
   ```bash
   python -m src.agent
   ```

## Example

**Input:**
```
Alice: Finished the API endpoint yesterday. Working on frontend now.
      Blocked by missing staging environment.
Bob: Wrapped up the database migration. Starting on tests today.
Carol: Still on the design review. Hope to be done by Friday.
```

**Output:**
```json
{
  "done": [
    "API endpoint",
    "Database migration"
  ],
  "in_progress": [
    "Frontend development",
    "Design review"
  ],
  "blockers": [
    {
      "description": "Missing staging environment",
      "owner": "Alice",
      "priority": "high"
    }
  ],
  "actions": [
    {
      "task": "Provision staging environment",
      "owner": "DevOps",
      "deadline": "EOD tomorrow"
    }
  ]
}
```

## Evaluation Results

| Metric | Target | Current |
|--------|--------|---------|
| Extraction accuracy | 90% | 0% |
| Golden dataset size | 50 | 0 |
| Blocker detection | 85% | — |
| False positives | <5% | — |

*Baseline will be established once golden dataset is created.*

## Development

### Project Structure
```
src/
  ├── agent.py       – Main agent definition (LangGraph)
  ├── tools.py       – Tool implementations
  └── config.py      – Configuration and constants
data/
  ├── golden_dataset/  – 50 labeled examples for eval
  └── traces/          – Langfuse trace logs
tests/
  └── test_agent.py  – Unit and integration tests
```

### Adding Tools

Edit `src/tools.py`:
```python
@tool
def my_tool(input: str) -> str:
    """Description for the model."""
    return result
```

Then add to agent in `src/agent.py`.

### Running Evals

```bash
python -m tests.eval_golden_dataset
```

## What I Learned

- **Context engineering** — how to structure tool descriptions and system prompts for better model reasoning
- **Tool design** — error messages must be actionable, not just descriptive
- **Evals as discipline** — golden datasets are the single highest-leverage habit
- **LangGraph abstractions** — state management, checkpointing, retry logic

## Links

- [Blog post: Building Your First Agent](https://example.com) — coming soon
- [LinkedIn](https://linkedin.com/in/kristian-kocsis/) — follow for updates
- [Langfuse traces](https://app.langfuse.com/) — production visibility

## License

MIT

---

**Status:** Early development (v0.1)  
**Last updated:** April 2026
