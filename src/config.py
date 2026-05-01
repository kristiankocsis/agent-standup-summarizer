import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")

# Langfuse Configuration (optional)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = LANGFUSE_PUBLIC_KEY is not None

# Agent Configuration
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# System prompt for the agent
SYSTEM_PROMPT = """You are a Standup Meeting Summarizer Agent. Given a standup transcript, produce a complete structured summary covering Done, In Progress, Blockers, and Actions.

Always follow this exact three-step sequence:

STEP 1 — extract_blockers(transcript)
Call with the full transcript.
Returns JSON: {"blockers": [{description, owner, severity}], "risks": [{description, owner}]}

STEP 2 — summarize_progress(transcript)
Call with the full transcript.
Returns JSON: {"done": [...], "in_progress": [...], "actions": [{task, owner, deadline}]}

STEP 3 — format_output(done, in_progress, blockers, actions)
Combine results from steps 1 and 2:
  done       = the "done" list from step 2
  in_progress = the "in_progress" list from step 2
  blockers   = the "blockers" list from step 1
  actions    = the "actions" list from step 2
Call format_output with these four lists. Use empty list [] for any category with no items.

After format_output completes, present the summary clearly to the user.
Do not skip any step. Do not call the same tool twice.
"""

# Validation
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Please configure it in .env file.")

if VERBOSE:
    print(f"[OK] Configuration loaded")
    print(f"  Model: {MODEL}")
    print(f"  Langfuse: {'enabled' if LANGFUSE_ENABLED else 'disabled'}")
