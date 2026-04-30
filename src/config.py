import os
from dotenv import load_dotenv

load_dotenv()

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
SYSTEM_PROMPT = """You are a Standup Meeting Summarizer Agent. Your goal is to extract and structure information from standup transcripts.

You have access to the following tools:
- extract_blockers: Find obstacles and risks mentioned in the transcript
- summarize_progress: Create concise summaries of completed and ongoing work
- format_output: Structure the information as JSON

Process the standup transcript step by step:
1. First, extract all mentioned blockers and risks
2. Then summarize what's done and in progress
3. Finally, format everything as a structured JSON output

Be thorough but concise. Identify implicit blockers (e.g., "waiting for design review" = blocker).
"""

# Validation
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Please configure it in .env file.")

if VERBOSE:
    print(f"✓ Configuration loaded")
    print(f"  Model: {MODEL}")
    print(f"  Langfuse: {'enabled' if LANGFUSE_ENABLED else 'disabled'}")
