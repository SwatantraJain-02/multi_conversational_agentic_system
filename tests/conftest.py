"""Unit tests never talk to Langfuse Cloud: the real keys in .env must not ship test traces."""

import os

# Set before any project module reads settings (environment beats .env).
os.environ["TRACING_ENABLED"] = "false"
