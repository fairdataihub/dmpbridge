"""
Central configuration for dmpbridge.
Edit this file, set environment variables, or create a .env file to change settings.
"""

import os
from pathlib import Path

# Load .env if present — provider settings live there.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; export variables in your shell instead

# ── Provider ───────────────────────────────────────────────────────────────────

# LLM backend — only "ollama" is supported.
# Override at runtime:  DMPBRIDGE_PROVIDER=ollama dmpbridge document.pdf
PROVIDER = os.getenv("DMPBRIDGE_PROVIDER", "ollama")

# Model name served by Ollama.
# Examples: "llama3.3:70b"  "llama3.1:8b"  "qwen2-vl:7b"
MODEL = os.getenv("DMPBRIDGE_MODEL", "llama3.3:70b")

# Ollama server URL.
HOST = os.getenv("DMPBRIDGE_HOST", "http://localhost:11434")

# ── Batch settings ────────────────────────────────────────────────────────────

# How many blocks to send per LLM request.
# Reduce if the model returns empty responses; increase for faster throughput.
BATCH_SIZE = int(os.getenv("DMPBRIDGE_BATCH_SIZE", "10"))
