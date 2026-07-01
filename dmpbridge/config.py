"""
Central configuration for dmpbridge.
Edit this file, set environment variables, or create a .env file to change settings.
"""

import os
from pathlib import Path

# Load .env if present — API keys and provider settings live there.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; export variables in your shell instead

# ── Provider ───────────────────────────────────────────────────────────────────

# Which LLM backend to use.  Options: "ollama" | "openai" | "anthropic" | "gemini"
# Override at runtime:  DMPBRIDGE_PROVIDER=openai dmpbridge document.pdf
PROVIDER = os.getenv("DMPBRIDGE_PROVIDER", "ollama")

# Model name for the selected provider.
# Ollama:    "llama3.3:70b"  "llama3.1:8b"  "mistral:latest"
# OpenAI:    "gpt-4o"  "gpt-4o-mini"
# Anthropic: "claude-sonnet-4-6"  "claude-haiku-4-5-20251001"
# Gemini:    "gemini-2.0-flash"  "gemini-1.5-pro"
MODEL = os.getenv("DMPBRIDGE_MODEL", "llama3.3:70b")

# Ollama server URL — only used when PROVIDER = "ollama".
HOST = os.getenv("DMPBRIDGE_HOST", "http://localhost:11434")

# ── Batch settings ────────────────────────────────────────────────────────────

# How many blocks to send per LLM request.
# Reduce if the model returns empty responses; increase for faster throughput.
BATCH_SIZE = int(os.getenv("DMPBRIDGE_BATCH_SIZE", "10"))

# ── API keys ──────────────────────────────────────────────────────────────────
# Set these in your .env file — never hardcode keys here.

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
