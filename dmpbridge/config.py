"""
Central configuration for dmpbridge.
Edit this file to change the default model without touching any other code.
"""

# ── Ollama settings ───────────────────────────────────────────────────────────

# Model to use for classification.
# Any model installed in Ollama works — change this to switch models.
# Examples:
#   "llama3.2:latest"   — fast, 2 GB (recommended for testing)
#   "llama3.1:8b"       — good accuracy, 4.7 GB
#   "llama3:8b"         — older llama3, 4.7 GB
#   "llama3.3:8b"       — newer, pull with: ollama pull llama3.3:8b
#   "deepseek-r1:latest"— reasoning model, slower but thorough
#   "mistral:latest"    — alternative, 4 GB
MODEL = "llama3.3:70b"

# Ollama server URL. Change if running Ollama on another machine.
HOST = "http://localhost:11434"

# ── Batch settings ────────────────────────────────────────────────────────────

# How many blocks to send to the LLM per request.
# Reduce if the model returns empty responses; increase for faster throughput.
BATCH_SIZE = 10
