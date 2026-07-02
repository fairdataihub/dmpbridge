"""Custom exception hierarchy for dmpbridge."""


class DmpBridgeError(Exception):
    """Base class for all dmpbridge errors."""


class ConfigurationError(DmpBridgeError):
    """Missing or invalid configuration: API keys, unsupported providers, etc."""


class ProviderConnectionError(DmpBridgeError):
    """Cannot reach the LLM provider (Ollama not running, API unreachable, etc.)."""


class ExtractionError(DmpBridgeError):
    """PDF extraction or page-rendering failed."""
