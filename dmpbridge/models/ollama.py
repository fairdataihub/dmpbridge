"""Ollama model backend."""
import requests

from ..prompts import OUTPUT_SCHEMA
from ..utils import ProviderConnectionError, get_logger

logger = get_logger(__name__)


class OllamaModel:
    """Call a locally running Ollama server.

    Uses ``format=OUTPUT_SCHEMA`` on every request so Ollama grammar-enforces
    valid JSON output — no need for separate JSON repair on the caller side.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.3:70b"``.
    host:
        Base URL of the Ollama server.
    num_ctx:
        Context window size passed to Ollama.  32 768 comfortably fits a full
        DMP document for whole-doc inference.
    """

    def __init__(
        self,
        model:   str,
        host:    str,
        num_ctx: int = 32768,
    ) -> None:
        self.model   = model
        self.host    = host.rstrip("/")
        self.num_ctx = num_ctx
        self._verify_connection()

    def complete(self, system: str, prompt: str) -> str:
        """Send *system* + *prompt* to Ollama and return the raw text response."""
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model":      self.model,
                "system":     system,
                "prompt":     prompt,
                "stream":     False,
                "format":     OUTPUT_SCHEMA,
                "keep_alive": -1,   # keep model in VRAM indefinitely
                "options": {
                    "temperature": 0.0,
                    "num_ctx":     self.num_ctx,
                    "num_gpu":     -1,
                },
            },
            timeout=3600,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _verify_connection(self) -> None:
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ProviderConnectionError(
                f"Ollama is not reachable at {self.host}.\n"
                "Install and start it: https://ollama.com\n"
                f"Then pull the model:  ollama pull {self.model}\n"
                f"Details: {exc}"
            ) from exc
