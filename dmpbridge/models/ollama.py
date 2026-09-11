"""Ollama model backend."""
import requests

from ..utils import ProviderConnectionError, get_logger

logger = get_logger(__name__)


class OllamaModel:
    """Call a locally running Ollama server.

    Every call must pass a ``schema`` — Ollama grammar-enforces the response
    against it, so there is no need for separate JSON repair on the caller
    side.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.3:70b"``.
    host:
        Base URL of the Ollama server.
    num_ctx:
        Context window size passed to Ollama.  32 768 comfortably fits a full
        DMP document for whole-doc inference.
    num_predict:
        Cap on generated tokens.  ``None`` (the default) is Ollama's own
        default: no limit at all — with context shifting, generation can run
        indefinitely.  Set it for prompts where the model may not stop on its
        own: a full schema in JSON mode once fell into a repetition loop
        (the same contributor block, over and over) for ~40 minutes.
    """

    def __init__(
        self,
        model:       str,
        host:        str,
        num_ctx:     int = 32768,
        num_predict: int | None = None,
    ) -> None:
        self.model       = model
        self.host        = host.rstrip("/")
        self.num_ctx     = num_ctx
        self.num_predict = num_predict
        self._verify_connection()

    def complete(self, system: str, prompt: str, *, schema: dict) -> str:
        """Send *system* + *prompt* to Ollama and return the raw text response.

        ``temperature`` stays pinned at ``0.0`` regardless of caller — this is
        what makes runs reproducible.
        """
        options = {"temperature": 0.0, "num_ctx": self.num_ctx}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model":      self.model,
                "system":     system,
                "prompt":     prompt,
                "stream":     False,
                "format":     schema,
                "keep_alive": -1,   # keep model in VRAM indefinitely
                "options":    options,
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
