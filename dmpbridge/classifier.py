"""Batch classifier — sliding-window batching over a ModelBackend.

:class:`BaseClassifier` owns the batching and context-window logic.
The actual model call is delegated to a :class:`~dmpbridge.models.ModelBackend`,
so adding a new provider requires only a new backend in ``dmpbridge/models/``,
not a new classifier subclass.

Use :func:`get_classifier` to get a ready-to-use instance.
"""
from . import config
from .models import ModelBackend, get_model
from .parsers import parse_llm_json
from .prompt import LABELS, SYSTEM_PROMPT, build_batch_prompt
from .logging_setup import get_logger

logger = get_logger(__name__)

BATCH_SIZE   = config.BATCH_SIZE
CONTEXT_SIZE = 3


class BaseClassifier:
    """Sliding-window batch classification over any :class:`ModelBackend`.

    Parameters
    ----------
    model_backend:
        A configured backend that implements ``complete(system, prompt) -> str``.
    batch_size:
        Number of blocks per LLM request.
    context_size:
        Number of already-labeled blocks prepended as sliding context.
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        batch_size:    int = BATCH_SIZE,
        context_size:  int = CONTEXT_SIZE,
    ) -> None:
        self._model      = model_backend
        self.batch_size  = batch_size
        self.context_size = context_size

    def classify_blocks(self, blocks: list[dict]) -> list[dict]:
        """Classify all blocks in document order with a sliding context window.

        1. Copy the original blocks.
        2. Process in batches of ``batch_size``.
        3. Prepend the last ``context_size`` labeled blocks as context.
        4. Call the model and merge predicted labels back into the result list.
        """
        result = [dict(b) for b in blocks]

        for start in range(0, len(result), self.batch_size):
            batch   = result[start : start + self.batch_size]
            context = result[max(0, start - self.context_size) : start]

            logger.info("  Classifying blocks %d–%d …", start, start + len(batch) - 1)

            prompt  = build_batch_prompt(batch, start, context)
            raw     = self._model.complete(SYSTEM_PROMPT, prompt)
            labels  = parse_llm_json(raw, label=f"offset={start}")

            for entry in labels:
                idx = entry.get("id")
                lbl = entry.get("label", "answer.text")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result


def get_classifier(
    provider:     str | None = None,
    model:        str | None = None,
    host:         str | None = None,
    batch_size:   int = BATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
) -> BaseClassifier:
    """Return a :class:`BaseClassifier` wired to the right model backend.

    Falls back to ``config`` defaults when arguments are omitted.
    """
    backend = get_model(
        provider or config.PROVIDER,
        model    or config.MODEL,
        host=host,
    )
    return BaseClassifier(backend, batch_size=batch_size, context_size=context_size)
