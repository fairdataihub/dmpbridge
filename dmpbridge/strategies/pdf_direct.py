"""PDF-direct strategy — send raw PDF bytes to Claude in a single call.

Unlike the batch and wholedoc strategies, this strategy does **not** use
pdfplumber.  The raw PDF is base64-encoded and sent to Claude's document API;
Claude both extracts and classifies the content in one shot.

This produces paragraph-level blocks rather than line-level blocks, so the
block count is much lower (~27% of pdfplumber).  Bounding boxes are not
available.

Only the Anthropic provider is supported because this feature relies on
Claude's PDF vision capability.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.pdf_direct import PdfDirectStrategy

    strategy = PdfDirectStrategy(model="claude-opus-4-8")
    blocks   = strategy.run(Path("document.pdf"))
"""
from pathlib import Path

from .. import config
from ..exceptions import ConfigurationError
from ..logging_setup import get_logger
from ..pdf_direct import classify_pdf

logger = get_logger(__name__)


class PdfDirectStrategy:
    """Send the raw PDF to Claude; Claude extracts and classifies in one call.

    Parameters
    ----------
    model:
        Anthropic model that supports the PDF document API.
    api_key:
        Anthropic API key — falls back to ``config.ANTHROPIC_API_KEY``.
    """

    def __init__(
        self,
        model:   str = "claude-opus-4-8",
        api_key: str | None = None,
    ) -> None:
        self.model   = model
        self._api_key = api_key or config.ANTHROPIC_API_KEY
        if not self._api_key:
            raise ConfigurationError(
                "PdfDirectStrategy requires ANTHROPIC_API_KEY to be set."
            )

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Send *pdf_path* to Claude and return labeled paragraph-level blocks.

        1. Read the PDF as raw bytes and base64-encode it.
        2. Send to Claude's document API with the classification prompt.
        3. Parse and validate the returned JSON array.
        """
        logger.info("[pdf_direct] sending %s to %s …", pdf_path.name, self.model)
        blocks = classify_pdf(pdf_path, self.model, self._api_key)
        logger.info("[pdf_direct] %d blocks returned", len(blocks))
        return blocks
