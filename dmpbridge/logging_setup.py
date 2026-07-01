"""Shared logging configuration for dmpbridge.

Usage in modules:
    from .logging_setup import get_logger
    logger = get_logger(__name__)

Usage at script / CLI entry points (call once before any work):
    from dmpbridge.logging_setup import setup_logging
    setup_logging(verbose=True)
"""
import logging
import logging.handlers
from pathlib import Path

_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "dmpbridge.log"

# Third-party loggers that are noisy at INFO/DEBUG level.
_NOISY_LOGGERS = (
    "pdfminer",
    "pdfminer.pdfpage",
    "pdfminer.pdfdocument",
    "pdfminer.pdfinterp",
    "pdfminer.converter",
    "pdfminer.cmapdb",
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a dmpbridge module or script."""
    return logging.getLogger(name)


def setup_logging(*, verbose: bool = False) -> None:
    """Configure console and rotating-file handlers on the root logger.

    Subsequent calls are no-ops when handlers are already attached.
    Console output uses a clean single-line format; the log file includes
    timestamps and module names for post-run debugging.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    root  = logging.getLogger()

    if root.handlers:
        root.setLevel(min(root.level or logging.WARNING, level))
        return

    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.addHandler(console)
    root.addHandler(file_handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR)
