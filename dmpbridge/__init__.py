"""dmpbridge — PDF extraction and structure labeling pipeline."""
import logging

from .core import convert_file, process_pdf, to_structured
from .strategies.wholedoc import WholeDocStrategy

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["process_pdf", "to_structured", "convert_file", "WholeDocStrategy"]
__version__ = "0.1.0"
