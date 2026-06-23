"""dmpbridge — PDF extraction and structure labeling pipeline."""

from .pipeline import process_pdf
from .converter import to_structured, convert_file

__all__ = ["process_pdf", "to_structured", "convert_file"]
__version__ = "0.1.0"
