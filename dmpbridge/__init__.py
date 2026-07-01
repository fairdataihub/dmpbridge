"""dmpbridge — PDF extraction and structure labeling pipeline."""
import logging

from .pipeline import process_pdf
from .converter import to_structured, convert_file

# Standard library best-practice: add a NullHandler so callers that don't call
# setup_logging() never see "No handlers could be found for logger 'dmpbridge'".
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["process_pdf", "to_structured", "convert_file"]
__version__ = "0.1.0"
