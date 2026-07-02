"""LLM output parsers.

Sub-modules
-----------
json_parser
    Parse raw LLM text into a list of dicts, handling markdown fences,
    dict-wrapped arrays, and empty responses.
"""
from .json_parser import parse_llm_json

__all__ = ["parse_llm_json"]
