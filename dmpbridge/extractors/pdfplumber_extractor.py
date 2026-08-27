"""pdfplumber-backed extractor — whole-document text with visual-signal markers."""
from __future__ import annotations

import json
from pathlib import Path

from .base import BaseExtractor


def native_result_dict(pdf_path: Path, include_chars: bool = False) -> dict:
    """Everything pdfplumber reads from the PDF, before any rule is applied.

    The counterpart of Docling's ``native_result_dict``: the raw evidence the
    marker rules in :mod:`~dmpbridge.preprocess.pdfplumber_reader` work from,
    so a marker can be traced back to what produced it. Per page — every word
    with its font name, size and box; every drawn rectangle, line and curve
    (a thin rectangle under a word is how an underline is detected);
    hyperlink annotations with their URI; image count — plus the document's
    body-font profile that bold and size are judged against. Characters are
    the bulk of the data and are left out unless *include_chars* is set.
    """
    import pdfplumber

    from ..preprocess.pdfplumber_reader import get_body_font_profile

    def box(o):
        return {k: round(float(o[k]), 2) for k in ("x0", "x1", "top", "bottom")}

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        body_size, body_font = get_body_font_profile(pdf)
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["fontname", "size"])
            entry = {
                "page_no": page.page_number,
                "size": {"width": float(page.width), "height": float(page.height)},
                "words": [{"text": w["text"], "fontname": w["fontname"],
                           "size": round(float(w["size"]), 2), "upright": w["upright"], **box(w)}
                          for w in words],
                "rects": [{**box(r), "height": round(float(r["height"]), 2),
                           "stroke": r.get("stroke"), "fill": r.get("fill")}
                          for r in page.rects],
                "lines": [box(ln) for ln in page.lines],
                "curves": len(page.curves),
                "images": len(page.images),
                "hyperlinks": [{**box(h), "uri": h.get("uri")}
                               for h in getattr(page, "hyperlinks", [])],
            }
            if include_chars:
                entry["chars"] = [{"text": c["text"], "fontname": c["fontname"],
                                   "size": round(float(c["size"]), 2), **box(c)}
                                  for c in page.chars]
            pages.append(entry)

    return {
        "tool": "pdfplumber",
        "version": pdfplumber.__version__,
        "file": pdf_path.name,
        "body_font": {"size": body_size, "fontname": body_font},
        "pages": pages,
    }


class PdfplumberExtractor(BaseExtractor):
    """Wrap pdfplumber's font-baseline visual-signal extraction as a BaseExtractor.

    No additional dependencies — pdfplumber is already a core requirement.

    Does not segment the document into blocks at extraction time: it returns
    the whole document as a single text string (wrapped in a one-item list so
    the return shape still matches :class:`BaseExtractor`), with words
    visually emphasized relative to the document's own body-text baseline
    wrapped in ``**...**`` and italic words in ``_..._``.
    :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy` classifies this
    text and splits it into labeled entries in one model call — extraction
    and labeling are not separate steps.

    Parameters
    ----------
    save_native:
        Also write pdfplumber's raw reading of the PDF as
        ``1_extracted/pdfplumber/<stem>.native.json`` — see
        :func:`native_result_dict`. Off by default; does not change the
        extracted text.
    native_chars:
        Include every character in the native file (several MB per document).
        Only used when ``save_native`` is on.
    """

    name = "pdfplumber"

    def __init__(self, save_native: bool = False, native_chars: bool = False) -> None:
        self._save_native = save_native
        self._native_chars = native_chars

    def extract(self, pdf_path: Path) -> list[dict]:
        from ..preprocess import extract_text_for_llm
        if self._save_native:
            self._save_side_file(pdf_path, "native.json",
                                 json.dumps(native_result_dict(pdf_path, self._native_chars),
                                            indent=2, ensure_ascii=False))
        return [{"text": extract_text_for_llm(pdf_path)}]

    def _save_side_file(self, pdf_path: Path, suffix: str, content: str) -> None:
        """Write ``<stem>.<suffix>`` next to the cached stage-1 JSON. Best-effort."""
        try:
            from ..core.paths import EXTRACTED_DIR
            out = EXTRACTED_DIR / self.name / f"{pdf_path.stem}.{suffix}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
        except OSError:
            pass
