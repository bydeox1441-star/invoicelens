"""Route each PDF through text/OCR -> layout rules -> optional AI -> validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf

from .layouts import PARSERS, detect_layout
from .models import REQUIRED_FIELDS, Invoice, Issue, Result
from .text import digital_lines, is_scanned, ocr_lines
from .validate import LOW_OCR_CONFIDENCE, flag_duplicates, validate_invoice

AIExtractor = Callable[[Path], Invoice]


def process_file(path: Path, ai: AIExtractor | None = None) -> Result:
    path = Path(path)
    with pymupdf.open(path) as doc:
        scanned = is_scanned(doc)
        lines, confidence = ocr_lines(doc) if scanned else (digital_lines(doc), None)

    layout = detect_layout(lines)
    invoice, method = Invoice(), "none"
    if layout in PARSERS:
        invoice = PARSERS[layout](lines)
        method = "ocr+rules" if scanned else "rules"

    extra: list[Issue] = []
    rules_incomplete = any(getattr(invoice, f) in (None, "") for f in REQUIRED_FIELDS)
    if ai is not None and (layout == "unknown" or rules_incomplete):
        try:
            invoice, method = ai(path), "ai"
        except Exception as e:  # noqa: BLE001 - any AI failure becomes a review reason
            extra.append(Issue("AI_ERROR", f"AI step failed: {e}"))

    if layout == "unknown" and method == "none":
        extra.append(Issue("UNKNOWN_LAYOUT", "layout not recognised and AI step is off"))
    if confidence is not None and confidence < LOW_OCR_CONFIDENCE:
        extra.append(Issue("LOW_OCR_CONFIDENCE", f"low OCR confidence ({confidence:.0f}/100)"))

    issues = extra + (validate_invoice(invoice) if method != "none" else [])
    return Result(file=path.name, invoice=invoice, method=method, layout=layout,
                  scanned=scanned, ocr_confidence=confidence, issues=issues)


def process_batch(paths: list[Path], ai: AIExtractor | None = None,
                  progress: Callable[[int, int, str], None] | None = None) -> list[Result]:
    results = []
    for n, p in enumerate(paths, start=1):
        if progress:
            progress(n, len(paths), Path(p).name)
        results.append(process_file(p, ai))
    flag_duplicates(results)
    return results
