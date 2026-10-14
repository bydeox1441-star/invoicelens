"""Tolerant parsers for amounts and dates, including common OCR confusions."""

from __future__ import annotations

import re
from datetime import datetime

# Characters OCR often returns in place of digits inside numeric tokens.
_OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})

_AMOUNT_RE = re.compile(r"-?\d[\d.,]*\d|\d")
# Only letters touching a digit or separator are treated as misread digits,
# so currency codes such as "USD" are left alone.
_OCR_IN_NUMBER = re.compile(r"(?<=[\d.,])[OolI|SB]|[OolI|SB](?=[\d.,]?\d)")


def fix_ocr_digits(text: str) -> str:
    return _OCR_IN_NUMBER.sub(lambda m: m.group(0).translate(_OCR_DIGIT_FIXES), text)


def parse_amount(text: str | None, decimal: str | None = None) -> float | None:
    """Parse '1,234.56', '1.234,56 €', '$ 99.00', 'EUR 7.5' into a float.

    `decimal` forces the decimal separator ('.' or ','). Without it the last
    separator followed by exactly 1-2 digits is treated as decimal.
    """
    if text is None:
        return None
    match = _AMOUNT_RE.search(fix_ocr_digits(text))
    if not match:
        return None
    token = match.group(0)
    if decimal is None:
        last = max(token.rfind("."), token.rfind(","))
        decimal = token[last] if last != -1 and len(token) - last - 1 in (1, 2) else "."
    thousands = "," if decimal == "." else "."
    token = token.replace(thousands, "").replace(decimal, ".")
    try:
        return round(float(token), 2)
    except ValueError:
        return None


_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%d.%m.%Y"]


def parse_date(text: str | None, formats: list[str] | None = None) -> str | None:
    """Return an ISO date (YYYY-MM-DD) or None. Tries layout formats first."""
    if not text:
        return None
    candidate = re.sub(r"\s+", " ", text.strip().strip(".:"))
    candidate = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", candidate)
    for fmt in (formats or []) + _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_text(text: str) -> str:
    """Lowercase alphanumerics only; used for fuzzy label matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())
