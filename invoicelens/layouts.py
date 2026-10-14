"""Rule-based extractors for the three known invoice layouts.

Each parser receives reading-order text lines (from the PDF text layer or
OCR) and returns an Invoice. Parsers never invent values: anything they can't
find stays None and is reported by the validation step.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Invoice, LineItem
from .parsing import normalize_text, parse_amount, parse_date

# --------------------------------------------------------------- detection

MARKERS = {
    "layout_a": ["billto", "totaldueusd", "salestax", "customertaxid", "descriptionqtyrateamount"],
    "layout_b": ["invoiceno", "netamount", "customervatno", "totaleur", "itemquantityunitpricelinetotal"],
    "layout_c": ["documentinvoice", "grosstotal", "recipientvatid", "suppliervatid", "paymentdue"],
}
MIN_MARKERS = 2


def detect_layout(lines: list[str]) -> str:
    normalized = [normalize_text(line) for line in lines]
    scores = {
        layout: sum(any(m in line for line in normalized) for m in markers) for layout, markers in MARKERS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_MARKERS else "unknown"


# ----------------------------------------------------------------- helpers


def _find(lines: list[str], pattern: str, group: int = 1) -> str | None:
    rx = re.compile(pattern, re.IGNORECASE)
    for line in lines:
        if m := rx.search(line):
            return m.group(group).strip()
    return None


def _line_after(lines: list[str], label: str) -> str | None:
    """Text of the line following the first line whose normalized text equals label."""
    for i, line in enumerate(lines[:-1]):
        if normalize_text(line) == label:
            return lines[i + 1].strip()
    return None


def clean_tax_id(value: str | None) -> str | None:
    if not value:
        return None
    v = re.sub(r"\s", "", value.upper())
    # The letter O never appears in the supported formats: it's an OCR zero.
    return v[:2] + v[2:].replace("O", "0") if len(v) > 2 else v


def _parse_rate(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


_NUM = r"[$€]?\s?-?\d[\d.,]*"
_ROW3 = re.compile(rf"^(?P<desc>.+?)\s+(?P<qty>\d+(?:[.,]\d+)?)\s+(?P<unit>{_NUM})\s+(?P<amount>{_NUM})\s*€?$")
_ROW2 = re.compile(rf"^(?P<desc>.+?)\s+(?P<unit>{_NUM})\s+(?P<amount>{_NUM})\s*€?$")


def _items_between(lines: list[str], start: str, end: str, decimal: str, strip_position: bool = False) -> list[LineItem]:
    """Parse table rows between a header line (regex `start`) and a totals line (regex `end`)."""
    items: list[LineItem] = []
    inside = False
    for line in lines:
        if not inside:
            inside = bool(re.search(start, line, re.IGNORECASE))
            continue
        if re.search(end, line, re.IGNORECASE):
            break
        text = line.strip()
        if strip_position:
            # First column is the position number; OCR sometimes misreads it.
            first, _, rest = text.partition(" ")
            if rest and (first.isdigit() or len(first) <= 3):
                text = rest
        if m := _ROW3.match(text):
            qty = parse_amount(m["qty"], decimal)
            unit, amount = parse_amount(m["unit"], decimal), parse_amount(m["amount"], decimal)
        elif m := _ROW2.match(text):
            # Quantity column missing (OCR dropped it): infer it only when it is a whole number.
            unit, amount = parse_amount(m["unit"], decimal), parse_amount(m["amount"], decimal)
            ratio = amount / unit if unit else None
            qty = float(round(ratio)) if ratio and abs(ratio - round(ratio)) < 1e-6 else None
        else:
            continue
        items.append(LineItem(description=m["desc"].strip(), quantity=qty, unit_price=unit, amount=amount))
    return items


# ----------------------------------------------------------------- parsers


def parse_layout_a(lines: list[str]) -> Invoice:
    us_date = ["%b %d, %Y", "%B %d, %Y"]
    tax = re.compile(r"Sales Tax \(([\d.,]+)\s*%\)\s+(.+)$", re.IGNORECASE)
    tax_match = next((m for line in lines if (m := tax.search(line))), None)
    currency = _find(lines, r"Total Due \(([A-Z]{3})\)") or ("USD" if any("$" in line for line in lines) else None)
    return Invoice(
        invoice_number=_find(lines, r"Invoice\s*#\s*:?\s*([A-Z0-9][A-Z0-9\-/]+)"),
        invoice_date=parse_date(_find(lines, r"\bDate:\s*([A-Za-z]{3,9}\.? \d{1,2},? \d{4})"), us_date),
        due_date=parse_date(_find(lines, r"\bDue:\s*([A-Za-z]{3,9}\.? \d{1,2},? \d{4})"), us_date),
        supplier_name=_line_after(lines, "invoice"),
        supplier_tax_id=clean_tax_id(_find(lines, r"\bEIN\s*:?\s*(\d{2}-?\d{7})")),
        customer_name=_line_after(lines, "billto"),
        customer_tax_id=clean_tax_id(_find(lines, r"Customer Tax ID:?\s*(\S+)")),
        currency=currency,
        subtotal=parse_amount(_find(lines, r"Subtotal\s+(.+)$"), "."),
        tax_rate=_parse_rate(tax_match.group(1)) if tax_match else None,
        tax_amount=parse_amount(tax_match.group(2), ".") if tax_match else None,
        total=parse_amount(_find(lines, r"Total Due \([A-Z]{3}\)\s+(.+)$"), "."),
        line_items=_items_between(lines, r"Description.*Qty.*Rate", r"^Subtotal", "."),
    )


def parse_layout_b(lines: list[str]) -> Invoice:
    eu_date = ["%d/%m/%Y"]
    first = lines[0] if lines else ""
    tax = re.compile(r"^VAT\s*([\d.,]+)\s*%\s+(.+)$", re.IGNORECASE)
    tax_match = next((m for line in lines if (m := tax.search(line.strip()))), None)
    return Invoice(
        invoice_number=_find(lines, r"Invoice No\.?:?\s*(\S+)"),
        invoice_date=parse_date(_find(lines, r"Invoice Date:?\s*(\d{1,2}/\d{1,2}/\d{4})"), eu_date),
        due_date=parse_date(_find(lines, r"Due Date:?\s*(\d{1,2}/\d{1,2}/\d{4})"), eu_date),
        supplier_name=re.sub(r"\s*INVOICE\s*$", "", first).strip() or None,
        supplier_tax_id=clean_tax_id(_find(lines, r"^\s*VAT No\.?:?\s*(.+)$")),
        customer_name=_line_after(lines, "customer"),
        customer_tax_id=clean_tax_id(_find(lines, r"Customer VAT No\.?:?\s*(.+)$")),
        currency=_find(lines, r"\bTotal (EUR|USD|GBP)\b"),
        subtotal=parse_amount(_find(lines, r"Net amount\s+(.+)$"), "."),
        tax_rate=_parse_rate(tax_match.group(1)) if tax_match else None,
        tax_amount=parse_amount(tax_match.group(2), ".") if tax_match else None,
        total=parse_amount(_find(lines, r"\bTotal (?:EUR|USD|GBP)\s+(.+)$"), "."),
        line_items=_items_between(lines, r"Unit Price.*Line Total", r"^Net amount", "."),
    )


def parse_layout_c(lines: list[str]) -> Invoice:
    first = lines[0] if lines else ""
    tax = re.compile(r"^VAT\s*([\d.,]+)\s*%\s+(.+)$", re.IGNORECASE)
    tax_match = next((m for line in lines if (m := tax.search(line.strip()))), None)
    total_line = _find(lines, r"Gross total\s+(.+)$")
    return Invoice(
        invoice_number=_find(lines, r"\bNumber:?\s*([A-Z0-9][A-Z0-9\-/]+)"),
        invoice_date=parse_date(_find(lines, r"Issued:?\s*(\d{4}-\d{2}-\d{2})")),
        due_date=parse_date(_find(lines, r"Payment due:?\s*(\d{4}-\d{2}-\d{2})")),
        supplier_name=re.sub(r"\s*Document:?\s*Invoice\s*$", "", first, flags=re.IGNORECASE).strip() or None,
        supplier_tax_id=clean_tax_id(_find(lines, r"Supplier VAT ID:?\s*(\S+)")),
        customer_name=_line_after(lines, "recipient"),
        customer_tax_id=clean_tax_id(_find(lines, r"Recipient VAT ID:?\s*(\S+)")),
        currency="EUR" if total_line and "€" in total_line else None,
        subtotal=parse_amount(_find(lines, r"^\s*Net\s+(\d.+)$"), ","),
        tax_rate=_parse_rate(tax_match.group(1)) if tax_match else None,
        tax_amount=parse_amount(tax_match.group(2), ",") if tax_match else None,
        total=parse_amount(total_line, ","),
        line_items=_items_between(lines, r"Pos\..*Description", r"^\s*Net\s+\d", ",", strip_position=True),
    )


PARSERS: dict[str, Callable[[list[str]], Invoice]] = {
    "layout_a": parse_layout_a,
    "layout_b": parse_layout_b,
    "layout_c": parse_layout_c,
}
