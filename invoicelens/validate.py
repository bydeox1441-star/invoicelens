"""Business checks that decide whether an extracted invoice needs human review."""

from __future__ import annotations

from datetime import date

from .models import REQUIRED_FIELDS, Invoice, Issue, Result
from .taxid import validate_tax_id

MONEY_TOLERANCE = 0.01
TAX_TOLERANCE = 0.02  # rounding of percentage-based tax
LOW_OCR_CONFIDENCE = 70.0  # mean Tesseract word confidence; see README for calibration


def validate_invoice(inv: Invoice) -> list[Issue]:
    issues: list[Issue] = []

    missing = sorted(f for f in REQUIRED_FIELDS if getattr(inv, f) in (None, ""))
    if missing:
        issues.append(Issue("MISSING_FIELD", f"missing {', '.join(missing)}"))

    if inv.invoice_date and inv.due_date and date.fromisoformat(inv.due_date) < date.fromisoformat(inv.invoice_date):
        issues.append(Issue("DATE_ORDER", f"due date {inv.due_date} is before invoice date {inv.invoice_date}"))

    if not inv.line_items:
        issues.append(Issue("NO_LINE_ITEMS", "no line items found"))
    else:
        for n, item in enumerate(inv.line_items, start=1):
            if None in (item.quantity, item.unit_price, item.amount):
                issues.append(Issue("LINE_INCOMPLETE", f"line {n} has missing quantity, price or amount"))
            elif abs(item.quantity * item.unit_price - item.amount) > MONEY_TOLERANCE:
                issues.append(Issue("LINE_ARITHMETIC", f"line {n}: {item.quantity:g} x {item.unit_price} != {item.amount}"))
        amounts = [i.amount for i in inv.line_items if i.amount is not None]
        if inv.subtotal is not None and abs(sum(amounts) - inv.subtotal) > MONEY_TOLERANCE:
            issues.append(Issue("LINES_MISMATCH", f"line items add up to {sum(amounts):.2f}, subtotal says {inv.subtotal:.2f}"))

    if None not in (inv.subtotal, inv.tax_rate, inv.tax_amount):
        expected_tax = inv.subtotal * inv.tax_rate / 100
        if abs(expected_tax - inv.tax_amount) > TAX_TOLERANCE:
            issues.append(Issue("TAX_MISMATCH", f"{inv.tax_rate:g}% of {inv.subtotal:.2f} is {expected_tax:.2f}, tax says {inv.tax_amount:.2f}"))

    if None not in (inv.subtotal, inv.tax_amount, inv.total):
        expected_total = inv.subtotal + inv.tax_amount
        if abs(expected_total - inv.total) > MONEY_TOLERANCE:
            issues.append(Issue("TOTAL_MISMATCH", f"subtotal + tax = {expected_total:.2f}, total says {inv.total:.2f}"))

    for label, value in (("supplier", inv.supplier_tax_id), ("customer", inv.customer_tax_id)):
        if label == "customer" and not value:
            continue  # customer tax ID is optional on many invoices
        ok, why = validate_tax_id(value)
        if not ok:
            issues.append(Issue("TAX_ID_INVALID", f"{label} tax ID: {why}"))

    return issues


def flag_duplicates(results: list[Result]) -> None:
    """Flag invoices whose (supplier, number) already appeared earlier in the batch."""
    seen: dict[tuple[str, str], str] = {}
    for r in results:
        inv = r.invoice
        if not inv.invoice_number:
            continue
        supplier = (inv.supplier_tax_id or inv.supplier_name or "").upper()
        key = (supplier, inv.invoice_number.upper())
        if key in seen:
            r.issues.append(Issue("DUPLICATE", f"same supplier and invoice number as {seen[key]}"))
        else:
            seen[key] = r.file
