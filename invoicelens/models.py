"""Data model shared by every extraction path (rules, OCR, AI)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# The 12 header fields every invoice is reduced to, in export order.
HEADER_FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "supplier_name",
    "supplier_tax_id",
    "customer_name",
    "customer_tax_id",
    "currency",
    "subtotal",
    "tax_rate",
    "tax_amount",
    "total",
]

AMOUNT_FIELDS = {"subtotal", "tax_rate", "tax_amount", "total"}
DATE_FIELDS = {"invoice_date", "due_date"}
REQUIRED_FIELDS = {"invoice_number", "invoice_date", "supplier_name", "currency", "subtotal", "total"}


@dataclass
class LineItem:
    description: str
    quantity: float | None
    unit_price: float | None
    amount: float | None


@dataclass
class Invoice:
    invoice_number: str | None = None
    invoice_date: str | None = None  # ISO 8601 (YYYY-MM-DD)
    due_date: str | None = None
    supplier_name: str | None = None
    supplier_tax_id: str | None = None
    customer_name: str | None = None
    customer_tax_id: str | None = None
    currency: str | None = None  # ISO 4217 (USD, EUR)
    subtotal: float | None = None
    tax_rate: float | None = None  # percent, e.g. 21.0
    tax_amount: float | None = None
    total: float | None = None
    line_items: list[LineItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Invoice":
        items = [LineItem(**li) for li in data.get("line_items", [])]
        header = {k: data.get(k) for k in HEADER_FIELDS}
        return cls(**header, line_items=items)


@dataclass
class Issue:
    code: str  # machine-readable, e.g. TOTAL_MISMATCH
    message: str  # human-readable reason shown in the review column


@dataclass
class Result:
    file: str
    invoice: Invoice
    method: str  # rules | ocr+rules | ai | none
    layout: str  # layout_a | layout_b | layout_c | unknown
    scanned: bool
    ocr_confidence: float | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.issues)

    @property
    def review_reason(self) -> str:
        return "; ".join(i.message for i in self.issues)
