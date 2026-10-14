"""AI fallback: extract an invoice with Claude when no rule-based layout matches.

The PDF is sent as a document block (Claude reads both text-layer and scanned
PDFs) and the answer is constrained to a strict JSON schema, so the response
always has the same shape. Values are then parsed and validated exactly like
rule-based output: the model is never trusted blindly.

Disabled unless explicitly requested (`--ai`), because it sends the document
to the Anthropic API.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import anthropic

from .layouts import clean_tax_id
from .models import HEADER_FIELDS, Invoice, LineItem
from .parsing import parse_amount, parse_date

DEFAULT_MODEL = os.environ.get("INVOICELENS_MODEL", "claude-opus-5-5")

_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        **{f: _STR for f in HEADER_FIELDS},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"description": _STR, "quantity": _STR, "unit_price": _STR, "amount": _STR},
                "required": ["description", "quantity", "unit_price", "amount"],
                "additionalProperties": False,
            },
        },
    },
    "required": HEADER_FIELDS + ["line_items"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract data from a single invoice for an accounts-payable system.

Copy values exactly as printed. If a field is not on the document, return an empty string for it: an empty value is sent to a human for review, while a guessed value would be booked as fact.

Formats:
- Dates: YYYY-MM-DD.
- Amounts, quantities and tax_rate: plain numbers with a dot as decimal separator and no thousands separators or currency symbols (1234.56). tax_rate is the percentage (21 for 21%).
- currency: ISO 4217 code (EUR, USD). Infer it from the symbol only when the symbol is unambiguous (€ -> EUR).
- supplier_tax_id / customer_tax_id: the VAT, CIF/NIF or EIN exactly as printed, without the label.
- supplier is the party issuing the invoice; customer is the party being billed.
- line_items: one entry per billed line, in document order."""


class AIExtractionError(RuntimeError):
    pass


class ClaudeExtractor:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def __call__(self, pdf_path: Path) -> Invoice:
        data = base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("ascii")
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                # Server-side fallback: if the request is declined by a safety
                # classifier, the API retries it on a substitute model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                            {"type": "text", "text": "Extract this invoice."},
                        ],
                    }
                ],
            )
        except anthropic.AuthenticationError as e:
            raise AIExtractionError("Anthropic API authentication failed (check ANTHROPIC_API_KEY)") from e
        except anthropic.RateLimitError as e:
            raise AIExtractionError("Anthropic API rate limit reached; retry later") from e
        except anthropic.APIStatusError as e:
            raise AIExtractionError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise AIExtractionError("could not reach the Anthropic API") from e

        if response.stop_reason == "refusal":
            raise AIExtractionError("the model declined to process this document")
        if response.stop_reason == "max_tokens":
            raise AIExtractionError("the model response was cut off (max_tokens)")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise AIExtractionError("the model returned no text")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise AIExtractionError("the model returned invalid JSON") from e
        return invoice_from_ai(raw)


def _blank(v: str | None) -> str | None:
    v = (v or "").strip()
    return v or None


def invoice_from_ai(raw: dict) -> Invoice:
    """Convert the all-strings JSON answer into a typed Invoice."""
    items = [
        LineItem(
            description=(li.get("description") or "").strip(),
            quantity=parse_amount(_blank(li.get("quantity")), "."),
            unit_price=parse_amount(_blank(li.get("unit_price")), "."),
            amount=parse_amount(_blank(li.get("amount")), "."),
        )
        for li in raw.get("line_items", [])
    ]
    rate = _blank(raw.get("tax_rate"))
    currency = _blank(raw.get("currency"))
    return Invoice(
        invoice_number=_blank(raw.get("invoice_number")),
        invoice_date=parse_date(_blank(raw.get("invoice_date"))),
        due_date=parse_date(_blank(raw.get("due_date"))),
        supplier_name=_blank(raw.get("supplier_name")),
        supplier_tax_id=clean_tax_id(_blank(raw.get("supplier_tax_id"))),
        customer_name=_blank(raw.get("customer_name")),
        customer_tax_id=clean_tax_id(_blank(raw.get("customer_tax_id"))),
        currency=currency.upper()[:3] if currency else None,
        subtotal=parse_amount(_blank(raw.get("subtotal")), "."),
        tax_rate=float(rate) if rate and rate.replace(".", "", 1).isdigit() else None,
        tax_amount=parse_amount(_blank(raw.get("tax_amount")), "."),
        total=parse_amount(_blank(raw.get("total")), "."),
        line_items=items,
    )
