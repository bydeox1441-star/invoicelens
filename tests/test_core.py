import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invoicelens.ai import SCHEMA, AIExtractionError, ClaudeExtractor, invoice_from_ai
from invoicelens.layouts import detect_layout, parse_layout_c
from invoicelens.models import Invoice, LineItem
from invoicelens.parsing import parse_amount, parse_date
from invoicelens.pipeline import process_batch, process_file
from invoicelens.taxid import validate_tax_id
from invoicelens.validate import validate_invoice

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "invoices"


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("text,decimal,expected", [
    ("$2,347.00", None, 2347.00),
    ("1.581,48 €", ",", 1581.48),
    ("1.581,48 €", None, 1581.48),
    ("EUR 7.5", None, 7.5),
    ("USD 1,2O0.5O", None, 1200.50),  # OCR letter O inside the number
    ("no number", None, None),
])
def test_parse_amount(text, decimal, expected):
    assert parse_amount(text, decimal) == expected


def test_parse_date_formats():
    assert parse_date("Sep 19, 2026") == "2026-09-19"
    assert parse_date("05/10/2026", ["%d/%m/%Y"]) == "2026-10-05"
    assert parse_date("2026-1O-07") == "2026-10-07"
    assert parse_date("not a date") is None


# ------------------------------------------------------------------- tax ids

@pytest.mark.parametrize("value,ok", [
    ("ESB33218199", True), ("B33218199", True), ("ESB33218190", False),
    ("12345678Z", True), ("12345678A", False),
    ("DE101338908", True), ("DE101338909", False),
    ("FR88702632297", True), ("FR00702632297", False),
    ("NL553035110B01", True), ("91-2867825", True), ("GB123", False),
])
def test_tax_ids(value, ok):
    assert validate_tax_id(value)[0] is ok


# ----------------------------------------------------------------- validation

def _invoice(**kw):
    base = dict(invoice_number="X-1", invoice_date="2026-10-01", due_date="2026-10-31", supplier_name="A",
                supplier_tax_id="DE101338908", currency="EUR", subtotal=100.0, tax_rate=19.0, tax_amount=19.0,
                total=119.0, line_items=[LineItem("Thing", 2, 50.0, 100.0)])
    base.update(kw)
    return Invoice(**base)


def test_clean_invoice_has_no_issues():
    assert validate_invoice(_invoice()) == []


@pytest.mark.parametrize("changes,code", [
    (dict(total=129.0), "TOTAL_MISMATCH"),
    (dict(tax_amount=21.0, total=121.0), "TAX_MISMATCH"),
    (dict(due_date="2026-09-01"), "DATE_ORDER"),
    (dict(supplier_tax_id="DE101338909"), "TAX_ID_INVALID"),
    (dict(invoice_number=None), "MISSING_FIELD"),
    (dict(line_items=[LineItem("Thing", 2, 50.0, 90.0)]), "LINE_ARITHMETIC"),
    (dict(line_items=[]), "NO_LINE_ITEMS"),
])
def test_validation_catches(changes, code):
    assert code in [i.code for i in validate_invoice(_invoice(**changes))]


# ------------------------------------------------------------------- layouts

def test_layout_c_parses_ocr_noise_in_position_column():
    lines = [
        "Kessler Maschinenbau GmbH Document: Invoice", "Supplier VAT ID: DE101338908",
        "Number: RE-26-10707", "Issued: 2026-09-17", "Payment due: 2026-10-17", "Recipient",
        "Nordlicht Medien GmbH", "Pos. Description Qty Price Total",
        "OND Conveyor belt 5 m 1 578,00 578,00", "Net 578,00 €", "VAT 19 % 109,82 €", "Gross total 687,82 €",
    ]
    assert detect_layout(lines) == "layout_c"
    inv = parse_layout_c(lines)
    assert inv.line_items[0].description == "Conveyor belt 5 m"
    assert inv.total == 687.82 and inv.currency == "EUR"
    assert validate_invoice(inv) == []


def test_unknown_layout():
    assert detect_layout(["Bill for professional services", "Total to pay €10.00"]) == "unknown"


# --------------------------------------------------------------- end to end

@pytest.mark.skipif(not SAMPLES.exists(), reason="run synthetic/generate.py first")
def test_digital_invoice_matches_ground_truth():
    truth = {t["file"]: t for t in json.loads((ROOT / "samples" / "ground_truth.json").read_text())}
    result = process_file(SAMPLES / "layout_b_01.pdf")
    assert result.method == "rules" and not result.needs_review
    assert result.invoice.to_dict() == Invoice.from_dict(truth["layout_b_01.pdf"]["invoice"]).to_dict()


@pytest.mark.skipif(not SAMPLES.exists(), reason="run synthetic/generate.py first")
def test_duplicate_is_flagged_in_batch():
    results = process_batch([SAMPLES / "layout_a_03.pdf", SAMPLES / "layout_a_06.pdf"])
    assert not results[0].needs_review
    assert [i.code for i in results[1].issues] == ["DUPLICATE"]


# ------------------------------------------------------------------- AI path

AI_ANSWER = {
    "invoice_number": "LTS-0410", "invoice_date": "2026-09-22", "due_date": "2026-10-07",
    "supplier_name": "Lumen Translation Studio S.L.", "supplier_tax_id": "B38637948",
    "customer_name": "Cobalt Retail S.A.", "customer_tax_id": "A16184954", "currency": "EUR",
    "subtotal": "730.47", "tax_rate": "21", "tax_amount": "153.40", "total": "883.87",
    "line_items": [
        {"description": "Translation", "quantity": "3045", "unit_price": "0.11", "amount": "334.95"},
        {"description": "Subtitling", "quantity": "52", "unit_price": "6.47", "amount": "336.44"},
        {"description": "Proofreading", "quantity": "1477", "unit_price": "0.04", "amount": "59.08"},
    ],
}


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=self.text)])


def _fake_client(text, stop_reason="end_turn"):
    messages = FakeMessages(text, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_ai_schema_is_strict():
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])


@pytest.mark.skipif(not SAMPLES.exists(), reason="run synthetic/generate.py first")
def test_ai_path_used_for_unknown_layout():
    client, messages = _fake_client(json.dumps(AI_ANSWER))
    result = process_file(SAMPLES / "layout_d_01.pdf", ai=ClaudeExtractor(client=client))
    assert result.method == "ai" and not result.needs_review, result.review_reason
    request = messages.calls[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["messages"][0]["content"][0]["type"] == "document"
    assert request["fallbacks"] == "default"


def test_ai_values_are_still_validated():
    bad = dict(AI_ANSWER, total="900.00")
    assert "TOTAL_MISMATCH" in [i.code for i in validate_invoice(invoice_from_ai(bad))]


def test_ai_refusal_raises():
    client, _ = _fake_client("", stop_reason="refusal")
    with pytest.raises(AIExtractionError):
        ClaudeExtractor(client=client)(SAMPLES / "layout_d_01.pdf")
