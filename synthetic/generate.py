"""Generate the synthetic InvoiceLens test set.

Creates 30 fictitious invoices in four layouts:
  - layout_a  "Modern"   US supplier, USD, "Oct 14, 2026" dates       (8)
  - layout_b  "Classic"  ES/NL supplier, EUR, DD/MM/YYYY, bordered     (8)
  - layout_c  "Compact"  DE/FR supplier, EUR, 1.234,56 numbers, ISO    (8)
  - layout_d  "Letter"   prose-style bill with no extraction rules     (6)

10 of them are turned into image-only "scans" (rotation, noise, blur, JPEG),
one of them deliberately bad. A few invoices carry designed anomalies
(arithmetic error, duplicate, invalid tax ID, due date before issue date) so
the validation layer has something real to catch.

Every company, person and number is invented. Output:
  samples/invoices/*.pdf
  samples/ground_truth.json

Usage: python synthetic/generate.py [--out samples] [--seed 42]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invoicelens.taxid import french_vat_key, german_vat_check_digit, spanish_cif_control  # noqa: E402

# --------------------------------------------------------------------------- ids


def cif(rng: random.Random, letter: str) -> str:
    body = letter + "".join(str(rng.randint(0, 9)) for _ in range(7))
    digit, ctrl_letter = spanish_cif_control(body)
    return "ES" + body + (ctrl_letter if letter in "PQRSNW" else digit)


def de_vat(rng: random.Random) -> str:
    first8 = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(7))
    return "DE" + first8 + german_vat_check_digit(first8)


def fr_vat(rng: random.Random) -> str:
    siren = str(rng.randint(100_000_000, 999_999_999))
    return "FR" + french_vat_key(siren) + siren


def nl_vat(rng: random.Random) -> str:
    return "NL" + str(rng.randint(100_000_000, 999_999_999)) + "B01"


def ein(rng: random.Random) -> str:
    return f"{rng.randint(10, 99)}-{rng.randint(1_000_000, 9_999_999)}"


# ---------------------------------------------------------------------- catalog


@dataclass
class Party:
    name: str
    address: list[str]
    tax_id: str


SERVICES_US = [
    ("Dashboard development (sprint)", 1, 1800, 4200),
    ("Data pipeline maintenance - monthly", 1, 650, 1400),
    ("Analytics consulting (hours)", 4, 95, 160),
    ("Cloud hosting - managed plan", 1, 120, 480),
    ("Report automation setup", 1, 400, 950),
    ("Training workshop (half day)", 1, 600, 1100),
]
SERVICES_EU = [
    ("Website maintenance", 1, 180, 450),
    ("UX audit", 1, 900, 2400),
    ("Pallet storage (per pallet/month)", 12, 9, 18),
    ("Freight Madrid - Rotterdam", 1, 650, 1300),
    ("Technical support (hours)", 3, 45, 85),
    ("Translation EN>ES (per 1,000 words)", 4, 70, 110),
]
PARTS_DE = [
    ("Spare part kit KX-200", 2, 120, 1450),
    ("Hydraulic pump service", 1, 380, 990),
    ("Bearing set 6204-2RS", 10, 6, 15),
    ("On-site technician (hours)", 4, 65, 95),
    ("Calibration certificate", 1, 140, 260),
    ("Conveyor belt 5 m", 1, 410, 760),
]
TRANSLATION = [
    ("Translation EN>ES, product catalogue", 4200, 0.07, 0.11),
    ("Proofreading, terms and conditions", 1800, 0.03, 0.05),
    ("Subtitling, onboarding videos", 35, 6, 9),
]


def money_en(v: float) -> str:
    return f"{v:,.2f}"


def money_eu(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def make_items(rng: random.Random, catalog, n_min=1, n_max=5) -> list[dict]:
    items = []
    for desc, qty, lo, hi in rng.sample(catalog, rng.randint(n_min, min(n_max, len(catalog)))):
        q = qty if qty == 1 else rng.randint(max(1, int(qty * 0.5)), int(qty * 1.5))
        unit = round(rng.uniform(lo, hi), 2 if hi < 1 else 0 if hi > 300 else 2)
        items.append({"description": desc, "quantity": float(q), "unit_price": unit, "amount": round(q * unit, 2)})
    return items


# ---------------------------------------------------------------------- layouts


def draw_layout_a(c: canvas.Canvas, inv: dict, sup: Party, cus: Party) -> None:
    w, h = LETTER
    c.setFillColor(colors.HexColor("#DCE8F5"))
    c.rect(0, h - 80, w, 80, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#1B2A41"))
    c.setFont("Helvetica-Bold", 28)
    c.drawString(50, h - 55, "INVOICE")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 115, sup.name)
    c.setFont("Helvetica", 10)
    for i, line in enumerate(sup.address + [f"EIN {sup.tax_id}"]):
        c.drawString(50, h - 132 - 14 * i, line)
    c.setFont("Helvetica", 10)
    c.drawString(370, h - 180, f"Invoice #: {inv['invoice_number']}")
    c.drawString(370, h - 194, f"Date: {fmt_date(inv['invoice_date'], '%b %d, %Y')}")
    c.drawString(370, h - 208, f"Due: {fmt_date(inv['due_date'], '%b %d, %Y')}")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, h - 240, "Bill To")
    c.setFont("Helvetica", 10)
    c.drawString(50, h - 256, cus.name)
    for i, line in enumerate(cus.address):
        c.drawString(50, h - 270 - 14 * i, line)
    c.drawString(50, h - 270 - 14 * len(cus.address), f"Customer Tax ID: {cus.tax_id}")
    y = h - 350
    c.setFillColor(colors.HexColor("#EEF2F7"))
    c.rect(45, y - 6, w - 90, 20, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Description")
    c.drawRightString(390, y, "Qty")
    c.drawRightString(470, y, "Rate")
    c.drawRightString(560, y, "Amount")
    c.setFont("Helvetica", 10)
    for item in inv["line_items"]:
        y -= 22
        c.drawString(50, y, item["description"])
        c.drawRightString(390, y, f"{item['quantity']:g}")
        c.drawRightString(470, y, "$" + money_en(item["unit_price"]))
        c.drawRightString(560, y, "$" + money_en(item["amount"]))
    y -= 40
    c.drawRightString(470, y, "Subtotal")
    c.drawRightString(560, y, "$" + money_en(inv["subtotal"]))
    c.drawRightString(470, y - 16, f"Sales Tax ({inv['tax_rate']:g}%)")
    c.drawRightString(560, y - 16, "$" + money_en(inv["tax_amount"]))
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(470, y - 36, "Total Due (USD)")
    c.drawRightString(560, y - 36, "$" + money_en(inv["total"]))
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 60, "Thank you for your business. Payment terms: Net 30.")


def draw_layout_b(c: canvas.Canvas, inv: dict, sup: Party, cus: Party) -> None:
    w, h = A4
    c.setFont("Times-Bold", 18)
    c.drawString(50, h - 60, sup.name)
    c.setFont("Times-Roman", 10)
    for i, line in enumerate(sup.address + [f"VAT No.: {sup.tax_id}"]):
        c.drawString(50, h - 78 - 13 * i, line)
    c.setFont("Times-Bold", 22)
    c.drawRightString(w - 50, h - 60, "INVOICE")
    c.rect(330, h - 190, w - 380, 62)
    c.setFont("Times-Roman", 10)
    c.drawString(340, h - 145, f"Invoice No.: {inv['invoice_number']}")
    c.drawString(340, h - 162, f"Invoice Date: {fmt_date(inv['invoice_date'], '%d/%m/%Y')}")
    c.drawString(340, h - 179, f"Due Date: {fmt_date(inv['due_date'], '%d/%m/%Y')}")
    c.setFont("Times-Bold", 11)
    c.drawString(50, h - 215, "Customer:")
    c.setFont("Times-Roman", 10)
    c.drawString(50, h - 230, cus.name)
    for i, line in enumerate(cus.address + [f"Customer VAT No.: {cus.tax_id}"]):
        c.drawString(50, h - 243 - 13 * i, line)
    top = h - 320
    cols = [50, 330, 400, 480, w - 50]
    rows = len(inv["line_items"]) + 1
    for r in range(rows + 1):
        c.line(cols[0], top - 22 * r, cols[-1], top - 22 * r)
    for x in cols:
        c.line(x, top, x, top - 22 * rows)
    c.setFont("Times-Bold", 10)
    for x, label in zip(cols, ["Item", "Quantity", "Unit Price", "Line Total"]):
        c.drawString(x + 5, top - 15, label)
    c.setFont("Times-Roman", 10)
    for r, item in enumerate(inv["line_items"], start=1):
        y = top - 22 * r - 15
        c.drawString(cols[0] + 5, y, item["description"])
        c.drawRightString(cols[2] - 5, y, f"{item['quantity']:g}")
        c.drawRightString(cols[3] - 5, y, money_en(item["unit_price"]))
        c.drawRightString(cols[4] - 5, y, money_en(item["amount"]))
    y = top - 22 * rows - 30
    c.drawString(380, y, "Net amount")
    c.drawRightString(w - 55, y, money_en(inv["subtotal"]))
    c.drawString(380, y - 16, f"VAT {inv['tax_rate']:g}%")
    c.drawRightString(w - 55, y - 16, money_en(inv["tax_amount"]))
    c.setFont("Times-Bold", 11)
    c.drawString(380, y - 36, "Total EUR")
    c.drawRightString(w - 55, y - 36, money_en(inv["total"]))
    c.setFont("Times-Roman", 9)
    c.drawString(50, 70, "Payment by bank transfer within 30 days. IBAN: XX00 0000 0000 0000 0000 (demo)")


def draw_layout_c(c: canvas.Canvas, inv: dict, sup: Party, cus: Party) -> None:
    w, h = A4
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, h - 50, sup.name)
    c.setFont("Helvetica", 9)
    for i, line in enumerate(sup.address):
        c.drawString(40, h - 64 - 11 * i, line)
    c.drawString(40, h - 64 - 11 * len(sup.address), f"Supplier VAT ID: {sup.tax_id}")
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(w - 40, h - 50, "Document: Invoice")
    c.setFont("Helvetica", 9)
    c.drawString(390, h - 120, f"Number: {inv['invoice_number']}")
    c.drawString(390, h - 132, f"Issued: {inv['invoice_date']}")
    c.drawString(390, h - 144, f"Payment due: {inv['due_date']}")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, h - 170, "Recipient")
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 182, cus.name)
    for i, line in enumerate(cus.address + [f"Recipient VAT ID: {cus.tax_id}"]):
        c.drawString(40, h - 194 - 11 * i, line)
    y = h - 270
    c.setStrokeColor(colors.grey)
    c.line(40, y + 12, w - 40, y + 12)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Pos.")
    c.drawString(75, y, "Description")
    c.drawRightString(390, y, "Qty")
    c.drawRightString(470, y, "Price")
    c.drawRightString(w - 40, y, "Total")
    c.line(40, y - 5, w - 40, y - 5)
    c.setFont("Helvetica", 9)
    for pos, item in enumerate(inv["line_items"], start=1):
        y -= 16
        c.drawString(40, y, str(pos))
        c.drawString(75, y, item["description"])
        c.drawRightString(390, y, f"{item['quantity']:g}")
        c.drawRightString(470, y, money_eu(item["unit_price"]))
        c.drawRightString(w - 40, y, money_eu(item["amount"]))
    c.line(40, y - 8, w - 40, y - 8)
    y -= 28
    c.drawString(390, y, "Net")
    c.drawRightString(w - 40, y, money_eu(inv["subtotal"]) + " €")
    c.drawString(390, y - 13, f"VAT {inv['tax_rate']:g} %")
    c.drawRightString(w - 40, y - 13, money_eu(inv["tax_amount"]) + " €")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(390, y - 30, "Gross total")
    c.drawRightString(w - 40, y - 30, money_eu(inv["total"]) + " €")


def draw_layout_d(c: canvas.Canvas, inv: dict, sup: Party, cus: Party) -> None:
    w, h = A4
    c.setFont("Helvetica", 11)
    c.drawString(60, h - 70, sup.name + "  |  " + sup.address[0])
    c.drawString(60, h - 86, f"CIF {sup.tax_id[2:]}")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, h - 130, "Bill for professional services")
    c.setFont("Helvetica", 11)
    issued = fmt_date(inv["invoice_date"], "%d %B %Y").lstrip("0")
    due = fmt_date(inv["due_date"], "%d %B %Y").lstrip("0")
    c.drawString(60, h - 160, f"Reference {inv['invoice_number']}, issued on {issued}.")
    c.drawString(60, h - 176, f"Payable within 15 days, by {due}.")
    c.drawString(60, h - 206, f"Client: {cus.name}, CIF {cus.tax_id[2:]}")
    y = h - 250
    for item in inv["line_items"]:
        unit = f"€{item['unit_price']:.2f}"
        qty = f"{item['quantity']:,.0f}"
        c.drawString(60, y, f"{item['description']} ({qty} units at {unit})")
        c.drawRightString(w - 60, y, f"€{money_en(item['amount'])}")
        y -= 20
    y -= 20
    c.drawString(60, y, "Amount before tax")
    c.drawRightString(w - 60, y, f"€{money_en(inv['subtotal'])}")
    c.drawString(60, y - 18, f"IVA at {inv['tax_rate']:g} percent")
    c.drawRightString(w - 60, y - 18, f"€{money_en(inv['tax_amount'])}")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(60, y - 40, "Total to pay")
    c.drawRightString(w - 60, y - 40, f"€{money_en(inv['total'])}")


DRAW = {"layout_a": draw_layout_a, "layout_b": draw_layout_b, "layout_c": draw_layout_c, "layout_d": draw_layout_d}
PAGE = {"layout_a": LETTER, "layout_b": A4, "layout_c": A4, "layout_d": A4}


def fmt_date(iso: str, fmt: str) -> str:
    return date.fromisoformat(iso).strftime(fmt)


# ------------------------------------------------------------------------ scans


def degrade_to_scan(pdf_path: Path, rng: random.Random, bad: bool = False) -> None:
    """Re-save a digital PDF as an image-only PDF that looks scanned."""
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for page in src:
        dpi = 110 if bad else 170
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        angle = rng.uniform(-2.5, 2.5) if bad else rng.uniform(-1.0, 1.0)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 10_000)).normal(0, 28 if bad else 9, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(1.3 if bad else 0.5))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=22 if bad else 60)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    src.close()
    out.save(pdf_path)


# ---------------------------------------------------------------------- builder


def build(out_dir: Path, seed: int) -> list[dict]:
    rng = random.Random(seed)
    inv_dir = out_dir / "invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)
    for old in inv_dir.glob("*.pdf"):
        old.unlink()

    suppliers = {
        "layout_a": [
            Party("Brightline Analytics LLC", ["410 Market Street, Suite 300", "San Francisco, CA 94111"], ein(rng)),
            Party("Harbor & Pine Consulting LLC", ["88 Water Street", "New York, NY 10005"], ein(rng)),
        ],
        "layout_b": [
            Party("Almagro Soluciones Digitales S.L.", ["Calle de Alcalá 120", "28009 Madrid, Spain"], cif(rng, "B")),
            Party("Van der Berg Logistics B.V.", ["Havenstraat 14", "3011 Rotterdam, Netherlands"], nl_vat(rng)),
        ],
        "layout_c": [
            Party("Kessler Maschinenbau GmbH", ["Industriestraße 7", "70565 Stuttgart, Germany"], de_vat(rng)),
            Party("Atelier Moreau SARL", ["12 rue des Tanneurs", "69002 Lyon, France"], fr_vat(rng)),
        ],
        "layout_d": [Party("Lumen Translation Studio S.L.", ["Carrer de Pallars 85, 08018 Barcelona"], cif(rng, "B"))],
    }
    customers = {
        "layout_a": [Party("Meridian Health Partners Inc.", ["2200 Lake Shore Drive", "Chicago, IL 60616"], ein(rng))],
        "layout_b": [
            Party("Cobalt Retail S.A.", ["Paseo de Gracia 55", "08007 Barcelona, Spain"], cif(rng, "A")),
            Party("Polder Foods B.V.", ["Keizersgracht 210", "1016 Amsterdam, Netherlands"], nl_vat(rng)),
        ],
        "layout_c": [
            Party("Nordlicht Medien GmbH", ["Schanzenstraße 22", "20357 Hamburg, Germany"], de_vat(rng)),
            Party("Studio Lumière SAS", ["5 avenue Foch", "75116 Paris, France"], fr_vat(rng)),
        ],
        "layout_d": [Party("Cobalt Retail S.A.", ["Paseo de Gracia 55", "08007 Barcelona, Spain"], cif(rng, "A"))],
    }
    plan = {
        "layout_a": dict(count=8, catalog=SERVICES_US, currency="USD", rates=[8.875, 7.25], prefix="INV-2026-"),
        "layout_b": dict(count=8, catalog=SERVICES_EU, currency="EUR", rates=[21.0], prefix="F2026/"),
        "layout_c": dict(count=8, catalog=PARTS_DE, currency="EUR", rates=[19.0, 20.0], prefix="RE-26-"),
        "layout_d": dict(count=6, catalog=TRANSLATION, currency="EUR", rates=[21.0], prefix="LTS-"),
    }
    scanned = {("layout_a", 1), ("layout_a", 4), ("layout_a", 6), ("layout_b", 2), ("layout_b", 5),
               ("layout_b", 7), ("layout_c", 2), ("layout_c", 5), ("layout_c", 7), ("layout_d", 3)}
    bad_scan = ("layout_c", 7)

    records = []
    start = date(2026, 9, 1)
    for layout, p in plan.items():
        seq = rng.randint(100, 800)
        for idx in range(p["count"]):
            sup = suppliers[layout][idx % len(suppliers[layout])]
            cus = customers[layout][idx % len(customers[layout])]
            rate = p["rates"][idx % len(p["rates"])]
            if layout == "layout_c":
                rate = 19.0 if sup.tax_id.startswith("DE") else 20.0
            items = make_items(rng, p["catalog"], 1, 3 if layout == "layout_d" else 5)
            subtotal = round(sum(i["amount"] for i in items), 2)
            tax = round(subtotal * rate / 100, 2)
            issued = start + timedelta(days=rng.randint(0, 40))
            due = issued + timedelta(days=15 if layout == "layout_d" else 30)
            seq += rng.randint(1, 9)
            number = f"{p['prefix']}{seq:04d}" if layout != "layout_c" else f"{p['prefix']}{seq + 10000}"
            inv = {
                "invoice_number": number,
                "invoice_date": issued.isoformat(),
                "due_date": due.isoformat(),
                "supplier_name": sup.name,
                "supplier_tax_id": sup.tax_id,
                "customer_name": cus.name,
                "customer_tax_id": cus.tax_id,
                "currency": p["currency"],
                "subtotal": subtotal,
                "tax_rate": rate,
                "tax_amount": tax,
                "total": round(subtotal + tax, 2),
                "line_items": items,
            }
            expected: list[str] = []
            supplier = sup
            # Designed anomalies -------------------------------------------------
            if (layout, idx) == ("layout_b", 3):
                inv["total"] = round(inv["total"] + 10.0, 2)  # supplier arithmetic error
                expected.append("TOTAL_MISMATCH")
            if (layout, idx) == ("layout_a", 5):
                dup = next(r for r in records if r["file"].startswith("layout_a_03"))
                inv = json.loads(json.dumps(dup["invoice"]))  # same invoice sent twice
                supplier = suppliers[layout][0]
                expected.append("DUPLICATE")
            if (layout, idx) == ("layout_c", 4):
                broken = sup.tax_id[:-1] + str((int(sup.tax_id[-1]) + 3) % 10)
                supplier = Party(sup.name, sup.address, broken)
                inv["supplier_tax_id"] = broken
                expected.append("TAX_ID_INVALID")
            if (layout, idx) == ("layout_a", 7):
                inv["due_date"] = (issued - timedelta(days=5)).isoformat()
                expected.append("DATE_ORDER")
            is_scan = (layout, idx) in scanned
            if (layout, idx) == bad_scan:
                expected.append("LOW_OCR_CONFIDENCE")
            if layout == "layout_d":
                expected.append("UNKNOWN_LAYOUT")  # only expected when the AI step is off
                # The letter layout prints Spanish IDs without the ES prefix.
                inv["supplier_tax_id"] = sup.tax_id[2:]
                inv["customer_tax_id"] = cus.tax_id[2:]

            name = f"{layout}_{idx + 1:02d}{'_scan' if is_scan else ''}.pdf"
            path = inv_dir / name
            c = canvas.Canvas(str(path), pagesize=PAGE[layout])
            c.setTitle(f"Invoice {inv['invoice_number']}")
            DRAW[layout](c, inv, supplier, cus)
            c.showPage()
            c.save()
            if is_scan:
                degrade_to_scan(path, rng, bad=(layout, idx) == bad_scan)
            records.append({"file": name, "layout": layout, "scanned": is_scan,
                            "expected_flags": expected, "invoice": inv})

    (out_dir / "ground_truth.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "samples"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    records = build(Path(args.out), args.seed)
    scans = sum(r["scanned"] for r in records)
    print(f"Generated {len(records)} invoices ({scans} scanned) in {args.out}/invoices")


if __name__ == "__main__":
    main()
