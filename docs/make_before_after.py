"""Build docs/before_after.png: a scanned invoice next to the data extracted from it.

Usage: python docs/make_before_after.py [samples/invoices/layout_b_03_scan.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from invoicelens.models import HEADER_FIELDS  # noqa: E402
from invoicelens.pipeline import process_file  # noqa: E402
from invoicelens.text import render_page  # noqa: E402

FONT = "/System/Library/Fonts/Helvetica.ttc"
INK, MUTED, LINE, ACCENT, OK = "#1B2A41", "#6B7280", "#E5E7EB", "#2563EB", "#047857"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


def main(pdf: Path) -> Path:
    result = process_file(pdf)
    with pymupdf.open(pdf) as doc:
        page = render_page(doc[0], dpi=120).convert("RGB")
    page = page.crop((0, 0, page.width, int(page.height * 0.62)))
    scale = 900 / page.height
    page = page.resize((int(page.width * scale), 900))

    W, H = page.width + 980, 1040
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Scanned PDF (synthetic)", fill=MUTED, font=font(22))
    img.paste(page, (40, 80))
    d.rectangle((40, 80, 40 + page.width, 80 + page.height), outline=LINE, width=2)

    x0 = page.width + 110
    ax, ay = x0 - 58, 530  # arrow drawn as shapes: not every font has the glyph
    d.line((ax, ay, ax + 30, ay), fill=ACCENT, width=6)
    d.polygon([(ax + 28, ay - 12), (ax + 44, ay), (ax + 28, ay + 12)], fill=ACCENT)
    d.text((x0, 30), "Extracted and validated", fill=MUTED, font=font(22))
    y = 80
    status = "Auto-accepted: all checks passed" if not result.needs_review else f"Needs review: {result.review_reason}"
    d.rounded_rectangle((x0, y, W - 40, y + 50), radius=10, fill="#ECFDF5" if not result.needs_review else "#FFF4CE")
    d.text((x0 + 18, y + 13), status, fill=OK if not result.needs_review else "#92400E", font=font(20, bold=True))
    y += 75
    meta = f"method: {result.method}   ·   OCR confidence: {result.ocr_confidence:.0f}/100" if result.ocr_confidence else f"method: {result.method}"
    d.text((x0, y), meta, fill=MUTED, font=font(18))
    y += 40
    for name in HEADER_FIELDS:
        value = getattr(result.invoice, name)
        if isinstance(value, float):
            value = f"{value:g} %" if name == "tax_rate" else f"{value:,.2f}"
        d.line((x0, y, W - 40, y), fill=LINE, width=1)
        d.text((x0 + 4, y + 9), name, fill=MUTED, font=font(19))
        d.text((x0 + 260, y + 9), str(value), fill=INK, font=font(19, bold=True))
        y += 40
    d.line((x0, y, W - 40, y), fill=LINE, width=1)
    y += 25
    d.text((x0, y), f"{len(result.invoice.line_items)} line items", fill=MUTED, font=font(18))
    y += 32
    for li in result.invoice.line_items:
        d.text((x0 + 4, y), li.description[:34], fill=INK, font=font(18))
        d.text((x0 + 420, y), f"{li.quantity:g} × {li.unit_price:,.2f}", fill=MUTED, font=font(18))
        d.text((W - 160, y), f"{li.amount:,.2f}", fill=INK, font=font(18, bold=True))
        y += 30
    out = ROOT / "docs" / "before_after.png"
    img.save(out, optimize=True)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "invoices" / "layout_b_03_scan.pdf"
    print(main(target))
