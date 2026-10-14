"""Turn a PDF into reading-order text lines, from the text layer or via OCR.

Both paths return the same shape (a list of strings, top to bottom, words
joined left to right) so the layout parsers don't care where text came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageFilter

# A page with fewer extractable characters than this is treated as scanned.
MIN_TEXT_CHARS = 40
OCR_DPI = 300


@dataclass
class Word:
    x: float
    y: float  # vertical centre
    h: float
    text: str


# Leftovers of table borders in OCR output (a lone hyphen is kept: it can be content).
_NOISE_TOKEN = re.compile(r"^(?:[|_—–=~`‘’\"]+|[.,']+)$")


def _group_lines(words: list[Word], drop_noise: bool = False) -> list[str]:
    """Group words whose vertical centres are close into lines.

    The tolerance comes from the median word height on the page, so neither
    tiny tokens ("-") nor oversized OCR garbage can split or merge rows.
    """
    if drop_noise:
        words = [w for w in words if not _NOISE_TOKEN.match(w.text)]
    if not words:
        return []
    tolerance = max(3.0, 0.5 * float(np.median([w.h for w in words])))
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.y, w.x)):
        if lines and abs(sum(x.y for x in lines[-1]) / len(lines[-1]) - w.y) <= tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [" ".join(w.text for w in sorted(line, key=lambda w: w.x)) for line in lines]


def is_scanned(doc: pymupdf.Document) -> bool:
    return all(len(page.get_text("text").strip()) < MIN_TEXT_CHARS for page in doc)


def digital_lines(doc: pymupdf.Document) -> list[str]:
    out: list[str] = []
    for page in doc:
        words = [Word(x0, (y0 + y1) / 2, y1 - y0, t) for x0, y0, x1, y1, t, *_ in page.get_text("words")]
        out.extend(_group_lines(words))
    return out


def render_page(page: pymupdf.Page, dpi: int = OCR_DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def _long_runs(mask: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Pixels belonging to runs of at least k consecutive True values along axis."""
    m = np.moveaxis(mask, axis, -1).astype(np.int32)
    n = m.shape[-1]
    zeros = np.zeros(m.shape[:-1] + (1,), dtype=np.int32)
    cs = np.concatenate([zeros, np.cumsum(m, axis=-1)], axis=-1)
    starts = (cs[..., k:] - cs[..., :-k]) == k  # a full run begins at index i
    cs2 = np.concatenate([zeros, np.cumsum(starts, axis=-1, dtype=np.int32)], axis=-1)
    j = np.arange(n)
    lo = np.clip(j - k + 1, 0, n - k + 1)
    hi = np.clip(j + 1, 0, n - k + 1)
    covered = (cs2[..., hi] - cs2[..., lo]) > 0
    return np.moveaxis(covered, -1, axis)


def estimate_skew(img: Image.Image, max_angle: float = 3.0, step: float = 0.1) -> float:
    """Angle (degrees) that makes text rows most horizontal (projection profile)."""
    small = img.resize((800, int(800 * img.height / img.width)))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        rotated = np.asarray(small.rotate(float(angle), fillcolor=255)) < 128
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def clean_for_ocr(img: Image.Image) -> Image.Image:
    """Deskew, remove speckle noise and table rules, which make Tesseract emit garbage."""
    img = img.rotate(estimate_skew(img), resample=Image.BICUBIC, fillcolor=255)
    img = img.filter(ImageFilter.MedianFilter(3))
    arr = np.asarray(img).copy()
    dark = arr < 140
    k = img.width // 15  # ~0.5 inch at 300 dpi: far longer than any glyph stroke
    rules = _long_runs(dark, k, axis=1) | _long_runs(dark, k, axis=0)
    # Grow the mask by one pixel so anti-aliased edges of the rules go too.
    rules |= np.roll(rules, 1, 0) | np.roll(rules, -1, 0) | np.roll(rules, 1, 1) | np.roll(rules, -1, 1)
    arr[rules] = 255
    return Image.fromarray(arr)


def ocr_lines(doc: pymupdf.Document) -> tuple[list[str], float]:
    """OCR every page. Returns (lines, mean word confidence 0-100)."""
    out: list[str] = []
    confidences: list[float] = []
    for page in doc:
        image = clean_for_ocr(render_page(page))
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 3")
        words = []
        for text, conf, top, height, left in zip(data["text"], data["conf"], data["top"], data["height"], data["left"]):
            if not text.strip() or float(conf) < 0:
                continue
            words.append(Word(left, top + height / 2, height, text))
            confidences.append(float(conf))
        out.extend(_group_lines(words, drop_noise=True))
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return out, round(mean_conf, 1)
