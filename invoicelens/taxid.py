"""Format and checksum validation for the tax IDs that appear in the demo set.

Covers Spanish NIF/NIE/CIF, French and German VAT numbers (with checksums),
Dutch VAT numbers and US EINs (format only). Anything else is reported as
"unrecognized" rather than silently accepted.
"""

from __future__ import annotations

import re

_NIF_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
_CIF_CONTROL_LETTERS = "JABCDEFGHI"


def _clean(value: str) -> str:
    return re.sub(r"[\s.\-]", "", value.upper())


def _spanish_nif(body: str) -> bool:
    if re.fullmatch(r"\d{8}[A-Z]", body):
        return _NIF_LETTERS[int(body[:8]) % 23] == body[8]
    if re.fullmatch(r"[XYZ]\d{7}[A-Z]", body):
        number = str("XYZ".index(body[0])) + body[1:8]
        return _NIF_LETTERS[int(number) % 23] == body[8]
    return False


def spanish_cif_control(letter_and_digits: str) -> tuple[str, str]:
    """Return (digit_control, letter_control) for the first 8 CIF characters."""
    digits = letter_and_digits[1:8]
    even = sum(int(d) for d in digits[1::2])
    odd = sum(sum(divmod(int(d) * 2, 10)) for d in digits[0::2])
    control = (10 - (even + odd) % 10) % 10
    return str(control), _CIF_CONTROL_LETTERS[control]


def _spanish_cif(body: str) -> bool:
    if not re.fullmatch(r"[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]", body):
        return False
    digit, letter = spanish_cif_control(body[:8])
    if body[0] in "PQRSNW":
        return body[8] == letter
    if body[0] in "ABEH":
        return body[8] == digit
    return body[8] in (digit, letter)


def french_vat_key(siren: str) -> str:
    return f"{(12 + 3 * (int(siren) % 97)) % 97:02d}"


def german_vat_check_digit(first8: str) -> str:
    product = 10
    for d in first8:
        total = (int(d) + product) % 10 or 10
        product = (2 * total) % 11
    check = 11 - product
    return "0" if check == 10 else str(check % 10)


def validate_tax_id(value: str | None) -> tuple[bool, str]:
    """Return (is_valid, explanation)."""
    if not value:
        return False, "tax ID missing"
    v = _clean(value)
    if v.startswith("ES"):
        v = v[2:]
    if re.fullmatch(r"[0-9XYZ]\d{7}[A-Z]", v):
        return (True, "valid Spanish NIF/NIE") if _spanish_nif(v) else (False, "Spanish NIF/NIE checksum fails")
    if re.fullmatch(r"[A-W]\d{7}[0-9A-J]", v):
        return (True, "valid Spanish CIF") if _spanish_cif(v) else (False, "Spanish CIF checksum fails")
    if m := re.fullmatch(r"FR([0-9A-Z]{2})(\d{9})", v):
        key, siren = m.groups()
        if not key.isdigit():
            return True, "French VAT (alphanumeric key, format only)"
        return (True, "valid French VAT") if key == french_vat_key(siren) else (False, "French VAT key fails")
    if m := re.fullmatch(r"DE(\d{9})", v):
        digits = m.group(1)
        ok = german_vat_check_digit(digits[:8]) == digits[8]
        return (True, "valid German VAT") if ok else (False, "German VAT checksum fails")
    if re.fullmatch(r"NL\d{9}B\d{2}", v):
        return True, "Dutch VAT (format only)"
    if re.fullmatch(r"\d{2}-?\d{7}", _clean(value).replace("EIN", "")):
        return True, "US EIN (format only)"
    return False, f"unrecognized tax ID format '{value}'"
