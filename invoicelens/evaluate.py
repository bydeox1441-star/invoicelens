"""Measure extraction quality against the synthetic ground truth.

The headline number is not raw accuracy but *silent errors*: invoices that
passed every check (no review flag) and still contain a wrong value. Those
are the ones that would reach the books unnoticed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .models import AMOUNT_FIELDS, HEADER_FIELDS, Invoice, Result


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().casefold()


def field_matches(name: str, got, want) -> bool:
    if want in (None, ""):
        return got in (None, "")
    if got in (None, ""):
        return False
    if name in AMOUNT_FIELDS:
        return abs(float(got) - float(want)) < 0.005
    if name.endswith("tax_id"):
        return re.sub(r"[\s\-.]", "", str(got)).upper() == re.sub(r"[\s\-.]", "", str(want)).upper()
    return _norm(got) == _norm(want)


def lines_match(got: Invoice, want: Invoice) -> bool:
    if len(got.line_items) != len(want.line_items):
        return False
    for g, w in zip(got.line_items, want.line_items):
        if any(a is None or abs(a - b) >= 0.005 for a, b in
               ((g.quantity, w.quantity), (g.unit_price, w.unit_price), (g.amount, w.amount))):
            return False
        if SequenceMatcher(None, _norm(g.description), _norm(w.description)).ratio() < 0.9:
            return False
    return True


@dataclass
class DocScore:
    file: str
    layout: str
    scanned: bool
    method: str
    fields_correct: int
    fields_total: int
    wrong_fields: list[str]
    lines_ok: bool
    flagged: bool
    issue_codes: list[str]
    expected_flags: list[str]
    caught: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def fully_correct(self) -> bool:
        return not self.wrong_fields and self.lines_ok


def score(results: list[Result], truth_path: Path, ai_enabled: bool) -> tuple[list[DocScore], dict]:
    truth = {t["file"]: t for t in json.loads(Path(truth_path).read_text())}
    docs: list[DocScore] = []
    for r in results:
        t = truth[r.file]
        want = Invoice.from_dict(t["invoice"])
        wrong = [f for f in HEADER_FIELDS if not field_matches(f, getattr(r.invoice, f), getattr(want, f))]
        expected = [e for e in t["expected_flags"] if not (ai_enabled and e == "UNKNOWN_LAYOUT")]
        codes = [i.code for i in r.issues]
        docs.append(DocScore(
            file=r.file, layout=t["layout"], scanned=t["scanned"], method=r.method,
            fields_correct=len(HEADER_FIELDS) - len(wrong), fields_total=len(HEADER_FIELDS),
            wrong_fields=wrong, lines_ok=lines_match(r.invoice, want), flagged=r.needs_review,
            issue_codes=codes, expected_flags=expected,
            caught=[e for e in expected if e in codes], missed=[e for e in expected if e not in codes],
        ))

    attempted = [d for d in docs if d.method != "none"]
    accepted = [d for d in docs if not d.flagged]
    expected_total = sum(len(d.expected_flags) for d in docs)
    summary = {
        "documents": len(docs),
        "scanned": sum(d.scanned for d in docs),
        "extraction_attempted": len(attempted),
        "header_field_accuracy": round(
            sum(d.fields_correct for d in attempted) / max(1, sum(d.fields_total for d in attempted)), 4),
        "line_items_fully_correct": sum(d.lines_ok for d in attempted),
        "documents_fully_correct": sum(d.fully_correct for d in docs),
        "auto_accepted": len(accepted),
        "auto_accepted_and_correct": sum(d.fully_correct for d in accepted),
        "silent_errors": [d.file for d in accepted if not d.fully_correct],
        "flagged_for_review": len(docs) - len(accepted),
        "designed_anomalies": expected_total,
        "designed_anomalies_caught": sum(len(d.caught) for d in docs),
        "false_alarms": [d.file for d in docs if d.flagged and d.fully_correct and not d.expected_flags],
    }
    return docs, summary


def render_markdown(docs: list[DocScore], summary: dict, ai_enabled: bool) -> str:
    s = summary
    out = [
        "# InvoiceLens evaluation",
        "",
        f"AI step: {'on' if ai_enabled else 'off'}. Synthetic data set ({s['documents']} invoices, {s['scanned']} scanned).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Header field accuracy (documents with an extraction) | {s['header_field_accuracy']:.1%} |",
        f"| Documents fully correct (12 fields + all line items) | {s['documents_fully_correct']} / {s['documents']} |",
        f"| Auto-accepted (no review flag) | {s['auto_accepted']} |",
        f"| Auto-accepted and fully correct | {s['auto_accepted_and_correct']} |",
        f"| **Silent errors** (accepted but wrong) | **{len(s['silent_errors'])}** {', '.join(s['silent_errors'])} |",
        f"| Flagged for review | {s['flagged_for_review']} |",
        f"| Designed anomalies caught | {s['designed_anomalies_caught']} / {s['designed_anomalies']} |",
        f"| False alarms (flagged but fully correct, no anomaly) | {len(s['false_alarms'])} {', '.join(s['false_alarms'])} |",
        "",
        "| File | Method | Fields | Lines | Flagged | Issues | Expected anomaly | Wrong fields |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in docs:
        out.append(
            f"| {d.file} | {d.method} | {d.fields_correct}/{d.fields_total} | {'ok' if d.lines_ok else 'wrong'} | "
            f"{'yes' if d.flagged else 'no'} | {', '.join(d.issue_codes) or '-'} | "
            f"{', '.join(d.expected_flags) or '-'}{' (missed)' if d.missed else ''} | {', '.join(d.wrong_fields) or '-'} |"
        )
    return "\n".join(out) + "\n"
