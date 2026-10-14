"""Command line interface.

  python -m invoicelens process samples/invoices --out output [--ai] [--formats xlsx,csv,json]
  python -m invoicelens evaluate samples/invoices --truth samples/ground_truth.json [--ai]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import render_markdown, score
from .export import write_csv, write_excel, write_json
from .pipeline import process_batch


def _ai(enabled: bool):
    if not enabled:
        return None
    from .ai import ClaudeExtractor

    return ClaudeExtractor()


def _progress(n: int, total: int, name: str) -> None:
    print(f"\r[{n}/{total}] {name:<40}", end="", file=sys.stderr, flush=True)


def _pdfs(folder: str) -> list[Path]:
    paths = sorted(Path(folder).glob("*.pdf"))
    if not paths:
        sys.exit(f"No PDF files found in {folder}")
    return paths


def cmd_process(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = process_batch(_pdfs(args.folder), _ai(args.ai), _progress)
    print(file=sys.stderr)
    formats = set(args.formats.split(","))
    written = []
    if "xlsx" in formats:
        written.append(write_excel(results, out / "invoices.xlsx"))
    if "csv" in formats:
        written.extend(write_csv(results, out))
    if "json" in formats:
        written.append(write_json(results, out / "invoices.json"))
    flagged = [r for r in results if r.needs_review]
    print(f"Processed {len(results)} documents: {len(results) - len(flagged)} auto-accepted, {len(flagged)} need review.")
    for r in flagged:
        print(f"  - {r.file}: {r.review_reason}")
    print("Wrote: " + ", ".join(str(p) for p in written))


def cmd_evaluate(args) -> None:
    results = process_batch(_pdfs(args.folder), _ai(args.ai), _progress)
    print(file=sys.stderr)
    docs, summary = score(results, Path(args.truth), ai_enabled=args.ai)
    report = render_markdown(docs, summary, ai_enabled=args.ai)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"Report written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="invoicelens", description="Extract structured data from invoice PDFs.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="extract a folder of PDFs")
    p.add_argument("folder")
    p.add_argument("--out", default="output")
    p.add_argument("--formats", default="xlsx,csv,json")
    p.add_argument("--ai", action="store_true", help="use Claude for unknown layouts (sends PDFs to the Anthropic API)")
    p.set_defaults(func=cmd_process)

    e = sub.add_parser("evaluate", help="score extraction against a ground-truth file")
    e.add_argument("folder")
    e.add_argument("--truth", required=True)
    e.add_argument("--report", default="output/evaluation.md")
    e.add_argument("--ai", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
