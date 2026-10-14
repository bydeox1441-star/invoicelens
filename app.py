"""InvoiceLens web demo (Streamlit).

    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st

from invoicelens.evaluate import score
from invoicelens.export import invoices_frame, lines_frame, results_json, write_excel
from invoicelens.models import HEADER_FIELDS
from invoicelens.pipeline import process_batch
from invoicelens.text import render_page

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples" / "invoices"
TRUTH = ROOT / "samples" / "ground_truth.json"

st.set_page_config(page_title="InvoiceLens", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .demo-note {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("InvoiceLens")
st.caption("Invoice PDFs in, validated Excel / CSV / JSON out. Known layouts are parsed with rules, "
           "scans go through OCR, and every row is checked before it's accepted.")
st.markdown('<p class="demo-note">Demo project. All invoices in the sample set are synthetic: '
            'every company, person and number is invented.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input")
    source = st.radio("Documents", ["Sample set (30 synthetic invoices)", "Upload PDFs"])
    uploads = st.file_uploader("PDF invoices", type="pdf", accept_multiple_files=True) if source == "Upload PDFs" else None
    use_ai = st.toggle("AI step for unknown layouts", value=False,
                       help="Sends unrecognised documents to the Anthropic API. Needs ANTHROPIC_API_KEY.")
    run = st.button("Extract", type="primary", use_container_width=True)


@st.cache_resource(show_spinner=False)
def _sample_results(ai: bool):
    return _process(sorted(SAMPLES.glob("*.pdf")), ai)


def _process(paths: list[Path], ai: bool):
    extractor = None
    if ai:
        from invoicelens.ai import ClaudeExtractor

        extractor = ClaudeExtractor()
    bar = st.progress(0.0, text="Starting")
    results = process_batch(paths, extractor, lambda n, t, name: bar.progress(n / t, text=f"{n}/{t} {name}"))
    bar.empty()
    return results


if run:
    if source.startswith("Sample"):
        st.session_state.results = _sample_results(use_ai)
        st.session_state.paths = {p.name: p for p in SAMPLES.glob("*.pdf")}
        st.session_state.is_sample = True
    elif uploads:
        tmp = Path(tempfile.mkdtemp(prefix="invoicelens_"))
        paths = []
        for f in uploads:
            p = tmp / Path(f.name).name
            p.write_bytes(f.getvalue())
            paths.append(p)
        st.session_state.results = _process(paths, use_ai)
        st.session_state.paths = {p.name: p for p in paths}
        st.session_state.is_sample = False
    else:
        st.warning("Upload at least one PDF.")

results = st.session_state.get("results")
if not results:
    st.info("Choose the sample set or upload PDFs, then press **Extract**.")
    st.stop()

inv_df = invoices_frame(results)
flagged = int(inv_df["needs_review"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(results))
c2.metric("Auto-accepted", len(results) - flagged)
c3.metric("Needs review", flagged)
c4.metric("Scanned (OCR)", int(inv_df["scanned"].sum()))

st.subheader("Extracted invoices")
show = inv_df[["file", "invoice_number", "invoice_date", "supplier_name", "currency", "total", "method"]].copy()
show.insert(1, "status", inv_df["needs_review"].map({True: "Review", False: "OK"}))
show["review_reason"] = inv_df["review_reason"]
show = show.astype(object).where(show.notna(), "—")


def _highlight(row):
    return ["background-color: #FFF4CE" if row["status"] == "Review" else "" for _ in row]


st.dataframe(
    show.style.apply(_highlight, axis=1).format({"total": lambda v: v if v == "—" else f"{v:,.2f}"}),
    use_container_width=True, hide_index=True, height=min(36 * (len(show) + 1) + 2, 520),
    column_config={"review_reason": st.column_config.TextColumn("review reason", width="large"),
                   "status": st.column_config.TextColumn("status", width="small")},
)

d1, d2, d3, _ = st.columns([1, 1, 1, 3])
with tempfile.TemporaryDirectory() as tmp:
    xlsx = write_excel(results, Path(tmp) / "invoices.xlsx").read_bytes()
d1.download_button("Excel", xlsx, "invoices.xlsx", use_container_width=True)
csv_zip = io.BytesIO()
with zipfile.ZipFile(csv_zip, "w") as z:
    z.writestr("invoices.csv", inv_df.to_csv(index=False))
    z.writestr("line_items.csv", lines_frame(results).to_csv(index=False))
d2.download_button("CSV (zip)", csv_zip.getvalue(), "invoices_csv.zip", use_container_width=True)
d3.download_button("JSON", json.dumps(results_json(results), indent=2, ensure_ascii=False),
                   "invoices.json", use_container_width=True)

st.subheader("Document detail")
by_name = {r.file: r for r in results}
default = next((r.file for r in results if r.needs_review and r.method != "none"), results[0].file)
choice = st.selectbox("Document", list(by_name), index=list(by_name).index(default))
r = by_name[choice]
left, right = st.columns([5, 6])
with left:
    with pymupdf.open(st.session_state.paths[choice]) as doc:
        st.image(render_page(doc[0], dpi=110), use_container_width=True)
with right:
    status = "Needs review" if r.needs_review else "Auto-accepted"
    conf = f" · OCR confidence {r.ocr_confidence:.0f}/100" if r.ocr_confidence is not None else ""
    st.markdown(f"**{status}** · method `{r.method}` · layout `{r.layout}`{conf}")
    for issue in r.issues:
        st.warning(f"{issue.code}: {issue.message}")
    def _fmt(name, v):
        if v is None:
            return "—"
        if name in ("subtotal", "tax_amount", "total"):
            return f"{v:,.2f}"
        return f"{v:g} %" if name == "tax_rate" else str(v)

    fields = pd.DataFrame({"field": HEADER_FIELDS, "value": [_fmt(f, getattr(r.invoice, f)) for f in HEADER_FIELDS]})
    st.dataframe(fields, hide_index=True, use_container_width=True, height=458)
    items = lines_frame([r]).drop(columns=["file", "invoice_number"])
    if len(items):
        st.dataframe(items.style.format({"quantity": "{:g}", "unit_price": "{:,.2f}", "amount": "{:,.2f}"}, na_rep="—"),
                     hide_index=True, use_container_width=True)

if st.session_state.get("is_sample") and TRUTH.exists():
    with st.expander("Accuracy against the synthetic ground truth"):
        _, summary = score(results, TRUTH, ai_enabled=use_ai)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Header field accuracy", f"{summary['header_field_accuracy']:.1%}")
        e2.metric("Silent errors", len(summary["silent_errors"]),
                  help="Accepted without a flag but containing a wrong value")
        e3.metric("Designed anomalies caught", f"{summary['designed_anomalies_caught']}/{summary['designed_anomalies']}")
        e4.metric("False alarms", len(summary["false_alarms"]))
        if summary["silent_errors"]:
            st.caption("Silent errors: " + ", ".join(summary["silent_errors"]))
