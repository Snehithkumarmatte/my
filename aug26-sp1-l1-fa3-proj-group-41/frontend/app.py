"""AI Invoice Auditor - human-in-the-loop dashboard (Streamlit, container port 8501).

Reads:  the `audit` schema in PostgreSQL, read-only (never the `erp` schema).
Writes: nothing directly. Uploads, corrections, re-validation and questions go to the
        agent over HTTP (AGENT_URL = http://agent:8000), the only service that writes data.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get("DATABASE_URL")
AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8000").rstrip("/")

st.set_page_config(page_title="AI Invoice Auditor", page_icon="🧾", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    .kpi { background: #fff; border: 1px solid #dfe8f3; border-radius: 14px; padding: .9rem 1rem; }
    .kpi .label { color: #5b6b7f; font-size: .85rem; }
    .kpi .value { font-size: 1.9rem; font-weight: 700; color: #1f2937; }
    .badge { display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-weight: 700; font-size: .85rem; }
    .badge-approve { background: #e7f8ef; color: #0c7a45; }
    .badge-manual_review { background: #fff0d9; color: #9a5200; }
    .badge-reject { background: #ffe0e3; color: #b42318; }
    .badge-pending { background: #edf2f7; color: #475467; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATABASE_URL:
    st.error("DATABASE_URL is not set - check the .env file used by docker compose.")
    st.stop()

REC_LABEL = {"approve": "✅ Approve", "manual_review": "🟠 Manual review", "reject": "⛔ Reject",
             "pending": "⏳ Pending"}
# correctable fields -> how to show the current value (must match the agent's HEADER_FIELDS/LINE_FIELDS)
HEADER_FIELDS = {"invoice_no": "Invoice number", "invoice_date": "Invoice date", "vendor_name": "Vendor name",
                 "vendor_id": "Vendor ID", "po_number": "PO number", "currency": "Currency",
                 "subtotal": "Subtotal", "tax_amount": "Tax amount", "total_amount": "Total amount"}
LINE_FIELDS = {"item_code": "Item code", "description": "Description", "qty": "Quantity",
               "unit_price": "Unit price", "total": "Line total"}


# ---------------------------------------------------------------- data access

def q(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5) as conn:
        conn.read_only = True
        return conn.execute(sql, params).fetchall()


def agent(method: str, path: str, quiet: bool = False, **kw):
    try:
        r = httpx.request(method, f"{AGENT_URL}{path}", timeout=600, **kw)
    except httpx.HTTPError as exc:
        if not quiet:
            st.warning(f"Agent service unreachable: {exc}")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.warning(f"Agent returned {r.status_code}: {detail}")
        return None
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text}


def invoices_df() -> pd.DataFrame:
    return pd.DataFrame(q(
        """SELECT invoice_id::text AS invoice_id, invoice_number, vendor_name, po_number, invoice_date,
                  currency, total_amount, source_language, translation_confidence,
                  validation_status, recommendation, extraction_method, processed_at,
                  regexp_replace(file_path, '^.*/', '') AS file_name,
                  (SELECT count(*) FROM audit.discrepancies d WHERE d.invoice_id = a.invoice_id) AS discrepancies,
                  (SELECT count(*) FROM audit.human_feedback f WHERE f.invoice_id = a.invoice_id) AS corrections
             FROM audit.invoice_audit a ORDER BY processed_at DESC"""))


def badge(value) -> str:
    key = str(value or "pending").lower()
    return f"<span class='badge badge-{key}'>{REC_LABEL.get(key, key.replace('_', ' ').title())}</span>"


def kpi(col, label: str, value) -> None:
    col.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
                 unsafe_allow_html=True)


def invoice_label(row) -> str:
    return f"{row['invoice_number'] or '(no number)'} · {row['vendor_name'] or row['file_name']} · {row['file_name']}"


# ---------------------------------------------------------------- sidebar

st.sidebar.title("🧾 Invoice Auditor")
nav = st.sidebar.radio("Navigation", ["Overview", "Review & correct", "Ask", "Upload", "System status"])
st.sidebar.divider()
if st.sidebar.button("📥 Check incoming folder now", use_container_width=True):
    with st.spinner("Processing new invoices…"):
        res = agent("POST", "/process")
    if res is not None:
        if res.get("status") == "busy":
            st.sidebar.info("A run is already in progress - new files will be picked up.")
        elif res.get("status") == "idle":
            st.sidebar.info("No new invoices in the incoming folder.")
        else:
            st.sidebar.success(f"{len(res.get('processed', []))} processed, {len(res.get('failed', []))} failed")
st.sidebar.caption(f"Agent: {AGENT_URL}")


# ---------------------------------------------------------------- pages

def page_overview() -> None:
    st.title("Overview")
    df = invoices_df()
    if df.empty:
        st.info("No invoices yet. Drop files into the incoming folder or use **Upload**.")
        return
    counts = df["recommendation"].value_counts()
    translated = df[df["source_language"].fillna("en") != "en"]["translation_confidence"].dropna()
    cols = st.columns(5)
    kpi(cols[0], "Invoices audited", len(df))
    kpi(cols[1], "Approve", int(counts.get("approve", 0)))
    kpi(cols[2], "Manual review", int(counts.get("manual_review", 0)))
    kpi(cols[3], "Reject", int(counts.get("reject", 0)))
    kpi(cols[4], "Avg translation confidence", f"{float(translated.astype(float).mean()):.2f}" if len(translated) else "—")

    st.subheader("Invoices")
    view = df[["invoice_number", "vendor_name", "po_number", "invoice_date", "currency", "total_amount",
               "source_language", "validation_status", "recommendation", "discrepancies", "corrections",
               "file_name"]].rename(columns={
        "invoice_number": "Invoice", "vendor_name": "Vendor", "po_number": "PO", "invoice_date": "Date",
        "currency": "Cur", "total_amount": "Total", "source_language": "Lang", "validation_status": "Status",
        "recommendation": "Recommendation", "discrepancies": "Issues", "corrections": "Corrections",
        "file_name": "File"})
    view["Recommendation"] = view["Recommendation"].map(lambda v: REC_LABEL.get(v, v))
    st.dataframe(view, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recommendations")
        st.bar_chart(counts.rename(index=lambda k: REC_LABEL.get(k, k)))
    with c2:
        st.subheader("Most frequent discrepancy fields")
        fields = q("""SELECT field_name, count(*) AS n FROM audit.discrepancies
                      GROUP BY field_name ORDER BY n DESC LIMIT 8""")
        if fields:
            st.bar_chart(pd.DataFrame(fields).set_index("field_name")["n"])
        else:
            st.caption("No discrepancies recorded.")


def _effective_values(inv: dict) -> tuple[dict, list[dict]]:
    """Values the agent would validate now: the stored extraction with corrections applied."""
    raw = inv.get("extracted_raw") or {}
    stored = (raw.get("revalidation") or {}).get("effective_translation_result") or raw.get("translation_result") or {}
    invoice = stored.get("translated_invoice") or raw.get("extracted_invoice") or {}
    header = {f: invoice.get("translated_vendor_name" if f == "vendor_name" else f) for f in HEADER_FIELDS}
    if header.get("vendor_name") is None:
        header["vendor_name"] = invoice.get("vendor_name")
    lines = []
    for line in invoice.get("line_items") or []:
        lines.append({f: line.get("translated_description", line.get("description")) if f == "description"
                      else line.get(f) for f in LINE_FIELDS})
    return header, lines


def page_review() -> None:
    st.title("Review & correct")
    df = invoices_df()
    if df.empty:
        st.info("No invoices are available for review yet.")
        return

    only_open = st.toggle("Show only invoices that need attention", value=False)
    pick = df[df["recommendation"] != "approve"] if only_open else df
    if pick.empty:
        st.success("Nothing needs attention.")
        return
    options = pick["invoice_id"].tolist()
    labels = {r["invoice_id"]: invoice_label(r) for _, r in pick.iterrows()}
    invoice_id = st.selectbox("Invoice", options, format_func=labels.get)

    inv = q("SELECT * FROM audit.invoice_audit WHERE invoice_id = %s", (invoice_id,))[0]
    raw = inv.get("extracted_raw") or {}
    report = (raw.get("revalidation") or {}).get("report") or raw.get("report") or {}
    email = raw.get("email") or {}

    st.markdown(
        f"### {inv['invoice_number'] or '(no invoice number)'} &nbsp; {badge(inv['recommendation'])}",
        unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Validation status", inv["validation_status"])
    cols[1].metric("Total", f"{inv['currency'] or ''} {inv['total_amount'] if inv['total_amount'] is not None else '—'}")
    cols[2].metric("Language", inv["source_language"] or "—")
    conf = inv["translation_confidence"]
    cols[3].metric("Translation confidence", f"{float(conf):.2f}" if conf is not None else "—")
    if email:
        st.caption(f"✉️ From {email.get('sender')} · “{email.get('subject')}” · received {email.get('received_timestamp')}")
    if inv.get("error_message"):
        st.warning(f"Processing notes: {inv['error_message']}")
    for reason in report.get("recommendation_reasons") or []:
        st.markdown(f"- {reason}")

    tabs = st.tabs(["Discrepancies", "Extracted data", "Original text / translation", "Report", "Corrections"])

    with tabs[0]:
        discs = q("""SELECT check_type AS "check", line_number AS line, item_code, field_name AS field,
                            invoice_value, erp_value AS "expected / ERP", deviation_pct AS "deviation_pct", severity
                       FROM audit.discrepancies WHERE invoice_id = %s ORDER BY id""", (invoice_id,))
        if discs:
            st.dataframe(pd.DataFrame(discs), hide_index=True, use_container_width=True)
        else:
            st.success("No discrepancies recorded.")

    with tabs[1]:
        header = {k: inv[c] for k, c in [("Invoice number", "invoice_number"), ("Date", "invoice_date"),
                                         ("Vendor", "vendor_name"), ("Vendor ID", "vendor_id"),
                                         ("PO", "po_number"), ("Currency", "currency"),
                                         ("Total", "total_amount"), ("Extraction method", "extraction_method"),
                                         ("File", "file_path"), ("Thread id", "thread_id")]}
        st.dataframe(pd.DataFrame([{"Field": k, "Value": "" if v is None else str(v)} for k, v in header.items()]),
                     hide_index=True, use_container_width=True)
        lines = q("""SELECT line_number AS "#", item_code, description, quantity, unit_price, line_total
                       FROM audit.invoice_line_items WHERE invoice_id = %s ORDER BY line_number""", (invoice_id,))
        st.markdown("**Line items**")
        if lines:
            st.dataframe(pd.DataFrame(lines), hide_index=True, use_container_width=True)
        else:
            st.caption("No line items extracted.")
        st.caption("These are the values as originally extracted. Corrections never overwrite them.")

    with tabs[2]:
        left, right = st.columns(2)
        left.text_area("Raw extracted text (pre-translation)", inv.get("raw_extracted_text") or "", height=360)
        right.text_area("English translation", inv.get("translated_text") or "(English invoice - not translated)",
                        height=360)

    with tabs[3]:
        try:
            r = httpx.get(f"{AGENT_URL}/invoices/{invoice_id}/report.html", timeout=30)
        except httpx.HTTPError as exc:
            r = None
            st.warning(f"Agent unreachable: {exc}")
        if r is not None and r.status_code == 200:
            st.components.v1.html(r.text, height=720, scrolling=True)
            st.link_button("Open JSON report", f"{AGENT_URL}/invoices/{invoice_id}/report.json")
        elif r is not None:
            st.info("No report file is available for this invoice.")

    with tabs[4]:
        header_now, lines_now = _effective_values(inv)
        st.markdown("Fix a wrongly extracted value, then re-validate. The correction is stored as a new "
                    "row in `human_feedback`; the original extraction stays as it was.")
        scope = st.radio("What do you want to correct?", ["Invoice header", "A line item"], horizontal=True)
        if scope == "Invoice header":
            field = st.selectbox("Field", list(HEADER_FIELDS), format_func=HEADER_FIELDS.get)
            current = header_now.get(field)
            field_name = field
        else:
            if not lines_now:
                st.info("This invoice has no extracted line items to correct.")
                field_name, current = None, None
            else:
                number = st.selectbox("Line", list(range(1, len(lines_now) + 1)),
                                      format_func=lambda n: f"{n}: {lines_now[n - 1].get('item_code') or ''} "
                                                            f"{lines_now[n - 1].get('description') or ''}")
                sub = st.selectbox("Field", list(LINE_FIELDS), format_func=LINE_FIELDS.get)
                current = lines_now[number - 1].get(sub)
                field_name = f"line:{number}:{sub}"
        if field_name:
            st.caption(f"Current value: **{'—' if current in (None, '') else current}**")
            with st.form("correction", clear_on_submit=True):
                value = st.text_input("Corrected value")
                who = st.text_input("Your name", value=st.session_state.get("auditor", ""))
                reval = st.checkbox("Re-validate immediately", value=True)
                if st.form_submit_button("Save correction", type="primary"):
                    if not value.strip() or not who.strip():
                        st.error("Enter both the corrected value and your name.")
                    else:
                        st.session_state["auditor"] = who.strip()
                        with st.spinner("Saving…"):
                            res = agent("POST", f"/invoices/{invoice_id}/corrections",
                                        json={"field_name": field_name, "corrected_value": value.strip(),
                                              "corrected_by": who.strip(), "revalidate": reval})
                        if res:
                            rv = res.get("revalidation")
                            if rv:
                                st.success(f"Saved. New recommendation: {REC_LABEL.get(rv['recommendation'], rv['recommendation'])}")
                            else:
                                st.success("Correction saved.")
                            if res.get("revalidation_error"):
                                st.warning(res["revalidation_error"])

        if st.button("🔁 Re-validate with all corrections"):
            with st.spinner("Re-validating against the ERP…"):
                res = agent("POST", f"/invoices/{invoice_id}/revalidate")
            if res:
                st.success(f"Recommendation: {REC_LABEL.get(res.get('recommendation'), res.get('recommendation'))} "
                           f"({res.get('corrections_applied')} correction(s) applied)")

        history = q("""SELECT corrected_at, field_name, original_value, corrected_value, corrected_by, revalidated
                         FROM audit.human_feedback WHERE invoice_id = %s ORDER BY corrected_at DESC""", (invoice_id,))
        st.markdown("**Correction history**")
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
        else:
            st.caption("No corrections yet.")


def page_ask() -> None:
    st.title("Ask about the invoices")
    st.caption("Counts, totals and statuses are answered exactly from the audit database. "
               "Open questions (“why was … flagged?”) use retrieval over invoices and reports, "
               "checked by a reflection step before being shown.")
    df = invoices_df()
    scope_opts = ["All invoices"] + (df["invoice_id"].tolist() if not df.empty else [])
    labels = {r["invoice_id"]: invoice_label(r) for _, r in df.iterrows()} if not df.empty else {}
    scope = st.selectbox("Limit to one invoice (optional)", scope_opts,
                         format_func=lambda v: v if v == "All invoices" else labels.get(v, v))

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("session_id", os.urandom(6).hex())
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    examples = ["How many invoices were rejected?", "What is the total amount for HafenLogistik?",
                "Why was INV-1001 flagged?", "Which invoices need manual review?"]
    st.caption("Try: " + " · ".join(examples))
    if prompt := st.chat_input("Ask a question about the processed invoices"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                body = {"question": prompt, "session_id": st.session_state.session_id}
                if scope != "All invoices":
                    body["invoice_id"] = scope
                res = agent("POST", "/query", json=body)
            if res:
                answer = res.get("answer") or "No answer."
                if res.get("route") == "sql":
                    meta = "Answered from PostgreSQL (SQL)"
                else:
                    ref = res.get("reflection") or {}
                    meta = (f"RAG · reflection score {float(ref.get('overall') or 0):.2f} · "
                            f"{'accepted' if res.get('accepted') else 'not accepted'} · "
                            f"retries {res.get('retry_count', 0)} · sources: {', '.join(res.get('sources_used') or []) or '—'}")
                st.markdown(answer)
                st.caption(meta)
                if res.get("sql"):
                    with st.expander("SQL used"):
                        st.code(res["sql"], language="sql")
                st.session_state.chat.append({"role": "assistant", "content": answer, "meta": meta})
            else:
                st.markdown("I could not answer that right now.")
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.session_state.session_id = os.urandom(6).hex()
        st.rerun()


def page_upload() -> None:
    st.title("Upload invoices")
    st.caption("Files are handed to the agent, which saves them in the monitored incoming folder and audits them.")
    uploaded = st.file_uploader("Invoice files (PDF, DOCX, PNG)", accept_multiple_files=True,
                                type=["pdf", "docx", "png"])
    c1, c2 = st.columns(2)
    sender = c1.text_input("Sender e-mail (optional)")
    subject = c2.text_input("Subject (optional)")
    if st.button("Upload & audit", type="primary", disabled=not uploaded):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded]
        data = {k: v for k, v in {"sender": sender, "subject": subject}.items() if v}
        with st.spinner("Uploading and auditing - OCR and translation can take a minute…"):
            res = agent("POST", "/upload", files=files, data=data)
        if res:
            for item in res.get("results", []):
                status = item.get("status")
                text = f"**{item['file']}** - {status}"
                if item.get("recommendation"):
                    text += f" &nbsp; {badge(item['recommendation'])}"
                if item.get("error"):
                    text += f"<br><small>{item['error']}</small>"
                st.markdown(text, unsafe_allow_html=True)


def page_status() -> None:
    st.title("System status")
    health = agent("GET", "/health", quiet=True)
    if health and health.get("status") == "healthy":
        st.success("Agent healthy - " + ", ".join(f"{k}: {v}" for k, v in health.items() if k != "status"))
    else:
        st.error("Agent is unavailable or a dependency is down. The dashboard is read-only until it recovers.")
    stats = agent("GET", "/stats", quiet=True)
    if stats:
        cols = st.columns(4)
        kpi(cols[0], "Audit rows (PostgreSQL)", stats.get("invoices", 0))
        kpi(cols[1], "Discrepancies", stats.get("discrepancies", 0))
        kpi(cols[2], "Human corrections", stats.get("corrections", 0))
        kpi(cols[3], "Vector chunks (Qdrant)", stats.get("indexed_chunks") if stats.get("indexed_chunks") is not None else "—")
        st.caption(f"Last processed: {stats.get('last_processed_at') or '—'}")
        if stats.get("last_run"):
            with st.expander("Last pipeline run"):
                st.json(stats["last_run"])
    if st.button("Rebuild vector index from PostgreSQL"):
        with st.spinner("Re-chunking and re-embedding stored text (no OCR, no file re-read)…"):
            res = agent("POST", "/reindex", json={"force": True})
        if res:
            st.success(f"Indexed {res.get('chunks_indexed')} chunks from {res.get('invoices_indexed')} invoice(s).")


{"Overview": page_overview, "Review & correct": page_review, "Ask": page_ask,
 "Upload": page_upload, "System status": page_status}[nav]()
