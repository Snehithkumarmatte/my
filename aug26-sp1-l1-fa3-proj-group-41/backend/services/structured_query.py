"""Structured questions are answered straight from PostgreSQL (spec 5.1).

"How many invoices were rejected?", "total amount for HafenLogistik", "status of
INV-1001" have exact answers in the audit schema, so they never touch Qdrant or the
LLM. Only open-ended questions ("why was ... flagged?") go through the RAG pipeline.

Every query here is a fixed, parameterised SELECT on the audit schema - user text is
never spliced into SQL - and the ERP schema is never read (that is mock-erp's job).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from backend.db import repository

_OPEN_ENDED = re.compile(r"\b(why|explain|how come|reason|describe|summari[sz]e|what happened|compare)\b", re.I)
_DOC_REF = re.compile(r"\b([A-Z]{2,4}-[A-Z0-9]{1,6}(?:-[A-Z0-9]{1,6})?)\b", re.I)

_REC_WORDS = {
    "reject": ("reject", "rejected", "rejection"),
    "approve": ("approve", "approved", "approval", "auto-approved"),
    "manual_review": ("manual review", "manual_review", "review", "needs review", "flagged for review"),
}
_STATUS_WORDS = {"failed": ("failed", "failing"), "passed": ("passed",), "partial": ("partial", "partially"),
                 "error": ("error", "errored", "could not be processed")}

_SELECT = """SELECT invoice_id::text AS invoice_id, invoice_number, vendor_name, po_number, invoice_date,
                    currency, total_amount, source_language, translation_confidence,
                    validation_status, recommendation,
                    regexp_replace(file_path, '^.*/', '') AS file_name
               FROM audit.invoice_audit"""


@dataclass
class StructuredAnswer:
    answer: str
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with repository.pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def _fmt_money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{Decimal(value):,.2f}"


def _label(row: dict) -> str:
    return row.get("invoice_number") or row.get("file_name") or row.get("invoice_id")


def _match_word(question: str, table: dict[str, tuple[str, ...]]) -> str | None:
    q = question.lower()
    for key, words in table.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            return key
    return None


def _match_vendor(question: str) -> str | None:
    """A vendor already in the audit trail whose name (or first word) appears in the question."""
    q = question.lower()
    names = [r["vendor_name"] for r in _rows(
        "SELECT DISTINCT vendor_name FROM audit.invoice_audit WHERE vendor_name IS NOT NULL")]
    for name in sorted(names, key=len, reverse=True):
        first = re.split(r"[\s,]+", name.strip())[0].lower()
        if name.lower() in q or (len(first) >= 4 and re.search(rf"\b{re.escape(first)}\b", q)):
            return name
    return None


def _list(rows: list[dict], limit: int = 15) -> str:
    out = [f"- {_label(r)} | {r.get('vendor_name') or 'unknown vendor'} | "
           f"{r.get('currency') or ''} {_fmt_money(r.get('total_amount'))} | "
           f"{r.get('validation_status')} / {r.get('recommendation')}" for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"... and {len(rows) - limit} more")
    return "\n".join(out)


def try_answer(question: str, invoice_id: str | None = None) -> StructuredAnswer | None:
    """Return an exact answer from SQL, or None when the question is open-ended."""
    q = question.strip()
    ql = q.lower()
    if _OPEN_ENDED.search(q):
        return None

    rec = _match_word(q, _REC_WORDS)
    status = _match_word(q, _STATUS_WORDS)
    vendor = _match_vendor(q)

    # one specific invoice / PO: "status of INV-1001", "total of PO-1004"
    refs = [r.upper() for r in _DOC_REF.findall(q)
            if any(ch.isdigit() for ch in r) and not r.upper().startswith(("SKU-", "VEND-"))]
    if refs or invoice_id:
        if invoice_id:
            sql = _SELECT + " WHERE invoice_id = %s"
            rows = _rows(sql, (invoice_id,))
        else:
            sql = _SELECT + " WHERE upper(invoice_number) = ANY(%s) OR upper(po_number) = ANY(%s)"
            rows = _rows(sql, (refs, refs))
        if not rows:
            return StructuredAnswer(f"No processed invoice matches {', '.join(refs)}.", sql) if refs else None
        if not re.search(r"\b(status|total|amount|recommend|currency|date|vendor|po|language|confidence|"
                         r"approved|rejected|value|due)\b", ql):
            return None                                   # e.g. "what is on INV-1001" -> RAG
        details = []
        for r in rows:
            details.append(
                f"Invoice {_label(r)} ({r['file_name']}): vendor {r.get('vendor_name') or 'n/a'}, "
                f"PO {r.get('po_number') or 'n/a'}, date {r.get('invoice_date') or 'n/a'}, "
                f"total {r.get('currency') or ''} {_fmt_money(r.get('total_amount'))}, "
                f"language {r.get('source_language') or 'n/a'} "
                f"(translation confidence {r.get('translation_confidence') if r.get('translation_confidence') is not None else 'n/a'}), "
                f"status {r['validation_status']}, recommendation {r['recommendation']}."
            )
        return StructuredAnswer("\n".join(details), sql, rows)

    where, params, what = [], [], []
    if rec:
        where.append("recommendation = %s"); params.append(rec)
        what.append({"reject": "recommended for rejection", "approve": "recommended for approval",
                     "manual_review": "flagged for manual review"}[rec])
    if status:
        where.append("validation_status = %s"); params.append(status); what.append(f"with validation status '{status}'")
    if vendor:
        where.append("vendor_name = %s"); params.append(vendor); what.append(f"from {vendor}")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    scope = " ".join(what) if what else "in the audit trail"

    # sums: "total amount for HafenLogistik", "total value of approved invoices"
    if re.search(r"\b(total|sum|value|spend|spent|amount)\b", ql) and re.search(
            r"\b(total|sum|how much|value|spend|spent)\b", ql):
        sql = ("SELECT currency, count(*) AS n, sum(total_amount) AS total FROM audit.invoice_audit"
               + clause + " GROUP BY currency ORDER BY currency NULLS LAST")
        rows = _rows(sql, tuple(params))
        if not rows:
            return StructuredAnswer(f"There are no invoices {scope}.", sql)
        parts = [f"{r['currency'] or 'no currency'} {_fmt_money(r['total'])} across {r['n']} invoice(s)"
                 for r in rows]
        return StructuredAnswer(f"Total invoice value {scope}: " + "; ".join(parts) + ".", sql, rows)

    # counts: "how many invoices were rejected?"
    if re.search(r"\b(how many|number of|count)\b", ql):
        if re.search(r"\bdiscrepanc", ql):
            sql = ("SELECT count(*) AS n FROM audit.discrepancies d JOIN audit.invoice_audit a USING (invoice_id)"
                   + clause.replace("recommendation", "a.recommendation").replace("vendor_name", "a.vendor_name")
                   .replace("validation_status", "a.validation_status"))
            n = _rows(sql, tuple(params))[0]["n"]
            return StructuredAnswer(f"There are {n} recorded discrepancies for invoices {scope}.", sql)
        sql = "SELECT count(*) AS n FROM audit.invoice_audit" + clause
        n = _rows(sql, tuple(params))[0]["n"]
        if not where:
            breakdown = _rows("SELECT recommendation, count(*) AS n FROM audit.invoice_audit "
                              "GROUP BY recommendation ORDER BY recommendation")
            extra = ", ".join(f"{r['n']} {r['recommendation'].replace('_', ' ')}" for r in breakdown)
            return StructuredAnswer(f"{n} invoices have been processed ({extra}).", sql, breakdown)
        return StructuredAnswer(f"{n} invoice(s) {scope}.", sql)

    # averages: "average translation confidence"
    if re.search(r"\b(average|avg|mean)\b", ql) and "confidence" in ql:
        sql = ("SELECT avg(translation_confidence) AS a, count(*) AS n FROM audit.invoice_audit"
               + (clause + " AND" if clause else " WHERE") + " source_language IS DISTINCT FROM 'en'")
        r = _rows(sql, tuple(params))[0]
        if not r["n"]:
            return StructuredAnswer("No translated (non-English) invoices have been processed yet.", sql)
        return StructuredAnswer(f"Average translation confidence over {r['n']} non-English invoice(s): "
                                f"{float(r['a'] or 0):.2f}.", sql)

    # lists: "which invoices were rejected?", "show invoices from SwiftMove"
    if where and re.search(r"\b(which|list|show|what|give|all|invoices)\b", ql):
        sql = _SELECT + clause + " ORDER BY processed_at DESC"
        rows = _rows(sql, tuple(params))
        if not rows:
            return StructuredAnswer(f"There are no invoices {scope}.", sql)
        return StructuredAnswer(f"{len(rows)} invoice(s) {scope}:\n" + _list(rows), sql, rows)

    return None
