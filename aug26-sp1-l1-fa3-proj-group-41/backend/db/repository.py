"""Audit-trail persistence (schema: audit).

The agent is the only service that writes application data. ERP tables are never
queried here: ERP data comes only from the mock-erp HTTP API.

The invoice file itself is never stored - only its path and SHA-256 checksum.
Human corrections are INSERTed into audit.human_feedback; the original extracted
values in audit.invoice_audit are never overwritten by a correction.
"""
from __future__ import annotations

import hashlib
import json
import atexit
import logging
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from backend.config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


@atexit.register
def _close_pool() -> None:
    if _pool is not None:
        _pool.close(timeout=1)

RECOMMENDATIONS = {"approve", "manual_review", "reject"}
SEVERITY = {"error": "high", "warning": "medium"}
EXTRACTION_METHODS = {"pdf": "pdfplumber", "docx": "docx", "image": "tesseract"}
DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y/%m/%d",
    "%Y.%m.%d", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%B %d %Y",
    "%b %d %Y", "%d %B, %Y",
)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True,
                               kwargs={"row_factory": dict_row})
    return _pool


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda obj: json.dumps(obj, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- monitor helpers

def checksum_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_audited(checksum: str) -> bool:
    """True once a file has a final outcome (anything but 'pending', including 'error')."""
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT validation_status FROM audit.invoice_audit WHERE file_checksum = %s",
            (checksum,),
        ).fetchone()
    return bool(row) and row["validation_status"] != "pending"


def record_discovered(path: str | Path) -> str:
    """Record a detected file (status 'pending'). Returns its invoice_id."""
    path = Path(path)
    checksum = checksum_file(path)
    with pool().connection() as conn:
        row = conn.execute(
            """INSERT INTO audit.invoice_audit (invoice_id, file_path, file_checksum)
               VALUES (%s, %s, %s)
               ON CONFLICT (file_checksum) DO UPDATE SET file_path = EXCLUDED.file_path
               RETURNING invoice_id""",
            (uuid.uuid4(), str(path), checksum),
        ).fetchone()
    return str(row["invoice_id"])


# ---------------------------------------------------------------- value helpers

def _dec(value: Any, limit: Decimal | None = None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    if limit is not None and abs(number) > limit:
        number = limit if number > 0 else -limit
    return number


def _date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _currency(value: Any) -> str | None:
    if isinstance(value, str) and len(value.strip()) == 3 and value.strip().isalpha():
        return value.strip().upper()
    return None


def _text(value: Any, max_len: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:max_len] if max_len else text


def _invoice_lines(translation: dict | None, extraction: dict) -> list[dict]:
    """Line items in English (translated) when available, else as extracted."""
    t_invoice = (translation or {}).get("translated_invoice") or {}
    if t_invoice.get("line_items"):
        return [{
            "item_code": line.get("item_code"),
            "description": line.get("translated_description") or line.get("original_description"),
            "qty": line.get("qty"), "unit_price": line.get("unit_price"), "total": line.get("total"),
        } for line in t_invoice["line_items"]]
    e_invoice = extraction.get("extracted_invoice") or {}
    return [{
        "item_code": line.get("item_code"), "description": line.get("description"),
        "qty": line.get("qty"), "unit_price": line.get("unit_price"), "total": line.get("total"),
    } for line in e_invoice.get("line_items") or []]


def render_invoice_text(invoice: dict, lines: list[dict]) -> str:
    """Readable English rendering of a translated invoice (stored as translated_text)."""
    out = [
        f"Invoice number: {invoice.get('invoice_no') or 'N/A'}",
        f"Invoice date: {invoice.get('invoice_date') or 'N/A'}",
        f"Vendor: {invoice.get('translated_vendor_name') or invoice.get('vendor_name') or 'N/A'}"
        f" (vendor id {invoice.get('vendor_id') or 'N/A'})",
        f"Purchase order: {invoice.get('po_number') or 'N/A'}",
        f"Currency: {invoice.get('currency') or 'N/A'}",
        f"Subtotal: {invoice.get('subtotal')}  Tax: {invoice.get('tax_amount')}  "
        f"Total: {invoice.get('total_amount')}",
        "Line items:",
    ]
    for n, line in enumerate(lines, 1):
        out.append(f"  {n}. {line.get('item_code') or '-'} | {line.get('description') or '-'} | "
                   f"qty {line.get('qty')} x unit price {line.get('unit_price')} = {line.get('total')}")
    extras = invoice.get("translated_additional_fields") or invoice.get("additional_fields") or {}
    for key, value in extras.items():
        out.append(f"{key}: {value}")
    return "\n".join(out)


def overall_status(validation: dict | None, business: dict | None, report: dict | None = None) -> str:
    """Status from the rules.yaml decision rule when the report has one, else derived."""
    if (report or {}).get("validation_status") in ("passed", "partial", "failed"):
        return report["validation_status"]
    v_status = (validation or {}).get("validation_status")
    b_status = (business or {}).get("validation_status")
    if v_status is None and b_status is None:
        return "error"
    if v_status == "failed" or b_status == "po_not_found":
        return "failed"
    if v_status == "passed" and b_status == "matched":
        return "passed"
    return "partial"


def _discrepancy_rows(invoice_id: str, validation: dict | None,
                      business: dict | None) -> list[tuple]:
    rows: list[tuple] = []
    validation = validation or {}
    for issue in validation.get("header_issues") or []:
        rows.append((invoice_id, "data", None, None, _text(issue.get("field_name"), 64) or "unknown",
                     _text(issue.get("actual")), _text(issue.get("expected")), None,
                     SEVERITY.get(issue.get("severity"), "medium")))
    for issue in validation.get("line_item_issues") or []:
        idx = issue.get("line_index")
        rows.append((invoice_id, "data", idx + 1 if isinstance(idx, int) and idx >= 0 else None, None,
                     _text(issue.get("field_name"), 64) or "unknown",
                     _text(issue.get("actual")), _text(issue.get("expected")), None,
                     SEVERITY.get(issue.get("severity"), "medium")))
    for d in (business or {}).get("discrepancies") or []:
        idx = d.get("line_index")
        rows.append((invoice_id, "business", idx + 1 if isinstance(idx, int) and idx >= 0 else None,
                     _text(d.get("item_code"), 64), _text(d.get("field_name"), 64) or "unknown",
                     _text(d.get("invoice_value")), _text(d.get("erp_value")),
                     _dec(d.get("difference_pct"), Decimal("9999.9999")),
                     SEVERITY.get(d.get("severity"), "medium")))
    return rows


def _replace_discrepancies(conn, invoice_id: str, validation: dict | None, business: dict | None) -> int:
    conn.execute("DELETE FROM audit.discrepancies WHERE invoice_id = %s", (invoice_id,))
    rows = _discrepancy_rows(invoice_id, validation, business)
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO audit.discrepancies
                   (invoice_id, check_type, line_number, item_code, field_name,
                    invoice_value, erp_value, deviation_pct, severity)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows,
            )
    return len(rows)


def _report_summary(report: dict | None) -> dict:
    report = report or {}
    return {
        "recommendation": report.get("recommendation"),
        "recommendation_reasons": report.get("recommendation_reasons") or [],
        "report_html_path": report.get("report_html_path"),
        "report_json_path": report.get("report_json_path"),
    }


def _email_meta(file_path: str) -> dict | None:
    """Simulated email header (.meta.json) kept alongside the extraction as variable extras."""
    from backend.tools.invoice_watcher_tool import find_email_meta
    try:
        return find_email_meta(file_path)[0]
    except Exception:  # noqa: BLE001 - the header is optional context
        return None


# ---------------------------------------------------------------- pipeline writes

def save_invoice_result(file_path: str, extraction: dict, translation: dict | None,
                        validation: dict | None, business: dict | None, report: dict | None,
                        thread_id: str | None) -> str:
    """Write the full outcome for one invoice in a single transaction."""
    invoice_id = record_discovered(file_path)
    e_invoice = extraction.get("extracted_invoice") or {}
    t_invoice = (translation or {}).get("translated_invoice") or {}
    header = t_invoice or {**e_invoice, "translated_vendor_name": e_invoice.get("vendor_name")}
    lines = _invoice_lines(translation, extraction)
    was_translated = bool((translation or {}).get("was_translated"))

    recommendation = (report or {}).get("recommendation")
    if recommendation not in RECOMMENDATIONS:
        recommendation = "manual_review"

    errors = list(extraction.get("extraction_errors") or []) + \
        list((translation or {}).get("translation_errors") or [])

    extracted_raw = {
        "extracted_invoice": e_invoice,
        "translation_result": translation,
        "validation_result": validation,
        "business_validation_result": business,
        "report": _report_summary(report),
        "invoice_date_text": header.get("invoice_date"),
        "email": _email_meta(file_path),
    }

    with pool().connection() as conn:
        conn.execute(
            """UPDATE audit.invoice_audit SET
                 extraction_method = %s, raw_extracted_text = %s, translated_text = %s,
                 source_language = %s, translation_confidence = %s,
                 invoice_number = %s, vendor_name = %s, vendor_id = %s, po_number = %s,
                 invoice_date = %s, currency = %s, total_amount = %s,
                 validation_status = %s, recommendation = %s, error_message = %s,
                 extracted_raw = %s, thread_id = %s, processed_at = now()
               WHERE invoice_id = %s""",
            (
                EXTRACTION_METHODS.get(extraction.get("file_format"), "failed"),
                extraction.get("raw_text") or None,
                render_invoice_text(t_invoice, lines) if was_translated else None,
                _text((translation or {}).get("source_language") or e_invoice.get("detected_language"), 8),
                _dec((translation or {}).get("translation_confidence")),
                _text(header.get("invoice_no"), 64),
                _text(header.get("translated_vendor_name") or header.get("vendor_name"), 255),
                _text(header.get("vendor_id"), 64),
                _text(header.get("po_number"), 64),
                _date(header.get("invoice_date")),
                _currency(header.get("currency")),
                _dec(header.get("total_amount")),
                overall_status(validation, business, report),
                recommendation,
                "; ".join(errors) or None,
                _jsonb(extracted_raw),
                thread_id,
                invoice_id,
            ),
        )
        conn.execute("DELETE FROM audit.invoice_line_items WHERE invoice_id = %s", (invoice_id,))
        if lines:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO audit.invoice_line_items
                       (invoice_id, line_number, item_code, description, quantity, unit_price, line_total)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    [(invoice_id, n, _text(line.get("item_code"), 64), _text(line.get("description")),
                      _dec(line.get("qty")), _dec(line.get("unit_price")), _dec(line.get("total")))
                     for n, line in enumerate(lines, 1)],
                )
        _replace_discrepancies(conn, invoice_id, validation, business)
    return invoice_id


def mark_failed(file_path: str, message: str, extraction: dict | None,
                thread_id: str | None) -> str:
    """Record a file that could not be processed (it stays in the audit trail)."""
    invoice_id = record_discovered(file_path)
    extraction = extraction or {}
    with pool().connection() as conn:
        conn.execute(
            """UPDATE audit.invoice_audit SET
                 extraction_method = %s, raw_extracted_text = %s,
                 validation_status = 'error', recommendation = 'manual_review',
                 error_message = %s, extracted_raw = %s, thread_id = %s, processed_at = now()
               WHERE invoice_id = %s""",
            (EXTRACTION_METHODS.get(extraction.get("file_format"), "failed"),
             extraction.get("raw_text") or None, message[:4000],
             _jsonb({"extraction": extraction, "email": _email_meta(file_path)}), thread_id, invoice_id),
        )
    return invoice_id


# ---------------------------------------------------------------- human-in-the-loop

def get_invoice(invoice_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM audit.invoice_audit WHERE invoice_id = %s",
                            (invoice_id,)).fetchone()


def get_discrepancies(invoice_id: str) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT * FROM audit.discrepancies WHERE invoice_id = %s ORDER BY id", (invoice_id,)
        ).fetchall()


def get_corrections(invoice_id: str) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT * FROM audit.human_feedback WHERE invoice_id = %s ORDER BY corrected_at, id",
            (invoice_id,),
        ).fetchall()


def add_correction(invoice_id: str, field_name: str, original_value: str | None,
                   corrected_value: str, corrected_by: str) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            """INSERT INTO audit.human_feedback
               (invoice_id, field_name, original_value, corrected_value, corrected_by)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (invoice_id, field_name[:64], original_value, corrected_value, corrected_by[:128]),
        ).fetchone()
    return int(row["id"])


def mark_feedback_revalidated(invoice_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("UPDATE audit.human_feedback SET revalidated = TRUE "
                     "WHERE invoice_id = %s AND revalidated = FALSE", (invoice_id,))


def save_revalidation(invoice_id: str, effective_translation: dict, validation: dict | None,
                      business: dict | None, report: dict | None, thread_id: str | None) -> None:
    """Store a re-validation outcome. Extracted values are left untouched; the corrected
    ('effective') invoice is kept in extracted_raw -> 'revalidation'."""
    recommendation = (report or {}).get("recommendation")
    if recommendation not in RECOMMENDATIONS:
        recommendation = "manual_review"
    revalidation = {
        "at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "effective_translation_result": effective_translation,
        "validation_result": validation,
        "business_validation_result": business,
        "report": _report_summary(report),
    }
    with pool().connection() as conn:
        conn.execute(
            """UPDATE audit.invoice_audit SET
                 validation_status = %s, recommendation = %s,
                 extracted_raw = COALESCE(extracted_raw, '{}'::jsonb) || jsonb_build_object('revalidation', %s::jsonb)
               WHERE invoice_id = %s""",
            (overall_status(validation, business, report), recommendation, _jsonb(revalidation), invoice_id),
        )
        _replace_discrepancies(conn, invoice_id, validation, business)


# ---------------------------------------------------------------- RAG indexing source

def list_indexable_ids() -> list[str]:
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT invoice_id FROM audit.invoice_audit WHERE validation_status <> 'pending' "
            "ORDER BY processed_at"
        ).fetchall()
    return [str(r["invoice_id"]) for r in rows]


def get_index_bundle(invoice_id: str) -> dict | None:
    invoice = get_invoice(invoice_id)
    if not invoice:
        return None
    return {"invoice": invoice, "discrepancies": get_discrepancies(invoice_id)}
