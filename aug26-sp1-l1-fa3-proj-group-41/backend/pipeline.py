"""End-to-end orchestration of the invoice audit.

    monitor -> extract -> translate -> validate -> ERP validate -> report
            -> persist to PostgreSQL (audit schema) -> index in Qdrant

Every agent is a LangGraph graph checkpointed in PostgreSQL (schema langgraph);
the run's thread id is stored on each audit row. Only one run happens at a time.
"""
from __future__ import annotations

import copy
import logging
import threading
from typing import Any

from backend.agents import (
    BusinessValidationAgent, ExtractorAgent, InvoiceMonitorAgent, ReportingAgent,
    TranslationAgent, ValidationAgent,
)
from backend.checkpointing import new_run_thread_id
from backend.db import repository
from backend.models import TranslationResult
from backend.services import qdrant_store

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()
_agents: dict[str, Any] | None = None

HEADER_FIELDS = {"invoice_no", "invoice_date", "po_number", "vendor_id", "vendor_name",
                 "currency", "subtotal", "tax_amount", "total_amount"}
LINE_FIELDS = {"item_code", "description", "qty", "unit_price", "total"}
NUMERIC_FIELDS = {"subtotal", "tax_amount", "total_amount", "qty", "unit_price", "total"}


def _get_agents() -> dict[str, Any]:
    global _agents
    if _agents is None:
        _agents = {
            "monitor": InvoiceMonitorAgent(),
            "extractor": ExtractorAgent(),
            "translator": TranslationAgent(),
            "validator": ValidationAgent(),
            "business": BusinessValidationAgent(),
            "reporter": ReportingAgent(),
        }
    return _agents


def _index_safely(invoice_id: str) -> int:
    try:
        return qdrant_store.index_invoice(invoice_id)
    except Exception as exc:          # the index is derived; it can be rebuilt later
        logger.error("Indexing failed for %s: %s", invoice_id, exc)
        return 0


# ---------------------------------------------------------------- batch processing

def process_new_invoices() -> dict[str, Any]:
    """Process every new file in the incoming folder. Returns a run summary."""
    if not _run_lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        return _process()
    finally:
        _run_lock.release()


def _process() -> dict[str, Any]:
    agents = _get_agents()
    thread_id = new_run_thread_id("run")

    monitor = agents["monitor"].run()
    if not monitor.new_emails:
        return {"status": "idle", "new_files": 0, "errors": monitor.errors}

    files = [f.file_path for email in monitor.new_emails for f in email.invoice_files]
    logger.info("Run %s: %d new file(s)", thread_id, len(files))

    extractions = agents["extractor"].run(monitor)
    usable = [e for e in extractions if e.extracted_invoice]
    translations = agents["translator"].run(usable) if usable else []
    validations = agents["validator"].run(translations) if translations else []
    business = agents["business"].run(translations, validations) if translations else []
    summary = agents["reporter"].run(translations, validations, business) if translations else None

    by_path = lambda items: {item.file_path: item.model_dump() for item in items}  # noqa: E731
    extraction_by = {e.file_path: e for e in extractions}
    translation_by, validation_by, business_by = by_path(translations), by_path(validations), by_path(business)
    report_by = by_path(summary.reports) if summary else {}

    processed, failed, chunks = [], [], 0
    for path in files:
        extraction = extraction_by.get(path)
        try:
            if extraction is None or not extraction.extracted_invoice or path not in translation_by:
                reason = "; ".join((extraction.extraction_errors if extraction else [])
                                   or ["Extraction produced no structured invoice"])
                invoice_id = repository.mark_failed(
                    path, reason, extraction.model_dump() if extraction else None, thread_id)
                failed.append({"file": path, "invoice_id": invoice_id, "reason": reason})
            else:
                invoice_id = repository.save_invoice_result(
                    path, extraction.model_dump(), translation_by.get(path), validation_by.get(path),
                    business_by.get(path), report_by.get(path), thread_id)
                processed.append({"file": path, "invoice_id": invoice_id,
                                  "recommendation": (report_by.get(path) or {}).get("recommendation")})
            chunks += _index_safely(invoice_id)
        except Exception as exc:
            logger.exception("Could not persist result for %s", path)
            failed.append({"file": path, "reason": f"persist failed: {exc}"})

    result = {"status": "completed", "thread_id": thread_id, "new_files": len(files),
              "processed": processed, "failed": failed, "chunks_indexed": chunks}
    logger.info("Run %s finished: %d processed, %d failed", thread_id, len(processed), len(failed))
    return result


# ---------------------------------------------------------------- human corrections

def field_value(translation_result: dict, field: str) -> Any:
    """Current value of a correctable field ('po_number' or 'line:2:unit_price')."""
    invoice = (translation_result or {}).get("translated_invoice") or {}
    if field.startswith("line:"):
        _, number, sub = field.split(":", 2)
        lines = invoice.get("line_items") or []
        index = int(number) - 1
        if not 0 <= index < len(lines):
            return None
        key = "translated_description" if sub == "description" else sub
        return lines[index].get(key)
    key = "translated_vendor_name" if field == "vendor_name" else field
    return invoice.get(key)


def validate_field_name(field: str) -> None:
    if field.startswith("line:"):
        parts = field.split(":")
        if len(parts) != 3 or not parts[1].isdigit() or parts[2] not in LINE_FIELDS:
            raise ValueError(f"Line fields look like 'line:<number>:<{'|'.join(sorted(LINE_FIELDS))}>'")
    elif field not in HEADER_FIELDS:
        raise ValueError(f"Correctable fields: {sorted(HEADER_FIELDS)} or line:<n>:<field>")


def _coerce(field: str, value: Any) -> Any:
    name = field.split(":")[-1]
    if name in NUMERIC_FIELDS and value not in (None, ""):
        return float(str(value).replace(",", ""))
    return value


def apply_corrections(translation_result: dict, corrections: list[dict]) -> dict:
    """Return a copy of the stored translation result with all corrections applied in order."""
    effective = copy.deepcopy(translation_result)
    invoice = effective.get("translated_invoice") or {}
    for correction in corrections:
        field, value = correction["field_name"], correction["corrected_value"]
        try:
            if field.startswith("line:"):
                _, number, sub = field.split(":", 2)
                lines = invoice.setdefault("line_items", [])
                index = int(number) - 1
                if 0 <= index < len(lines):
                    key = "translated_description" if sub == "description" else sub
                    lines[index][key] = _coerce(field, value)
            elif field in HEADER_FIELDS:
                key = "translated_vendor_name" if field == "vendor_name" else field
                invoice[key] = _coerce(field, value)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping correction %s=%r: %s", field, value, exc)
    effective["translated_invoice"] = invoice
    return effective


def revalidate(invoice_id: str) -> dict[str, Any]:
    """Re-run data validation, ERP validation and reporting with the human corrections
    applied. Nothing is re-extracted and no OCR is repeated."""
    row = repository.get_invoice(invoice_id)
    if not row:
        raise LookupError(f"Invoice {invoice_id} not found")
    stored = (row.get("extracted_raw") or {}).get("translation_result")
    if not stored or not stored.get("translated_invoice"):
        raise ValueError("This invoice has no stored extraction to re-validate (processing failed)")

    corrections = repository.get_corrections(invoice_id)
    effective = apply_corrections(stored, corrections)

    if not _run_lock.acquire(timeout=600):
        raise TimeoutError("A pipeline run is still in progress - try again shortly")
    try:
        agents = _get_agents()
        thread_id = new_run_thread_id("reval")
        translations = [TranslationResult(**effective)]
        validations = agents["validator"].run(translations)
        business = agents["business"].run(translations, validations)
        summary = agents["reporter"].run(translations, validations, business)
    finally:
        _run_lock.release()

    report = summary.reports[0].model_dump() if summary.reports else None
    repository.save_revalidation(
        invoice_id, effective,
        validations[0].model_dump() if validations else None,
        business[0].model_dump() if business else None,
        report, thread_id,
    )
    repository.mark_feedback_revalidated(invoice_id)
    _index_safely(invoice_id)
    return {
        "invoice_id": invoice_id,
        "thread_id": thread_id,
        "corrections_applied": len(corrections),
        "recommendation": (report or {}).get("recommendation"),
        "reasons": (report or {}).get("recommendation_reasons", []),
    }
