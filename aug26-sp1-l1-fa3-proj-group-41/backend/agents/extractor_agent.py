"""Extractor Agent - pulls structured invoice data out of PDF, DOCX and image attachments.

Graph:
    extract_all -> validate_extractions -> END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models import ExtractionResult, MonitorResult
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config

logger = logging.getLogger(__name__)

# fields an invoice can't really be audited without
CRITICAL_FIELDS = ("invoice_no", "po_number", "total_amount", "line_items")


class ExtractorState(TypedDict):
    monitored_emails: list[dict]       # MonitoredEmail dicts
    extraction_results: list[dict]     # ExtractionResult dicts
    errors: list[str]
    status: str                        # idle | extracted | completed | error


# nodes

def extract_all_node(state: ExtractorState) -> dict[str, Any]:
    """Run the data_harvester tool on every attachment of every new email."""
    logger.info("Extractor Agent - starting extraction ...")

    results: list[dict] = []
    errors: list[str] = []

    for email in state.get("monitored_emails", []):
        email_id = email.get("email_meta", {}).get("email_id", "unknown")
        files = email.get("invoice_files", [])

        if not files:
            errors.append(f"No attachments for email {email_id}")
            continue

        for invoice_file in files:
            filename = invoice_file.get("filename", "")
            logger.info("Extracting: %s (email: %s)", filename, email_id)

            try:
                result = invoke_tool_sync("data_harvester", {
                    "file_path": invoice_file.get("file_path", ""),
                    "email_id": email_id,
                })
            except Exception as exc:
                message = f"Extraction failed for {filename}: {exc}"
                errors.append(message)
                logger.error("ERROR: %s", message)
                continue

            results.append(result)
            errors.extend(result.get("extraction_errors") or [])

            invoice = result.get("extracted_invoice") or {}
            logger.info("OK: %s | Invoice#: %s | Items: %d",
                        filename, invoice.get("invoice_no") or "N/A", len(invoice.get("line_items") or []))

    logger.info("Extraction complete - %d file(s) processed, %d error(s)", len(results), len(errors))
    return {"extraction_results": results, "errors": errors, "status": "extracted"}


def validate_extractions_node(state: ExtractorState) -> dict[str, Any]:
    """Warn about files where nothing was extracted or critical fields are missing."""
    no_data = 0

    for result in state.get("extraction_results", []):
        filename = result.get("filename", "?")
        invoice = result.get("extracted_invoice")

        if not invoice:
            no_data += 1
            logger.warning("WARN: %s: no structured data extracted", filename)
            continue

        missing = [field for field in CRITICAL_FIELDS if _is_missing(invoice, field)]
        if missing:
            logger.warning("WARN: %s: missing fields: %s", filename, missing)

    if no_data:
        logger.info("Validation summary: %d no-data", no_data)

    return {"status": "completed", "errors": list(state.get("errors", []))}


def _is_missing(invoice: dict, field: str) -> bool:
    value = invoice.get(field)
    if field == "total_amount":
        return value is None          # a total of 0 is still a value
    return not value


# graph

def build_extractor_graph():
    workflow = StateGraph(ExtractorState)

    workflow.add_node("extract_all", extract_all_node)
    workflow.add_node("validate_extractions", validate_extractions_node)

    workflow.set_entry_point("extract_all")
    workflow.add_edge("extract_all", "validate_extractions")
    workflow.add_edge("validate_extractions", END)

    return compile_graph(workflow)


# agent

class ExtractorAgent:
    """Runs the extractor graph over the monitor's new emails."""

    def __init__(self) -> None:
        self.graph = build_extractor_graph()

    def run(self, monitor_result: MonitorResult) -> list[ExtractionResult]:
        logger.info("Starting Extractor Agent ...")

        if not monitor_result.new_emails:
            logger.info("No emails to extract - skipping.")
            return []

        initial_state: ExtractorState = {
            "monitored_emails": [email.model_dump() for email in monitor_result.new_emails],
            "extraction_results": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        return [ExtractionResult(**r) for r in final_state.get("extraction_results", [])]


# printing (used when the file is run directly)

def _print_results(results: list[ExtractionResult]) -> None:
    wide, thin = "=" * 80, "-" * 80
    print(f"\n{wide}\nEXTRACTOR AGENT - RESULTS\n{wide}")
    print(f"Total files extracted: {len(results)}")

    for number, result in enumerate(results, 1):
        print(f"\n{thin}\n[{number}] {result.filename} (email: {result.email_id})\n{thin}")
        print(f"  Format    : {result.file_format}")
        print(f"  File Path : {result.file_path}")

        _print_raw_text(result.raw_text)

        if result.extracted_invoice:
            _print_invoice(result.extracted_invoice)
        else:
            print("\n  No structured invoice data extracted.")

        if result.extraction_errors:
            print("\n  EXTRACTION ERRORS:")
            for error in result.extraction_errors:
                print(f"  - {error}")

    print(f"\n{wide}\nExtraction pipeline complete.\n{wide}")


def _print_raw_text(raw_text: str) -> None:
    rule = "-" * 50
    print(f"\n  {rule}\n  RAW EXTRACTED TEXT ({len(raw_text)} chars):\n  {rule}")
    if not raw_text.strip():
        print("  (no raw text extracted)")
        return
    for line in raw_text.strip().splitlines():
        print(f"  {line}")


def _print_invoice(invoice) -> None:
    rule = "-" * 50
    print(f"\n  {rule}\n  STRUCTURED INVOICE DATA:\n  {rule}")
    print(f"  Invoice #    : {invoice.invoice_no or 'N/A'}")
    print(f"  Invoice Date : {invoice.invoice_date or 'N/A'}")
    print(f"  PO Number    : {invoice.po_number or 'N/A'}")
    print(f"  Vendor ID    : {invoice.vendor_id or 'N/A'}")
    print(f"  Vendor Name  : {invoice.vendor_name or 'N/A'}")
    print(f"  Currency     : {invoice.currency or 'N/A'}")
    print(f"  Subtotal     : {invoice.subtotal}")
    print(f"  Tax Amount   : {invoice.tax_amount}")
    print(f"  Total Amount : {invoice.total_amount}")
    print(f"  Language     : {invoice.detected_language or 'N/A'}")

    print(f"\n  LINE ITEMS ({len(invoice.line_items)}):")
    if invoice.line_items:
        print(f"  {'Code':<14} {'Description':<35} {'Qty':>6} {'Price':>10} {'Total':>10}")
        print(f"  {'-' * 14} {'-' * 35} {'-' * 6} {'-' * 10} {'-' * 10}")
        for item in invoice.line_items:
            print(f"  {item.item_code or '?':<14} {(item.description or '')[:35]:<35} "
                  f"{_show(item.qty):>6} {_show(item.unit_price):>10} {_show(item.total):>10}")
    else:
        print("  (no line items found)")

    if invoice.additional_fields:
        print("\n  ADDITIONAL FIELDS:")
        for key, value in invoice.additional_fields.items():
            print(f"  {key}: {value}")


def _show(value) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging
    from backend.agents.invoice_monitor_agent import InvoiceMonitorAgent

    setup_logging()

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    _print_results(ExtractorAgent().run(monitor.run()))