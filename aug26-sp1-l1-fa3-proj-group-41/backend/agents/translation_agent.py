"""Translation Agent - turns every extracted invoice into English.

English invoices are passed through as they are; other languages are
translated by the lang_bridge tool.

Graph:
    translate_all -> verify_translations -> END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models.extraction import ExtractionResult
from backend.models.translation import TranslationResult
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config

logger = logging.getLogger(__name__)


class TranslationState(TypedDict):
    extraction_results: list[dict]     # ExtractionResult dicts
    translation_results: list[dict]    # TranslationResult dicts
    errors: list[str]
    status: str                        # idle | translated | completed | error


# nodes

def translate_all_node(state: TranslationState) -> dict[str, Any]:
    """Send every extraction result through the lang_bridge tool."""
    logger.info("Translation Agent - starting translations ...")

    results: list[dict] = []
    errors: list[str] = []

    for extraction in state.get("extraction_results", []):
        filename = extraction.get("filename", "?")
        email_id = extraction.get("email_id", "unknown")
        source_lang = (extraction.get("extracted_invoice") or {}).get("detected_language") or "?"

        logger.info("Translating: %s (email: %s, lang: %s)", filename, email_id, source_lang)

        try:
            result = invoke_tool_sync("lang_bridge", {"extraction_result": extraction})
        except Exception as exc:
            message = f"Translation failed for {filename}: {exc}"
            errors.append(message)
            logger.error("ERROR: %s", message)
            continue

        results.append(result)
        errors.extend(result.get("translation_errors") or [])

        invoice = result.get("translated_invoice") or {}
        vendor = invoice.get("translated_vendor_name", "N/A") if invoice else "N/A"
        items = len(invoice.get("line_items") or [])

        if result.get("was_translated"):
            logger.info("OK: %s | %s -> en | Vendor: %s | Items: %d", filename, source_lang, vendor, items)
        else:
            logger.info("PASSED: %s | already English - passed through | Vendor: %s | Items: %d",
                        filename, vendor, items)

    logger.info("Translation complete - %d file(s) processed, %d error(s)", len(results), len(errors))
    return {"translation_results": results, "errors": errors, "status": "translated"}


def verify_translations_node(state: TranslationState) -> dict[str, Any]:
    """Count translated / passed-through / failed files and warn about odd results."""
    translated = passed_through = failed = 0

    for result in state.get("translation_results", []):
        filename = result.get("filename", "?")
        invoice = result.get("translated_invoice")

        if not invoice:
            failed += 1
            logger.warning("WARN: %s: no translated invoice produced", filename)
            continue

        if result.get("was_translated"):
            translated += 1
            if not invoice.get("line_items"):
                logger.warning("WARN: %s: translation produced 0 line items", filename)
        else:
            passed_through += 1

        if result.get("translation_errors"):
            logger.warning("%s: translation errors: %s", filename, result["translation_errors"])

    logger.info("Translation summary: %d translated, %d passed-through, %d failed",
                translated, passed_through, failed)

    return {"status": "completed", "errors": list(state.get("errors", []))}


# graph

def build_translation_graph():
    workflow = StateGraph(TranslationState)

    workflow.add_node("translate_all", translate_all_node)
    workflow.add_node("verify_translations", verify_translations_node)

    workflow.set_entry_point("translate_all")
    workflow.add_edge("translate_all", "verify_translations")
    workflow.add_edge("verify_translations", END)

    return compile_graph(workflow)


# agent

class TranslationAgent:
    """Runs the translation graph over the extractor's results."""

    def __init__(self) -> None:
        self.graph = build_translation_graph()

    def run(self, extraction_results: list[ExtractionResult]) -> list[TranslationResult]:
        logger.info("Starting Translation Agent ...")

        if not extraction_results:
            logger.info("No extraction results to translate - skipping.")
            return []

        initial_state: TranslationState = {
            "extraction_results": [r.model_dump() for r in extraction_results],
            "translation_results": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        return [TranslationResult(**r) for r in final_state.get("translation_results", [])]


# printing (used when the file is run directly)

def _print_results(results: list[TranslationResult]) -> None:
    wide, thin = "=" * 80, "-" * 80
    print(f"\n{wide}\nTRANSLATION AGENT - RESULTS\n{wide}")
    print(f"Total files translated: {len(results)}")

    for number, result in enumerate(results, 1):
        print(f"\n{thin}\n[{number}] {result.filename} (email: {result.email_id})\n{thin}")
        print(f"  Source Language : {result.source_language or 'N/A'}")
        print(f"  Was Translated  : {'Yes' if result.was_translated else 'No (already English)'}")
        if result.translation_confidence is not None:
            print(f"  Confidence      : {result.translation_confidence:.2f}")

        if result.translated_invoice:
            _print_invoice(result.translated_invoice)
        else:
            print("\n  No translated invoice data produced.")

        if result.translation_errors:
            print("\n  TRANSLATION ERRORS:")
            for error in result.translation_errors:
                print(f"  - {error}")

    print(f"\n{wide}\nTranslation pipeline complete.\n{wide}")


def _print_invoice(invoice) -> None:
    rule = "-" * 50
    print(f"\n  {rule}\n  TRANSLATED INVOICE DATA:\n  {rule}")
    print(f"  Invoice #     : {invoice.invoice_no or 'N/A'}")
    print(f"  Invoice Date  : {invoice.invoice_date or 'N/A'}")
    print(f"  PO Number     : {invoice.po_number or 'N/A'}")
    print(f"  Vendor ID     : {invoice.vendor_id or 'N/A'}")

    if invoice.was_translated and invoice.original_vendor_name != invoice.translated_vendor_name:
        print(f"  Vendor (orig) : {invoice.original_vendor_name or 'N/A'}")
        print(f"  Vendor (en)   : {invoice.translated_vendor_name or 'N/A'}")
    else:
        print(f"  Vendor Name   : {invoice.translated_vendor_name or 'N/A'}")

    print(f"  Currency      : {invoice.currency or 'N/A'}")
    print(f"  Subtotal      : {invoice.subtotal}")
    print(f"  Tax Amount    : {invoice.tax_amount}")
    print(f"  Total Amount  : {invoice.total_amount}")

    _print_line_items(invoice)

    if invoice.translated_additional_fields:
        print("\n  ADDITIONAL FIELDS (English):")
        for key, value in invoice.translated_additional_fields.items():
            print(f"  {key}: {value}")

        if invoice.was_translated and invoice.original_additional_fields != invoice.translated_additional_fields:
            print("\n  ADDITIONAL FIELDS (Original):")
            for key, value in invoice.original_additional_fields.items():
                print(f"  {key}: {value}")


def _print_line_items(invoice) -> None:
    print(f"\n  LINE ITEMS ({len(invoice.line_items)}):")
    if not invoice.line_items:
        print("  (no line items found)")
        return

    numbers_header = f"{'Qty':>6} {'Price':>10} {'Total':>10}"
    numbers_rule = f"{'-' * 6} {'-' * 10} {'-' * 10}"

    if invoice.was_translated:
        # original and English description side by side
        print(f"  {'Code':<14} {'Original Description':<30} {'-> English':<30} {numbers_header}")
        print(f"  {'-' * 14} {'-' * 30} {'-' * 30} {numbers_rule}")
        for item in invoice.line_items:
            print(f"  {item.item_code or '?':<14} {(item.original_description or '')[:30]:<30} "
                  f"{(item.translated_description or '')[:30]:<30} {_numbers(item)}")
    else:
        print(f"  {'Code':<14} {'Description':<35} {numbers_header}")
        print(f"  {'-' * 14} {'-' * 35} {numbers_rule}")
        for item in invoice.line_items:
            print(f"  {item.item_code or '?':<14} {(item.translated_description or '')[:35]:<35} {_numbers(item)}")


def _numbers(item) -> str:
    show = lambda value: "-" if value is None else str(value)
    return f"{show(item.qty):>6} {show(item.unit_price):>10} {show(item.total):>10}"


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging
    from backend.agents.invoice_monitor_agent import InvoiceMonitorAgent
    from backend.agents.extractor_agent import ExtractorAgent

    setup_logging()

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    extraction_results = ExtractorAgent().run(monitor.run())
    _print_results(TranslationAgent().run(extraction_results))