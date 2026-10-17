"""Business Validation Agent - checks each invoice against the ERP records
(purchase order, vendor and SKU master).

Graph:
    business_validate_all -> summarize_business_validation -> END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models.translation import TranslationResult
from backend.models.validation import ValidationResult
from backend.models.business_validation import BusinessValidationResult
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config

logger = logging.getLogger(__name__)

STATUSES = ("matched", "discrepancies_found", "po_not_found", "skipped")


class BusinessValidationState(TypedDict):
    translation_results: list[dict]
    validation_results: list[dict]          # from the data completeness checker
    business_validation_results: list[dict]
    errors: list[str]
    status: str


# nodes

def business_validate_all_node(state: BusinessValidationState) -> dict[str, Any]:
    """Run the business_validation tool on every translated invoice."""
    logger.info("Business Validation Agent - starting ERP validation ...")

    # the completeness result for each file goes along as extra context
    validation_by_file = {v.get("filename", ""): v for v in state.get("validation_results", [])}

    results: list[dict] = []
    errors: list[str] = []

    for translation in state.get("translation_results", []):
        filename = translation.get("filename", "?")
        po_number = (translation.get("translated_invoice") or {}).get("po_number", "N/A")
        logger.info("Business validating: %s (PO: %s, email: %s)",
                    filename, po_number, translation.get("email_id", "unknown"))

        try:
            result = invoke_tool_sync("business_validation", {
                "translation_result": translation,
                "validation_result": validation_by_file.get(filename),
            })
        except Exception as exc:
            message = f"Business validation failed for {filename}: {exc}"
            errors.append(message)
            logger.error("ERROR: %s", message)
            continue

        results.append(result)
        status = result.get("validation_status", "?")
        n_errors = result.get("total_errors", 0)
        logger.info("%s %s - %s | Matched: %d | Errors: %d | Warnings: %d",
                    _tag(status, n_errors), filename, status.upper(),
                    result.get("matched_line_items", 0), n_errors, result.get("total_warnings", 0))

    logger.info("Business validation complete - %d file(s), %d error(s)", len(results), len(errors))
    return {"business_validation_results": results, "errors": errors, "status": "validated"}


def summarize_business_validation_node(state: BusinessValidationState) -> dict[str, Any]:
    """Count results per status and log every discrepancy found."""
    counts = dict.fromkeys(STATUSES, 0)

    for result in state.get("business_validation_results", []):
        status = result.get("validation_status", "skipped")
        counts[status if status in counts else "skipped"] += 1

        if status == "discrepancies_found":
            for d in result.get("discrepancies", []):
                severity = "[ERR]" if d.get("severity") == "error" else "[WARN]"
                logger.warning("%s %s: %s", severity, result.get("filename", "?"), d.get("message", "?"))

    logger.info("Business validation summary: %d matched, %d with discrepancies, %d PO-not-found, %d skipped",
                counts["matched"], counts["discrepancies_found"], counts["po_not_found"], counts["skipped"])

    return {"status": "completed", "errors": list(state.get("errors", []))}


def _tag(status: str, n_errors: int) -> str:
    if status == "matched":
        return "[OK]"
    return "[WARN]" if n_errors == 0 else "[ERR]"


# graph

def build_business_validation_graph():
    workflow = StateGraph(BusinessValidationState)

    workflow.add_node("business_validate_all", business_validate_all_node)
    workflow.add_node("summarize_business_validation", summarize_business_validation_node)

    workflow.set_entry_point("business_validate_all")
    workflow.add_edge("business_validate_all", "summarize_business_validation")
    workflow.add_edge("summarize_business_validation", END)

    return compile_graph(workflow)


# agent

class BusinessValidationAgent:
    """Runs the business validation graph over the translated invoices."""

    def __init__(self) -> None:
        self.graph = build_business_validation_graph()

    def run(self, translation_results: list[TranslationResult],
            validation_results: list[ValidationResult] | None = None) -> list[BusinessValidationResult]:
        logger.info("Starting Business Validation Agent ...")

        if not translation_results:
            logger.info("No translation results to validate - skipping.")
            return []

        initial_state: BusinessValidationState = {
            "translation_results": [r.model_dump() for r in translation_results],
            "validation_results": [r.model_dump() for r in (validation_results or [])],
            "business_validation_results": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        return [BusinessValidationResult(**r) for r in final_state.get("business_validation_results", [])]


# printing (used when the file is run directly)

def _print_results(results: list[BusinessValidationResult]) -> None:
    wide, thin = "=" * 80, "-" * 80
    count = lambda status: sum(1 for r in results if r.validation_status == status)

    print(f"\n{wide}\nBUSINESS VALIDATION AGENT - RESULTS\n{wide}")
    print(f"Total invoices validated: {len(results)}")
    print(f"Matched: {count('matched')}  |  Discrepancies: {count('discrepancies_found')}  |  "
          f"PO-Not-Found: {count('po_not_found')}  |  Skipped: {count('skipped')}")

    for number, bv in enumerate(results, 1):
        print(f"\n{thin}\n{_tag(bv.validation_status, bv.total_errors)} [{number}] {bv.filename}  "
              f"(email: {bv.email_id})\n{thin}")
        print(f"  Status        : {bv.validation_status.upper()}")
        print(f"  Is Valid      : {'Yes' if bv.is_valid else 'No'}")
        print(f"  Errors        : {bv.total_errors}")
        print(f"  Warnings      : {bv.total_warnings}")
        print(f"  Matched Items : {bv.matched_line_items}")

        if bv.erp_lookup:
            _print_erp_lookup(bv.erp_lookup)

        if bv.discrepancies:
            print(f"\n  DISCREPANCIES ({len(bv.discrepancies)}):")
            for d in bv.discrepancies:
                _print_discrepancy(d)
        else:
            print("\n  All line items match ERP records!")

    print(f"\n{wide}\nBusiness validation pipeline complete.\n{wide}")


def _print_erp_lookup(lookup) -> None:
    found = lambda ok: "found" if ok else "not found"
    print("\n  ERP LOOKUP:")
    print(f"    PO Number      : {lookup.po_number or 'N/A'} ({found(lookup.po_found)})")
    print(f"    Vendor ID      : {lookup.vendor_id or 'N/A'} ({found(lookup.vendor_found)})")
    if lookup.erp_vendor_name:
        print(f"    ERP Vendor     : {lookup.erp_vendor_name}")
    print(f"    ERP Currency   : {lookup.erp_currency or 'N/A'}")
    print(f"    ERP Line Items : {lookup.erp_line_items_count}")
    print(f"    Inv Line Items : {lookup.invoice_line_items_count}")


def _print_discrepancy(d) -> None:
    level = "ERROR" if d.severity == "error" else "WARN "
    where = f"[Row {d.line_index}]" if d.line_index >= 0 else "[Header]"
    print(f"    {level} {where} [{d.discrepancy_type}] {d.message}")
    if d.invoice_value or d.erp_value:
        print(f"          Invoice: {d.invoice_value or '-'}  |  ERP: {d.erp_value or '-'}")
    if d.difference_pct is not None:
        print(f"          Difference: {d.difference_pct:.1f}%")


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging
    from backend.agents.invoice_monitor_agent import InvoiceMonitorAgent
    from backend.agents.extractor_agent import ExtractorAgent
    from backend.agents.translation_agent import TranslationAgent
    from backend.agents.validation_agent import ValidationAgent

    setup_logging()

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    extraction_results = ExtractorAgent().run(monitor.run())
    translation_results = TranslationAgent().run(extraction_results)
    validation_results = ValidationAgent().run(translation_results)
    _print_results(BusinessValidationAgent().run(translation_results, validation_results))