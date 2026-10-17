"""Reporting Agent - builds the HTML/JSON audit report for every processed invoice
and gives each one a recommendation (approve, manual review or reject).

Graph:
    generate_reports -> summarize_reports -> END
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models.translation import TranslationResult
from backend.models.validation import ValidationResult
from backend.models.business_validation import BusinessValidationResult
from backend.models.reporting import InvoiceReport, ReportingSummary
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config

logger = logging.getLogger(__name__)

RECOMMENDATION_TAGS = {"approve": "[APPROVE]", "manual_review": "[REVIEW]", "reject": "[REJECT]"}


class ReportingState(TypedDict):
    translation_results: list[dict]
    validation_results: list[dict]
    business_validation_results: list[dict]
    reports: list[dict]
    errors: list[str]
    status: str


# nodes

def generate_reports_node(state: ReportingState) -> dict[str, Any]:
    """Run the insight_reporter tool for every invoice, giving it the results of all earlier stages."""
    logger.info("Reporting Agent - generating reports ...")

    validation_by_file = _by_filename(state.get("validation_results", []))
    business_by_file = _by_filename(state.get("business_validation_results", []))

    reports: list[dict] = []
    errors: list[str] = []

    for translation in state.get("translation_results", []):
        filename = translation.get("filename", "?")
        logger.info("Generating report: %s (email: %s)", filename, translation.get("email_id", "unknown"))

        try:
            report = invoke_tool_sync("insight_reporter", {
                "translation_result": translation,
                "validation_result": validation_by_file.get(filename),
                "business_validation_result": business_by_file.get(filename),
            })
        except Exception as exc:
            message = f"Report generation failed for {filename}: {exc}"
            errors.append(message)
            logger.error("ERROR: %s", message)
            continue

        reports.append(report)
        recommendation = report.get("recommendation", "?")
        logger.info("%s %s -> %s | Report: %s",
                    RECOMMENDATION_TAGS.get(recommendation, "[?]"), filename,
                    recommendation.upper(), report.get("report_html_path", "?"))

    logger.info("Report generation complete - %d report(s), %d error(s)", len(reports), len(errors))
    return {"reports": reports, "errors": errors, "status": "generated"}


def summarize_reports_node(state: ReportingState) -> dict[str, Any]:
    """Log how many invoices got each recommendation."""
    reports = state.get("reports", [])
    count = lambda rec: sum(1 for r in reports if r.get("recommendation") == rec)

    logger.info("Reporting summary: %d total - %d auto-approve, %d manual-review, %d reject",
                len(reports), count("approve"), count("manual_review"), count("reject"))

    return {"status": "completed", "errors": list(state.get("errors", []))}


def _by_filename(results: list[dict]) -> dict[str, dict]:
    return {r.get("filename", ""): r for r in results}


# graph

def build_reporting_graph():
    workflow = StateGraph(ReportingState)

    workflow.add_node("generate_reports", generate_reports_node)
    workflow.add_node("summarize_reports", summarize_reports_node)

    workflow.set_entry_point("generate_reports")
    workflow.add_edge("generate_reports", "summarize_reports")
    workflow.add_edge("summarize_reports", END)

    return compile_graph(workflow)


# agent

class ReportingAgent:
    """Runs the reporting graph and returns a ReportingSummary."""

    def __init__(self) -> None:
        self.graph = build_reporting_graph()

    def run(self, translation_results: list[TranslationResult],
            validation_results: list[ValidationResult],
            business_validation_results: list[BusinessValidationResult]) -> ReportingSummary:
        logger.info("Starting Reporting Agent ...")

        if not translation_results:
            logger.info("No results to report - skipping.")
            return ReportingSummary()

        initial_state: ReportingState = {
            "translation_results": [r.model_dump() for r in translation_results],
            "validation_results": [r.model_dump() for r in validation_results],
            "business_validation_results": [r.model_dump() for r in business_validation_results],
            "reports": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        reports = [InvoiceReport(**r) for r in final_state.get("reports", [])]
        count = lambda rec: sum(1 for r in reports if r.recommendation == rec)

        # all reports go to the same folder, so the first one tells us where
        output_dir = ""
        if reports and reports[0].report_html_path:
            output_dir = str(Path(reports[0].report_html_path).parent)

        return ReportingSummary(
            total_invoices=len(reports),
            auto_approved=count("approve"),
            manual_review=count("manual_review"),
            rejected=count("reject"),
            reports=reports,
            output_dir=output_dir,
        )


# printing (used when the file is run directly)

def _print_summary(summary: ReportingSummary) -> None:
    wide, thin = "=" * 80, "-" * 80
    print(f"\n{wide}\nREPORTING AGENT - SUMMARY\n{wide}")
    print(f"Total Invoices : {summary.total_invoices}")
    print(f"Auto-Approve   : {summary.auto_approved}")
    print(f"Manual Review  : {summary.manual_review}")
    print(f"Rejected       : {summary.rejected}")
    print(f"Output Dir     : {summary.output_dir}")

    for number, r in enumerate(summary.reports, 1):
        tag = RECOMMENDATION_TAGS.get(r.recommendation, "")
        print(f"\n{thin}\n{tag} [{number}] {r.filename} (email: {r.email_id})\n{thin}")
        print(f"  Invoice #              : {r.invoice_no or 'N/A'}")
        print(f"  PO #                   : {r.po_number or 'N/A'}")
        print(f"  Vendor                 : {r.vendor_id or 'N/A'} / {r.vendor_name or 'N/A'}")
        print(f"  Total                  : {r.currency or ''} {r.total_amount}")
        print(f"  Language               : {r.source_language or 'en'} "
              f"({'translated' if r.was_translated else 'original'})")
        print(f"  Translation Confidence : {r.translation_confidence:.0%}")
        print(f"  Data Validation        : {r.data_validation_status or 'N/A'} "
              f"(E:{r.data_validation_errors} W:{r.data_validation_warnings})")
        print(f"  Biz Validation         : {r.business_validation_status or 'N/A'} "
              f"(E:{r.business_validation_errors} W:{r.business_validation_warnings} "
              f"Matched:{r.business_validation_matched})")
        print(f"  Recommendation         : {r.recommendation.replace('_', ' ').upper()}")
        for reason in r.recommendation_reasons:
            print(f"    -> {reason}")
        print(f"  Report File            : {r.report_html_path or 'N/A'}")

    print(f"\n{wide}\nReporting pipeline complete - HTML reports saved.\n{wide}")


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging
    from backend.agents.invoice_monitor_agent import InvoiceMonitorAgent
    from backend.agents.extractor_agent import ExtractorAgent
    from backend.agents.translation_agent import TranslationAgent
    from backend.agents.validation_agent import ValidationAgent
    from backend.agents.business_validation_agent import BusinessValidationAgent

    setup_logging()

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    extraction_results = ExtractorAgent().run(monitor.run())
    translation_results = TranslationAgent().run(extraction_results)
    validation_results = ValidationAgent().run(translation_results)
    business_results = BusinessValidationAgent().run(translation_results, validation_results)
    _print_summary(ReportingAgent().run(translation_results, validation_results, business_results))