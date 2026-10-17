"""Invoice Data Validation Agent - checks every translated invoice for missing or wrong data
using the rules in config/rules.yaml.

Graph:
    validate_all -> summarize_validation -> END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models.translation import TranslationResult
from backend.models.validation import ValidationResult
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config

logger = logging.getLogger(__name__)

STATUS_TAGS = {"passed": "[OK]", "passed_with_warnings": "[WARN]"}     # anything else -> [ERR]


class ValidationState(TypedDict):
    translation_results: list[dict]    # TranslationResult dicts
    validation_results: list[dict]     # ValidationResult dicts
    errors: list[str]
    status: str                        # idle | validated | completed | error


# nodes

def validate_all_node(state: ValidationState) -> dict[str, Any]:
    """Run the data_completeness_checker tool on every translated invoice."""
    logger.info("Validation Agent - starting data validation ...")

    results: list[dict] = []
    errors: list[str] = []

    for translation in state.get("translation_results", []):
        filename = translation.get("filename", "?")
        logger.info("Validating: %s (email: %s)", filename, translation.get("email_id", "unknown"))

        try:
            result = invoke_tool_sync("data_completeness_checker", {"translation_result": translation})
        except Exception as exc:
            message = f"Validation failed for {filename}: {exc}"
            errors.append(message)
            logger.error("ERROR: %s", message)
            continue

        results.append(result)
        status = result.get("validation_status", "?")
        logger.info("%s %s - %s | Errors: %d | Warnings: %d",
                    STATUS_TAGS.get(status, "[ERR]"), filename, status.upper(),
                    result.get("total_errors", 0), result.get("total_warnings", 0))

    logger.info("Validation complete - %d file(s) validated, %d error(s)", len(results), len(errors))
    return {"validation_results": results, "errors": errors, "status": "validated"}


def summarize_validation_node(state: ValidationState) -> dict[str, Any]:
    """Count passed / warning / failed invoices and log the issues that matter."""
    results = state.get("validation_results", [])
    passed = with_warnings = failed = 0

    for result in results:
        filename = result.get("filename", "?")
        status = result.get("validation_status", "failed")

        if status == "passed":
            passed += 1
        elif status == "passed_with_warnings":
            with_warnings += 1
            for issue in _issues(result, "warning"):
                logger.warning("WARN: %s: %s", filename, issue.get("message"))
        else:
            failed += 1
            for issue in _issues(result, "error"):
                logger.error("ERROR: %s: %s", filename, issue.get("message"))

    logger.info("Validation summary: %d passed, %d with warnings, %d failed (total: %d)",
                passed, with_warnings, failed, len(results))

    return {"status": "completed", "errors": list(state.get("errors", []))}


def _issues(result: dict, severity: str) -> list[dict]:
    """Header issues first, then line-item issues, of one severity."""
    every_issue = result.get("header_issues", []) + result.get("line_item_issues", [])
    return [issue for issue in every_issue if issue.get("severity") == severity]


# graph

def build_validation_graph():
    workflow = StateGraph(ValidationState)

    workflow.add_node("validate_all", validate_all_node)
    workflow.add_node("summarize_validation", summarize_validation_node)

    workflow.set_entry_point("validate_all")
    workflow.add_edge("validate_all", "summarize_validation")
    workflow.add_edge("summarize_validation", END)

    return compile_graph(workflow)


# agent

class ValidationAgent:
    """Runs the validation graph over the translation results."""

    def __init__(self) -> None:
        self.graph = build_validation_graph()

    def run(self, translation_results: list[TranslationResult]) -> list[ValidationResult]:
        logger.info("Starting Validation Agent ...")

        if not translation_results:
            logger.info("No translation results to validate - skipping.")
            return []

        initial_state: ValidationState = {
            "translation_results": [r.model_dump() for r in translation_results],
            "validation_results": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        return [ValidationResult(**r) for r in final_state.get("validation_results", [])]


# printing (used when the file is run directly)

def _print_results(results: list[ValidationResult]) -> None:
    wide, thin = "=" * 80, "-" * 80
    count = lambda status: sum(1 for v in results if v.validation_status == status)

    print(f"\n{wide}\nVALIDATION AGENT - RESULTS\n{wide}")
    print(f"Total invoices validated: {len(results)}")
    print(f"Passed: {count('passed')} | Warnings: {count('passed_with_warnings')} | Failed: {count('failed')}")

    for number, v in enumerate(results, 1):
        tag = STATUS_TAGS.get(v.validation_status, "[ERR]")
        print(f"\n{thin}\n{tag} [{number}] {v.filename} (email: {v.email_id})\n{thin}")
        print(f"  Status        : {v.validation_status.upper()}")
        print(f"  Is Valid      : {'Yes' if v.is_valid else 'No'}")
        print(f"  Errors        : {v.total_errors}")
        print(f"  Warnings      : {v.total_warnings}")
        print(f"  Language      : {v.source_language or 'N/A'}")
        print(f"  Rules Applied : {', '.join(v.rules_applied)}")

        if v.header_issues:
            print(f"\n  HEADER ISSUES ({len(v.header_issues)}):")
            for issue in v.header_issues:
                _print_issue(issue, "")

        if v.line_item_issues:
            print(f"\n  LINE ITEM ISSUES ({len(v.line_item_issues)}):")
            for issue in v.line_item_issues:
                _print_issue(issue, f"[Row {issue.line_index}] ")

        if not v.header_issues and not v.line_item_issues:
            print("\n  No validation issues - all checks passed!")

    print(f"\n{wide}\nValidation pipeline complete.\n{wide}")


def _print_issue(issue, prefix: str) -> None:
    level = "ERROR" if issue.severity == "error" else "WARN "
    print(f"    {level} {prefix}[{issue.issue_type}] {issue.field_name}: {issue.message}")
    if issue.expected or issue.actual:
        print(f"          Expected: {issue.expected or '-'}  |  Actual: {issue.actual or '-'}")


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging
    from backend.agents.invoice_monitor_agent import InvoiceMonitorAgent
    from backend.agents.extractor_agent import ExtractorAgent
    from backend.agents.translation_agent import TranslationAgent

    setup_logging()

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    extraction_results = ExtractorAgent().run(monitor.run())
    translation_results = TranslationAgent().run(extraction_results)
    _print_results(ValidationAgent().run(translation_results))