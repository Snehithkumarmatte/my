"""Invoice Monitor Agent - looks in the incoming folder for invoices that haven't been audited yet.

Graph:
    scan_inbox -> evaluate_results -> mark_processed -> END
                  (no new invoices) -> END

Duplicate detection is done by the invoice_watcher tool using the file
checksums stored in PostgreSQL, so this agent keeps no state file of its own.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.models import MonitorResult
from backend.tool_dispatch import invoke_tool_sync
from backend.checkpointing import compile_graph, invoke_config
from backend.tools.invoice_watcher_tool import EMAILS_DIR

logger = logging.getLogger(__name__)


class MonitorState(TypedDict):
    scan_requested: bool
    monitor_result: dict | None
    new_email_ids: list[str]
    errors: list[str]
    status: str        # idle | scanned | completed_no_new | completed_with_new | completed | error


# nodes

def scan_inbox_node(state: MonitorState) -> dict[str, Any]:
    """Run the invoice_watcher tool over the incoming folder."""
    logger.info("Invoice Monitor Agent - scanning inbox ...")
    try:
        result = invoke_tool_sync("invoice_watcher", {"scan_directory": str(EMAILS_DIR)})
    except Exception as exc:
        logger.error("Scan failed: %s", exc)
        return {"monitor_result": None, "new_email_ids": [], "errors": [str(exc)], "status": "error"}

    logger.info("Scan complete - %d scanned, %d already processed, %d new",
                result["total_scanned"], result["already_processed"], result["newly_found"])

    return {
        "monitor_result": result,
        "new_email_ids": [email["email_meta"]["email_id"] for email in result.get("new_emails", [])],
        "errors": result.get("errors", []),
        "status": "scanned",
    }


def evaluate_results_node(state: MonitorState) -> dict[str, Any]:
    """Log what was found and say whether there is anything to process."""
    result = state.get("monitor_result")

    if not result or result.get("newly_found", 0) == 0:
        logger.info("DONE: No new emails to process.")
        return {"status": "completed_no_new"}

    logger.info("Found %d new email(s):", result["newly_found"])
    for email in result["new_emails"]:
        meta = email["email_meta"]
        files = [f["filename"] for f in email["invoice_files"]]
        logger.info("-> %s | From: %s | Lang: %s | Files: %s",
                    meta["email_id"], meta["sender"], meta["language"], files)

    return {"status": "completed_with_new"}


def mark_processed_node(state: MonitorState) -> dict[str, Any]:
    """Nothing to write here - the checksums in PostgreSQL already record what was seen."""
    return {"status": "completed"}


def _next_step(state: MonitorState) -> str:
    return "mark_processed" if state.get("status") == "completed_with_new" else "end"


# graph

def build_monitor_graph():
    workflow = StateGraph(MonitorState)

    workflow.add_node("scan_inbox", scan_inbox_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("mark_processed", mark_processed_node)

    workflow.set_entry_point("scan_inbox")
    workflow.add_edge("scan_inbox", "evaluate_results")
    workflow.add_conditional_edges("evaluate_results", _next_step,
                                   {"mark_processed": "mark_processed", "end": END})
    workflow.add_edge("mark_processed", END)

    return compile_graph(workflow)


# agent

class InvoiceMonitorAgent:
    """Runs the monitor graph and returns a MonitorResult."""

    def __init__(self) -> None:
        self.graph = build_monitor_graph()

    def run(self) -> MonitorResult:
        logger.info("Starting Invoice Monitor Agent ...")

        initial_state: MonitorState = {
            "scan_requested": True,
            "monitor_result": None,
            "new_email_ids": [],
            "errors": [],
            "status": "idle",
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        raw = final_state.get("monitor_result")
        if raw:
            return MonitorResult(**raw)
        return MonitorResult(errors=final_state.get("errors", []))

    def reset_processed_state(self) -> None:
        """Kept so older callers still work. Processed files are tracked by
        checksum in PostgreSQL, so there is no local state to reset."""
        logger.info("Processed state lives in PostgreSQL - nothing to reset locally.")

    @staticmethod
    def get_processed_ids() -> set[str]:
        """Kept for older callers; deduplication is done by PostgreSQL checksums."""
        return set()


def _print_results(result: MonitorResult) -> None:
    line = "=" * 60
    print(f"\n{line}\nINVOICE MONITOR AGENT - RESULTS\n{line}")
    print(f"Total emails scanned : {result.total_scanned}")
    print(f"Already processed    : {result.already_processed}")
    print(f"Newly found          : {result.newly_found}")

    if result.new_emails:
        print("\nNew Emails:")
        for email in result.new_emails:
            meta = email.email_meta
            print(f"- {meta.email_id}")
            print(f"    From     : {meta.sender}")
            print(f"    Subject  : {meta.subject}")
            print(f"    Language : {meta.language}")
            print(f"    Files    : {[f.filename for f in email.invoice_files]}")

    if result.errors:
        print(f"\nErrors: {result.errors}")
    print(line)


if __name__ == "__main__":
    from backend.config.logging_setup import setup_logging

    setup_logging()
    _print_results(InvoiceMonitorAgent().run())
