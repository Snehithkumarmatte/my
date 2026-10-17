
"""
Indexing Agent — parses and indexes invoice documents into FAISS.

LangGraph flow:
    index -> END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

# Logging setup for direct execution.
from backend.config.logging_setup import setup_logging

# Central dispatch function for the vector-indexing tool.
from backend.tool_dispatch import invoke_tool_sync

# LangFuse observability.
from backend.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


# ============================================================
# 1. STATE SCHEMA
# ============================================================
class IndexingState(TypedDict):
    # True means the tool should rebuild the index rather than
    # simply follow its normal update behavior.
    force_reindex: bool

    # Response returned by the vector-indexing tool.
    result: dict[str, Any]

    # Error message if indexing fails.
    error: str


# ============================================================
# 2. NODE: INDEX DOCUMENTS
# ============================================================
def index_node(state: IndexingState) -> IndexingState:
    """
    Invoke the vector_indexer tool.

    The tool handles document parsing, chunking, embedding, and
    storing/updating the FAISS index according to its implementation.
    """
    logger.info("Indexing Agent: building / updating FAISS index ...")

    try:
        # Pass the force_reindex option to the tool.
        result = invoke_tool_sync(
            "vector_indexer",
            {"force_reindex": state["force_reindex"]},
        )

        # Log the number of indexed chunks and the tool's status.
        logger.info(
            "Indexing complete: %d chunk(s) indexed, status=%s",
            result.get("chunks_indexed", 0),
            result.get("status", "?"),
        )

        # Store the result and clear any previous error.
        return {
            **state,
            "result": result,
            "error": "",
        }

    except Exception as exc:
        logger.error("ERROR: Indexing failed: %s", exc)

        # Store the error in graph state.
        # The public run() method raises it after graph execution.
        return {
            **state,
            "result": {},
            "error": str(exc),
        }


# ============================================================
# 3. GRAPH BUILDER
# ============================================================
def build_indexing_graph() -> StateGraph:
    """Build and compile the Indexing Agent LangGraph."""

    g = StateGraph(IndexingState)

    # Register the indexing node.
    g.add_node("index", index_node)

    # Indexing is the first and only operation.
    g.set_entry_point("index")
    g.add_edge("index", END)

    return g.compile()


# ============================================================
# 4. PUBLIC API
# ============================================================
class IndexingAgent:
    """Builds or updates the FAISS vector index."""

    def __init__(self) -> None:
        self._graph = build_indexing_graph()

    @observe(name="agent-rag-indexing")
    def run(
        self,
        force_reindex: bool = False,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """
        Run the indexing graph.

        Returns the indexing result dictionary.
        Raises RuntimeError if the tool invocation fails.
        """
        # Initialize graph state with the requested indexing mode.
        state = IndexingState(
            force_reindex=force_reindex,
            result={},
            error="",
        )

        cfg = {"callbacks": callbacks} if callbacks else {}

        # Execute the indexing node.
        final = self._graph.invoke(state, config=cfg)

        # Unlike the other agents, this public method explicitly
        # raises a RuntimeError when its node records an error.
        if final["error"]:
            raise RuntimeError(final["error"])

        return final["result"]


# ============================================================
# 5. CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    setup_logging()

    agent = IndexingAgent()

    # Explicitly request a full re-index.
    # This can be more expensive than an incremental update.
    res = agent.run(force_reindex=True)

    # Print a short summary of the indexing operation.
    print(
        f"Indexed {res.get('chunks_indexed', 0)} chunks "
        f"from {len(res.get('sources', []))} source(s)"
    )

    print(f"Index path: {res.get('index_path')}")
