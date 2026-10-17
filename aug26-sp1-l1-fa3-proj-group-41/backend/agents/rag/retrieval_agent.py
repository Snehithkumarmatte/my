
"""
Retrieval Agent — accepts a user query, generates its embedding,
and performs semantic similarity search over the FAISS index.

LangGraph flow:
    retrieve -> END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

# Logging setup for direct execution.
from backend.config.logging_setup import setup_logging

# Central dispatch function for invoking the semantic retriever.
from backend.tool_dispatch import invoke_tool_sync

# LangFuse observability.
from backend.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


# ============================================================
# 1. STATE SCHEMA
# ============================================================
class RetrievalState(TypedDict):
    # User's question or search query.
    query: str

    # Maximum number of candidate chunks requested.
    top_k: int

    # Optional invoice identifier used to constrain retrieval.
    invoice_id: str | None

    # Complete response returned by the retriever tool.
    result: dict[str, Any]

    # Error message if retrieval fails.
    error: str


# ============================================================
# 2. NODE: RETRIEVE CANDIDATE CHUNKS
# ============================================================
def retrieve_node(state: RetrievalState) -> RetrievalState:
    """
    Invoke semantic_retriever to search for relevant invoice chunks.

    The tool performs the actual semantic search and returns
    candidate chunks for the augmentation stage.
    """
    logger.info(
        "Retrieval Agent: query='%.80s'",
        state["query"],
    )

    try:
        # Pass the query, result limit, and optional invoice filter
        # to the semantic retriever.
        result = invoke_tool_sync(
            "semantic_retriever",
            {
                "query": state["query"],
                "top_k": state["top_k"],
                "invoice_id": state.get("invoice_id"),
            },
        )

        # Count the candidates returned by the tool.
        n = len(result.get("chunks", []))

        logger.info(
            "Retrieved %d candidate chunk(s)",
            n,
        )

        # Preserve the existing state and store the tool response.
        return {
            **state,
            "result": result,
            "error": "",
        }

    except Exception as exc:
        logger.error("ERROR: Retrieval failed: %s", exc)

        # Return an empty result instead of propagating the exception.
        # The status tells the caller that retrieval failed.
        return {
            **state,
            "result": {
                "query": state["query"],
                "chunks": [],
                "status": "error",
            },
            "error": str(exc),
        }


# ============================================================
# 3. GRAPH BUILDER
# ============================================================
def build_retrieval_graph() -> StateGraph:
    """Build and compile the Retrieval Agent graph."""

    g = StateGraph(RetrievalState)

    # Register the retrieval node.
    g.add_node("retrieve", retrieve_node)

    # Start retrieval immediately.
    g.set_entry_point("retrieve")

    # Finish after candidate retrieval.
    g.add_edge("retrieve", END)

    return g.compile()


# ============================================================
# 4. PUBLIC API
# ============================================================
class RetrievalAgent:
    """Performs semantic search over the invoice FAISS index."""

    def __init__(self) -> None:
        self._graph = build_retrieval_graph()

    @observe(name="agent-rag-retrieval")
    def run(
        self,
        query: str,
        top_k: int = 10,
        callbacks: list | None = None,
        invoice_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run semantic retrieval.

        Returns the tool's result dictionary, typically containing
        query, chunks, total_indexed, and status.
        """
        # Prepare the initial graph state.
        state = RetrievalState(
            query=query,
            top_k=top_k,
            invoice_id=invoice_id,
            result={},
            error="",
        )

        cfg = {"callbacks": callbacks} if callbacks else {}

        # Execute the retrieval node.
        final = self._graph.invoke(
            state,
            config=cfg,
        )

        # Return the retriever response, not the entire graph state.
        return final["result"]


# ============================================================
# 5. CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    setup_logging()

    import sys

    # Accept the query from command-line arguments.
    # If no query is provided, use a sample invoice question.
    q = " ".join(sys.argv[1:]) or (
        "Why was invoice INV_EN_006 rejected?"
    )

    # Create the agent and execute retrieval.
    agent = RetrievalAgent()
    res = agent.run(q)

    # Print the query and the number of retrieved chunks.
    print(f"Query: {res['query']}")

    print(
        f"Retrieved {len(res['chunks'])} chunks "
        f"(index has {res['total_indexed']} total)"
    )

    # Display a preview of the first three candidate chunks.
    for c in res["chunks"][:3]:
        print(
            f"[{c['score']:.3f}] {c['source']}: "
            f"{c['text'][:120]}"
        )
