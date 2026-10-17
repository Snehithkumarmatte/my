"""Augmentation Agent - re-ranks the retrieved chunks and assembles the context
that the Generation Agent will answer from (uses the chunk_ranker tool).

Graph:
    rerank -> END
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.tool_dispatch import invoke_tool_sync

logger = logging.getLogger(__name__)


class AugmentationState(TypedDict):
    query: str
    chunks: list[dict[str, Any]]
    top_n: int
    result: dict[str, Any]
    error: str


def rerank_node(state: AugmentationState) -> dict[str, Any]:
    logger.info("Augmentation Agent: re-ranking %d chunk(s) -> top %d", len(state["chunks"]), state["top_n"])
    try:
        result = invoke_tool_sync("chunk_ranker", {
            "query": state["query"], "chunks": state["chunks"], "top_n": state["top_n"],
        })
        return {"result": result, "error": ""}
    except Exception as exc:
        logger.error("Re-ranking failed, using retrieval order: %s", exc)
        return {
            "result": {"query": state["query"], "ranked_chunks": state["chunks"][:state["top_n"]],
                       "status": "fallback"},
            "error": str(exc),
        }


def build_augmentation_graph():
    graph = StateGraph(AugmentationState)
    graph.add_node("rerank", rerank_node)
    graph.set_entry_point("rerank")
    graph.add_edge("rerank", END)
    return graph.compile()


class AugmentationAgent:
    """Re-ranks retrieved chunks and returns {'ranked_chunks': [...], ...}."""

    def __init__(self) -> None:
        self._graph = build_augmentation_graph()

    def run(self, query: str, chunks: list[dict], top_n: int = 5,
            callbacks: list | None = None) -> dict[str, Any]:
        state = AugmentationState(query=query, chunks=chunks or [], top_n=top_n, result={}, error="")
        config = {"callbacks": callbacks} if callbacks else {}
        return self._graph.invoke(state, config=config)["result"]
