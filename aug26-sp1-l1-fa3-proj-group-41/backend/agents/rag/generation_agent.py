
"""
Generation Agent — uses the Response Synthesizer Tool to generate a
contextual answer from ranked chunks and conversation history.

LangGraph flow:
    generate -> END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

# Logging setup for direct execution.
from backend.config.logging_setup import setup_logging

# Central tool-dispatch function for the response synthesizer.
from backend.tool_dispatch import invoke_tool_sync

# LangFuse observability.

logger = logging.getLogger(__name__)


# ============================================================
# 1. STATE SCHEMA
# ============================================================
class GenerationState(TypedDict):
    # Question that the user wants answered.
    query: str

    # Relevant chunks selected by the augmentation stage.
    ranked_chunks: list[dict[str, Any]]

    # Previous conversation messages, represented as dictionaries.
    chat_history: list[dict[str, str]]

    # Response returned by the generation tool.
    result: dict[str, Any]

    # Error message if generation fails.
    error: str


# ============================================================
# 2. NODE: GENERATE ANSWER
# ============================================================
def generate_node(state: GenerationState) -> GenerationState:
    """
    Invoke the response_synthesizer tool to generate an answer.

    The tool is responsible for synthesizing the answer from
    the query, context chunks, and chat history.
    """
    logger.info(
        "Generation Agent: synthesizing answer | %d chunks | "
        "%d history turns",
        len(state["ranked_chunks"]),
        len(state["chat_history"]),
    )

    try:
        # Send the three pieces of context required by the tool:
        # 1. Current query
        # 2. Ranked evidence chunks
        # 3. Conversation history
        result = invoke_tool_sync(
            "response_synthesizer",
            {
                "query": state["query"],
                "ranked_chunks": state["ranked_chunks"],
                "chat_history": state["chat_history"],
            },
        )

        # Log the length of the generated answer.
        logger.info(
            "Answer generated (%d chars)",
            len(result.get("answer", "")),
        )

        # Store the tool response and clear the error field.
        return {
            **state,
            "result": result,
            "error": "",
        }

    except Exception as exc:
        logger.error("ERROR: Generation failed: %s", exc)

        # Return a structured error response so the caller can
        # distinguish generation failure from a normal answer.
        return {
            **state,
            "result": {
                "answer": f"Generation error: {exc}",
                "sources_used": [],
                "context_chunks": 0,
                "status": "error",
            },
            "error": str(exc),
        }


# ============================================================
# 3. GRAPH BUILDER
# ============================================================
def build_generation_graph() -> StateGraph:
    """Build and compile the Generation Agent graph."""

    g = StateGraph(GenerationState)

    # Register the answer-generation node.
    g.add_node("generate", generate_node)

    # Start directly at generation.
    g.set_entry_point("generate")

    # Finish after generating the answer.
    g.add_edge("generate", END)

    return g.compile()


# ============================================================
# 4. PUBLIC API
# ============================================================
class GenerationAgent:
    """Synthesizes a final LLM answer from the supplied context."""

    def __init__(self) -> None:
        self._graph = build_generation_graph()

    
    def run(
        self,
        query: str,
        ranked_chunks: list[dict],
        chat_history: list[dict] | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """
        Generate an answer.

        Returns a dictionary containing answer, sources_used,
        context_chunks, and status.
        """
        # Normalize missing history to an empty list.
        # This prevents the node from having to handle None.
        state = GenerationState(
            query=query,
            ranked_chunks=ranked_chunks,
            chat_history=chat_history or [],
            result={},
            error="",
        )

        cfg = {"callbacks": callbacks} if callbacks else {}

        # Execute the generation graph.
        final = self._graph.invoke(state, config=cfg)

        # Return the tool response, not the complete graph state.
        return final["result"]
