
"""
Reflection Agent — evaluates generated answers for quality,
completeness, and grounding, and decides whether to accept or retry.

LangGraph flow:
    reflect -> accept -> END
            -> retry -> reflect
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any, Literal

from langgraph.graph import StateGraph, END

# LLM client used to evaluate answer quality.
from backend.llm import LiteLLMClient

# Logging and model configuration.
from backend.config.logging_setup import setup_logging
from backend.config.settings import LITELLM_RAG_MODEL



logger = logging.getLogger(__name__)


# ============================================================
# 1. REFLECTION CONFIGURATION
# ============================================================

# Maximum number of regeneration attempts after the initial answer.
_MAX_RETRIES = 3

# System instructions for the LLM acting as a quality evaluator.
_REFLECTION_SYSTEM = """\
You are a quality-control evaluator for an AI invoice support agent.

Given a user question, the context used, and the generated answer, evaluate:
1. RELEVANCE  — Does the answer directly address the question? (0-1)
2. GROUNDEDNESS  — Is the answer supported by the provided context? (0-1)
               Flag if the answer contains facts NOT present in the context.
3. CONTEXT RELEVANCE — Is the retrieved context relevant to the question? (0-1)

Respond ONLY with valid JSON in exactly this format:
{
  "relevance": <number 0-1>,
  "groundedness": <number 0-1>,
  "context_relevance": <number 0-1>,
  "issues": ["<issue 1>", ...],
  "verdict": "accept" | "retry",
  "feedback": "<one-sentence improvement suggestion if verdict=retry, else empty>"
}

The application calculates the average of the three scores. Accept only when it is greater than 0.7.
"""


# ============================================================
# 2. LAZY LLM CLIENT INITIALIZATION
# ============================================================

# The client starts as None and is created only when first needed.
# Later calls reuse the same client.
_client: LiteLLMClient | None = None


def _get_client() -> LiteLLMClient:
    """Return the shared LLM client, creating it on first use."""
    global _client

    if _client is None:
        _client = LiteLLMClient()

    return _client


# ============================================================
# 3. STATE SCHEMA
# ============================================================
class ReflectionState(TypedDict):
    # Original user question.
    query: str

    # Optional invoice identifier used to constrain retrieval on retries.
    invoice_id: str | None

    # Evidence chunks supplied to the generation stage.
    ranked_chunks: list[dict[str, Any]]

    # Conversation history.
    chat_history: list[dict[str, str]]

    # Current generated answer being evaluated.
    answer: str

    # Sources reported by the generation stage.
    sources_used: list[str]

    # Scores, issues, verdict, and feedback from reflection.
    reflection: dict[str, Any]

    # Number of regeneration attempts already performed.
    retry_count: int

    # Answer ultimately returned to the caller.
    final_answer: str

    # Error information, if needed.
    error: str


# ============================================================
# 4. NODE: REFLECT ON THE ANSWER
# ============================================================
def reflect_node(state: ReflectionState) -> ReflectionState:
    """
    Ask an LLM to evaluate answer relevance and grounding.

    The application calculates the average of the three scores
    and overrides the model's verdict using its threshold.
    """
    logger.info("Reflection Agent: evaluating answer quality ...")

    # Construct a compact representation of the retrieved evidence.
    # Only the first four chunks are included, and each text is
    # truncated to 200 characters.
    context_summary = "\n".join(
        f"[{c.get('source', '?')}]: {c.get('text', '')[:200]}"
        for c in state["ranked_chunks"][:4]
    )

    # Give the evaluator the question, context, and generated answer.
    prompt = (
        f"Question: {state['query']}\n\n"
        f"Context used:\n{context_summary}\n\n"
        f"Generated answer:\n{state['answer']}"
    )

    try:
        client = _get_client()

        # Ask the configured LLM to act as a quality evaluator.
        resp = client.chat.completions.create(
            model=LITELLM_RAG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _REFLECTION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )

        # Extract the model's textual response.
        import json

        content = resp.choices[0].message.content or "{}"

        # Attempt to remove a surrounding Markdown JSON code fence.
        # The original implementation uses lstrip/rstrip, which
        # remove matching characters rather than exact substrings.
        content = (
            content.strip()
            .lstrip("```json")
            .rstrip("```")
            .strip()
        )

        # Convert the JSON string into a Python dictionary.
        reflection = json.loads(content)

        # Read the three quality scores.
        # Missing scores default to zero.
        scores = [
            float(reflection.get(key, 0))
            for key in (
                "relevance",
                "groundedness",
                "context_relevance",
            )
        ]

        # Calculate the unweighted average.
        reflection["overall"] = sum(scores) / 3

        # Enforce the application's threshold instead of trusting
        # the verdict text generated by the evaluator.
        reflection["verdict"] = (
            "accept"
            if reflection["overall"] > 0.7
            else "retry"
        )

    except Exception as exc:
        # Invalid JSON, an API failure, or another exception causes
        # the evaluator to produce a low-score retry result.
        logger.warning("Reflection parse error: %s", exc)

        reflection = {
            "verdict": "retry",
            "overall": 0,
            "relevance": 0,
            "groundedness": 0,
            "context_relevance": 0,
            "issues": ["Could not score response"],
            "feedback": (
                "Recheck the question against retrieved evidence."
            ),
        }

    logger.info(
        "Reflection scores: overall=%s, verdict=%s",
        reflection.get("overall", "?"),
        reflection.get("verdict", "?"),
    )

    # Store the reflection result while preserving other state fields.
    return {**state, "reflection": reflection}


# ============================================================
# 5. NODE: ACCEPT ANSWER
# ============================================================
def accept_node(state: ReflectionState) -> ReflectionState:
    """
    Return the generated answer only if it passes the score threshold.

    Otherwise, return a fixed message explaining that the answer
    could not be sufficiently verified.
    """
    accepted = state["reflection"].get("overall", 0) > 0.7

    if accepted:
        answer = state["answer"]
    else:
        answer = (
            "I could not verify a sufficiently relevant and grounded "
            "answer from the invoice evidence. Please review the "
            "invoice report."
        )

    return {
        **state,
        "final_answer": answer,
    }


# ============================================================
# 6. NODE: RETRY GENERATION
# ============================================================
def retry_node(state: ReflectionState) -> ReflectionState:
    """
    Retrieve evidence again, re-rank it, and regenerate the answer
    using the feedback from the previous reflection.
    """
    feedback = state["reflection"].get("feedback", "")

    logger.info(
        "Reflection: retrying generation (feedback: %s)",
        feedback,
    )

    # Local imports avoid importing these agent modules at the
    # top level of this file.
    from backend.agents.rag.generation_agent import GenerationAgent
    from backend.agents.rag.retrieval_agent import RetrievalAgent
    from backend.agents.rag.augmentation_agent import AugmentationAgent

    # Retrieve fresh candidate chunks.
    retrieved = RetrievalAgent().run(
        state["query"],
        invoice_id=state.get("invoice_id"),
    )

    # Re-rank the new candidates.
    reranked = AugmentationAgent().run(
        state["query"],
        retrieved.get("chunks", []),
    )

    # Prefer ranked results, falling back to the retrieved candidates.
    ranked_chunks = reranked.get(
        "ranked_chunks",
        retrieved.get("chunks", []),
    )

    gen = GenerationAgent()

    # Add reflection feedback to the history sent to generation.
    # This provides an improvement hint to the synthesizer.
    augmented_history = list(state["chat_history"]) + [
        {
            "role": "system",
            "content": (
                f"[Quality feedback]: {feedback} "
                "Improve the answer accordingly."
            ),
        }
    ]

    # Generate a replacement answer with the new context.
    result = gen.run(
        query=state["query"],
        ranked_chunks=ranked_chunks,
        chat_history=augmented_history,
    )

    new_answer = result.get("answer", state["answer"])

    # Update the state so the graph can evaluate the new answer.
    return {
        **state,
        "ranked_chunks": ranked_chunks,
        "answer": new_answer,
        "retry_count": state["retry_count"] + 1,
        "final_answer": new_answer,
    }


# ============================================================
# 7. CONDITIONAL ROUTING
# ============================================================
def _route(
    state: ReflectionState,
) -> Literal["accept", "retry"]:
    """
    Choose the next node based on the verdict and retry count.

    Retry only while the answer fails the threshold and the
    configured retry limit has not been reached.
    """
    if (
        state["reflection"].get("verdict") == "retry"
        and state["retry_count"] < _MAX_RETRIES
    ):
        return "retry"

    return "accept"


# ============================================================
# 8. GRAPH BUILDER
# ============================================================
def build_reflection_graph() -> StateGraph:
    """Build and compile the Reflection Agent graph."""

    g = StateGraph(ReflectionState)

    # Register all three nodes.
    g.add_node("reflect", reflect_node)
    g.add_node("accept", accept_node)
    g.add_node("retry", retry_node)

    # Every execution begins by evaluating the current answer.
    g.set_entry_point("reflect")

    # Reflection decides whether to accept or regenerate.
    g.add_conditional_edges(
        "reflect",
        _route,
        {
            "accept": "accept",
            "retry": "retry",
        },
    )

    # Accepted or withheld answers end the workflow.
    g.add_edge("accept", END)

    # A retry loops back to reflection so the new answer
    # is evaluated using the same quality criteria.
    g.add_edge("retry", "reflect")

    return g.compile()


# ============================================================
# 9. PUBLIC API
# ============================================================
class ReflectionAgent:
    """Evaluates answers and optionally regenerates them."""

    def __init__(self) -> None:
        self._graph = build_reflection_graph()

    
    def run(
        self,
        query: str,
        answer: str,
        ranked_chunks: list[dict],
        invoice_id: str | None = None,
        chat_history: list[dict] | None = None,
        sources_used: list[str] | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate an answer and return the final answer,
        reflection details, and retry count.
        """
        # Prepare the complete initial state.
        state = ReflectionState(
            query=query,
            invoice_id=invoice_id,
            ranked_chunks=ranked_chunks,
            chat_history=chat_history or [],
            answer=answer,
            sources_used=sources_used or [],
            reflection={},
            retry_count=0,
            final_answer="",
            error="",
        )

        cfg = {"callbacks": callbacks} if callbacks else {}

        # Run reflection, with retries if necessary.
        final = self._graph.invoke(
            state,
            config=cfg,
        )

        # Expose the important results rather than internal state.
        return {
            "final_answer": final["final_answer"] or answer,
            "reflection": final["reflection"],
            "retry_count": final["retry_count"],
        }
