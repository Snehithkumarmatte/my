"""Response synthesizer tool.

Writes the answer to the user's question from the ranked context chunks
and the recent conversation, using the Bedrock RAG model.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from backend.llm import invoke_bedrock_chat
from backend.config.settings import BEDROCK_RAG_MODEL

logger = logging.getLogger(__name__)

HISTORY_TURNS = 6          # last 3 question/answer pairs
NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the indexed data. "
    "Please ensure the pipeline has been run and the index is built."
)

SYSTEM_PROMPT = """\
You are an expert **Invoice Audit Support Agent** helping human support staff
understand invoice processing decisions made by an AI system.

You have access to:
- Detailed AI-generated audit reports for each invoice (verdict, issues, discrepancies)
- ERP master data (vendors, purchase orders, SKU catalogue)
- Validation rules applied during auditing

Guidelines:
1. Answer ONLY from the provided context.  If the context does not contain
   enough information, say so clearly — do not hallucinate facts.
2. When explaining a rejection or manual-review decision, quote the exact
   issues found (missing fields, discrepancies, ERP mismatches).
3. Use structured responses: use **bold** for key terms, and bullet points
   for lists of issues.
4. Keep answers concise but complete — no padding.
5. If asked about a specific invoice, identify it by filename or Invoice #
   and give a focused answer.
6. Maintain conversation context from chat history when answering follow-ups.
"""


@tool
def response_synthesizer_tool(
    query: str,
    ranked_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> dict:
    """Answer the question from the ranked chunks and the chat history.

    Args:
        query: The user's current question.
        ranked_chunks: Top chunks from chunk_ranker_tool.
        chat_history: Earlier turns as {'role': 'user'|'assistant', 'content': str}.

    Returns:
        dict with answer, sources_used, context_chunks and status.
    """
    if not ranked_chunks:
        return {"answer": NO_CONTEXT_ANSWER, "sources_used": [], "context_chunks": 0, "status": "no_context"}

    context, sources = build_context(ranked_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += recent_history(chat_history)
    messages.append({
        "role": "user",
        "content": f"Context from invoice audit system:\n\n{context}\n\n---\nUser question: {query}",
    })

    answer = invoke_bedrock_chat(
        model_id=BEDROCK_RAG_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    ) or "No response generated."

    logger.info("Synthesized answer for query='%.60s' | %d chunks | %d sources",
                query, len(ranked_chunks), len(sources))

    return {"answer": answer, "sources_used": sources, "context_chunks": len(ranked_chunks), "status": "ok"}


def build_context(chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Number the chunks for the prompt and collect their sources (no duplicates, in order)."""
    parts, sources = [], []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("final_score", chunk.get("score"))
        score = float(score) if isinstance(score, (int, float)) else 0.0
        parts.append(f"--- Context [{i}] | Source: {source} | Relevance: {score:.3f} ---\n{chunk.get('text', '')}")
        if source not in sources:
            sources.append(source)
    return "\n\n".join(parts), sources


def recent_history(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Last few user/assistant turns; anything malformed is skipped."""
    return [
        {"role": turn["role"], "content": turn["content"]}
        for turn in (chat_history or [])[-HISTORY_TURNS:]
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content")
    ]