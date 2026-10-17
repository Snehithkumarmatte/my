"""Chunk ranker tool.

Re-ranks the chunks returned by the retriever before they go to the
generation step. A chunk's final score is

    similarity * source weight + intent boost + keyword bonus

and near-duplicate chunks are dropped so the context is not filled
with the same text twice.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# how useful each kind of chunk usually is (the part before ":" in chunk["source"])
SOURCE_WEIGHTS = {
    "summary": 1.40,   # aggregate summary of all invoices
    "report":  1.20,   # per-invoice audit report
    "erp":     1.10,   # ERP master data
    "invoice": 1.05,   # raw invoice text
}

# if the question contains one of these phrases, boost chunks of that source type
INTENT_BOOSTS = [
    (["how many", "count", "total", "all invoices", "list invoices", "summary",
      "overview", "auto approved", "auto-approved", "auto approve",
      "manual review", "flagged", "rejected", "across invoices"], "summary", 0.30),
    (["discrepanc", "mismatch", "price mismatch", "why", "reason", "flagged",
      "manual review", "missing field", "error", "fail", "issue", "problem",
      "missing", "invoice_no", "currency missing", "data validation",
      "business validation", "erp cross", "decision rationale"], "report", 0.20),
    (["vendor id", "vendor ids", "all vendors", "list vendors", "show vendors",
      "vend-", "supplier", "company"], "erp", 0.25),
    (["sku", "item code", "all sku", "list sku", "all items"], "erp", 0.20),
    (["po number", "purchase order", "all po", "list po"], "erp", 0.20),
    (["confidence", "score", "approve", "manual", "review", "audit"], "report", 0.10),
]

KEYWORD_BONUS_PER_HIT = 0.02
MAX_KEYWORD_BONUS = 0.10
DUPLICATE_OVERLAP = 0.85   # high on purpose: chunks of one report share header text


@tool
def chunk_ranker_tool(query: str, chunks: list[dict[str, Any]], top_n: int = 8) -> dict:
    """Re-rank retrieved chunks and keep the best top_n.

    Combines the similarity score, a weight per source type, boosts for
    the kind of question being asked, and a small bonus for query words
    found in the chunk. Near-duplicate chunks are skipped.

    Args:
        query: The user's question.
        chunks: Chunks from semantic_retriever_tool (text, source, score, chunk_id).
        top_n: How many chunks to keep.

    Returns:
        dict with ranked_chunks, query and status.
    """
    if not chunks:
        return {"query": query, "ranked_chunks": [], "status": "no_chunks"}

    question = query.lower()
    query_words = [w for w in re.findall(r"\w+", question) if len(w) > 3]

    scored = sorted(
        (_score(chunk, question, query_words) for chunk in chunks),
        key=lambda c: c["final_score"],
        reverse=True,
    )

    selected: list[dict] = []
    selected_texts: list[str] = []
    for chunk in scored:
        if len(selected) >= top_n:
            break
        text = chunk.get("text", "").lower()
        if any(_overlap(text, other) > DUPLICATE_OVERLAP for other in selected_texts):
            continue
        selected.append(chunk)
        selected_texts.append(text)

    logger.info(
        "Ranker: %d candidates -> %d selected | top score=%.3f",
        len(chunks), len(selected), selected[0]["final_score"] if selected else 0,
    )
    return {"query": query, "ranked_chunks": selected, "status": "ok"}


def _score(chunk: dict, question: str, query_words: list[str]) -> dict:
    source_type = str(chunk.get("source", "")).split(":")[0]
    similarity = float(chunk.get("score") or 0.0)
    text = chunk.get("text", "").lower()

    intent_boost = sum(
        boost for phrases, target, boost in INTENT_BOOSTS
        if target == source_type and any(p in question for p in phrases)
    )
    keyword_hits = sum(1 for w in query_words if w in text)
    keyword_bonus = min(keyword_hits * KEYWORD_BONUS_PER_HIT, MAX_KEYWORD_BONUS)

    final = similarity * SOURCE_WEIGHTS.get(source_type, 1.0) + intent_boost + keyword_bonus
    return {**chunk, "final_score": final, "src_type": source_type}


def _overlap(a: str, b: str) -> float:
    """Share of distinct words two texts have in common (Jaccard)."""
    words_a, words_b = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)