"""Semantic retriever tool.

Finds the invoice and report chunks in Qdrant that are closest in
meaning to the user's question. Pass an invoice_id to search only the
chunks of that one invoice.
"""

from langchain_core.tools import tool

from backend.services.qdrant_store import retrieve


@tool
def semantic_retriever_tool(query: str, top_k: int = 10, invoice_id: str | None = None) -> dict:
    """Return the top_k most relevant chunks for the question.

    Args:
        query: The user's question.
        top_k: How many chunks to return.
        invoice_id: Only search this invoice's chunks (UUID), or None for all.

    Returns:
        dict with query, chunks, total_indexed (number of chunks returned) and status.
    """
    chunks = retrieve(query, limit=max(1, top_k), invoice_id=invoice_id)
    return {
        "query": query,
        "chunks": chunks,
        "total_indexed": len(chunks),
        "status": "ok" if chunks else "empty_index",
    }