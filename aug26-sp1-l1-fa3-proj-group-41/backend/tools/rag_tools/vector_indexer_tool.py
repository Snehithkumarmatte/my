"""Vector indexer tool.

Builds the RAG index: reads the invoice text and audit results already
stored in PostgreSQL, splits them into chunks, embeds them and writes
them to Qdrant. Qdrant is a derived store, so it can always be rebuilt
this way without re-reading any invoice file.
"""

from langchain_core.tools import tool

from backend.services.qdrant_store import index_all


@tool
def vector_indexer_tool(force_reindex: bool = False) -> dict:
    """Index every stored invoice and its report into Qdrant.

    Args:
        force_reindex: Drop the Qdrant collection and rebuild it from scratch.

    Returns:
        dict with status, chunks_indexed, invoices_indexed and sources.
    """
    return index_all(force_reindex=force_reindex)