"""Qdrant vector index - a DERIVED store, always rebuildable from PostgreSQL.

Chunks come from text already stored in audit.invoice_audit (raw/translated text)
plus a report text built from the audit row and its discrepancies. No invoice file
is ever re-read and no OCR is repeated to rebuild the index.

Payload per point: invoice_id, chunk_text, source_type (invoice|report), vendor,
language, plus source / filename / invoice_number for display.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from backend.config.settings import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL
from backend.db import repository
from backend.llm import embed_texts

logger = logging.getLogger(__name__)

PAYLOAD_INDEXES = ("invoice_id", "source_type", "vendor", "language")
_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=30)
    return _client


def ensure_collection() -> None:
    c = client()
    if c.collection_exists(QDRANT_COLLECTION):
        return
    c.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    for field in PAYLOAD_INDEXES:
        c.create_payload_index(QDRANT_COLLECTION, field_name=field,
                               field_schema=models.PayloadSchemaType.KEYWORD)
    logger.info("Qdrant collection '%s' created (dim=%d)", QDRANT_COLLECTION, EMBEDDING_DIM)


def _invoice_filter(invoice_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="invoice_id",
                                                     match=models.MatchValue(value=str(invoice_id)))])


# ---------------------------------------------------------------- chunking

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Character windows that prefer to break on a newline."""
    text = (text or "").strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            newline = text.rfind("\n", start + size // 2, end)
            if newline != -1:
                end = newline
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _label(row: dict) -> str:
    return row.get("invoice_number") or Path(row.get("file_path") or "unknown").name


def build_report_text(row: dict, discrepancies: list[dict]) -> str:
    raw = row.get("extracted_raw") or {}
    report = (raw.get("revalidation") or {}).get("report") or raw.get("report") or {}
    lines = [
        f"AUDIT REPORT for invoice {_label(row)} (file {Path(row.get('file_path') or '').name}, "
        f"invoice_id {row.get('invoice_id')})",
        f"Vendor: {row.get('vendor_name') or 'N/A'} | PO: {row.get('po_number') or 'N/A'} | "
        f"Date: {row.get('invoice_date') or 'N/A'} | Total: {row.get('currency') or ''} {row.get('total_amount')}",
        f"Validation status: {row.get('validation_status')} | Recommendation: {row.get('recommendation')}",
        f"Source language: {row.get('source_language') or 'en'} | "
        f"Translation confidence: {row.get('translation_confidence')}",
    ]
    if row.get("error_message"):
        lines.append(f"Processing errors: {row['error_message']}")
    reasons = report.get("recommendation_reasons") or []
    if reasons:
        lines.append("Reasons for the recommendation:")
        lines += [f"- {reason}" for reason in reasons]
    if discrepancies:
        lines.append(f"Discrepancies ({len(discrepancies)}):")
        for d in discrepancies:
            where = f"line {d['line_number']}" if d.get("line_number") else "header"
            dev = f" (deviation {float(d['deviation_pct']):.1f}%)" if d.get("deviation_pct") is not None else ""
            lines.append(
                f"- [{d.get('check_type')}/{d.get('severity')}] {where} {d.get('item_code') or ''} "
                f"{d.get('field_name')}: invoice={d.get('invoice_value')} "
                f"expected/ERP={d.get('erp_value')}{dev}"
            )
    else:
        lines.append("Discrepancies: none")
    return "\n".join(lines)


def build_documents(bundle: dict) -> list[tuple[str, str]]:
    row, discrepancies = bundle["invoice"], bundle["discrepancies"]
    header = (f"Invoice {_label(row)} | vendor {row.get('vendor_name') or 'N/A'} | "
              f"PO {row.get('po_number') or 'N/A'} | recommendation {row.get('recommendation')}")
    invoice_text = row.get("translated_text") or row.get("raw_extracted_text") or ""
    docs = [("invoice", f"{header}\n{chunk}") for chunk in chunk_text(invoice_text)]
    docs += [("report", chunk) for chunk in chunk_text(build_report_text(row, discrepancies), size=1500)]
    return docs


# ---------------------------------------------------------------- write

def has_points(invoice_id: str) -> bool:
    result = client().count(QDRANT_COLLECTION, count_filter=_invoice_filter(invoice_id), exact=True)
    return result.count > 0


def index_invoice(invoice_id: str) -> int:
    """(Re)index one invoice from PostgreSQL. Returns the number of chunks written."""
    ensure_collection()
    bundle = repository.get_index_bundle(invoice_id)
    if not bundle:
        return 0
    row = bundle["invoice"]
    docs = build_documents(bundle)
    c = client()
    c.delete(QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=_invoice_filter(invoice_id)))
    if not docs:
        return 0

    vectors = embed_texts([text for _, text in docs])
    label = _label(row)
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{invoice_id}:{source_type}:{n}")),
            vector=vector,
            payload={
                "invoice_id": str(invoice_id),
                "chunk_text": text,
                "source_type": source_type,
                "source": f"{source_type}:{label}",
                "vendor": row.get("vendor_name") or "",
                "language": row.get("source_language") or "en",
                "invoice_number": row.get("invoice_number"),
                "filename": Path(row.get("file_path") or "").name,
            },
        )
        for n, ((source_type, text), vector) in enumerate(zip(docs, vectors))
    ]
    c.upsert(QDRANT_COLLECTION, points=points, wait=True)
    logger.info("Indexed %s: %d chunk(s)", label, len(points))
    return len(points)


def index_all(force_reindex: bool = False) -> dict[str, Any]:
    """Index every processed invoice. Without force, invoices already in Qdrant are skipped."""
    c = client()
    if force_reindex and c.collection_exists(QDRANT_COLLECTION):
        c.delete_collection(QDRANT_COLLECTION)
    ensure_collection()

    ids = repository.list_indexable_ids()
    chunks = indexed = 0
    sources: list[str] = []
    for invoice_id in ids:
        if not force_reindex and has_points(invoice_id):
            continue
        written = index_invoice(invoice_id)
        if written:
            chunks += written
            indexed += 1
            sources.append(invoice_id)
    return {
        "status": "ok",
        "chunks_indexed": chunks,
        "invoices_indexed": indexed,
        "invoices_total": len(ids),
        "sources": sources,
        "index_path": f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}",
    }


# ---------------------------------------------------------------- read

def retrieve(query: str, limit: int = 10, invoice_id: str | None = None) -> list[dict]:
    ensure_collection()
    vector = embed_texts([query])[0]
    result = client().query_points(
        QDRANT_COLLECTION, query=vector, limit=limit, with_payload=True,
        query_filter=_invoice_filter(invoice_id) if invoice_id else None,
    )
    return [
        {
            "chunk_id": str(point.id),
            "text": (point.payload or {}).get("chunk_text", ""),
            "source": (point.payload or {}).get("source", ""),
            "score": float(point.score),
            "invoice_id": (point.payload or {}).get("invoice_id"),
            "source_type": (point.payload or {}).get("source_type"),
            "vendor": (point.payload or {}).get("vendor"),
        }
        for point in result.points
    ]


def collection_size() -> int:
    ensure_collection()
    return client().count(QDRANT_COLLECTION, exact=True).count
