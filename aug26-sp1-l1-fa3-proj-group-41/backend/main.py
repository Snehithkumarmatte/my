"""Agent service (container port 8000).

- Background poller: watches the incoming folder and runs the LangGraph pipeline.
- POST /process                         run the pipeline now
- POST /upload                          drop files into the incoming folder, then run the pipeline
- POST /query                           question: SQL for structured ones, RAG for open-ended ones
- POST /invoices/{id}/corrections       record a human correction (INSERT only)
- POST /invoices/{id}/revalidate        re-run validation with the corrections applied
- POST /reindex                         rebuild / top up the Qdrant index from PostgreSQL
- GET  /invoices/{id}/report.html|json  the generated report files
- GET  /health, GET /status, GET /stats
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
import psycopg
import redis
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from backend.config import settings
from backend.config.logging_setup import setup_logging

setup_logging()
log = logging.getLogger("agent")

REQUIRED_ENV = [
    "DATABASE_URL", "REDIS_URL", "QDRANT_URL", "ERP_BASE_URL",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME",
]

_stop = threading.Event()
_last_run: dict = {}


def check_dependencies() -> dict:
    """Raise if any dependency is unreachable; return per-service status."""
    status = {}
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=3) as conn:
        conn.execute("SELECT 1 FROM audit.invoice_audit LIMIT 1")
    status["postgres"] = "ok"
    redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=3).ping()
    status["redis"] = "ok"
    QdrantClient(url=os.environ["QDRANT_URL"], timeout=3).get_collections()
    status["qdrant"] = "ok"
    httpx.get(f"{os.environ['ERP_BASE_URL']}/health", timeout=3).raise_for_status()
    status["mock_erp"] = "ok"
    return status


def _poll_loop() -> None:
    from backend.pipeline import process_new_invoices
    from backend.services import qdrant_store

    try:   # top up the index from PostgreSQL (no-op if Qdrant already has everything)
        log.info("Startup index sync: %s", qdrant_store.index_all(force_reindex=False))
    except Exception:
        log.exception("Startup index sync failed")

    while not _stop.is_set():
        try:
            result = process_new_invoices()
            if result.get("status") == "completed":
                _last_run.clear()
                _last_run.update(result, finished_at=time.time())
        except Exception:
            log.exception("Pipeline run failed")
        _stop.wait(settings.POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")
    for attempt in range(1, 6):
        try:
            log.info("Dependencies ready: %s", check_dependencies())
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("Dependency check failed (attempt %d/5): %s", attempt, exc)
            time.sleep(3)
    else:
        raise RuntimeError("Dependencies unreachable - refusing to start")

    from backend.checkpointing import get_checkpointer
    from backend.services import qdrant_store
    get_checkpointer()                 # creates the langgraph.* tables
    qdrant_store.ensure_collection()

    if settings.AUTO_PROCESS:
        threading.Thread(target=_poll_loop, name="invoice-poller", daemon=True).start()
        log.info("Folder poller started (every %ss): %s", settings.POLL_INTERVAL_SECONDS, settings.INCOMING_DIR)
    yield
    _stop.set()


app = FastAPI(title="Invoice Auditor Agent", version="1.0", lifespan=lifespan)


def _uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise HTTPException(status_code=400, detail="invoice_id must be a UUID")


# ---------------------------------------------------------------- health / status

@app.get("/health")
def health():
    try:
        return {"status": "healthy", **check_dependencies()}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(exc)})


@app.get("/status")
def status():
    from backend.services import qdrant_store
    return {"last_run": _last_run, "indexed_chunks": qdrant_store.collection_size(),
            "incoming_dir": settings.INCOMING_DIR, "poll_interval_s": settings.POLL_INTERVAL_SECONDS}


@app.get("/stats")
def stats():
    """Audit-trail counts (PostgreSQL) and index size (Qdrant) for the dashboard."""
    from backend.db import repository
    from backend.services import qdrant_store
    with repository.pool().connection() as conn:
        by_rec = conn.execute("SELECT recommendation, count(*) AS n FROM audit.invoice_audit "
                              "GROUP BY recommendation").fetchall()
        by_status = conn.execute("SELECT validation_status, count(*) AS n FROM audit.invoice_audit "
                                 "GROUP BY validation_status").fetchall()
        totals = conn.execute(
            """SELECT (SELECT count(*) FROM audit.invoice_audit)   AS invoices,
                      (SELECT count(*) FROM audit.discrepancies)   AS discrepancies,
                      (SELECT count(*) FROM audit.human_feedback)  AS corrections,
                      (SELECT max(processed_at) FROM audit.invoice_audit) AS last_processed_at""").fetchone()
    try:
        chunks = qdrant_store.collection_size()
    except Exception:  # noqa: BLE001
        chunks = None
    return {**totals, "by_recommendation": {r["recommendation"]: r["n"] for r in by_rec},
            "by_status": {r["validation_status"]: r["n"] for r in by_status},
            "indexed_chunks": chunks, "last_run": _last_run}


# ---------------------------------------------------------------- pipeline

@app.post("/process")
def process_now():
    from backend.pipeline import process_new_invoices
    return process_new_invoices()


UPLOAD_EXTENSIONS = {".pdf", ".docx", ".png"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), sender: str | None = Form(None),
           subject: str | None = Form(None)):
    """Save uploaded invoices into the incoming folder (the same path the monitor watches),
    then run the pipeline. The UI never writes files or rows itself."""
    import json
    import re
    from backend.db import repository
    from backend.pipeline import process_new_invoices

    incoming = Path(settings.INCOMING_DIR)
    incoming.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for upload_file in files:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload_file.filename or "upload").name)
        ext = Path(name).suffix.lower()
        if ext not in UPLOAD_EXTENSIONS:
            rejected.append({"file": name, "status": "rejected", "error": f"unsupported type {ext or '?'}"})
            continue
        data = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            rejected.append({"file": name, "status": "rejected", "error": "file larger than 20 MB"})
            continue
        target = incoming / name
        n = 1
        while target.exists():
            target = incoming / f"{Path(name).stem}_{n}{ext}"
            n += 1
        target.write_bytes(data)
        if sender or subject:   # simulated email header, like the dataset's .meta.json files
            meta = {"sender": sender or "dashboard-upload", "subject": subject or target.name,
                    "received_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "language": "unknown", "attachments": [target.name]}
            target.with_name(f"{target.stem}.meta.json").write_text(json.dumps(meta, indent=2))
        saved.append((name, str(target.resolve()), repository.checksum_file(target)))

    run = process_new_invoices() if saved else {"status": "idle"}
    if run.get("status") == "busy":       # the poller is mid-run; it will pick the files up
        return {"run": run, "results": [{"file": n, "status": "queued"} for n, _, _ in saved] + rejected}

    outcome = {item["file"]: item for item in run.get("processed", []) + run.get("failed", [])}
    results = []
    for name, path, checksum in saved:
        item = outcome.get(path)
        if item is None:
            with repository.pool().connection() as conn:
                row = conn.execute("SELECT invoice_id::text AS invoice_id, recommendation, validation_status "
                                   "FROM audit.invoice_audit WHERE file_checksum = %s", (checksum,)).fetchone()
            results.append({"file": name, "status": "duplicate" if row else "queued",
                            "invoice_id": row and row["invoice_id"],
                            "recommendation": row and row["recommendation"],
                            "error": "already audited (same checksum)" if row else None})
        elif "reason" in item:
            results.append({"file": name, "status": "failed", "invoice_id": item.get("invoice_id"),
                            "recommendation": "manual_review", "error": item["reason"]})
        else:
            results.append({"file": name, "status": "processed", "invoice_id": item["invoice_id"],
                            "recommendation": item.get("recommendation"), "error": None})
    return {"run": {k: v for k, v in run.items() if k in ("status", "thread_id", "chunks_indexed")},
            "results": results + rejected}


def _report_path(invoice_id: str, kind: str) -> Path:
    from backend.db import repository
    row = repository.get_invoice(_uuid(invoice_id))
    if not row:
        raise HTTPException(status_code=404, detail="Invoice not found")
    raw = row.get("extracted_raw") or {}
    report = (raw.get("revalidation") or {}).get("report") or raw.get("report") or {}
    path = report.get(f"report_{kind}_path")
    if not path:
        raise HTTPException(status_code=404, detail="No report was generated for this invoice")
    # serve only files inside the reports folder
    resolved = Path(path).resolve()
    if Path(settings.REPORTS_DIR).resolve() not in resolved.parents or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Report file is not available")
    return resolved


@app.get("/invoices/{invoice_id}/report.html", response_class=HTMLResponse)
def report_html(invoice_id: str):
    return HTMLResponse(_report_path(invoice_id, "html").read_text(encoding="utf-8"))


@app.get("/invoices/{invoice_id}/report.json")
def report_json(invoice_id: str):
    return FileResponse(_report_path(invoice_id, "json"), media_type="application/json")


# ---------------------------------------------------------------- RAG

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    invoice_id: str | None = None
    session_id: str = "default"


@app.post("/query")
def query(request: QueryRequest):
    """Structured questions -> one parameterised SQL query on the audit schema.
    Open-ended questions -> RAG (retrieve, augment, generate, reflect)."""
    from backend.services.structured_query import try_answer
    invoice_id = _uuid(request.invoice_id) if request.invoice_id else None
    started = time.perf_counter()

    structured = try_answer(request.question, invoice_id=invoice_id)
    if structured is not None:
        return {"query": request.question, "answer": structured.answer, "route": "sql",
                "sql": " ".join(structured.sql.split()), "accepted": True, "sources_used": ["postgresql:audit"],
                "context_chunks": 0, "reflection": {}, "retry_count": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1), "invoice_id": invoice_id}

    from backend.services.rag_pipeline import get_rag_pipeline
    try:
        result = get_rag_pipeline().query(request.question, invoice_id=invoice_id,
                                          session_id=request.session_id)
    except Exception as exc:  # noqa: BLE001 - e.g. Bedrock unreachable
        log.exception("RAG query failed")
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {exc}")
    return {**asdict(result), "route": "rag"}


class ReindexRequest(BaseModel):
    force: bool = False


@app.post("/reindex")
def reindex(request: ReindexRequest):
    from backend.services import qdrant_store
    return qdrant_store.index_all(force_reindex=request.force)


# ---------------------------------------------------------------- human in the loop

class CorrectionRequest(BaseModel):
    field_name: str = Field(..., description="e.g. po_number, currency, total_amount, line:2:unit_price")
    corrected_value: str
    corrected_by: str = Field(..., min_length=1)
    revalidate: bool = True


@app.post("/invoices/{invoice_id}/corrections")
def add_correction(invoice_id: str, request: CorrectionRequest):
    from backend.db import repository
    from backend.pipeline import field_value, revalidate, validate_field_name

    invoice_id = _uuid(invoice_id)
    row = repository.get_invoice(invoice_id)
    if not row:
        raise HTTPException(status_code=404, detail="Invoice not found")
    try:
        validate_field_name(request.field_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    stored = (row.get("extracted_raw") or {}).get("translation_result") or {}
    original = field_value(stored, request.field_name)
    feedback_id = repository.add_correction(
        invoice_id, request.field_name, None if original is None else str(original),
        request.corrected_value, request.corrected_by)

    response = {"feedback_id": feedback_id, "invoice_id": invoice_id,
                "field_name": request.field_name, "original_value": original}
    if request.revalidate:
        try:
            response["revalidation"] = revalidate(invoice_id)
        except (ValueError, TimeoutError) as exc:
            response["revalidation_error"] = str(exc)
    return response


@app.post("/invoices/{invoice_id}/revalidate")
def revalidate_invoice(invoice_id: str):
    from backend.pipeline import revalidate
    try:
        return revalidate(_uuid(invoice_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
