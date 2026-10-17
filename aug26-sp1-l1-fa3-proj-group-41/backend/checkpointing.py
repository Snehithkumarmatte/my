"""LangGraph checkpointing in PostgreSQL (schema: langgraph).

Every agent graph is compiled with the same PostgresSaver. Each pipeline run gets a
run thread id (stored in audit.invoice_audit.thread_id); every graph invocation in
that run uses "<run id>:<suffix>" so different graphs never share a checkpoint.
"""
from __future__ import annotations

import atexit
import logging
import uuid
from contextvars import ContextVar

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from backend.config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

_run_thread_id: ContextVar[str | None] = ContextVar("run_thread_id", default=None)
_pool: ConnectionPool | None = None
_saver: PostgresSaver | None = None


@atexit.register
def _close_pool() -> None:
    if _pool is not None:
        _pool.close(timeout=1)


def get_checkpointer() -> PostgresSaver:
    """Create the saver once; setup() creates its tables inside the langgraph schema."""
    global _pool, _saver
    if _saver is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(
            DATABASE_URL, min_size=1, max_size=5, open=True,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": "-c search_path=langgraph"},
        )
        _saver = PostgresSaver(_pool)
        _saver.setup()
        logger.info("LangGraph checkpointer ready (schema: langgraph)")
    return _saver


def compile_graph(workflow):
    return workflow.compile(checkpointer=get_checkpointer())


def new_run_thread_id(prefix: str = "run") -> str:
    thread_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
    _run_thread_id.set(thread_id)
    return thread_id


def current_run_thread_id() -> str | None:
    return _run_thread_id.get()


def invoke_config() -> dict:
    base = _run_thread_id.get() or "adhoc"
    return {"configurable": {"thread_id": f"{base}:{uuid.uuid4().hex[:8]}"}}
