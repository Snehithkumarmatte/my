"""RAG pipeline: Retrieval -> Augmentation -> Generation -> Reflection.

Indexing runs separately (after each invoice is processed, and at startup for
anything missing), so a question never triggers re-extraction or re-indexing.

The Reflection Agent scores relevance, groundedness and context relevance; the
answer is shown only when their average is above 0.7, otherwise it re-retrieves
and regenerates, up to 3 retries.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from backend.agents.rag.augmentation_agent import AugmentationAgent
from backend.agents.rag.generation_agent import GenerationAgent
from backend.agents.rag.indexing_agent import IndexingAgent
from backend.agents.rag.reflection_agent import ReflectionAgent
from backend.agents.rag.retrieval_agent import RetrievalAgent

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.7


@dataclass
class ConversationMemory:
    """Sliding window of recent turns for one chat session."""
    history: list[dict[str, str]] = field(default_factory=list)
    max_messages: int = 20

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.history = self.history[-self.max_messages:]

    def get_history(self) -> list[dict[str, str]]:
        return list(self.history)


@dataclass
class RAGResult:
    query: str
    answer: str
    accepted: bool
    sources_used: list[str]
    context_chunks: int
    reflection: dict[str, Any]
    retry_count: int
    latency_ms: float
    invoice_id: str | None = None


class RAGPipeline:
    def __init__(self, retrieval_top_k: int = 10, rerank_top_n: int = 5) -> None:
        self._indexing_agent = IndexingAgent()
        self._retrieval_agent = RetrievalAgent()
        self._augmentation_agent = AugmentationAgent()
        self._generation_agent = GenerationAgent()
        self._reflection_agent = ReflectionAgent()
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_n = rerank_top_n
        self._sessions: dict[str, ConversationMemory] = {}
        self._sessions_lock = threading.Lock()

    def ensure_index(self, force_reindex: bool = False) -> dict[str, Any]:
        """Index anything in PostgreSQL that is missing from Qdrant (or rebuild with force)."""
        return self._indexing_agent.run(force_reindex=force_reindex)

    def _memory(self, session_id: str) -> ConversationMemory:
        with self._sessions_lock:
            return self._sessions.setdefault(session_id, ConversationMemory())

    def query(self, user_query: str, invoice_id: str | None = None,
              session_id: str = "default") -> RAGResult:
        started = time.perf_counter()
        memory = self._memory(session_id)

        logger.info("[RAG] 1/4 retrieval (invoice filter: %s)", invoice_id or "none")
        retrieved = self._retrieval_agent.run(query=user_query, top_k=self._retrieval_top_k,
                                              invoice_id=invoice_id)
        raw_chunks = retrieved.get("chunks", [])

        logger.info("[RAG] 2/4 augmentation")
        augmented = self._augmentation_agent.run(query=user_query, chunks=raw_chunks,
                                                 top_n=self._rerank_top_n)
        ranked = augmented.get("ranked_chunks", raw_chunks[:self._rerank_top_n])

        logger.info("[RAG] 3/4 generation")
        generated = self._generation_agent.run(query=user_query, ranked_chunks=ranked,
                                               chat_history=memory.get_history())

        logger.info("[RAG] 4/4 reflection")
        reflected = self._reflection_agent.run(
            query=user_query, answer=generated.get("answer", ""), ranked_chunks=ranked,
            invoice_id=invoice_id, chat_history=memory.get_history(),
            sources_used=generated.get("sources_used", []),
        )
        answer = reflected["final_answer"]
        reflection = reflected["reflection"]

        memory.add("user", user_query)
        memory.add("assistant", answer)

        latency = (time.perf_counter() - started) * 1000
        logger.info("[RAG] done in %.0f ms (overall=%s, retries=%d)",
                    latency, reflection.get("overall"), reflected["retry_count"])
        return RAGResult(
            query=user_query,
            answer=answer,
            accepted=float(reflection.get("overall", 0) or 0) > ACCEPT_THRESHOLD,
            sources_used=generated.get("sources_used", []),
            context_chunks=generated.get("context_chunks", 0),
            reflection=reflection,
            retry_count=reflected["retry_count"],
            latency_ms=round(latency, 1),
            invoice_id=invoice_id,
        )

    def clear_memory(self, session_id: str = "default") -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)


_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline()
        return _pipeline
