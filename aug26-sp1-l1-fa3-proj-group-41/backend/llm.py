"""All LLM and embedding calls go through here (LiteLLM -> AWS Bedrock).

Responses are cached in Redis under llm:{sha256(prompt)} and embeddings under
emb:{sha256(text)}, each with a TTL. Redis is disposable: if it is down or
flushed, calls simply go to Bedrock again - nothing breaks, it is only slower.
"""
from __future__ import annotations

import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any

import litellm
import redis

from backend.config.settings import CACHE_TTL_SECONDS, EMBEDDING_MODEL, REDIS_URL

logger = logging.getLogger(__name__)
litellm.drop_params = True          # ignore params a given Bedrock model doesn't support

_redis: redis.Redis | None = None
_redis_checked = False


def _cache() -> redis.Redis | None:
    global _redis, _redis_checked
    if not _redis_checked:
        _redis_checked = True
        try:
            client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2,
                                          decode_responses=True)
            client.ping()
            _redis = client
        except redis.RedisError as exc:
            logger.warning("Redis unavailable - running without cache: %s", exc)
            _redis = None
    return _redis


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    cache = _cache()
    if cache is None:
        return None
    try:
        return cache.get(key)
    except redis.RedisError:
        return None


def _cache_set(key: str, value: str) -> None:
    cache = _cache()
    if cache is None or not value:
        return
    try:
        cache.setex(key, CACHE_TTL_SECONDS, value)
    except redis.RedisError:
        pass


def invoke_bedrock_chat(model_id: str, messages: list[dict], max_tokens: int = 1024,
                        temperature: float = 0.0, use_cache: bool = True) -> str:
    """Chat completion through LiteLLM. Identical prompts are answered from Redis."""
    key = "llm:" + _sha({"model": model_id, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens})
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            logger.info("LLM cache hit (%s)", key[:16])
            return hit

    response = litellm.completion(model=model_id, messages=messages, max_tokens=max_tokens,
                                  temperature=temperature, num_retries=2)
    text = response.choices[0].message.content or ""
    if use_cache:
        _cache_set(key, text)
    return text


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed each text (Titan takes one input per call). Cached per text."""
    vectors: list[list[float]] = []
    for text in texts:
        key = "emb:" + _sha({"model": EMBEDDING_MODEL, "text": text})
        hit = _cache_get(key)
        if hit is not None:
            vectors.append(json.loads(hit))
            continue
        response = litellm.embedding(model=EMBEDDING_MODEL, input=[text])
        item = response.data[0]
        vector = item["embedding"] if isinstance(item, dict) else item.embedding
        _cache_set(key, json.dumps(vector))
        vectors.append(vector)
    return vectors


# --- OpenAI-style client used by reflection_agent: client.chat.completions.create(...) ---

class _Completions:
    def create(self, model: str, messages: list[dict], temperature: float = 0.0,
               max_tokens: int = 1024, **_ignored):
        text = invoke_bedrock_chat(model, messages, max_tokens=max_tokens, temperature=temperature)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class LiteLLMClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())
