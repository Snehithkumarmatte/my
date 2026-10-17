"""Runtime settings, read from environment variables (supplied by compose env_file).

No credential has a default here: DATABASE_URL and the AWS keys must come from the
environment. Service addresses default to the compose service names.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent          # .../backend

# --- data stores (service names + CONTAINER ports) ---
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "invoices")

# --- mock ERP: the ONLY route to ERP data ---
ERP_API_BASE_URL = os.getenv("ERP_BASE_URL", "http://mock-erp:8000").rstrip("/")

# --- folders (bind-mounted volumes in compose) ---
INCOMING_DIR = os.getenv("INCOMING_DIR", str(BACKEND_DIR / "data" / "incoming"))
REPORTS_DIR = os.getenv("REPORTS_DIR", str(BACKEND_DIR / "data" / "reports"))

# --- models (LiteLLM model strings) ---
_HAIKU = "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_NOVA = "bedrock/converse/global.amazon.nova-2-lite-v1:0"
BEDROCK_REASONING_MODEL = os.getenv("LLM_REASONING_MODEL", _NOVA)
BEDROCK_RAG_MODEL = os.getenv("LLM_RAG_MODEL", _HAIKU)
BEDROCK_TRANSLATION_MODEL = os.getenv("LLM_TRANSLATION_MODEL", _HAIKU)
LITELLM_RAG_MODEL = BEDROCK_RAG_MODEL                         # name used by reflection_agent
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bedrock/amazon.titan-embed-text-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- behaviour ---
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
AUTO_PROCESS = os.getenv("AUTO_PROCESS", "true").lower() in ("1", "true", "yes")
