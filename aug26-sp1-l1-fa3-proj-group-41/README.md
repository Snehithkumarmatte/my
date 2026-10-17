# AI Invoice Auditor

A containerised, multilingual invoice validation system. Invoices (PDF, DOCX, scanned PNG; English, Spanish, German) dropped into a monitored folder are extracted, translated, validated against an ERP service backed by PostgreSQL, reported on, and indexed for natural-language questions. A Streamlit dashboard lets an auditor review, correct and re-validate.

The whole stack (six containers) starts with one command, and the audit trail and RAG index survive container destruction.

---

## 1. Architecture

```
                         ┌──────────────── docker compose network: auditor-net ────────────────┐
 Auditor's browser ──►   │  streamlit-ui :8501 ──HTTP──► agent :8000 ──HTTP──► mock-erp :8000   │
 (localhost:8501)        │       │ read audit                │  writes audit        │ reads erp  │
                         │       ▼                           ▼                      ▼            │
 incoming/ folder ──vol──►│  ┌─────────────── postgres:16 :5432 (volume pg_data) ──────────────┐ │
 reports/ folder  ◄─vol── │  │ schema erp (mock-erp) · audit (agent writes, UI reads) · langgraph│ │
                         │  └────────────────────────────────────────────────────────────────┘ │
                         │   agent ──► qdrant :6333 (volume qdrant_data, derived index)          │
                         │   agent ──► redis :6379  (no volume, cache only)                       │
                         └───────────────────────────────────────────────┬──────────────────────┘
                                                                         ▼
                                                              AWS Bedrock (via LiteLLM, in-process)
```

| Service | Image / build | Host:container | Responsibility |
|---|---|---|---|
| `agent` | `backend/Dockerfile` → `invoice-agent:1.0` | 8000:8000 | LangGraph agents: monitor, extractor, translation, validation, business validation, reporting, RAG. Only writer of application data. `/health` endpoint. |
| `mock-erp` | `mock_erp/Dockerfile` → `invoice-mock-erp:1.0` | 8001:8000 | FastAPI over the `erp` schema: `/po/{po}`, `/vendor/{id}`, `/sku/{code}`. Only route to ERP data. |
| `streamlit-ui` | `frontend/Dockerfile` → `invoice-streamlit-ui:1.0` | 8501:8501 | Human-in-the-loop dashboard. Reads `audit` (read-only connection), calls the agent for every change. |
| `postgres` | `postgres:16` | 5432:5432 | System of record: ERP master data, audit trail, LangGraph checkpoints. |
| `qdrant` | `qdrant/qdrant:v1.12.4` | 6333:6333 | Vector index of invoice + report text. Rebuildable from PostgreSQL. |
| `redis` | `redis:7.2` | 6379:6379 | Expiring cache for LLM responses (`llm:{sha256(prompt)}`) and embeddings. |

Inside the network every service is addressed by **service name and container port** (`postgres:5432`, `qdrant:6333`, `redis:6379`, `mock-erp:8000`, `agent:8000`). Host ports are only for your browser.

### Processing flow (one invoice)

1. File lands in `backend/data/incoming/` (bind-mounted into the agent). The file never enters the database.
2. **Invoice Monitor** computes a SHA-256 checksum, skips anything already in `audit.invoice_audit`, records new files (including ones that later fail), and reads the companion `.meta.json` email header.
3. **Extractor** gets raw text — `pdfplumber` for digital PDFs, Tesseract OCR (`eng+spa+deu`) for scans/PNGs, `python-docx` for DOCX — and stores the full raw text. Fields are parsed by Bedrock; if Bedrock is unavailable a deterministic rule-based parser takes over, so a model outage never stops extraction.
4. **Translation** detects the language and translates non-English content (cached in Redis) with a confidence score.
5. **Invoice Data Validation** checks completeness and arithmetic against `backend/config/rules.yaml`.
6. **Business Validation** calls `mock-erp` over HTTP and compares quantity, unit price and currency with the PO, applying the tolerances from `rules.yaml`. One discrepancy row per mismatched field.
7. **Reporting** applies the `decision_rules` in `rules.yaml` (first match wins) → `approve | manual_review | reject`, writes the outcome to PostgreSQL and JSON + HTML report files.
8. **Indexing** chunks the stored text and report, embeds it and writes to Qdrant with payload `invoice_id, chunk_text, source_type, vendor, language`.

Every agent is a LangGraph graph checkpointed by `PostgresSaver` into the `langgraph` schema; the run's `thread_id` is stored on the audit row.

### Questions

- **Structured questions** ("How many invoices were rejected?", "Total amount for HafenLogistik?", "Status of INV-1001?") are answered directly from PostgreSQL with fixed, parameterised SQL — no LLM, no vector search.
- **Open-ended questions** ("Why was INV-1001 flagged?") go through the RAG graph: Retrieval (optionally filtered to one invoice) → Augmentation (rerank) → Generation (LiteLLM) → Reflection (RAG triad: relevance, groundedness, context relevance). The answer is shown only when the average score is > 0.7; otherwise it re-retrieves and regenerates, up to 3 retries.

---

## 2. Data architecture

| Store | Holds | Survives `down`? | Survives `down -v`? |
|---|---|---|---|
| Filesystem (bind mounts) | Original invoice files, generated report files | Yes | Yes (host folders) |
| PostgreSQL (`pg_data` volume) | ERP master data, audit trail, LangGraph checkpoints | Yes | **No** |
| Qdrant (`qdrant_data` volume) | Embeddings + chunk payloads | Yes | **No** (rebuildable) |
| Redis (no volume) | LLM/embedding cache with TTL | No — and it doesn't need to | No |

PostgreSQL has three schemas in one database (`backend/data/sql/01_init_schema.sql`):

- `erp` — `vendors`, `purchase_orders`, `po_line_items`, `sku_master` (owned by mock-erp; data in `02_erp_data.sql`).
- `audit` — `invoice_audit`, `invoice_line_items`, `discrepancies`, `human_feedback` (agent writes, UI reads).
- `langgraph` — created automatically by the LangGraph Postgres checkpointer.

Design rules that are enforced in code:

- Line items are rows in `audit.invoice_line_items`, not JSON inside the invoice row.
- A human correction is an **INSERT** into `audit.human_feedback`; the original extracted values are never overwritten. Re-validation applies the corrections on top of the stored extraction.
- Money is `NUMERIC`, dates are `DATE`/`TIMESTAMPTZ`; the complete raw text is kept in `raw_extracted_text`; format-specific extras and the email header go in `extracted_raw` (JSONB).
- The agent and UI never query `erp.*` — ERP data only comes through the mock-erp API.
- Qdrant is derived from PostgreSQL: `POST /reindex` (or the button on *System status*) rebuilds it from stored text with no OCR and no file re-read. On startup the agent tops up anything missing.

---

## 3. Running it

### Prerequisites
Docker Engine with Compose v2, and AWS credentials with Bedrock access to the models in `.env.example`.

### Configure and start

```bash
git clone <repo-url> && cd aug26-sp1-l1-fa3-proj-group-41
cp .env.example .env
#   edit .env: set POSTGRES_PASSWORD (and the same password inside DATABASE_URL),
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME, and APP_UID=$(id -u)
docker compose up --build -d
docker compose ps        # all six services should become "healthy"
```

Startup order is enforced by health checks: postgres (accepting connections *and* ERP data loaded), qdrant and redis → mock-erp → agent → streamlit-ui. The schema loads automatically on the first start of an empty `pg_data` volume.

| URL | What |
|---|---|
| http://localhost:8501 | Dashboard |
| http://localhost:8000/docs | Agent API |
| http://localhost:8001/docs | Mock ERP API |
| http://localhost:6333/dashboard | Qdrant |

The sample invoices in `backend/data/incoming/` are processed automatically within one poll interval. To simulate new e-mails arriving, copy files from `backend/data/samples/` into `backend/data/incoming/`, or use **Upload** in the dashboard.

### Agent API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Checks postgres, redis, qdrant and mock-erp; 503 if any is down |
| POST | `/process` | Process new files in the incoming folder now |
| POST | `/upload` | Save uploaded files into the incoming folder, then process |
| POST | `/query` | `{"question": ..., "invoice_id": optional}` → SQL or RAG answer |
| POST | `/invoices/{id}/corrections` | `{"field_name", "corrected_value", "corrected_by", "revalidate"}` |
| POST | `/invoices/{id}/revalidate` | Re-run validation + reporting with all corrections applied |
| GET | `/invoices/{id}/report.html` · `/report.json` | Generated reports |
| POST | `/reindex` | `{"force": true}` rebuilds Qdrant from PostgreSQL |
| GET | `/stats`, `/status` | Counts and last run |

Correctable fields: `invoice_no, invoice_date, vendor_name, vendor_id, po_number, currency, subtotal, tax_amount, total_amount`, or `line:<n>:<item_code|description|qty|unit_price|total>`.

---

## 4. Verification checklist (section 9 of the brief)

```bash
# Check 1 - single-command startup from a clean state
docker compose down -v
docker compose up --build -d
docker compose ps                       # six services, all (healthy)

# Check 2 - end-to-end processing
cp backend/data/samples/INV_ES_003.* backend/data/incoming/    # or upload in the UI
docker compose logs -f agent            # monitor -> extract -> translate -> validate -> report -> index
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "SELECT invoice_number, source_language, translation_confidence, validation_status, recommendation FROM audit.invoice_audit;"'

# Check 3 - persistence across restart (volumes kept)
docker compose down
docker compose up -d
#   audit rows are still there (query above) and the Ask page still answers
#   "Why was INV-1001 flagged?" - nothing is reprocessed (checksums already audited)

# Check 4 - deliberate data loss (volumes deleted)
docker compose down -v
docker compose up -d
#   audit trail and Qdrant collection are empty: -v removed pg_data and qdrant_data.
#   (The invoice files are on the host, so the monitor re-processes them from scratch.)

# Check 5 - no secrets in images
docker history invoice-agent:1.0 --no-trunc | grep -iE "aws|secret|password" || echo "clean"
docker history invoice-mock-erp:1.0 --no-trunc | grep -iE "aws|secret|password" || echo "clean"
docker history invoice-streamlit-ui:1.0 --no-trunc | grep -iE "aws|secret|password" || echo "clean"

# Check 6 - service-name resolution from inside the agent
docker compose exec agent bash -c 'getent hosts postgres qdrant redis mock-erp'
docker compose exec agent python -c "import urllib.request;print(urllib.request.urlopen('http://mock-erp:8000/health').read())"
```

Other useful commands: `docker compose logs -f <service>`, `docker compose exec <service> env`, `docker compose config`, `docker system df`.

---

## 5. Containerisation notes

- Base images pinned (`python:3.11.9-slim`, `postgres:16`, `qdrant/qdrant:v1.12.4`, `redis:7.2`); never `latest`.
- Layer order in each Dockerfile: OS packages → `requirements.txt` + `pip install` → application code, so editing code does not reinstall dependencies. Apt lists are removed in the layer that creates them.
- The agent image installs `tesseract-ocr` with `eng`, `spa`, `deu` language packs (the dataset's languages) and `poppler-utils` for PDF rendering.
- Each build context has a `.dockerignore` excluding `.env`, `__pycache__/`, `*.pyc`, `.git/` and virtual environments. No credential is in any Dockerfile or image layer — secrets arrive at runtime through `env_file`.
- All three images run as an unprivileged `appuser`. The agent's UID is a build arg (`APP_UID`, default 1000) so it can write to the bind-mounted `incoming/` and `reports/` folders.
- Logs go to stdout/stderr only; no log files inside containers.
- `restart: unless-stopped` recovers from crashes; it does **not** fix a mis-ordered startup (that's what `depends_on: condition: service_healthy` is for — a restart policy would just hide a service looping on a dead dependency). The agent and mock-erp also fail fast with a clear error if a dependency is unreachable after a few retries.

### Design questions you may be asked

- **Which services scale horizontally?** `mock-erp` (stateless, read-only) and `streamlit-ui` (stateless) scale freely. The agent's API scales, but the folder poller must run once (or move to a queue / leader election) to avoid two workers racing on the same file; the checksum `UNIQUE` constraint prevents duplicate audit rows either way. PostgreSQL scales vertically or with read replicas; Qdrant can be clustered; Redis is a cache.
- **Why a checksum, not a filename?** Vendors rename files; the same bytes arriving twice under different names have the same SHA-256, so they're skipped.
- **Why is the invoice file not in PostgreSQL?** Binary blobs in a relational database are an anti-pattern. Here the file stays on a mounted volume and the database stores its path and checksum; in production this would be object storage (S3) with the key stored in the row.
- **Why LiteLLM in-process?** Simplicity for one service. In production it would be its own gateway service so rate limiting, cost tracking and credential handling are centralised.
- **Why can Redis be lost?** It passes the test "if flushed, does anything break or does it just get slower?" — only slower. Audit state, checkpoints and the vector index all live in durable stores.

---

## 6. Repository layout

```
compose.yaml              six services, two named volumes, one network
.env.example              every variable the stack needs (copy to .env)
backend/                  agent service (Dockerfile, .dockerignore, requirements.txt)
  main.py                 FastAPI app + background folder poller
  pipeline.py             orchestration, corrections, re-validation
  agents/                 LangGraph agents (monitor, extractor, translation, validation,
                          business validation, reporting) and agents/rag/ (indexing,
                          retrieval, augmentation, generation, reflection)
  tools/                  tools the agents call (data_harvester, rule_parser, lang_bridge, ...)
  services/               qdrant_store, rag_pipeline, structured_query (SQL route)
  db/repository.py        all audit-schema reads/writes
  config/                 rules.yaml, prompts.yaml, personas.yaml, settings, logging
  data/sql/               01_init_schema.sql (erp + audit), 02_erp_data.sql (ERP master data)
  data/incoming/          monitored folder (bind-mounted)
  data/samples/           copies of the dataset for simulating new arrivals
  data/reports/           generated JSON/HTML reports (bind-mounted)
mock_erp/                 ERP API (Dockerfile, .dockerignore, requirements.txt, main.py)
frontend/                 Streamlit dashboard (Dockerfile, .dockerignore, requirements.txt, app.py)
```

## 7. Model choices

| Task | Model (configurable in `.env`) |
|---|---|
| Field extraction, reasoning, report text | Amazon Nova Lite (`LLM_REASONING_MODEL`) |
| Translation | `LLM_TRANSLATION_MODEL` |
| RAG generation and reflection | `LLM_RAG_MODEL` |
| Embeddings | Amazon Titan Text Embeddings (`EMBEDDING_MODEL`, 1536-d) |

The brief names Cohere Command R+ for translation and RAG. It is not enabled in our Bedrock account, so `.env.example` uses Claude Haiku 4.5 instead. Switching back is a one-line change to the `LLM_*` variables (for example `bedrock/cohere.command-r-plus-v1:0`).
