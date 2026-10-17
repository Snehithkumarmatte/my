SET search_path TO erp, audit, public;

CREATE SCHEMA IF NOT EXISTS erp;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS langgraph;   -- filled by LangGraph PostgresSaver.setup()

-- =========================  ERP (owned by mock-erp, read-only)  =========================
CREATE TABLE IF NOT EXISTS erp.vendors(
    vendor_id   VARCHAR(64)  PRIMARY KEY,
    vendor_name VARCHAR(255) NOT NULL,
    country     VARCHAR(64)  NOT NULL,
    currency    CHAR(3)      NOT NULL
);

CREATE TABLE IF NOT EXISTS erp.sku_master(
    item_code VARCHAR(64)  PRIMARY KEY,
    category  VARCHAR(64)  NOT NULL,
    uom       VARCHAR(32)  NOT NULL,
    gst_rate  NUMERIC(5,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS erp.purchase_orders(
    po_number VARCHAR(64) PRIMARY KEY,
    vendor_id VARCHAR(64) REFERENCES erp.vendors(vendor_id)
);

CREATE TABLE IF NOT EXISTS erp.po_line_items(
    id          BIGSERIAL PRIMARY KEY,
    po_number   VARCHAR(64) REFERENCES erp.purchase_orders(po_number),
    item_code   VARCHAR(64) REFERENCES erp.sku_master(item_code),
    description TEXT          NOT NULL,
    quantity    NUMERIC(12,3) NOT NULL,
    unit_price  NUMERIC(14,4) NOT NULL,
    currency    CHAR(3)       NOT NULL
);

-- =========================  AUDIT (agent writes, UI reads)  =========================
CREATE TABLE IF NOT EXISTS audit.invoice_audit(
    invoice_id             UUID PRIMARY KEY,
    file_path              VARCHAR(512) NOT NULL,          -- reference, never the file
    file_checksum          VARCHAR(64)  UNIQUE NOT NULL,   -- sha256, dedup
    extraction_method      VARCHAR(32)  NOT NULL DEFAULT 'pending',  -- pdfplumber|docx|tesseract|failed
    raw_extracted_text     TEXT,                           -- NULL only if extraction failed
    translated_text        TEXT,
    source_language        VARCHAR(8),
    translation_confidence NUMERIC(4,3),
    invoice_number         VARCHAR(64),
    vendor_name            VARCHAR(255),
    vendor_id              VARCHAR(64),                    -- resolved via mock-erp; no FK on purpose
    po_number              VARCHAR(64),
    invoice_date           DATE,
    currency               CHAR(3),
    total_amount           NUMERIC(14,2),
    validation_status      VARCHAR(32) NOT NULL DEFAULT 'pending',
    recommendation         VARCHAR(32) NOT NULL DEFAULT 'pending',
    error_message          TEXT,                           -- why processing failed, if it did
    extracted_raw          JSONB,                          -- variable extras + email meta.json
    thread_id              VARCHAR(128),                   -- link to langgraph checkpoint
    processed_at           TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT chk_validation_status
        CHECK (validation_status IN ('pending','passed','partial','failed','error')),
    CONSTRAINT chk_recommendation
        CHECK (recommendation IN ('pending','approve','manual_review','reject'))
);

CREATE TABLE IF NOT EXISTS audit.invoice_line_items(
    id          BIGSERIAL PRIMARY KEY,
    invoice_id  UUID REFERENCES audit.invoice_audit(invoice_id),
    line_number INTEGER,
    item_code   VARCHAR(64),
    description TEXT,
    quantity    NUMERIC(12,3),
    unit_price  NUMERIC(14,4),
    line_total  NUMERIC(14,2)
);

CREATE TABLE IF NOT EXISTS audit.discrepancies(
    id            BIGSERIAL PRIMARY KEY,
    invoice_id    UUID REFERENCES audit.invoice_audit(invoice_id),
    check_type    VARCHAR(16) NOT NULL DEFAULT 'business',  -- data | business
    line_number   INTEGER,                                  -- NULL = header-level issue
    item_code     VARCHAR(64),
    field_name    VARCHAR(64) NOT NULL,                     -- unit_price | quantity | currency | total | invoice_number ...
    invoice_value TEXT,
    erp_value     TEXT,
    deviation_pct NUMERIC(8,4),
    severity      VARCHAR(16) NOT NULL,                     -- low | medium | high
    detected_at   TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT chk_check_type CHECK (check_type IN ('data','business')),
    CONSTRAINT chk_severity   CHECK (severity IN ('low','medium','high'))
);

-- corrections are INSERTs here, never UPDATEs of the original extraction
CREATE TABLE IF NOT EXISTS audit.human_feedback(
    id              BIGSERIAL PRIMARY KEY,
    invoice_id      UUID REFERENCES audit.invoice_audit(invoice_id),
    field_name      VARCHAR(64)  NOT NULL,
    original_value  TEXT,
    corrected_value TEXT         NOT NULL,
    corrected_by    VARCHAR(128) NOT NULL,
    corrected_at    TIMESTAMPTZ  DEFAULT now(),
    revalidated     BOOLEAN      DEFAULT FALSE
);

-- =========================  INDEXES  =========================
CREATE INDEX IF NOT EXISTS idx_po_lines_po           ON erp.po_line_items(po_number);
CREATE INDEX IF NOT EXISTS idx_line_items_invoice    ON audit.invoice_line_items(invoice_id);
CREATE INDEX IF NOT EXISTS idx_discrepancies_invoice ON audit.discrepancies(invoice_id);
CREATE INDEX IF NOT EXISTS idx_feedback_invoice      ON audit.human_feedback(invoice_id);
CREATE INDEX IF NOT EXISTS idx_audit_vendor          ON audit.invoice_audit(vendor_name);
CREATE INDEX IF NOT EXISTS idx_audit_recommendation  ON audit.invoice_audit(recommendation);
CREATE INDEX IF NOT EXISTS idx_audit_processed       ON audit.invoice_audit(processed_at);