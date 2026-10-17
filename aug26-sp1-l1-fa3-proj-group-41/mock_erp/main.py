"""Mock ERP API – read-only HTTP facade over the `erp` schema.

Response shapes (unchanged from the old JSON files):
    PO     -> {po_number, vendor_id, line_items: [{item_code, description, qty, unit_price, currency}]}
    vendor -> {vendor_id, vendor_name, country, currency}
    SKU    -> {item_code, category, uom, gst_rate}

DATABASE_URL must come from the environment (env_file in compose). No default on purpose.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from decimal import Decimal

import psycopg
from fastapi import APIRouter, FastAPI, HTTPException, Query
from psycopg.rows import dict_row

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)  # stdout only
logger = logging.getLogger("mock-erp")

DATABASE_URL = os.environ.get("DATABASE_URL")


# ---------- startup: fail fast if the DB or schema isn't there ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set – refusing to start")
    for attempt in range(1, 6):
        try:
            with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
                conn.execute("SELECT 1 FROM erp.vendors LIMIT 1")  # proves schema is loaded
            logger.info("Connected to ERP database")
            break
        except psycopg.Error as exc:
            logger.warning("ERP DB not ready (attempt %d/5): %s", attempt, exc)
            time.sleep(2)
    else:
        raise RuntimeError("Cannot reach ERP database – exiting")
    yield


# ---------- database helpers ----------
def _query(sql: str, params: tuple = ()) -> list[dict]:
    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5) as conn:
            rows = conn.execute(sql, params).fetchall()
    except psycopg.OperationalError as exc:
        logger.error("ERP database not reachable: %s", exc)
        raise HTTPException(status_code=503, detail="ERP database is not reachable")
    return [_clean(r) for r in rows]


def _clean(row: dict) -> dict:
    """NUMERIC -> float, CHAR(3) padding stripped."""
    out = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            v = float(v)
        elif isinstance(v, str):
            v = v.strip()
        out[k] = v
    return out


def _not_found(what: str, key: str):
    raise HTTPException(status_code=404, detail=f"{what} '{key}' not found in ERP system")


PO_SQL = "SELECT po_number, vendor_id FROM erp.purchase_orders"
LINES_SQL = """
    SELECT po_number, item_code, description, quantity AS qty, unit_price, currency
    FROM erp.po_line_items
"""
VENDOR_SQL = "SELECT vendor_id, vendor_name, country, currency FROM erp.vendors"
SKU_SQL = "SELECT item_code, category, uom, gst_rate FROM erp.sku_master"


def _attach_lines(orders: list[dict], lines: list[dict]) -> list[dict]:
    by_po: dict[str, list[dict]] = {}
    for line in lines:
        by_po.setdefault(line.pop("po_number"), []).append(line)
    for order in orders:
        order["line_items"] = by_po.get(order["po_number"], [])
    return orders


# ---------- routes ----------
router = APIRouter()


@router.get("/health")
def health_check():
    counts = _query("""
        SELECT (SELECT count(*) FROM erp.purchase_orders) AS po_records,
               (SELECT count(*) FROM erp.vendors)         AS vendors,
               (SELECT count(*) FROM erp.sku_master)      AS skus
    """)[0]
    return {"status": "healthy", **counts}


@router.get("/po")
def list_purchase_orders():
    orders = _query(PO_SQL + " ORDER BY po_number")
    lines = _query(LINES_SQL + " ORDER BY po_number, id")
    return _attach_lines(orders, lines)


@router.get("/po/{po_number}")
def get_purchase_order(po_number: str):
    key = po_number.strip().upper()
    orders = _query(PO_SQL + " WHERE upper(po_number) = %s", (key,))
    if not orders:
        _not_found("Purchase Order", po_number)
    lines = _query(LINES_SQL + " WHERE upper(po_number) = %s ORDER BY id", (key,))
    return _attach_lines(orders, lines)[0]


@router.get("/vendors")
def list_vendors(name: str | None = Query(None, description="Case-insensitive partial name match")):
    """Invoices carry vendor *names*, so ?name= lets the agent resolve them to an ID."""
    if name:
        return _query(VENDOR_SQL + " WHERE vendor_name ILIKE %s ORDER BY vendor_id",
                      (f"%{name.strip()}%",))
    return _query(VENDOR_SQL + " ORDER BY vendor_id")


@router.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    rows = _query(VENDOR_SQL + " WHERE upper(vendor_id) = %s", (vendor_id.strip().upper(),))
    if not rows:
        _not_found("Vendor", vendor_id)
    return rows[0]


@router.get("/skus")
def list_skus():
    return _query(SKU_SQL + " ORDER BY item_code")


@router.get("/sku/{item_code}")
def get_sku(item_code: str):
    rows = _query(SKU_SQL + " WHERE upper(item_code) = %s", (item_code.strip().upper(),))
    if not rows:
        _not_found("SKU", item_code)
    return rows[0]


app = FastAPI(title="Mock ERP API", version="2.1.0", lifespan=lifespan)
app.include_router(router)                                                  # spec paths: /po/{po}
app.include_router(router, prefix="/api/erp", include_in_schema=False)      # old paths keep working

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))