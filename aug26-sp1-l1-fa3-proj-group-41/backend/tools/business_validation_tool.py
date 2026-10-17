"""Business validation tool.

Compares a translated invoice with the ERP records (purchase order,

vendor master and SKU master) and lists every mismatch it finds.

All ERP data comes from the mock ERP API, never from local files.

"""

from __future__ import annotations

import logging

from pathlib import Path

import httpx

import yaml

from langchain_core.tools import tool

from backend.models.business_validation import (

    BusinessValidationResult,

    ERPLookupResult,

    LineItemDiscrepancy,

)

from backend.config.settings import ERP_API_BASE_URL

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/

RULES_PATH = BASE_DIR / "config" / "rules.yaml"

DEFAULT_TOLERANCES = {"price_difference_percent": 5, "quantity_difference_percent": 0}

_rules: dict | None = None



@tool

def business_validation_tool(translation_result: dict, validation_result: dict | None = None) -> dict:

    """Check a translated invoice against the ERP purchase order.

    Looks for vendor, currency, quantity and unit price mismatches, items

    that are not on the PO, and PO items that are missing from the invoice.

    Args:

        translation_result: TranslationResult as a dict.

        validation_result: ValidationResult from the completeness checker (optional).

    Returns:

        BusinessValidationResult as a dict.

    """

    filename = translation_result.get("filename", "?")

    invoice = translation_result.get("translated_invoice") or {}

    po_number = invoice.get("po_number")

    vendor_id = invoice.get("vendor_id")

    invoice_currency = invoice.get("currency")

    invoice_lines = invoice.get("line_items", [])

    # fields every result shares

    base = {

        "email_id": translation_result.get("email_id", "unknown"),

        "filename": filename,

        "file_path": translation_result.get("file_path", ""),

        "source_language": translation_result.get("source_language"),

    }

    # no PO number -> nothing to compare against

    if not po_number:

        logger.warning("%s: no PO number found - skipping business validation", filename)

        return BusinessValidationResult(

            **base,

            is_valid=False,

            validation_status="skipped",

            erp_lookup=ERPLookupResult(invoice_line_items_count=len(invoice_lines)),

            discrepancies=[_header_issue(

                "po_number", "item_not_in_po",

                "No PO number on invoice — cannot cross-reference with ERP",

            )],

            total_errors=1,

        ).model_dump()

    # load everything we need from the ERP

    po = _fetch_po(po_number)

    vendor = _fetch_vendor(vendor_id) if vendor_id else None

    invoice_codes = {str(line.get("item_code") or "").upper() for line in invoice_lines}

    skus = {code: _fetch_sku(code) for code in invoice_codes if code}

    if po is None:

        logger.warning("[PO NOT FOUND] %s: PO '%s' not found in ERP", filename, po_number)

        return BusinessValidationResult(

            **base,

            is_valid=False,

            validation_status="po_not_found",

            erp_lookup=ERPLookupResult(

                po_number=po_number,

                po_found=False,

                vendor_id=vendor_id,

                vendor_found=vendor is not None,

                invoice_line_items_count=len(invoice_lines),

            ),

            discrepancies=[_header_issue(

                "po_number", "item_not_in_po",

                f"PO '{po_number}' not found in ERP system",

                invoice_value=po_number,

            )],

            total_errors=1,

        ).model_dump()

    po_lines = po.get("line_items", [])

    po_vendor_id = po.get("vendor_id")

    po_currency = po_lines[0].get("currency") if po_lines else None

    # PO lines looked up by item code, and by description for lines without a code

    po_by_code = {line["item_code"].upper(): line for line in po_lines if line.get("item_code")}

    po_by_desc = {

        (line.get("description") or "").strip().lower(): line

        for line in po_lines if (line.get("description") or "").strip()

    }

    issues: list[LineItemDiscrepancy] = []

    # header checks: vendor and currency

    if vendor_id and po_vendor_id and vendor_id.upper() != po_vendor_id.upper():

        issues.append(_header_issue(

            "vendor_id", "vendor_mismatch",

            f"Invoice vendor '{vendor_id}' does not match PO vendor '{po_vendor_id}'",

            invoice_value=vendor_id, erp_value=po_vendor_id,

        ))

    if vendor_id and vendor is None:

        issues.append(_header_issue(

            "vendor_id", "vendor_mismatch",

            f"Vendor '{vendor_id}' not found in ERP Vendor Master",

            severity="warning", invoice_value=vendor_id,

        ))

    if invoice_currency and po_currency and invoice_currency.upper() != po_currency.upper():

        issues.append(_header_issue(

            "currency", "currency_mismatch",

            f"Invoice currency '{invoice_currency}' does not match ERP PO currency '{po_currency}'",

            invoice_value=invoice_currency, erp_value=po_currency,

        ))

    # line item checks

    tolerances = _load_rules().get("tolerances", {})

    price_tolerance = tolerances.get("price_difference_percent", 5)

    qty_tolerance = tolerances.get("quantity_difference_percent", 0)

    matched_count = 0

    matched_po_codes: set[str] = set()

    for idx, line in enumerate(invoice_lines):

        code = (line.get("item_code") or "").upper()

        desc = (

            line.get("translated_description")

            or line.get("description")

            or line.get("original_description")

            or ""

        ).strip().lower()

        # find the matching PO line: by item code first, then by description

        po_line, matched_by = None, ""

        if code and code in po_by_code:

            po_line, matched_by = po_by_code[code], "item_code"

        elif desc:

            po_line = _match_by_description(desc, po_by_desc)

            if po_line:

                matched_by = "description"

                logger.info(

                    "     Line [%d]: item_code missing - matched by description '%s' -> ERP item '%s'",

                    idx, desc, po_line.get("item_code", "?"),

                )

        if po_line is None:

            issues.append(LineItemDiscrepancy(

                line_index=idx,

                item_code=line.get("item_code"),

                field_name="item_code",

                discrepancy_type="item_not_in_po",

                severity="error",

                message=(f"Line [{idx}]: item '{line.get('item_code') or '(no code)'}' "

                         f"(desc: '{desc or '?'}') not found in ERP PO '{po_number}'"),

                invoice_value=line.get("item_code") or desc,

            ))

            if code and skus.get(code) is None:

                issues.append(LineItemDiscrepancy(

                    line_index=idx,

                    item_code=line.get("item_code"),

                    field_name="item_code",

                    discrepancy_type="item_not_in_sku_master",

                    severity="warning",

                    message=f"Line [{idx}]: item '{line.get('item_code')}' not found in ERP SKU Master",

                    invoice_value=line.get("item_code"),

                ))

            continue

        po_code = po_line.get("item_code", "").upper()

        matched_po_codes.add(po_code)

        label = code or po_code or desc

        line_ok = True

        for field, kind, tolerance in (

            ("qty", "quantity_mismatch", qty_tolerance),

            ("unit_price", "price_mismatch", price_tolerance),

        ):

            issue = _compare_numbers(

                idx, label, field, kind, tolerance, matched_by,

                invoice_value=line.get(field),

                erp_value=po_line.get(field),

                item_code=line.get("item_code") or po_line.get("item_code"),

            )

            if issue:

                issues.append(issue)

                line_ok = False

        if line_ok:

            matched_count += 1

    # PO items the invoice never mentions

    for po_code, po_line in po_by_code.items():

        if po_code not in matched_po_codes:

            issues.append(LineItemDiscrepancy(

                line_index=-1,

                item_code=po_line.get("item_code"),

                field_name="item_code",

                discrepancy_type="missing_item_on_invoice",

                severity="warning",

                message=(f"ERP PO item '{po_line.get('item_code')}' ({po_line.get('description', '?')}) "

                         f"is in the PO but missing from the invoice"),

                erp_value=po_line.get("item_code"),

            ))

    errors = sum(1 for i in issues if i.severity == "error")

    warnings = sum(1 for i in issues if i.severity == "warning")

    status = "discrepancies_found" if issues else "matched"

    logger.info(

        "%s - %s | matched: %d/%d | errors: %d | warnings: %d",

        filename, status.upper(), matched_count, len(invoice_lines), errors, warnings,

    )

    return BusinessValidationResult(

        **base,

        is_valid=errors == 0,

        validation_status=status,

        erp_lookup=ERPLookupResult(

            po_number=po_number,

            po_found=True,

            vendor_id=vendor_id,

            vendor_found=vendor is not None,

            erp_line_items_count=len(po_lines),

            invoice_line_items_count=len(invoice_lines),

            erp_currency=po_currency,

            erp_vendor_name=vendor.get("vendor_name") if vendor else None,

        ),

        discrepancies=issues,

        total_errors=errors,

        total_warnings=warnings,

        matched_line_items=matched_count,

    ).model_dump()



# helpers

def _header_issue(field: str, kind: str, message: str, severity: str = "error",

                  invoice_value: str | None = None, erp_value: str | None = None) -> LineItemDiscrepancy:

    """A discrepancy that belongs to the invoice header, not to one line."""

    return LineItemDiscrepancy(

        line_index=-1, field_name=field, discrepancy_type=kind, severity=severity,

        message=message, invoice_value=invoice_value, erp_value=erp_value,

    )



def _percent_diff(invoice_value: float, erp_value: float) -> float:

    if erp_value != 0:

        return abs((invoice_value - erp_value) / erp_value) * 100

    return 100.0 if invoice_value != 0 else 0.0



def _compare_numbers(idx, label, field, kind, tolerance, matched_by,

                     invoice_value, erp_value, item_code) -> LineItemDiscrepancy | None:

    """Return a discrepancy if the invoice value is outside the tolerance."""

    if invoice_value is None or erp_value is None:

        return None

    try:

        inv, erp = float(invoice_value), float(erp_value)

    except (ValueError, TypeError):

        return None

    diff = _percent_diff(inv, erp)

    if diff <= tolerance:

        return None

    name = "qty" if field == "qty" else "price"

    return LineItemDiscrepancy(

        line_index=idx,

        item_code=item_code,

        field_name=field,

        discrepancy_type=kind,

        severity="error",

        message=(f"Line [{idx}] {label}: {name} mismatch — invoice={inv}, ERP={erp} "

                 f"(diff: {diff:.1f}%, tolerance: {tolerance}%) [matched by {matched_by}]"),

        invoice_value=str(inv),

        erp_value=str(erp),

        difference_pct=round(diff, 2),

    )



def _match_by_description(desc: str, po_by_desc: dict[str, dict]) -> dict | None:

    """Match a line without an item code to a PO line by its description.

    Tries an exact match, then one description containing the other,

    then the best word overlap (needs at least 60%).

    """

    desc = desc.strip().lower()

    if not desc:

        return None

    if desc in po_by_desc:

        return po_by_desc[desc]

    for po_desc, po_line in po_by_desc.items():

        if desc in po_desc or po_desc in desc:

            return po_line

    words = set(desc.split())

    best_line, best_ratio = None, 0.0

    for po_desc, po_line in po_by_desc.items():

        po_words = set(po_desc.split())

        if not po_words or not words:

            continue

        ratio = len(words & po_words) / max(len(words), len(po_words))

        if ratio > best_ratio:

            best_line, best_ratio = po_line, ratio

    return best_line if best_ratio >= 0.6 else None



def _load_rules() -> dict:

    """Read rules.yaml once and keep it in memory."""

    global _rules

    if _rules is None:

        if RULES_PATH.exists():

            with open(RULES_PATH, encoding="utf-8") as fh:

                _rules = yaml.safe_load(fh)

        else:

            _rules = {"tolerances": dict(DEFAULT_TOLERANCES)}

    return _rules



def _erp_get(path: str, what: str) -> dict | None:

    """GET a record from the mock ERP. Returns None when it does not exist."""

    try:

        resp = httpx.get(f"{ERP_API_BASE_URL}{path}", timeout=5.0)

    except httpx.HTTPError as exc:

        raise RuntimeError(f"ERP API unavailable for {what}: {exc}") from exc

    if resp.status_code == 404:

        return None

    if resp.status_code != 200:

        raise RuntimeError(f"ERP API returned {resp.status_code} for {what}")

    return resp.json()



def _fetch_po(po_number: str) -> dict | None:

    return _erp_get(f"/po/{po_number}", f"PO {po_number}")



def _fetch_vendor(vendor_id: str) -> dict | None:

    return _erp_get(f"/vendor/{vendor_id}", f"vendor {vendor_id}")



def _fetch_sku(item_code: str) -> dict | None:

    return _erp_get(f"/sku/{item_code}", f"SKU {item_code}")