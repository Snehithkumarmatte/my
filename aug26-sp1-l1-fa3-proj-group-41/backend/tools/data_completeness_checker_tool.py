"""Data completeness checker tool.

Checks a translated invoice against the rules in config/rules.yaml:
required fields, data types, date format, currency, line item maths
(qty x unit price = total) and invoice maths (lines add up to the
subtotal, subtotal + tax = total).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import yaml
from langchain_core.tools import tool

from backend.models.validation import FieldIssue, LineItemIssue, ValidationResult

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
RULES_PATH = BASE_DIR / "config" / "rules.yaml"

DEFAULT_RULES = {
    "required_fields": {
        "header": ["invoice_no", "invoice_date", "vendor_id", "currency", "total_amount"],
        "line_item": ["item_code", "description", "qty", "unit_price", "total"],
    },
    "data_types": {
        "invoice_no": "str", "invoice_date": "date", "vendor_id": "str", "currency": "str",
        "total_amount": "float", "qty": "float", "unit_price": "float", "total": "float",
    },
    "tolerances": {
        "price_difference_percent": 5,
        "quantity_difference_percent": 0,
        "tax_difference_percent": 2,
    },
}

# rules.yaml field name -> key used in the translated invoice
HEADER_KEYS = {"vendor_name": "translated_vendor_name"}
LINE_KEYS = {"description": "translated_description"}
LINE_ONLY_FIELDS = ("qty", "unit_price", "total")

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d-%b-%Y",        # 25-May-2025
    "%b %d, %Y",       # May 25, 2025
    "%B %d, %Y",       # January 15, 2025
    "%d %B %Y",        # 25 January 2025
    "%d %b %Y",        # 25 Jan 2025
    "%B %d %Y",        # January 15 2025
    "%b %d %Y",        # Jan 15 2025
    "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y", "%Y.%m.%d",
    "%d %B, %Y",       # 25 January, 2025
]

_rules: dict | None = None


@tool
def data_completeness_checker_tool(translation_result: dict) -> dict:
    """Validate a translated invoice for completeness and correctness.

    Runs every check from config/rules.yaml and returns the issues found.

    Args:
        translation_result: TranslationResult as a dict.

    Returns:
        ValidationResult as a dict.
    """
    rules = _load_rules()
    filename = translation_result.get("filename", "?")
    invoice = translation_result.get("translated_invoice") or {}
    lines = invoice.get("line_items", [])

    header_issues: list[FieldIssue] = []
    line_issues: list[LineItemIssue] = []

    header_issues += check_required_header_fields(invoice, rules)
    header_issues += check_header_types(invoice, rules)
    header_issues += check_date(invoice)
    header_issues += check_currency(invoice, rules)
    line_issues += check_required_line_fields(lines, rules)
    line_issues += check_line_math(lines, rules)
    header_issues += check_invoice_math(invoice, lines, rules)

    rules_applied = [
        "required_header_fields", "header_data_types", "date_format", "currency_validation",
        "required_line_item_fields", "line_item_math", "invoice_math",
    ]

    all_issues = header_issues + line_issues
    errors = sum(1 for i in all_issues if i.severity == "error")
    warnings = sum(1 for i in all_issues if i.severity == "warning")

    if errors:
        status = "failed"
    elif warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"

    logger.info("%s - %s | errors: %d | warnings: %d", filename, status.upper(), errors, warnings)

    return ValidationResult(
        email_id=translation_result.get("email_id", "unknown"),
        filename=filename,
        file_path=translation_result.get("file_path", ""),
        is_valid=errors == 0,
        validation_status=status,
        header_issues=header_issues,
        line_item_issues=line_issues,
        total_errors=errors,
        total_warnings=warnings,
        rules_applied=rules_applied,
        source_language=translation_result.get("source_language"),
    ).model_dump()


# rules

def _load_rules() -> dict:
    """Read rules.yaml once and keep it in memory."""
    global _rules
    if _rules is None:
        if RULES_PATH.exists():
            with open(RULES_PATH, encoding="utf-8") as fh:
                _rules = yaml.safe_load(fh)
            logger.info("RULES: Loaded validation rules from %s", RULES_PATH)
        else:
            logger.warning("Rules file not found at %s - using defaults", RULES_PATH)
            _rules = DEFAULT_RULES
    return _rules


def _is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _pct_diff(actual: float, expected: float) -> float:
    return abs((actual - expected) / expected) * 100


# header checks

def check_required_header_fields(invoice: dict, rules: dict) -> list[FieldIssue]:
    issues = []
    for field in rules.get("required_fields", {}).get("header", []):
        value = invoice.get(HEADER_KEYS.get(field, field))
        if _is_empty(value):
            issues.append(FieldIssue(
                field_name=field,
                issue_type="missing",
                severity="error",
                message=f"Required header field '{field}' is missing or empty",
                expected="non-empty value",
                actual="null" if value is None else str(value),
            ))
    return issues


def check_header_types(invoice: dict, rules: dict) -> list[FieldIssue]:
    """Numbers must be numbers (error), text must be text (warning).
    Dates are checked separately and missing values by the required check."""
    issues = []
    for field, expected in rules.get("data_types", {}).items():
        if field in LINE_ONLY_FIELDS:
            continue
        value = invoice.get(HEADER_KEYS.get(field, field))
        if value is None:
            continue
        type_name = type(value).__name__

        if expected == "float" and not isinstance(value, (int, float)):
            issues.append(FieldIssue(
                field_name=field, issue_type="wrong_type", severity="error",
                message=f"Field '{field}' expected numeric but got {type_name}: {value}",
                expected="float", actual=type_name,
            ))
        elif expected == "str" and not isinstance(value, str):
            issues.append(FieldIssue(
                field_name=field, issue_type="wrong_type", severity="warning",
                message=f"Field '{field}' expected string but got {type_name}",
                expected="str", actual=type_name,
            ))
    return issues


def check_date(invoice: dict) -> list[FieldIssue]:
    value = invoice.get("invoice_date")
    if value is None:
        return []

    if not isinstance(value, str):
        return [FieldIssue(
            field_name="invoice_date", issue_type="wrong_type", severity="error",
            message=f"invoice_date is not a string: {value}",
            expected="date string", actual=type(value).__name__,
        )]

    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return []
        except ValueError:
            pass

    return [FieldIssue(
        field_name="invoice_date", issue_type="invalid_value", severity="warning",
        message=f"invoice_date '{value}' could not be parsed with common date formats — may need manual review",
        expected="parseable date string", actual=value,
    )]


def check_currency(invoice: dict, rules: dict) -> list[FieldIssue]:
    """Currency must be in accepted_currencies. Symbols like € are mapped
    to their code first using currency_symbol_map."""
    currency = invoice.get("currency")
    accepted = rules.get("accepted_currencies", [])
    if currency is None or not isinstance(currency, str) or not accepted:
        return []

    code = rules.get("currency_symbol_map", {}).get(currency.strip(), currency.strip()).upper()
    if code in [c.upper() for c in accepted]:
        return []

    action = rules.get("validation_policies", {}).get("invalid_currency_action", "reject")
    return [FieldIssue(
        field_name="currency",
        issue_type="invalid_value",
        severity="error" if action == "reject" else "warning",
        message=f"Currency '{currency}' is not in the accepted list: {accepted}",
        expected=f"one of {accepted}",
        actual=currency,
    )]


# line item checks

def check_required_line_fields(lines: list[dict], rules: dict) -> list[LineItemIssue]:
    issues = []
    required = rules.get("required_fields", {}).get("line_item", [])
    for idx, line in enumerate(lines):
        for field in required:
            value = line.get(LINE_KEYS.get(field, field))
            if _is_empty(value):
                issues.append(LineItemIssue(
                    line_index=idx,
                    field_name=field,
                    issue_type="missing",
                    severity="error",
                    message=f"Line item [{idx}]: required field '{field}' is missing or empty",
                    expected="non-empty value",
                    actual="null" if value is None else str(value),
                ))
    return issues


def check_line_math(lines: list[dict], rules: dict) -> list[LineItemIssue]:
    """qty x unit_price should equal the line total, within the tolerance."""
    tolerances = rules.get("tolerances", {})
    tolerance = tolerances.get("line_item_math_tolerance_percent",
                               tolerances.get("price_difference_percent", 5))
    issues = []

    for idx, line in enumerate(lines):
        qty, price, total = line.get("qty"), line.get("unit_price"), line.get("total")
        if qty is None or price is None or total is None:
            continue
        try:
            expected, actual = float(qty) * float(price), float(total)
        except (ValueError, TypeError):
            continue          # non-numeric values are reported by the type checks

        if expected == 0 and actual == 0:
            continue
        diff = 100.0 if expected == 0 else _pct_diff(actual, expected)
        if diff > tolerance:
            issues.append(LineItemIssue(
                line_index=idx,
                field_name="total",
                issue_type="math_error",
                severity="error",
                message=(f"Line item [{idx}]: qty({qty}) × unit_price({price}) = {expected:.2f}, "
                         f"but total = {actual:.2f} (diff: {diff:.1f}%, tolerance: {tolerance}%)"),
                expected=f"{expected:.2f}",
                actual=f"{actual:.2f}",
            ))
    return issues


# invoice level maths

def check_invoice_math(invoice: dict, lines: list[dict], rules: dict) -> list[FieldIssue]:
    """Line totals should add up to the subtotal, and subtotal + tax to the total."""
    tolerances = rules.get("tolerances", {})
    price_tolerance = tolerances.get("price_difference_percent", 5)
    tax_tolerance = tolerances.get("tax_difference_percent", 2)
    subtotal = invoice.get("subtotal")
    issues = []

    # 1. sum of line totals vs subtotal (only when every line has a numeric total)
    line_sum = _sum_line_totals(lines)
    if line_sum is not None and subtotal is not None and lines:
        try:
            sub = float(subtotal)
            if sub != 0:
                diff = _pct_diff(line_sum, sub)
                if diff > price_tolerance:
                    issues.append(FieldIssue(
                        field_name="subtotal", issue_type="math_error", severity="warning",
                        message=f"Sum of line totals ({line_sum:.2f}) ≠ subtotal ({sub:.2f}) — diff: {diff:.1f}%",
                        expected=f"{line_sum:.2f}", actual=f"{sub:.2f}",
                    ))
        except (ValueError, TypeError):
            pass

    # 2. subtotal + tax vs total_amount
    total_amount, tax = invoice.get("total_amount"), invoice.get("tax_amount")
    if subtotal is not None and total_amount is not None:
        try:
            sub, total = float(subtotal), float(total_amount)
            tax_value = float(tax) if tax is not None else 0.0
            expected = sub + tax_value
            if total != 0:
                diff = _pct_diff(expected, total)
                if diff > tax_tolerance:
                    issues.append(FieldIssue(
                        field_name="total_amount", issue_type="math_error", severity="warning",
                        message=(f"subtotal ({sub:.2f}) + tax ({tax_value:.2f}) = {expected:.2f}, "
                                 f"but total_amount = {total:.2f} — diff: {diff:.1f}%"),
                        expected=f"{expected:.2f}", actual=f"{total:.2f}",
                    ))
        except (ValueError, TypeError):
            pass

    return issues


def _sum_line_totals(lines: list[dict]) -> float | None:
    """Sum of line totals, or None if any line is missing a usable total."""
    total = 0.0
    for line in lines:
        try:
            total += float(line["total"])
        except (KeyError, ValueError, TypeError):
            return None
    return total