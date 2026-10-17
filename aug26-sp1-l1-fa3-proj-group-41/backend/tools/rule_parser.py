"""Deterministic (no-LLM) invoice parser.

Used by the data harvester when Bedrock is unavailable or returns unusable JSON,
so a model outage never stops extraction. It understands the label vocabulary of
the dataset languages (English, Spanish, German), currency symbols, pipe-separated
line-item tables, OCR'd space-separated rows and single-item "Quantity / Unit Price"
invoices. Dates are normalised to ISO (YYYY-MM-DD).

It returns the same dict shape the LLM is asked to produce, so the rest of the
pipeline does not care which parser ran.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# ------------------------------------------------------------------ vocabulary

_INVOICE_NO_LABELS = r"(?:invoice\s*(?:no\.?|number|#)?|factura\s*(?:n[oº°]\.?|n[uú]mero)?|rechnungs?\s*-?\s*(?:nummer|nr\.?))"
_DATE_LABELS = r"(?:invoice\s*date|date|fecha(?:\s*de\s*factura)?|rechnungsdatum|datum)"
_SUBTOTAL_LABELS = r"(?:sub\s*-?total|zwischensumme|base\s*imponible|nettobetrag)"
_TAX_LABELS = r"(?:tax|vat|gst|iva|mwst|ust|impuesto)"
_TOTAL_LABELS = r"(?:total\s*amount\s*due|total\s*amount|amount\s*due|grand\s*total|total|gesamtbetrag|gesamtsumme|importe\s*total)"

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "₹": "INR", "£": "GBP", "¥": "JPY"}
_ISO_CODES = ("USD", "EUR", "INR", "GBP", "JPY", "VND", "AUD", "CAD", "CHF", "SGD")

_MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Spanish
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    # German
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6, "juli": 7,
    "oktober": 10, "dezember": 12,
}

_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"
_SKIP_ROW = re.compile(rf"^\s*(?:{_SUBTOTAL_LABELS}|{_TAX_LABELS}|{_TOTAL_LABELS})\b", re.I)


# ------------------------------------------------------------------ helpers

def _num(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", text)
    if not cleaned or not re.search(r"\d", cleaned):
        return None
    # "1.234,56" (European) -> 1234.56 ; "1,234.56" -> 1234.56
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        cleaned = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalise_date(text: str | None) -> str | None:
    """Return YYYY-MM-DD for the common formats in the dataset, else the input."""
    if not text:
        return None
    # keep only the date-looking part of the value ("03-Dec-2024 Payment Ref ..." -> "03-Dec-2024")
    found = re.search(
        r"\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{4}"
        r"|\d{1,2}[\s\-./]+(?:de\s+)?[A-Za-zäéíóú]+\.?[\s\-./]+(?:de\s+)?\d{4}"
        r"|[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}",
        text,
    )
    raw = (found.group(0) if found else text).strip().rstrip(".")
    lowered = raw.lower().replace(" de ", " ").replace(",", " ")
    lowered = re.sub(r"\s+", " ", lowered)

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass

    # 14 March 2025 | 10 abril 2025 | 25-May-2025 | March 14 2025
    m = re.match(r"^(\d{1,2})[\s\-./]+([a-zäéíóú]+)\.?[\s\-./]+(\d{4})$", lowered)
    if m and m.group(2) in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^([a-zäéíóú]+)\.?\s+(\d{1,2})\s+(\d{4})$", lowered)
    if m and m.group(1) in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
    return raw


def _labelled(text: str, labels: str, value: str = r"(.+)") -> str | None:
    """Value after 'Label:' at the start of a line (first match)."""
    m = re.search(rf"^\s*{labels}\s*(?:\([^)]*\))?\s*[:#]?\s*{value}\s*$", text, re.I | re.M)
    return m.group(1).strip() if m else None


def detect_currency(text: str) -> str | None:
    """Only a currency that is visibly present - never inferred from country or vendor."""
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    for code in _ISO_CODES:
        if re.search(rf"\b{code}\b", text):
            return code
    return None


def detect_language(text: str) -> str | None:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text) if text.strip() else None
    except Exception:  # noqa: BLE001 - language is a hint only
        return None


# ------------------------------------------------------------------ line items

def _pipe_rows(text: str) -> list[dict]:
    items = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 4:
            continue
        numbers = [_num(c) for c in cells[-3:]]
        if any(n is None for n in numbers):
            continue                                   # header row
        code_match = re.match(r"^[A-Z]{2,5}-?\d+[A-Z0-9-]*$", cells[0])
        code = cells[0] if code_match else None
        desc_cells = cells[1:-3] if code else cells[:-3]
        items.append({
            "item_code": code,
            "description": " ".join(desc_cells) or None,
            "qty": numbers[0], "unit_price": numbers[1], "total": numbers[2],
        })
    return items


def _space_rows(text: str) -> list[dict]:
    """OCR rows like 'Fuel Surcharge 1 120 120' or 'SKU-1 Boxes 25 4.5 112.5'."""
    items = []
    pattern = re.compile(rf"^\s*(?:([A-Z]{{2,5}}-\d+)\s+)?([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .&/()-]*?)\s+({_NUM})\s+[$€£₹]?({_NUM})\s+[$€£₹]?({_NUM})\s*$")
    for line in text.splitlines():
        if _SKIP_ROW.match(line):
            continue
        m = pattern.match(line)
        if not m:
            continue
        qty, price, total = _num(m.group(3)), _num(m.group(4)), _num(m.group(5))
        if None in (qty, price, total):
            continue
        # accept only rows whose arithmetic is roughly consistent (filters address lines etc.)
        if total and abs(qty * price - total) > max(1.0, 0.05 * total):
            continue
        items.append({"item_code": m.group(1), "description": m.group(2).strip(),
                      "qty": qty, "unit_price": price, "total": total})
    return items


def _single_item(text: str) -> list[dict]:
    """'Description: X / Quantity: 200 units / Unit Price: $1.25' layout."""
    qty = _num(_labelled(text, r"(?:quantity|qty|cantidad|menge)"))
    price = _num(_labelled(text, r"(?:unit\s*price|precio(?:\s*unitario)?|einzelpreis|preis)"))
    if qty is None or price is None:
        return []
    desc = _labelled(text, r"(?:description|descripci[oó]n|beschreibung)")
    code = _labelled(text, r"(?:item\s*code|sku|art[ií]culo|artikelcode)")
    return [{"item_code": code, "description": desc, "qty": qty, "unit_price": price,
             "total": round(qty * price, 2)}]


# ------------------------------------------------------------------ public API

def parse_invoice_text(text: str) -> dict[str, Any]:
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    invoice_no = _labelled(text, _INVOICE_NO_LABELS, r"([A-Z0-9][A-Z0-9/\-]{2,})")
    if not invoice_no:
        m = re.search(r"\b((?:INV|FAC|RE)-[A-Z0-9\-]{3,})\b", text)
        invoice_no = m.group(1) if m else None

    po = re.search(r"\bPO[-\s]?(\d{3,})\b", text, re.I)
    vendor_id = re.search(r"\b(VEND-\d+)\b", text, re.I)

    items = _pipe_rows(text) or _single_item(text) or _space_rows(text)

    subtotal = _num(_labelled(text, _SUBTOTAL_LABELS))
    tax = _num(_labelled(text, _TAX_LABELS))
    total = _num(_labelled(text, _TOTAL_LABELS))

    extras: dict[str, Any] = {"parser": "rules"}
    rate = re.search(rf"^\s*{_TAX_LABELS}\s*\((\d+(?:\.\d+)?)\s*%\)", text, re.I | re.M)
    if rate:
        extras["tax_rate_pct"] = float(rate.group(1))

    return {
        "invoice_no": invoice_no,
        "invoice_date": normalise_date(_labelled(text, _DATE_LABELS)),
        "po_number": f"PO-{po.group(1)}" if po else None,
        "vendor_id": vendor_id.group(1).upper() if vendor_id else None,
        "vendor_name": lines[0] if lines else None,
        "currency": detect_currency(text),
        "total_amount": total,
        "tax_amount": tax,
        "subtotal": subtotal,
        "line_items": items,
        "detected_language": detect_language(text),
        "additional_fields": extras,
    }
