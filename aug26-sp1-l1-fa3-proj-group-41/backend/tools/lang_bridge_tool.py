"""Lang-bridge tool.

Detects the invoice language and translates the free-text fields (line item
descriptions, additional fields) into English with a Bedrock model. Numbers,
codes, dates, currency codes and company names are never translated.

English invoices are passed through with confidence 1.0. Translations go
through backend.llm, which caches them in Redis, so the same content is never
translated twice.
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import tool
from langdetect import DetectorFactory, LangDetectException, detect_langs

from backend.config.settings import BEDROCK_TRANSLATION_MODEL
from backend.llm import invoke_bedrock_chat
from backend.models.translation import TranslatedInvoice, TranslatedLineItem, TranslationResult
from backend.tools.data_harvester_tool import strip_code_fence

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0          # make langdetect deterministic

TRANSLATION_PROMPT = """\
You translate invoice content into English for a financial audit system.

You receive JSON with:
  "vendor_name": string or null
  "descriptions": list of line item descriptions (strings or null)
  "additional_fields": object of extra invoice fields

Rules:
- Translate every description and every additional field (keys and text values) into clear English.
- NEVER translate or change company / vendor names, item codes (SKU-...), PO numbers (PO-...),
  vendor ids (VEND-...), numbers, dates or currency codes. Return vendor_name exactly as given.
- Keep "descriptions" the same length and in the same order; keep null as null.
- Rate how faithful and complete your translation is as "confidence" between 0 and 1.

Return ONLY valid JSON, no markdown:
{"vendor_name": ..., "descriptions": [...], "additional_fields": {...}, "confidence": 0.0}
"""


def detect_language(extraction: dict) -> tuple[str, float]:
    """langdetect on the raw text; fall back to the extractor's guess when unsure."""
    invoice = extraction.get("extracted_invoice") or {}
    hint = str(invoice.get("detected_language") or "en").lower()[:2]
    text = extraction.get("raw_text") or ""
    if len(text.strip()) >= 20:
        try:
            best = detect_langs(text)[0]
            if best.prob >= 0.80:
                return best.lang[:2], float(best.prob)
        except LangDetectException:
            pass
    return hint, 0.5


def _build_invoice(invoice: dict, translated: dict | None, language: str,
                   was_translated: bool) -> TranslatedInvoice:
    translated = translated or {}
    descriptions = translated.get("descriptions") or []
    lines = []
    for n, line in enumerate(invoice.get("line_items") or []):
        english = descriptions[n] if n < len(descriptions) and descriptions[n] else None
        lines.append(TranslatedLineItem(
            item_code=line.get("item_code"),
            original_description=line.get("description"),
            translated_description=english or line.get("description"),
            qty=line.get("qty"),
            unit_price=line.get("unit_price"),
            total=line.get("total"),
        ))
    extras = invoice.get("additional_fields") or {}
    return TranslatedInvoice(
        invoice_no=invoice.get("invoice_no"),
        invoice_date=invoice.get("invoice_date"),
        po_number=invoice.get("po_number"),
        vendor_id=invoice.get("vendor_id"),
        original_vendor_name=invoice.get("vendor_name"),
        translated_vendor_name=invoice.get("vendor_name"),      # company names are never translated
        currency=invoice.get("currency"),
        total_amount=invoice.get("total_amount"),
        tax_amount=invoice.get("tax_amount"),
        subtotal=invoice.get("subtotal"),
        line_items=lines,
        source_language=language,
        was_translated=was_translated,
        original_additional_fields=extras,
        translated_additional_fields=translated.get("additional_fields") or extras,
    )


@tool
def lang_bridge_tool(extraction_result: dict) -> dict:
    """Translate an extracted invoice into English (pass-through when already English).

    Args:
        extraction_result: ExtractionResult as a dict.

    Returns:
        TranslationResult as a dict, including translation_confidence (0-1).
    """
    base = {
        "email_id": extraction_result.get("email_id", "unknown"),
        "filename": extraction_result.get("filename", "?"),
        "file_path": extraction_result.get("file_path", ""),
    }
    invoice = extraction_result.get("extracted_invoice")
    if not invoice:
        return TranslationResult(**base, translation_errors=["No structured invoice data to translate"]).model_dump()

    language, detection_confidence = detect_language(extraction_result)
    logger.info("%s: detected language '%s' (p=%.2f)", base["filename"], language, detection_confidence)

    if language == "en":
        return TranslationResult(
            **base, source_language="en", was_translated=False, translation_confidence=1.0,
            translated_invoice=_build_invoice(invoice, None, "en", False),
        ).model_dump()

    lines = invoice.get("line_items") or []
    payload = {
        "vendor_name": invoice.get("vendor_name"),
        "descriptions": [line.get("description") for line in lines],
        "additional_fields": invoice.get("additional_fields") or {},
    }
    errors: list[str] = []
    try:
        reply = invoke_bedrock_chat(
            model_id=BEDROCK_TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        translated = json.loads(strip_code_fence(reply or "{}"))
        if len(translated.get("descriptions") or []) != len(lines):
            errors.append("Translation returned a different number of line descriptions")
        confidence = max(0.0, min(1.0, float(translated.get("confidence", 0.8))))
        result_invoice = _build_invoice(invoice, translated, language, True)
        was_translated = True
    except Exception as exc:     # keep the pipeline going with the untranslated text
        logger.error("Translation failed for %s: %s", base["filename"], exc)
        errors.append(f"Translation failed: {exc}")
        confidence = 0.0
        result_invoice = _build_invoice(invoice, None, language, False)
        was_translated = False

    return TranslationResult(
        **base, source_language=language, was_translated=was_translated,
        translation_confidence=round(confidence, 3), translated_invoice=result_invoice,
        translation_errors=errors,
    ).model_dump()
