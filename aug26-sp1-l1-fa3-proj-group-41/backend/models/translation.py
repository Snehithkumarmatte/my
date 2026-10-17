"""
Translation Models
==================
Pydantic schemas for the translated invoice data produced by the
Translation Agent / Lang-Bridge Tool.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class TranslatedLineItem(BaseModel):
    """A single line item with both original and English-translated text."""
    item_code: Optional[str] = Field(
        None, description="Product / SKU code (unchanged)"
    )
    original_description: Optional[str] = Field(
        None, description="Item description in the original language"
    )
    translated_description: Optional[str] = Field(
        None,
        description="Item description translated to English "
                    "(same as original if already English)",
    )
    qty: Optional[float] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price")
    total: Optional[float] = Field(
        None, description="Line total (qty × unit_price)"
    )


class TranslatedInvoice(BaseModel):
    """Structured invoice where every free-text field has been translated
    to English while numeric / code fields remain unchanged."""

    #  Header fields (translated) 
    invoice_no: Optional[str] = Field(None, description="Invoice number (unchanged)")
    invoice_date: Optional[str] = Field(None, description="Invoice date (unchanged)")
    po_number: Optional[str] = Field(None, description="PO reference (unchanged)")
    vendor_id: Optional[str] = Field(None, description="Vendor ID (unchanged)")

    original_vendor_name: Optional[str] = Field(
        None, description="Vendor name in original language"
    )
    translated_vendor_name: Optional[str] = Field(
        None, description="Vendor name translated to English"
    )

    currency: Optional[str] = Field(None, description="Currency code (unchanged)")
    total_amount: Optional[float] = Field(None, description="Grand total (unchanged)")
    tax_amount: Optional[float] = Field(None, description="Tax amount (unchanged)")
    subtotal: Optional[float] = Field(None, description="Subtotal (unchanged)")

    #  Line items (translated) 
    line_items: List[TranslatedLineItem] = Field(
        default_factory=list,
        description="Line items with translated descriptions",
    )

    #  Language info 
    source_language: Optional[str] = Field(
        None,
        description="ISO 639-1 code of the detected source language",
    )
    target_language: str = Field(
        default="en",
        description="Target language — always 'en' (English)",
    )
    was_translated: bool = Field(
        default=False,
        description="True if translation was actually performed "
                    "(False when source is already English)",
    )

    #  Extra 
    original_additional_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields in the original language",
    )
    translated_additional_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields translated to English",
    )


class TranslationResult(BaseModel):
    """Result returned by the Lang-Bridge Tool for one extraction."""
    email_id: str = Field(..., description="Email ID this belongs to")
    filename: str = Field(..., description="Source filename")
    file_path: str = Field(..., description="Absolute path to source file")
    source_language: Optional[str] = Field(
        None, description="Detected source language"
    )
    was_translated: bool = Field(
        default=False,
        description="Whether translation was performed",
    )
    translation_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Confidence reported by the translation model (None = not translated)",
    )
    translated_invoice: Optional[TranslatedInvoice] = Field(
        None, description="Fully translated invoice data"
    )
    translation_errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during translation",
    )
