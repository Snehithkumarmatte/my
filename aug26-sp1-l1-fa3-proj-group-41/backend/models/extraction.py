"""
Extraction Models
=================
Pydantic schemas for extracted invoice data — line items, structured
invoice fields, and extraction results.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ExtractedLineItem(BaseModel):
   """A single line item extracted from an invoice."""
   item_code: Optional[str] = Field(
       None, description="Product / SKU code (e.g. SKU-001)"
   )
   description: Optional[str] = Field(
       None, description="Item description — in original language"
   )
   qty: Optional[float] = Field(None, description="Quantity")
   unit_price: Optional[float] = Field(None, description="Unit price")
   total: Optional[float] = Field(
       None, description="Line total (qty × unit_price)"
   )


class ExtractedInvoice(BaseModel):
   """Structured data extracted from a single invoice document."""
   #  Header fields 
   invoice_no: Optional[str] = Field(None, description="Invoice number")
   invoice_date: Optional[str] = Field(
       None, description="Invoice date as string (any format found in doc)"
   )
   po_number: Optional[str] = Field(
       None, description="Purchase Order reference (e.g. PO-1001)"
   )
   vendor_id: Optional[str] = Field(
       None, description="Vendor ID (e.g. VEND-001)"
   )
   vendor_name: Optional[str] = Field(None, description="Vendor name")
   currency: Optional[str] = Field(
       None, description="Currency code or symbol found"
   )
   total_amount: Optional[float] = Field(
       None, description="Grand total on the invoice"
   )
   tax_amount: Optional[float] = Field(
       None, description="Tax / GST amount if present"
   )
   subtotal: Optional[float] = Field(
       None, description="Subtotal before tax"
   )

   #  Line items 
   line_items: List[ExtractedLineItem] = Field(
       default_factory=list, description="Extracted line items"
   )

   #  Meta 
   detected_language: Optional[str] = Field(
       None, description="Language detected in the document text"
   )
   additional_fields: Dict[str, Any] = Field(
       default_factory=dict,
       description="Any other key-value pairs found in the invoice",
   )


class ExtractionResult(BaseModel):
   """Result returned by the Data Harvester Tool for one file."""
   email_id: str = Field(..., description="Email ID this extraction belongs to")
   filename: str = Field(..., description="Source filename")
   file_path: str = Field(..., description="Absolute path to the source file")
   file_format: str = Field(
       ..., description="Detected format: pdf | docx | image | unknown"
   )
   raw_text: str = Field(
       default="", description="Full raw text extracted from the document"
   )
   extracted_invoice: Optional[ExtractedInvoice] = Field(
       None, description="Structured invoice data parsed from raw text"
   )
   extraction_errors: List[str] = Field(
       default_factory=list,
       description="Errors encountered during extraction",
   )
