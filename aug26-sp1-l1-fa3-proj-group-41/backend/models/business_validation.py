"""
Business Validation Models
==========================
Pydantic schemas for business-level validation results produced by the
Business Validation Agent / Business Validation Tool.

These models capture discrepancies found when cross-referencing invoice
data against enterprise system (ERP) records at the **line-item level**.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class LineItemDiscrepancy(BaseModel):
   """A single discrepancy found when comparing an invoice line item
   against the corresponding ERP Purchase Order line item."""
   line_index: int = Field(
       ..., description="Zero-based index of the line item in the invoice"
   )
   item_code: Optional[str] = Field(
       None, description="SKU / item code of the line item"
   )
   field_name: str = Field(
       ..., description="Name of the field with the discrepancy"
   )
   discrepancy_type: Literal[
       "price_mismatch",
       "quantity_mismatch",
       "item_not_in_po",
       "item_not_in_sku_master",
       "currency_mismatch",
       "vendor_mismatch",
       "extra_item_on_invoice",
       "missing_item_on_invoice",
   ] = Field(
       ..., description="Category of the discrepancy"
   )
   severity: Literal["error", "warning"] = Field(
       default="error",
       description="error = must fix, warning = review recommended",
   )
   message: str = Field(
       ..., description="Human-readable description of the discrepancy"
   )
   invoice_value: Optional[str] = Field(
       None, description="Value found on the invoice"
   )
   erp_value: Optional[str] = Field(
       None, description="Value found in the ERP system"
   )
   difference_pct: Optional[float] = Field(
       None,
       description="Percentage difference (for numeric mismatches)",
   )


class ERPLookupResult(BaseModel):
   """Summary of the ERP lookup performed for one invoice."""
   po_number: Optional[str] = Field(
       None, description="PO number looked up"
   )
   po_found: bool = Field(
       default=False, description="Whether the PO was found in ERP"
   )
   vendor_id: Optional[str] = Field(
       None, description="Vendor ID looked up"
   )
   vendor_found: bool = Field(
       default=False, description="Whether the vendor was found in ERP"
   )
   erp_line_items_count: int = Field(
       default=0, description="Number of line items in the ERP PO"
   )
   invoice_line_items_count: int = Field(
       default=0, description="Number of line items on the invoice"
   )
   erp_currency: Optional[str] = Field(
       None, description="Currency from the ERP PO"
   )
   erp_vendor_name: Optional[str] = Field(
       None, description="Vendor name from ERP master"
   )


class BusinessValidationResult(BaseModel):
   """Complete business validation report for a single invoice.

   This is the output of cross-referencing invoice data against
   ERP Purchase Orders, Vendor Master, and SKU Master.
   """
   email_id: str = Field(
       ..., description="Email ID this validation belongs to"
   )
   filename: str = Field(..., description="Source filename")
   file_path: str = Field(
       ..., description="Absolute path to the source file"
   )

   #  Overall verdict 
   is_valid: bool = Field(
       default=True,
       description="True if no errors found (warnings are OK)",
   )
   validation_status: Literal[
       "matched", "discrepancies_found", "po_not_found",
       "vendor_not_found", "skipped",
   ] = Field(
       default="matched",
       description="Overall business validation status",
   )

   #  ERP Lookup info 
   erp_lookup: Optional[ERPLookupResult] = Field(
       None, description="Summary of the ERP lookup"
   )

   #  Discrepancies 
   discrepancies: List[LineItemDiscrepancy] = Field(
       default_factory=list,
       description="List of all discrepancies found",
   )

   #  Counts 
   total_errors: int = Field(
       default=0, description="Count of severity=error discrepancies"
   )
   total_warnings: int = Field(
       default=0, description="Count of severity=warning discrepancies"
   )
   matched_line_items: int = Field(
       default=0, description="Number of line items that matched ERP"
   )

   #  Pass-through context 
   source_language: Optional[str] = Field(
       None, description="Language of the invoice"
   )
