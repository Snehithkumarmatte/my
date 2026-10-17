"""Reporting models - the audit report for one invoice and the run summary."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class InvoiceReport(BaseModel):
    """Audit report for one invoice: results of every pipeline stage plus the recommendation."""

    # identity
    email_id: str = Field(..., description="Email ID")
    filename: str = Field(..., description="Source filename")
    file_path: str = Field(..., description="Absolute file path")

    # invoice header (translated)
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    po_number: Optional[str] = None
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None

    # translation
    source_language: Optional[str] = None
    was_translated: bool = False
    translation_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # data validation
    data_validation_status: Optional[str] = None
    data_validation_errors: int = 0
    data_validation_warnings: int = 0
    data_validation_issues: List[Dict[str, Any]] = Field(default_factory=list)

    # business validation
    business_validation_status: Optional[str] = None
    business_validation_errors: int = 0
    business_validation_warnings: int = 0
    business_validation_matched: int = 0
    erp_po_found: bool = False
    erp_vendor_found: bool = False
    business_discrepancies: List[Dict[str, Any]] = Field(default_factory=list)

    # line items (translated)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)

    # recommendation - same values as the audit.invoice_audit CHECK constraint
    recommendation: Literal["approve", "manual_review", "reject"] = "manual_review"
    recommendation_reasons: List[str] = Field(default_factory=list)
    # overall status from the matching rules.yaml decision rule: passed | partial | failed
    validation_status: Optional[Literal["passed", "partial", "failed"]] = None

    # report files
    report_html_path: Optional[str] = None
    report_json_path: Optional[str] = None


class ReportingSummary(BaseModel):
    """Summary across all invoices processed in one pipeline run."""
    total_invoices: int = 0
    auto_approved: int = 0
    manual_review: int = 0
    rejected: int = 0
    reports: List[InvoiceReport] = Field(default_factory=list)
    output_dir: str = ""
