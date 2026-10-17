"""Invoice Auditor - agents package.

The pipeline runs these agents in order:
    InvoiceMonitorAgent -> ExtractorAgent -> TranslationAgent
    -> ValidationAgent -> BusinessValidationAgent -> ReportingAgent

The data models are re-exported here too, so callers can do
    from backend.agents import ExtractorAgent, ExtractionResult
"""

from backend.models import (
    EmailMeta,
    InvoiceFile,
    MonitoredEmail,
    MonitorResult,
    ExtractedLineItem,
    ExtractedInvoice,
    ExtractionResult,
    TranslatedLineItem,
    TranslatedInvoice,
    TranslationResult,
    FieldIssue,
    LineItemIssue,
    ValidationResult,
    LineItemDiscrepancy,
    ERPLookupResult,
    BusinessValidationResult,
    InvoiceReport,
    ReportingSummary,
)

from .invoice_monitor_agent import InvoiceMonitorAgent
from .extractor_agent import ExtractorAgent
from .translation_agent import TranslationAgent
from .validation_agent import ValidationAgent
from .business_validation_agent import BusinessValidationAgent
from .reporting_agent import ReportingAgent

__all__ = [
    # models
    "EmailMeta", "InvoiceFile", "MonitoredEmail", "MonitorResult",
    "ExtractedLineItem", "ExtractedInvoice", "ExtractionResult",
    "TranslatedLineItem", "TranslatedInvoice", "TranslationResult",
    "FieldIssue", "LineItemIssue", "ValidationResult",
    "LineItemDiscrepancy", "ERPLookupResult", "BusinessValidationResult",
    "InvoiceReport", "ReportingSummary",
    # agents
    "InvoiceMonitorAgent", "ExtractorAgent", "TranslationAgent",
    "ValidationAgent", "BusinessValidationAgent", "ReportingAgent",
]