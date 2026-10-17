"""
Invoice Auditor — Models Package
=================================
All Pydantic data models live here, split by domain.

Import from this package directly::

   from backend.models import EmailMeta, ExtractedInvoice, ...
"""

from .monitoring import (
   EmailMeta,
   InvoiceFile,
   MonitoredEmail,
   MonitorResult,
)
from .extraction import (
   ExtractedLineItem,
   ExtractedInvoice,
   ExtractionResult,
)
from .translation import (
   TranslatedLineItem,
   TranslatedInvoice,
   TranslationResult,
)
from .validation import (
   FieldIssue,
   LineItemIssue,
   ValidationResult,
)
from .business_validation import (
   LineItemDiscrepancy,
   ERPLookupResult,
   BusinessValidationResult,
)
from .reporting import (
   InvoiceReport,
   ReportingSummary,
)

__all__ = [
   # Monitoring
   "EmailMeta",
   "InvoiceFile",
   "MonitoredEmail",
   "MonitorResult",
   # Extraction
   "ExtractedLineItem",
   "ExtractedInvoice",
   "ExtractionResult",
   # Translation
   "TranslatedLineItem",
   "TranslatedInvoice",
   "TranslationResult",
   # Validation
   "FieldIssue",
   "LineItemIssue",
   "ValidationResult",
   # Business Validation
   "LineItemDiscrepancy",
   "ERPLookupResult",
   "BusinessValidationResult",
   # Reporting
   "InvoiceReport",
   "ReportingSummary",
]
