

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class FieldIssue(BaseModel):
   """A single validation issue for a header-level field."""
   field_name: str = Field(
       ..., description="Name of the field that has an issue"
   )
   issue_type: Literal[
       "missing", "wrong_type", "invalid_value", "math_error"
   ] = Field(
       ..., description="Category of the issue"
   )
   severity: Literal["error", "warning"] = Field(
       default="error",
       description="error = must fix, warning = review recommended",
   )
   message: str = Field(
       ..., description="Human-readable description of the issue"
   )
   expected: Optional[str] = Field(
       None, description="What was expected (e.g. 'float', 'USD')"
   )
   actual: Optional[str] = Field(
       None, description="What was actually found"
   )


class LineItemIssue(BaseModel):
   """A validation issue tied to a specific line item."""
   line_index: int = Field(
       ..., description="Zero-based index of the line item in the list"
   )
   field_name: str = Field(
       ..., description="Name of the line-item field that has an issue"
   )
   issue_type: Literal[
       "missing", "wrong_type", "invalid_value", "math_error"
   ] = Field(
       ..., description="Category of the issue"
   )
   severity: Literal["error", "warning"] = Field(
       default="error",
       description="error = must fix, warning = review recommended",
   )
   message: str = Field(
       ..., description="Human-readable description of the issue"
   )
   expected: Optional[str] = Field(None, description="Expected value")
   actual: Optional[str] = Field(None, description="Actual value found")


class ValidationResult(BaseModel):
   """Complete validation report for a single invoice."""
   email_id: str = Field(
       ..., description="Email ID this validation belongs to"
   )
   filename: str = Field(..., description="Source filename")
   file_path: str = Field(
       ..., description="Absolute path to the source file"
   )

  
   is_valid: bool = Field(
       default=True,
       description="True if no errors found (warnings are OK)",
   )
   validation_status: Literal[
       "passed", "failed", "passed_with_warnings"
   ] = Field(
       default="passed",
       description="Overall validation status",
   )

   #  Issue details 
   header_issues: List[FieldIssue] = Field(
       default_factory=list,
       description="Validation issues on header-level fields",
   )
   line_item_issues: List[LineItemIssue] = Field(
       default_factory=list,
       description="Validation issues on individual line items",
   )

   #  Counts 
   total_errors: int = Field(
       default=0, description="Count of severity=error issues"
   )
   total_warnings: int = Field(
       default=0, description="Count of severity=warning issues"
   )

   #  Rules applied 
   rules_applied: List[str] = Field(
       default_factory=list,
       description="Names of the rule checks that were executed",
   )

   #  Pass-through context 
   source_language: Optional[str] = Field(
       None, description="Language of the invoice"
   )