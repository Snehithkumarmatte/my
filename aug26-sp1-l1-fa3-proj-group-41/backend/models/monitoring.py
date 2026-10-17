

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class EmailMeta(BaseModel):
   """Metadata of a single incoming email (parsed from *.meta.json)."""
   email_id: str = Field(
       ..., description="Unique ID derived from the meta.json filename"
   )
   sender: str = Field(..., description="Sender email address")
   subject: str = Field(..., description="Email subject line")
   received_timestamp: datetime = Field(
       ..., description="Timestamp when the email was received"
   )
   language: str = Field(
       ..., description="ISO language code of the email/invoice (e.g. en, de, es)"
   )
   attachments: List[str] = Field(
       default_factory=list,
       description="List of attachment filenames referenced in the email",
   )


class InvoiceFile(BaseModel):
   """Represents a single invoice attachment discovered on the filesystem."""
   filename: str = Field(..., description="Name of the file on disk")
   file_path: str = Field(..., description="Absolute path to the file")
   file_extension: str = Field(
       ..., description="Lowercase extension including the dot, e.g. '.pdf'"
   )
   file_size_bytes: int = Field(..., description="Size of the file in bytes")


class MonitoredEmail(BaseModel):
   """A fully resolved email record — metadata + verified attachment paths."""
   email_meta: EmailMeta
   meta_file_path: str = Field(
       ..., description="Absolute path to the .meta.json file"
   )
   invoice_files: List[InvoiceFile] = Field(
       default_factory=list,
       description="Verified invoice attachments found on disk",
   )
   status: str = Field(
       default="new",
       description="Processing status: new | processing | processed | error",
   )


class MonitorResult(BaseModel):
   """Aggregated result returned by the Invoice Monitor Agent."""
   new_emails: List[MonitoredEmail] = Field(default_factory=list)
   total_scanned: int = Field(
       0, description="Total meta.json files found in the inbox"
   )
   already_processed: int = Field(
       0, description="Emails skipped because they were already processed"
   )
   newly_found: int = Field(
       0, description="Emails that are new and ready for processing"
   )
   errors: List[str] = Field(
       default_factory=list,
       description="Non-fatal errors encountered during the scan",
   )
