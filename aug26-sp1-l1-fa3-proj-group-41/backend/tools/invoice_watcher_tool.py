"""Invoice watcher tool.

Scans the incoming folder for invoice files (PDF, DOCX, PNG) and returns
the ones that have not been audited yet. A file counts as "already
audited" when its content checksum is in PostgreSQL, so a renamed copy
of the same invoice is still skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool

from backend.config.settings import INCOMING_DIR
from backend.db import repository
from backend.models import EmailMeta, InvoiceFile, MonitoredEmail, MonitorResult

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
INVOICES_DIR = Path(INCOMING_DIR)
EMAILS_DIR = INVOICES_DIR                                   # kept for the monitor agent
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png"}


@tool
def invoice_watcher_tool(scan_directory: str | None = None) -> dict:
    """Find new, unaudited invoice files in the incoming folder.

    Every new file is recorded in PostgreSQL as discovered; files whose
    checksum is already audited are skipped.

    Args:
        scan_directory: Folder to scan instead of the configured incoming folder.

    Returns:
        MonitorResult as a dict (new_emails, counts, errors).
    """
    folder = Path(scan_directory) if scan_directory else INVOICES_DIR
    if not folder.exists():
        return MonitorResult(errors=[f"Incoming directory not found: {folder}"]).model_dump()

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    new_emails: list[MonitoredEmail] = []
    errors: list[str] = []
    seen: set[str] = set()      # catches the same invoice twice in one scan (renamed copy)
    skipped = 0

    for path in files:
        try:
            checksum = repository.checksum_file(path)
            if checksum in seen or repository.is_audited(checksum):
                skipped += 1
                continue
            seen.add(checksum)
            repository.record_discovered(path)
            new_emails.append(_to_monitored_email(path, checksum))
        except Exception as exc:
            errors.append(f"Error processing {path.name}: {exc}")
            logger.error("Failed to process %s: %s", path, exc)

    return MonitorResult(
        new_emails=new_emails,
        total_scanned=len(files),
        already_processed=skipped,
        newly_found=len(new_emails),
        errors=errors,
    ).model_dump()


def find_email_meta(path: str | Path) -> tuple[dict | None, Path | None]:
    """The simulated email header for an attachment.

    Looks for '<stem>.meta.json' next to the file first, then for any *.meta.json in the
    same folder whose "attachments" list names the file (vendors rename files freely,
    and INV_DE_004.meta.json may list a .docx while a .pdf arrives).
    """
    path = Path(path)
    candidates = [path.with_name(f"{path.stem}.meta.json")]
    try:
        candidates += sorted(p for p in path.parent.glob("*.meta.json") if p not in candidates)
    except OSError:
        pass
    for meta_path in candidates:
        if not meta_path.is_file():
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable email meta %s: %s", meta_path.name, exc)
            continue
        if not isinstance(meta, dict):
            continue
        attachments = [str(a) for a in meta.get("attachments") or []]
        if meta_path == candidates[0] or path.name in attachments:
            return meta, meta_path
    return None, None


def _to_monitored_email(path: Path, checksum: str) -> MonitoredEmail:
    """Wrap a file as a MonitoredEmail, using its .meta.json email header when there is one."""
    stat = path.stat()
    invoice_file = InvoiceFile(
        filename=path.name,
        file_path=str(path.resolve()),
        file_extension=path.suffix.lower(),
        file_size_bytes=stat.st_size,
    )
    header, meta_path = find_email_meta(path)
    header = header or {}
    meta = {
        "email_id": f"file-{checksum[:16]}",
        "sender": str(header.get("sender") or "folder-monitor"),
        "subject": str(header.get("subject") or path.name),
        "received_timestamp": header.get("received_timestamp") or stat.st_mtime,
        "language": str(header.get("language") or "unknown"),
        "attachments": [path.name],
    }
    try:
        email_meta = EmailMeta(**meta)
    except ValueError:                       # bad timestamp in the meta file
        email_meta = EmailMeta(**{**meta, "received_timestamp": stat.st_mtime})
    return MonitoredEmail(email_meta=email_meta, meta_file_path=str(meta_path or ""),
                          invoice_files=[invoice_file], status="new")
