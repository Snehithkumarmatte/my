"""Insight reporter tool.

Builds the final audit report for one invoice from the translation,
data validation and business validation results, decides the
recommendation (approve / manual_review / reject) and writes the report
as an HTML file for people and a JSON file for machines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import yaml
from langchain_core.tools import tool

from backend.config.settings import REPORTS_DIR
from backend.models.reporting import InvoiceReport

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BASE_DIR.parent
RULES_PATH = BASE_DIR / "config" / "rules.yaml"

_rules: dict | None = None


@tool
def insight_reporter_tool(
    translation_result: dict,
    validation_result: dict | None = None,
    business_validation_result: dict | None = None,
) -> dict:
    """Generate the audit report (HTML + JSON) for a single invoice.

    Args:
        translation_result: TranslationResult as a dict.
        validation_result: ValidationResult as a dict (optional).
        business_validation_result: BusinessValidationResult as a dict (optional).

    Returns:
        InvoiceReport as a dict, including report_html_path and report_json_path.
    """
    rules = _load_rules()
    # REPORTS_DIR (env) is the folder compose mounts as a volume; rules.yaml is only a fallback
    out_dir = Path(REPORTS_DIR or rules.get("reporting", {}).get("output_dir", "data/reports"))
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(translation_result, validation_result, business_validation_result, rules)

    base_name = report.filename.replace(" ", "_").rsplit(".", 1)[0]
    html_path = out_dir / f"{report.email_id}_{base_name}_report.html"
    json_path = html_path.with_suffix(".json")

    html_path.write_text(render_html(report, rules), encoding="utf-8")
    report.report_html_path = str(html_path.resolve())
    report.report_json_path = str(json_path.resolve())
    json_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    logger.info("Report generated: %s -> %s [%s]", report.filename, report.recommendation.upper(), html_path.name)
    return report.model_dump()


def _load_rules() -> dict:
    global _rules
    if _rules is None:
        if RULES_PATH.exists():
            with open(RULES_PATH, encoding="utf-8") as fh:
                _rules = yaml.safe_load(fh) or {}
        else:
            _rules = {}
    return _rules


# building the report

def build_report(tr: dict, vr: dict | None, bvr: dict | None, rules: dict) -> InvoiceReport:
    """Combine the outputs of every pipeline stage into one InvoiceReport."""
    vr, bvr = vr or {}, bvr or {}
    invoice = tr.get("translated_invoice") or {}
    erp = bvr.get("erp_lookup") or {}

    line_items = [
        {
            "item_code": line.get("item_code"),
            "original_description": line.get("original_description"),
            "translated_description": line.get("translated_description"),
            "qty": line.get("qty"),
            "unit_price": line.get("unit_price"),
            "total": line.get("total"),
        }
        for line in invoice.get("line_items", [])
    ]
    recommendation, status, reasons = decide_recommendation(tr, vr, bvr, rules)
    confidence = tr.get("translation_confidence")

    return InvoiceReport(
        email_id=tr.get("email_id", "unknown"),
        filename=tr.get("filename", "unknown"),
        file_path=tr.get("file_path", ""),
        invoice_no=invoice.get("invoice_no"),
        invoice_date=invoice.get("invoice_date"),
        po_number=invoice.get("po_number"),
        vendor_id=invoice.get("vendor_id"),
        vendor_name=invoice.get("translated_vendor_name"),
        currency=invoice.get("currency"),
        subtotal=invoice.get("subtotal"),
        tax_amount=invoice.get("tax_amount"),
        total_amount=invoice.get("total_amount"),
        source_language=tr.get("source_language"),
        was_translated=tr.get("was_translated", False),
        translation_confidence=1.0 if confidence is None else confidence,
        data_validation_status=vr.get("validation_status"),
        data_validation_errors=vr.get("total_errors", 0),
        data_validation_warnings=vr.get("total_warnings", 0),
        data_validation_issues=list(vr.get("header_issues", [])) + list(vr.get("line_item_issues", [])),
        business_validation_status=bvr.get("validation_status"),
        business_validation_errors=bvr.get("total_errors", 0),
        business_validation_warnings=bvr.get("total_warnings", 0),
        business_validation_matched=bvr.get("matched_line_items", 0),
        erp_po_found=erp.get("po_found", False),
        erp_vendor_found=erp.get("vendor_found", False),
        business_discrepancies=list(bvr.get("discrepancies", [])),
        line_items=line_items,
        recommendation=recommendation,
        recommendation_reasons=reasons,
        validation_status=status,
    )


def _issue_text(issue: dict) -> str:
    return str(issue.get("message") or
               f"{issue.get('field_name')}: {issue.get('issue_type') or issue.get('discrepancy_type')}")


def evaluate_conditions(tr: dict, vr: dict, bvr: dict, rules: dict) -> dict[str, list[str]]:
    """Evidence for every condition named in rules.yaml -> decision_rules."""
    critical = set(rules.get("critical_fields") or [])
    accepted = {c.upper() for c in rules.get("accepted_currencies") or []}
    threshold = float((rules.get("validation_policies") or {}).get("auto_approve_confidence_threshold", 0.85))

    data_issues = list(vr.get("header_issues") or []) + list(vr.get("line_item_issues") or [])
    header_fields = {i.get("field_name") for i in vr.get("header_issues") or []}
    discrepancies = list(bvr.get("discrepancies") or [])
    found: dict[str, list[str]] = {name: [] for name in (
        "missing_critical_field", "invalid_currency", "any_high_discrepancy",
        "any_medium_discrepancy", "missing_noncritical_field",
        "translation_confidence_below_threshold")}

    for issue in data_issues:
        text = _issue_text(issue)
        is_header = issue in (vr.get("header_issues") or [])
        if issue.get("issue_type") == "missing":
            if is_header and issue.get("field_name") in critical:
                found["missing_critical_field"].append(text)
            else:
                found["missing_noncritical_field"].append(text)
        elif issue.get("field_name") == "currency" and issue.get("issue_type") == "invalid_value":
            found["invalid_currency"].append(text)
        elif issue.get("severity") == "error":
            found["any_high_discrepancy"].append(text)
        else:
            found["any_medium_discrepancy"].append(text)

    invoice = tr.get("translated_invoice") or {}
    currency = (invoice.get("currency") or "").strip().upper()
    if currency and accepted and currency not in accepted and "currency" not in header_fields:
        found["invalid_currency"].append(f"Currency {currency} is not an accepted currency")

    if bvr.get("validation_status") == "po_not_found":
        found["any_high_discrepancy"].append("Purchase order was not found in the ERP service")
    for d in discrepancies:
        key = "any_high_discrepancy" if d.get("severity") == "error" else "any_medium_discrepancy"
        found[key].append(_issue_text(d))

    confidence = tr.get("translation_confidence")
    if tr.get("was_translated") or (tr.get("source_language") not in (None, "en")):
        if confidence is None or float(confidence) < threshold:
            found["translation_confidence_below_threshold"].append(
                f"Translation confidence {0.0 if confidence is None else float(confidence):.2f} "
                f"is below the auto-approve threshold {threshold:.2f}")
    return found


def decide_recommendation(tr: dict, vr: dict, bvr: dict, rules: dict) -> tuple[str, str, list[str]]:
    """Apply rules.yaml -> decision_rules top to bottom; the first matching rule wins.

    Returns (recommendation, validation_status, reasons)."""
    found = evaluate_conditions(tr, vr, bvr, rules)
    policy = rules.get("decision_rules") or [
        {"when": "missing_critical_field", "recommendation": "reject", "validation_status": "failed"},
        {"when": "invalid_currency", "recommendation": "reject", "validation_status": "failed"},
        {"when": "any_high_discrepancy", "recommendation": "reject", "validation_status": "failed"},
        {"when": "any_medium_discrepancy", "recommendation": "manual_review", "validation_status": "partial"},
        {"when": "missing_noncritical_field", "recommendation": "manual_review", "validation_status": "partial"},
        {"when": "translation_confidence_below_threshold", "recommendation": "manual_review",
         "validation_status": "partial"},
        {"when": "only_low_or_none", "recommendation": "approve", "validation_status": "passed"},
    ]
    for rule in policy:
        condition = rule.get("when")
        evidence = found.get(condition, [])
        if condition == "only_low_or_none" or evidence:
            others = [f"{name.replace('_', ' ')}: {item}" for name, items in found.items()
                      if name != condition for item in items]
            if condition == "only_low_or_none":
                reasons = ["Required fields and configured invoice checks passed",
                           "ERP comparisons found no discrepancies above tolerance"] + others
            else:
                reasons = [f"Rule '{condition}' matched"] + evidence + others
            return rule.get("recommendation", "manual_review"), rule.get("validation_status", "partial"), reasons
    return "manual_review", "partial", ["No decision rule matched"]


# HTML

COLOURS = {
    "approve":       {"bg": "#0f5132", "light": "#d1e7dd", "text": "#0f5132", "bar": "#198754"},
    "manual_review": {"bg": "#664d03", "light": "#fff3cd", "text": "#664d03", "bar": "#ffc107"},
    "reject":        {"bg": "#842029", "light": "#f8d7da", "text": "#842029", "bar": "#dc3545"},
}
ICONS = {"approve": "✓", "manual_review": "!", "reject": "✗"}

CSS = Template("""\
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; line-height: 1.55; background: #f0f2f5; color: #1e2330;
  }
  a { color: inherit; text-decoration: none; }

  /*  Page wrapper  */
  .page { max-width: 980px; margin: 24px auto; }

  /*  Top bar  */
  .topbar {
    background: #1e2330; color: #fff; border-radius: 10px 10px 0 0;
    padding: 18px 28px; display: flex; align-items: center; justify-content: space-between;
  }
  .topbar-left { display: flex; align-items: center; gap: 14px; }
  .topbar-icon {
    width: 38px; height: 38px; border-radius: 8px;
    background: $bar; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; font-weight: 700; flex-shrink: 0;
  }
  .topbar-title { font-size: 14px; font-weight: 600; }
  .topbar-sub { font-size: 11px; color: #8892a4; margin-top: 1px; }
  .topbar-date { font-size: 11px; color: #8892a4; text-align: right; }

  /*  Verdict strip  */
  .verdict {
    background: $light; border-left: 4px solid $bar;
    padding: 10px 28px; display: flex; align-items: center; gap: 10px;
    font-size: 12px; color: $text;
  }
  .verdict-icon {
    width: 22px; height: 22px; border-radius: 50%;
    background: $bar; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; flex-shrink: 0;
  }
  .verdict-label { font-weight: 700; font-size: 12px; }
  .verdict-sub { color: #555; font-size: 11px; }

  /*  Card shell  */
  .card {
    background: #fff; border-radius: 0; border-bottom: 1px solid #e5e8ed;
    padding: 20px 28px;
  }
  .card:last-of-type { border-radius: 0 0 10px 10px; border-bottom: none; }
  .card-title {
    font-size: 11px; font-weight: 700; letter-spacing: .6px;
    text-transform: uppercase; color: #8892a4; margin-bottom: 14px;
    display: flex; align-items: center; gap: 7px;
  }
  .card-title::before {
    content: ""; display: block; width: 3px; height: 13px;
    border-radius: 2px; background: $bar;
  }

  /*  Field grid  */
  .fgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 16px; }
  .fgrid-3 { grid-template-columns: repeat(3, 1fr); }
  .fgrid-2 { grid-template-columns: repeat(2, 1fr); }
  .fl { font-size: 10px; color: #8892a4; text-transform: uppercase;
         letter-spacing: .4px; margin-bottom: 2px; }
  .fv { font-size: 12.5px; font-weight: 500; color: #1e2330; }
  .fv-lg { font-size: 15px; font-weight: 700; color: #1e2330; }
  .fv-mono { font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
              font-size: 11.5px; }
  .fv-muted { font-size: 11px; color: #8892a4; word-break: break-all; }

  /*  Scorecard row  */
  .scorecard { display: flex; gap: 12px; }
  .sc {
    flex: 1; background: #f8f9fb; border: 1px solid #e5e8ed;
    border-radius: 8px; padding: 12px 14px; text-align: center;
  }
  .sc-val { font-size: 20px; font-weight: 700; line-height: 1.2; }
  .sc-lbl { font-size: 10px; color: #8892a4; margin-top: 2px; letter-spacing: .3px; }
  .sc-green .sc-val { color: #198754; }
  .sc-amber .sc-val { color: #d97706; }
  .sc-red   .sc-val { color: #dc3545; }
  .sc-blue  .sc-val { color: #0d6efd; }
  .sc-dark  .sc-val { color: #1e2330; }

  /*  Confidence bar  */
  .conf-wrap { margin-top: 8px; }
  .conf-track {
    height: 6px; background: #e5e8ed; border-radius: 3px; overflow: hidden;
    margin-top: 5px;
  }
  .conf-fill { height: 100%; border-radius: 3px; }
  .conf-labels {
    display: flex; justify-content: space-between;
    font-size: 10px; color: #8892a4; margin-top: 3px;
  }

  /*  Pill badges  */
  .pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 20px; font-size: 10.5px; font-weight: 600;
  }
  .pill-green  { background: #d1e7dd; color: #0f5132; }
  .pill-amber  { background: #fff3cd; color: #664d03; }
  .pill-red    { background: #f8d7da; color: #842029; }
  .pill-blue   { background: #cfe2ff; color: #084298; }
  .pill-grey   { background: #e9ecef; color: #495057; }

  /*  Tables  */
  table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
  thead tr { background: #f8f9fb; }
  th {
    padding: 8px 10px; text-align: left; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .4px; color: #8892a4;
    border-bottom: 1px solid #e5e8ed; white-space: nowrap;
  }
  td { padding: 8px 10px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: #f8f9fb; }
  .tr-err td  { background: #fff5f5; }
  .tr-warn td { background: #fffdf0; }
  td.num { text-align: right; font-family: "SF Mono", Consolas, monospace; font-size: 11.5px; }
  th.num { text-align: right; }

  /*  ERP lookup row  */
  .erp-row { display: flex; gap: 24px; margin-top: 12px; margin-bottom: 4px; }
  .erp-item { display: flex; align-items: center; gap: 7px; font-size: 12px; }
  .erp-label { color: #8892a4; font-size: 11px; }

  /*  Reasons list  */
  .reasons { list-style: none; }
  .reasons li {
    padding: 6px 0 6px 16px; border-bottom: 1px solid #f0f2f5;
    position: relative; font-size: 12px; color: #3a4258;
  }
  .reasons li:last-child { border-bottom: none; }
  .reasons li::before {
    content: ""; position: absolute; left: 0; top: 12px;
    width: 6px; height: 6px; border-radius: 50%; background: $bar;
  }

  /*  Pass banner  */
  .pass-banner {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: #d1e7dd; border-radius: 6px; color: #0f5132;
    font-size: 12px; font-weight: 500; margin-top: 10px;
  }

  /*  Footer  */
  .footer {
    text-align: center; font-size: 10.5px; color: #8892a4;
    padding: 14px 28px; background: #f8f9fb;
    border-radius: 0 0 10px 10px; border-top: 1px solid #e5e8ed;
  }""")


def render_html(report: InvoiceReport, rules: dict) -> str:
    settings = rules.get("reporting", {})
    colours = COLOURS.get(report.recommendation, COLOURS["manual_review"])
    generated_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    sections = [page_start(report, colours, generated_at), invoice_details(report)]
    if settings.get("include_translation_confidence", True):
        sections.append(confidence_section(report, colours))
    sections.append(data_validation_section(report))
    if settings.get("include_discrepancy_summary", True):
        sections.append(business_validation_section(report))
    sections.append(line_items_section(report))
    sections.append(reasons_section(report))
    sections.append(page_end(generated_at))
    return "\n".join(sections)


def page_start(report: InvoiceReport, colours: dict, generated_at: str) -> str:
    icon = ICONS.get(report.recommendation, "?")
    label = report.recommendation.replace("_", " ").title()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit Report · {esc(report.filename)}</title>
<style>
{CSS.substitute(colours)}
</style>
</head>
<body>
<div class="page">

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-icon">{icon}</div>
    <div>
      <div class="topbar-title">Invoice Audit Report</div>
      <div class="topbar-sub">{esc(report.filename)}&ensp;·&ensp;{esc(report.email_id)}</div>
    </div>
  </div>
  <div class="topbar-date">Generated<br>{generated_at}</div>
</div>

<!-- Verdict strip -->
<div class="verdict">
  <div class="verdict-icon">{icon}</div>
  <div>
    <span class="verdict-label">{label}</span>
    <span class="verdict-sub">&ensp;·&ensp;Translation confidence: <strong>{report.translation_confidence:.0%}</strong></span>
  </div>
</div>"""


def page_end(generated_at: str) -> str:
    return f"""
<div class="footer">
  AI Invoice Auditor &nbsp;·&nbsp; {generated_at}
</div>

</div><!-- /page -->
</body>
</html>"""


def field(label: str, value: str, css: str = "fv") -> str:
    return f"""    <div>
      <div class="fl">{label}</div>
      <div class="{css}">{value}</div>
    </div>"""


def invoice_details(r: InvoiceReport) -> str:
    fields = "\n".join([
        field("Invoice No.", esc(r.invoice_no or "—"), "fv fv-mono"),
        field("Invoice Date", esc(r.invoice_date or "—")),
        field("PO Number", esc(r.po_number or "—"), "fv fv-mono"),
        field("Currency", esc(r.currency or "—")),
        field("Vendor ID", esc(r.vendor_id or "—"), "fv fv-mono"),
        field("Vendor Name", esc(r.vendor_name or "—")),
        field("Subtotal", fmt_amount(r.subtotal, r.currency), "fv fv-mono"),
        field("Tax", fmt_amount(r.tax_amount, r.currency), "fv fv-mono"),
    ])
    return f"""
<div class="card">
  <div class="card-title">Invoice Details</div>
  <div class="fgrid">
{fields}
  </div>
  <div style="margin-top:14px; padding-top:12px; border-top:1px solid #f0f2f5;
              display:flex; align-items:baseline; gap:10px;">
    <span style="font-size:10px; color:#8892a4; text-transform:uppercase;
                 letter-spacing:.4px;">Total Amount</span>
    <span class="fv-lg">{fmt_amount(r.total_amount, r.currency)}</span>
  </div>
  <div style="margin-top:8px;">
    <div class="fl">Source File</div>
    <div class="fv-muted">{esc(r.file_path)}</div>
  </div>
</div>"""


def confidence_section(r: InvoiceReport, colours: dict) -> str:
    pct = r.translation_confidence * 100
    colour = colours["bar"]
    if r.was_translated:
        language = f'<span class="pill pill-blue">{esc(r.source_language or "?")} → en &nbsp;Translated</span>'
    else:
        language = '<span class="pill pill-green">Original English</span>'
    return f"""
<div class="card">
  <div class="card-title">Confidence &amp; Language</div>
  <div style="display:flex; gap:32px; align-items:flex-start; flex-wrap:wrap;">
    <div style="flex:1; min-width:240px;">
      <div class="fl">Source Language</div>
      <div class="fv" style="margin-top:3px;">{language}</div>
    </div>
    <div style="flex:2; min-width:280px;">
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
      <div class="fl">Translation Confidence</div>
        <div style="font-size:16px; font-weight:700; color:{colour};">{pct:.0f}%</div>
      </div>
      <div class="conf-track" style="margin-top:6px; height:8px;">
        <div class="conf-fill" style="width:{pct:.0f}%; background:{colour};"></div>
      </div>
      <div class="conf-labels">
        <span>0%</span>
        <span style="color:{colour}; font-weight:600;">{pct:.0f}%</span>
        <span>100%</span>
      </div>
      <div style="font-size:10px; color:#8892a4; margin-top:4px;">
        Confidence reported by the translation stage
      </div>
    </div>
  </div>
</div>"""


def data_validation_section(r: InvoiceReport) -> str:
    if r.data_validation_issues:
        rows = "".join(f"""
      <tr class="{row_class(i)}">
        <td>{severity_pill(i)}</td><td>{scope(i)}</td>
        <td class="fv-mono" style="font-size:11.5px;">{esc(i.get("field_name", "—"))}</td>
        <td>{esc(i.get("issue_type", "—"))}</td><td>{esc(i.get("message", "—"))}</td>
      </tr>""" for i in r.data_validation_issues)
        body = f"""
  <table>
    <thead><tr>
      <th>Sev.</th><th>Scope</th><th>Field</th><th>Check</th><th>Detail</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>"""
    else:
        body = '<div class="pass-banner">&#10003;&ensp;All data completeness checks passed — no issues found.</div>'

    return f"""
<div class="card">
  <div class="card-title">Data Validation</div>
  <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:4px;">
    <div>{status_pill(r.data_validation_status or "N/A")}</div>
    <div style="font-size:11px; color:#8892a4;">
      <span style="color:#dc3545; font-weight:600;">{r.data_validation_errors}</span> error(s)
      &ensp;·&ensp;
      <span style="color:#d97706; font-weight:600;">{r.data_validation_warnings}</span> warning(s)
    </div>
  </div>
  {body}
</div>"""


def business_validation_section(r: InvoiceReport) -> str:
    if r.business_discrepancies:
        rows = ""
        for d in r.business_discrepancies:
            diff = d.get("difference_pct")
            rows += f"""
      <tr class="{row_class(d)}">
        <td>{severity_pill(d)}</td><td>{scope(d)}</td><td>{esc(d.get("discrepancy_type", "—"))}</td>
        <td>{esc(d.get("message", "—"))}</td>
        <td class="num">{esc(d.get("invoice_value") or "—")}</td>
        <td class="num">{esc(d.get("erp_value") or "—")}</td>
        <td class="num" style="color:#d97706;">{f"{diff:+.1f}%" if diff is not None else "—"}</td>
      </tr>"""
        body = f"""
  <table>
    <thead><tr>
      <th>Sev.</th><th>Scope</th><th>Type</th><th>Detail</th>
      <th class="num">Invoice</th><th class="num">ERP</th><th class="num">Diff</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>"""
    else:
        body = '<div class="pass-banner">&#10003;&ensp;All line items matched ERP records — no discrepancies.</div>'

    return f"""
<div class="card">
  <div class="card-title">Business Validation · ERP Cross-Reference</div>
  <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:4px;">
    <div>{status_pill(r.business_validation_status or "N/A")}</div>
    <div style="font-size:11px; color:#8892a4;">
      <span style="color:#198754; font-weight:600;">{r.business_validation_matched}</span> matched
      &ensp;·&ensp;
      <span style="color:#dc3545; font-weight:600;">{r.business_validation_errors}</span> error(s)
      &ensp;·&ensp;
      <span style="color:#d97706; font-weight:600;">{r.business_validation_warnings}</span> warning(s)
    </div>
  </div>
  <div class="erp-row">
    <div class="erp-item">
      <span class="erp-label">PO</span>
      <span class="fv-mono" style="font-size:11.5px;">{esc(r.po_number or "—")}</span>
      {found_pill(r.erp_po_found)}
    </div>
    <div class="erp-item">
      <span class="erp-label">Vendor</span>
      <span class="fv-mono" style="font-size:11.5px;">{esc(r.vendor_id or "—")}</span>
      {found_pill(r.erp_vendor_found)}
    </div>
  </div>
  {body}
</div>"""


def line_items_section(r: InvoiceReport) -> str:
    if not r.line_items:
        return """
<div class="card">
  <div class="card-title">Line Items</div>
  <p style="color:#8892a4; font-size:12px;">No line items extracted.</p>
</div>"""

    rows = ""
    for n, line in enumerate(r.line_items, 1):
        original = str(line.get("original_description") or "—")
        translated = str(line.get("translated_description") or "—")
        description = esc(translated)
        if r.was_translated and original not in (translated, "—"):
            description += f'<br><span style="font-size:10px;color:#8892a4;">{esc(original)}</span>'
        rows += f"""
    <tr>
      <td style="color:#8892a4; font-size:11px; width:28px;">{n}</td>
      <td class="fv-mono" style="font-size:11.5px; white-space:nowrap;">{esc(line.get("item_code") or "—")}</td>
      <td>{description}</td>
      <td class="num">{fmt_num(line.get("qty"))}</td>
      <td class="num">{fmt_num(line.get("unit_price"))}</td>
      <td class="num" style="font-weight:600;">{fmt_num(line.get("total"))}</td>
    </tr>"""

    return f"""
<div class="card">
  <div class="card-title">Line Items <span style="font-weight:400; color:#8892a4;">({len(r.line_items)})</span></div>
  <table>
    <thead><tr>
      <th style="width:28px;">#</th>
      <th>Item Code</th>
      <th>Description</th>
      <th class="num">Qty</th>
      <th class="num">Unit Price</th>
      <th class="num">Total</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def reasons_section(r: InvoiceReport) -> str:
    if not r.recommendation_reasons:
        return ""
    items = "".join(f"<li>{esc(reason)}</li>" for reason in r.recommendation_reasons)
    return f"""
<div class="card">
  <div class="card-title">Decision Rationale</div>
  <ul class="reasons">{items}</ul>
</div>"""


# small helpers

def esc(value) -> str:
    """Escape text for HTML."""
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def scope(issue: dict) -> str:
    """Row number for line issues, 'Header' for header issues (line_index None or -1)."""
    idx = issue.get("line_index")
    return f"Row&nbsp;{idx}" if idx is not None and idx >= 0 else "Header"


def row_class(issue: dict) -> str:
    return "tr-err" if issue.get("severity", "error") == "error" else "tr-warn"


def severity_pill(issue: dict) -> str:
    if issue.get("severity", "error") == "error":
        return '<span class="pill pill-red">Error</span>'
    return '<span class="pill pill-amber">Warning</span>'


def found_pill(found: bool) -> str:
    if found:
        return '<span class="pill pill-green">&#10003;&ensp;Found</span>'
    return '<span class="pill pill-red">&#10007;&ensp;Not Found</span>'


def status_pill(status: str) -> str:
    if status in ("passed", "matched"):
        cls = "pill-green"
    elif status in ("passed_with_warnings", "discrepancies_found", "vendor_not_found"):
        cls = "pill-amber"
    elif status in ("failed", "po_not_found", "skipped"):
        cls = "pill-red"
    else:
        cls = "pill-grey"
    return f'<span class="pill {cls}">{status.lower().replace("_", " ").title()}</span>'


def fmt_amount(value: float | None, currency: str | None) -> str:
    if value is None:
        return "N/A"
    return f"{currency} {value:,.2f}" if currency else f"{value:,.2f}"


def fmt_num(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value)}" if value == int(value) else f"{value:,.2f}"