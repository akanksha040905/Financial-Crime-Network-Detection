"""
Investigation Report Generator for Financial Crime and AML Compliance.

Produces:
  1. Formal PDF Investigation Dossier (using ReportLab when available).
  2. Standalone HTML Printable Investigation Dossier (zero-dependency fallback).
  3. Structured CSV & JSON export bundles for audit trails and regulatory filings.
"""

from __future__ import annotations

import datetime
import html
import io
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

HAS_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _extract_pattern_props(p: Any) -> Tuple[str, str, str, str]:
    """Safely extract pattern fields whether p is a PatternFinding dataclass, dict, or object."""
    if isinstance(p, dict):
        return (
            str(p.get("pattern_type", "Typology")),
            str(p.get("severity", "Medium")),
            str(p.get("description", "")),
            str(p.get("evidence_summary", ""))
        )
    return (
        str(getattr(p, "pattern_type", "Typology")),
        str(getattr(p, "severity", "Medium")),
        str(getattr(p, "description", "")),
        str(getattr(p, "evidence_summary", ""))
    )


def _safe_float(d: Any, key: str, default: float = 0.0) -> float:
    if d is None:
        return default
    val = d.get(key, default) if hasattr(d, "get") else getattr(d, key, default)
    if isinstance(val, (dict, pd.Series)):
        val = list(val.values())[0] if isinstance(val, dict) else val.iloc[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(d: Any, key: str, default: int = 0) -> int:
    return int(_safe_float(d, key, float(default)))


def _safe_str(d: Any, key: str, default: str = "") -> str:
    if d is None:
        return default
    val = d.get(key, default) if hasattr(d, "get") else getattr(d, key, default)
    if isinstance(val, (dict, pd.Series)):
        val = list(val.values())[0] if isinstance(val, dict) else val.iloc[0]
    return str(val)


def generate_account_investigation_pdf(
    account_id: str,
    account_summary: Dict[str, Any],
    graph_metrics: Dict[str, Any],
    patterns: List[Any],
    transactions: pd.DataFrame,
    shap_narratives: List[str]
) -> bytes:
    """
    Generate a formal PDF investigation dossier for an individual flagged account.
    Returns PDF bytes.
    """
    if not HAS_REPORTLAB:
        # Fallback to HTML format if reportlab is not installed
        html_content = generate_account_investigation_html(
            account_id, account_summary, graph_metrics, patterns, transactions, shap_narratives
        )
        return html_content.encode("utf-8")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#12305E")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#5B6779")
    )

    heading2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#12305E"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#101A2E")
    )

    elements = []

    # Header Banner
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph("FINANCIAL CRIME INVESTIGATION DOSSIER", title_style))
    elements.append(Paragraph(f"CONFIDENTIAL // AML COMPLIANCE AUDIT TRAIL // Case Ref: INV-{account_id}-{datetime.datetime.now().strftime('%Y%m%d')}", subtitle_style))
    elements.append(Paragraph(f"Generated: {now_str} | Target Account: <b>{account_id}</b>", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#12305E"), spaceAfter=12))

    # 1. Executive Summary & Risk Profile
    elements.append(Paragraph("1. Account Risk Profile", heading2_style))
    
    risk_score = _safe_float(account_summary, "risk_score", 0.0)
    risk_cat = _safe_str(account_summary, "risk_category", "Low")
    inflow = _safe_float(account_summary, "total_received", 0.0)
    outflow = _safe_float(account_summary, "total_sent", 0.0)
    net_flow = _safe_float(account_summary, "net_flow", 0.0)
    tx_count = _safe_int(account_summary, "total_tx_count", 0)
    susp_count = _safe_int(account_summary, "suspicious_tx_count", 0)

    cat_color = colors.HexColor("#8B0000") if risk_cat in ("Critical", "High") else colors.HexColor("#1F9D8B")

    summary_data = [
        [Paragraph("<b>Risk Score:</b>", body_style), Paragraph(f"<b>{risk_score:.1f} / 100</b>", body_style),
         Paragraph("<b>Risk Tier:</b>", body_style), Paragraph(f"<b>{risk_cat.upper()}</b>", body_style)],
        [Paragraph("<b>Total Inflow:</b>", body_style), Paragraph(f"${inflow:,.2f}", body_style),
         Paragraph("<b>Total Outflow:</b>", body_style), Paragraph(f"${outflow:,.2f}", body_style)],
        [Paragraph("<b>Net Retention:</b>", body_style), Paragraph(f"${net_flow:,.2f}", body_style),
         Paragraph("<b>Total Transactions:</b>", body_style), Paragraph(f"{tx_count:,} ({susp_count} flagged)", body_style)]
    ]

    t_summary = Table(summary_data, colWidths=[100, 160, 120, 160])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE5")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 10))

    # 2. Graph Centrality & Topological Indicators
    elements.append(Paragraph("2. Network Topology & Centrality Indicators", heading2_style))
    in_deg = graph_metrics.get("in_degree", 0)
    out_deg = graph_metrics.get("out_degree", 0)
    bc = graph_metrics.get("betweenness_centrality", 0.0)
    pr = graph_metrics.get("pagerank", 0.0)
    comm_id = graph_metrics.get("community_id", 0)

    graph_data = [
        [Paragraph("<b>In-Degree (Senders):</b>", body_style), Paragraph(f"{in_deg}", body_style),
         Paragraph("<b>Out-Degree (Receivers):</b>", body_style), Paragraph(f"{out_deg}", body_style)],
        [Paragraph("<b>Betweenness Centrality:</b>", body_style), Paragraph(f"{bc:.5f}", body_style),
         Paragraph("<b>PageRank Score:</b>", body_style), Paragraph(f"{pr:.5f}", body_style)],
        [Paragraph("<b>Community Cluster ID:</b>", body_style), Paragraph(f"Cluster #{comm_id}", body_style),
         Paragraph("<b>Total Counterparties:</b>", body_style), Paragraph(f"{in_deg + out_deg}", body_style)]
    ]
    t_graph = Table(graph_data, colWidths=[130, 130, 130, 150])
    t_graph.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE5")),
        ('PADDING', (0, 0), (-1, -1), 5)
    ]))
    elements.append(t_graph)
    elements.append(Spacer(1, 10))

    # 3. Suspicious AML Patterns Detected
    elements.append(Paragraph("3. Detected AML Typologies & Structural Indicators", heading2_style))
    if not patterns:
        elements.append(Paragraph("No automated network typologies flagged for this account.", body_style))
    else:
        pat_rows = [[Paragraph("<b>Typology</b>", body_style), Paragraph("<b>Severity</b>", body_style), Paragraph("<b>Details & Evidence</b>", body_style)]]
        for p in patterns[:6]:
            p_type, sev, desc, ev = _extract_pattern_props(p)
            full_detail = f"{desc} {ev}"
            pat_rows.append([
                Paragraph(f"<b>{p_type}</b>", body_style),
                Paragraph(sev, body_style),
                Paragraph(full_detail, body_style)
            ])
        t_pat = Table(pat_rows, colWidths=[140, 70, 330])
        t_pat.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF0F4")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE5")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        elements.append(t_pat)
    elements.append(Spacer(1, 10))

    # 4. Machine Learning & SHAP Explainability Drivers
    elements.append(Paragraph("4. Machine Learning Risk Drivers (SHAP Attribution)", heading2_style))
    if not shap_narratives:
        elements.append(Paragraph("Local model feature attribution not generated or model is unsupervised.", body_style))
    else:
        for narr in shap_narratives[:5]:
            clean_text = narr.replace("<b>", "").replace("</b>", "")
            elements.append(Paragraph(f"• {clean_text}", body_style))
            elements.append(Spacer(1, 3))
    elements.append(Spacer(1, 10))

    # 5. Underlying Transaction Ledger (Sample)
    elements.append(Paragraph("5. Relevant Transaction Ledger (Top Records)", heading2_style))
    if transactions.empty:
        elements.append(Paragraph("No individual transaction records available.", body_style))
    else:
        tx_rows = [[
            Paragraph("<b>Timestamp</b>", body_style),
            Paragraph("<b>TX ID</b>", body_style),
            Paragraph("<b>Direction</b>", body_style),
            Paragraph("<b>Counterparty</b>", body_style),
            Paragraph("<b>Amount</b>", body_style),
            Paragraph("<b>Risk</b>", body_style)
        ]]
        for _, row in transactions.head(10).iterrows():
            is_inflow = str(row.get("receiver_account", "")) == account_id
            direction = "INFLOW" if is_inflow else "OUTFLOW"
            cparty = str(row.get("sender_account" if is_inflow else "receiver_account", ""))
            amt = float(row.get("amount", 0.0))
            score = float(row.get("risk_score", 0.0))
            ts = str(row.get("timestamp", ""))[:19]
            tid = str(row.get("transaction_id", ""))[:12]

            tx_rows.append([
                Paragraph(ts, body_style),
                Paragraph(tid, body_style),
                Paragraph(direction, body_style),
                Paragraph(cparty[:14], body_style),
                Paragraph(f"${amt:,.2f}", body_style),
                Paragraph(f"{score:.0f}", body_style)
            ])
        t_tx = Table(tx_rows, colWidths=[100, 80, 65, 120, 100, 75])
        t_tx.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF0F4")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE5")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        elements.append(t_tx)

    # 6. Compliance Sign-Off
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("6. Compliance Officer Disposition & Sign-off", heading2_style))
    sign_data = [
        [Paragraph("<b>Investigator Name:</b> _________________________", body_style),
         Paragraph("<b>Date:</b> _________________________", body_style)],
        [Paragraph("<b>Case Action:</b> [  ] SAR Filing  [  ] Account Freeze  [  ] Cleared  [  ] Further Review", body_style),
         Paragraph("<b>Signature:</b> _________________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[270, 270])
    t_sign.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(t_sign)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_account_investigation_html(
    account_id: str,
    account_summary: Dict[str, Any],
    graph_metrics: Dict[str, Any],
    patterns: List[Any],
    transactions: pd.DataFrame,
    shap_narratives: List[str]
) -> str:
    """
    Generate a standalone printable HTML dossier for an individual flagged account.
    Styled with enterprise AML formatting and print media rules.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_score = _safe_float(account_summary, "risk_score", 0.0)
    risk_cat = _safe_str(account_summary, "risk_category", "Low")
    inflow = _safe_float(account_summary, "total_received", 0.0)
    outflow = _safe_float(account_summary, "total_sent", 0.0)
    net_flow = _safe_float(account_summary, "net_flow", 0.0)
    tx_count = _safe_int(account_summary, "total_tx_count", 0)
    susp_count = _safe_int(account_summary, "suspicious_tx_count", 0)

    cat_badge_class = "badge-alert" if risk_cat in ("Critical", "High") else "badge-calm"

    # Pattern rows
    pat_html = ""
    if patterns:
        for p in patterns[:8]:
            p_type, sev, desc, ev = _extract_pattern_props(p)
            pat_html += f"""
            <tr>
                <td><b>{html.escape(p_type)}</b></td>
                <td><span class="badge {cat_badge_class}">{html.escape(sev)}</span></td>
                <td>{html.escape(desc)} <em>{html.escape(ev)}</em></td>
            </tr>
            """
    else:
        pat_html = "<tr><td colspan='3'>No automated network typologies flagged for this account.</td></tr>"

    # Narrative rows
    narr_html = ""
    if shap_narratives:
        for n in shap_narratives[:6]:
            narr_html += f"<li>{n}</li>"
    else:
        narr_html = "<li>Model is unsupervised or local attribution was not computed.</li>"

    # Transaction rows
    tx_html = ""
    if not transactions.empty:
        for _, row in transactions.head(15).iterrows():
            is_inflow = str(row.get("receiver_account", "")) == account_id
            direction = "INFLOW" if is_inflow else "OUTFLOW"
            cparty = str(row.get("sender_account" if is_inflow else "receiver_account", ""))
            amt = float(row.get("amount", 0.0))
            score = float(row.get("risk_score", 0.0))
            ts = str(row.get("timestamp", ""))[:19]
            tid = str(row.get("transaction_id", ""))
            tx_html += f"""
            <tr>
                <td>{html.escape(ts)}</td>
                <td><code>{html.escape(tid)}</code></td>
                <td><b>{direction}</b></td>
                <td>{html.escape(cparty)}</td>
                <td style="text-align:right;">${amt:,.2f}</td>
                <td style="text-align:right;"><b>{score:.0f}</b></td>
            </tr>
            """
    else:
        tx_html = "<tr><td colspan='6'>No individual transactions recorded.</td></tr>"

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Investigation Dossier - {html.escape(account_id)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #101A2E; background: #FFF; }}
.header {{ border-bottom: 2px solid #12305E; padding-bottom: 12px; margin-bottom: 20px; }}
.header h1 {{ margin: 0 0 6px 0; color: #12305E; font-size: 22px; }}
.header .sub {{ color: #5B6779; font-size: 12px; }}
h2 {{ color: #12305E; font-size: 15px; margin: 20px 0 10px; border-bottom: 1px solid #D5DBE5; padding-bottom: 4px; }}
.grid-cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
.card {{ background: #F8FAFC; border: 1px solid #D5DBE5; border-radius: 4px; padding: 10px 12px; }}
.card .val {{ font-size: 20px; font-weight: 700; color: #12305E; }}
.card .lbl {{ font-size: 11px; color: #5B6779; text-transform: uppercase; margin-top: 3px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }}
th {{ background: #EDF0F4; color: #101A2E; text-align: left; padding: 6px 8px; border: 1px solid #D5DBE5; }}
td {{ padding: 6px 8px; border: 1px solid #D5DBE5; vertical-align: top; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
.badge-alert {{ background: #FDE3E7; color: #8B0000; }}
.badge-calm {{ background: #EAF6F4; color: #1F9D8B; }}
ul {{ margin: 6px 0 16px 20px; font-size: 12px; line-height: 1.6; }}
.signoff {{ margin-top: 30px; border: 1px dashed #B9C4D3; padding: 14px; border-radius: 4px; font-size: 12px; }}
@media print {{ body {{ margin: 15mm; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<div class="no-print" style="margin-bottom: 16px;">
    <button onclick="window.print()" style="padding: 8px 16px; background: #12305E; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;">Print / Save as PDF</button>
</div>
<div class="header">
    <h1>ANTI-MONEY LAUNDERING INVESTIGATION DOSSIER</h1>
    <div class="sub">CONFIDENTIAL // AML COMPLIANCE AUDIT TRAIL // Case Ref: INV-{html.escape(account_id)} // Generated: {now_str}</div>
</div>

<div class="grid-cards">
    <div class="card">
        <div class="val">{risk_score:.1f} / 100</div>
        <div class="lbl">Risk Score ({html.escape(risk_cat)})</div>
    </div>
    <div class="card">
        <div class="val">${inflow:,.2f}</div>
        <div class="lbl">Total Inflow</div>
    </div>
    <div class="card">
        <div class="val">${outflow:,.2f}</div>
        <div class="lbl">Total Outflow</div>
    </div>
    <div class="card">
        <div class="val">{tx_count:,} ({susp_count} Flagged)</div>
        <div class="lbl">Total Transactions</div>
    </div>
</div>

<h2>1. Network Position & Graph Centralities</h2>
<table>
    <tr>
        <th>Metric</th><th>Value</th><th>Metric</th><th>Value</th>
    </tr>
    <tr>
        <td><b>In-Degree (Senders)</b></td><td>{graph_metrics.get('in_degree', 0)}</td>
        <td><b>Out-Degree (Receivers)</b></td><td>{graph_metrics.get('out_degree', 0)}</td>
    </tr>
    <tr>
        <td><b>Betweenness Centrality</b></td><td>{graph_metrics.get('betweenness_centrality', 0.0):.5f}</td>
        <td><b>PageRank Score</b></td><td>{graph_metrics.get('pagerank', 0.0):.5f}</td>
    </tr>
    <tr>
        <td><b>Community ID</b></td><td>Cluster #{graph_metrics.get('community_id', 0)}</td>
        <td><b>Total Counterparties</b></td><td>{graph_metrics.get('in_degree', 0) + graph_metrics.get('out_degree', 0)}</td>
    </tr>
</table>

<h2>2. Detected Suspicious Typologies & Graph Patterns</h2>
<table>
    <tr><th>Typology</th><th>Severity</th><th>Evidence & Finding</th></tr>
    {pat_html}
</table>

<h2>3. Model Explainability & SHAP Risk Drivers</h2>
<ul>
    {narr_html}
</ul>

<h2>4. Transaction Activity Log (Top Records)</h2>
<table>
    <tr><th>Timestamp</th><th>Transaction ID</th><th>Direction</th><th>Counterparty</th><th>Amount</th><th>Risk Score</th></tr>
    {tx_html}
</table>

<div class="signoff">
    <b>Compliance Officer Review & Disposition:</b><br><br>
    Investigator Name: ____________________________________ &nbsp;&nbsp;&nbsp;&nbsp; Date: ____________________<br><br>
    Action Taken: [ &nbsp; ] SAR Filing &nbsp;&nbsp;&nbsp; [ &nbsp; ] Account Freeze &nbsp;&nbsp;&nbsp; [ &nbsp; ] Cleared &nbsp;&nbsp;&nbsp; [ &nbsp; ] Enhanced Monitoring<br><br>
    Signature: ____________________________________________
</div>
</body>
</html>
"""
    return html_page


def export_bundle_to_csv_dict(
    df_cleaned: pd.DataFrame,
    df_tx_risk: pd.DataFrame,
    df_account_risk: pd.DataFrame,
    df_metrics: pd.DataFrame,
    pattern_findings: List[Any]
) -> Dict[str, str]:
    """
    Export all investigation tables to a dictionary of CSV strings.
    """
    # Pattern findings table
    pat_rows = []
    for p in pattern_findings:
        pat_rows.append({
            "pattern_type": getattr(p, "pattern_type", ""),
            "severity": getattr(p, "severity", ""),
            "primary_account": getattr(p, "primary_account", ""),
            "involved_accounts": "; ".join(getattr(p, "involved_accounts", [])),
            "description": getattr(p, "description", ""),
            "evidence_summary": getattr(p, "evidence_summary", ""),
            "total_amount": getattr(p, "total_amount", 0.0),
            "tx_ids": "; ".join(getattr(p, "transaction_ids", []))
        })
    df_pat = pd.DataFrame(pat_rows)

    return {
        "cleaned_transactions.csv": df_cleaned.to_csv(index=False),
        "transaction_risk_register.csv": df_tx_risk.to_csv(index=False),
        "account_risk_register.csv": df_account_risk.to_csv(index=True),
        "graph_analytics_metrics.csv": df_metrics.to_csv(index=True),
        "suspicious_patterns_findings.csv": df_pat.to_csv(index=False)
    }
