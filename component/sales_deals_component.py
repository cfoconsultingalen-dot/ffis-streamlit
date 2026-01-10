# pages/sales_deals_component.py

from __future__ import annotations
from html import escape
from typing import Any, Dict, List

import streamlit as st
from streamlit.components.v1 import html as st_html


# ----------------------------
# Helpers
# ----------------------------
def _s(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)

def _lst(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return []

def _render_kpi_cells(items: List[Dict[str, Any]], *, max_n: int) -> str:
    cells = []
    for it in (items or [])[:max_n]:
        label = escape(_s(it.get("label"), "—"))
        value = escape(_s(it.get("value"), "—"))
        note = _s(it.get("note"), "").strip()
        note_html = f"<div class='kpi-note'>{escape(note)}</div>" if note else ""
        cells.append(
            f"""
            <div class="kpi-cell">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              {note_html}
            </div>
            """
        )
    return "\n".join(cells)

def _render_bullets(bullets: List[str], fallback: str) -> str:
    items = [b.strip() for b in (bullets or []) if str(b).strip()]
    if not items:
        items = [fallback]
    lis = "\n".join([f"<li>{escape(i)}</li>" for i in items])
    return f"<ul class='bullets'>{lis}</ul>"

def _render_checklist(items: List[str], fallback: str) -> str:
    rows = [i.strip() for i in (items or []) if str(i).strip()]
    if not rows:
        rows = [fallback]
    out = []
    for r in rows:
        out.append(
            f"""
            <div class="check-row">
              <div class="check-dot"></div>
              <div class="check-text">{escape(r)}</div>
            </div>
            """
        )
    return "\n".join(out)

def _issue_row(icon: str, text: str) -> str:
    icon = escape(_s(icon, "alert-circle"))
    text = escape(_s(text, ""))
    return f"""
      <div class="issue-row">
        <i data-lucide="{icon}" class="icon issue-icon"></i>
        <div class="issue-text">{text}</div>
      </div>
    """

def _status_class(state: str) -> str:
    s = (state or "").strip().lower()
    if s in ("risk", "error", "red", "danger", "high"):
        return "status-risk"
    if s in ("warning", "amber", "watch"):
        return "status-warning"
    return "status-success"


# ----------------------------
# HTML shell (same tokens as Team Spending)
# ----------------------------
def _html_shell(inner: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/lucide@latest"></script>
<style>
:root {{
  --bg-app:#F8FAFC;
  --bg-card:#FFFFFF;
  --border:#E5E7EB;

  --text-h1:#0F172A;
  --text-h2:#1F2937;
  --text-body:#3A454B;
  --text-muted:#64748B;

  --success-bg:#ECFDF3;
  --success-border:#BBF7D0;

  --warning-bg:#FFFBEB;
  --warning-border:#FDE68A;

  --risk-bg:#FEF2F2;
  --risk-border:#FCA5A5;

  --radius:12px;
}}

body {{
  margin:0;
  background:var(--bg-app);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
}}

.main {{
  max-width:1100px;
  margin:0 auto;
  padding:24px 24px 48px;
  display:flex;
  flex-direction:column;
  gap:32px;
}}

.icon {{
  width:20px;
  height:20px;
  color:#64748B;
  stroke-width:1.8;
}}

.h1 {{
  display:flex;
  align-items:center;
  gap:10px;
  font-size:20px;
  font-weight:600;
  line-height:28px;
  color:var(--text-h1);
}}

.subtitle {{
  margin-top:6px;
  font-size:14px;
  line-height:22px;
  color:var(--text-muted);
}}

.card {{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:20px;
}}

.card-title {{
  display:flex;
  align-items:center;
  gap:10px;
  font-size:16px;
  font-weight:600;
  line-height:24px;
  color:var(--text-h2);
}}

.card-subtitle {{
  margin-top:6px;
  font-size:14px;
  line-height:22px;
  color:var(--text-muted);
}}

.kpi-grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:16px; }}
.kpi-grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:16px; }}
.kpi-grid-2 {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:16px; }}

.kpi-cell {{
  border:1px solid var(--border);
  border-radius:12px;
  background:#fff;
  padding:16px;
}}

.kpi-label {{
  font-size:12px;
  font-weight:500;
  line-height:18px;
  color:var(--text-muted);
  margin-bottom:8px;
}}

.kpi-value {{
  font-size:18px;
  font-weight:600;
  line-height:26px;
  color:var(--text-h1);
}}

.kpi-note {{
  margin-top:10px;
  font-size:12px;
  line-height:18px;
  color:var(--text-muted);
}}

.status-strip {{
  margin-top:14px;
  border-radius:10px;
  padding:10px 12px;
  border:1px solid var(--border);
  font-size:14px;
  line-height:22px;
  color:var(--text-body);
}}

.status-success {{ background:var(--success-bg); border-color:var(--success-border); }}
.status-warning {{ background:var(--warning-bg); border-color:var(--warning-border); }}
.status-risk    {{ background:var(--risk-bg);    border-color:var(--risk-border); }}

.helper {{
  margin-top:12px;
  font-size:13px;
  line-height:20px;
  color:var(--text-muted);
}}

.chart-shell {{
  margin-top:14px;
  border:1px solid var(--border);
  border-radius:10px;
  background:linear-gradient(135deg,#EEF2F7,#F8FAFC);
  padding:12px;
}}

.chart-title {{
  margin-top:12px;
  font-size:14px;
  font-weight:600;
  color:var(--text-h2);
}}

.chart-explain {{
  margin-top:10px;
  font-size:14px;
  line-height:22px;
  color:var(--text-body);
}}

.plotly-wrap {{
  width:100%;
  overflow:hidden;
}}

.plotly-wrap > div {{
  width:100% !important;
}}

.empty {{
  margin-top:10px;
  font-size:13px;
  line-height:20px;
  color:var(--text-muted);
}}

.issue-list {{
  margin-top:14px;
  display:flex;
  flex-direction:column;
  gap:10px;
}}

.issue-row {{
  display:flex;
  align-items:flex-start;
  gap:10px;
}}

.issue-icon {{
  margin-top:2px;
}}

.issue-text {{
  font-size:14px;
  line-height:22px;
  color:var(--text-body);
}}

.inline-link {{
  margin-top:12px;
  display:inline-block;
  font-size:14px;
  line-height:22px;
  color:var(--text-h2);
  text-decoration:none;
}}

.inline-link:hover {{ text-decoration:underline; }}

.bullets {{
  margin:12px 0 0 0;
  padding-left:18px;
  color:var(--text-body);
  font-size:14px;
  line-height:22px;
}}
.bullets li {{ margin:8px 0; }}

.check-row {{
  display:flex;
  align-items:flex-start;
  gap:12px;
  padding:10px 0;
}}
.check-dot {{
  width:18px;
  height:18px;
  border-radius:6px;
  border:1px solid var(--border);
  margin-top:2px;
  background:#fff;
}}
.check-text {{
  font-size:14px;
  line-height:22px;
  color:var(--text-body);
}}
</style>
</head>

<body>
<div class="main">
{inner}
</div>

<script>lucide.createIcons();</script>
</body>
</html>
"""


# ----------------------------
# Public renderer
# ----------------------------
def render_sales_deals(data: Dict[str, Any], *, height: int = 2100) -> None:
    header = data.get("header", {}) or {}

    ca = data.get("card_a_revenue_health", {}) or {}
    cb = data.get("card_b_revenue_now_next", {}) or {}
    cc = data.get("card_c_pipeline_reality", {}) or {}
    cd = data.get("card_d_revenue_risk_data_issues", {}) or {}
    ce = data.get("card_e_what_this_means", {}) or {}
    cf = data.get("card_f_action_plan", {}) or {}

    # Header
    title = escape(_s(header.get("title"), "Revenue & Deals"))
    subtitle = escape(_s(header.get("subtitle"), ""))
    icon_header = escape(_s(header.get("icon"), "handshake"))

    # Card A
    icon_a = escape(_s(ca.get("icon"), "activity"))
    a_kpis_html = _render_kpi_cells(ca.get("kpis", []) or [], max_n=4)
    st_state = _status_class(_s(ca.get("status_strip", {}).get("state"), "success"))
    st_msg = escape(_s(ca.get("status_strip", {}).get("message"), ""))

    # Card B
    icon_b = escape(_s(cb.get("icon"), "calendar"))
    b_kpis_html = _render_kpi_cells(cb.get("kpis", []) or [], max_n=2)
    b_helper = escape(_s(cb.get("helper_text"), "").strip())

    # Card C
    icon_c = escape(_s(cc.get("icon"), "bar-chart-3"))
    c_kpis_html = _render_kpi_cells(cc.get("kpis", []) or [], max_n=3)

    chart = cc.get("chart", {}) or {}
    chart_title = escape(_s(chart.get("title"), "Waterfall Chart"))
    plotly_html = _s(chart.get("plotly_html"), "").strip()
    empty_text = escape(_s(chart.get("empty_text"), "No pipeline data to display."))
    expl = escape(_s(cc.get("explanation"), "").strip())

    chart_block = ""
    if plotly_html:
        chart_block = f"""
        <div class="chart-shell">
          <div class="chart-title">{chart_title}</div>
          <div class="plotly-wrap">{plotly_html}</div>
        </div>
        """
    else:
        chart_block = f"""
        <div class="chart-shell">
          <div class="chart-title">{chart_title}</div>
          <div class="empty">{empty_text}</div>
        </div>
        """

    # Card D issues
    icon_d = escape(_s(cd.get("icon"), "shield-alert"))
    issues = cd.get("issues", []) or []
    issue_rows_html = []
    for it in issues[:8]:
        issue_rows_html.append(_issue_row(_s(it.get("icon"), "alert-circle"), _s(it.get("text"), "")))
    if not issue_rows_html:
        issue_rows_html = [_issue_row("check-circle", "No risk or data integrity issues detected for this month.")]
    issue_rows_html = "\n".join(issue_rows_html)

    inline_link = cd.get("inline_link", {}) or {}
    link_label = _s(inline_link.get("label"), "").strip()
    link_href = _s(inline_link.get("href"), "").strip()
    link_html = ""
    if link_label and link_href:
        link_html = f"<a class='inline-link' href='{escape(link_href)}' target='_blank' rel='noopener'>{escape(link_label)} →</a>"

    # Card E bullets
    icon_e = escape(_s(ce.get("icon"), "sparkles"))
    e_fallback = _s(ce.get("fallback"), "No narrative insights available yet.")
    e_bullets_html = _render_bullets(_lst(ce.get("bullets")), fallback=e_fallback)

    # Card F checklist
    icon_f = escape(_s(cf.get("icon"), "circle-check-big"))
    f_fallback = _s(cf.get("fallback"), "No actions triggered — maintain pipeline hygiene and update weekly.")
    f_items_html = _render_checklist(_lst(cf.get("items")), fallback=f_fallback)

    inner = f"""
  <!-- HEADER -->
  <div>
    <div class="h1">
      <i data-lucide="{icon_header}" class="icon"></i>
      {title}
    </div>
    <div class="subtitle">{subtitle}</div>
  </div>

  <!-- CARD A -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_a}" class="icon"></i>{escape(_s(ca.get("title"), "Revenue health (this month)"))}</div>
    <div class="card-subtitle">{escape(_s(ca.get("subtitle"), ""))}</div>
    <div class="kpi-grid-4">{a_kpis_html}</div>
    {"<div class='status-strip "+st_state+"'>"+st_msg+"</div>" if st_msg else ""}
  </div>

  <!-- CARD B -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_b}" class="icon"></i>{escape(_s(cb.get("title"), "Revenue now & next"))}</div>
    <div class="card-subtitle">{escape(_s(cb.get("subtitle"), ""))}</div>
    <div class="kpi-grid-2">{b_kpis_html}</div>
    {"<div class='helper'>"+b_helper+"</div>" if b_helper else ""}
  </div>

  <!-- CARD C -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_c}" class="icon"></i>{escape(_s(cc.get("title"), "Pipeline Reality"))}</div>
    <div class="card-subtitle">{escape(_s(cc.get("subtitle"), ""))}</div>
    <div class="kpi-grid-3">{c_kpis_html}</div>
    {chart_block}
    {"<div class='chart-explain'>"+expl+"</div>" if expl else ""}
  </div>

  <!-- CARD D -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_d}" class="icon"></i>{escape(_s(cd.get("title"), "Revenue Risk & Data Issues"))}</div>
    <div class="card-subtitle">{escape(_s(cd.get("subtitle"), ""))}</div>
    <div class="issue-list">{issue_rows_html}</div>
    {link_html}
  </div>

  <!-- CARD E -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_e}" class="icon"></i>{escape(_s(ce.get("title"), "What this means"))}</div>
    {e_bullets_html}
  </div>

  <!-- CARD F -->
  <div class="card">
    <div class="card-title"><i data-lucide="{icon_f}" class="icon"></i>{escape(_s(cf.get("title"), "Action plan – next 7 days"))}</div>
    <div style="margin-top:6px;"></div>
    {f_items_html}
  </div>
"""
    st_html(_html_shell(inner), height=height, scrolling=False)
