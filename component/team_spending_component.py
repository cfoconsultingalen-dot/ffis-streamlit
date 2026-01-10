# pages/team_spending_component.py

from __future__ import annotations
from html import escape
from typing import Any, Dict, List

import streamlit as st
from streamlit.components.v1 import html as st_html


# ---------- helpers (keep yours as-is) ----------
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

def _callout_class(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k in ("risk", "red", "danger", "high"):
        return "callout-risk"
    return "callout-warning"


# ---------- shared HTML shell ----------
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
  --warning-bg:#FFFBEB;
  --risk-bg:#FEF2F2;

  --radius:12px;
}}

body {{
  margin:0;
  background:var(--bg-app);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
}}

.main {{
  max-width:1180px;
  margin:0 auto;
  padding:24px 24px 24px;
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

.footnote {{
  margin-top:12px;
  font-size:12px;
  line-height:18px;
  color:var(--text-muted);
}}

.callouts {{
  display:flex;
  flex-direction:column;
  gap:12px;
  margin-top:16px;
}}

.callout {{
  border-radius:10px;
  padding:12px;
  border:1px solid var(--border);
}}

.callout-warning {{
  background:var(--warning-bg);
  border-color:#FDE68A;
}}

.callout-risk {{
  background:var(--risk-bg);
  border-color:#FCA5A5;
}}

.callout-title {{
  font-size:14px;
  font-weight:600;
  line-height:22px;
  color:var(--text-h2);
}}

.callout-body {{
  margin-top:6px;
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

.empty {{
  margin-top:12px;
  font-size:13px;
  line-height:20px;
  color:var(--text-muted);
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


def render_team_spending_top(data: Dict[str, Any], *, height: int = 1400) -> None:
    """
    Renders: Header + Card A + Card B + Card C (ONLY).
    Card D is rendered in Streamlit so the chart can sit inside it.
    """
    header = data.get("header", {}) or {}

    card_a = data.get("card_a_team_cost_health", {}) or {}
    card_b = data.get("card_b_payroll_snapshot", {}) or {}
    card_c = data.get("card_c_hiring_sensitivity", {}) or {}

    title = escape(_s(header.get("title"), "Team & Payroll"))
    subtitle = escape(_s(header.get("subtitle"), ""))

    icon_header = _s(header.get("icon"), "users")
    icon_a = _s(card_a.get("icon"), "heart-pulse")
    icon_b = _s(card_b.get("icon"), "wallet")
    icon_c = _s(card_c.get("icon"), "target")

    a_kpis_html = _render_kpi_cells(card_a.get("kpis", []) or [], max_n=4)
    b_kpis_html = _render_kpi_cells(card_b.get("kpis", []) or [], max_n=4)
    c_kpis_html = _render_kpi_cells(card_c.get("kpis", []) or [], max_n=3)

    b_footnote = _s(card_b.get("footnote_helper"), "").strip()

    inner = f"""
  <!-- HEADER -->
  <div>
    <div class="h1">
      <i data-lucide="{escape(icon_header)}" class="icon"></i>
      {title}
    </div>
    <div class="subtitle">{subtitle}</div>
  </div>

  <!-- CARD A -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_a)}" class="icon"></i>{escape(_s(card_a.get("title"), "Team Cost Health"))}</div>
    <div class="card-subtitle">{escape(_s(card_a.get("subtitle"), "Monthly payroll and headcount versus budget"))}</div>
    <div class="kpi-grid-4">{a_kpis_html}</div>
  </div>

  <!-- CARD B -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_b)}" class="icon"></i>{escape(_s(card_b.get("title"), "Current Payroll Snapshot"))}</div>
    <div class="card-subtitle">{escape(_s(card_b.get("subtitle"), "Fully loaded monthly people cost"))}</div>
    <div class="kpi-grid-4">{b_kpis_html}</div>
    {"<div class='footnote'>"+escape(b_footnote)+"</div>" if b_footnote else ""}
  </div>

  <!-- CARD C -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_c)}" class="icon"></i>{escape(_s(card_c.get("title"), "Hiring Sensitivity & Burn"))}</div>
    <div class="card-subtitle">{escape(_s(card_c.get("subtitle"), "Incremental cost and runway sensitivity"))}</div>
    <div class="kpi-grid-3">{c_kpis_html}</div>
  </div>
"""
    st_html(_html_shell(inner), height=height, scrolling=False)


def render_team_spending_bottom(data: Dict[str, Any], *, height: int = 1400) -> None:
    """
    Renders: Card E + Card F + Card G (ONLY).
    """
    card_e = data.get("card_e_risk_compliance", {}) or {}
    card_f = data.get("card_f_what_this_means", {}) or {}
    card_g = data.get("card_g_action_plan", {}) or {}

    icon_e = _s(card_e.get("icon"), "shield-alert")
    icon_f = _s(card_f.get("icon"), "sparkles")
    icon_g = _s(card_g.get("icon"), "circle-check-big")

    # Callouts
    e_callouts = card_e.get("callouts", []) or []
    callouts_html = []
    for c in e_callouts:
        c_title = escape(_s(c.get("title"), ""))
        c_body = escape(_s(c.get("message"), ""))
        c_kind = _s(c.get("kind"), "warning")
        cls = _callout_class(c_kind)
        if not (c_title or c_body):
            continue
        callouts_html.append(
            f"""
            <div class="callout {cls}">
              <div class="callout-title">{c_title}</div>
              <div class="callout-body">{c_body}</div>
            </div>
            """
        )
    callouts_html = "\n".join(callouts_html) if callouts_html else "<div class='empty'>No payroll risk signals triggered for this month.</div>"

    # Link
    e_link = card_e.get("inline_link", {}) or {}
    e_link_label = _s(e_link.get("label"), "").strip()
    e_link_href = _s(e_link.get("href"), "").strip()
    link_html = ""
    if e_link_label and e_link_href:
        link_html = f"<a class='inline-link' href='{escape(e_link_href)}' target='_blank' rel='noopener'>{escape(e_link_label)} →</a>"

    # Bullets / checklist
    f_bullets_html = _render_bullets(_lst(card_f.get("bullets")), fallback="No insights available yet — connect payroll history to generate narrative.")
    g_items_html = _render_checklist(_lst(card_g.get("items")), fallback="No actions suggested yet — once inputs are complete, actions will appear here.")

    inner = f"""
  <!-- CARD E -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_e)}" class="icon"></i>{escape(_s(card_e.get("title"), "Payroll risk & compliance"))}</div>
    <div class="card-subtitle">{escape(_s(card_e.get("subtitle"), "Potential risks and upcoming payroll obligations"))}</div>
    <div class="callouts">{callouts_html}</div>
    {link_html}
  </div>

  <!-- CARD F -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_f)}" class="icon"></i>{escape(_s(card_f.get("title"), "What this means"))}</div>
    {f_bullets_html}
  </div>

  <!-- CARD G -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_g)}" class="icon"></i>{escape(_s(card_g.get("title"), "Action plan – next 7 days"))}</div>
    <div style="margin-top:6px;"></div>
    {g_items_html}
  </div>
"""
    st_html(_html_shell(inner), height=height, scrolling=False)


def render_team_spending_card_d(data: Dict[str, Any], *, height: int = 460) -> None:
    """
    Renders: Card D (Headcount & cost drivers) — HTML-based, consistent with other cards.
    Expects:
      data["card_d_headcount_cost_drivers"]["kpis"] (3 KPIs)
      data["card_d_headcount_cost_drivers"]["plotly_html"] (string, optional)
      data["card_d_headcount_cost_drivers"]["empty_text"] (string, optional)
    """
    card_d = data.get("card_d_headcount_cost_drivers", {}) or {}

    icon_d = _s(card_d.get("icon"), "line-chart")
    title = escape(_s(card_d.get("title"), "Headcount & cost drivers"))
    subtitle = escape(_s(card_d.get("subtitle"), "What’s driving changes in team size and payroll"))

    kpis_html = _render_kpi_cells(card_d.get("kpis", []) or [], max_n=3)

    # Chart handling
    plotly_html = _s(card_d.get("plotly_html"), "").strip()
    empty_text = escape(_s(card_d.get("empty_text"), "Headcount trend unavailable (no payroll roles / insufficient history)."))

    chart_block = ""
    if plotly_html:
        chart_block = f"""
        <div style="
          margin-top:12px;
          border:1px solid var(--border);
          border-radius:10px;
          background:#FFFFFF;
          padding:12px;
        ">
          {plotly_html}
        </div>
        """
    else:
        chart_block = f"<div class='empty' style='margin-top:12px;'>{empty_text}</div>"

    inner = f"""
  <!-- CARD D -->
  <div class="card">
    <div class="card-title"><i data-lucide="{escape(icon_d)}" class="icon"></i>{title}</div>
    <div class="card-subtitle">{subtitle}</div>

    <div class="kpi-grid-3" style="margin-top:16px;">{kpis_html}</div>

    <div style="margin-top:14px; font-size:14px; font-weight:600; color:var(--text-h2);">
      Headcount trend (past 6 months)
    </div>
    {chart_block}
  </div>
"""
    st_html(_html_shell(inner), height=height, scrolling=False)
