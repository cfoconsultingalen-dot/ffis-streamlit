# pages/funding_strategy_component.py
import streamlit as st
from streamlit.components.v1 import html as st_html


def _esc(x):
    if x is None:
        return ""
    return (
        str(x)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _join(items):
    return "\n".join(items) if items else ""


def _kpi_cell(k: dict) -> str:
    note = k.get("note", "")
    note_html = f"<div class='kpi-note'>{_esc(note)}</div>" if note else ""
    return f"""
    <div class="kpi">
      <div class="kpi-label">{_esc(k.get("label",""))}</div>
      <div class="kpi-value">{_esc(k.get("value","—"))}</div>
      {note_html}
    </div>
    """


def _base_styles() -> str:
    return """
    <style>
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 0 16px 0;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: #F8FAFC;
      }

      .header {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 18px;
      }
      .h1 {
        font-size: 20px;
        line-height: 28px;
        font-weight: 600;
        color: #0F172A;
      }
      .meta {
        font-size: 14px;
        line-height: 22px;
        font-weight: 400;
        color: #64748B;
      }

      .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 32px;
      }

      
      .wrap { padding-bottom: 0 !important; }
      .card:last-child { margin-bottom: 0 !important; }
      .card-title {
        font-size: 14px;
        line-height: 22px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 8px;
      }
      .sub {
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
        margin-bottom: 12px;
      }
      .hint {
        margin-top: 10px;
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
      }

      .kpi-grid { display: grid; gap: 16px; margin-bottom: 14px; }
      .kpi-4 { grid-template-columns: repeat(4, 1fr); }
      .kpi { display: flex; flex-direction: column; gap: 6px; }
      .kpi-label { font-size: 12px; line-height: 18px; color: #64748B; font-weight: 400; }
      .kpi-value { font-size: 18px; line-height: 24px; color: #0F172A; font-weight: 600; white-space: nowrap; }
      .kpi-note { font-size: 11px; line-height: 16px; color: #64748B; }

      .status { width: 100%; border-radius: 10px; padding: 10px 12px; font-size: 13px; line-height: 20px; font-weight: 500; }
      .success { background: #ECFDF3; color: #166534; border: 1px solid #BBF7D0; }
      .warning { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
      .risk { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
      .info { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }

      .chartbox { border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px; background: #FFFFFF; overflow: hidden; }
      .chart-inner { width: 100%; overflow: hidden; }
      .chartbox .plotly-graph-div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; }
      .chartbox svg, .chartbox canvas { max-width: 100% !important; }

      .empty { color: #64748B; padding: 16px; border: 1px dashed #E5E7EB; border-radius: 12px; }

      .bullets, .checks { margin: 0; padding-left: 18px; color: #3A454B; font-size: 13px; line-height: 22px; }
      .bullets li, .checks li { margin: 8px 0; }

      @media (max-width: 1100px) { .wrap { padding-left: 12px; padding-right: 12px; } }
      @media (max-width: 900px) { .kpi-4 { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """


def render_funding_strategy_top(data: dict, height: int = 980):
    """
    Renders only:
    - Header
    - Card A (Runway & Burn Health)
    - Card B (Cash cliff & timing risk)
    """
    header = data.get("header", {}) or {}
    cards = data.get("cards", {}) or {}

    card_a = cards.get("runway_burn_health", {}) or {}
    a_kpis = card_a.get("kpis", []) or []
    a_status = card_a.get("status_strip", {}) or {}
    a_state = str(a_status.get("state", "success")).lower()
    a_msg = a_status.get("message", "")

    state_class = {
        "success": "status success",
        "warning": "status warning",
        "risk": "status risk",
        "negative": "status risk",
        "info": "status info",
    }.get(a_state, "status success")

    card_b = cards.get("cash_cliff_risk", {}) or {}
    b_chart = (card_b.get("chart", {}) or {})
    b_plotly_html = b_chart.get("plotly_html", "")
    b_empty = b_chart.get("empty_text", "No data.")
    b_caption = card_b.get("caption", "")

    html = f"""
    <div class="wrap">
      <div class="header">
        <div class="h1">{_esc(header.get("title","Funding Strategy"))}</div>
        <div class="meta">{_esc(header.get("subtitle",""))}</div>
      </div>

      <div class="card">
        <div class="card-title">{_esc(card_a.get("title","Runway & Burn Health"))}</div>
        <div class="sub">{_esc(card_a.get("subtitle",""))}</div>
        <div class="kpi-grid kpi-4">
          {_join([_kpi_cell(k) for k in a_kpis[:4]])}
        </div>
        <div class="{state_class}">{_esc(a_msg)}</div>
      </div>

      <div class="card">
        <div class="card-title">{_esc(card_b.get("title","Cash cliff & timing risk"))}</div>
        <div class="sub">{_esc(card_b.get("subtitle",""))}</div>
        <div class="chartbox">
          <div class="chart-inner">
            {b_plotly_html if b_plotly_html else f"<div class='empty'>{_esc(b_empty)}</div>"}
          </div>
        </div>
        <div class="hint">{_esc(b_caption)}</div>
      </div>
    </div>
    {_base_styles()}
    """
    st_html(html, height=height, scrolling=False)


def render_funding_strategy_bottom(data: dict, height: int = 900):
    """
    Renders only:
    - Card D (What this means)
    - Card E (Action plan)
    No header, no runway, no chart.
    """
    cards = data.get("cards", {}) or {}

    card_d = cards.get("what_this_means", {}) or {}
    d_bullets = card_d.get("bullets", []) or []
    d_fallback = card_d.get("fallback", "Run scenarios to generate narrative.")
    if not d_bullets:
        d_bullets = [d_fallback]
    d_li = "\n".join([f"<li>{_esc(b)}</li>" for b in d_bullets[:5]])

    card_e = cards.get("action_plan_30_90", {}) or {}
    e_items = card_e.get("items", []) or []
    e_fallback = card_e.get("fallback", "Run scenarios to generate actions.")
    if not e_items:
        e_items = [e_fallback]
    e_li = "\n".join([f"<li>{_esc(a)}</li>" for a in e_items[:6]])

    html = f"""
    <div class="wrap">
      <div class="card">
        <div class="card-title">{_esc(card_d.get("title","What this means"))}</div>
        <ul class="bullets">{d_li}</ul>
      </div>

      <div class="card">
        <div class="card-title">{_esc(card_e.get("title","Action plan — next 30–90 days"))}</div>
        <ul class="checks">{e_li}</ul>
      </div>
    </div>
    {_base_styles()}
    """
    st_html(html, height=height, scrolling=False)
