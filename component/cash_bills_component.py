# pages/cash_bills_component.py
import streamlit as st
from streamlit.components.v1 import html as st_html

def render_cash_bills(data: dict, height: int = 3000):
    """
    Cash & Bills component renderer (Nordic-minimalist).

    Updates in this version:
    ✅ Card E: table full width, ~4 rows visible + scroll
    ✅ Card E: chart moved BELOW table (full width)
    ✅ Card E: Plotly width/height constrained (no overflow, axes visible)
    """

    # ---------------------------
    # helpers
    # ---------------------------
    def esc(x):
        if x is None:
            return ""
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def join_items(items):
        if not items:
            return ""
        return "\n".join(items)

    # ---------------------------
    # unpack
    # ---------------------------
    header = data.get("header", {})
    cards = data.get("cards", {})

    # Card A
    card_a = cards.get("cash_danger_14w", {})
    a_kpis = card_a.get("kpis", [])
    a_status = card_a.get("status_strip", {}) or {}
    a_state = a_status.get("state", "success")
    a_msg = a_status.get("message", "")

    # Card B
    card_b = cards.get("short_term_liquidity_4w", {})
    b_kpis = card_b.get("kpis", [])
    b_helper = card_b.get("helper_text", "")

    # Card C
    card_c = cards.get("ar_ap_ageing", {})
    ar = card_c.get("ar", {}) or {}
    ap = card_c.get("ap", {}) or {}
    ar_link = ar.get("drilldown_link")
    ap_link = ap.get("drilldown_link")

    # Card D
    card_d = cards.get("commitments", {})
    comm_rows = card_d.get("rows", []) or []

    # Card E
    card_e = cards.get("cashflow_14w", {})
    e_plotly_html = (card_e.get("chart", {}) or {}).get("plotly_html", "")
    e_empty = (card_e.get("chart", {}) or {}).get("empty_text", "No data.")
    e_table_cols = card_e.get("table_columns", []) or []
    e_table_rows = card_e.get("table_rows", []) or []
    e_caption = card_e.get("caption", "")

    # Card F
    card_f = cards.get("what_this_means", {})
    f_bullets = card_f.get("bullets", []) or []
    f_fallback = card_f.get("fallback", "No narrative available yet.")

    # Card G
    card_g = cards.get("action_plan", {})
    g_items = card_g.get("items", []) or []
    g_fallback = card_g.get("fallback", "No actions triggered.")

    # ---------------------------
    # status class mapping
    # ---------------------------
    state_class = {
        "success": "status success",
        "warning": "status warning",
        "risk": "status risk",
        "negative": "status risk",
    }.get(str(a_state).lower(), "status success")

    # ---------------------------
    # render pieces
    # ---------------------------
    def kpi_cell(k):
        return f"""
        <div class="kpi">
          <div class="kpi-label">{esc(k.get("label",""))}</div>
          <div class="kpi-value">{esc(k.get("value","—"))}</div>
        </div>
        """

    def mini_kpi(k):
        return f"""
        <div class="mini">
          <div class="mini-label">{esc(k.get("label",""))}</div>
          <div class="mini-value">{esc(k.get("value","—"))}</div>
        </div>
        """

    def simple_table(columns, rows):
        th = "".join([f"<th>{esc(c)}</th>" for c in columns])
        trs = []
        for r in rows:
            if isinstance(r, dict):
                tds = "".join([f"<td>{esc(r.get(c,''))}</td>" for c in columns])
            else:
                tds = "".join([f"<td>{esc(x)}</td>" for x in r])
            trs.append(f"<tr>{tds}</tr>")
        body = "\n".join(trs) if trs else f"<tr><td colspan='{max(1,len(columns))}' class='empty'>—</td></tr>"
        return f"""
        <table class="tbl">
          <thead><tr>{th}</tr></thead>
          <tbody>{body}</tbody>
        </table>
        """

    # tables
    ar_cols = ar.get("columns", ["Bucket", "Total amount"])
    ap_cols = ap.get("columns", ["Bucket", "Total amount"])

    comm_cols = card_d.get("columns", ["Direction", "Who/What", "Amount", "Expected date"])
    comm_rows_fmt = comm_rows

    # Card E table
    e_tbl = simple_table(e_table_cols, e_table_rows) if e_table_cols else ""

    # What this means bullets
    if not f_bullets:
        f_bullets = [f_fallback]
    f_li = "\n".join([f"<li>{esc(b)}</li>" for b in f_bullets[:4]])

    # Action plan bullets
    if not g_items:
        g_items = [g_fallback]
    g_li = "\n".join([f"<li>{esc(a)}</li>" for a in g_items[:5]])

    # ---------------------------
    # full html
    # ---------------------------
    html = f"""
    <div class="wrap">
      <div class="header">
        <div class="h1">{esc(header.get("title","Cash & Bills"))}</div>
        <div class="meta">{esc(header.get("subtitle",""))}</div>
      </div>

      <!-- CARD A -->
      <div class="card">
        <div class="card-title">{esc(card_a.get("title","Cash danger summary (next 14 weeks)"))}</div>
        <div class="kpi-grid kpi-6">
          {join_items([kpi_cell(k) for k in a_kpis])}
        </div>
        <div class="{state_class}">{esc(a_msg)}</div>
      </div>

      <!-- CARD B -->
      <div class="card">
        <div class="card-title">{esc(card_b.get("title","Short-term liquidity (next 4 weeks)"))}</div>
        <div class="mini-grid">
          {join_items([mini_kpi(k) for k in b_kpis])}
        </div>
        <div class="helper">{esc(b_helper)}</div>
      </div>

      <!-- CARD C -->
      <div class="card">
        <div class="card-title">{esc(card_c.get("title","AR & AP ageing"))}</div>
        <div class="split-2">
          <div>
            <div class="sub">{esc(ar.get("subtitle","Invoices coming in (AR)"))}</div>
            {simple_table(ar_cols, ar.get("rows", []) or [])}
            {"<a class='drill' href='" + esc(ar_link.get("href","")) + "'>" + esc(ar_link.get("label","")) + "</a>"
            if isinstance(ar_link, dict) and ar_link.get("href") else ""}
            <div class="hint">{esc(ar.get("hint",""))}</div>
          </div>
          <div>
            <div class="sub">{esc(ap.get("subtitle","Bills to pay (AP)"))}</div>
            {simple_table(ap_cols, ap.get("rows", []) or [])}
            {"<a class='drill' href='" + esc(ap_link.get("href","")) + "'>" + esc(ap_link.get("label","")) + "</a>"
            if isinstance(ap_link, dict) and ap_link.get("href") else ""}
            <div class="hint">{esc(ap.get("hint",""))}</div>
          </div>
        </div>
      </div>

      <!-- CARD D -->
      <div class="card">
        <div class="card-title">{esc(card_d.get("title","Cash commitments coming up"))}</div>
        {simple_table(comm_cols, comm_rows_fmt)}
        <div class="hint">{esc(card_d.get("caption",""))}</div>
      </div>

      <!-- CARD E -->
      <div class="card">
        <div class="card-title">{esc(card_e.get("title","14-week cashflow (early warning)"))}</div>

        <!-- TABLE (full width, scroll after ~4 rows) -->
        <div class="table-scroll full">
          {e_tbl if e_tbl else f"<div class='empty'>{esc(e_empty)}</div>"}
        </div>

        <!-- CHART (below table, full width, contained) -->
        <div class="chartbox full">
          <div class="chart-title">
            {esc((card_e.get("chart", {}) or {}).get("title","Weekly projected closing cash balance"))}
          </div>
          <div class="chart-inner">
            {e_plotly_html if e_plotly_html else f"<div class='empty'>{esc(e_empty)}</div>"}
          </div>
        </div>

        <div class="hint">{esc(e_caption)}</div>
      </div>

      <!-- CARD F -->
      <div class="card">
        <div class="card-title">{esc(card_f.get("title","What this means"))}</div>
        <ul class="bullets">{f_li}</ul>
      </div>

      <!-- CARD G -->
      <div class="card">
        <div class="card-title">{esc(card_g.get("title","Action plan – next 7–14 days"))}</div>
        <ul class="checks">{g_li}</ul>
      </div>

    </div>

    <style>
      .wrap {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 0 48px 0;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: #F8FAFC;
      }}
      .header {{
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 18px;
      }}
      .h1 {{
        font-size: 20px;
        line-height: 28px;
        font-weight: 600;
        color: #0F172A;
      }}
      .meta {{
        font-size: 14px;
        line-height: 22px;
        font-weight: 400;
        color: #64748B;
      }}
      .card {{
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 32px;
      }}
      .card-title {{
        font-size: 14px;
        line-height: 22px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 12px;
      }}
      .sub {{
        font-size: 13px;
        line-height: 20px;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 10px;
      }}

      .kpi-grid {{
        display: grid;
        gap: 16px;
        margin-bottom: 14px;
      }}
      .kpi-6 {{ grid-template-columns: repeat(6, 1fr); }}
      .kpi {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .kpi-label {{
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
        font-weight: 400;
      }}
      .kpi-value {{
        font-size: 18px;
        line-height: 24px;
        color: #0F172A;
        font-weight: 600;
        white-space: nowrap;
      }}

      .mini-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 10px;
      }}
      .mini {{
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 14px 14px;
        background: #FFFFFF;
      }}
      .mini-label {{
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 6px;
      }}
      .mini-value {{
        font-size: 16px;
        line-height: 22px;
        color: #0F172A;
        font-weight: 600;
      }}

      .helper {{
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
      }}
      .hint {{
        margin-top: 10px;
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
      }}

      .status {{
        width: 100%;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 20px;
        font-weight: 500;
      }}
      .success {{ background: #ECFDF3; color: #166534; border: 1px solid #BBF7D0; }}
      .warning {{ background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }}
      .risk {{ background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }}

      .split-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
      }}

      .tbl {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        line-height: 18px;
        color: #0F172A;
      }}
      .tbl th {{
        text-align: left;
        padding: 10px 10px;
        border-bottom: 1px solid #E5E7EB;
        color: #64748B;
        font-weight: 600;
        font-size: 12px;
        background: #FFFFFF;
      }}
      .tbl td {{
        padding: 10px 10px;
        border-bottom: 1px solid #F1F5F9;
        vertical-align: top;
      }}

      .empty {{
        color: #64748B;
        padding: 16px;
        border: 1px dashed #E5E7EB;
        border-radius: 12px;
      }}

      /* ✅ Card E: table full width with scroll */
      .table-scroll {{
        max-height: 320px;   /* ~4 rows */
        overflow: auto;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
      }}
      .table-scroll.full {{
        width: 100%;
      }}
      .table-scroll .tbl {{
        margin: 0;
      }}

      /* ✅ Card E: chart below table, full width, no spill */
      .chartbox {{
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 12px;
        background: #FFFFFF;
        margin-top: 14px;
      }}
      .chartbox.full {{
        width: 100%;
        overflow: hidden;
      }}

      .chart-title {{
        font-size: 12px;
        line-height: 18px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 8px;
      }}

      .chart-inner {{
        width: 100%;
        overflow: hidden;
        min-height: 200px; /* gives axes room */
      }}

      /* Hard containment for Plotly */
      .chartbox .plotly-graph-div {{
        width: 100% !important;
        max-width: 100% !important;
        height: 280px !important;
      }}
      .chartbox svg,
      .chartbox canvas {{
        max-width: 100% !important;
      }}

      .bullets, .checks {{
        margin: 0;
        padding-left: 18px;
        color: #3A454B;
        font-size: 13px;
        line-height: 22px;
      }}
      .bullets li, .checks li {{
        margin: 8px 0;
      }}

      .drill {{
        display: inline-block;
        margin-top: 10px;
        font-size: 12px;
        line-height: 18px;
        color: #0F172A;
        text-decoration: none;
        font-weight: 600;
      }}
      .drill:hover {{
        text-decoration: underline;
      }}

      @media (max-width: 1100px) {{
        .wrap {{ padding-left: 12px; padding-right: 12px; }}
      }}
      @media (max-width: 900px) {{
        .kpi-6 {{ grid-template-columns: repeat(2, 1fr); }}
        .mini-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .split-2 {{ grid-template-columns: 1fr; }}
      }}
    </style>
    """

    st_html(html, height=height, scrolling=False)
