# pages/business_overview_component.py

from __future__ import annotations

from html import escape
from typing import Any, Dict, List

import streamlit as st
from streamlit.components.v1 import html as st_html


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


def _status_class(state: str) -> str:
    s = (state or "").strip().upper()
    if s in ("AT_RISK", "AT RISK", "RISK"):
        return "negative"
    if s in ("WATCH", "WARNING"):
        return "warning"
    return "success"


def _render_kpis(kpis: List[Dict[str, Any]]) -> str:
    icon_by_id = {
        "cash": "wallet",
        "burn": "flame",
        "runway": "hourglass",
        "revenue": "trending-up",
    }

    cards = []
    for k in (kpis or [])[:4]:
        kid = str(k.get("id", "")).strip().lower()
        icon = icon_by_id.get(kid, "")
        icon_html = f'<i data-lucide="{icon}" class="icon"></i>' if icon else ""

        label = escape(_s(k.get("label"), "—"))
        value = escape(_s(k.get("value"), "—"))
        note = _s(k.get("note"), "").strip()
        note_html = f'<div class="kpi-note">{escape(note)}</div>' if note else ""

        cards.append(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{icon_html}{label}</div>
              <div class="kpi-value">{value}</div>
              {note_html}
            </div>
            """
        )
    return "\n".join(cards)


def _render_bullets(bullets: List[str], fallback: str) -> str:
    items = [b.strip() for b in (bullets or []) if str(b).strip()]
    if not items:
        items = [fallback]
    lis = "\n".join([f"<li>{escape(i)}</li>" for i in items])
    return f"<ul class='bullets'>{lis}</ul>"


def _render_focus_actions(items: Any, fallback: str) -> str:
    """
    Accepts:
      - list[str]  -> renders bullets
      - list[dict] -> expects {"action": "...", "why": "..."} and renders action + why line under it
    """
    if not items:
        return f"<ul class='bullets'><li>{escape(fallback)}</li></ul>"

    # list of dicts with action/why
    if isinstance(items, list) and items and isinstance(items[0], dict):
        blocks = []
        for obj in items[:6]:
            action = str(obj.get("action") or "").strip()
            why = str(obj.get("why") or "").strip()
            if not action:
                continue

            why_html = f"<div class='why'>{escape(why)}</div>" if why else ""
            blocks.append(
                f"""
                <div class="focus-item">
                  <div class="focus-action">• {escape(action)}</div>
                  {why_html}
                </div>
                """
            )

        if not blocks:
            return f"<ul class='bullets'><li>{escape(fallback)}</li></ul>"

        return "<div class='focus-list'>" + "\n".join(blocks) + "</div>"

    # list of strings
    if isinstance(items, list):
        as_strings = [str(x).strip() for x in items if str(x).strip()]
        if not as_strings:
            as_strings = [fallback]
        lis = "\n".join([f"<li>{escape(i)}</li>" for i in as_strings[:6]])
        return f"<ul class='bullets'>{lis}</ul>"

    return f"<ul class='bullets'><li>{escape(fallback)}</li></ul>"


def render_business_overview(data: Dict[str, Any], *, height: int = 1100) -> None:
    """
    Presentation-only HTML renderer.
    IMPORTANT: Real charts must be rendered in Streamlit below this component.
    """

    header = data.get("header", {}) or {}
    exec_summary = data.get("exec_summary", {}) or {}
    kpis = data.get("kpis", []) or []
    month_summary = data.get("month_summary", {}) or {}
    focus_next = data.get("focus_next", {}) or {}

    # Header
    title = escape(_s(header.get("title"), "Business at a Glance"))
    subtitle = escape(_s(header.get("subtitle"), ""))

    # Exec summary
    bh = exec_summary.get("business_health", {}) or {}
    pf = exec_summary.get("primary_focus", {}) or {}

    bh_label = escape(_s(bh.get("label"), "Business Health"))
    bh_state = _s(bh.get("state"), "STABLE").upper()
    bh_message = escape(_s(bh.get("message"), "—"))

    bh_support = _s(bh.get("supporting_line"), "").strip()
    bh_support_html = f"<div class='support'>{escape(bh_support)}</div>" if bh_support else ""

    pf_label = escape(_s(pf.get("label"), "Primary Focus"))
    pf_headline = escape(_s(pf.get("headline"), "Maintain discipline — no action required"))

    # Month summary (bullets)
    month_title = escape(_s(month_summary.get("title"), "This month in 20 sec"))
    month_bullets = _lst(month_summary.get("bullets"))
    month_bullets_html = _render_bullets(
        month_bullets,
        fallback="No summary available for this month yet."
    )

    # Focus next (Action + Why)
    focus_title = escape(_s(focus_next.get("title"), "What to Focus on Next"))
    focus_items = focus_next.get("items")  # preferred: list[{"action","why"}]
    if focus_items is None:
        # backward compatible: maybe you still pass bullets as strings
        focus_items = focus_next.get("bullets")

    focus_html = _render_focus_actions(
        focus_items,
        fallback="Maintain focus — no defensible CFO-grade action triggered from this page’s KPIs."
    )

    kpis_html = _render_kpis(kpis)

    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/lucide@latest"></script>
<style>
:root {{
  --bg-app: #F8FAFC;
  --bg-card: #FFFFFF;
  --border: #E5E7EB;

  --text-h1: #0F172A;
  --text-h2: #1F2937;
  --text-body: #3A454B;
  --text-muted: #64748B;

  --success-bg: #ECFDF3;
  --warning-bg: #FFFBEB;
  --negative-bg: #FEF2F2;

  --radius-card: 12px;
}}

body {{
  margin: 0;
  background: var(--bg-app);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}

.main {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 32px 32px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}}

.icon {{
  width: 20px;
  height: 20px;
  color: #64748B;
  stroke-width: 1.8;
}}

.h1 {{
  display:flex;
  align-items:center;
  gap:10px;
  font-size: 20px;
  font-weight: 600;
  line-height: 28px;
  color: var(--text-h1);
}}

.subtitle {{
  font-size: 14px;
  line-height: 22px;
  color: var(--text-muted);
}}

.section-title {{
  display:flex;
  align-items:center;
  gap:10px;
  font-size: 16px;
  font-weight: 600;
  line-height: 24px;
  color: var(--text-h2);
}}

.label {{
  display:flex;
  align-items:center;
  gap:8px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 6px;
}}

.headline {{
  font-size: 16px;
  font-weight: 600;
  color: var(--text-h2);
}}

.support {{
  font-size: 13px;
  line-height: 20px;
  color: var(--text-muted);
  margin-top: 6px;
}}

.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 22px;
}}

.grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}

.grid-4 {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
}}

.kpi-card {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  background: #fff;
  min-height: 92px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}}

.kpi-label {{
  display:flex;
  align-items:center;
  gap:8px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 10px;
}}

.kpi-value {{
  font-size: 18px;
  font-weight: 600;
  color: var(--text-h1);
  line-height: 24px;
}}

.kpi-note {{
  margin-top: 10px;
  font-size: 12px;
  line-height: 20px;
  color: var(--text-muted);
}}

.status {{
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  margin-top: 6px;
  color: var(--text-h1);
}}

.success {{ background: var(--success-bg); }}
.warning {{ background: var(--warning-bg); }}
.negative {{ background: var(--negative-bg); }}

.bullets {{
  margin: 12px 0 0 0;
  padding-left: 18px;
  color: var(--text-body);
  font-size: 14px;
  line-height: 22px;
}}
.bullets li {{ margin: 10px 0; }}

/* Focus list (action + why) */
.focus-list {{
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.focus-action {{
  font-size: 14px;
  line-height: 22px;
  color: var(--text-body);
  font-weight: 500;
}}
.why {{
  margin-left: 18px;
  margin-top: 4px;
  font-size: 12px;
  line-height: 18px;
  color: var(--text-muted);
}}
</style>
</head>

<body>
<div class="main">

  <!-- HEADER -->
  <div>
    <div class="h1">
      <i data-lucide="heart-pulse" class="icon"></i>
      {title}
    </div>
    <div class="subtitle">{subtitle}</div>
  </div>

  <!-- EXEC SUMMARY -->
  <div class="card grid-2">
    <div>
      <div class="label">
        <i data-lucide="heart-pulse" class="icon"></i>
        {bh_label}
      </div>
      <div class="status {_status_class(bh_state)}">
        {escape(bh_state.replace("_"," "))} — {bh_message}
      </div>
      {bh_support_html}
    </div>

    <div>
      <div class="label">
        <i data-lucide="target" class="icon"></i>
        {pf_label}
      </div>
      <div class="headline">{pf_headline}</div>
    </div>
  </div>

  <!-- KPI ROW -->
  <div class="grid-4">
    {kpis_html}
  </div>

  <!-- THIS MONTH (TITLE OUTSIDE CARD) -->
  <div>
    <div class="section-title">
      <i data-lucide="timer" class="icon"></i>
      {month_title}
    </div>
    <div class="card">
      {month_bullets_html}
    </div>
  </div>

  <!-- FOCUS NEXT (TITLE OUTSIDE CARD) -->
  <div>
    <div class="section-title">
      <i data-lucide="circle-check-big" class="icon"></i>
      {focus_title}
    </div>
    <div class="card">
      {focus_html}
    </div>
  </div>

</div>

<script>
  lucide.createIcons();
</script>
</body>
</html>
    """

    st_html(html_doc, height=height, scrolling=False)
