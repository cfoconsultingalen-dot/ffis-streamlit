# ui/components.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import streamlit as st

from ui.theme import COLORS, TYPE, LAYOUT
from ui.styles import inject_global_css


# ---------------------------
# Page wrapper
# ---------------------------
def start_page_container() -> None:
    """Call at the beginning of each page."""
    inject_global_css()
    st.markdown('<div class="ui-wrap">', unsafe_allow_html=True)


def end_page_container() -> None:
    """Call at the end of each page."""
    st.markdown("</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(f'<h1 class="ui-h1">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="ui-quote">{subtitle}</p>', unsafe_allow_html=True)


# ---------------------------
# Card context manager
# ---------------------------
@contextmanager
def card(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None):
    icon_html = f"{icon} " if icon else ""
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="ui-h2">{icon_html}{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="ui-muted">{subtitle}</p>', unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


@contextmanager
def nested_card(title: Optional[str] = None):
    st.markdown('<div class="ui-card-nested">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="ui-h3">{title}</div>', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# KPI helpers
# ---------------------------
def kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <p class="ui-kpi-label">{label}</p>
        <p class="ui-kpi-value">{value}</p>
        """,
        unsafe_allow_html=True,
    )


def kpi_grid(items: Sequence[Tuple[str, str]], cols: int, gap: str = "small") -> None:
    """
    items: list of (label, value)
    cols: number of columns on the row
    """
    columns = st.columns(cols, gap=gap)
    for i, (label, value) in enumerate(items):
        with columns[i % cols]:
            kpi(label, value)


# ---------------------------
# Status + helper
# ---------------------------
def status_strip(text: str, kind: str = "success") -> None:
    """
    kind: success | warning | negative
    """
    kind = kind.lower().strip()
    cls = {
        "success": "ui-status ui-status-success",
        "warning": "ui-status ui-status-warning",
        "negative": "ui-status ui-status-negative",
    }.get(kind, "ui-status ui-status-success")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def helper_text(text: str) -> None:
    st.markdown(f'<div class="ui-helper">{text}</div>', unsafe_allow_html=True)


# ---------------------------
# Bullets + checklists
# ---------------------------
def bullet_box(title: str, bullets: Sequence[str], icon: Optional[str] = None) -> None:
    with card(title=title, icon=icon):
        bullets_html = "".join([f"<li>{b}</li>" for b in bullets])
        st.markdown(f'<div class="ui-bullets"><ul>{bullets_html}</ul></div>', unsafe_allow_html=True)


def checklist_box(title: str, items: Sequence[str], key_prefix: str, icon: Optional[str] = None) -> None:
    with card(title=title, icon=icon):
        for i, t in enumerate(items):
            st.checkbox(t, key=f"{key_prefix}_{i}")


# ---------------------------
# Simple table renderer
# ---------------------------
def table_simple(
    df: pd.DataFrame,
    right_align_cols: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> None:
    """
    Renders a simple Nordic HTML table.
    right_align_cols: list of column names to right-align (e.g. numeric columns).
    """
    if df is None or df.empty:
        st.info("No data available.")
        return

    if max_rows is not None:
        df = df.head(max_rows)

    right_align_cols = set(right_align_cols or [])
    cols = list(df.columns)

    # Header
    ths = "".join([f"<th>{c}</th>" for c in cols])

    # Rows
    trs = []
    for _, row in df.iterrows():
        tds = []
        for c in cols:
            val = row[c]
            cls = "ui-right" if c in right_align_cols else ""
            tds.append(f'<td class="{cls}">{val}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")

    html = f"""
    <table class="ui-table">
      <thead><tr>{ths}</tr></thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------
# Callouts (for risk/compliance)
# ---------------------------
def callout(text: str, kind: str = "warning") -> None:
    kind = kind.lower().strip()
    cls = {
        "warning": "ui-callout ui-callout-warning",
        "negative": "ui-callout ui-callout-negative",
        "neutral": "ui-callout",
    }.get(kind, "ui-callout ui-callout-warning")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
