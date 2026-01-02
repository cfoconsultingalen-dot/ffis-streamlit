# ui/styles.py
import streamlit as st
from ui.theme import COLORS, TYPE, LAYOUT

def inject_global_css() -> None:
    """
    Injects Nordic minimalist base styling:
    - App background
    - Max width centered container
    - Card styling
    - Typography classes
    - KPI styling
    - Status strips
    - Simple tables
    """
    st.markdown(
        f"""
<style>
/* ---------- App base ---------- */
.stApp {{
  background: {COLORS.app_bg};
  font-family: {TYPE.font_family};
}}

/* ---------- Centered max-width container ---------- */
.ui-wrap {{
  max-width: {LAYOUT.max_width_px}px;
  margin: 0 auto;
  padding: {LAYOUT.page_pad_top_px}px 0 {LAYOUT.page_pad_bottom_px}px 0;
}}

/* ---------- Cards ---------- */
.ui-card {{
  background: {COLORS.card_bg};
  border: 1px solid {COLORS.border};
  border-radius: {LAYOUT.card_radius_px}px;
  padding: {LAYOUT.card_pad_px}px;
  margin-bottom: {LAYOUT.section_gap_px}px;
}}

/* Smaller nested card (e.g., Funding Details) */
.ui-card-nested {{
  background: {COLORS.card_bg};
  border: 1px solid {COLORS.border};
  border-radius: 10px;
  padding: 16px;
}}

/* ---------- Typography ---------- */
.ui-h1 {{
  margin: 0;
  font-size: {TYPE.h1_size}px;
  line-height: {TYPE.h1_lh}px;
  font-weight: {TYPE.h1_weight};
  letter-spacing: {TYPE.h1_letter_spacing};
  color: {COLORS.text_h1};
}}

.ui-h2 {{
  margin: 0 0 14px 0;
  font-size: {TYPE.h2_size}px;
  line-height: {TYPE.h2_lh}px;
  font-weight: {TYPE.h2_weight};
  letter-spacing: {TYPE.h2_letter_spacing};
  color: {COLORS.text_h2};
  display: flex;
  align-items: center;
  gap: 8px;
}}

.ui-h3 {{
  margin: 0 0 8px 0;
  font-size: {TYPE.h3_size}px;
  line-height: {TYPE.h3_lh}px;
  font-weight: {TYPE.h3_weight};
  letter-spacing: {TYPE.h3_letter_spacing};
  color: {COLORS.text_h3};
}}

.ui-body {{
  margin: 0;
  font-size: {TYPE.body_size}px;
  line-height: {TYPE.body_lh}px;
  font-weight: {TYPE.body_weight};
  letter-spacing: {TYPE.body_letter_spacing};
  color: {COLORS.text_body};
}}

.ui-muted {{
  margin: 0;
  font-size: {TYPE.body_size}px;
  line-height: {TYPE.body_lh}px;
  font-weight: {TYPE.body_weight};
  color: {COLORS.text_muted};
}}

.ui-quote {{
  margin-top: 6px;
  margin-bottom: 0;
  font-size: {TYPE.quote_size}px;
  line-height: {TYPE.quote_lh}px;
  font-weight: {TYPE.quote_weight};
  letter-spacing: {TYPE.quote_letter_spacing};
  color: {COLORS.text_muted};
}}

/* ---------- KPI ---------- */
.ui-kpi-label {{
  margin: 0;
  font-size: {TYPE.kpi_label_size}px;
  line-height: {TYPE.kpi_label_lh}px;
  font-weight: {TYPE.kpi_label_weight};
  letter-spacing: {TYPE.kpi_label_letter_spacing};
  color: {COLORS.text_muted};
}}

.ui-kpi-value {{
  margin: 2px 0 0 0;
  font-size: {TYPE.kpi_value_size}px;
  line-height: {TYPE.kpi_value_lh}px;
  font-weight: {TYPE.kpi_value_weight};
  letter-spacing: {TYPE.kpi_value_letter_spacing};
  color: {COLORS.text_kpi_value};
}}

/* ---------- Status strips ---------- */
.ui-status {{
  margin-top: 14px;
  border-radius: 8px;
  padding: 10px 12px;
  font-size: {TYPE.kpi_label_size}px;
  line-height: {TYPE.kpi_label_lh}px;
  font-weight: {TYPE.kpi_label_weight};
}}

.ui-status-success {{
  background: {COLORS.status_success_bg};
  color: {COLORS.status_success_text};
}}

.ui-status-warning {{
  background: {COLORS.status_warning_bg};
  color: {COLORS.status_warning_text};
}}

.ui-status-negative {{
  background: {COLORS.status_negative_bg};
  color: {COLORS.status_negative_text};
}}

/* ---------- Helper line ---------- */
.ui-helper {{
  margin-top: 10px;
  color: {COLORS.text_muted};
  font-size: {TYPE.kpi_label_size}px;
  line-height: {TYPE.kpi_label_lh}px;
}}

/* ---------- Tables (simple, Nordic) ---------- */
.ui-table {{
  width: 100%;
  border-collapse: collapse;
}}

.ui-table th {{
  text-align: left;
  padding: 10px 8px;
  border-bottom: 1px solid {COLORS.border};
  font-size: {TYPE.kpi_label_size}px;
  line-height: {TYPE.kpi_label_lh}px;
  font-weight: {TYPE.kpi_label_weight};
  color: {COLORS.text_h3};
}}

.ui-table td {{
  padding: 10px 8px;
  border-bottom: 1px solid {COLORS.border};
  font-size: {TYPE.body_size}px;
  line-height: 20px;
  font-weight: {TYPE.body_weight};
  color: {COLORS.text_body};
}}

.ui-right {{
  text-align: right;
}}

/* ---------- Bullet blocks ---------- */
.ui-bullets ul {{
  margin: 0;
  padding-left: 18px;
}}
.ui-bullets li {{
  margin: 8px 0;
}}

/* ---------- Soft callouts (used in risk/compliance) ---------- */
.ui-callout {{
  border-radius: 10px;
  padding: 10px 12px;
  font-size: {TYPE.body_size}px;
  line-height: {TYPE.body_lh}px;
  color: {COLORS.text_body};
  border: 1px solid {COLORS.border};
}}
.ui-callout-warning {{
  background: {COLORS.status_warning_bg};
}}
.ui-callout-negative {{
  background: {COLORS.status_negative_bg};
}}

</style>
        """,
        unsafe_allow_html=True,
    )
