# ui/streamlit_cards.py
import streamlit as st

def inject_nordic_card_css() -> None:
    st.markdown(
        """
<style>
/* --- Nordic tokens (match your HTML component) --- */
:root{
  --bg-app:#F8FAFC;
  --bg-card:#FFFFFF;
  --border:#E5E7EB;
  --text-h1:#0F172A;
  --text-h2:#1F2937;
  --text-body:#3A454B;
  --text-muted:#64748B;
  --radius:12px;
}

/* Wrapper that mimics HTML .card */
.nordic-card{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}

/* Title + subtitle styles */
.nordic-card-title{
  font-size:16px;
  font-weight:600;
  line-height:24px;
  color: var(--text-h2);
  margin: 0;
  display:flex;
  align-items:center;
  gap:10px;
}

.nordic-card-subtitle{
  margin-top:6px;
  font-size:14px;
  line-height:22px;
  color: var(--text-muted);
}

/* KPI label + value alignment (Streamlit metric tweaks) */
.nordic-metric-row [data-testid="stMetricLabel"]{
  font-size:12px !important;
  font-weight:500 !important;
  color: var(--text-muted) !important;
}

.nordic-metric-row [data-testid="stMetricValue"]{
  font-size:18px !important;
  font-weight:600 !important;
  color: var(--text-h1) !important;
  line-height: 26px !important;
}

/* Keep chart from looking “detached” */
.nordic-chart-title{
  margin-top: 10px;
  font-size:14px;
  font-weight:600;
  color: var(--text-h2);
}

.nordic-chart-shell{
  margin-top:12px;
  border:1px solid var(--border);
  border-radius:10px;
  background:#FFFFFF;
  padding:12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )
