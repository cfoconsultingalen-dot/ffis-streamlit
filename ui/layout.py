# ui/layout.py
import streamlit as st

def apply_global_layout():
    st.markdown(
        """
        <style>
        /* Expand usable width */
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }

        /* Tighten KPI cards */
        div[data-testid="stMetric"] {
            padding: 0.75rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
