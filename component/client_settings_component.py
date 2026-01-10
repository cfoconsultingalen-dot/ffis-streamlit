# pages/client_settings_component.py
import streamlit as st
from streamlit.components.v1 import html as st_html

def inject_client_settings_css() -> None:
    """
    Nordic styling for Client Settings.
    Safe to call multiple times.
    """
    st.markdown(
        """
        <style>
        /* ---------- Page canvas ---------- */
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        section.main,
        section.main > div {
          background: #F8FAFC !important;
        }

        /* ---------- Main content width ---------- */
        div.block-container {
          max-width: 1100px !important;
          padding-top: 24px !important;
          padding-bottom: 48px !important;
        }

        /* ---------- Global typography ---------- */
        .stApp, .stApp * {
          font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
        }

        /* ---------- Header (match your nm-h1 / nm-sub spec) ---------- */
        .nm-page-header {
          max-width: 1100px;
          margin: 0 auto 24px auto;
        }
        .nm-h1 {
          font-size: 20px;
          line-height: 28px;
          font-weight: 600;
          color: #0F172A;
          margin: 0 0 6px 0;
        }
        .nm-sub {
          font-size: 14px;
          line-height: 22px;
          font-weight: 400;
          color: #64748B;
          margin: 0;
        }

        /* ---------- Nordic card shell (Streamlit bordered containers) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nordic_anchor) {
          background: #FFFFFF !important;
          border: 1px solid #E5E7EB !important;
          border-radius: 12px !important;
          padding: 0 !important;
          margin: 0 0 24px 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nordic_anchor) > div {
          padding: 20px !important;
        }

        .nordic-card-title {
          font-size: 14px;
          line-height: 22px;
          font-weight: 600;
          color: #0F172A;
          margin: 0 0 8px 0;
        }
        .nordic-card-sub {
          font-size: 12px;
          line-height: 18px;
          font-weight: 400;
          color: #64748B;
          margin: 0 0 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_client_settings_header(data: dict) -> None:
    title = (data or {}).get("title", "Client Settings")
    subtitle = (data or {}).get("subtitle", "")

    st.markdown(
        f"""
        <div class="nm-page-header">
          <div class="nm-h1">{title}</div>
          <div class="nm-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
