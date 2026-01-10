# pages/collaboration_hub_component.py
import streamlit as st
from streamlit.components.v1 import html as st_html

def inject_collaboration_hub_css():
    st.markdown(
        """
        <style>
          /* ---------- Page background (real Streamlit wrappers) ---------- */
          div[data-testid="stAppViewContainer"],
          div[data-testid="stMain"],
          section.main,
          section.main > div {
            background: #F8FAFC !important;
          }

          /* ---------- Global text typography (DO NOT use .stApp *) ---------- */
          html, body, .stApp {
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          }

          /* Apply Inter only to common text elements */
          p, li, label, small, strong, em,
          .stMarkdown, .stText, .stCaption,
          input, textarea, button, select,
          [data-testid="stTextInput"], [data-testid="stTextArea"],
          [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
          [data-testid="stDateInput"], [data-testid="stButton"] {
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          }

          /* Base text calm */
          .stMarkdown, .stText, label, p, li { color: #3A454B !important; }
          .stCaption, small { color: #64748B !important; }

          /* Reduce top padding slightly */
          div.block-container {
            max-width: 1100px !important;
            padding-top: 24px !important;
            padding-bottom: 48px !important;
          }

          /* ---------- CRITICAL: Restore icon font everywhere ---------- */
          /* Streamlit uses Material Symbols ligatures for arrows/chevrons */
          span.material-symbols-outlined,
          span.material-symbols-rounded,
          span.material-symbols-sharp,
          i.material-icons,
          i.material-icons-round,
          i.material-icons-outlined,
          i.material-icons-two-tone,
          [class*="material-symbols"] {
            font-family: "Material Symbols Outlined" !important;
            font-weight: 400 !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            line-height: 1 !important;
            -webkit-font-smoothing: antialiased !important;
          }

          /* ---------- Nordic card wrapper: ONLY bordered containers with the anchor ---------- */
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.nordic_anchor) {
            max-width: 1100px !important;
            margin: 0 auto 32px auto !important;
            background: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 12px !important;
          }
          div[data-testid="stVerticalBlockBorderWrapper"]:has(.nordic_anchor) > div {
            padding: 20px !important;
          }

          /* Card title/subtitle */
          .nordic-card-title {
            font-size: 14px !important;
            line-height: 22px !important;
            font-weight: 600 !important;
            color: #0F172A !important;
            margin: 0 0 8px 0 !important;
          }
          .nordic-card-sub {
            font-size: 12px !important;
            line-height: 18px !important;
            color: #64748B !important;
            margin: 0 0 12px 0 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )




def render_collaboration_hub_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="nm-page-header">
          <div class="nm-h1">{title}</div>
          <div class="nm-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

