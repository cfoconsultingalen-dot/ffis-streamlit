# pages/configuration_component.py
import streamlit as st

def inject_configuration_css() -> None:
    st.markdown(
        """
        <style>
        /* Global typography */
        .stApp, .stApp *{
          font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
        }

        /* Page canvas */
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        section.main,
        section.main > div{
          background: #F8FAFC !important;
        }

        /* Main width */
        div.block-container{
          max-width: 1100px !important;
          padding-top: 24px !important;
          padding-bottom: 48px !important;
        }

        /* Headings hierarchy */
        .nm-h1{
          font-size: 20px;
          line-height: 28px;
          font-weight: 600;
          color: #0F172A;
          margin: 0 0 6px 0;
        }
        .nm-sub{
          font-size: 14px;
          line-height: 22px;
          font-weight: 400;
          color: #64748B;
          margin: 0 0 18px 0;
        }

        /* Card title + subtitle */
        .nordic-card-title{
          font-size: 14px;
          line-height: 22px;
          font-weight: 600;
          color: #0F172A;
          margin: 0 0 8px 0;
        }
        .nordic-card-sub{
          font-size: 12px;
          line-height: 18px;
          color: #64748B;
          margin: 0 0 12px 0;
        }

        /* Streamlit bordered container → Nordic card */
        div[data-testid="stVerticalBlockBorderWrapper"]{
          background: #FFFFFF !important;
          border: 1px solid #E5E7EB !important;
          border-radius: 12px !important;
          padding: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div{
          padding: 20px !important;
        }

        /* Keep text calm */
        .stMarkdown, .stText, label, p, li { color: #3A454B; }

        /* =========================================
           EXPANDER FIX (guaranteed)
           Removes "keyboard_arrow_right/down" text
        ========================================= */

        /* Hide Streamlit’s toggle icon span that becomes text */
        div[data-testid="stExpander"] span[data-testid="stExpanderToggleIcon"]{
          display: none !important;
        }

        /* Remove default marker */
        div[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
        div[data-testid="stExpander"] summary { list-style: none !important; }

        /* Create our own arrow */
        div[data-testid="stExpander"] details > summary{
          position: relative !important;
          padding-left: 18px !important;
        }

        /* Closed arrow */
        div[data-testid="stExpander"] details > summary::before{
          content: "▸";
          position: absolute;
          left: 0;
          top: 50%;
          transform: translateY(-50%);
          font-size: 14px;
          line-height: 1;
          color: #64748B;
        }

        /* Open arrow rotates */
        div[data-testid="stExpander"] details[open] > summary::before{
          transform: translateY(-50%) rotate(90deg);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )



def render_configuration_header(*, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='nm-h1'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='nm-sub'>{subtitle}</div>", unsafe_allow_html=True)


def card_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='nordic-card-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='nordic-card-sub'>{subtitle}</div>", unsafe_allow_html=True)
