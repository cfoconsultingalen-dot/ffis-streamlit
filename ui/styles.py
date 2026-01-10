# ui/styles.py
import streamlit as st

def inject_global_nordic_styles():
    st.markdown(
        """
        <style>
        /* ===============================
           Page header (GLOBAL)
           =============================== */

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

        /* ===============================
           Card titles (GLOBAL)
           =============================== */

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
