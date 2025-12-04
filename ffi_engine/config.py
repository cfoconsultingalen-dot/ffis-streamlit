import os
from typing import Optional

from supabase import create_client, Client

# Try to import streamlit if available (for st.secrets on Cloud)
try:
    import streamlit as st
except ImportError:
    st = None  # when running plain Python scripts


_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Returns a singleton Supabase client.
    Works both:
    - locally (reads from .env)
    - on Streamlit Cloud (reads from st.secrets)
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    # 1) Streamlit Cloud: use st.secrets
    url = None
    key = None

    if st is not None and hasattr(st, "secrets") and "SUPABASE_URL" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    else:
        # 2) Local: read from environment variables (which come from .env)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Supabase URL/KEY not found. Check .env or Streamlit secrets.")

    _supabase_client = create_client(url, key)
    return _supabase_client
