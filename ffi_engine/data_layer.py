from typing import List, Dict, Any

from .config import get_supabase_client


def fetch_clients() -> List[Dict[str, Any]]:
    """
    Return a list of client rows from the 'clients' table.
    """
    supabase = get_supabase_client()
    response = supabase.table("clients").select("*").order("created_at").execute()
    # response.data is a list of dicts
    return response.data or []


def fetch_client_names() -> List[str]:
    """
    Convenience helper: returns just the client names.
    """
    rows = fetch_clients()
    if not rows:
        return []
    return [row.get("name", "Unnamed client") for row in rows]
