import streamlit as st
from typing import List,Dict
from datetime import date, datetime, timedelta
import pandas as pd
from supabase import create_client
import altair as alt
import calendar
import os
import time
import numpy as np
import io
import matplotlib.pyplot as plt

#import Fundtion for Board Reportinf
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

# ---------- Supabase config ----------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase_client()

st.set_page_config(page_title="FFIS – Founder Financial Intelligence", layout="wide")


def fetch_scenarios_for_client(client_id: str):
    """Return list of saved scenarios for this client, newest first."""
    try:
        resp = (
            supabase.table("scenarios")
            .select("*")
            .eq("client_id", str(client_id))
            .order("created_at", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print("Error fetching scenarios:", e)
        return []


def save_scenario_for_client(
    client_id: str,
    name: str,
    controls: dict,
    base_month: date | None = None,
    description: str | None = None,
):
    """Insert a new named scenario for this client."""
    try:
        payload = {
            "client_id": str(client_id),
            "name": name.strip(),
            "controls": controls,
            "base_month": base_month.isoformat() if base_month else None,
            "description": (description or "").strip() or None,
        }
        supabase.table("scenarios").insert(payload).execute()
        return True
    except Exception as e:
        print("Error saving scenario:", e)
        return False


def fetch_scenario_by_id(scenario_id: int):
    """Get a single scenario by numeric ID."""
    try:
        resp = (
            supabase.table("scenarios")
            .select("*")
            .eq("id", scenario_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception as e:
        print("Error fetching scenario by id:", e)
        return None


def delete_scenario_by_id(scenario_id: int) -> bool:
    """Optional: allow deleting a saved scenario from the UI."""
    try:
        supabase.table("scenarios").delete().eq("id", scenario_id).execute()
        return True
    except Exception as e:
        print("Error deleting scenario:", e)
        return False


def fetch_opex_for_client(client_id) -> pd.DataFrame:
    supabase = get_supabase_client()
    resp = (
        supabase.table("opex_entries")
        .select("*")
        .eq("client_id", str(client_id))
        .execute()
    )
    data = resp.data or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df["cash_out"] = pd.to_numeric(df.get("cash_out"), errors="coerce").fillna(0.0)
    df["cash_in"] = pd.to_numeric(df.get("cash_in"), errors="coerce").fillna(0.0)
    return df


def fetch_operating_other_income_for_client(client_id) -> pd.DataFrame:
    supabase = get_supabase_client()
    resp = (
        supabase.table("operating_other_income")
        .select("*")
        .eq("client_id", str(client_id))
        .execute()
    )
    data = resp.data or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df["cash_in"] = pd.to_numeric(df.get("cash_in"), errors="coerce").fillna(0.0)
    return df


def fetch_investing_flows_for_client(client_id) -> pd.DataFrame:
    supabase = get_supabase_client()
    resp = (
        supabase.table("investing_cash_movements")
        .select("*")
        .eq("client_id", str(client_id))
        .execute()
    )
    data = resp.data or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df["cash_out"] = pd.to_numeric(df.get("cash_out"), errors="coerce").fillna(0.0)
    df["cash_in"] = pd.to_numeric(df.get("cash_in"), errors="coerce").fillna(0.0)
    return df


def fetch_financing_flows_for_client(client_id) -> pd.DataFrame:
    supabase = get_supabase_client()
    resp = (
        supabase.table("financing_cash_movements")
        .select("*")
        .eq("client_id", str(client_id))
        .execute()
    )
    data = resp.data or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df["cash_out"] = pd.to_numeric(df.get("cash_out"), errors="coerce").fillna(0.0)
    df["cash_in"] = pd.to_numeric(df.get("cash_in"), errors="coerce").fillna(0.0)
    return df


def page_client_settings():
    st.title("⚙️ Client Settings")

    if not selected_client_id:
        st.info("Select a client from the sidebar to configure settings.")
        return

    # Always load settings once at top
    settings = get_client_settings(selected_client_id)

    # ----------------- Currency -----------------
    st.subheader("💱 Currency Settings")
    render_currency_settings_section(selected_client_id)

    st.markdown("---")

    # ----------------- Opening cash -----------------
    st.subheader("💰 Opening cash")

    opening_cash = st.number_input(
        "Opening cash balance at start of focus month",
        min_value=0.0,
        value=float(settings.get("opening_cash_start", 0.0)),
        step=1000.0,
        help="Used as the starting cash when you rebuild the 12-month cashflow engine.",
    )

    if st.button("Save opening cash"):
        supabase.table("client_settings").upsert(
            {
                "client_id": str(selected_client_id),
                "opening_cash_start": float(opening_cash),
            }
        ).execute()
        get_client_settings.clear()
        st.success("Opening cash saved.")

    st.markdown("---")

    # ----------------- AR / AP defaults -----------------
    st.subheader("📌 AR / AP Default Rules")
    col1, col2 = st.columns(2)

    with col1:
        ar_days = st.number_input(
            "Default AR collection days",
            min_value=0,
            max_value=180,
            value=settings["ar_default_days"],
        )
    with col2:
        ap_days = st.number_input(
            "Default AP payment days",
            min_value=0,
            max_value=180,
            value=settings["ap_default_days"],
        )

    if st.button("Save AR/AP defaults"):
        supabase.table("client_settings").upsert(
            {
                "client_id": str(selected_client_id),
                "ar_default_days": int(ar_days),
                "ap_default_days": int(ap_days),
            }
        ).execute()
        get_client_settings.clear()
        st.success("AR/AP defaults saved.")

    st.markdown("---")

    # ----------------- Runway / overspend thresholds -----------------
    st.subheader("🚨 Runway & Overspend Thresholds")
    col1, col2, col3 = st.columns(3)

    with col1:
        runway_min = st.number_input(
            "Minimum runway months",
            min_value=0.0,
            max_value=24.0,
            value=settings["runway_min_months"],
            step=0.5,
        )
    with col2:
        warn_pct = st.number_input(
            "Overspend warning %",
            min_value=0.0,
            max_value=100.0,
            value=settings["overspend_warn_pct"],
        )
    with col3:
        danger_pct = st.number_input(
            "Overspend danger %",
            min_value=0.0,
            max_value=100.0,
            value=settings["overspend_high_pct"],
        )

    if st.button("Save runway & overspend thresholds"):
        supabase.table("client_settings").upsert(
            {
                "client_id": str(selected_client_id),
                "runway_min_months": float(runway_min),
                "overspend_warn_pct": float(warn_pct),
                "overspend_high_pct": float(danger_pct),
            }
        ).execute()
        get_client_settings.clear()
        st.success("Risk thresholds saved.")

    st.markdown("---")

    # ----------------- Revenue recognition -----------------
    st.subheader("📊 Revenue Recognition Method")

    method = st.selectbox(
        "Select default revenue recognition method",
        ["saas", "milestone", "straight_line", "instant", "deferred"],
        index=["saas", "milestone", "straight_line", "instant", "deferred"]
        .index(settings["revenue_recognition_method"]),
    )

    if st.button("Save revenue recognition method"):
        supabase.table("client_settings").upsert(
            {
                "client_id": str(selected_client_id),
                "revenue_recognition_method": method,
            }
        ).execute()
        get_client_settings.clear()
        st.success("Revenue recognition method updated.")

# ---------- Helpers (data + utilities) ----------

@st.cache_data(ttl=60)
def load_clients():
    """
    Load clients from Supabase as a list of dicts: [{'id': ..., 'name': ...}, ...]
    Falls back to a single demo client if table is empty or fails.
    """
    try:
        response = supabase.table("clients").select("id, name").order("created_at").execute()
        rows = response.data or []
        if rows:
            return rows
    except Exception:
        pass

    # Fallback demo client (no id)
    return [{"id": None, "name": "Demo Startup"}]



# ---------- Payroll helpers ----------

@st.cache_data(ttl=60)
def fetch_payroll_positions_for_client(client_id):
    """
    Return payroll positions for a client as a DataFrame.

    Matches Supabase schema:

        base_salary_annual  -- annual base at 1.0 FTE
        super_rate_pct      -- e.g. 11.0
        payroll_tax_pct     -- e.g. 3.0
        fte                 -- FTE fraction
        start_date / end_date

    Handles missing columns defensively.
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("payroll_positions")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        print("Error fetching payroll positions:", e)
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Numeric fields
    for col in ["base_salary_annual", "super_rate_pct", "payroll_tax_pct", "fte"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Fallback defaults
            if col == "base_salary_annual":
                df[col] = 0.0
            elif col == "super_rate_pct":
                df[col] = 0.0
            elif col == "payroll_tax_pct":
                df[col] = 0.0
            elif col == "fte":
                df[col] = 1.0

    # Dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        else:
            df[col] = None

    # Optional contractor flag – not in your schema, so default False
    if "is_contractor" not in df.columns:
        df["is_contractor"] = False
    else:
        df["is_contractor"] = df["is_contractor"].fillna(False).astype(bool)

    return df

def upsert_payroll_position(
    client_id,
    position_id,
    role_name,
    employee_name,
    fte,
    base_salary_annual,
    super_rate_pct,
    payroll_tax_pct,
    start_date,
    end_date,
    notes,
) -> bool:
    """
    Insert or update a payroll position row.
    """
    if client_id is None:
        return False

    payload = {
        "client_id": str(client_id),
        "role_name": role_name or None,
        "employee_name": employee_name or None,
        "fte": float(fte) if fte is not None else 1.0,
        "base_salary_annual": float(base_salary_annual or 0),
        "super_rate_pct": float(super_rate_pct or 0),
        "payroll_tax_pct": float(payroll_tax_pct or 0),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "notes": notes or None,
    }

    try:
        if position_id is None:
            supabase.table("payroll_positions").insert(payload).execute()
        else:
            supabase.table("payroll_positions").update(payload).eq("id", position_id).execute()
        return True
    except Exception as e:
        print("Error upserting payroll position:", e)
        return False


def delete_payroll_position(position_id) -> bool:
    if not position_id:
        return False
    try:
        supabase.table("payroll_positions").delete().eq("id", position_id).execute()
        return True
    except Exception as e:
        print("Error deleting payroll position:", e)
        return False


from typing import List, Optional
import pandas as pd

# -----------------------------------------
# Central task fetcher for Collaboration Hub
# -----------------------------------------
def fetch_tasks_for_collab_hub(
    client_id: str,
    status_filter: Optional[List[str]] = None,
    page_filter: Optional[List[str]] = None,
    owner_filter: Optional[List[str]] = None,
    search_text: str = "",
) -> pd.DataFrame:
    """
    Fetch tasks for the central Collaboration Hub board.
    Uses existing `tasks` table:
      - status: open / in_progress / done / cancelled
      - owner_name: assignee
      - page_name: originating page (business_overview, sales_deals, etc)
    """
    query = (
        supabase.table("tasks")
        .select("*")
        .eq("client_id", client_id)
    )

    if status_filter:
        query = query.in_("status", status_filter)

    if page_filter:
        query = query.in_("page_name", page_filter)

    if owner_filter:
        query = query.in_("owner_name", owner_filter)

    search_text = (search_text or "").strip()
    if search_text:
        like_expr = f"%{search_text}%"
        # OR across title/description
        query = query.or_(
            f"title.ilike.{like_expr},description.ilike.{like_expr}"
        )

    resp = (
        query
        .order("due_date", desc=False)
        .order("priority", desc=False)
        .order("created_at", desc=False)
        .execute()
    )

    data = resp.data or []
    return pd.DataFrame(data)


# -----------------------------------------
# Distinct owners for filter dropdown
# -----------------------------------------
def get_task_owner_options(client_id: str) -> List[str]:
    """
    Distinct owner_name values, for filter dropdowns.
    We deduplicate in Python instead of using `distinct=` in select().
    """
    resp = (
        supabase.table("tasks")
        .select("owner_name")
        .eq("client_id", client_id)
        .execute()
    )
    rows = resp.data or []
    owners = [r["owner_name"] for r in rows if r.get("owner_name")]
    return sorted(set(owners))


# -----------------------------------------
# Distinct pages for filter dropdown
# -----------------------------------------
def get_task_page_options(client_id: str) -> List[str]:
    """
    Known page slugs + any additional page_name values found in tasks.
    """
    base_pages = [
        "business_overview",
        "sales_deals",
        "team_spending",
        "cash_bills",
        "alerts_todos",
        "collaboration_hub",
    ]

    resp = (
        supabase.table("tasks")
        .select("page_name")
        .eq("client_id", client_id)
        .execute()
    )
    rows = resp.data or []
    from_db = [r["page_name"] for r in rows if r.get("page_name")]

    all_pages = sorted(set(base_pages + from_db))
    return all_pages


def upsert_payroll_position(
    client_id,
    position_id,
    role_name,
    employee_name,
    fte,
    base_salary_annual,
    super_rate_pct,
    payroll_tax_pct,
    start_date,
    end_date,
    notes,
) -> bool:
    """
    Insert or update a payroll position row.
    If position_id is None -> insert; otherwise update that row.
    """
    if client_id is None:
        return False

    payload = {
        "client_id": str(client_id),
        "role_name": role_name or None,
        "employee_name": employee_name or None,
        "fte": float(fte) if fte is not None else 1.0,
        "base_salary_annual": float(base_salary_annual or 0),
        "super_rate_pct": float(super_rate_pct or 0),
        "payroll_tax_pct": float(payroll_tax_pct or 0),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "notes": notes or None,
    }

    try:
        if position_id is None:
            supabase.table("payroll_positions").insert(payload).execute()
        else:
            supabase.table("payroll_positions").update(payload).eq("id", position_id).execute()
        return True
    except Exception as e:
        print("Error upserting payroll position:", e)
        return False


def delete_payroll_position(position_id) -> bool:
    if not position_id:
        return False
    try:
        supabase.table("payroll_positions").delete().eq("id", position_id).execute()
        return True
    except Exception as e:
        print("Error deleting payroll position:", e)
        return False


def get_month_options(n_months: int = 18):
    """Return a list of month labels (most recent first)."""
    today = date.today().replace(day=1)
    months = []
    for i in range(n_months):
        m = today - pd.DateOffset(months=i)
        months.append(m.strftime("%b %Y"))  # e.g. "Dec 2025"
    return months


def parse_month_label_to_date(label: str) -> date:
    """
    Convert 'Dec 2025' → date(2025, 12, 1)
    """
    return pd.to_datetime(label, format="%b %Y").date().replace(day=1)

@st.cache_data(ttl=60)
def fetch_dept_monthly_for_client(client_id):
    """
    Fetch department-level monthly data for this client from dept_monthly.
    Expected columns (flexible):
      - month_date or month
      - department
      - budget_total or budget
      - actual_total or actual
      - variance, variance_pct (optional)
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("dept_monthly")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse month column
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    elif "month" in df.columns:
        df["month_date"] = pd.to_datetime(df["month"], errors="coerce")

    return df


def fetch_hiring_monthly_for_client(client_id):
    """
    Fetch hiring + headcount monthly data for this client from hiring_monthly.
    Expected columns (flexible):
      - month_date or month
      - department
      - roles_count
      - new_payroll
      - cumulative_roles
      - cumulative_payroll
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("hiring_monthly")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse month column
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    elif "month" in df.columns:
        df["month_date"] = pd.to_datetime(df["month"], errors="coerce")

    return df


def compute_payroll_by_month(client_id, month_index):
    """
    Compute total payroll cash-out for each month in month_index.

    Rules:
    - Each row in payroll_positions represents a role.
    - A role is active in a month if:
         start_month <= month <= end_month (or no end_date = open-ended).
    - Monthly cost = (base_salary_annual / 12 * FTE)
                     + super
                     + payroll tax & on-costs
    - Returns: dict {month_start_timestamp -> float(total_cost_for_that_month)}
    """
    # Make month_index a clean list of month starts (normalize to first of month)
    month_list = [
        pd.to_datetime(m).to_period("M").to_timestamp()
        for m in month_index
    ]
    if not month_list:
        return {}

    df_positions = fetch_payroll_positions_for_client(client_id)

    if df_positions is None or df_positions.empty:
        return {m: 0.0 for m in month_list}

    pos = df_positions.copy()

    # Coerce dates
    pos["start_date"] = pd.to_datetime(pos.get("start_date"), errors="coerce")
    pos["end_date"] = pd.to_datetime(pos.get("end_date"), errors="coerce")

    # Coerce numerics safely
    for col in ["base_salary_annual", "fte", "super_rate_pct", "payroll_tax_pct"]:
        if col not in pos.columns:
            pos[col] = 0.0
        pos[col] = pd.to_numeric(pos[col], errors="coerce").fillna(0.0)

    # Pre-normalise start/end month buckets (first day of month)
    pos["start_month"] = pos["start_date"].dt.to_period("M").dt.to_timestamp()
    pos["end_month"] = pos["end_date"].dt.to_period("M").dt.to_timestamp()

    # For roles with no start_date, treat as not active (you can relax this if you like)
    pos = pos[pos["start_month"].notna()]
    if pos.empty:
        return {m: 0.0 for m in month_list}

    # Precompute monthly cost per role (constant across months)
    # base monthly salary
    base_monthly = (pos["base_salary_annual"] / 12.0) * pos["fte"]

    super_monthly = base_monthly * (pos["super_rate_pct"] / 100.0)
    tax_monthly = base_monthly * (pos["payroll_tax_pct"] / 100.0)

    pos["monthly_cost"] = (base_monthly + super_monthly + tax_monthly).fillna(0.0)

    # Now build totals per month
    totals = {}

    for m in month_list:
        # Active roles in month m:
        #   start_month <= m AND (end_month is NaT OR m <= end_month)
        active_mask = (pos["start_month"] <= m) & (
            pos["end_month"].isna() | (pos["end_month"] >= m)
        )
        active_roles = pos[active_mask]

        if active_roles.empty:
            totals[m] = 0.0
        else:
            totals[m] = float(active_roles["monthly_cost"].sum())

    return totals


@st.cache_data(ttl=60)
def fetch_baseline_monthly_for_client(client_id):
    """
    Fetch baseline monthly numbers for this client from baseline_monthly.

    Expected (flexible) columns:
      - month_date or month or period
      - closing_cash  (used for the cash curve + CF engine)
      - optional: operating_cf, investing_cf, financing_cf, free_cash_flow
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("baseline_monthly")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date", desc=False)
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalise month_date
    month_col = None
    for cand in ["month_date", "month", "period"]:
        if cand in df.columns:
            month_col = cand
            break

    if month_col is None:
        return pd.DataFrame()

    df["month_date"] = pd.to_datetime(df[month_col], errors="coerce")
    df = df[df["month_date"].notna()]
    if df.empty:
        return pd.DataFrame()

    return df


def upload_csv_to_table_for_client(client_id, csv_file, table_name: str):
    """
    Simple helper to upload a CSV into a Supabase table for the current client.
    - Expects the CSV columns to match the table columns (except client_id).
    - Adds/overwrites client_id on each row.
    """
    if client_id is None:
        return False, "No client selected."

    try:
        df = pd.read_csv(csv_file)

        # Force client_id column
        df["client_id"] = str(client_id)

        records = df.to_dict(orient="records")
        if not records:
            return False, "CSV has no rows."

        supabase.table(table_name).insert(records).execute()
        return True, f"Inserted {len(records)} rows into {table_name}."
    except Exception as e:
        return False, f"Upload failed: {e}"

def get_engine_window_for_scenarios(
    client_id: str,
    base_month: date,
    n_months: int = 12,
) -> pd.DataFrame | None:
    """
    Pull a clean 12-month baseline window from cashflow_summary
    for scenarios. Does NOT write anything back.

    Ensures numeric columns and reconstructs opening cash.
    """
    engine_df = fetch_cashflow_summary_for_client(client_id)
    if engine_df is None or engine_df.empty:
        return None

    df = engine_df.copy()
    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df = df.dropna(subset=["month_date"])

    start = pd.to_datetime(base_month).replace(day=1)
    end = start + pd.DateOffset(months=n_months)

    df = df[(df["month_date"] >= start) & (df["month_date"] < end)].copy()
    if df.empty:
        return None

    df = df.sort_values("month_date").reset_index(drop=True)

    # Ensure required numeric fields
    for col in ["operating_cf", "investing_cf", "financing_cf", "free_cash_flow", "closing_cash"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Reconstruct opening cash series
    closing_0 = float(df.loc[0, "closing_cash"] or 0.0)
    free_0 = float(df.loc[0, "free_cash_flow"] or 0.0)
    opening_0 = closing_0 - free_0

    openings = [opening_0]
    for i in range(1, len(df)):
        openings.append(float(df.loc[i - 1, "closing_cash"] or 0.0))

    df["opening_cash"] = openings

    return df


#try Supabase servcer again if error arise
def sb_execute_with_retry(fn, label: str, tries: int = 3, base_wait: float = 0.6):
    """
    Runs a Supabase query with retry.
    Returns:
      - result (whatever fn returns) on success
      - None on failure after retries
    """
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"[WARN] {label} attempt {i+1}/{tries} failed: {e}")
            time.sleep(base_wait * (i + 1))

    print(f"[ERROR] {label} failed after {tries} tries: {last_err}")
    return None

def fetch_cashflow_summary_for_client(client_id: str) -> pd.DataFrame | None:
    """
    Fetch cashflow_summary rows for a client.

    Returns:
        - None  -> Supabase connection problem (sb_execute_with_retry returned None)
        - empty DataFrame -> call succeeded but no rows
        - non-empty DataFrame -> valid engine data
    """
    if client_id is None:
        return pd.DataFrame()

    def _call():
        return (
            supabase
            .table("cashflow_summary")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date")
            .execute()
        )

    res = sb_execute_with_retry(_call, label="fetch cashflow_summary")
    if res is None:
        print("[ERROR] fetch_cashflow_summary_for_client -> connection problem (None)")
        return None

    rows = res.data or []
    if not rows:
        print("[DEBUG] fetch_cashflow_summary_for_client -> 0 rows for", client_id)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
        df = df[df["month_date"].notna()].sort_values("month_date")

    print(f"[DEBUG] fetch_cashflow_summary_for_client -> {len(df)} rows for {client_id}")
    return df



def fetch_kpis_for_client_month(
    client_id: str,
    month_date: date | datetime,
) -> dict | None:
    """
    Get a single KPI row for (client, month_date) from kpi_monthly.

    If nothing exists, derive a KPI row from:
      - AR (cash in)
      - AP + payroll (+ optional opex) (cash out)
      - cashflow_summary (closing cash + runway)
    and just RETURN it (no upsert).
    """
    if client_id is None or month_date is None:
        return None

    # Normalise to first of month (as date + ISO string)
    month_dt = pd.to_datetime(month_date).date().replace(day=1)
    month_iso = month_dt.isoformat()

    # ---------- 1) Try to read from kpi_monthly directly ----------
    try:
        resp = (
            supabase.table("kpi_monthly")
            .select("*")
            .eq("client_id", client_id)
            .eq("month_date", month_iso)
            .execute()
        )

        rows = []
        if hasattr(resp, "data"):
            rows = resp.data or []
        elif isinstance(resp, dict):
            rows = resp.get("data", []) or []

        if rows:
            return rows[0]

    except Exception as e:
        print("Error fetching from kpi_monthly:", e)
        # Don't return yet — we can still derive

    # ---------- 2) Derive KPIs from AR/AP + payroll + engine ----------
    try:
        base_month_ts = pd.to_datetime(month_dt).replace(day=1)
        as_of_date = base_month_ts.date()

        # Settings + AR/AP
        settings = get_client_settings(client_id) or {}
        ar_days = int(settings.get("ar_default_days", 30))
        ap_days = int(settings.get("ap_default_days", 30))

        df_ar, df_ap = fetch_ar_ap_for_client(client_id)

        # ----- AR: expected cash-in this month -----
        money_in = 0.0
        if df_ar is not None and not df_ar.empty:
            df_ar = df_ar.copy()
            df_ar = add_ar_aging(df_ar, as_of=as_of_date, ar_default_days=ar_days)

            df_ar["expected_date"] = pd.to_datetime(df_ar.get("expected_date"), errors="coerce")
            df_ar["amount"] = pd.to_numeric(df_ar.get("amount"), errors="coerce").fillna(0.0)

            if "status" in df_ar.columns:
                df_ar = df_ar[~df_ar["status"].astype(str).str.lower().isin(["paid", "closed", "settled"])]

            mask_ar = df_ar["expected_date"].dt.to_period("M") == base_month_ts.to_period("M")
            money_in = float(df_ar.loc[mask_ar, "amount"].sum())

        # ----- AP: expected cash-out this month -----
        ap_cash = 0.0
        if df_ap is not None and not df_ap.empty:
            df_ap = df_ap.copy()
            df_ap = add_ap_aging(df_ap, as_of=as_of_date, ap_default_days=ap_days)

            if "pay_expected_date" in df_ap.columns:
                pay_col = "pay_expected_date"
            elif "expected_payment_date" in df_ap.columns:
                pay_col = "expected_payment_date"
            else:
                pay_col = "due_date"

            df_ap[pay_col] = pd.to_datetime(df_ap.get(pay_col), errors="coerce")
            df_ap["amount"] = pd.to_numeric(df_ap.get("amount"), errors="coerce").fillna(0.0)

            if "status" in df_ap.columns:
                df_ap = df_ap[~df_ap["status"].astype(str).str.lower().isin(["paid", "closed", "settled"])]

            mask_ap = df_ap[pay_col].dt.to_period("M") == base_month_ts.to_period("M")
            ap_cash = float(df_ap.loc[mask_ap, "amount"].sum())

        # ----- Payroll this month -----
        payroll_cash = 0.0
        try:
            payroll_by_month = compute_payroll_by_month(client_id, [base_month_ts])
            payroll_cash = float(payroll_by_month.get(base_month_ts, 0.0))
        except Exception as e:
            print("[WARN] compute_payroll_by_month failed:", e)
            payroll_cash = 0.0

        # ----- Optional Opex -----
        opex_cash = 0.0
        try:
            df_opex = fetch_opex_for_client(client_id)
            if df_opex is not None and not df_opex.empty:
                df_opex = df_opex.copy()
                df_opex["month_bucket"] = (
                    pd.to_datetime(df_opex.get("month_date"), errors="coerce")
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
                mask_ox = df_opex["month_bucket"] == base_month_ts
                opex_cash = float(pd.to_numeric(df_opex.loc[mask_ox, "amount"], errors="coerce").fillna(0.0).sum())
        except Exception:
            opex_cash = 0.0

        money_out = ap_cash + payroll_cash + opex_cash

        # ----- Engine: closing cash + runway -----
        closing_cash = None
        runway_months = None
        effective_burn = None

        engine_df = fetch_cashflow_summary_for_client(client_id)

        if engine_df is None:
            # connection failure: keep cash/runway as None (don’t fake zeros)
            pass
        elif engine_df.empty:
            # success but no engine rows yet
            pass
        else:
            df_eng = engine_df.copy()
            df_eng["month_date"] = pd.to_datetime(df_eng["month_date"], errors="coerce").dt.date
            row_eng = df_eng[df_eng["month_date"] == month_dt]

            if not row_eng.empty and "closing_cash" in row_eng.columns:
                closing_cash = float(row_eng["closing_cash"].iloc[0] or 0.0)

            # your runway helper supports focus_month keyword; safe to use
            runway_months, effective_burn = compute_runway_and_effective_burn_from_df(
            engine_df,
            selected_month_start,
        )

        kpi_row = {
            "client_id": client_id,
            "month_date": month_iso,
            "revenue": float(money_in),
            "burn": float(money_out),
            "cash_balance": closing_cash,      # None if engine unavailable
            "runway_months": runway_months,    # None if engine unavailable
        }

        print(
            f"Derived KPIs from cashflow_summary/AR/AP for client={client_id}, "
            f"month_date={month_iso}: {kpi_row}"
        )

        return kpi_row

    except Exception as e:
        print("Error deriving KPIs from cashflow_summary:", e)
        return None



def fetch_pipeline_for_client(client_id):
    """
    Fetch all revenue pipeline deals for the given client.
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("revenue_pipeline")
            .select("*")
            .eq("client_id", str(client_id))
            .order("created_at", desc=True)
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse dates
    for col in ["start_month", "end_month", "created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

#POC Revenue Recongition
@st.cache_data(ttl=60)
def fetch_poc_progress_for_client(client_id) -> pd.DataFrame:
    """
    Fetch all percentage-of-completion rows for this client.
    Each row is cumulative pct_complete for a given deal + month_date.
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("deal_poc_progress")
            .select("*")
            .eq("client_id", str(client_id))
            .order("deal_id")
            .order("month_date")
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        print("Error fetching POC progress:", e)
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    if "pct_complete" in df.columns:
        df["pct_complete"] = pd.to_numeric(df["pct_complete"], errors="coerce").fillna(0.0)
    return df

def upsert_poc_progress_row(
    client_id,
    deal_id: int,
    month_date: date,
    pct_complete: float,
    notes: str | None = None,
) -> bool:
    """
    Upsert a single POC row for (deal_id, month_date).
    pct_complete is cumulative 0–100.
    """
    if client_id is None or deal_id is None or month_date is None:
        return False

    try:
        payload = {
            "client_id": str(client_id),
            "deal_id": int(deal_id),
            "month_date": month_date.isoformat(),
            "pct_complete": float(pct_complete),
            "notes": notes or None,
        }

        (
            supabase
            .table("deal_poc_progress")
            .upsert(payload, on_conflict="deal_id,month_date")
            .execute()
        )

        # Clear cache so engine picks up new % immediately
        st.cache_data.clear()
        return True
    except Exception as e:
        print("Error upserting POC progress:", e)
        return False



#Milstone Revenue Recongition
@st.cache_data(ttl=60)
def fetch_milestones_for_deal(client_id, deal_id) -> pd.DataFrame:
    """
    Load all milestone rows for a single deal.
    """
    if client_id is None or deal_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("deal_milestones")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("deal_id", int(deal_id))
            .order("month_date")
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.date
    df["percent_of_contract"] = pd.to_numeric(
        df.get("percent_of_contract"), errors="coerce"
    ).fillna(0.0)

    return df


def save_milestones_for_deal(client_id, deal_id, milestones_df: pd.DataFrame) -> bool:
    """
    Replace milestones for this deal with the rows from milestones_df.
    Expected columns in milestones_df:
      - month_date (date or string)
      - percent_of_contract (float, 0–100)
      - milestone_name (optional)
    """
    if client_id is None or deal_id is None:
        return False

    if milestones_df is None or milestones_df.empty:
        # nothing to save
        return False

    df = milestones_df.copy()

    # Normalise and clean
    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.date
    df["percent_of_contract"] = pd.to_numeric(
        df.get("percent_of_contract"), errors="coerce"
    ).fillna(0.0)

    df = df[df["month_date"].notna() & (df["percent_of_contract"] > 0)]
    if df.empty:
        return False

    # Build payload
    payload = []
    for _, row in df.iterrows():
        payload.append(
            {
                "client_id": str(client_id),
                "deal_id": int(deal_id),
                "month_date": row["month_date"].isoformat(),
                "percent_of_contract": float(row["percent_of_contract"]),
                "milestone_name": row.get("milestone_name") or None,
            }
        )

    try:
        # Delete existing milestones for this deal
        supabase.table("deal_milestones").delete().eq(
            "client_id", str(client_id)
        ).eq("deal_id", int(deal_id)).execute()

        # Insert new ones
        supabase.table("deal_milestones").insert(payload).execute()

        # Clear cache so fetch_milestones_for_deal + revenue schedule recompute
        st.cache_data.clear()

        return True
    except Exception as e:
        print("Error saving milestones:", e)
        return False


def build_expected_revenue_curve(
    df_pipeline: pd.DataFrame,
    base_month: date,
    n_months: int = 12,
):
    """
    Build a simple expected revenue curve:
    For each deal, we place value_total * (probability_pct/100)
    into its start_month (or created_at month if start_month missing).
    """
    if df_pipeline is None or df_pipeline.empty:
        return pd.DataFrame()

    # Generate month buckets starting from base_month
    month_index = pd.date_range(
        start=pd.to_datetime(base_month),
        periods=n_months,
        freq="MS",  # month start
    )
    months = {m: 0.0 for m in month_index}

    for _, row in df_pipeline.iterrows():
        # Decide which month to use
        start_dt = row.get("start_month")
        if pd.isna(start_dt):
            start_dt = row.get("created_at")

        if pd.isna(start_dt):
            continue

        # Snap to first day of month
        month_key = pd.to_datetime(start_dt).to_period("M").to_timestamp()

        if month_key not in months:
            # Ignore deals outside our window for now
            continue

        value = row.get("value_total") or 0.0
        prob = row.get("probability_pct") or 0.0
        expected_value = float(value) * float(prob) / 100.0

        months[month_key] += expected_value

    # Build DataFrame for chart
    out_df = pd.DataFrame(
        {
            "month_date": month_index,
            "month_label": [m.strftime("%b %Y") for m in month_index],
            "expected_revenue": [months[m] for m in month_index],
        }
    )
    return out_df

@st.cache_data(ttl=60)
def build_revenue_schedule_for_client(
    client_id,
    base_month: date,
    n_months: int = 12,
) -> pd.DataFrame:
    """
    Build recognised revenue *by deal + month* for this client.

    - Uses a single revenue_recognition_method from client_settings
      (saas, milestone, poc, straight_line, usage, point_in_time)
    - Ignores deals with effective probability below `min_revenue_prob_pct`
    - Ignores stage = 'closed_lost'
    - If stage = 'closed_won' → probability = 100%
    - For milestone method, uses deal_milestones table to phase revenue
    - For SaaS / straight_line / point_in_time, uses pipeline dates + (optional) contract_months
    """

    if client_id is None or base_month is None:
        return pd.DataFrame()

    settings = get_client_settings(client_id)
    method = (settings.get("revenue_recognition_method") or "saas").lower()
    min_prob_pct = float(settings.get("min_revenue_prob_pct", 20))  # default 20%

    # Pipeline deals
    df = fetch_pipeline_for_client(client_id)
    # POC progress table (may be empty) – we keep this ready for future true-up logic
    df_poc_all = fetch_poc_progress_for_client(client_id)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["value_total"] = pd.to_numeric(df.get("value_total"), errors="coerce").fillna(0.0)
    df["probability_pct"] = pd.to_numeric(df.get("probability_pct"), errors="coerce").fillna(0.0)

    # Normalise dates
    for col in ["start_month", "end_month", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Effective probability based on stage
    def _effective_prob(row):
        stage = str(row.get("stage", "")).lower()
        if stage == "closed_lost":
            return 0.0
        if stage == "closed_won":
            return 1.0
        p = float(row.get("probability_pct", 0.0)) / 100.0
        return max(0.0, min(p, 1.0))

    df["effective_prob"] = df.apply(_effective_prob, axis=1)
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    # Drop low-probability deals
    df = df[df["effective_prob_pct"] >= min_prob_pct]
    if df.empty:
        return pd.DataFrame()

    # Month buckets we care about
    month_index = pd.date_range(
        start=pd.to_datetime(base_month),
        periods=n_months,
        freq="MS",
    )
    month_set = set(month_index)

    # Helper: figure out per-deal spreading for the simple methods
    def _spread_evenly(start_ts, end_ts, total_amount, months_to_spread):
        """
        Returns list of (month_bucket, amount) for this deal.
        """
        if total_amount == 0 or pd.isna(start_ts) or months_to_spread <= 0:
            return []

        start_bucket = pd.to_datetime(start_ts).to_period("M").to_timestamp()

        local_months: list[pd.Timestamp] = []
        for m in month_index:
            if m < start_bucket:
                continue
            if end_ts is not None and not pd.isna(end_ts):
                end_bucket = pd.to_datetime(end_ts).to_period("M").to_timestamp()
                if m > end_bucket:
                    continue
            local_months.append(m)
            if len(local_months) >= months_to_spread:
                break

        if not local_months:
            return []

        per_month = float(total_amount) / len(local_months)
        return [(m, per_month) for m in local_months]

    # ------------- MAIN LOOP: build schedule rows -------------
    rows: list[dict] = []

    for _, row in df.iterrows():
        deal_id = row.get("id")
        gross_val = float(row.get("value_total", 0.0))
        eff_prob = float(row.get("effective_prob", 0.0))
        effective_val = gross_val * eff_prob

        if effective_val == 0:
            continue

        start_ts = row.get("start_month")
        end_ts = row.get("end_month")
        created_ts = row.get("created_at")

        if pd.isna(start_ts):
            start_ts = created_ts
        if pd.isna(start_ts):
            # No usable timing → skip
            continue

        # Revenue type label (for charts)
        revenue_type = method  # e.g. 'saas', 'milestone', 'poc', 'straight_line', etc.

        # ---------- CASE 1: Milestone method ----------
        if method == "milestone":
            ms_df = fetch_milestones_for_deal(client_id, deal_id)
            used_any = False

            if ms_df is not None and not ms_df.empty:
                for _, ms in ms_df.iterrows():
                    md = ms.get("month_date")
                    pct = float(ms.get("percent_of_contract") or 0.0)

                    if pd.isna(md) or pct <= 0:
                        continue

                    bucket = pd.to_datetime(md).to_period("M").to_timestamp()
                    if bucket not in month_set:
                        continue

                    allocated = effective_val * (pct / 100.0)
                    if allocated == 0:
                        continue

                    rows.append(
                        {
                            "client_id": str(client_id),
                            "deal_id": deal_id,
                            "month_date": bucket,
                            "revenue_amount": round(allocated, 2),
                            "revenue_type": revenue_type,
                        }
                    )
                    used_any = True

            # If we used milestones, skip fallback
            if used_any:
                continue

            # Fallback: no milestones → recognise all in the start month
            start_bucket = pd.to_datetime(start_ts).to_period("M").to_timestamp()
            if start_bucket in month_set:
                rows.append(
                    {
                        "client_id": str(client_id),
                        "deal_id": deal_id,
                        "month_date": start_bucket,
                        "revenue_amount": round(effective_val, 2),
                        "revenue_type": revenue_type,
                    }
                )
            continue

        # ---------- CASE 2: SaaS subscription ----------
        if method == "saas":
            # Optional per-deal contract length
            contract_months = None
            if "contract_months" in df.columns:
                try:
                    contract_months = int(row.get("contract_months") or 0)
                except Exception:
                    contract_months = None

            if not contract_months or contract_months <= 0:
                contract_months = int(settings.get("saas_default_months", 12) or 12)

            allocations = _spread_evenly(
                start_ts,
                None,  # no fixed end date needed for SaaS
                effective_val,
                months_to_spread=contract_months,
            )

        # ---------- CASE 3: Straight-line service / simple POC baseline ----------
        elif method in ["straight_line", "poc"]:
            # If we have an explicit end date, use the actual duration
            if not pd.isna(end_ts):
                start_bucket = pd.to_datetime(start_ts).to_period("M").to_timestamp()
                end_bucket = pd.to_datetime(end_ts).to_period("M").to_timestamp()
                # number of months inclusive
                n = max(
                    1,
                    (end_bucket.to_period("M") - start_bucket.to_period("M")).n + 1,
                )
                months_to_spread = n
            else:
                # No end date: try contract_months, else fallback to project_default_months (e.g. 6)
                contract_months = None
                if "contract_months" in df.columns:
                    try:
                        contract_months = int(row.get("contract_months") or 0)
                    except Exception:
                        contract_months = None

                if not contract_months or contract_months <= 0:
                    contract_months = int(settings.get("project_default_months", 6) or 6)

                months_to_spread = contract_months

            allocations = _spread_evenly(
                start_ts,
                end_ts,
                effective_val,
                months_to_spread=months_to_spread,
            )

            # NOTE: Right now POC uses this straight-line baseline.
            # The POC true-up logic using df_poc_all can be layered on top later.

        # ---------- CASE 4: Point-in-time goods delivery ----------
        elif method == "point_in_time":
            start_bucket = pd.to_datetime(start_ts).to_period("M").to_timestamp()
            allocations = []
            if start_bucket in month_set:
                allocations.append((start_bucket, effective_val))

        # ---------- CASE 5: Usage-based (temporary rule) ----------
        elif method == "usage":
            allocations = _spread_evenly(
                start_ts,
                None,
                effective_val,
                months_to_spread=3,  # short-term spreading as placeholder
            )

        # ---------- CASE 6: Fallback ----------
        else:
            allocations = _spread_evenly(
                start_ts,
                None,
                effective_val,
                months_to_spread=6,
            )

        # Add rows for all non-milestone methods
        for bucket, amt in allocations:
            if bucket not in month_set or amt == 0:
                continue

            rows.append(
                {
                    "client_id": str(client_id),
                    "deal_id": deal_id,
                    "month_date": bucket,
                    "revenue_amount": round(amt, 2),
                    "revenue_type": revenue_type,
                }
            )

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows)
    out_df["month_date"] = pd.to_datetime(out_df["month_date"], errors="coerce")
    out_df = out_df[out_df["month_date"].notna()]
    out_df = out_df.sort_values("month_date")
    out_df["month_label"] = out_df["month_date"].dt.strftime("%b %Y")

    return out_df

# ---------- Revenue recognition engine (6 modes) ----------

def _append_revenue_row(
    rows: list,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    deal_row: pd.Series,
    method_label: str,
    month_value,
    amount: float,
):
    """
    Internal helper: add a revenue row if month_value is within the window and amount != 0.
    """
    if amount is None:
        return
    try:
        amount = float(amount)
    except Exception:
        return
    if abs(amount) < 1e-6:
        return

    if pd.isna(month_value):
        return

    month_ts = pd.to_datetime(month_value, errors="coerce")
    if pd.isna(month_ts):
        return

    # Snap to first of month
    month_ts = month_ts.to_period("M").to_timestamp()

    if not (window_start <= month_ts < window_end):
        return

    rows.append(
        {
            "month_date": month_ts,
            "revenue_amount": amount,
            "revenue_type": method_label,
            "deal_id": deal_row.get("id"),
            "deal_name": deal_row.get("deal_name"),
            "customer_name": deal_row.get("customer_name"),
        }
    )


def _determine_method_key(method_raw: str) -> str:
    """
    Take whatever is stored in revenue_pipeline.method and normalise to a key:
      'saas', 'milestone', 'poc', 'straight', 'usage', 'point'
    """
    m = (method_raw or "").lower()

    if "usage" in m:
        return "usage"
    if "percent" in m or "poc" in m:
        return "poc"
    if "straight" in m or "service" in m:
        return "straight"
    if "point" in m or "goods" in m:
        return "point"
    if "mile" in m or "project" in m:
        return "milestone"
    # default: treat as SaaS
    return "saas"


def _schedule_saas(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    SaaS subscription revenue:
      - value_total spread evenly over contract_length_months (or 12 if missing)
      - optional annual uplift from annual_uplift_pct
    """
    total = float(deal.get("value_total") or 0.0) * prob_factor
    if total == 0:
        return

    start_raw = deal.get("start_month") or deal.get("created_at")
    if pd.isna(start_raw):
        return
    start_ts = pd.to_datetime(start_raw, errors="coerce")
    if pd.isna(start_ts):
        return
    start_ts = start_ts.to_period("M").to_timestamp()

    # Contract length in months
    contract_len = deal.get("contract_length_months") or deal.get("contract_length")
    try:
        contract_len = int(contract_len) if contract_len is not None else 12
    except Exception:
        contract_len = 12
    if contract_len <= 0:
        contract_len = 12

    uplift = 0.0
    try:
        uplift = float(deal.get("annual_uplift_pct") or 0.0)
    except Exception:
        uplift = 0.0

    base_mrr = total / contract_len if contract_len else 0.0

    for i in range(contract_len):
        month_ts = start_ts + pd.DateOffset(months=i)
        # Which "year" of the contract is this month in (0,1,2,...)
        year_idx = i // 12
        if uplift > 0:
            factor = (1 + uplift / 100.0) ** year_idx
        else:
            factor = 1.0
        amount = base_mrr * factor
        _append_revenue_row(
            rows,
            window_start,
            window_end,
            deal,
            "SaaS subscription",
            month_ts,
            amount,
        )


def _schedule_straight_line(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    Straight-line service revenue:
      - evenly from start_month to service_end_month (or end_month)
    """
    total = float(deal.get("value_total") or 0.0) * prob_factor
    if total == 0:
        return

    start_raw = deal.get("start_month") or deal.get("created_at")
    end_raw = deal.get("service_end_month") or deal.get("end_month")

    if pd.isna(start_raw) or pd.isna(end_raw):
        return

    start_ts = pd.to_datetime(start_raw, errors="coerce")
    end_ts = pd.to_datetime(end_raw, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return

    start_ts = start_ts.to_period("M").to_timestamp()
    end_ts = end_ts.to_period("M").to_timestamp()

    if end_ts < start_ts:
        return

    month_index = pd.date_range(start=start_ts, end=end_ts, freq="MS")
    n = len(month_index)
    if n <= 0:
        return

    per_month = total / n

    for m in month_index:
        _append_revenue_row(
            rows,
            window_start,
            window_end,
            deal,
            "Straight-line service",
            m,
            per_month,
        )


def _schedule_milestone(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    Milestone / project revenue:
      - expects milestone_data JSONB = list of {name, amount, month}
      - if missing, fallback: recognise full amount in start_month
    """
    total = float(deal.get("value_total") or 0.0) * prob_factor
    milestone_data = deal.get("milestone_data")

    if milestone_data:
        # Expect list of dicts: {"name": ..., "amount": ..., "month": "2025-12-01"}
        for ms in milestone_data:
            try:
                ms_amount = float(ms.get("amount", 0.0)) * prob_factor
            except Exception:
                ms_amount = 0.0
            ms_month = ms.get("month") or ms.get("month_date")
            _append_revenue_row(
                rows,
                window_start,
                window_end,
                deal,
                "Milestone project",
                ms_month,
                ms_amount,
            )
    else:
        # Fallback: full amount in start_month
        start_raw = deal.get("start_month") or deal.get("created_at")
        if not pd.isna(start_raw):
            _append_revenue_row(
                rows,
                window_start,
                window_end,
                deal,
                "Milestone project",
                start_raw,
                total,
            )


def _schedule_poc(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    Percentage-of-completion (simple v1):

      Option A: poc_schedule JSONB with [{"month": ..., "percent": 30}, ...]
        -> revenue = total * percent/100

      Option B (fallback): behave like straight-line between start_month and end_month.
    """
    total = float(deal.get("value_total") or 0.0) * prob_factor
    poc_schedule = deal.get("poc_schedule")

    if poc_schedule:
        for item in poc_schedule:
            try:
                pct = float(item.get("percent", 0.0))
            except Exception:
                pct = 0.0
            month_val = item.get("month") or item.get("month_date")
            amount = total * (pct / 100.0)
            _append_revenue_row(
                rows,
                window_start,
                window_end,
                deal,
                "Percentage-of-completion",
                month_val,
                amount,
            )
    else:
        # Fallback: straight-line
        _schedule_straight_line(deal, window_start, window_end, rows, prob_factor)


def _schedule_usage(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    Usage-based revenue:
      usage_data JSONB = list of {month: '2025-12-01', units: 320, rate: 0.1}
    """
    usage_data = deal.get("usage_data")
    if not usage_data:
        return

    for item in usage_data:
        try:
            units = float(item.get("units", 0.0))
            rate = float(item.get("rate", 0.0))
        except Exception:
            continue
        amount = units * rate * prob_factor
        month_val = item.get("month") or item.get("month_date")
        _append_revenue_row(
            rows,
            window_start,
            window_end,
            deal,
            "Usage-based",
            month_val,
            amount,
        )


def _schedule_point_in_time(deal: pd.Series, window_start, window_end, rows: list, prob_factor: float):
    """
    Point-in-time goods delivery:
      - if delivery_date present, use that
      - else use start_month
    """
    total = float(deal.get("value_total") or 0.0) * prob_factor
    if total == 0:
        return

    delivery_raw = deal.get("delivery_date") or deal.get("start_month")
    if pd.isna(delivery_raw):
        return

    _append_revenue_row(
        rows,
        window_start,
        window_end,
        deal,
        "Point-in-time goods",
        delivery_raw,
        total,
    )



def get_effective_probability(row: dict, min_prob: float) -> float:
    """
    Compute effective probability for revenue recognition.

    - closed_lost -> 0
    - closed_won  -> 100
    - other stages -> probability_pct if >= min_prob, else 0 (paused)
    """
    stage = str(row.get("stage", "")).lower()
    prob_raw = row.get("probability_pct", 0) or 0
    try:
        prob = float(prob_raw)
    except Exception:
        prob = 0.0

    if stage == "closed_lost":
        return 0.0
    if stage == "closed_won":
        return 100.0
    if prob < min_prob:
        return 0.0
    return prob


@st.cache_data(ttl=60)
def compute_revenue_schedule(client_id, base_month: date, n_months: int = 12) -> pd.DataFrame:
    """
    Revenue recognition engine (v1):
    Turns revenue_pipeline into a monthly expected revenue schedule.

    - Uses probability-weighted value_total
    - SaaS / Accrual: spread evenly from start_month to end_month (if end given),
      otherwise all in start_month
    - Milestone / other methods: full amount in start_month
    """
    if client_id is None or base_month is None:
        return pd.DataFrame()

    df_pipeline = fetch_pipeline_for_client(client_id)
    if df_pipeline is None or df_pipeline.empty:
        return pd.DataFrame()

    # Month buckets
    month_index = pd.date_range(
        start=pd.to_datetime(base_month),
        periods=n_months,
        freq="MS",
    )
    buckets = {m: 0.0 for m in month_index}

    for _, row in df_pipeline.iterrows():
        total = float(row.get("value_total") or 0.0)
        prob = float(row.get("probability_pct") or 0.0) / 100.0
        method = str(row.get("method") or "SaaS").lower()

        # Choose start month
        start = row.get("start_month")
        if pd.isna(start):
            start = row.get("created_at")
        if pd.isna(start):
            continue

        start = pd.to_datetime(start).to_period("M").to_timestamp()
        if start not in buckets:
            # For now, ignore deals outside our window
            continue

        # SaaS / Accrual – spread evenly if end_month exists
        if method in ["saas", "accrual"]:
            end = row.get("end_month")
            if pd.isna(end):
                # No explicit end -> recognise all in start month
                amt = total * prob
                buckets[start] += amt
            else:
                end = pd.to_datetime(end).to_period("M").to_timestamp()
                if end < start:
                    end = start

                periods = pd.date_range(start=start, end=end, freq="MS")
                if len(periods) == 0:
                    continue

                amt_per_month = (total * prob) / len(periods)
                for m in periods:
                    if m in buckets:
                        buckets[m] += amt_per_month
        else:
            # Milestone / other: full amount at start
            amt = total * prob
            buckets[start] += amt

    out_df = pd.DataFrame(
        {
            "month_date": list(buckets.keys()),
            "expected_revenue": list(buckets.values()),
        }
    ).sort_values("month_date")

    return out_df
def page_investing_financing():
    top_header("Funding Strategy")

    # -------------------------------
    # Guards
    # -------------------------------
    if not selected_client_id:
        st.info("Select a business from the sidebar to see funding strategy.")
        return

    if not selected_month_start:
        st.info("Pick a focus month in the navbar to analyse funding strategy.")
        return

    currency_code, currency_symbol = get_client_currency(selected_client_id)
    settings = get_client_settings(selected_client_id) or {}

    # ---- Load engine + flows ----
    engine_df = fetch_cashflow_summary_for_client(selected_client_id)
    if engine_df is None:
        st.error("Could not load cashflow engine (Supabase disconnected). Please refresh.")
        return
    if engine_df.empty:
        st.info("No cashflow engine rows yet. Rebuild cashflow from Business overview.")
        return

    df_inv = fetch_investing_flows_for_client(selected_client_id)
    if df_inv is None:
        df_inv = pd.DataFrame()

    df_fin = fetch_financing_flows_for_client(selected_client_id)
    if df_fin is None:
        df_fin = pd.DataFrame()


    # ---- Runway + effective burn from engine_df ----
    runway_from_engine, effective_burn = compute_runway_and_effective_burn_from_df(
        engine_df,
        selected_month_start,
    )

    if runway_from_engine is None or effective_burn is None:
        st.info("Not enough cashflow engine data yet. Rebuild cashflow from Business overview.")
        return

    current_runway = float(runway_from_engine)
    current_burn = float(effective_burn)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _month_start(x):
        return pd.to_datetime(x).to_period("M").to_timestamp()

    def _prep_engine_monthly(df: pd.DataFrame) -> pd.DataFrame:
        ef = df.copy()
        ef["month_date"] = pd.to_datetime(ef["month_date"], errors="coerce")
        ef = ef[ef["month_date"].notna()].sort_values("month_date")
        ef["month_date"] = ef["month_date"].dt.to_period("M").dt.to_timestamp()

        # Ensure numeric cashflow cols exist
        for c in ["operating_cf", "investing_cf", "financing_cf", "net_cash", "closing_cash"]:
            if c in ef.columns:
                ef[c] = pd.to_numeric(ef[c], errors="coerce").fillna(0.0)

        # Compute net_cash if missing
        if "net_cash" not in ef.columns or ef["net_cash"].isna().all():
            if all(c in ef.columns for c in ["operating_cf", "investing_cf", "financing_cf"]):
                ef["net_cash"] = ef["operating_cf"] + ef["investing_cf"] + ef["financing_cf"]
            else:
                ef["net_cash"] = 0.0

        # If closing_cash missing, we still allow charts using cumulative net
        if "closing_cash" not in ef.columns or ef["closing_cash"].isna().all():
            ef["closing_cash"] = np.nan

        return ef

    def _derive_opening_cash_for_first_row(ef: pd.DataFrame) -> float:
        # Best: opening = closing - net for focus month
        focus = _month_start(selected_month_start)
        row = ef[ef["month_date"] == focus]
        if not row.empty:
            closing = float(row["closing_cash"].iloc[0]) if "closing_cash" in row.columns else 0.0
            net = float(row["net_cash"].iloc[0]) if "net_cash" in row.columns else 0.0
            if not np.isnan(closing):
                return closing - net

        # Fallback: use earliest available closing - net
        first = ef.iloc[0]
        closing = float(first.get("closing_cash", 0.0) or 0.0)
        net = float(first.get("net_cash", 0.0) or 0.0)
        return (closing - net) if not np.isnan(closing) else 0.0

    def _recompute_closing_cash_series(ef: pd.DataFrame, opening_cash: float) -> pd.DataFrame:
        ef2 = ef.copy().sort_values("month_date")
        closes = []
        opens = []
        o = float(opening_cash)

        for _, r in ef2.iterrows():
            opens.append(o)
            net = float(r.get("net_cash", 0.0) or 0.0)
            c = o + net
            closes.append(c)
            o = c

        ef2["opening_cash_sim"] = opens
        ef2["closing_cash_sim"] = closes
        return ef2

    # -------------------------------
    # Prepare engine window (12 months)
    # -------------------------------
    ef = _prep_engine_monthly(engine_df)

    focus_start = _month_start(selected_month_start)
    start = focus_start
    end = start + pd.DateOffset(months=12)
    ef_12 = ef[(ef["month_date"] >= start) & (ef["month_date"] < end)].copy()

    # Existing cash and cliff from engine (use actual closing_cash if present; else use simulated)
    existing_cash = 0.0
    cliff_date = None

    if not ef_12.empty:
        r0 = ef_12[ef_12["month_date"] == focus_start]
        if not r0.empty and "closing_cash" in ef_12.columns and not np.isnan(float(r0["closing_cash"].iloc[0])):
            existing_cash = float(r0["closing_cash"].iloc[0] or 0.0)

        if "closing_cash" in ef_12.columns and ef_12["closing_cash"].notna().any():
            below = ef_12[ef_12["closing_cash"] <= 0]
            if not below.empty:
                cliff_date = below["month_date"].iloc[0].date()
        else:
            # fallback to simulated cliff using cumulative net
            opening0 = _derive_opening_cash_for_first_row(ef_12)
            ef_tmp = _recompute_closing_cash_series(ef_12, opening0)
            below = ef_tmp[ef_tmp["closing_cash_sim"] <= 0]
            if not below.empty:
                cliff_date = below["month_date"].iloc[0].date()
            # approximate "existing cash" from sim
            r0 = ef_tmp[ef_tmp["month_date"] == focus_start]
            if not r0.empty:
                existing_cash = float(r0["closing_cash_sim"].iloc[0] or 0.0)

    # ===============================
    # NEW STRUCTURE
    # Left column = Decision + timeline + survival + cash curve + recommendation strip
    # Right column = Raise KPI + simulator + funding mix tabs
    # ===============================
    col_left, col_right = st.columns([2, 1])

    # -------------------------------
    # LEFT: Funding decision panel + timeline (and later, survival + cash curve + strip)
    # -------------------------------
    with col_left:
        st.subheader("🚀 Funding decision panel")

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Cash in bank (engine)", f"{currency_symbol}{existing_cash:,.0f}")
        with colB:
            st.metric("Effective burn", f"{currency_symbol}{current_burn:,.0f} / month")
        with colC:
            st.metric("Runway (forward)", f"{current_runway:.1f} months")
        with colD:
            st.metric("Cash cliff", cliff_date.strftime("%b %Y") if cliff_date else "No cliff in forecast")

        if cliff_date:
            st.warning(
                f"Your forecast goes cash-negative around **{cliff_date.strftime('%b %Y')}**. "
                "To avoid last-minute fundraising, start planning your raise **2–3 months earlier**."
            )
        else:
            st.success("Cash does not go negative in your current forecast window. Funding is optional—optimize growth efficiency.")

        # ---- Timeline chart (next 12 months) directly under KPIs ----
        st.markdown("---")
        st.subheader("📆 Investing & financing timeline (next 12 months)")

        if ef_12.empty:
            st.caption("No cashflow engine data in this 12-month window.")
        else:
            plot_cols = [c for c in ["investing_cf", "financing_cf"] if c in ef_12.columns]
            if not plot_cols:
                st.caption("No investing or financing cashflows in this 12-month window yet.")
            else:
                long_df = ef_12.melt(
                    id_vars="month_date",
                    value_vars=plot_cols,
                    var_name="Type",
                    value_name="Amount",
                )
                type_labels = {
                    "investing_cf": "Investing (CAPEX / R&D / assets)",
                    "financing_cf": "Financing (Equity / Debt / Repayments)",
                }
                long_df["Type"] = long_df["Type"].map(type_labels).fillna(long_df["Type"])

                chart = (
                    alt.Chart(long_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                        y=alt.Y("Amount:Q", title=f"Cashflow ({currency_code})"),
                        color=alt.Color("Type:N", title="Flow"),
                        tooltip=[
                            alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                            "Type:N",
                            alt.Tooltip("Amount:Q", title="Cashflow", format=",.0f"),
                        ],
                    )
                )
                st.altair_chart(chart.interactive(), width="stretch")
                st.caption("Positive = cash in. Negative = cash out.")

    # -------------------------------
    # RIGHT: Raise KPI (top) + Raise simulator (middle) + Funding mix tabs (bottom)
    # -------------------------------
    with col_right:
        st.subheader("🧮 Raise plan")

        # ---- Simulator inputs ----
        default_target_runway = float(settings.get("target_runway_months", 12) or 12)
        target_runway = st.number_input(
            "Target runway after raise (months)",
            min_value=3.0,
            max_value=36.0,
            step=1.0,
            value=default_target_runway,
            help="How many months of runway you want after funding lands.",
        )

        burn_change_pct = st.slider(
            "Planned burn change for growth plan",
            min_value=-50,
            max_value=150,
            value=0,
            help="Increase for hiring/marketing; decrease for cost cuts.",
        )

        adjusted_burn = current_burn * (1 + burn_change_pct / 100.0)
        if adjusted_burn <= 0:
            st.error("Adjusted burn must be > 0 to calculate a funding plan.")
            return

        raise_month = st.date_input(
            "When will the funding land? (month)",
            value=selected_month_start,
            help="Model a realistic timing for when cash hits the bank.",
            key="raise_month_picker",
        )
        raise_month = pd.to_datetime(raise_month).to_period("M").to_timestamp().date()
        raise_month_ts = pd.to_datetime(raise_month).to_period("M").to_timestamp()

        # ---- Funding amount guidance (buffers) ----
        st.markdown("##### Funding amount guidance")
        safety_buffer_pct = st.slider(
            "Safety buffer (%)",
            min_value=0,
            max_value=50,
            value=int(settings.get("funding_safety_buffer_pct", 15) or 15),
            help="Adds a buffer so you don’t raise too little.",
        )
        one_off_costs = st.number_input(
            f"One-off raise costs ({currency_code})",
            min_value=0.0,
            step=5000.0,
            value=float(settings.get("one_off_raise_costs", 0.0) or 0.0),
            help="Legal, audit, hiring fees, equipment step-changes, etc.",
        )
        wc_buffer = st.number_input(
            f"Working-capital buffer ({currency_code})",
            min_value=0.0,
            step=5000.0,
            value=float(settings.get("working_capital_buffer", 0.0) or 0.0),
            help="Extra buffer if collections are slow / AR timing risk.",
        )

        # ---- Recommended raise ----
        base_required_cash = adjusted_burn * float(target_runway)
        buffer_mult = 1.0 + (float(safety_buffer_pct) / 100.0)
        required_cash = (base_required_cash * buffer_mult) + float(one_off_costs) + float(wc_buffer)
        funding_gap = max(0.0, required_cash - existing_cash)

        max_raise = max(funding_gap * 2, adjusted_burn * 6, 50_000.0)
        raise_amount = st.slider(
            "Raise amount (what you plan to raise)",
            min_value=0.0,
            max_value=float(max_raise),
            value=float(funding_gap),
            step=float(max(1000.0, max_raise / 50)),
        )

        # ---- KPI block (TOP of right column, as requested) ----
        new_cash_buffer_simple = existing_cash + raise_amount
        new_runway_est_simple = (new_cash_buffer_simple / adjusted_burn) if adjusted_burn else None

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Recommended raise", f"{currency_symbol}{funding_gap:,.0f}")
        with k2:
            st.metric("Planned raise", f"{currency_symbol}{raise_amount:,.0f}")
        with k3:
            st.metric("Runway after raise (simple)", f"{new_runway_est_simple:.1f}m" if new_runway_est_simple else "—")

        st.caption(
            f"Adjusted burn: **{currency_symbol}{adjusted_burn:,.0f}/month** · "
            f"Target runway: **{target_runway:.0f} months** · "
            f"Buffers: **{safety_buffer_pct}% + {currency_symbol}{one_off_costs:,.0f} + {currency_symbol}{wc_buffer:,.0f}**"
        )

        # -------------------------------
        # Funding Mix (Equity / Debt / Grants) – upgraded
        # -------------------------------
        st.markdown("---")
        st.subheader("🏦 Funding mix options")

        tab_eq, tab_debt, tab_grants = st.tabs(["Equity", "Debt", "Grants"])

        with tab_eq:
            st.markdown("#### Equity impact (post-money + ownership remaining)")

            pre_money = st.number_input(
                "Pre-money valuation (rough)",
                min_value=0.0,
                step=100000.0,
                value=float(settings.get("rough_pre_money", 0.0) or 0.0),
            )

            founder_ownership_pct = st.number_input(
                "Current founder ownership (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=float(settings.get("founder_ownership_pct", 100.0) or 100.0),
                help="If you have co-founders/investors already, enter the founder’s current %.",
            )

            if pre_money > 0 and raise_amount > 0:
                post_money = pre_money + raise_amount
                dilution = (raise_amount / post_money) * 100.0
                ownership_remaining = founder_ownership_pct * (pre_money / post_money)

                st.info(
                    f"Post-money ≈ **{currency_symbol}{post_money:,.0f}** · "
                    f"Dilution ≈ **{dilution:.1f}%** · "
                    f"Founder ownership remaining ≈ **{ownership_remaining:.1f}%**"
                )
            else:
                st.caption("Enter pre-money valuation and a raise amount to see post-money, dilution, and ownership remaining.")

        with tab_debt:
            st.markdown("#### Debt sanity (DSCR-style + runway impact)")

            interest = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.5)
            term_months = st.number_input("Loan term (months)", min_value=3, max_value=84, value=24, step=1)

            if raise_amount > 0 and term_months > 0:
                r = (interest / 100.0) / 12.0
                if r == 0:
                    pmt = raise_amount / term_months
                else:
                    pmt = raise_amount * (r * (1 + r) ** term_months) / (((1 + r) ** term_months) - 1)

                # “DSCR-style” sanity: compare ability to carry repayments vs cash burn
                # We treat adjusted_burn as baseline cash cost. Debt adds repayment pressure.
                burn_with_debt = adjusted_burn + float(pmt)
                runway_with_debt = (existing_cash / burn_with_debt) if burn_with_debt > 0 else None

                # Coverage proxy: repayment as % of burn (lower is healthier)
                repayment_share = (pmt / adjusted_burn) * 100.0 if adjusted_burn > 0 else None

                st.info(
                    f"Estimated monthly repayment ≈ **{currency_symbol}{pmt:,.0f} / month**"
                )

                cA, cB = st.columns(2)
                with cA:
                    st.metric("Repayment as % of burn", f"{repayment_share:.0f}%" if repayment_share is not None else "—")
                with cB:
                    st.metric("Runway with debt repayment", f"{runway_with_debt:.1f}m" if runway_with_debt else "—")

                if runway_with_debt is not None and runway_with_debt < current_runway:
                    st.warning(
                        "Debt repayments shorten runway. If you’re already tight, consider equity, a smaller debt tranche, "
                        "or push the repayment start later."
                    )
            else:
                st.caption("Set a raise amount and loan term to estimate repayment pressure and runway impact.")

        with tab_grants:
            st.markdown("#### Grants (one-off inflows)")
            grant_amount = st.number_input("Expected grant amount", min_value=0.0, step=1000.0, value=0.0)
            if grant_amount > 0:
                grant_runway = (existing_cash + grant_amount) / adjusted_burn
                st.info(
                    f"A **{currency_symbol}{grant_amount:,.0f}** grant would lift simple runway to ~**{grant_runway:.1f} months**."
                )
            else:
                st.caption("Enter a grant amount to see runway lift.")

    # ==========================================================
    # BELOW THE MAIN KPI BLOCK (LEFT COLUMN CONTINUES):
    # 1) Can we survive until the raise lands?
    # 2) Cash curve before vs after raise
    # 3) Funding recommendation strip
    # ==========================================================
    with col_left:
        st.markdown("---")
        st.subheader("🧱 Can you survive until the raise lands?")

        if ef_12.empty:
            st.caption("No engine data to test survival.")
        else:
            # Base series (simulated if needed)
            opening0 = _derive_opening_cash_for_first_row(ef_12)
            base_series = _recompute_closing_cash_series(ef_12, opening0)

            # Find worst cash before raise month (inclusive)
            before_raise = base_series[base_series["month_date"] <= raise_month_ts].copy()
            min_before = float(before_raise["closing_cash_sim"].min()) if not before_raise.empty else None

            survives = (min_before is not None and min_before > 0)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Raise lands (month)", raise_month_ts.strftime("%b %Y"))
            with c2:
                st.metric("Min projected cash before raise", f"{currency_symbol}{min_before:,.0f}" if min_before is not None else "—")
            with c3:
                st.metric("Survive until raise?", "Yes ✅" if survives else "No ⚠️")

            if not survives:
                st.error(
                    "You are projected to run out of cash **before** the raise lands. "
                    "Options: raise earlier, cut burn now, pull forward collections, or arrange a bridge."
                )
            else:
                st.success("You remain cash-positive until the raise lands (based on current forecast).")

        # -------------------------------
        # Cash curve before vs after raise
        # -------------------------------
        st.markdown("---")
        st.subheader("📈 Cash curve: before vs after raise")

        if ef_12.empty:
            st.caption("No engine data to plot cash curve.")
        else:
            # Base curve
            opening0 = _derive_opening_cash_for_first_row(ef_12)
            base_series = _recompute_closing_cash_series(ef_12, opening0)

            # After-raise curve: inject raise into financing_cf in raise_month, recompute
            sim = ef_12.copy()
            if "financing_cf" in sim.columns:
                sim.loc[sim["month_date"] == raise_month_ts, "financing_cf"] = (
                    sim.loc[sim["month_date"] == raise_month_ts, "financing_cf"].fillna(0.0) + float(raise_amount)
                )
                # re-net
                if all(c in sim.columns for c in ["operating_cf", "investing_cf", "financing_cf"]):
                    sim["net_cash"] = sim["operating_cf"] + sim["investing_cf"] + sim["financing_cf"]
            else:
                # If financing_cf not present, we still can add to net_cash
                sim.loc[sim["month_date"] == raise_month_ts, "net_cash"] = (
                    sim.loc[sim["month_date"] == raise_month_ts, "net_cash"].fillna(0.0) + float(raise_amount)
                )

            after_series = _recompute_closing_cash_series(sim, opening0)

            plot_df = pd.DataFrame({
                "month_date": base_series["month_date"],
                "Base forecast": base_series["closing_cash_sim"],
                "After raise": after_series["closing_cash_sim"],
            })

            plot_long = plot_df.melt("month_date", var_name="Scenario", value_name="Closing cash")

            chart = (
                alt.Chart(plot_long)
                .mark_line()
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("Closing cash:Q", title=f"Projected closing cash ({currency_code})"),
                    color=alt.Color("Scenario:N", title="Scenario"),
                    tooltip=[
                        alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                        "Scenario:N",
                        alt.Tooltip("Closing cash:Q", title="Closing cash", format=",.0f"),
                    ],
                )
            )
            st.altair_chart(chart.interactive(), width="stretch")

            # quick delta
            base_end = float(base_series["closing_cash_sim"].iloc[-1] or 0.0)
            after_end = float(after_series["closing_cash_sim"].iloc[-1] or 0.0)
            st.caption(
                f"End-of-window cash: Base **{currency_symbol}{base_end:,.0f}** → After raise **{currency_symbol}{after_end:,.0f}**."
            )

        # -------------------------------
        # Funding recommendation strip
        # -------------------------------
        st.markdown("---")
        st.subheader("🟨 Funding recommendation")

        # Determine urgency
        recommendation = []
        sev = "green"

        # When to start raise: cliff - 90 days (configurable later)
        raise_start_by = None
        if cliff_date:
            raise_start_by = (pd.to_datetime(cliff_date) - pd.DateOffset(days=90)).date()

        # Use survive flag if available
        try:
            survives_flag = survives
        except Exception:
            survives_flag = True

        if cliff_date and not survives_flag:
            sev = "red"
            recommendation.append(
                f"Cash goes negative around **{pd.to_datetime(cliff_date).strftime('%b %Y')}** and you don’t survive until the raise month."
            )
            recommendation.append("Move raise earlier **or** implement immediate burn reduction / bridge funding.")
        elif cliff_date and current_runway < 6:
            sev = "amber"
            recommendation.append(
                f"Cash goes negative around **{pd.to_datetime(cliff_date).strftime('%b %Y')}** (runway ~{current_runway:.1f} months)."
            )
            if raise_start_by:
                recommendation.append(f"Start fundraising by **{pd.to_datetime(raise_start_by).strftime('%b %Y')}** to avoid a last-minute raise.")
        elif current_runway < 6:
            sev = "amber"
            recommendation.append(f"Runway is **{current_runway:.1f} months**. You’re not in immediate danger, but you should prepare a raise.")
            recommendation.append("Tighten burn discipline + improve collections to extend runway while you prepare.")
        else:
            sev = "green"
            recommendation.append("Runway looks healthy in the current forecast window.")
            recommendation.append("Funding is optional—use it strategically for growth, pricing power, or product acceleration.")

        recommendation.append(
            f"Recommended raise (incl. buffers): **{currency_symbol}{funding_gap:,.0f}** · Planned: **{currency_symbol}{raise_amount:,.0f}**"
        )

        msg = "  \n".join([f"- {x}" for x in recommendation])

        if sev == "red":
            st.error(msg)
        elif sev == "amber":
            st.warning(msg)
        else:
            st.success(msg)

    # -------------------------------
    # Admin expander + collaboration (unchanged)
    # -------------------------------
    st.markdown("---")
    with st.expander("📚 Detailed investing & financing entries (admin)", expanded=False):
        col_left_admin, col_right_admin = st.columns([2, 1])

        with col_left_admin:
            st.markdown("### Investing flows")
            if df_inv is None or df_inv.empty:
                st.caption("No investing flows yet.")
            else:
                inv_view = df_inv.copy()
                inv_view["month_date"] = pd.to_datetime(inv_view["month_date"], errors="coerce")
                inv_view = inv_view.sort_values("month_date")
                inv_view["Month"] = inv_view["month_date"].dt.strftime("%b %Y")
                cols = ["id", "Month"] + [c for c in ["amount", "category", "notes"] if c in inv_view.columns]
                st.dataframe(inv_view[cols], width="stretch")

            st.markdown("### Financing flows")
            if df_fin is None or df_fin.empty:
                st.caption("No financing flows yet.")
            else:
                fin_view = df_fin.copy()
                fin_view["month_date"] = pd.to_datetime(fin_view["month_date"], errors="coerce")
                fin_view = fin_view.sort_values("month_date")
                fin_view["Month"] = fin_view["month_date"].dt.strftime("%b %Y")
                cols = ["id", "Month"] + [c for c in ["amount", "category", "notes"] if c in fin_view.columns]
                st.dataframe(fin_view[cols], width="stretch")

        with col_right_admin:
            st.markdown("### ➕ Add investing flow")
            with st.form(key="new_investing_flow_form"):
                inv_date = st.date_input("Month", value=selected_month_start, key="inv_date")
                inv_amount = st.number_input("Amount", value=0.0, step=1000.0, key="inv_amount")
                inv_category = st.selectbox(
                    "Category",
                    ["Capex", "R&D / product build", "Equipment & tools", "Fitout / leasehold", "Software / intangibles", "Other"],
                    key="inv_category",
                )
                inv_notes = st.text_area("Notes", value="", key="inv_notes")
                if st.form_submit_button("Save investing flow"):
                    ok = create_investing_flow(selected_client_id, month_date=inv_date, amount=inv_amount, category=inv_category, notes=inv_notes)
                    if ok:
                        st.success("Saved. Rebuild cashflow to refresh projections.")
                        st.rerun()
                    else:
                        st.error("Could not save investing flow.")

            st.markdown("### ➕ Add financing flow")
            with st.form(key="new_financing_flow_form"):
                fin_date = st.date_input("Month", value=selected_month_start, key="fin_date")
                fin_amount = st.number_input("Amount", value=0.0, step=1000.0, key="fin_amount")
                fin_category = st.selectbox(
                    "Category",
                    ["Equity raise", "SAFE / convertible", "Loan drawdown", "Loan repayment", "Interest payment", "Dividend", "Other"],
                    key="fin_category",
                )
                fin_notes = st.text_area("Notes", value="", key="fin_notes")
                if st.form_submit_button("Save financing flow"):
                    ok = create_financing_flow(selected_client_id, month_date=fin_date, amount=fin_amount, category=fin_category, notes=fin_notes)
                    if ok:
                        st.success("Saved. Rebuild cashflow to refresh projections.")
                        st.rerun()
                    else:
                        st.error("Could not save financing flow.")

    # Collaboration layer at bottom
    st.markdown("---")
    col_comments, col_tasks = st.columns([3, 1])
    with col_comments:
        comments_block("investing_financing")
    with col_tasks:
        with st.expander("✅ Tasks for this page", expanded=False):
            tasks_block("investing_financing")



##Pipeline health panel#
def build_pipeline_health_panel(df_pipeline, focus_month, currency_symbol):
    if df_pipeline is None or df_pipeline.empty:
        return ("warning", ["No deals yet. Add your top 5–10 active deals to see pipeline health."], [])

    df = df_pipeline.copy()
    df["value_total"] = pd.to_numeric(df.get("value_total"), errors="coerce").fillna(0.0)
    df["probability_pct"] = pd.to_numeric(df.get("probability_pct"), errors="coerce").fillna(0.0)
    df["stage"] = df.get("stage", "").astype(str).str.lower()

    # closing month logic
    closing = pd.to_datetime(df.get("end_month"), errors="coerce")
    startm = pd.to_datetime(df.get("start_month"), errors="coerce")
    df["closing_month"] = closing.fillna(startm)

    focus = pd.to_datetime(focus_month).replace(day=1)
    d90 = focus + pd.DateOffset(months=3)

    open_mask = df["stage"].isin(["idea","proposal","demo","contract"])
    df_open = df[open_mask].copy()
    df_open["weighted"] = df_open["value_total"] * df_open["probability_pct"] / 100.0

    # KPIs
    weighted_total = float(df_open["weighted"].sum())
    weighted_90d = float(df_open.loc[(df_open["closing_month"] >= focus) & (df_open["closing_month"] < d90), "weighted"].sum())

    # Warnings
    decisions = []
    metrics = []
    severity = "success"

    missing_dates = int(df_open["closing_month"].isna().sum())
    if missing_dates > 0:
        severity = "warning"
        decisions.append(f"{missing_dates} open deals have no expected close month. Add dates to make forecasting meaningful.")

    early_stage_weighted = float(df_open.loc[df_open["stage"].isin(["idea","proposal"]), "weighted"].sum())
    if weighted_total > 0 and (early_stage_weighted / weighted_total) > 0.6:
        severity = "warning"
        decisions.append("Pipeline is mostly early-stage (Idea/Proposal). Push deals to Demo/Contract to de-risk the forecast.")

    if weighted_90d == 0 and weighted_total > 0:
        severity = "warning"
        decisions.append("No weighted pipeline is closing in the next ~90 days. You may face a near-term revenue gap.")

    # Concentration
    top = df_open.sort_values("weighted", ascending=False).head(2)["weighted"].sum()
    if weighted_total > 0 and (top / weighted_total) > 0.5:
        severity = "warning"
        decisions.append("Pipeline is highly concentrated in 1–2 deals. Create backup deals to reduce risk.")

    metrics.append(("Weighted pipeline (total)", f"{currency_symbol}{weighted_total:,.0f}"))
    metrics.append(("Weighted closing in ~90 days", f"{currency_symbol}{weighted_90d:,.0f}"))

    if not decisions:
        decisions.append("Pipeline health looks reasonable. Focus on closing the next best 2–3 deals and keeping dates/probabilities current.")

    return severity, decisions, metrics

def build_sales_next_actions(df_pipeline):
    if df_pipeline is None or df_pipeline.empty:
        return ["Add your top 5–10 active deals (name, value, stage, win chance, close month)."]

    df = df_pipeline.copy()
    df["stage"] = df.get("stage", "").astype(str).str.lower()

    actions = []
    if (df["stage"] == "idea").any():
        actions.append("Move **Idea → Proposal**: qualify ICP, confirm pain, book next meeting, set a close month.")
    if (df["stage"] == "proposal").any():
        actions.append("Move **Proposal → Demo**: confirm decision criteria, timeline, stakeholders; schedule demo on calendar.")
    if (df["stage"] == "demo").any():
        actions.append("Move **Demo → Contract**: identify champion, confirm procurement/legal, send pricing + implementation plan.")
    if (df["stage"] == "contract").any():
        actions.append("Close **Contract → Closed won**: push signature, confirm start date, invoice/PO, remove blockers weekly.")

    if not actions:
        actions.append("No active deals in selling stages. Add opportunities or restart outbound to build pipeline.")

    return actions[:4]

def build_pipeline_data_checks(df_pipeline):
    if df_pipeline is None or df_pipeline.empty:
        return []

    df = df_pipeline.copy()
    df["value_total"] = pd.to_numeric(df.get("value_total"), errors="coerce")
    df["probability_pct"] = pd.to_numeric(df.get("probability_pct"), errors="coerce")
    df["stage"] = df.get("stage", "").astype(str).str.lower()

    startm = pd.to_datetime(df.get("start_month"), errors="coerce")
    endm = pd.to_datetime(df.get("end_month"), errors="coerce")
    closing = endm.fillna(startm)

    issues = []
    if df["value_total"].isna().sum() > 0 or (df["value_total"].fillna(0) <= 0).sum() > 0:
        issues.append("Some deals have **$0 / blank value**. Add an estimate to make pipeline meaningful.")
    if df["probability_pct"].isna().sum() > 0:
        issues.append("Some deals have **blank win chance**. Set 10–90% for open deals.")
    if closing.isna().sum() > 0:
        issues.append("Some deals have **no expected close month**. Add close month to improve forecasting.")
    if ((df["stage"] == "closed_won") & (df["probability_pct"].fillna(0) < 90)).any():
        issues.append("Some **Closed won** deals have low probability. Set Closed won deals to 100% for consistency.")
    if ((df["stage"] == "closed_lost") & (df["probability_pct"].fillna(0) > 0)).any():
        issues.append("Some **Closed lost** deals still have probability > 0. Set to 0% to avoid confusion.")

    return issues


def page_sales_deals():
    top_header("Sales & Deals")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    # ---------- Revenue recognition settings for this business ----------
    settings = get_client_settings(selected_client_id)
    current_method = (settings.get("revenue_recognition_method") or "saas").lower()

    method_labels = {
        "saas": "SaaS subscription (spread over 12 months)",
        "milestone": "Milestone / project (milestones by month)",
        "poc": "Percentage-of-completion (spread between start & end)",
        "straight_line": "Straight-line service (start–end months)",
        "usage": "Usage-based (placeholder spreading)",
        "point_in_time": "Point-in-time goods delivery (lump on delivery month)",
    }

    # Human-facing label for storing in the pipeline `method` column
    method_display_map = {
        "saas": "SaaS subscription",
        "milestone": "Milestone project",
        "poc": "Percentage-of-completion",
        "straight_line": "Straight-line service",
        "usage": "Usage-based",
        "point_in_time": "Point-in-time goods",
    }

    method_options = list(method_labels.keys())
    default_index = (
        method_options.index(current_method)
        if current_method in method_options
        else 0
    )

    selected_method_label = st.selectbox(
        "Revenue recognition method for this business",
        options=[method_labels[m] for m in method_options],
        index=default_index,
        help=(
            "One method per client. All deals in the pipeline use this method "
            "for recognised revenue in your dashboards."
        ),
    )

    # Map label back to code
    reverse_map = {v: k for k, v in method_labels.items()}
    selected_method_code = reverse_map[selected_method_label]

    # Label stored on each deal
    current_method_display = method_display_map.get(
        selected_method_code,
        "Straight-line service",
    )

    if st.button("Save revenue recognition method"):
        ok = save_revenue_method(selected_client_id, selected_method_code)
        if ok:
            st.success("Revenue recognition method updated for this business.")
            st.rerun()
        else:
            st.error("Could not update method. Please try again.")

    st.markdown("---")

    # ---------- Load pipeline ----------
    with st.spinner("Loading your pipeline..."):
        try:
            df_pipeline = fetch_pipeline_for_client(selected_client_id)
        except Exception as e:
            st.error("Could not load pipeline from Supabase (pipeline table).")
            st.caption(f"{type(e).__name__}: {e}")
            # Fall back to empty DataFrame so the rest of the page doesn’t crash
            df_pipeline = pd.DataFrame()

   # Normalize "no data" cases
    if df_pipeline is None:
        df_pipeline = pd.DataFrame()

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # ---------- Compute simple pipeline KPIs ----------
    total_pipeline = 0.0
    weighted_pipeline = 0.0
    open_deals_count = 0

    if not df_pipeline.empty:
        df_pipeline = df_pipeline.copy()
        df_pipeline["value_total"] = pd.to_numeric(
            df_pipeline.get("value_total"), errors="coerce"
        ).fillna(0.0)
        df_pipeline["probability_pct"] = pd.to_numeric(
            df_pipeline.get("probability_pct"), errors="coerce"
        ).fillna(0.0)

        total_pipeline = float(df_pipeline["value_total"].sum())
        weighted_pipeline = float(
            (df_pipeline["value_total"] * df_pipeline["probability_pct"] / 100.0).sum()
        )
        open_stages = ["idea", "proposal", "demo", "contract"]
        open_deals_count = int(df_pipeline["stage"].astype(str).str.lower().isin(open_stages).sum())



    


    # --------------------------------
    # Main layout: LEFT (overview) / RIGHT (deal editor)
    # --------------------------------
    col_main, col_side = st.columns([2, 1])

    # ===== LEFT: Pipeline overview =====
    with col_main:
        st.subheader("📂 Pipeline overview")

        if df_pipeline.empty:
            st.info("No deals in the pipeline yet for this business.")
        else:
            # KPI row
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric(
                    "Total pipeline",
                    f"{currency_symbol}{total_pipeline:,.0f}",
                )
            with k2:
                st.metric(
                    "Weighted pipeline",
                    f"{currency_symbol}{weighted_pipeline:,.0f}",
                )
            with k3:
                st.metric("Open deals", open_deals_count)

            # ---- Define "this year" window = next 12 months from focus month ----
            horizon_start = pd.to_datetime(selected_month_start).replace(day=1)
            horizon_end = horizon_start + pd.DateOffset(months=12)

            st.caption(
                f"Views below use deals closing between "
                f"{horizon_start.strftime('%b %Y')} and "
                f"{(horizon_end - pd.DateOffset(days=1)).strftime('%b %Y')} "
                "(next 12 months window)."
            )
            #health panel#

            sev, decisions, metrics = build_pipeline_health_panel(df_pipeline, selected_month_start, currency_symbol)

          
            if metrics:
                mcols = st.columns(len(metrics))
                for i, (label, val) in enumerate(metrics):
                    mcols[i].metric(label, val)

            if sev == "error":
                st.error("\n".join([f"• {d}" for d in decisions]))
            elif sev == "warning":
                st.warning("\n".join([f"• {d}" for d in decisions]))
            else:
                st.success("\n".join([f"• {d}" for d in decisions]))


            st.markdown("### ✅ What to do this week")
            next_actions = build_sales_next_actions(df_pipeline)
            st.info("\n".join([f"• {a}" for a in next_actions]))


            issues = build_pipeline_data_checks(df_pipeline)
            if issues:
                st.warning("**Data checks**\n" + "\n".join([f"• {i}" for i in issues]))


            # ---- Pipeline waterfall (Closed to date + pipeline stages) ----
            st.markdown("#### 🎯 Pipeline waterfall – closed + weighted pipeline (this 12-month year)")

            df_stage = df_pipeline.copy()
            df_stage["stage"] = df_stage.get("stage", "").astype(str).str.lower()

            # Prefer end_month as closing month; fall back to start_month
            def parse_month(col_name):
                if col_name in df_stage.columns:
                    return pd.to_datetime(df_stage[col_name], errors="coerce")
                return pd.to_datetime(pd.Series([pd.NaT] * len(df_stage)), errors="coerce")

            end_month = parse_month("end_month")
            start_month = parse_month("start_month")
            closing_month = end_month.fillna(start_month)
            df_stage["closing_month"] = closing_month

            in_horizon = (
                (df_stage["closing_month"] >= horizon_start)
                & (df_stage["closing_month"] < horizon_end)
            )
            no_date_mask = df_stage["closing_month"].isna()
            effective_mask = in_horizon | no_date_mask

            df_stage_window = df_stage[effective_mask].copy()

            # Closed to date in this window = sum closed_won
            closed_mask = df_stage_window["stage"] == "closed_won"
            closed_to_date = float(df_stage_window.loc[closed_mask, "value_total"].sum())

            # Open pipeline stages (idea / proposal / demo / contract), weighted
            open_mask = df_stage_window["stage"].isin(
                ["idea", "proposal", "demo", "contract"]
            )
            df_open = df_stage_window[open_mask].copy()

            if df_open.empty and closed_to_date == 0:
                st.caption(
                    "No closed or active deals in the next 12-month window to show in the waterfall yet."
                )
            else:
                if not df_open.empty:
                    df_open["weighted_value"] = (
                        df_open["value_total"] * df_open["probability_pct"] / 100.0
                    )
                else:
                    df_open["weighted_value"] = 0.0

                stage_labels = {
                    "closed_to_date": "Closed to date",
                    "idea": "Idea",
                    "proposal": "Proposal",
                    "demo": "Demo",
                    "contract": "Contract",
                }

                # --- Build rows, sorting open stages ascending by amount ---
                rows = []

                # Closed bar always first (bottom of the waterfall)
                if closed_to_date != 0:
                    rows.append(
                        {
                            "key": "closed_to_date",
                            "Stage": stage_labels["closed_to_date"],
                            "amount": closed_to_date,
                        }
                    )

                # Aggregate weighted pipeline by stage
                if not df_open.empty:
                    agg_open = (
                        df_open.groupby("stage", as_index=False)["weighted_value"]
                        .sum()
                    )
                else:
                    agg_open = pd.DataFrame(columns=["stage", "weighted_value"])

                open_rows = []
                for s in ["idea", "proposal", "demo", "contract"]:
                    val = float(
                        agg_open.loc[agg_open["stage"] == s, "weighted_value"].sum()
                    )
                    if val != 0:
                        open_rows.append(
                            {
                                "key": s,
                                "Stage": stage_labels[s],
                                "amount": val,
                            }
                        )

                # Sort open stages by ascending amount so the waterfall steps up nicely
                open_rows = sorted(open_rows, key=lambda r: r["amount"])

                if rows:
                    # closed first, then the ascending open stages
                    rows = rows + open_rows
                else:
                    # edge case: no closed deals, just open stages
                    rows = open_rows

                if rows:
                    wf_df = pd.DataFrame(rows)

                    # Cumulative start/end for each stage component
                    wf_df["start"] = wf_df["amount"].cumsum() - wf_df["amount"]
                    wf_df["end"] = wf_df["amount"].cumsum()

                    # --- Final bar: Total est. sales (closed + weighted pipeline) ---
                    total_all = float(wf_df["amount"].sum())
                    total_row = {
                        "key": "total_est",
                        "Stage": "Total est. sales (12-month)",
                        "amount": total_all,
                        "start": 0.0,
                        "end": total_all,
                    }

                    wf_df_total = pd.concat(
                        [wf_df, pd.DataFrame([total_row])],
                        ignore_index=True,
                    )

                    # Base waterfall bars
                    wf_chart = (
                        alt.Chart(wf_df_total)
                        .mark_bar()
                        .encode(
                            x=alt.X("Stage:N", title="Stage"),
                            y=alt.Y(
                                "start:Q",
                                title=f"Closed + pipeline ({currency_code})",
                            ),
                            y2="end:Q",
                            tooltip=[
                                alt.Tooltip("Stage:N", title="Stage"),
                                alt.Tooltip(
                                    "amount:Q",
                                    title="Stage contribution",
                                    format=",.0f",
                                ),
                                alt.Tooltip(
                                    "end:Q",
                                    title="Cumulative total",
                                    format=",.0f",
                                ),
                            ],
                        )
                    )

                    # Label only the final total bar at the top
                    label_chart = (
                        alt.Chart(
                            wf_df_total[wf_df_total["key"] == "total_est"]
                        )
                        .mark_text(dy=-6)
                        .encode(
                            x="Stage:N",
                            y="end:Q",
                            text=alt.Text("end:Q", format=",.0f"),
                        )
                    )

                    st.altair_chart(
                        alt.layer(wf_chart, label_chart).interactive(),
                        use_container_width=True,
                    )
                    st.caption(
                        "Waterfall starts with **Closed to date**, then adds each open stage "
                        "(Idea → Proposal → Demo → Contract) in ascending size for the next 12 months. "
                        "The last bar **Total est. sales (12-month)** is the sum of closed + weighted pipeline – "
                        "the number on top of that bar is your headline sales target."
                    )
                else:
                    st.caption("No stages with non-zero amounts to display in the waterfall.")

            # ---- Pipeline table (in dropdown) ----
            display_cols = []
            for col in [
                "deal_name",
                "customer_name",
                "stage",
                "value_total",
                "probability_pct",
                "start_month",
                "end_month",
                "method",
            ]:
                if col in df_pipeline.columns:
                    display_cols.append(col)

            if display_cols:
                view = df_pipeline[display_cols].copy()
                view = view.rename(
                    columns={
                        "deal_name": "Deal",
                        "customer_name": "Customer",
                        "stage": "Stage",
                        "value_total": "Value",
                        "probability_pct": "Win chance (%)",
                        "start_month": "Start month",
                        "end_month": "End month",
                        "method": "Revenue type",
                    }
                )
                with st.expander("View full pipeline list", expanded=False):
                    st.dataframe(view, width="stretch")
            else:
                st.caption("No pipeline columns available to show in the table yet.")

    # ===== RIGHT: New / update deal =====
    with col_side:
        st.subheader("✍️ Create / update a deal")

        selected_deal_id = None
        selected_deal_row = None

        # Build dropdown of existing deals
        deal_options = []
        deal_id_map = {}
        if not df_pipeline.empty:
            for _, r in df_pipeline.iterrows():
                deal_id = r.get("id")
                name = r.get("deal_name", "Unnamed")
                customer = r.get("customer_name") or ""
                stage = r.get("stage", "")
                prob = r.get("probability_pct", 0)
                label = f"{name} – {customer} [{stage}, {prob}%]"
                deal_options.append(label)
                deal_id_map[label] = deal_id

        sel_label = st.selectbox(
            "Existing deals (optional)",
            ["(New deal)"] + deal_options,
            key="sales_deals_existing_select",
        )

        if sel_label != "(New deal)" and deal_options:
            selected_deal_id = deal_id_map[sel_label]
            selected_deal_row = df_pipeline[df_pipeline["id"] == selected_deal_id].iloc[0]

        # Helper to read values from selected row
        def get_val(col, default=None):
            if selected_deal_row is not None and col in selected_deal_row:
                return selected_deal_row[col]
            return default

        # ---- New / update form ----
        st.markdown("#### Deal details")

        with st.form(key="sales_deals_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                deal_name = st.text_input(
                    "Deal name",
                    value=str(get_val("deal_name", "")) if selected_deal_id else "",
                )
                customer_name = st.text_input(
                    "Customer (optional)",
                    value=str(get_val("customer_name", "")) if selected_deal_id else "",
                )
            with col_f2:
                value_total = st.number_input(
                    f"Total value ({currency_code})",
                    min_value=0.0,
                    step=1000.0,
                    value=float(get_val("value_total", 0.0) or 0.0),
                )
                probability_pct = st.slider(
                    "Win chance (%)",
                    min_value=0,
                    max_value=100,
                    value=int(get_val("probability_pct", 50) or 50),
                )

            col_f3, col_f4 = st.columns(2)
            with col_f3:
                stage = st.selectbox(
                    "Stage",
                    ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"],
                    index=(
                        ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"]
                        .index(str(get_val("stage", "proposal")))
                        if str(get_val("stage", "proposal")) in
                        ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"]
                        else 1
                    ),
                )
            with col_f4:
                start_month_input = st.date_input(
                    "Start / expected close month",
                    value=(
                        pd.to_datetime(get_val("start_month"))
                        if get_val("start_month", None)
                        else selected_month_start
                    ),
                )

            st.markdown(
                f"**Revenue type for this business:** {current_method_display}  \n"
                "To change this, update the revenue recognition method at the top of this page."
            )
            method = current_method_display

            # Contract length in months (only needed for some methods)
            contract_months = None
            if selected_method_code == "saas":
                contract_months = st.number_input(
                    "Contract length (months)",
                    min_value=1,
                    max_value=60,
                    value=int(
                        settings.get("saas_default_months", 12) or 12
                    ),
                    step=1,
                    help="How many months to spread the subscription over.",
                )
            elif selected_method_code in ["straight_line", "poc"]:
                contract_months = st.number_input(
                    "Project duration (months)",
                    min_value=1,
                    max_value=60,
                    value=int(
                        settings.get("project_default_months", 6) or 6
                    ),
                    step=1,
                    help="How many months to spread the project revenue over if no end date is set.",
                )

            commentary = st.text_area(
                "Notes (optional)",
                value=str(get_val("commentary", "")) if selected_deal_id else "",
            )

            if selected_deal_id:
                submit_label = "Update deal"
            else:
                submit_label = "Create deal"

            submitted = st.form_submit_button(submit_label)

            if submitted:
                if not deal_name.strip():
                    st.warning("Deal name is required.")
                else:
                    if selected_deal_id:
                        # For updates we keep using your helper that updates stage + probability
                        ok = update_pipeline_deal(
                            selected_deal_id,
                            deal_name=deal_name,
                            customer_name=customer_name,
                            value_total=value_total,
                            probability_pct=probability_pct,
                            stage=stage,
                            start_month=start_month_input,
                            end_month=None,
                            commentary=commentary,
                            contract_months=contract_months,
                        )

                        if ok:
                            st.success("Deal updated.")
                        else:
                            st.error("Could not update deal. Please try again.")
                    else:
                        # Create new Deal
                        ok = create_pipeline_deal(
                            selected_client_id,
                            deal_name=deal_name,
                            customer_name=customer_name,
                            value_total=value_total,
                            probability_pct=probability_pct,
                            method=method,  # fixed per client
                            stage=stage,
                            start_month=start_month_input,
                            end_month=None,  # no end date from the form
                            commentary=commentary,
                            contract_months=contract_months,
                        )
                        if ok:
                            st.success("Deal added to pipeline.")
                        else:
                            st.error(
                                "Could not create deal. Please check inputs or try again later."
                            )
                    st.rerun()

        # ---- Milestones (only if method == 'milestone' and we have a selected deal) ----
        if selected_method_code == "milestone" and selected_deal_id is not None:
            st.markdown("#### 🧱 Milestones for this deal")

            ms_df = fetch_milestones_for_deal(selected_client_id, selected_deal_id)

            if ms_df.empty:
                ms_df = pd.DataFrame(
                    {
                        "month_date": [selected_month_start],
                        "percent_of_contract": [100.0],
                        "milestone_name": ["Initial milestone"],
                    }
                )

            editable = ms_df[["month_date", "percent_of_contract", "milestone_name"]].copy()

            edited = st.data_editor(
                editable,
                num_rows="dynamic",
                key=f"milestones_editor_{selected_deal_id}",
                width="stretch",
            )

            st.caption(
                "Add a row per milestone. "
                "Percent of contract can total 100%, or we’ll normalise it automatically."
            )

            if st.button("Save milestones", key=f"save_milestones_{selected_deal_id}"):
                ok_ms = save_milestones_for_deal(
                    selected_client_id, selected_deal_id, edited
                )
                if ok_ms:
                    st.success(
                        "Milestones saved. Revenue schedule will update automatically."
                    )
                    st.rerun()
                else:
                    st.error(
                        "Could not save milestones. "
                        "Check dates and percentages and try again."
                    )

    # ---------- POC progress editor (only if method == 'poc') ----------
    if selected_method_code == "poc":
        st.markdown("---")
        st.subheader("🏗 Percentage-of-completion (projects)")

        if df_pipeline.empty:
            st.caption("No deals yet. Add a project deal to track % complete.")
        else:
            # Limit to non-lost deals
            poc_deals = df_pipeline[
                df_pipeline["stage"].str.lower() != "closed_lost"
            ].copy()

            if poc_deals.empty:
                st.caption("No active POC projects to update.")
            else:
                options = []
                id_map = {}
                for _, r in poc_deals.iterrows():
                    d_id = r.get("id")
                    name = r.get("deal_name", "Unnamed")
                    cust = r.get("customer_name") or ""
                    label = f"{name} – {cust} (id={d_id})"
                    options.append(label)
                    id_map[label] = d_id

                selected_poc_label = st.selectbox(
                    "Pick a project to update % complete (cumulative)",
                    ["-- Select --"] + options,
                    key="poc_deal_select",
                )

                if selected_poc_label != "-- Select --":
                    deal_id = id_map[selected_poc_label]

                    # Let the user choose which month they are updating
                    selected_month_for_poc = st.date_input(
                        "Which month is this % complete for?",
                        value=selected_month_start,
                        help="We treat this as the month-end / reporting month.",
                        key="poc_month_picker",
                    )

                    # Convert chosen date to first of that month for storage
                    month_bucket = (
                        pd.to_datetime(selected_month_for_poc)
                        .to_period("M")
                        .to_timestamp()
                        .date()
                    )

                    # Load existing % for this deal + month (if any)
                    df_poc_all = fetch_poc_progress_for_client(selected_client_id)
                    existing_pct = 0.0
                    existing_notes = ""

                    if df_poc_all is not None and not df_poc_all.empty:
                        df_poc_all = df_poc_all.copy()
                        df_poc_all["month_date"] = pd.to_datetime(
                            df_poc_all["month_date"], errors="coerce"
                        )
                        df_match = df_poc_all[
                            (df_poc_all["deal_id"] == deal_id)
                            & (df_poc_all["month_date"].dt.date == month_bucket)
                        ]
                        if not df_match.empty:
                            existing_pct = float(df_match["pct_complete"].iloc[0] or 0.0)
                            existing_notes = str(df_match["notes"].iloc[0] or "")

                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        pct_input = st.number_input(
                            "Cumulative % complete (0–100)",
                            min_value=0.0,
                            max_value=100.0,
                            step=1.0,
                            value=existing_pct,
                            key="poc_pct_input",
                        )
                    with col_p2:
                        poc_notes = st.text_area(
                            f"Notes for {month_bucket.strftime('%b %Y')} (optional)",
                            value=existing_notes,
                            key="poc_notes_input",
                        )

                    if st.button("Save % complete for this month", key="poc_save_btn"):
                        ok = upsert_poc_progress_row(
                            selected_client_id,
                            deal_id=deal_id,
                            month_date=month_bucket,
                            pct_complete=pct_input,
                            notes=poc_notes,
                        )
                        if ok:
                            st.success(
                                f"Saved {pct_input:.1f}% complete for this project "
                                f"({month_bucket.strftime('%b %Y')}). "
                                "Revenue true-up will be reflected in the schedule."
                            )
                            st.rerun()
                        else:
                            st.error("Could not save % complete. Please try again.")

    # ---------- Revenue snapshot (this month + next 3 months) ----------
    st.markdown("---")
    st.subheader("📌 Revenue snapshot (this + next 3 months)")

    snap_df = build_revenue_schedule_for_client(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if snap_df is None or snap_df.empty:
        st.caption(
            "No recognised revenue schedule yet. "
            "Add deals with a revenue type + start month to see this view."
        )
    else:
        snap_df = snap_df.copy()
        snap_df["month_date"] = pd.to_datetime(snap_df["month_date"], errors="coerce")
        snap_df = snap_df[snap_df["month_date"].notna()]

        # Decide which column holds the recognised revenue
        value_col = None
        for cand in ["recognised_revenue", "revenue_amount", "amount"]:
            if cand in snap_df.columns:
                value_col = cand
                break

        if value_col is None:
            st.caption("Revenue schedule is missing a recognised revenue column.")
        else:
            focus_start = pd.to_datetime(selected_month_start).replace(day=1)
            future_end = focus_start + pd.DateOffset(months=4)

            window = snap_df[
                (snap_df["month_date"] >= focus_start)
                & (snap_df["month_date"] < future_end)
            ].copy()

            if window.empty:
                st.caption("No recognised revenue in this 4-month window yet.")
            else:
                monthly = (
                    window.groupby("month_date", as_index=False)[value_col]
                    .sum()
                    .sort_values("month_date")
                )
                monthly["Month"] = monthly["month_date"].dt.strftime("%b %Y")

                this_row = monthly[monthly["month_date"] == focus_start]
                this_rev = float(this_row[value_col].iloc[0]) if not this_row.empty else 0.0

                next_3_rev = float(
                    monthly[monthly["month_date"] > focus_start][value_col].sum()
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(
                        "This month (recognised revenue)",
                        f"{currency_symbol}{this_rev:,.0f}",
                    )
                with c2:
                    st.metric(
                        "Next 3 months (total recognised)",
                        f"{currency_symbol}{next_3_rev:,.0f}",
                    )

                table_view = monthly[["Month", value_col]].rename(
                    columns={value_col: "Recognised revenue"}
                )
                st.dataframe(table_view, width="stretch")


    st.markdown("### 🧠 Revenue confidence (next 90 days)")
    confidence_lines = []
    if next_3_rev <= 0:
        confidence_lines.append("No recognised revenue in the next 3 months → **high risk**. Pull forward deals or reduce burn.")
    else:
        confidence_lines.append(f"Recognised revenue next 3 months: **{currency_symbol}{next_3_rev:,.0f}**.")

    # Optional: compare to payroll/burn later if you want
    st.info("\n".join([f"• {l}" for l in confidence_lines]))


    # ---------- Revenue recognition engine output ----------
    st.markdown("---")
    st.subheader("📆 Revenue recognition over time (by method)")

    rev_df = build_revenue_schedule_for_client(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if rev_df is None or rev_df.empty:
        st.caption(
            "No revenue recognition schedule yet. "
            "Add a few deals and make sure they have a revenue type + start month."
        )
    else:
        rev_df = rev_df.copy()
        rev_df["month_date"] = pd.to_datetime(rev_df["month_date"], errors="coerce")
        rev_df = rev_df[rev_df["month_date"].notna()]
        rev_df = rev_df.sort_values("month_date")
        rev_df["Month"] = rev_df["month_date"].dt.strftime("%b %Y")

        # 1) Summary by month (total)
        total_col = None
        for cand in ["revenue_amount", "recognised_revenue", "amount"]:
            if cand in rev_df.columns:
                total_col = cand
                break

        if total_col is None:
            st.caption("Revenue schedule is missing a recognised revenue column.")
        else:
            monthly_total = (
                rev_df.groupby(["month_date", "Month"], as_index=False)[total_col]
                .sum()
                .rename(columns={total_col: "Total revenue"})
            )

            # Chart – total revenue over time
            chart = (
                alt.Chart(monthly_total)
                .mark_line()
                .encode(
                    x=alt.X(
                        "month_date:T",
                        title="Month",
                        axis=alt.Axis(format="%b %Y"),
                    ),
                    y=alt.Y("Total revenue:Q", title="Recognised revenue"),
                    tooltip=[
                        alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                        alt.Tooltip("Total revenue:Q", title="Revenue", format=",.0f"),
                    ],
                )
            )
            st.altair_chart(chart.interactive(), width="stretch")

    # ---------- Bottom: comments + tasks in right expander ----------
    st.markdown("---")
    col_bottom_left, col_bottom_right = st.columns([2, 1])
    with col_bottom_right:
        with st.expander("🤝 Team space: comments & actions", expanded=False):
            comments_block("sales_deals")
            tasks_block("sales_deals")


def create_investing_flow(
    client_id,
    month_date: date,
    amount: float,
    category: str | None = None,
    notes: str | None = None,
) -> bool:
    if client_id is None or month_date is None:
        return False

    try:
        payload = {
            "client_id": str(client_id),
            "month_date": month_date.isoformat(),
            "amount": float(amount),
            "category": category or None,
            "notes": notes or None,
        }
        resp = supabase.table("investing_flows").insert(payload).execute()

        error_obj = getattr(resp, "error", None)
        if error_obj:
            print("Error creating investing_flow:", error_obj)
            return False

        if isinstance(resp, dict) and resp.get("error"):
            print("Error creating investing_flow:", resp["error"])
            return False

        return True
    except Exception as e:
        print("Exception creating investing_flow:", e)
        return False


def create_financing_flow(
    client_id,
    month_date: date,
    amount: float,
    category: str | None = None,
    notes: str | None = None,
) -> bool:
    if client_id is None or month_date is None:
        return False

    try:
        payload = {
            "client_id": str(client_id),
            "month_date": month_date.isoformat(),
            "amount": float(amount),
            "category": category or None,
            "notes": notes or None,
        }
        resp = supabase.table("financing_flows").insert(payload).execute()

        error_obj = getattr(resp, "error", None)
        if error_obj:
            print("Error creating financing_flow:", error_obj)
            return False

        if isinstance(resp, dict) and resp.get("error"):
            print("Error creating financing_flow:", resp["error"])
            return False

        return True
    except Exception as e:
        print("Exception creating financing_flow:", e)
        return False



def create_pipeline_deal(
    client_id,
    deal_name,
    customer_name,
    value_total,
    probability_pct,
    method,
    stage,
    start_month,
    end_month,
    commentary,
    contract_months=None,
):
    """
    Create a new pipeline deal for this client.

    - start_month / end_month can be date, datetime or string
    - contract_months (int) is optional, used for SaaS / straight-line / POC spreading
    """

    # Normalise dates to ISO strings if needed
    def _to_iso(d):
        if d is None:
            return None
        if hasattr(d, "isoformat"):
            return d.isoformat()
        return str(d)

    payload = {
        "client_id": str(client_id),
        "deal_name": deal_name,
        "customer_name": customer_name,
        "value_total": float(value_total) if value_total is not None else 0.0,
        "probability_pct": float(probability_pct) if probability_pct is not None else 0.0,
        "method": method,
        "stage": stage,
        "start_month": _to_iso(start_month),
        "end_month": _to_iso(end_month),
        "commentary": commentary,
    }

    # 🔹 Optional contract length in months
    if contract_months is not None:
        try:
            payload["contract_months"] = int(contract_months)
        except Exception:
            # if user typed something weird, just skip it
            pass

    try:
        # 🔴 IMPORTANT: correct table name
        supabase.table("revenue_pipeline").insert(payload).execute()

        # Clear cache so revenue schedule etc picks up the new deal
        try:
            st.cache_data.clear()
        except Exception:
            pass

        return True
    except Exception as e:
        print("Error creating deal:", e)
        return False
    
def update_pipeline_deal(
    deal_id,
    deal_name=None,
    customer_name=None,
    value_total=None,
    probability_pct=None,
    stage=None,
    start_month=None,
    end_month=None,
    commentary=None,
    method=None,
    contract_months=None,
):
    try:
        payload = {}

        def _to_month_iso(d):
            dt = pd.to_datetime(d, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.to_period("M").to_timestamp().date().isoformat()

        if deal_name is not None:
            payload["deal_name"] = str(deal_name).strip()

        if customer_name is not None:
            payload["customer_name"] = str(customer_name).strip()

        if value_total is not None:
            payload["value_total"] = float(value_total)

        if probability_pct is not None:
            payload["probability_pct"] = float(probability_pct)

        if stage is not None:
            stage_norm = str(stage).lower()
            payload["stage"] = stage_norm

            # enforce rules
            if stage_norm == "closed_won":
                payload["probability_pct"] = 100.0
            elif stage_norm == "closed_lost":
                payload["probability_pct"] = 0.0

        if start_month is not None:
            payload["start_month"] = _to_month_iso(start_month)

        if end_month is not None:
            payload["end_month"] = _to_month_iso(end_month)

        if commentary is not None:
            payload["commentary"] = str(commentary)

        if method is not None:
            payload["method"] = str(method)

        if contract_months is not None:
            payload["contract_months"] = int(contract_months)

        # Nothing to update
        if not payload:
            return True

        res = (
            supabase.table("revenue_pipeline")   # ✅ FIXED TABLE NAME
            .update(payload)
            .eq("id", deal_id)
            .execute()
        )

        return True

    except Exception as e:
        print(f"[ERROR] update_pipeline_deal failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------
# Investing & Financing flows – simple fetch helpers
# ---------------------------------------------------------

def fetch_investing_flows_for_client(client_id: str) -> pd.DataFrame | None:
    """
    Read investing_flows for this client.

    Returns:
        - None  -> Supabase connection problem
        - empty DataFrame -> call succeeded but no rows
        - non-empty DataFrame -> valid data
    """
    if not client_id:
        return pd.DataFrame()

    def _call():
        return (
            supabase
            .table("investing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )

    resp = sb_execute_with_retry(_call, label="fetch investing_flows")
    if resp is None:
        print("[ERROR] fetch_investing_flows_for_client -> connection problem (None)")
        return None

    data = getattr(resp, "data", None) if hasattr(resp, "data") else resp.get("data")
    df = pd.DataFrame(data or [])
    print(f"[DEBUG] fetch_investing_flows_for_client -> {len(df)} rows for {client_id}")
    return df


def fetch_financing_flows_for_client(client_id: str) -> pd.DataFrame | None:
    """
    Read financing_flows for this client.

    Returns:
        - None  -> Supabase connection problem
        - empty DataFrame -> call succeeded but no rows
        - non-empty DataFrame -> valid data
    """
    if not client_id:
        return pd.DataFrame()

    def _call():
        return (
            supabase
            .table("financing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )

    resp = sb_execute_with_retry(_call, label="fetch financing_flows")
    if resp is None:
        print("[ERROR] fetch_financing_flows_for_client -> connection problem (None)")
        return None

    data = getattr(resp, "data", None) if hasattr(resp, "data") else resp.get("data")
    df = pd.DataFrame(data or [])
    print(f"[DEBUG] fetch_financing_flows_for_client -> {len(df)} rows for {client_id}")
    return df


@st.cache_data(ttl=60)
def fetch_ar_ap_for_client(client_id):
    """
    Fetch AR and AP rows for the given client from ar_ap_tracker.
    Returns two DataFrames: df_ar, df_ap
    """
    if client_id is None:
        return pd.DataFrame(), pd.DataFrame()

    try:
        res = (
            supabase
            .table("ar_ap_tracker")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)

    # Format dates nicely if they exist
    for col in ["due_date", "expected_payment_date", "created_at", "updated_at", "issued_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Split into AR and AP
    if "type" in df.columns:
        df_ar = df[df["type"] == "AR"].copy()
        df_ap = df[df["type"] == "AP"].copy()
    else:
        df_ar = pd.DataFrame()
        df_ap = pd.DataFrame()

    # Get client defaults
    settings = get_client_settings(client_id)
    client_ar_days = settings["ar_default_days"]

    # For AR: compute expected_date
    if not df_ar.empty:
        def _compute_expected(row):
            issued = row.get("issued_date")
            row_days = row.get("default_receipt_days")
            try:
                if issued is not None:
                    days = row_days if row_days is not None else client_ar_days
                    return issued + timedelta(days=int(days))
            except Exception:
                pass
            return row.get("due_date")

        df_ar["expected_date"] = df_ar.apply(_compute_expected, axis=1)

        # Keep everything as proper datetimes (not Python date)
        for col in ["issued_date", "due_date", "expected_date"]:
            if col in df_ar.columns:
                df_ar[col] = pd.to_datetime(df_ar[col], errors="coerce")


    return df_ar, df_ap

def add_ar_aging(df_ar: pd.DataFrame, as_of: date, ar_default_days: int) -> pd.DataFrame:
    """
    Add AR ageing columns to df_ar:

      - expected_date (if missing, recomputed using issued_date + default days, fallback due_date)
      - days_past_expected  -> days overdue based on expected_date
      - aging_bucket        -> "0–30", "30–60", "60–90", "90+ days overdue", or "Not due / current"
      - is_over_default     -> True if days_past_expected > 0 (i.e. beyond default terms)

    Only counts invoices that are not fully paid/closed as overdue.
    """
    if df_ar is None or df_ar.empty:
        return df_ar

    df = df_ar.copy()

    # Normalise date-like columns to python date objects
    def _to_date(x):
        if pd.isna(x):
            return None
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return None

    for col in ["issued_date", "due_date", "expected_date"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_date)

    # Recompute expected_date if missing, using the same logic as your overdue function
    if "expected_date" not in df.columns or df["expected_date"].isna().all():
        def _compute_expected(row):
            issued = row.get("issued_date")
            row_days = row.get("default_receipt_days")
            try:
                if issued is not None:
                    days = row_days if row_days is not None else ar_default_days
                    return issued + timedelta(days=int(days))
            except Exception:
                pass
            # fallback
            return row.get("due_date")
        df["expected_date"] = df.apply(_compute_expected, axis=1)
    else:
        # At least normalise existing values
        df["expected_date"] = df["expected_date"].apply(_to_date)

    # Helper: determine if status is "paid" or closed
    def _is_paid(row):
        status = str(row.get("status", "")).lower()
        return status in ["paid", "closed", "settled"]

    def _days_past_expected(row):
        if _is_paid(row):
            return 0
        exp = row.get("expected_date")
        if not exp:
            return 0
        # as_of is a python date already
        return (as_of - exp).days

    df["days_past_expected"] = df.apply(_days_past_expected, axis=1)

    def _bucket(days):
        # Not overdue yet
        if days <= 0:
            return "Not due / current"
        if days <= 30:
            return "0–30 days overdue"
        if days <= 60:
            return "30–60 days overdue"
        if days <= 90:
            return "60–90 days overdue"
        return "90+ days overdue"

    df["aging_bucket"] = df["days_past_expected"].apply(_bucket)

    # Over default terms as soon as it's past expected_date
    df["is_over_default"] = df["days_past_expected"] > 0

    return df

def rebuild_cashflow_summary_for_client(client_id):
    """
    Core cashflow engine v1.

    Logic:
      - Read baseline_monthly for this client
      - For each month:
          * take closing_cash from baseline_monthly
          * compute net change vs previous month = operating_cf (for now)
          * investing_cf = 0, financing_cf = 0 (v1 placeholder)
          * free_cash_flow = operating_cf + investing_cf
          * cash_danger_flag = True if closing_cash <= 0
      - Upsert into cashflow_summary on (client_id, month_date)
    """
    if client_id is None:
        return False, "No client selected."

    df_base = fetch_baseline_monthly_for_client(client_id)
    if df_base.empty:
        return False, "No baseline_monthly data found for this client."

    # Make sure we have a month_date and closing_cash-like column
    if "month_date" not in df_base.columns:
        return False, "baseline_monthly has no month_date column after normalisation."

    # Try to find closing_cash / cash balance column
    cash_col = None
    for cand in ["closing_cash", "cash_balance", "closing_cash_balance"]:
        if cand in df_base.columns:
            cash_col = cand
            break

    if cash_col is None:
        return False, "baseline_monthly has no closing_cash / cash_balance column."

    # Sort by month
    df_base = df_base.sort_values("month_date").reset_index(drop=True)

    records = []
    prev_closing = None

    for _, row in df_base.iterrows():
        month_ts = row["month_date"]
        if pd.isna(month_ts):
            continue

        # Convert to python date
        if isinstance(month_ts, datetime):
            month_d = month_ts.date().replace(day=1)
        else:
            month_d = pd.to_datetime(month_ts).date().replace(day=1)

        raw_closing = row.get(cash_col)
        try:
            closing_cash = float(raw_closing) if raw_closing is not None else None
        except (TypeError, ValueError):
            closing_cash = None

        # Net change vs previous month (simple v1: treat as operating_CF)
        if prev_closing is None or closing_cash is None:
            operating_cf = None
        else:
            operating_cf = closing_cash - prev_closing

        prev_closing = closing_cash if closing_cash is not None else prev_closing

        investing_cf = 0.0  # placeholder – later you can feed real capex etc.
        financing_cf = 0.0  # placeholder – later you can feed raises, loans, repayments

        # Free cash flow (v1 = operating + investing, ignoring financing)
        free_cash_flow = None
        if operating_cf is not None:
            free_cash_flow = operating_cf + investing_cf

        cash_danger_flag = bool(
            (closing_cash is not None) and (closing_cash <= 0)
        )

        rec = {
            "client_id": str(client_id),
            "month_date": month_d.isoformat(),
            "operating_cf": operating_cf,
            "investing_cf": investing_cf,
            "financing_cf": financing_cf,
            "free_cash_flow": free_cash_flow,
            "closing_cash": closing_cash,
            "cash_danger_flag": cash_danger_flag,
        }
        records.append(rec)

    if not records:
        return False, "No valid months were found to write to cashflow_summary."

    try:
        # Upsert on (client_id, month_date) as per your unique constraint
        supabase.table("cashflow_summary").upsert(
            records,
            on_conflict="client_id,month_date",
        ).execute()
        return True, f"Rebuilt cashflow_summary for {len(records)} month(s)."
    except Exception as e:
        return False, f"Failed to upsert cashflow_summary: {e}"


def add_ap_aging(df_ap: pd.DataFrame, as_of: date, ap_default_days: int) -> pd.DataFrame:
    """
    Add AP ageing columns to df_ap:

      - pay_expected_date      -> expected payment date (expected_payment_date, or due_date,
                                  or issued_date + default days)
      - days_past_expected     -> days you've gone past that expected payment date
      - aging_bucket           -> "0–30", "30–60", "60–90", "90+ days overdue", or "Not due / current"
      - is_over_default        -> True if days_past_expected > 0 (i.e. you are beyond your terms)

    Only counts bills that are not fully paid/closed as overdue.
    """
    if df_ap is None or df_ap.empty:
        return df_ap

    df = df_ap.copy()

    # Normalise date-like columns to python date objects
    def _to_date(x):
        if pd.isna(x):
            return None
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        try:
                return pd.to_datetime(x).date()
        except Exception:
            return None

    for col in ["issued_date", "due_date", "expected_payment_date"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_date)

    # Compute pay_expected_date priority:
    # 1) expected_payment_date
    # 2) due_date
    # 3) issued_date + default days
    def _compute_expected(row):
        if row.get("expected_payment_date"):
            return row.get("expected_payment_date")
        if row.get("due_date"):
            return row.get("due_date")
        issued = row.get("issued_date")
        if issued:
            row_days = row.get("default_payment_days")
            try:
                days = row_days if row_days is not None else ap_default_days
                return issued + timedelta(days=int(days))
            except Exception:
                return None
        return None

    df["pay_expected_date"] = df.apply(_compute_expected, axis=1)

    # Helper: determine if bill is already paid/closed
    def _is_paid(row):
        status = str(row.get("status", "")).lower()
        return status in ["paid", "closed", "settled"]

    def _days_past_expected(row):
        if _is_paid(row):
            return 0
        exp = row.get("pay_expected_date")
        if not exp:
            return 0
        return (as_of - exp).days

    df["days_past_expected"] = df.apply(_days_past_expected, axis=1)

    def _bucket(days):
        if days <= 0:
            return "Not due / current"
        if days <= 30:
            return "0–30 days overdue"
        if days <= 60:
            return "30–60 days overdue"
        if days <= 90:
            return "60–90 days overdue"
        return "90+ days overdue"

    df["aging_bucket"] = df["days_past_expected"].apply(_bucket)

    # Over default terms as soon as it's past expected payment date
    df["is_over_default"] = df["days_past_expected"] > 0

    return df


def build_cash_commitments(df_ar: pd.DataFrame, df_ap: pd.DataFrame, limit: int = 7):
    frames = []

    if not df_ar.empty:
        tmp = df_ar.copy()
        tmp["direction"] = "Cash in"
        tmp["who"] = tmp.get("counterparty", "")
        tmp["amount"] = tmp.get("amount", 0)

        # Prefer expected_date → expected_payment_date → due_date
        if "expected_date" in tmp.columns:
            tmp["date"] = tmp["expected_date"]
        else:
            tmp["date"] = tmp.get("expected_payment_date").fillna(tmp.get("due_date"))

        frames.append(tmp[["direction", "who", "amount", "date"]])

    if not df_ap.empty:
        tmp = df_ap.copy()
        tmp["direction"] = "Cash out"
        tmp["who"] = tmp.get("counterparty", "")
        tmp["amount"] = tmp.get("amount", 0)

        tmp["date"] = tmp.get("expected_payment_date").fillna(tmp.get("due_date"))
        frames.append(tmp[["direction", "who", "amount", "date"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # 🔥 FIX: normalise 'date' to Python date objects
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date

    # Now safe to sort
    combined = combined.sort_values("date", na_position="last").head(limit)

    return combined


def save_comment(client_id, page_name: str, body: str, month_start: date):
    """
    Insert a comment row into the comments table.
    """
    if not body or not body.strip():
        return False

    data = {
        "client_id": str(client_id) if client_id else None,
        "page_name": page_name,
        "context_type": "generic",
        "context_id": None,
        "month_date": month_start.isoformat() if month_start else None,
        "author_name": "Internal CFO",  # later replace with real user
        "author_role": "advisor",
        "body": body.strip(),
    }

    try:
        supabase.table("comments").insert(data).execute()
        return True
    except Exception:
        return False

@st.cache_data(ttl=60)
def fetch_recent_comments_for_client(client_id, limit: int = 50):
    """
    Fetch recent comments for this client across all pages/months.
    """
    if client_id is None:
        return []

    try:
        res = (
            supabase
            .table("comments")
            .select("*")
            .eq("client_id", str(client_id))
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def fetch_overdue_ar_for_client(client_id, as_of: date | None = None):
    """
    Fetch overdue AR rows for this client based on expected receipt date:
      expected_date = issued_date + default_receipt_days (row-level)
                      or issued_date + ar_default_days (client-level)
                      or fallback: due_date
    """
    if client_id is None:
        return pd.DataFrame()

    if as_of is None:
        as_of = date.today()

    # Get client-level defaults
    settings = get_client_settings(client_id)
    client_ar_days = settings["ar_default_days"]

    try:
        res = (
            supabase
            .table("ar_ap_tracker")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("type", "AR")
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse dates
    for col in ["issued_date", "due_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Compute expected_date
    def _compute_expected(row):
        issued = row.get("issued_date")
        row_days = row.get("default_receipt_days")
        try:
            if issued is not None:
                days = row_days if row_days is not None else client_ar_days
                return issued + timedelta(days=int(days))
        except Exception:
            pass
        # fallback
        return row.get("due_date")

    df["expected_date"] = df.apply(_compute_expected, axis=1)

    df = df[df["expected_date"].notna()]

   
    # Make sure expected_date is a date, not full datetime
    if "expected_date" in df.columns:
        df["expected_date"] = pd.to_datetime(df["expected_date"], errors="coerce").dt.date
    else:
        return pd.DataFrame()  # nothing to check

    # Make sure as_of is also a date object
    if isinstance(as_of, pd.Timestamp):
        as_of_date = as_of.date()
    else:
        as_of_date = as_of  # already a date

    # Keep only rows with a valid expected_date and overdue
    df = df[df["expected_date"].notna()]
    df = df[df["expected_date"] < as_of_date]


    # Exclude fully paid if we have a status column
    if "status" in df.columns:
        df = df[~df["status"].str.lower().isin(["paid", "closed", "settled"])]

    return df

@st.cache_data(ttl=60)
def build_12m_pnl_and_cash_view(
    client_id,
    base_month: date,
    n_months: int = 12,
) -> pd.DataFrame:
    """
    12-month view combining:
      - recognised revenue (from revenue engine)
      - payroll cost (from payroll_positions)
      - vendor cost (from AP)
      - cash in (AR receipts)
      - cash out (AP + payroll)
      - operating CF + closing cash

    Returns one row per month with:
      month_date, revenue_pnl, payroll_pnl, vendor_pnl,
      total_cost_pnl, net_burn_pnl,
      cash_in_ar, cash_out_ap, cash_out_payroll,
      operating_cf, closing_cash, cash_danger_flag
    """

    if client_id is None or base_month is None:
        return pd.DataFrame()

    # Month index (month starts)
    month_index = pd.date_range(
        start=pd.to_datetime(base_month),
        periods=n_months,
        freq="MS",
    )
    month_set = set(month_index)

    # ------------------------------
    # 1) Recognised revenue by month
    # ------------------------------
    rev_df = build_revenue_schedule_for_client(
        client_id,
        base_month=base_month,
        n_months=n_months,
    )

    if rev_df is None or rev_df.empty:
        rev_series = pd.Series(0.0, index=month_index)
    else:
        tmp = rev_df.copy()
        tmp["month_date"] = pd.to_datetime(tmp["month_date"], errors="coerce")
        tmp = tmp[tmp["month_date"].notna()]

        # pick revenue column
        value_col = None
        for cand in ["revenue_amount", "recognised_revenue", "amount"]:
            if cand in tmp.columns:
                value_col = cand
                break

        if value_col is None:
            rev_series = pd.Series(0.0, index=month_index)
        else:
            g = (
                tmp.groupby("month_date", as_index=True)[value_col]
                .sum()
            )
            # reindex so we have all months
            rev_series = g.reindex(month_index).fillna(0.0)

    # ------------------------------
    # 2) AR / AP by month (cash view)
    # ------------------------------
    df_ar, df_ap = fetch_ar_ap_for_client(client_id)

    # ---- AR ----
    if not df_ar.empty:
        df_ar = df_ar.copy()
        df_ar["amount"] = pd.to_numeric(df_ar.get("amount"), errors="coerce").fillna(0.0)

        if "expected_date" in df_ar.columns:
            df_ar["expected_date"] = pd.to_datetime(df_ar["expected_date"], errors="coerce")
        else:
            if "due_date" in df_ar.columns:
                df_ar["expected_date"] = pd.to_datetime(df_ar["due_date"], errors="coerce")
            else:
                df_ar["expected_date"] = pd.NaT

        df_ar["bucket_month"] = (
            df_ar["expected_date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        ar_monthly = (
            df_ar.groupby("bucket_month", as_index=True)["amount"]
            .sum()
        )
        ar_series = ar_monthly.reindex(month_index).fillna(0.0)
    else:
        ar_series = pd.Series(0.0, index=month_index)

    # ---- AP ----
    if not df_ap.empty:
        df_ap = df_ap.copy()
        df_ap["amount"] = pd.to_numeric(df_ap.get("amount"), errors="coerce").fillna(0.0)

        pay_col = "expected_payment_date"
        if pay_col not in df_ap.columns:
            pay_col = "due_date" if "due_date" in df_ap.columns else None

        if pay_col is not None:
            df_ap[pay_col] = pd.to_datetime(df_ap[pay_col], errors="coerce")
            df_ap["bucket_month"] = (
                df_ap[pay_col]
                .dt.to_period("M")
                .dt.to_timestamp()
            )
        else:
            df_ap["bucket_month"] = pd.NaT

        ap_monthly = (
            df_ap.groupby("bucket_month", as_index=True)["amount"]
            .sum()
        )
        ap_series = ap_monthly.reindex(month_index).fillna(0.0)
    else:
        ap_series = pd.Series(0.0, index=month_index)

    # For P&L view, treat vendor_pnl = AP amount in that month
    vendor_pnl_series = ap_series.copy()

    # ------------------------------
    # 3) Payroll by month
    # ------------------------------
    payroll_by_month = compute_payroll_by_month(client_id, month_index)
    payroll_series = pd.Series(
        [float(payroll_by_month.get(m, 0.0)) for m in month_index],
        index=month_index,
    )

    # ------------------------------
    # 4) Opening cash and monthly loop
    # ------------------------------
    # Opening cash from KPI for base month
    opening_cash = None
    kpi_row = fetch_kpis_for_client_month(client_id, base_month)
    if kpi_row and "cash_balance" in kpi_row:
        try:
            opening_cash = float(kpi_row["cash_balance"])
        except Exception:
            opening_cash = None

    if opening_cash is None:
        opening_cash = 120_000.0  # sensible fallback

    current_cash = opening_cash
    rows = []

    for m in month_index:
        revenue_pnl = float(rev_series.get(m, 0.0))
        payroll_pnl = float(payroll_series.get(m, 0.0))
        vendor_pnl = float(vendor_pnl_series.get(m, 0.0))
        total_cost_pnl = payroll_pnl + vendor_pnl
        net_burn_pnl = total_cost_pnl - revenue_pnl  # +ve = burning cash

        cash_in_ar = float(ar_series.get(m, 0.0))
        cash_out_ap = float(ap_series.get(m, 0.0))
        cash_out_payroll = payroll_pnl  # assume paid in same month for now

        operating_cf = cash_in_ar - cash_out_ap - cash_out_payroll
        investing_cf = 0.0
        financing_cf = 0.0

        closing_cash = current_cash + operating_cf + investing_cf + financing_cf
        danger = closing_cash <= 0

        rows.append(
            {
                "client_id": str(client_id),
                "month_date": m,
                "revenue_pnl": round(revenue_pnl, 2),
                "payroll_pnl": round(payroll_pnl, 2),
                "vendor_pnl": round(vendor_pnl, 2),
                "total_cost_pnl": round(total_cost_pnl, 2),
                "net_burn_pnl": round(net_burn_pnl, 2),
                "cash_in_ar": round(cash_in_ar, 2),
                "cash_out_ap": round(cash_out_ap, 2),
                "cash_out_payroll": round(cash_out_payroll, 2),
                "operating_cf": round(operating_cf, 2),
                "investing_cf": round(investing_cf, 2),
                "financing_cf": round(financing_cf, 2),
                "closing_cash": round(closing_cash, 2),
                "cash_danger_flag": danger,
            }
        )

        current_cash = closing_cash

    out_df = pd.DataFrame(rows)
    out_df["month_date"] = pd.to_datetime(out_df["month_date"], errors="coerce")
    out_df = out_df[out_df["month_date"].notna()]
    out_df = out_df.sort_values("month_date")
    out_df["month_label"] = out_df["month_date"].dt.strftime("%b %Y")

    return out_df


def fetch_opex_for_client(client_id: str) -> pd.DataFrame:
    """
    Load operating_opex rows for a client.
    One row = one month + category + amount (cash out).
    """
    if not client_id:
        return pd.DataFrame()

    try:
        resp = (
            supabase.table("operating_opex")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )
    except Exception as e:
        print("Error fetching operating_opex:", e)
        return pd.DataFrame()

    data = getattr(resp, "data", None) if hasattr(resp, "data") else resp.get("data")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    return df


def recompute_cashflow_from_ar_ap(
    client_id,
    base_month: date,
    n_months: int = 12,
    opening_cash_hint: float | None = None,
) -> bool:
    """
    Rebuild cashflow_summary for this client using:

      - AR expected cash-in
      - AP expected cash-out
      - Payroll cash-out
      - (Optionally) Opex, Investing, Financing

    and store only the aggregated fields in cashflow_summary:
      month_date, operating_cf, investing_cf, financing_cf,
      free_cash_flow, closing_cash, cash_danger_flag, notes.
    """
    if client_id is None or base_month is None:
        return False

    try:
        # Normalise base_month to first of month
        base_month_ts = pd.to_datetime(base_month).replace(day=1)
        as_of_date = base_month_ts.date()

        # Month index for forecast horizon
        month_index = pd.date_range(start=base_month_ts, periods=n_months, freq="MS")

        # ---------- Settings + AR/AP ----------
        settings = get_client_settings(client_id)


        
        ar_days = int(settings.get("ar_default_days", 30))
        ap_days = int(settings.get("ap_default_days", 30))

        df_ar, df_ap = fetch_ar_ap_for_client(client_id)

        # ----- AR -----
        if df_ar is not None and not df_ar.empty:
            df_ar = df_ar.copy()
            df_ar = add_ar_aging(df_ar, as_of=as_of_date, ar_default_days=ar_days)

            df_ar["expected_date"] = pd.to_datetime(
                df_ar.get("expected_date"), errors="coerce"
            )
            df_ar["amount"] = pd.to_numeric(
                df_ar.get("amount"), errors="coerce"
            ).fillna(0.0)

            if "status" in df_ar.columns:
                df_ar = df_ar[
                    ~df_ar["status"].str.lower().isin(["paid", "closed", "settled"])
                ]
        else:
            df_ar = pd.DataFrame(columns=["expected_date", "amount"])

        # ----- AP -----
        if df_ap is not None and not df_ap.empty:
            df_ap = df_ap.copy()
            df_ap = add_ap_aging(df_ap, as_of=as_of_date, ap_default_days=ap_days)

            if "pay_expected_date" in df_ap.columns:
                pay_col = "pay_expected_date"
            elif "expected_payment_date" in df_ap.columns:
                pay_col = "expected_payment_date"
            else:
                pay_col = "due_date"

            df_ap[pay_col] = pd.to_datetime(df_ap.get(pay_col), errors="coerce")
            df_ap["amount"] = pd.to_numeric(
                df_ap.get("amount"), errors="coerce"
            ).fillna(0.0)

            if "status" in df_ap.columns:
                df_ap = df_ap[
                    ~df_ap["status"].str.lower().isin(["paid", "closed", "settled"])
                ]
        else:
            df_ap = pd.DataFrame(columns=["due_date", "amount"])

        # --- Opex (extra operating expenses by month, optional table) ---
        try:
            df_opex = fetch_opex_for_client(client_id)
            if df_opex is not None and not df_opex.empty:
                df_opex = df_opex.copy()
                df_opex["month_bucket"] = (
                    pd.to_datetime(df_opex["month_date"], errors="coerce")
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
            else:
                df_opex = pd.DataFrame(columns=["month_bucket", "amount"])
        except Exception:
            df_opex = pd.DataFrame(columns=["month_bucket", "amount"])

        # --- Investing & Financing moves (optional tables) ---
        try:
            df_inv = fetch_investing_flows_for_client(client_id)
        except Exception:
            df_inv = None
        try:
            df_fin = fetch_financing_flows_for_client(client_id)
        except Exception:
            df_fin = None

        inv_by_month = {}
        if df_inv is not None and not df_inv.empty:
            df_inv = df_inv.copy()
            df_inv["month_bucket"] = (
                pd.to_datetime(df_inv["month_date"], errors="coerce")
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            inv_agg = (
                df_inv.groupby("month_bucket", as_index=False)["amount"]
                .sum()
                .rename(columns={"month_bucket": "month_date", "amount": "amount"})
            )
            inv_by_month = {
                row["month_date"]: float(row["amount"] or 0.0)
                for _, row in inv_agg.iterrows()
            }

        fin_by_month = {}
        if df_fin is not None and not df_fin.empty:
            df_fin = df_fin.copy()
            df_fin["month_bucket"] = (
                pd.to_datetime(df_fin["month_date"], errors="coerce")
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            fin_agg = (
                df_fin.groupby("month_bucket", as_index=False)["amount"]
                .sum()
                .rename(columns={"month_bucket": "month_date", "amount": "amount"})
            )
            fin_by_month = {
                row["month_date"]: float(row["amount"] or 0.0)
                for _, row in fin_agg.iterrows()
            }

        # ---------- Payroll by month ----------
        payroll_by_month = compute_payroll_by_month(client_id, month_index)

                # ---------- Opening cash for the first month ----------
        if opening_cash_hint is not None:
            opening_cash_0 = float(opening_cash_hint)
        else:
            kpi_row = fetch_kpis_for_client_month(client_id, as_of_date)
            opening_cash_0 = 0.0
            if kpi_row and "cash_balance" in kpi_row:
                try:
                    opening_cash_0 = float(kpi_row["cash_balance"])
                except Exception:
                    opening_cash_0 = 0.0


        rows = []
        current_closing = opening_cash_0

        for m in month_index:
            opening_cash = current_closing

            # AR cash-in
            if not df_ar.empty:
                mask_ar = df_ar["expected_date"].dt.to_period("M") == m.to_period("M")
                cash_in_ar = float(df_ar.loc[mask_ar, "amount"].sum())
            else:
                cash_in_ar = 0.0

            # AP cash-out
            if not df_ap.empty:
                if "pay_expected_date" in df_ap.columns:
                    pay_col = "pay_expected_date"
                elif "expected_payment_date" in df_ap.columns:
                    pay_col = "expected_payment_date"
                else:
                    pay_col = "due_date"

                if pay_col in df_ap.columns:
                    mask_ap = df_ap[pay_col].dt.to_period("M") == m.to_period("M")
                    cash_out_ap = float(df_ap.loc[mask_ap, "amount"].sum())
                else:
                    cash_out_ap = 0.0
            else:
                cash_out_ap = 0.0

            # Payroll cash-out
            payroll_cash = float(payroll_by_month.get(m, 0.0))

            # Opex cash-out
            if not df_opex.empty:
                mask_ox = df_opex["month_bucket"] == m
                opex_cash = float(df_opex.loc[mask_ox, "amount"].sum())
            else:
                opex_cash = 0.0

            # Investing & Financing (positive = cash in, negative = out)
            investing_cf = float(inv_by_month.get(m, 0.0))
            financing_cf = float(fin_by_month.get(m, 0.0))

            # Operating CF: inflows - outflows
            operating_cf = cash_in_ar - cash_out_ap - payroll_cash - opex_cash

            # Free CF = operating + investing + financing
            free_cf = operating_cf + investing_cf + financing_cf

            # Roll closing cash
            closing_cash = opening_cash + free_cf

            rows.append(
                {
                    "client_id": str(client_id),
                    "month_date": m.date().isoformat(),
                    "operating_cf": round(operating_cf, 2),
                    "investing_cf": round(investing_cf, 2),
                    "financing_cf": round(financing_cf, 2),
                    "free_cash_flow": round(free_cf, 2),
                    "closing_cash": round(closing_cash, 2),
                    "cash_danger_flag": False,
                    "notes": None,
                }
            )

            current_closing = closing_cash

        if not rows:
            print("No rows computed for cashflow_summary recompute.")
            return False

        # ---------- Replace existing rows for this client (no ON CONFLICT) ----------
        try:
            supabase.table("cashflow_summary").delete().eq("client_id", str(client_id)).execute()
        except Exception as e:
            print("Warning deleting old cashflow_summary rows:", e)

        try:
            resp = supabase.table("cashflow_summary").insert(rows).execute()
        except Exception as e:
            print("Error recomputing cashflow_summary:", e)
            return False

        # If the client library returns a dict with 'error'
        if isinstance(resp, dict) and resp.get("error"):
            print("Error recomputing cashflow_summary:", resp["error"])
            return False

        print(f"Recomputed {len(rows)} rows into cashflow_summary")
        return True

    except Exception as e:
        print("Error recomputing cashflow_summary:", e)
        return False




def debug_cashflow_engine(
    client_id,
    base_month: date,
    n_months: int = 3,
    opening_cash_hint: float | None = None,
) -> pd.DataFrame:
    """
    Recompute a simple cashflow view from:
      - AR expected cash-in
      - AP expected cash-out
      - Payroll cash-out
    and compare it to what's stored in cashflow_summary.

    Returns a dataframe with:
      month_date, opening_cash_calc, cash_in_ar, cash_out_ap,
      payroll_cash, operating_cf_calc, investing_cf_engine,
      financing_cf_engine, closing_cash_calc, closing_cash_engine,
      diff_closing
    """
    if client_id is None or base_month is None:
        return pd.DataFrame()

    # Normalise base_month to first of month as Timestamp
    base_month_ts = pd.to_datetime(base_month).replace(day=1)

    # Use a plain date for AR/AP aging helpers (to avoid Timestamp - date issues)
    as_of_date = base_month_ts.date()

    # Month range for debug window
    month_index = pd.date_range(start=base_month_ts, periods=n_months, freq="MS")

    # --- Settings + AR/AP ---
    settings = get_client_settings(client_id)
    ar_days = int(settings.get("ar_default_days", 30))
    ap_days = int(settings.get("ap_default_days", 30))

    df_ar, df_ap = fetch_ar_ap_for_client(client_id)

    # ---------- AR ----------
    if df_ar is not None and not df_ar.empty:
        df_ar = df_ar.copy()

        # Use as_of_date (datetime.date) so it matches aging logic
        df_ar = add_ar_aging(df_ar, as_of=as_of_date, ar_default_days=ar_days)

        df_ar["expected_date"] = pd.to_datetime(
            df_ar.get("expected_date"), errors="coerce"
        )
        df_ar["amount"] = pd.to_numeric(
            df_ar.get("amount"), errors="coerce"
        ).fillna(0.0)

        if "status" in df_ar.columns:
            df_ar = df_ar[
                ~df_ar["status"].str.lower().isin(["paid", "closed", "settled"])
            ]
    else:
        df_ar = pd.DataFrame(columns=["expected_date", "amount"])

    # ---------- AP ----------
    if df_ap is not None and not df_ap.empty:
        df_ap = df_ap.copy()

        # Again: as_of_date as a plain date
        df_ap = add_ap_aging(df_ap, as_of=as_of_date, ap_default_days=ap_days)

        # Choose the best payment date column
        if "pay_expected_date" in df_ap.columns:
            pay_col = "pay_expected_date"
        elif "expected_payment_date" in df_ap.columns:
            pay_col = "expected_payment_date"
        else:
            pay_col = "due_date"

        df_ap[pay_col] = pd.to_datetime(df_ap.get(pay_col), errors="coerce")
        df_ap["amount"] = pd.to_numeric(
            df_ap.get("amount"), errors="coerce"
        ).fillna(0.0)

        if "status" in df_ap.columns:
            df_ap = df_ap[
                ~df_ap["status"].str.lower().isin(["paid", "closed", "settled"])
            ]
    else:
        df_ap = pd.DataFrame(columns=["due_date", "amount"])

    # --- Payroll per month ---
    payroll_by_month = compute_payroll_by_month(client_id, month_index)

    # --- Engine output from Supabase ---
    engine_df = fetch_cashflow_summary_for_client(client_id)
    if engine_df is None or engine_df.empty:
        st.warning("No rows in cashflow_summary yet – rebuild cashflow first.")
        return pd.DataFrame()

    engine_df = engine_df.copy()
    engine_df["month_date"] = pd.to_datetime(
        engine_df["month_date"], errors="coerce"
    )
    engine_df["month_date"] = (
        engine_df["month_date"].dt.to_period("M").dt.to_timestamp()
    )

    # --- Opening cash for first month ---
    if opening_cash_hint is not None:
        opening_cash_0 = float(opening_cash_hint)
    else:
        kpi_row = fetch_kpis_for_client_month(client_id, as_of_date)
        opening_cash_0 = 0.0
        if kpi_row and "cash_balance" in kpi_row:
            try:
                opening_cash_0 = float(kpi_row["cash_balance"])
            except Exception:
                opening_cash_0 = 0.0

    rows = []
    current_closing = opening_cash_0

    for m in month_index:
        opening_cash = current_closing

        # ----- AR in this month -----
        if not df_ar.empty and "expected_date" in df_ar.columns:
            mask_ar = df_ar["expected_date"].dt.to_period("M") == m.to_period("M")
            cash_in_ar = float(df_ar.loc[mask_ar, "amount"].sum())
        else:
            cash_in_ar = 0.0

        # ----- AP out this month -----
        if not df_ap.empty:
            if "pay_expected_date" in df_ap.columns:
                pay_col = "pay_expected_date"
            elif "expected_payment_date" in df_ap.columns:
                pay_col = "expected_payment_date"
            else:
                pay_col = "due_date"

            if pay_col in df_ap.columns:
                mask_ap = df_ap[pay_col].dt.to_period("M") == m.to_period("M")
                cash_out_ap = float(df_ap.loc[mask_ap, "amount"].sum())
            else:
                cash_out_ap = 0.0
        else:
            cash_out_ap = 0.0

        # ----- Payroll this month -----
        payroll_cash = float(payroll_by_month.get(m, 0.0))

        # Our operating CF definition (for debug)
        operating_cf_calc = cash_in_ar - cash_out_ap - payroll_cash

        # ----- Engine numbers for same month -----
        row_eng = engine_df[engine_df["month_date"] == m]
        if not row_eng.empty:
            investing_cf_engine = float(
                row_eng.get("investing_cf", 0.0).iloc[0] or 0.0
            )
            financing_cf_engine = float(
                row_eng.get("financing_cf", 0.0).iloc[0] or 0.0
            )
            closing_cash_engine = float(
                row_eng.get("closing_cash", 0.0).iloc[0] or 0.0
            )
        else:
            investing_cf_engine = 0.0
            financing_cf_engine = 0.0
            closing_cash_engine = float("nan")

        closing_cash_calc = (
            opening_cash + operating_cf_calc + investing_cf_engine + financing_cf_engine
        )

        rows.append(
            {
                "month_date": m,
                "opening_cash_calc": round(opening_cash, 2),
                "cash_in_ar": round(cash_in_ar, 2),
                "cash_out_ap": round(cash_out_ap, 2),
                "payroll_cash": round(payroll_cash, 2),
                "operating_cf_calc": round(operating_cf_calc, 2),
                "investing_cf_engine": round(investing_cf_engine, 2),
                "financing_cf_engine": round(financing_cf_engine, 2),
                "closing_cash_calc": round(closing_cash_calc, 2),
                "closing_cash_engine": (
                    round(closing_cash_engine, 2)
                    if pd.notna(closing_cash_engine)
                    else None
                ),
            }
        )

        current_closing = closing_cash_calc

    debug_df = pd.DataFrame(rows)
    debug_df["diff_closing"] = (
        debug_df["closing_cash_engine"] - debug_df["closing_cash_calc"]
    )

    return debug_df



def fetch_investing_flows_for_client(client_id) -> pd.DataFrame:
    if client_id is None:
        return pd.DataFrame()

    try:
        resp = (
            supabase.table("investing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date", desc=False)
            .execute()
        )

        data = getattr(resp, "data", None)
        if data is None and isinstance(resp, dict):
            data = resp.get("data", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df

    except Exception as e:
        print("Error fetching investing_flows:", e)
        return pd.DataFrame()


def fetch_financing_flows_for_client(client_id) -> pd.DataFrame:
    if client_id is None:
        return pd.DataFrame()

    try:
        resp = (
            supabase.table("financing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date", desc=False)
            .execute()
        )

        data = getattr(resp, "data", None)
        if data is None and isinstance(resp, dict):
            data = resp.get("data", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df

    except Exception as e:
        print("Error fetching financing_flows:", e)
        return pd.DataFrame()


def fetch_cash_curve_for_client(client_id, base_month: date | None, n_months: int = 12) -> pd.DataFrame:
    """
    Try to build a month-by-month cash curve for this client.

    Priority:
      1) cashflow_summary (if exists) with month_date + closing_cash
      2) baseline_monthly (if exists) with month_date + closing_cash
      3) fallback: simple synthetic curve for demo

    Returns DataFrame with columns:
      - month_date (datetime)
      - month_label (e.g. "Dec 2025")
      - closing_cash (float)
    """
    if client_id is None:
        return _build_synthetic_cash_curve(base_month, n_months)

    # Helper to filter & clean a table if present
    def _try_table(table_name: str, amount_col: str):
        try:
            res = (
                supabase
                .table(table_name)
                .select("*")
                .eq("client_id", str(client_id))
                .execute()
            )
            rows = res.data or []
        except Exception:
            rows = []

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Identify month column
        month_col = None
        for cand in ["month_date", "month", "period"]:
            if cand in df.columns:
                month_col = cand
                break
        if month_col is None or amount_col not in df.columns:
            return pd.DataFrame()

        df["month_date"] = pd.to_datetime(df[month_col], errors="coerce")
        df = df[df["month_date"].notna()]

        if base_month is not None:
            start = pd.to_datetime(base_month)
            end = start + pd.DateOffset(months=n_months)
            df = df[(df["month_date"] >= start) & (df["month_date"] < end)]

        if df.empty:
            return pd.DataFrame()

        out = df[["month_date", amount_col]].copy()
        out = out.rename(columns={amount_col: "closing_cash"})
        out = out.sort_values("month_date")
        out["month_label"] = out["month_date"].dt.strftime("%b %Y")
        return out

    # 1) Try cashflow_summary
    df = _try_table("cashflow_summary", "closing_cash")
    if not df.empty:
        return df

    # 2) Try baseline_monthly
    df = _try_table("baseline_monthly", "closing_cash")
    if not df.empty:
        return df

    # 3) Fallback synthetic curve
    return _build_synthetic_cash_curve(base_month, n_months)


def _build_synthetic_cash_curve(base_month: date | None, n_months: int = 12) -> pd.DataFrame:
    """
    Simple placeholder cash curve if DB has no data yet.
    Starts at 120k, burns 10k per month.
    """
    if base_month is None:
        base_month = date.today().replace(day=1)

    month_index = pd.date_range(
        start=pd.to_datetime(base_month),
        periods=n_months,
        freq="MS",
    )
    starting_cash = 120_000
    burn_per_month = 10_000

    cash_values = []
    for i in range(n_months):
        cash_values.append(starting_cash - burn_per_month * i)

    df = pd.DataFrame(
        {
            "month_date": month_index,
            "closing_cash": cash_values,
        }
    )
    df["month_label"] = df["month_date"].dt.strftime("%b %Y")
    return df


from datetime import date, datetime

def compute_runway_and_effective_burn_from_df(
    engine_df: pd.DataFrame,
    month_ref: date | datetime,
) -> tuple[float | None, float | None]:
    """
    Forward-looking runway + effective burn from an existing cashflow_summary DataFrame.

    - Runway:
        first month where closing_cash <= 0, counted as:
        0 = this month, 1 = next month, etc.
        If no cash-out in the horizon -> runway_months stays None.

    - Effective burn:
        3-month average of negative cashflow (operating_cf/free_cash_flow/etc),
        returned as a positive number (e.g. 25k burn per month).
    """
    if engine_df is None or engine_df.empty:
        print("[DEBUG] compute_runway_and_effective_burn_from_df -> empty/no engine_df")
        return None, None

    df = engine_df.copy()
    df["month_date"] = pd.to_datetime(df.get("month_date"), errors="coerce")
    df = df[df["month_date"].notna()].sort_values("month_date").reset_index(drop=True)
    if df.empty:
        print("[DEBUG] compute_runway_and_effective_burn_from_df -> no valid month_date rows")
        return None, None

    # Normalise reference to first of month (Timestamp)
    start_month = pd.to_datetime(month_ref).to_period("M").to_timestamp()

    # --- Find base row (focus month) ---
    mask_exact = df["month_date"] == start_month
    if mask_exact.any():
        base_idx = int(mask_exact.idxmax())
    else:
        mask_future = df["month_date"] > start_month
        base_idx = int(mask_future.idxmax()) if mask_future.any() else len(df) - 1

    df_future = df.loc[base_idx:].reset_index(drop=True)

    # ======================
    # 1) Runway from closing_cash
    # ======================
    runway_months: float | None = None

    if "closing_cash" in df_future.columns:
        cash_series = pd.to_numeric(df_future["closing_cash"], errors="coerce")
        neg_mask = cash_series.notna() & (cash_series <= 0)

        if neg_mask.any():
            first_neg_idx = int(neg_mask.idxmax())
            runway_months = float(first_neg_idx)

    # ======================
    # 2) Effective burn (3-month forward avg)
    # ======================
    effective_burn: float | None = None

    burn_source_cols = [
        "operating_cf",      # from your sample engine logs
        "free_cash_flow",
        "net_cash_flow",
        "net_cashflow",
        "total_cash_flow",
    ]
    burn_col = next((c for c in burn_source_cols if c in df.columns), None)

    if burn_col is not None:
        window = df.loc[base_idx: base_idx + 2].copy()
        vals = pd.to_numeric(window[burn_col], errors="coerce")
        avg_cf = vals.mean()
        if pd.notna(avg_cf) and avg_cf < 0:
            effective_burn = float(-avg_cf)  # positive number

    # Fallback: derive burn from slope of closing_cash
    if effective_burn is None and "closing_cash" in df.columns:
        window = df.loc[base_idx: base_idx + 2].copy()
        cc = pd.to_numeric(window["closing_cash"], errors="coerce")
        delta = cc.diff().mean()   # negative = cash falling
        if pd.notna(delta) and delta < 0:
            effective_burn = float(-delta)

    # ======================
    # 3) Fallback runway = cash_now / burn
    # ======================
    if runway_months is None and effective_burn is not None and effective_burn > 0:
        try:
            cash_now = pd.to_numeric(df_future.loc[0, "closing_cash"], errors="coerce")
            cash_now = float(cash_now) if pd.notna(cash_now) else 0.0
            runway_months = (cash_now / effective_burn) if cash_now > 0 else 0.0
        except Exception as e:
            print("[WARN] runway fallback error:", e)

    print(
        f"[DEBUG] compute_runway_and_effective_burn_from_df -> "
        f"runway={runway_months}, eff_burn={effective_burn}"
    )
    return runway_months, effective_burn



def compute_danger_month_from_cash(df: pd.DataFrame, amount_col: str = "closing_cash"):
    """
    Given a DF with month_date, month_label, and closing_cash, find the first
    month where cash is <= 0. Returns (month_label, month_date) or (None, None).
    """
    if df is None or df.empty or amount_col not in df.columns:
        return None, None

    df_sorted = df.sort_values("month_date")
    for _, row in df_sorted.iterrows():
        value = row[amount_col]
        try:
            if float(value) <= 0:
                return row.get("month_label"), row.get("month_date")
        except Exception:
            continue

    return None, None


def fetch_comments_for_client_page(client_id: str, page_name: str):
    """Return list of comments for a given client + page, newest first."""
    try:
        resp = (
            supabase.table("comments")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("page_name", page_name)
            .order("created_at", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print("Error fetching comments:", e)
        return []


def create_comment_for_client_page(
    client_id: str,
    page_name: str,
    comment_text: str,
    author_name: str = "Team member",
):
    """Insert a new comment for a given client + page."""
    try:
        data = {
            "client_id": str(client_id),
            "page_name": page_name,
            "comment_text": comment_text,
            "author_name": author_name,
            "created_at": datetime.now(datetime.UTC).isoformat(),
        }
        supabase.table("comments").insert(data).execute()
        return True
    except Exception as e:
        print("Error creating comment:", e)
        return False


# ---------- Sidebar: global controls ----------

st.sidebar.title("FFIS")

# Client selector (multi-tenant ready)
clients_data = load_clients()
client_names = [c["name"] for c in clients_data]

selected_client_name = st.sidebar.selectbox("Select business", client_names)

# Find the matching client_id
selected_client_id = None
for c in clients_data:
    if c["name"] == selected_client_name:
        selected_client_id = c["id"]
        break

# Month selector
month_options = get_month_options(18)
selected_month_label = st.sidebar.selectbox("Focus month", month_options)
selected_month_start = parse_month_label_to_date(selected_month_label)

st.sidebar.markdown("---")



def save_task(client_id, page_name: str, title: str, month_start: date):
    """
    Insert a task row into the tasks table.
    Uses existing schema: title, status, priority, owner_name, etc.
    """
    if not title or not title.strip():
        return False

    data = {
        "client_id": str(client_id) if client_id else None,
        "page_name": page_name,
        "month_date": month_start.isoformat() if month_start else None,
        "title": title.strip(),
        "status": "open",
        "priority": "normal",
        "owner_name": "Internal CFO",  # later replace with real user
    }

    try:
        supabase.table("tasks").insert(data).execute()
        return True
    except Exception:
        return False
    
def update_task_status(task_id, status: str = "done"):
    """
    Update the status of a task (e.g. mark as done).
    """
    try:
        supabase.table("tasks").update({"status": status}).eq("id", task_id).execute()
        return True
    except Exception:
        return False

# ---------------- Currency helpers ----------------

DEFAULT_CURRENCY_CODE = "AUD"

CURRENCY_SYMBOLS = {
    "AUD": "A$",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "NZD": "NZ$",
    "CAD": "C$",
}
def format_money(value, currency_symbol: str = "$", none_placeholder: str = "—") -> str:
    """
    Nicely format a numeric value as money with the right currency symbol.
    If value is None or a placeholder, returns an em dash.
    """
    if value is None or value == "—":
        return none_placeholder

    try:
        num = float(value)
    except (TypeError, ValueError):
        return none_placeholder

    return f"{currency_symbol}{num:,.0f}"


def get_client_currency(client_id: str) -> tuple[str, str]:
    """
    Return (currency_code, currency_symbol) for a client.

    Source of truth: client_settings, using one of:
    - currency_code
    - base_currency
    - currency

    Fallback = AUD.
    """
    if not client_id:
        return "AUD", "A$"

    try:
        settings = get_client_settings(client_id) or {}
        code = (
            settings.get("currency_code")
            or settings.get("base_currency")
            or settings.get("currency")
            or "AUD"
        )
        code = str(code).upper()
    except Exception as e:
        print("Error loading client currency, falling back to AUD:", e)
        code = "AUD"

    symbol_map = {
        "AUD": "A$",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "NZD": "NZ$",
        "CAD": "C$",
        "INR": "₹",
    }
    symbol = symbol_map.get(code, code + " ")

    print(f"[DEBUG] get_client_currency -> code={code}, symbol={symbol}")
    return code, symbol




def format_money_for_client(client_id: str | None, value) -> str:
    """
    Format a numeric value into a currency string for this client.
    Handles None / dash cases gracefully.
    """
    if value is None or value == "—":
        return "—"

    try:
        num = float(value)
    except Exception:
        return str(value)

    _, symbol = get_client_currency(client_id)
    return f"{symbol}{num:,.0f}"


def get_client_settings(client_id: str) -> dict:
    try:
        resp = (
            supabase.table("client_settings")
            .select("*")
            .eq("client_id", str(client_id))
            .single()
            .execute()
        )
        row = resp.data or {}
    except Exception:
        row = {}

    return {
        "ar_default_days": int(row.get("ar_default_days", 30)),
        "ap_default_days": int(row.get("ap_default_days", 30)),
        "runway_min_months": float(row.get("runway_min_months", 3.0)),
        "overspend_warn_pct": float(row.get("overspend_warn_pct", 10.0)),
        "overspend_high_pct": float(row.get("overspend_high_pct", 25.0)),
        "revenue_recognition_method": row.get("revenue_recognition_method", "instant"),
        # 👇 NEW: opening cash for engine
        "opening_cash_start": float(row.get("opening_cash_start", 0.0) or 0.0),
    }



def save_alert_thresholds(client_id, runway_min_months: float,
                          overspend_warn_pct: float, overspend_high_pct: float):
    """
    Upsert alert thresholds for this client into client_settings.
    (Keeps existing AR/AP defaults as they are.)
    """
    if client_id is None:
        return False

    try:
        current = get_client_settings(client_id)
        payload = {
            "client_id": str(client_id),
            # keep existing AR/AP defaults
            "ar_default_days": int(current["ar_default_days"]),
            "ap_default_days": int(current["ap_default_days"]),
            # update thresholds
            "runway_min_months": float(runway_min_months),
            "overspend_warn_pct": float(overspend_warn_pct),
            "overspend_high_pct": float(overspend_high_pct),
            "updated_at": datetime.utcnow().isoformat(),
        }
        supabase.table("client_settings").upsert(payload, on_conflict="client_id").execute()
         # Clear cached settings so new thresholds are picked up
        st.cache_data.clear()
        return True
    except Exception:
        return False

def save_revenue_method(client_id, method: str):
    """
    Save the default revenue recognition method for this client.
    Method is one of: 'saas', 'milestone', 'poc', 'straight_line', 'usage', 'point_in_time'.
    """
    if client_id is None:
        return False

    valid_methods = {
        "saas",
        "milestone",
        "poc",
        "straight_line",
        "usage",
        "point_in_time",
    }
    if method not in valid_methods:
        method = "saas"

    try:
        current = get_client_settings(client_id)
        payload = {
            "client_id": str(client_id),
            "ar_default_days": int(current["ar_default_days"]),
            "ap_default_days": int(current["ap_default_days"]),
            "runway_min_months": float(current["runway_min_months"]),
            "overspend_warn_pct": float(current["overspend_warn_pct"]),
            "overspend_high_pct": float(current["overspend_high_pct"]),
            # NEW
            "revenue_recognition_method": method,
            "updated_at": datetime.utcnow().isoformat(),
        }
        supabase.table("client_settings").upsert(payload, on_conflict="client_id").execute()
        st.cache_data.clear()
        return True
    except Exception:
        return False


def save_client_settings(
    client_id,
    ar_days: int | None = None,
    ap_days: int | None = None,
    **extra_fields,
):
    """
    Upsert client settings (AR/AP days + any extra fields such as currency_code).
    Works with:
        save_client_settings(client_id, 30, 45)
        save_client_settings(client_id, currency_code="USD")
    """
    if client_id is None:
        return False

    payload = {
        "client_id": str(client_id),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # Only include AR/AP if provided
    if ar_days is not None:
        payload["ar_default_days"] = int(ar_days)
    if ap_days is not None:
        payload["ap_default_days"] = int(ap_days)

    # Extra fields e.g. currency_code="USD"
    if extra_fields:
        payload.update(extra_fields)

    try:
        supabase.table("client_settings") \
            .upsert(payload, on_conflict="client_id") \
            .execute()

        # Invalidate cached settings so the UI sees latest values
        st.cache_data.clear()
        return True
    except Exception as e:
        print("Error in save_client_settings:", e)
        return False



def render_currency_settings_section(client_id: str) -> None:
    st.subheader("Currency settings")

    if not client_id:
        st.info("Select a business first.")
        return

    settings = get_client_settings(client_id) or {}

    current_code = (
        settings.get("currency_code")
        or settings.get("base_currency")
        or settings.get("currency")
        or "AUD"
    )
    current_code = str(current_code).upper()

    currency_options = ["AUD", "USD", "EUR", "GBP", "NZD", "CAD", "INR"]
    try:
        idx = currency_options.index(current_code)
    except ValueError:
        idx = 0

    selected_code = st.selectbox(
        "Base currency for this business",
        options=currency_options,
        index=idx,
        help="All KPIs and charts will use this as the display currency (no FX conversion yet).",
        key="client_currency_select",
    )

    if st.button("Save currency", key="save_client_currency_btn"):
        ok = save_client_settings(client_id, currency_code=selected_code)
        if ok:
            st.success("Currency updated for this business.")
            st.rerun()
        else:
            st.error("Could not update currency. Please try again.")



def fetch_tasks_for_page(client_id, page_name: str, month_start: date):
    """
    Fetch ALL tasks (any status) for this client + page + month.
    """
    if client_id is None or month_start is None:
        return []

    try:
        res = (
            supabase
            .table("tasks")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("page_name", page_name)
            .eq("month_date", month_start.isoformat())
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception:
        return []

def sort_alerts_by_severity(alerts: list[dict]) -> list[dict]:
    """
    Sort alerts by severity (critical > high > medium > low),
    then by created_at (newest first).
    """
    if not alerts:
        return []

    severity_order = {
        "critical": 3,
        "high": 2,
        "medium": 1,
        "low": 0,
    }

    def alert_key(a):
        sev = str(a.get("severity", "medium")).lower()
        score = severity_order.get(sev, 1)
        created = a.get("created_at") or ""
        return (-score, created)  # higher severity first, then newest

    return sorted(alerts, key=alert_key)

def ensure_overdue_ar_alert(client_id, as_of: date | None = None, min_days_overdue: int = 14):
    """
    Look at overdue AR for this client and create/resolve an 'ar_overdue' alert.

    Logic:
      - overdue based on expected_date (issued_date + default_receipt_days, fallback due_date)
      - if any rows with days_overdue > min_days_overdue -> create alert for this month
      - otherwise resolve existing alerts for this month
    """
    if client_id is None:
        return

    if as_of is None:
        as_of = date.today()

    df_overdue = fetch_overdue_ar_for_client(client_id, as_of=as_of)
    if df_overdue.empty or "expected_date" not in df_overdue.columns:
        _resolve_ar_alerts_for_month(client_id, as_of)
        return

    df_overdue = df_overdue.copy()
    # days overdue = today - expected receipt date
    df_overdue["days_overdue"] = df_overdue["expected_date"].apply(
        lambda d: (as_of - d).days if pd.notnull(d) else 0
    )

    # Keep only invoices seriously overdue
    df_overdue = df_overdue[df_overdue["days_overdue"] > min_days_overdue]

    if df_overdue.empty:
        _resolve_ar_alerts_for_month(client_id, as_of)
        return

    # Aggregate stats
    count_invoices = len(df_overdue)
    total_amount = float(df_overdue.get("amount", 0).fillna(0).sum())
    max_days_overdue = int(df_overdue["days_overdue"].max())

    # Decide severity
    if max_days_overdue > 60 or total_amount > 100_000:
        severity = "critical"
    elif max_days_overdue > 30 or total_amount > 50_000:
        severity = "high"
    else:
        severity = "medium"

    month_start = as_of.replace(day=1)

    # Avoid duplicate active alerts
    try:
        existing = (
            supabase
            .table("alerts")
            .select("id")
            .eq("client_id", str(client_id))
            .eq("alert_type", "ar_overdue")
            .eq("month_date", month_start.isoformat())
            .eq("is_active", True)
            .execute()
        )
        if existing.data:
            return
    except Exception:
        return

    msg = (
        f"{count_invoices} customer invoice(s) are overdue more than {min_days_overdue} days "
        f"past their expected date, totalling approx ${total_amount:,.0f}. "
        f"Oldest is about {max_days_overdue} days late."
    )

    data = {
        "client_id": str(client_id),
        "page_name": "cash_bills",
        "alert_type": "ar_overdue",
        "severity": severity,
        "message": msg,
        "month_date": month_start.isoformat(),
        "context_type": "ar_ap",
        "context_id": None,
        "is_active": True,
        "is_dismissed": False,
    }

    try:
        supabase.table("alerts").insert(data).execute()
    except Exception:
        pass


def _resolve_ar_alerts_for_month(client_id, as_of: date):
    """
    Internal helper: resolve existing ar_overdue alerts for the month of 'as_of'.
    """
    try:
        month_start = as_of.replace(day=1)
        supabase.table("alerts").update(
            {
                "is_active": False,
                "is_dismissed": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("client_id", str(client_id)) \
         .eq("alert_type", "ar_overdue") \
         .eq("month_date", month_start.isoformat()) \
         .eq("is_active", True) \
         .execute()
    except Exception:
        pass

def _resolve_cash_danger_alerts_for_month(client_id, month_start: date):
    """
    Internal helper: resolve existing cash_danger alerts for the given month.
    """
    if client_id is None or month_start is None:
        return

    try:
        supabase.table("alerts").update(
            {
                "is_active": False,
                "is_dismissed": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("client_id", str(client_id)) \
         .eq("alert_type", "cash_danger") \
         .eq("month_date", month_start.isoformat()) \
         .eq("is_active", True) \
         .execute()
    except Exception:
        pass


def ensure_cash_danger_alert_for_month(client_id, month_start: date):
    """
    Use cashflow_summary for this client + month to manage a 'cash_danger' alert.

      - If cash_danger_flag is TRUE or closing_cash <= 0 -> ensure a critical alert.
      - If FALSE or no row                                -> resolve any existing alert.
    """
    if client_id is None or month_start is None:
        return

    df = fetch_cashflow_summary_for_client(client_id)
    if df is None or df.empty:
        _resolve_cash_danger_alerts_for_month(client_id, month_start)
        return

    # Make sure month_date is datetime-like for .dt
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    else:
        _resolve_cash_danger_alerts_for_month(client_id, month_start)
        return

    if "cash_danger_flag" not in df.columns:
        _resolve_cash_danger_alerts_for_month(client_id, month_start)
        return

    # Compare by month (YYYY-MM)
    df["month_key"] = df["month_date"].dt.to_period("M")
    target_key = pd.to_datetime(month_start).to_period("M")

    row = df[df["month_key"] == target_key]
    if row.empty:
        _resolve_cash_danger_alerts_for_month(client_id, month_start)
        return

    row = row.iloc[0]
    flag = bool(row.get("cash_danger_flag", False))
    closing_cash = row.get("closing_cash", None)

    # Optional: treat <= 0 as danger even if flag is False
    try:
        closing_cash_val = float(closing_cash) if closing_cash is not None else None
    except Exception:
        closing_cash_val = None

    is_danger = flag or (closing_cash_val is not None and closing_cash_val <= 0)

    # If not a danger month -> resolve any existing alerts
    if not is_danger:
        _resolve_cash_danger_alerts_for_month(client_id, month_start)
        return

    # Avoid duplicate active alerts
    try:
        existing = (
            supabase
            .table("alerts")
            .select("id")
            .eq("client_id", str(client_id))
            .eq("alert_type", "cash_danger")
            .eq("month_date", month_start.isoformat())
            .eq("is_active", True)
            .execute()
        )
        if existing.data:
            return
    except Exception:
        return

    cash_str = ""
    try:
        if closing_cash_val is not None:
            cash_str = f" (projected cash ~${closing_cash_val:,.0f})"
    except Exception:
        pass

    msg = (
        f"Projected cash is at or below zero this month{cash_str}. "
        "This is your cliff month – plan funding or cuts before this point."
    )

    data = {
        "client_id": str(client_id),
        "page_name": "business_overview",
        "alert_type": "cash_danger",
        "severity": "critical",
        "message": msg,
        "month_date": month_start.isoformat(),
        "context_type": "cashflow_summary",
        "context_id": None,
        "is_active": True,
        "is_dismissed": False,
    }

    try:
        supabase.table("alerts").insert(data).execute()
    except Exception:
        pass


def ensure_runway_alert_for_month(client_id, month_start: date):
    """
    Check KPI for the given client + month and ensure we have the right runway alert:
      - If runway < client-specific runway_min_months -> create 'runway_low' alert.
      - If runway >= threshold -> resolve any existing runway_low alerts for that month.
    """
    if client_id is None or month_start is None:
        return

    # NEW: use per-client threshold
    settings = get_client_settings(client_id)
    min_months = float(settings["runway_min_months"])


    # Get the KPI row for this month
    kpis = fetch_kpis_for_client_month(client_id, month_start)
    if not kpis:
        return

    runway_value = kpis.get("runway_months")
    try:
        runway = float(runway_value) if runway_value is not None else None
    except (TypeError, ValueError):
        runway = None

    if runway is None:
        return

    # Helper: resolve (deactivate) existing runway alerts if any
    def resolve_existing_runway_alerts():
        try:
            supabase.table("alerts").update(
                {
                    "is_active": False,
                    "is_dismissed": True,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("client_id", str(client_id)) \
             .eq("alert_type", "runway_low") \
             .eq("month_date", month_start.isoformat()) \
             .eq("is_active", True) \
             .execute()
        except Exception:
            pass

    # If runway is healthy, just resolve any old alerts and return
    if runway >= min_months:
        resolve_existing_runway_alerts()
        return

    # Decide severity based on how low it is
    if runway < 2:
        severity = "critical"
    elif runway < min_months:
        severity = "high"
    else:
        severity = "medium"

    # Check if an active alert for this month already exists
    try:
        existing = (
            supabase
            .table("alerts")
            .select("id")
            .eq("client_id", str(client_id))
            .eq("alert_type", "runway_low")
            .eq("month_date", month_start.isoformat())
            .eq("is_active", True)
            .execute()
        )
        if existing.data:
            # Already have an active alert – optionally we could update severity/message, but skip for now
            return
    except Exception:
        # If we can't check, fail quietly (don't break the app)
        return

    # Build a founder-friendly message
    msg = f"Runway is around {runway:.1f} months. Consider reducing spend or planning your next raise."

    data = {
        "client_id": str(client_id),
        "page_name": "business_overview",
        "alert_type": "runway_low",
        "severity": severity,
        "message": msg,
        "month_date": month_start.isoformat(),
        "context_type": "kpi",
        "context_id": None,
        "is_active": True,
        "is_dismissed": False,
    }

    try:
        supabase.table("alerts").insert(data).execute()
    except Exception:
        # If insert fails, don't kill the UI
        pass

@st.cache_data(ttl=60)
def fetch_alerts_for_client(client_id, only_active: bool = True, limit: int = 50):
    """
    Fetch alerts for this client from alerts table.
    Returns active or all alerts sorted by newest first.
    """
    if client_id is None:
        return []

    try:
        query = (
            supabase
            .table("alerts")
            .select("*")
            .eq("client_id", str(client_id))
        )

        if only_active:
            query = query.eq("is_active", True)

        # Apply ordering + limit once
        query = query.order("created_at", desc=True).limit(limit)

        res = query.execute()
        return res.data or []

    except Exception:
        return []



def dismiss_alert(alert_id: int):
    """
    Mark an alert as dismissed (inactive) in the alerts table.
    """
    try:
        supabase.table("alerts").update(
            {
                "is_active": False,
                "is_dismissed": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", alert_id).execute()
        return True
    except Exception:
        return False

@st.cache_data(ttl=60)
def fetch_all_open_tasks(client_id):
    """
    Fetch all open tasks for this client (any page/month).
    """
    if client_id is None:
        return []

    try:
        res = (
            supabase
            .table("tasks")
            .select("*")
            .eq("client_id", str(client_id))
            .neq("status", "done")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return res.data or []
    except Exception:
        return []

# ---------- Reusable UI pieces ----------

def comments_block(page_name: str):
    """Simple shared comments UI, scoped by page_name + client."""
    st.markdown("---")
    st.subheader("💬 Team comments")

    if not selected_client_id:
        st.caption("Pick a business first to see comments.")
        return

    key_prefix = f"{selected_client_id}_{page_name}"

    # Load existing comments (page + client)
    comments = fetch_comments_for_client_page(selected_client_id, page_name)

    if not comments:
        st.caption("No comments yet. Start the discussion for this page.")
    else:
        for c in comments:
            author = c.get("author_name", "Someone on your team")
            created = c.get("created_at", "")
            text = c.get("comment_text", "")

            st.markdown(f"**{author}** · _{created}_")
            st.write(text)
            st.markdown("---")

    st.markdown("#### Add a new comment")

    new_comment = st.text_area(
        "What do you want others to know about this view?",
        key=f"comment_input_{key_prefix}",
        placeholder="E.g. 'December payroll is higher because of bonuses.'",
    )

    if st.button("Post comment", key=f"comment_btn_{key_prefix}"):
        if new_comment.strip():
            ok = create_comment_for_client_page(
                selected_client_id,
                page_name=page_name,
                comment_text=new_comment.strip(),
            )
            if ok:
                st.success("Comment posted.")
                st.rerun()
            else:
                st.error("Could not save comment. Please try again.")
        else:
            st.warning("Type something before posting.")


def tasks_block(page_name: str):
    """
    Lightweight task list per client + page.

    Expects Supabase table 'tasks' with columns:
      id, client_id, page_name, title, owner, status, priority, due_date, created_at
    """
    st.markdown("### ✅ Tasks for this page")

    if not selected_client_id:
        st.info("Select a business first to manage tasks.")
        return

    # ---- Load existing tasks ----
    try:
        resp = (
            supabase.table("tasks")
            .select("*")
            .eq("client_id", str(selected_client_id))
            .eq("page_name", page_name)
            .order("status", desc=False)
            .order("due_date", desc=False)
            .order("created_at", desc=False)
            .execute()
        )
        tasks = resp.data or []
    except Exception as e:
        st.error("Couldn't load tasks right now.")
        print("Error fetching tasks:", e)
        tasks = []

    # ---- Show tasks with quick status updates ----
    if not tasks:
        st.caption("No tasks yet for this page.")
    else:
        for t in tasks:
            tid = t.get("id")
            title = t.get("title") or "(no title)"
            owner = t.get("owner") or "Unassigned"
            status = t.get("status") or "Open"
            priority = t.get("priority") or "Medium"
            due = t.get("due_date") or ""

            row_cols = st.columns([4, 2, 2, 2, 1])
            with row_cols[0]:
                st.markdown(f"**{title}**")
                st.caption(f"👤 {owner}")
            with row_cols[1]:
                st.caption(f"Priority: **{priority}**")
            with row_cols[2]:
                st.caption(f"Due: {due or '—'}")
            with row_cols[3]:
                new_status = st.selectbox(
                    "Status",
                    options=["Open", "In progress", "Done"],
                    index=["Open", "In progress", "Done"].index(status)
                    if status in ["Open", "In progress", "Done"]
                    else 0,
                    key=f"task_status_{tid}",
                )
            with row_cols[4]:
                if st.button("💾", key=f"task_save_{tid}", help="Save status"):
                    try:
                        supabase.table("tasks").update(
                            {"status": new_status}
                        ).eq("id", tid).execute()
                        st.success("Updated.")
                        st.rerun()
                    except Exception as e:
                        st.error("Could not update task.")
                        print("Error updating task:", e)

    st.markdown("#### ➕ Add a new task")

    with st.form(key=f"task_form_{page_name}"):
        title = st.text_input("Task title")
        owner = st.text_input("Owner (name or role)", value="Founder")
        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox(
                "Priority",
                options=["Low", "Medium", "High"],
                index=1,
            )
        with col2:
            due_date = st.date_input(
                "Due date (optional)",
                value=None,
                key=f"task_due_{page_name}",
            )

        submitted = st.form_submit_button("Create task")

        if submitted:
            if not title.strip():
                st.warning("Task title is required.")
            else:
                try:
                    # date_input may return None or a date; convert to iso if non-empty
                    due_iso = None
                    if due_date:
                        try:
                            due_iso = due_date.isoformat()
                        except Exception:
                            due_iso = None

                    data = {
                        "client_id": str(selected_client_id),
                        "page_name": page_name,
                        "title": title.strip(),
                        "owner": owner.strip() or "Founder",
                        "status": "Open",
                        "priority": priority,
                        "due_date": due_iso,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    supabase.table("tasks").insert(data).execute()
                    st.success("Task created.")
                    st.rerun()
                except Exception as e:
                    st.error("Could not create task.")
                    print("Error inserting task:", e)



def top_header(title: str):
    st.title(title)
    st.caption(f"{selected_client_name} • Focus month: {selected_month_label}")

def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or val == "—":
            return default
        return float(val)
    except Exception:
        return default


def build_kpi_change_explanations(client_id, month_date: date) -> dict:
    """
    Compare this month vs previous month for KPI values and
    return:
      {
        "deltas": {
            "money_in": float | None,
            "money_out": float | None,
            "cash": float | None,
            "runway": float | None,
        },
        "explanations": {
            "money_in": str,
            "money_out": str,
            "cash": str,
            "runway": str,
        },
    }
    If previous month is missing, deltas will be None and text will say
    it's the first tracked month.
    """
    result = {
        "deltas": {
            "money_in": None,
            "money_out": None,
            "cash": None,
            "runway": None,
        },
        "explanations": {
            "money_in": "First month tracked – no prior month to compare.",
            "money_out": "First month tracked – no prior month to compare.",
            "cash": "First month tracked – no prior month to compare.",
            "runway": "First month tracked – no prior month to compare.",
        },
    }

    if client_id is None or month_date is None:
        return result

    this_month_ts = pd.to_datetime(month_date).replace(day=1)
    prev_month_ts = (this_month_ts - pd.DateOffset(months=1)).date()

    curr = fetch_kpis_for_client_month(client_id, this_month_ts.date())
    prev = fetch_kpis_for_client_month(client_id, prev_month_ts)

    if not curr or not prev:
        # No previous row – keep defaults
        return result

    # Extract numeric values
    curr_in = _safe_float(curr.get("revenue", 0.0))
    prev_in = _safe_float(prev.get("revenue", 0.0))

    curr_out = _safe_float(curr.get("burn", 0.0))
    prev_out = _safe_float(prev.get("burn", 0.0))

    curr_cash = _safe_float(curr.get("cash_balance", 0.0))
    prev_cash = _safe_float(prev.get("cash_balance", 0.0))

    curr_runway = _safe_float(curr.get("runway_months", 0.0))
    prev_runway = _safe_float(prev.get("runway_months", 0.0))

    di = curr_in - prev_in
    do = curr_out - prev_out
    dc = curr_cash - prev_cash
    dr = curr_runway - prev_runway

    result["deltas"]["money_in"] = di
    result["deltas"]["money_out"] = do
    result["deltas"]["cash"] = dc
    result["deltas"]["runway"] = dr

    # Helper to build short change sentence
    def _change_sentence(delta: float, label: str) -> str:
        if abs(delta) < 1e-6:
            return f"{label} is roughly flat vs last month."
        direction = "up" if delta > 0 else "down"
        return f"{label} is {direction} by {abs(delta):,.0f} vs last month."

    result["explanations"]["money_in"] = _change_sentence(di, "Money in")
    result["explanations"]["money_out"] = _change_sentence(do, "Money out")

    if abs(dc) < 1e-6:
        result["explanations"]["cash"] = (
            "Cash in bank is roughly flat vs last month."
        )
    else:
        direction = "higher" if dc > 0 else "lower"
        result["explanations"]["cash"] = (
            f"Cash in bank is {direction} by {abs(dc):,.0f} vs last month."
        )

    if abs(dr) < 1e-3:
        result["explanations"]["runway"] = (
            "Runway is about the same as last month."
        )
    else:
        direction = "longer" if dr > 0 else "shorter"
        result["explanations"]["runway"] = (
            f"Runway is {direction} by {abs(dr):.1f} months vs last month."
        )

    return result


def build_month_summary(
    kpis: dict | None,
    deltas: dict | None,
    alerts_for_month: list[dict] | None = None,
) -> str:
    """
    Build a short, founder-friendly summary of the month using:
      - KPIs (money in, out, cash, runway)
      - Deltas vs last month
      - Key alerts (optional)
    """
    if not kpis:
        return (
            "We don't have enough data to summarise this month yet. "
            "Once AR/AP, payroll and cash are loaded, this will turn into a "
            "simple story about what happened to your money."
        )

    money_in = float(kpis.get("revenue") or 0.0)
    money_out = float(kpis.get("burn") or 0.0)
    cash = float(kpis.get("cash_balance") or 0.0)
    runway = kpis.get("runway_months")

    d_money_in = (deltas or {}).get("money_in")
    d_money_out = (deltas or {}).get("money_out")
    d_cash = (deltas or {}).get("cash")
    d_runway = (deltas or {}).get("runway")

    parts = []

    # 1) Headline on money in / out
    if money_in == 0 and money_out == 0:
        parts.append("No meaningful cash movement recorded this month yet.")
    else:
        if money_in >= money_out:
            parts.append(
                f"You brought in about ${money_in:,.0f} and spent about ${money_out:,.0f}, "
                "so cashflow from operations was **positive** this month."
            )
        else:
            parts.append(
                f"You brought in about ${money_in:,.0f} and spent about ${money_out:,.0f}, "
                "so cashflow from operations was **negative** this month."
            )

    # 2) Changes vs last month (if we have deltas)
    change_bits = []
    if d_money_in is not None and d_money_in != 0:
        direction = "up" if d_money_in > 0 else "down"
        change_bits.append(f"Money in is {direction} by ${abs(d_money_in):,.0f} vs last month")
    if d_money_out is not None and d_money_out != 0:
        direction = "up" if d_money_out > 0 else "down"
        change_bits.append(f"Money out is {direction} by ${abs(d_money_out):,.0f}")
    if d_cash is not None and d_cash != 0:
        direction = "up" if d_cash > 0 else "down"
        change_bits.append(f"Cash in bank is {direction} by ${abs(d_cash):,.0f}")

    if change_bits:
        parts.append(" ".join(change_bits) + ".")

    # 3) Cash + runway
    if cash > 0:
        if isinstance(runway, (int, float)) and runway is not None:
            parts.append(
                f"You finished the month with about ${cash:,.0f} in the bank, "
                f"which gives roughly **{runway:.1f} months of runway** at the recent burn rate."
            )
        else:
            parts.append(
                f"You finished the month with about ${cash:,.0f} in the bank. "
                "We don't have enough history yet to compute a stable runway."
            )
    else:
        parts.append(
            "Cash in bank is effectively at or below zero. "
            "You are in a critical cash position and should treat this as urgent."
        )

    # 4) Alerts (pull just 1–2 key ones)
    if alerts_for_month:
        # Sort so highest severity first
        sev_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        alerts_sorted = sorted(
            alerts_for_month,
            key=lambda a: sev_order.get(str(a.get("severity", "medium")).lower(), 1),
            reverse=True,
        )

        top = alerts_sorted[:2]
        alert_msgs = []
        for a in top:
            msg = a.get("message", "")
            if msg:
                alert_msgs.append(msg)

        if alert_msgs:
            parts.append(
                "Top issues to pay attention to: " + " | ".join(alert_msgs)
            )

    return " ".join(parts)


def build_month_summary_insight(
    client_id: str,
    focus_month: date,
) -> str:
    """
    Generate a short, opinionated 'CFO-style' summary for the focus month.

    Uses cashflow_summary + computed runway/effective burn to answer:
      - What kind of month was this? (strong / controlled burn / tight / critical)
      - What mainly drove it? (operating vs investing vs financing)
      - What should the founder pay attention to next?
    """
    try:
        engine_df = fetch_cashflow_summary_for_client(client_id)
        if engine_df is None or engine_df.empty:
            return (
                "We don't have any cashflow engine rows yet for this business. "
                "Once AR/AP, payroll and investing/financing flows are in and you've rebuilt "
                "the cashflow, this summary will explain what drove the month."
            )

        df = engine_df.copy()
        df["month_date"] = pd.to_datetime(
            df["month_date"], errors="coerce"
        ).dt.date
        df = df.sort_values("month_date")

        focus = pd.to_datetime(focus_month).date().replace(day=1)

        # ---- This month row ----
        this_row = df[df["month_date"] == focus]
        if this_row.empty:
            return (
                "We don't yet have a cashflow row for this month. "
                "Rebuild the cashflow engine, or pick a month that has data."
            )

        closing_cash = float(this_row.get("closing_cash", 0.0).iloc[0] or 0.0)
        free_cf_this = float(this_row.get("free_cash_flow", 0.0).iloc[0] or 0.0)
        investing_cf_this = float(
            this_row.get("investing_cf", 0.0).iloc[0] or 0.0
        )
        financing_cf_this = float(
            this_row.get("financing_cf", 0.0).iloc[0] or 0.0
        )
        operating_cf_this = float(
            this_row.get("operating_cf", 0.0).iloc[0] or 0.0
        )

        # ---- Previous month (for comparison) ----
        prev_row = df[df["month_date"] < focus].tail(1)
        closing_prev = None
        free_cf_prev = None
        if not prev_row.empty:
            closing_prev = float(
                prev_row.get("closing_cash", 0.0).iloc[0] or 0.0
            )
            free_cf_prev = float(
                prev_row.get("free_cash_flow", 0.0).iloc[0] or 0.0
            )

        # ---- Runway & effective burn ----
        runway_months, effective_burn = compute_runway_and_effective_burn_from_df(
        engine_df,
        selected_month_start,
    )

        # ------------------------------------------------------------------
        # 1) Classify the month
        # ------------------------------------------------------------------
        headline_parts: list[str] = []

        # Cash movement this month
        if free_cf_this > 0:
            headline_parts.append("Cash **increased** this month.")
        elif free_cf_this < 0:
            headline_parts.append("Cash **decreased** this month.")
        else:
            headline_parts.append("Cash was **roughly flat** this month.")

        # Runway view
        if runway_months is not None:
            rm = runway_months
            if rm < 3:
                headline_parts.append(
                    f"Runway is **critical** at about {rm:.1f} months."
                )
            elif rm < 6:
                headline_parts.append(
                    f"Runway is **tight**, around {rm:.1f} months."
                )
            elif rm < 12:
                headline_parts.append(
                    f"Runway is **comfortable** at about {rm:.1f} months, "
                    "with a controlled burn."
                )
            else:
                headline_parts.append(
                    f"Runway is **strong** at roughly {rm:.1f} months."
                )
        else:
            if effective_burn and effective_burn > 0:
                headline_parts.append(
                    "You're burning cash, but there isn't enough history yet "
                    "to estimate runway."
                )
            else:
                headline_parts.append(
                    "There isn't enough history yet to estimate runway."
                )

        headline = " ".join(headline_parts)

        # ------------------------------------------------------------------
        # 2) Identify the main driver of the month
        # ------------------------------------------------------------------
        driver_parts: list[str] = []

        # Magnitudes for attribution
        total_movement = abs(free_cf_this)
        inv_abs = abs(investing_cf_this)
        fin_abs = abs(financing_cf_this)
        op_abs = abs(operating_cf_this)

        # If there's very little movement, keep this simple
        if total_movement < 1e-6:
            driver_parts.append(
                "This month’s net cash movement was very small, so there "
                "were no major cash drivers."
            )
        else:
            # Decide which component is the biggest contributor
            # (operating vs investing vs financing)
            biggest = "operating"
            biggest_val = op_abs

            if inv_abs > biggest_val:
                biggest = "investing"
                biggest_val = inv_abs
            if fin_abs > biggest_val:
                biggest = "financing"
                biggest_val = fin_abs

            share = biggest_val / total_movement if total_movement > 0 else 0.0

            if biggest == "operating":
                if share > 0.6:
                    driver_parts.append(
                        "Most of the movement came from **operating cashflow** "
                        "(revenue vs operating spend)."
                    )
                else:
                    driver_parts.append(
                        "Operating cashflow mattered, but **investing/financing "
                        "moves also played a role**."
                    )
            elif biggest == "investing":
                driver_parts.append(
                    "The main driver was **investing activity** – things like "
                    "capex, asset purchases or disposals."
                )
            else:  # financing
                driver_parts.append(
                    "The main driver was **financing activity** – equity raises, "
                    "loan drawdowns/repayments or dividends."
                )

        # Also mention whether movement is a one-off vs recurring
        if abs(investing_cf_this) > 0 and abs(investing_cf_this) > 0.3 * total_movement:
            driver_parts.append(
                "Part of this looks like a **one-off investing move** "
                "(not a recurring monthly pattern)."
            )
        if abs(financing_cf_this) > 0 and abs(financing_cf_this) > 0.3 * total_movement:
            driver_parts.append(
                "There is also a **financing impact** this month "
                "(funding or repayments)."
            )

        driver_sentence = " ".join(driver_parts)

        # ------------------------------------------------------------------
        # 3) Suggest next actions / focus areas
        # ------------------------------------------------------------------
        action_parts: list[str] = []

        # If runway is short
        if runway_months is not None and runway_months < 6:
            action_parts.append(
                "Priority should be **extending runway** – "
                "review hiring pace, contractor spend and large supplier costs."
            )
        elif runway_months is not None and runway_months >= 12:
            action_parts.append(
                "Given the healthy runway, focus can stay on **growth and ROI** "
                "rather than immediate cost cutting."
            )

        # If burn is increasing vs last month
        if free_cf_prev is not None and free_cf_prev < 0 and free_cf_this < free_cf_prev:
            action_parts.append(
                "Burn is **higher than last month** – it’s worth checking "
                "for permanent step-ups in payroll or recurring vendor spend."
            )

        # If very little movement and strong cash
        if total_movement < 1000 and closing_cash > 0:
            action_parts.append(
                "Cash movement is small relative to balance – it may be time "
                "to decide whether to **deploy excess cash** or keep it as buffer."
            )

        if not action_parts:
            # Fallback generic guidance
            action_parts.append(
                "Use this view to confirm whether the month behaved as planned – "
                "if not, start with operating spend, then any large "
                "investing or financing moves."
            )

        actions_sentence = " ".join(action_parts)

        # ------------------------------------------------------------------
        # Final combined summary (single paragraph)
        # ------------------------------------------------------------------
        return f"{headline} {driver_sentence} {actions_sentence}"

    except Exception as e:
        print("Error building month summary insight:", e)
        return (
            "We hit an issue while building this month's summary. "
            "Check that cashflow_summary has data for this client and month."
        )

def render_cash_engine_debug_tile(client_id: str, base_month: date):
    """
    Debug tile: shows how opening cash, net free cashflow and closing cash
    relate over the 12-month window currently in view.

    Helps catch double-counting or weird jumps in the engine.
    """
    if not client_id or not base_month:
        st.caption("No business selected for cash debug.")
        return

    engine_df = fetch_cashflow_summary_for_client(client_id)
    if engine_df is None or engine_df.empty:
        st.caption("No cashflow engine data yet – rebuild cashflow first.")
        return

    df = engine_df.copy()
    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df = df.dropna(subset=["month_date"])

    start = pd.to_datetime(base_month).replace(day=1)
    end = start + pd.DateOffset(months=12)

    window = df[(df["month_date"] >= start) & (df["month_date"] < end)].copy()
    window = window.sort_values("month_date")

    if window.empty:
        st.caption("No engine rows in this 12-month window yet.")
        return

    # Make sure columns exist
    for col in ["free_cash_flow", "closing_cash"]:
        if col not in window.columns:
            window[col] = 0.0

    first_row = window.iloc[0]
    last_row = window.iloc[-1]

    closing_0 = float(first_row.get("closing_cash", 0.0) or 0.0)
    free_cf_0 = float(first_row.get("free_cash_flow", 0.0) or 0.0)

    # Opening cash is implied by the first month: closing = opening + free_cf
    opening_0 = closing_0 - free_cf_0

    net_free_cf = float(window["free_cash_flow"].sum() or 0.0)
    expected_closing = opening_0 + net_free_cf

    closing_actual = float(last_row.get("closing_cash", 0.0) or 0.0)
    diff = closing_actual - expected_closing

    # Currency handling
    curr_code, curr_symbol = get_client_currency(client_id)

    try:
        fmt = format_money  # use your existing helper if present
    except NameError:
        def fmt(value, currency_symbol=curr_symbol or "$"):
            if value is None:
                return "—"
            try:
                return f"{currency_symbol}{float(value):,.0f}"
            except Exception:
                return str(value)

    st.markdown("#### 🧪 Cash engine sanity check")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Opening cash (start of window)", fmt(opening_0, curr_symbol))
    with c2:
        st.metric("Net free cash over 12 months", fmt(net_free_cf, curr_symbol))
    with c3:
        st.metric("Expected closing (open + net)", fmt(expected_closing, curr_symbol))
    with c4:
        st.metric(
            "Engine closing (last row)",
            fmt(closing_actual, curr_symbol),
            delta=fmt(diff, curr_symbol) if diff != 0 else None,
            help="Delta = engine closing minus recalculated closing.",
        )

    st.caption(
        "If the expected closing (open + net free CF) and engine closing differ a lot, "
        "it may indicate double counting or an off opening balance."
    )


def render_quick_scenario_block_for_overview():
    """
    Lightweight scenario playground for the Business Overview page.
    Uses the same engine as page_scenarios but without saved-scenario management.
    """

    if not selected_client_id or not selected_month_start:
        st.info("Pick a business and focus month at the top before running scenarios.")
        return

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    base_df = get_engine_window_for_scenarios(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if base_df is None or base_df.empty:
        st.caption(
            "No cashflow engine data yet – go to **Business at a Glance → Rebuild cashflow** first."
        )
        return

    base_df = base_df.sort_values("month_date").reset_index(drop=True)

    # Shared month selector for scenario start
    month_opts = base_df["month_date"].dt.to_period("M").drop_duplicates()
    month_label_map = {p.strftime("%b %Y"): p.to_timestamp() for p in month_opts}
    month_labels = list(month_label_map.keys())

    col_output, col_controls = st.columns([2, 1])

    # ------------------------------------
    # RIGHT: presets + levers
    # ------------------------------------
    with col_controls:
        st.markdown("#### Scenario type")

        preset = st.selectbox(
            "Quick scenario preset",
            options=[
                "Revenue push (+20%)",
                "Hire 1 key role (+$15k/month)",
                "Cut supplier/operating spend by 20%",
                "Raise $250k now",
                "Raise $250k + monthly top-up",
                "Custom mix",
            ],
        )

        controls = {
            "rev_change_pct": 0.0,
            "collections_boost_pct": 0.0,
            "extra_payroll_per_month": 0.0,
            "spend_change_pct": 0.0,
            "extra_capex_one_off": 0.0,
            "extra_capex_recurring": 0.0,
            "equity_raise": 0.0,
            "recurring_funding": 0.0,
        }

        start_month_label = st.selectbox(
            "Scenario starts from month",
            options=month_labels,
        )
        start_month_ts = month_label_map[start_month_label]

        is_custom = preset == "Custom mix"
        disabled = not is_custom

        tab_rev, tab_hire, tab_spend, tab_capex, tab_fund = st.tabs(
            [
                "Revenue & customers",
                "Hiring & team",
                "Bills & spend",
                "Capex & assets",
                "Funding & runway",
            ]
        )

        # Revenue & collections
        with tab_rev:
            rev_change_pct = st.slider(
                "Change in cash from customers (%)",
                min_value=-50,
                max_value=100,
                value=20 if preset == "Revenue push (+20%)" else 0,
                step=5,
                disabled=disabled,
            )
            collections_boost_pct = st.slider(
                "Change from collecting invoices earlier / later (%)",
                min_value=-20,
                max_value=20,
                value=0,
                step=5,
                disabled=disabled,
            )
            if is_custom:
                controls["rev_change_pct"] = rev_change_pct
                controls["collections_boost_pct"] = collections_boost_pct

        # Hiring
        with tab_hire:
            extra_headcount_cost = st.number_input(
                f"Extra monthly payroll from that month ({currency_code})",
                value=15_000.0 if preset == "Hire 1 key role (+$15k/month)" else 0.0,
                step=1000.0,
                disabled=disabled,
            )
            if is_custom:
                controls["extra_payroll_per_month"] = extra_headcount_cost

        # Spend
        with tab_spend:
            spend_change_pct = st.slider(
                "Change in supplier / operating spend (%)",
                min_value=-50,
                max_value=50,
                value=-20 if preset == "Cut supplier/operating spend by 20%" else 0,
                step=5,
                disabled=disabled,
            )
            if is_custom:
                controls["spend_change_pct"] = spend_change_pct

        # Capex (kept simple)
        with tab_capex:
            extra_capex_one_off = st.number_input(
                f"One-off capex / asset sale in start month ({currency_code})",
                value=0.0,
                step=5000.0,
                disabled=disabled,
            )
            if is_custom:
                controls["extra_capex_one_off"] = extra_capex_one_off

        # Funding
        with tab_fund:
            equity_raise = st.number_input(
                f"One-off equity / loan cash-in in start month ({currency_code})",
                value=250_000.0 if preset in ["Raise $250k now", "Raise $250k + monthly top-up"] else 0.0,
                step=25_000.0,
                disabled=disabled,
            )
            recurring_funding = st.number_input(
                f"Recurring funding each month from start ({currency_code})",
                value=10_000.0 if preset == "Raise $250k + monthly top-up" else 0.0,
                step=5000.0,
                disabled=disabled,
            )
            if is_custom:
                controls["equity_raise"] = equity_raise
                controls["recurring_funding"] = recurring_funding

        # Apply non-custom presets
        if not is_custom:
            if preset == "Revenue push (+20%)":
                controls["rev_change_pct"] = 20.0
            elif preset == "Hire 1 key role (+$15k/month)":
                controls["extra_payroll_per_month"] = 15_000.0
            elif preset == "Cut supplier/operating spend by 20%":
                controls["spend_change_pct"] = -20.0
            elif preset == "Raise $250k now":
                controls["equity_raise"] = 250_000.0
            elif preset == "Raise $250k + monthly top-up":
                controls["equity_raise"] = 250_000.0
                controls["recurring_funding"] = 10_000.0

        run_clicked = st.button("Run this scenario", key="bo_run_scenario")

    # ------------------------------------
    # LEFT: simple outputs
    # ------------------------------------
    with col_output:
        if not run_clicked:
            st.info("Choose a preset on the right and click **Run this scenario**.")
            return

        if not any(abs(v) > 0 for v in controls.values()):
            st.info("Nothing changed yet – try a preset or adjust the sliders.")
            return

        scen_df, (base_label, base_date), (scen_label, scen_date) = run_founder_scenario(
            base_df=base_df,
            start_month=start_month_ts,
            controls=controls,
        )

        st.caption(f"Scenario run: **{preset} starting {start_month_label}**")

        st.markdown("#### 📌 Runway & danger month")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Baseline danger month", base_label or "None in 12 months")
        with col_m2:
            delta_txt = None
            if base_date is not None and scen_date is not None:
                month_diff = (scen_date.to_period("M") - base_date.to_period("M")).n
                if month_diff != 0:
                    sign_word = "later" if month_diff > 0 else "earlier"
                    delta_txt = f"{abs(month_diff)} month(s) {sign_word}"
            st.metric(
                "Scenario danger month",
                scen_label or "None in 12 months",
                delta=delta_txt,
            )

        # Plain-language summary
        st.markdown("#### 🧠 What this means in plain language")

        min_base = float(base_df["closing_cash"].min())
        min_scen = float(scen_df["closing_cash_scenario"].min())
        diff_min = min_scen - min_base

        lines = []
        if diff_min > 0:
            lines.append(
                f"- Your **lowest cash balance improves by {currency_symbol}{diff_min:,.0f}**."
            )
        elif diff_min < 0:
            lines.append(
                f"- Your **lowest cash balance drops by {currency_symbol}{abs(diff_min):,.0f}**."
            )

        if base_label and scen_label and base_label != scen_label:
            lines.append(
                f"- The month where cash gets tight moves from **{base_label} → {scen_label}**."
            )

        if controls.get("extra_payroll_per_month", 0.0):
            lines.append("- This includes extra monthly payroll for new hires.")

        if controls.get("equity_raise", 0.0) or controls.get("recurring_funding", 0.0):
            lines.append("- This includes extra cash in from funding (equity / loans / top-ups).")

        if controls.get("spend_change_pct", 0.0) < 0:
            lines.append("- You are **cutting operating spend**, which improves runway.")
        elif controls.get("spend_change_pct", 0.0) > 0:
            lines.append("- You are **increasing operating spend**, which reduces runway.")

        if not lines:
            lines.append("- This scenario only makes a small change in the next 12 months.")

        st.markdown("\n".join(lines))

        # Simple cash curve chart
        st.markdown("---")
        st.markdown(f"#### 📉 Cash curve – baseline vs scenario ({currency_code})")

        chart_df = pd.DataFrame(
            {
                "month_date": scen_df["month_date"],
                "Baseline closing cash": base_df["closing_cash"].values,
                "Scenario closing cash": scen_df["closing_cash_scenario"].values,
            }
        )
        chart_long = chart_df.melt(
            id_vars="month_date", var_name="Series", value_name="Cash"
        )

        cash_chart = (
            alt.Chart(chart_long)
            .mark_line()
            .encode(
                x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("Cash:Q", title=f"Closing cash ({currency_code})"),
                color=alt.Color("Series:N", title=""),
                tooltip=[
                    alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                    "Series:N",
                    alt.Tooltip("Cash:Q", title="Cash", format=",.0f"),
                ],
            )
        )

        st.altair_chart(cash_chart.interactive(), width="stretch")



def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_health_verdict(
    runway_months: float | None,
    cash_balance: float | None,
    money_in: float | None,
    money_out: float | None,
    active_alerts: list | None,
):
    """
    Returns: (status, headline, message)
    status: "good" | "warn" | "risk"
    """
    runway = _safe_float(runway_months, default=None) if runway_months is not None else None
    cash = _safe_float(cash_balance, default=None) if cash_balance is not None else None
    rev = _safe_float(money_in, 0.0)
    burn = _safe_float(money_out, 0.0)

    # alert severity count
    critical = 0
    high = 0
    if active_alerts:
        for a in active_alerts:
            sev = str(a.get("severity", "")).lower()
            if sev == "critical":
                critical += 1
            elif sev == "high":
                high += 1

    # Basic health logic (simple + founder-friendly)
    if runway is not None and runway <= 2.0:
        status = "risk"
        headline = "🔴 Business health: At risk"
        msg = "Runway is very tight. You should act immediately on cash preservation and start funding prep now."
        return status, headline, msg

    if runway is not None and runway <= 4.0:
        status = "warn"
        headline = "🟡 Business health: Tight runway"
        msg = "Cash is stable short-term, but runway is tight. Tighten costs/collections and plan funding early."
        # If also many severe alerts, escalate tone
        if critical > 0 or high >= 2:
            msg = "Runway is tight and risk signals are elevated. Prioritise cash control this month and prepare funding."
        return status, headline, msg

    # If runway looks fine, but burn > revenue (or revenue is zero), still caution
    if burn > 0 and rev >= 0 and burn > rev and (runway is None or runway <= 8.0):
        status = "warn"
        headline = "🟡 Business health: Watching burn"
        msg = "Runway is okay for now, but spend is outpacing inflows. Keep burn controlled and improve collections."
        return status, headline, msg

    status = "good"
    headline = "🟢 Business health: Stable"
    msg = "Cash position looks stable in the near term. Focus on efficient growth and keep an eye on upcoming commitments."
    return status, headline, msg


def _compute_recovery_vs_funding_signal(
    runway_months: float | None,
    money_in: float | None,
    money_out: float | None,
):
    """
    Returns: (mode, message)
    mode: "recover" | "mixed" | "funding"
    """
    runway = _safe_float(runway_months, default=None) if runway_months is not None else None
    rev = _safe_float(money_in, 0.0)
    burn = _safe_float(money_out, 0.0)

    # Heuristics: founders need a clear fork
    if runway is not None and runway <= 3.0:
        return "funding", "Funding likely required unless you can cut burn fast. Start funding prep now (2–3 months lead time)."

    if runway is not None and runway <= 6.0 and burn > rev:
        return "mixed", "You may recover operationally (reduce burn / speed up collections), but also start light funding prep as a backup."

    return "recover", "You likely can recover through operating levers (collections, costs, pricing) without immediate funding—monitor runway monthly."


def page_business_overview():
    top_header("Business at a Glance")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    # --- Client settings (for opening cash, thresholds, etc.) ---
    settings = get_client_settings(selected_client_id)

    # --- Currency for this client ---
    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # Pull KPIs from Supabase
    kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up runway alerts for this month (now reads client settings)
    ensure_runway_alert_for_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up cash danger alert based on cashflow_summary
    ensure_cash_danger_alert_for_month(selected_client_id, selected_month_start)

    # ---------- Runway & effective burn from engine ----------
    
    engine_df = fetch_cashflow_summary_for_client(selected_client_id)
    runway_from_engine, effective_burn = compute_runway_and_effective_burn_from_df(
        engine_df=engine_df,
        month_ref=selected_month_start,
    )

    # ---------- Compute KPI deltas & explanations vs last month ----------
    kpi_change = build_kpi_change_explanations(
        selected_client_id,
        selected_month_start,
    )
    deltas = kpi_change["deltas"]
    expl = kpi_change["explanations"]

    # Defaults for display
    money_in_val = "—"
    money_out_val = "—"
    cash_in_bank_val = "—"
    runway_months_val = "—"

    money_in_delta_label = None
    money_out_delta_label = None
    cash_delta_label = None
    runway_delta_label = None

    if kpis:
        # Base values from kpi_monthly row
        money_in_val = kpis.get("revenue", "—")
        money_out_val = kpis.get("burn", "—")
        cash_in_bank_val = kpis.get("cash_balance", "—")
        runway_months_val = kpis.get("runway_months", "—")

        # Deltas (for st.metric)
        if deltas["money_in"] is not None:
            money_in_delta_label = f"{deltas['money_in']:+,.0f}"
        if deltas["money_out"] is not None:
            money_out_delta_label = f"{deltas['money_out']:+,.0f}"
        if deltas["cash"] is not None:
            cash_delta_label = f"{deltas['cash']:+,.0f}"
        if deltas["runway"] is not None:
            runway_delta_label = f"{deltas['runway']:+.1f} mo"

    # If we have a better runway from the engine, show that
    if runway_from_engine is not None:
        runway_months_val = f"{runway_from_engine:.1f}"

    # ---------- Top KPI row with "why it changed" under each ----------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(
            f"💰 Money in ({currency_code})",
            format_money(money_in_val, currency_symbol),
            delta=money_in_delta_label,
            help=f"Total cash in for the month (in {currency_code})",
        )
        st.caption(expl["money_in"])
    with kpi_col2:
        st.metric(
            f"💸 Money out ({currency_code})",
            format_money(money_out_val, currency_symbol),
            delta=money_out_delta_label,
            help=f"Total cash out for the month (in {currency_code})",
        )
        st.caption(expl["money_out"])
    with kpi_col3:
        st.metric(
            f"🏦 Cash in bank ({currency_code})",
            format_money(cash_in_bank_val, currency_symbol),
            delta=cash_delta_label,
        )
        st.caption(expl["cash"])
    with kpi_col4:
        st.metric(
            "🧯 Runway (months)",
            "—" if runway_months_val in (None, "—") else f"{float(runway_months_val):.1f}",
            delta=runway_delta_label,
        )
        st.caption(expl["runway"])

        # ---------- Health verdict (explicit) ----------
    # Use engine runway if available, else KPI runway
    runway_num = None
    try:
        runway_num = float(runway_months_val) if runway_months_val not in (None, "—") else None
    except Exception:
        runway_num = None

    # Pull numeric cash/rev/burn for verdict
    cash_num = None
    if kpis:
        cash_num = _safe_float(kpis.get("cash_balance"), default=None)

    rev_num = _safe_float(money_in_val, 0.0) if money_in_val != "—" else 0.0
    burn_num = _safe_float(money_out_val, 0.0) if money_out_val != "—" else 0.0

    # Use alerts list (already fetched later) -> quick fetch here for verdict
    active_alerts = fetch_alerts_for_client(selected_client_id, only_active=True, limit=50) or []

    status, headline, msg = _compute_health_verdict(
        runway_months=runway_num,
        cash_balance=cash_num,
        money_in=rev_num,
        money_out=burn_num,
        active_alerts=active_alerts,
    )

    if status == "risk":
        st.error(f"{headline}\n\n{msg}")
    elif status == "warn":
        st.warning(f"{headline}\n\n{msg}")
    else:
        st.success(f"{headline}\n\n{msg}")

    # ---------- Key alerts ----------
    st.markdown("---")
    st.subheader("🚨 Key alerts for this month")

    all_alerts = fetch_alerts_for_client(selected_client_id, only_active=True, limit=50)
    if not all_alerts:
        st.caption("No critical alerts right now. Keep doing what you're doing. 🙌")
    else:
        alerts_for_month = []
        for a in all_alerts:
            month_value = a.get("month_date")
            if month_value and selected_month_start:
                try:
                    mdt = pd.to_datetime(month_value).date().replace(day=1)
                    if mdt == selected_month_start:
                        alerts_for_month.append(a)
                except Exception:
                    alerts_for_month.append(a)
            else:
                alerts_for_month.append(a)

        if not alerts_for_month:
            alerts_for_month = all_alerts

        alerts_sorted = sort_alerts_by_severity(alerts_for_month)
        top_alerts = alerts_sorted[:3]

        for alert in top_alerts:
            sev = str(alert.get("severity", "medium")).lower()
            msg = alert.get("message", "")
            atype = alert.get("alert_type", "alert")
            label = f"[{atype}] {msg}"

            if sev in ["critical", "high"]:
                st.error(label)
            elif sev == "medium":
                st.warning(label)
            else:
                st.info(label)

    st.markdown("---")

    # ---------- This month in 20 seconds ----------
    st.subheader("📌 This month in 20 seconds")

    summary_text = build_month_summary_insight(
        selected_client_id,
        selected_month_start,
    )
    st.info(summary_text)

    st.markdown("---")

    # ---------- Cashflow engine controls ----------
    st.subheader("🔁 Cashflow engine controls")

    if st.button("Rebuild cashflow for next 12 months"):
        opening_hint = settings.get("opening_cash_start")
        ok = recompute_cashflow_from_ar_ap(
            selected_client_id,
            base_month=selected_month_start,
            n_months=12,
            opening_cash_hint=opening_hint,
        )
        if ok:
            st.success("Cashflow updated from AR/AP, payroll, opex and investing/financing moves.")
            st.rerun()
        else:
            st.error("Could not rebuild cashflow. Check AR/AP data and try again.")

    # ---------- Cashflow engine – 12-month breakdown (moved up) ----------
    st.markdown("---")
    st.subheader("🔍 Cashflow engine – 12-month breakdown")

    engine_df = fetch_cashflow_summary_for_client(selected_client_id)

    if engine_df is None:
        print("DEBUG loaded engine rows: 0 (None)")
    else:
        print("DEBUG loaded engine rows:", len(engine_df))
        print(engine_df.head())

    if engine_df is None or engine_df.empty:
        st.caption(
            "No cashflow engine data yet. "
            "Hit **'Rebuild cashflow for next 12 months'** above to generate it."
        )
    else:
        start = pd.to_datetime(selected_month_start).replace(day=1)
        end = start + pd.DateOffset(months=12)

        engine_df = engine_df.copy()
        engine_df["month_date"] = pd.to_datetime(engine_df["month_date"], errors="coerce")

        view = engine_df[
            (engine_df["month_date"] >= start) & (engine_df["month_date"] < end)
        ].copy()

        if view.empty:
            st.caption(
                "No cashflow rows in cashflow_summary for this window yet. "
                "Try rebuilding cashflow or picking a different focus month."
            )
        else:
            view["Month"] = view["month_date"].dt.strftime("%b %Y")

            cols = ["Month"]
            for c in [
                "operating_cf",
                "investing_cf",
                "financing_cf",
                "free_cash_flow",
                "closing_cash",
                "cash_danger_flag",
            ]:
                if c in view.columns:
                    cols.append(c)

            view = view[cols].rename(
                columns={
                    "operating_cf": "Operating CF",
                    "investing_cf": "Investing CF",
                    "financing_cf": "Financing CF",
                    "free_cash_flow": "Free cash flow",
                    "closing_cash": "Closing cash",
                    "cash_danger_flag": "Danger month?",
                }
            )

            st.dataframe(view, width="stretch")
            st.caption(
                "This is the raw output of your core cashflow engine "
                "that feeds the alerts and scenario views."
            )

    # ------------------------------------------------------------------
    # Payroll vs bills vs Net operating cash (6-month view)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💼 Payroll vs bills vs Net operating cash")

    window_months = pd.date_range(
        start=pd.to_datetime(selected_month_start),
        periods=6,
        freq="MS",
    )

    engine_df_all = fetch_cashflow_summary_for_client(selected_client_id)
    df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)
    payroll_by_month = compute_payroll_by_month(selected_client_id, window_months)

    if engine_df_all is None or engine_df_all.empty:
        st.caption("No cashflow engine data yet – rebuild cashflow first.")
    else:
        engine_df_all = engine_df_all.copy()
        engine_df_all["month_date"] = pd.to_datetime(
            engine_df_all["month_date"], errors="coerce"
        )

        engine_window = engine_df_all[
            engine_df_all["month_date"].isin(window_months)
        ][["month_date", "operating_cf"]].copy()

        # ---------- AP: monthly cash-out ----------
        ap_monthly = pd.DataFrame({"month_date": window_months})

        if df_ap is not None and not df_ap.empty:
            df_ap = df_ap.copy()
            df_ap["amount"] = pd.to_numeric(df_ap.get("amount"), errors="coerce").fillna(0.0)

            pay_col = "expected_payment_date"
            if pay_col not in df_ap.columns:
                pay_col = "due_date" if "due_date" in df_ap.columns else None

            if pay_col is not None:
                df_ap[pay_col] = pd.to_datetime(df_ap[pay_col], errors="coerce")
                df_ap["bucket_month"] = df_ap[pay_col].dt.to_period("M").dt.to_timestamp()

                ap_agg = (
                    df_ap.groupby("bucket_month", as_index=False)["amount"]
                    .sum()
                    .rename(columns={"bucket_month": "month_date", "amount": "ap_cash_out"})
                )

                ap_monthly = ap_monthly.merge(ap_agg, on="month_date", how="left")

        if "ap_cash_out" not in ap_monthly.columns:
            ap_monthly["ap_cash_out"] = 0.0

        ap_monthly["ap_cash_out"] = ap_monthly["ap_cash_out"].fillna(0.0)

        payroll_series = pd.Series(
            [float(payroll_by_month.get(m, 0.0)) for m in window_months],
            index=window_months,
            name="payroll_cash",
        ).reset_index().rename(columns={"index": "month_date"})

        merged = pd.DataFrame({"month_date": window_months})
        merged = merged.merge(engine_window, on="month_date", how="left")
        merged = merged.merge(ap_monthly, on="month_date", how="left")
        merged = merged.merge(payroll_series, on="month_date", how="left")

        merged["operating_cf"] = merged["operating_cf"].fillna(0.0)
        merged["ap_cash_out"] = merged["ap_cash_out"].fillna(0.0)
        merged["payroll_cash"] = merged["payroll_cash"].fillna(0.0)

        merged["month_date"] = pd.to_datetime(merged["month_date"], errors="coerce")
        merged = merged.sort_values("month_date")

        plot_df = merged.melt(
            id_vars="month_date",
            value_vars=["operating_cf", "payroll_cash", "ap_cash_out"],
            var_name="Series",
            value_name="Amount",
        )

        series_labels = {
            "operating_cf": "Net operating CF",
            "payroll_cash": "Payroll (cash out)",
            "ap_cash_out": "Bills / AP (cash out)",
        }
        plot_df["Series"] = plot_df["Series"].map(series_labels)

        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "month_date:T",
                    title="Month",
                    axis=alt.Axis(format="%b %Y"),
                ),
                y=alt.Y("Amount:Q", title=f"Amount (cash, {currency_code})"),
                color=alt.Color("Series:N", title=""),
                tooltip=[
                    alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                    "Series:N",
                    alt.Tooltip("Amount:Q", title="Amount", format=",.0f"),
                ],
            )
        )

        st.altair_chart(chart.interactive(), width="stretch")
        st.caption(
            "Shows how much of your monthly cash movement is driven by payroll vs supplier bills, "
            "compared to net operating cashflow."
        )

        # ------------------------------------------------------------------
    # Recovery vs Funding signal (clear fork)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Recovery vs funding signal")

    mode, signal_msg = _compute_recovery_vs_funding_signal(
        runway_months=runway_num,
        money_in=rev_num,
        money_out=burn_num,
    )

    if mode == "funding":
        st.error(f"🔴 {signal_msg}  → Go to **Funding Strategy** page.")
    elif mode == "mixed":
        st.warning(f"🟡 {signal_msg}  → Work on recovery + explore **Funding Strategy**.")
    else:
        st.info(f"🟢 {signal_msg}  → Focus on operating improvements first.")


        # ------------------------------------------------------------------
    # Ranked next actions (end-of-page action box)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("✅ Ranked next actions (do these first)")

    actions = []

    # 1) If runway tight -> funding prep + cash control
    if runway_num is not None and runway_num <= 4.0:
        actions.append("Start funding prep: build your raise plan, timeline, and target amount (Funding Strategy page).")
        actions.append("Freeze / slow discretionary spend and hiring until runway improves.")
        actions.append("Aggressively chase overdue AR and pull forward collections (Cash & Bills page).")
    else:
        # 2) If burn > revenue -> focus burn discipline + collections
        if burn_num > rev_num:
            actions.append("Reduce burn drivers (payroll + vendor commitments). Identify 1–2 cuts that move the needle.")
            actions.append("Improve collections: tighten terms, follow-ups, incentives for early payment (Cash & Bills page).")
            actions.append("Review pricing/discounting: ensure growth isn’t destroying cash.")
        else:
            actions.append("Keep burn controlled and invest only where ROI is clear (avoid unnecessary fixed commitments).")
            actions.append("Strengthen collection discipline so AR doesn’t become a hidden cash leak (Cash & Bills page).")
            actions.append("Run a quick scenario to stress-test the next 90 days (Scenarios section above).")

    # Render top 3 only (keep it sharp)
    actions = actions[:3]
    st.markdown("\n".join([f"{i+1}. {a}" for i, a in enumerate(actions)]))
    st.caption("This list is intentionally short. The goal is focus, not a to-do dump.")


        # ---------- Quick scenarios (founder-friendly) ----------
    st.markdown("---")
    st.subheader("🔮 Recovery Scenarios")

    render_quick_scenario_block_for_overview()


    # ---------- Collaboration layer (comments + tasks) ----------
    st.markdown("---")
    col_comments, col_tasks = st.columns([3, 1])

    with col_comments:
        comments_block("business_overview")

    with col_tasks:
        with st.expander("✅ Tasks for this page", expanded=False):
            tasks_block("business_overview")



def run_multi_lever_scenario(
    base_df: pd.DataFrame,
    start_month_ts: pd.Timestamp,
    op_delta_pct: float = 0.0,
    extra_payroll_per_month: float = 0.0,
    dept_burn_delta_pct: float = 0.0,
    extra_capex_per_month: float = 0.0,
    one_off_capex: float = 0.0,
    one_off_financing: float = 0.0,
    recurring_financing: float = 0.0,
) -> pd.DataFrame:
    """
    Generic scenario engine that takes the 12-month baseline cashflow engine
    (base_df) and applies different levers to produce a scenario cash curve.

    All inputs are incremental changes on top of the baseline:
      - op_delta_pct: % change to operating_cf (net operating cash) from start month
      - extra_payroll_per_month: additional (negative) payroll cost per month
      - dept_burn_delta_pct: extra % change applied only to negative operating_cf
      - extra_capex_per_month: extra investing CF per month (− = more capex, + = asset sales)
      - one_off_capex: one-off investing CF in the start month
      - one_off_financing: one-off financing CF in the start month (equity / loan inflow)
      - recurring_financing: recurring financing CF each month from start month
    """
    df = base_df.copy()
    df = df.sort_values("month_date").reset_index(drop=True)

    # Safety: make sure required columns exist & numeric
    for col in ["operating_cf", "investing_cf", "financing_cf", "free_cash_flow", "closing_cash"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    mask = df["month_date"] >= start_month_ts

    # ---------- Operating CF scenario ----------
    df["operating_cf_scenario"] = df["operating_cf"]

    # 1) Global % change on net operating cash from start month
    if op_delta_pct != 0.0:
        factor = 1.0 + op_delta_pct / 100.0
        df.loc[mask, "operating_cf_scenario"] = df.loc[mask, "operating_cf_scenario"] * factor

    # 2) Extra payroll (always cash out => negative impact)
    if extra_payroll_per_month != 0.0:
        df.loc[mask, "operating_cf_scenario"] = (
            df.loc[mask, "operating_cf_scenario"] - float(extra_payroll_per_month)
        )

    # 3) Extra burn % on the negative portion only (e.g. department-level increases)
    if dept_burn_delta_pct != 0.0:
        burn_factor = 1.0 + dept_burn_delta_pct / 100.0
        neg_mask = (df["operating_cf_scenario"] < 0.0) & mask
        df.loc[neg_mask, "operating_cf_scenario"] = (
            df.loc[neg_mask, "operating_cf_scenario"] * burn_factor
        )

    # ---------- Investing CF scenario ----------
    df["investing_cf_scenario"] = df["investing_cf"]

    # Extra monthly capex or asset sale from start month
    if extra_capex_per_month != 0.0:
        df.loc[mask, "investing_cf_scenario"] = (
            df.loc[mask, "investing_cf_scenario"] + float(extra_capex_per_month)
        )

    # One-off capex / asset sale in the start month
    if one_off_capex != 0.0:
        df.loc[df["month_date"] == start_month_ts, "investing_cf_scenario"] = (
            df.loc[df["month_date"] == start_month_ts, "investing_cf_scenario"]
            + float(one_off_capex)
        )

    # ---------- Financing CF scenario ----------
    df["financing_cf_scenario"] = df["financing_cf"]

    # One-off financing in start month (equity / new loan)
    if one_off_financing != 0.0:
        df.loc[df["month_date"] == start_month_ts, "financing_cf_scenario"] = (
            df.loc[df["month_date"] == start_month_ts, "financing_cf_scenario"]
            + float(one_off_financing)
        )

    # Recurring financing each month from start month (e.g. monthly loan drawdown)
    if recurring_financing != 0.0:
        df.loc[mask, "financing_cf_scenario"] = (
            df.loc[mask, "financing_cf_scenario"] + float(recurring_financing)
        )

    # ---------- Rebuild free CF & scenario closing cash ----------
    df["free_cash_flow_scenario"] = (
        df["operating_cf_scenario"]
        + df["investing_cf_scenario"]
        + df["financing_cf_scenario"]
    )

    # Derive the implied baseline opening cash for month 0
    first_row = df.iloc[0]
    base_opening_0 = float(first_row["closing_cash"]) - float(first_row["free_cash_flow"])

    closing_vals = []
    opening = base_opening_0
    for _, r in df.iterrows():
        free_cf_s = float(r["free_cash_flow_scenario"])
        closing = opening + free_cf_s
        closing_vals.append(closing)
        opening = closing

    df["closing_cash_scenario"] = closing_vals

    return df


def run_founder_scenario(
    base_df: pd.DataFrame,
    start_month: pd.Timestamp,
    controls: dict,
):
    """
    Apply founder-friendly scenario levers on top of baseline engine rows.

    controls may include:
      - rev_change_pct: % change to positive part of operating_cf
      - collections_boost_pct: % change to total operating_cf (proxy for earlier receipts)
      - extra_payroll_per_month: negative cashflow per month from start
      - spend_change_pct: % change to negative part of operating_cf (bills/opex)
      - extra_capex_one_off: one-off investing CF at start month (negative = more capex)
      - extra_capex_recurring: extra investing CF each month from start
      - equity_raise: one-off financing inflow at start month
      - recurring_funding: recurring financing inflow from start
    """
    df = base_df.copy().sort_values("month_date").reset_index(drop=True)

    # Default values for all levers
    rev_change_pct = float(controls.get("rev_change_pct", 0.0))
    collections_boost_pct = float(controls.get("collections_boost_pct", 0.0))
    extra_payroll_per_month = float(controls.get("extra_payroll_per_month", 0.0))
    spend_change_pct = float(controls.get("spend_change_pct", 0.0))
    extra_capex_one_off = float(controls.get("extra_capex_one_off", 0.0))
    extra_capex_recurring = float(controls.get("extra_capex_recurring", 0.0))
    equity_raise = float(controls.get("equity_raise", 0.0))
    recurring_funding = float(controls.get("recurring_funding", 0.0))

    mask = df["month_date"] >= start_month

    # --- Split operating CF into "good" (customer cash in) & "bad" (spend) parts ---
    op_base = df["operating_cf"].astype(float)
    positive_part = op_base.clip(lower=0.0)            # approx "cash from customers"
    negative_part = op_base.where(op_base < 0.0, 0.0)  # approx "bills + payroll + opex"

    # 1) Revenue lever → boost positive part from start month
    pos_factor = 1.0 + rev_change_pct / 100.0
    positive_scn = positive_part.copy()
    positive_scn[mask] = positive_scn[mask] * pos_factor

    # 2) Spend lever → change negative part (bills, opex) from start month
    spend_factor = 1.0 + spend_change_pct / 100.0
    negative_scn = negative_part.copy()
    negative_scn[mask] = negative_scn[mask] * spend_factor

    # 3) Collections lever → nudge whole operating CF (proxy for timing change)
    coll_factor = 1.0 + collections_boost_pct / 100.0

    op_scenario = (positive_scn + negative_scn) * coll_factor

    # 4) Hiring lever → extra payroll per month (negative cashflow)
    if extra_payroll_per_month != 0.0:
        op_scenario[mask] = op_scenario[mask] - abs(extra_payroll_per_month)

    df["operating_cf_scenario"] = op_scenario

    # --- Investing CF scenario (capex / asset sales) ---
    inv_base = df["investing_cf"].astype(float)
    inv_scenario = inv_base.copy()

    if extra_capex_recurring != 0.0:
        inv_scenario[mask] = inv_scenario[mask] + extra_capex_recurring

    if extra_capex_one_off != 0.0:
        inv_scenario[df["month_date"] == start_month] = (
            inv_scenario[df["month_date"] == start_month] + extra_capex_one_off
        )

    df["investing_cf_scenario"] = inv_scenario

    # --- Financing CF scenario (equity, loans, top-ups) ---
    fin_base = df["financing_cf"].astype(float)
    fin_scenario = fin_base.copy()

    if recurring_funding != 0.0:
        fin_scenario[mask] = fin_scenario[mask] + recurring_funding

    if equity_raise != 0.0:
        fin_scenario[df["month_date"] == start_month] = (
            fin_scenario[df["month_date"] == start_month] + equity_raise
        )

    df["financing_cf_scenario"] = fin_scenario

    # --- Recompute free CF + closing cash path for scenario ---
    df["free_cash_flow_scenario"] = (
        df["operating_cf_scenario"]
        + df["investing_cf_scenario"]
        + df["financing_cf_scenario"]
    )

    # Derive baseline opening_0 from first row
    closing_0 = float(df.loc[0, "closing_cash"] or 0.0)
    free_0 = float(df.loc[0, "free_cash_flow"] or 0.0)
    opening_0 = closing_0 - free_0

    closing_vals = []
    opening = opening_0
    for _, r in df.iterrows():
        free_sc = float(r["free_cash_flow_scenario"])
        closing = opening + free_sc
        closing_vals.append(closing)
        opening = closing

    df["closing_cash_scenario"] = closing_vals

    # --- Danger month comparison (baseline vs scenario) ---
    # baseline: use original closing_cash
    base_label, base_date = compute_danger_month_from_cash(
        df,  # positional arg
        "closing_cash",
    )

    # scenario: rename scenario column to closing_cash
    scen_tmp = df.rename(columns={"closing_cash_scenario": "closing_cash"})
    scen_label, scen_date = compute_danger_month_from_cash(
        scen_tmp,  # positional arg
        "closing_cash",
    )

    return df, (base_label, base_date), (scen_label, scen_date)



def render_scenario_view(
    scenario_name: str,
    base_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    currency_code: str,
):
    """
    Shared visualisation block for any scenario:
      - danger month comparison
      - cash curve chart
      - breakdown table
    """
    # --- Danger month comparison ---
    baseline_label, baseline_date = compute_danger_month_from_cash(
        base_df.copy(), "closing_cash"
    )

    scen_tmp = scen_df.rename(columns={"closing_cash_scenario": "closing_cash"})
    scen_label, scen_date = compute_danger_month_from_cash(
        scen_tmp, "closing_cash"
    )

    st.markdown("---")
    st.subheader(f"📌 {scenario_name} – danger month impact")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            "Baseline danger month",
            baseline_label or "None in 12 months",
        )
    with col_m2:
        delta_txt = None
        if baseline_date is not None and scen_date is not None:
            month_diff = (scen_date.to_period("M") - baseline_date.to_period("M")).n
            if month_diff != 0:
                sign_word = "later" if month_diff > 0 else "earlier"
                delta_txt = f"{abs(month_diff)} month(s) {sign_word}"
        st.metric(
            "Scenario danger month",
            scen_label or "None in 12 months",
            delta=delta_txt,
        )

    # --- Cash curve chart (baseline vs scenario) ---
    st.markdown("---")
    st.subheader(f"📉 Cash curve – baseline vs scenario ({currency_code})")

    chart_df = pd.DataFrame(
        {
            "month_date": scen_df["month_date"],
            "Baseline closing cash": base_df["closing_cash"].values,
            "Scenario closing cash": scen_df["closing_cash_scenario"].values,
        }
    )

    chart_df = chart_df.melt(
        id_vars="month_date",
        var_name="Series",
        value_name="Cash",
    )

    cash_chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X(
                "month_date:T",
                title="Month",
                axis=alt.Axis(format="%b %Y"),
            ),
            y=alt.Y("Cash:Q", title=f"Closing cash ({currency_code})"),
            color=alt.Color("Series:N", title=""),
            tooltip=[
                alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                "Series:N",
                alt.Tooltip("Cash:Q", title="Cash", format=",.0f"),
            ],
        )
    )

    st.altair_chart(cash_chart.interactive(), width="stretch")

    # --- Scenario table ---
    st.markdown("---")
    st.subheader("🔍 Scenario breakdown table")

    view = scen_df[
        [
            "month_date",
            "operating_cf",
            "operating_cf_scenario",
            "investing_cf",
            "investing_cf_scenario",
            "financing_cf",
            "financing_cf_scenario",
            "closing_cash",
            "closing_cash_scenario",
        ]
    ].copy()

    view["Month"] = view["month_date"].dt.strftime("%b %Y")
    view = view[
        [
            "Month",
            "operating_cf",
            "operating_cf_scenario",
            "investing_cf",
            "investing_cf_scenario",
            "financing_cf",
            "financing_cf_scenario",
            "closing_cash",
            "closing_cash_scenario",
        ]
    ].rename(
        columns={
            "operating_cf": "Op CF (base)",
            "operating_cf_scenario": "Op CF (scenario)",
            "investing_cf": "Invest CF (base)",
            "investing_cf_scenario": "Invest CF (scenario)",
            "financing_cf": "Fin CF (base)",
            "financing_cf_scenario": "Fin CF (scenario)",
            "closing_cash": "Closing cash (base)",
            "closing_cash_scenario": "Closing cash (scenario)",
        }
    )

    st.dataframe(view, width="stretch")

def page_scenarios():
    top_header("Scenarios & What-if")

    st.markdown(
        "Answer simple questions like **“Can I hire now?”**, "
        "**“What if revenue grows 20%?”**, or **“When do I need to raise?”** "
        "without touching your actual data."
    )

    if not selected_client_id or not selected_month_start:
        st.info("Pick a business and focus month at the top before running scenarios.")
        return

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    base_df = get_engine_window_for_scenarios(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if base_df is None or base_df.empty:
        st.warning(
            "No cashflow engine data yet for this business. "
            "Go to **Business at a Glance → Rebuild cashflow for next 12 months** first."
        )
        return

    base_df = base_df.sort_values("month_date").reset_index(drop=True)

    # Shared month selector for all scenarios
    month_opts = base_df["month_date"].dt.to_period("M").drop_duplicates()
    month_label_map = {p.strftime("%b %Y"): p.to_timestamp() for p in month_opts}
    month_labels = list(month_label_map.keys())

    st.markdown("---")
    st.subheader("🎛 Choose your scenario")

    # Layout: left = outputs, right = controls
    col_output, col_controls = st.columns([2, 1])

    # ----------------------------
    # RIGHT: PRESETS + CONTROLS
    # ----------------------------
    with col_controls:
        st.markdown("#### Scenario type")

        preset = st.selectbox(
            "Quick scenario preset",
            options=[
                "Custom",
                "Revenue push (+20%)",
                "Hire 1 key role (+$15k/month)",
                "Cut supplier/operating spend by 20%",
                "Raise $250k now",
                "Raise $250k + monthly top-up",
            ],
            help="Presets override the sliders below when you run the scenario.",
        )

        # Base controls dict (sliders will fill this for Custom)
        controls = {
            "rev_change_pct": 0.0,
            "collections_boost_pct": 0.0,
            "extra_payroll_per_month": 0.0,
            "spend_change_pct": 0.0,
            "extra_capex_one_off": 0.0,
            "extra_capex_recurring": 0.0,
            "equity_raise": 0.0,
            "recurring_funding": 0.0,
        }

        # Shared start month
        start_month_label = st.selectbox(
            "Scenario starts from month",
            options=month_labels,
        )
        start_month_ts = month_label_map[start_month_label]

        # If preset != Custom, we disable manual sliders (to avoid confusion)
        widgets_disabled = preset != "Custom"

        tab_rev, tab_hire, tab_spend, tab_capex, tab_fund = st.tabs(
            [
                "Revenue & customers",
                "Hiring & team",
                "Bills & spend",
                "Capex & assets",
                "Funding & runway",
            ]
        )

        # --- Revenue & customers ---
        with tab_rev:
            st.caption("Change revenue and how quickly customers pay you.")
            rev_change_pct = st.slider(
                "Change in cash from customers (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                help="Approximate effect of more/less sales and fewer cancellations.",
                disabled=widgets_disabled,
            )
            collections_boost_pct = st.slider(
                "Change from collecting invoices earlier / later (%)",
                min_value=-20,
                max_value=20,
                value=0,
                step=5,
                help="Positive = customers effectively pay earlier (better cash timing).",
                disabled=widgets_disabled,
            )
            if preset == "Custom":
                controls["rev_change_pct"] = rev_change_pct
                controls["collections_boost_pct"] = collections_boost_pct

        # --- Hiring & team ---
        with tab_hire:
            st.caption("Test the impact of new hires or hiring freeze.")
            extra_headcount_cost = st.number_input(
                f"Extra monthly payroll from that month ({currency_code})",
                value=0.0,
                step=1000.0,
                help="Total extra team cost per month (salary + on-costs). Use positive numbers.",
                disabled=widgets_disabled,
            )
            if preset == "Custom":
                controls["extra_payroll_per_month"] = extra_headcount_cost

        # --- Bills & spend ---
        with tab_spend:
            st.caption("Change how much you spend on suppliers, tools, marketing, etc.")
            spend_change_pct = st.slider(
                "Change in supplier / operating spend (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Negative = cut spend. Positive = spend more.",
                disabled=widgets_disabled,
            )
            if preset == "Custom":
                controls["spend_change_pct"] = spend_change_pct

        # --- Capex & assets ---
        with tab_capex:
            st.caption("Buy or sell big assets like equipment or fit-out.")
            extra_capex_one_off = st.number_input(
                f"One-off asset purchase / sale in start month ({currency_code})",
                value=0.0,
                step=5000.0,
                help="Negative = buy asset (cash out). Positive = sell asset (cash in).",
                disabled=widgets_disabled,
            )
            extra_capex_recurring = st.number_input(
                f"Recurring extra capex / asset cash each month ({currency_code})",
                value=0.0,
                step=1000.0,
                help="Negative = ongoing capex. Positive = regular asset proceeds.",
                disabled=widgets_disabled,
            )
            if preset == "Custom":
                controls["extra_capex_one_off"] = extra_capex_one_off
                controls["extra_capex_recurring"] = extra_capex_recurring

        # --- Funding & runway ---
        with tab_fund:
            st.caption("Test raising equity, loans or monthly investor top-ups.")
            equity_raise = st.number_input(
                f"One-off equity / loan cash-in in start month ({currency_code})",
                value=0.0,
                step=25000.0,
                help="Total cash received in that month from investors or lenders.",
                disabled=widgets_disabled,
            )
            recurring_funding = st.number_input(
                f"Recurring funding each month from start ({currency_code})",
                value=0.0,
                step=5000.0,
                help="Monthly investor top-ups or drawdowns.",
                disabled=widgets_disabled,
            )
            if preset == "Custom":
                controls["equity_raise"] = equity_raise
                controls["recurring_funding"] = recurring_funding

        # --- Apply preset overrides on top of base controls ---
        final_controls = controls.copy()

        if preset == "Revenue push (+20%)":
            final_controls["rev_change_pct"] = 20.0

        elif preset == "Hire 1 key role (+$15k/month)":
            final_controls["extra_payroll_per_month"] = 15_000.0

        elif preset == "Cut supplier/operating spend by 20%":
            final_controls["spend_change_pct"] = -20.0

        elif preset == "Raise $250k now":
            final_controls["equity_raise"] = 250_000.0

        elif preset == "Raise $250k + monthly top-up":
            final_controls["equity_raise"] = 250_000.0
            final_controls["recurring_funding"] = 10_000.0

        st.markdown("—")
        st.markdown("#### 💾 Save this scenario")

        scen_name = st.text_input(
            "Scenario name",
            value=f"{preset} @ {start_month_label}",
            help="E.g. 'Plan A – Hire dev now', 'Plan B – Cut burn & raise later'.",
        )
        scen_desc = st.text_area(
            "Short note (optional)",
            value="",
            help="Why this scenario exists – for your future self / board.",
        )

        if st.button("Save scenario", key="save_scenario_btn"):
            if not scen_name.strip():
                st.warning("Give this scenario a name before saving.")
            else:
                ok = save_scenario_for_client(
                    client_id=selected_client_id,
                    name=scen_name,
                    controls=final_controls,
                    base_month=selected_month_start,
                    description=scen_desc,
                )
                if ok:
                    st.success("Scenario saved.")
                else:
                    st.error("Could not save scenario. Check logs.")

        run_clicked = st.button("Run this scenario")

    # ----------------------------
    # LEFT: SAVED SCENARIOS + OUTPUTS
    # ----------------------------
    with col_output:
        # --- Saved scenarios selector (NEW) ---
        saved_scenarios = fetch_scenarios_for_client(selected_client_id)
        run_saved = False  # default
        scenario_label = preset  # what we'll show in the caption

        chosen_controls_from_saved = None
        chosen_base_month = None
        chosen_name_from_saved = None

        if saved_scenarios:
            st.markdown("#### 📂 Run a saved scenario")

            labels = [
                f"{s['name']} (saved {s['created_at'][:10]})"
                for s in saved_scenarios
            ]
            id_map = {
                labels[i]: saved_scenarios[i]["id"] for i in range(len(labels))
            }

            sel_label = st.selectbox(
                "Saved scenarios",
                options=["(None)"] + labels,
                key="saved_scenario_select",
            )

            col_ss1, col_ss2 = st.columns([1, 1])
            with col_ss1:
                run_saved = st.button("Run saved scenario", key="run_saved_btn")
            with col_ss2:
                delete_saved = st.button("Delete selected", key="delete_saved_btn")

            if sel_label != "(None)":
                chosen_id = id_map[sel_label]
                scen_row = fetch_scenario_by_id(chosen_id)
                if scen_row:
                    chosen_controls_from_saved = scen_row.get("controls") or {}
                    chosen_name_from_saved = scen_row.get("name")
                    chosen_base_month = scen_row.get("base_month")

            # Delete scenario if requested
            if saved_scenarios and sel_label != "(None)" and 'delete_saved' in locals() and delete_saved:
                if delete_scenario_by_id(chosen_id):
                    st.success("Scenario deleted.")
                    st.rerun()

            # If user presses "Run saved scenario", override controls + start month
            if run_saved and chosen_controls_from_saved:
                final_controls = chosen_controls_from_saved
                scenario_label = f"Saved: {chosen_name_from_saved or 'Scenario'}"
                if chosen_base_month:
                    try:
                        start_month_ts = pd.to_datetime(chosen_base_month).replace(day=1)
                    except Exception:
                        pass

        # --- Guard rails before running ---
        if not any(abs(v) > 0 for v in final_controls.values()):
            st.info(
                "Pick a preset on the right, set Custom sliders, "
                "or choose a saved scenario, then click **Run**."
            )
            return

        if not (run_clicked or run_saved):
            st.info(
                "Adjust the preset/controls on the right, or choose a saved scenario, "
                "then click **Run this scenario** or **Run saved scenario**."
            )
            return

        # --- Run scenario using current final_controls + start_month_ts ---
        scen_df, (base_label, base_date), (scen_label, scen_date) = run_founder_scenario(
            base_df=base_df,
            start_month=start_month_ts,
            controls=final_controls,
        )

        # --- Display which scenario was used ---
        st.caption(f"Scenario run: **{scenario_label}**")

        # --- Runway / danger month metrics ---
        st.subheader("📌 Runway & danger month")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric(
                "Baseline danger month",
                base_label or "None in 12 months",
            )

        with col_m2:
            delta_txt = None
            if base_date is not None and scen_date is not None:
                month_diff = (scen_date.to_period("M") - base_date.to_period("M")).n
                if month_diff != 0:
                    sign_word = "later" if month_diff > 0 else "earlier"
                    delta_txt = f"{abs(month_diff)} month(s) {sign_word}"
            st.metric(
                "Scenario danger month",
                scen_label or "None in 12 months",
                delta=delta_txt,
            )

        # --- Plain-English summary for founders ---
        st.markdown("#### 🧠 What this scenario means (in plain language)")

        min_base = float(base_df["closing_cash"].min())
        min_scen = float(scen_df["closing_cash_scenario"].min())
        diff_min = min_scen - min_base

        summary_lines = []

        if diff_min > 0:
            summary_lines.append(
                f"- Your **lowest cash balance improves by {currency_symbol}{diff_min:,.0f}**."
            )
        elif diff_min < 0:
            summary_lines.append(
                f"- Your **lowest cash balance drops by {currency_symbol}{abs(diff_min):,.0f}**."
            )

        if base_label and scen_label and base_label != scen_label:
            summary_lines.append(
                f"- The month where cash gets tight moves from **{base_label} → {scen_label}**."
            )

        if final_controls.get("extra_payroll_per_month", 0.0):
            summary_lines.append(
                "- This includes the extra monthly payroll you entered (new hires or raises)."
            )

        if final_controls.get("equity_raise", 0.0) or final_controls.get("recurring_funding", 0.0):
            summary_lines.append(
                "- This scenario also includes your funding changes (equity / loans / top-ups)."
            )

        if final_controls.get("spend_change_pct", 0.0) < 0:
            summary_lines.append("- You are **cutting operating spend**, which improves runway.")
        elif final_controls.get("spend_change_pct", 0.0) > 0:
            summary_lines.append("- You are **increasing operating spend**, which reduces runway.")

        if not summary_lines:
            summary_lines.append(
                "- This scenario makes only a small change to your cash position within 12 months."
            )

        st.markdown("\n".join(summary_lines))

        # --- Cash curve chart ---
        st.markdown("---")
        st.subheader(f"📉 Cash curve – baseline vs scenario ({currency_code})")

        chart_df = pd.DataFrame(
            {
                "month_date": scen_df["month_date"],
                "Baseline closing cash": base_df["closing_cash"].values,
                "Scenario closing cash": scen_df["closing_cash_scenario"].values,
            }
        )

        chart_long = chart_df.melt(
            id_vars="month_date", var_name="Series", value_name="Cash"
        )

        cash_chart = (
            alt.Chart(chart_long)
            .mark_line()
            .encode(
                x=alt.X(
                    "month_date:T",
                    title="Month",
                    axis=alt.Axis(format="%b %Y"),
                ),
                y=alt.Y("Cash:Q", title=f"Closing cash ({currency_code})"),
                color=alt.Color("Series:N", title=""),
                tooltip=[
                    alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                    "Series:N",
                    alt.Tooltip("Cash:Q", title="Cash", format=",.0f"),
                ],
            )
        )

        st.altair_chart(cash_chart.interactive(), width="stretch")

        # --- Scenario breakdown table ---
        st.markdown("---")
        st.subheader("🔍 Scenario breakdown by month")

        view = scen_df[
            [
                "month_date",
                "operating_cf",
                "operating_cf_scenario",
                "investing_cf",
                "investing_cf_scenario",
                "financing_cf",
                "financing_cf_scenario",
                "closing_cash",
                "closing_cash_scenario",
            ]
        ].copy()

        view["Month"] = view["month_date"].dt.strftime("%b %Y")
        view = view[
            [
                "Month",
                "operating_cf",
                "operating_cf_scenario",
                "investing_cf",
                "investing_cf_scenario",
                "financing_cf",
                "financing_cf_scenario",
                "closing_cash",
                "closing_cash_scenario",
            ]
        ].rename(
            columns={
                "operating_cf": "Op CF (base)",
                "operating_cf_scenario": "Op CF (scenario)",
                "investing_cf": "Invest CF (base)",
                "investing_cf_scenario": "Invest CF (scenario)",
                "financing_cf": "Fin CF (base)",
                "financing_cf_scenario": "Fin CF (scenario)",
                "closing_cash": "Closing cash (base)",
                "closing_cash_scenario": "Closing cash (scenario)",
            }
        )

        st.dataframe(view, width="stretch")
        st.caption(
            "Scenario is simulated on top of your engine. "
            "It does **not** modify your underlying AR/AP, payroll or opex data."
        )

    # --- Collaboration layer for scenarios ---
    st.markdown("---")
    comments_block("scenarios")
    tasks_block("scenarios")


# ---------- Page: Team & Spending ----------

def page_team_spending():
    top_header("Team & Spending")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    ensure_dept_overspend_alert(selected_client_id, selected_month_start)

    # ------------------------------------------------------------------
    # Load core data once
    # ------------------------------------------------------------------
    with st.spinner("Loading team and spending data..."):
        df_positions = fetch_payroll_positions_for_client(selected_client_id)
        df_hiring = fetch_hiring_monthly_for_client(selected_client_id)

        # NOTE: If dept view moved to Budget vs Actuals page, remove this next line
        # df_dept = fetch_dept_monthly_for_client(selected_client_id)

        df_ar_ap = fetch_ar_ap_for_client(selected_client_id)
        kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    if isinstance(df_ar_ap, tuple):
        df_ar, df_ap = df_ar_ap
    else:
        df_ar, df_ap = None, None

    # ------------------------------------------------------------------
    # Month setup
    # ------------------------------------------------------------------
    month_ts = pd.to_datetime(selected_month_start)

    # ------------------------------------------------------------------
    # Payroll (this month)
    # ------------------------------------------------------------------
    payroll_by_month = compute_payroll_by_month(selected_client_id, pd.date_range(start=month_ts, periods=1, freq="MS"))
    month_payroll = float(payroll_by_month.get(month_ts, 0.0))

    # ------------------------------------------------------------------
    # AP (this month)
    # ------------------------------------------------------------------
    month_ap = 0.0
    if df_ap is not None and not df_ap.empty:
        df_ap = df_ap.copy()
        df_ap["amount"] = pd.to_numeric(df_ap.get("amount"), errors="coerce").fillna(0.0)

        pay_col = "expected_payment_date"
        if pay_col not in df_ap.columns:
            pay_col = "due_date" if "due_date" in df_ap.columns else None

        if pay_col is not None:
            df_ap[pay_col] = pd.to_datetime(df_ap[pay_col], errors="coerce")
            df_ap["bucket_month"] = df_ap[pay_col].dt.to_period("M").dt.to_timestamp()
            month_ap = float(df_ap.loc[df_ap["bucket_month"] == month_ts, "amount"].sum())

    total_cash_out = month_payroll + month_ap
    payroll_share = (month_payroll / total_cash_out * 100.0) if total_cash_out else 0.0

    # ------------------------------------------------------------------
    # Revenue + runway (from KPIs)
    # ------------------------------------------------------------------
    month_revenue = 0.0
    if kpis:
        try:
            month_revenue = float(kpis.get("revenue") or 0.0)
        except Exception:
            month_revenue = 0.0

    payroll_vs_rev_pct = (month_payroll / month_revenue * 100.0) if month_revenue else None

    runway_months = None
    try:
        runway_months = float(kpis.get("runway_months")) if kpis else None
    except Exception:
        runway_months = None

    # ------------------------------------------------------------------
    # Headcount / FTE (active this month)
    # ------------------------------------------------------------------
    headcount_fte = 0.0
    avg_cost_per_fte = 0.0
    if df_positions is not None and not df_positions.empty:
        pos = df_positions.copy()
        pos["start_date"] = pd.to_datetime(pos.get("start_date"), errors="coerce")
        pos["end_date"] = pd.to_datetime(pos.get("end_date"), errors="coerce")

        active_mask = (pos["start_date"] <= month_ts) & (pos["end_date"].isna() | (pos["end_date"] >= month_ts))
        pos_active = pos[active_mask]

        if not pos_active.empty:
            headcount_fte = float(pos_active.get("fte", 0.0).fillna(0.0).sum())
            if headcount_fte > 0 and month_payroll > 0:
                avg_cost_per_fte = month_payroll / headcount_fte

    # ------------------------------------------------------------------
    # Compute payroll delta (next 6 months) BEFORE decision panel
    # ------------------------------------------------------------------
    month_index_6 = pd.date_range(start=month_ts, periods=6, freq="MS")
    payroll_by_month_6 = compute_payroll_by_month(selected_client_id, month_index_6)

    delta_payroll_6m = 0.0
    if payroll_by_month_6:
        vals = [float(payroll_by_month_6.get(m, 0.0)) for m in month_index_6]
        if len(vals) >= 2:
            delta_payroll_6m = vals[-1] - vals[0]

    # ------------------------------------------------------------------
    # 1) Top KPIs
    # ------------------------------------------------------------------
    st.subheader("👥 Payroll snapshot for this month")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Payroll cash this month", f"{currency_symbol}{month_payroll:,.0f}",
                  help="Total payroll cash out for this month from the payroll engine.")
    with k2:
        st.metric("Supplier bills (AP) this month", f"{currency_symbol}{month_ap:,.0f}",
                  help="Supplier bills expected to be paid this month.")
    with k3:
        st.metric("Payroll share of tracked cash-out", f"{payroll_share:.0f}%",
                  help="Payroll ÷ (Payroll + Supplier bills) for this month.")
    with k4:
        st.metric("Payroll as % of revenue",
                  "—" if payroll_vs_rev_pct is None else f"{payroll_vs_rev_pct:.0f}%",
                  help="How much of this month's revenue is spent on team costs.")

    if headcount_fte > 0:
        st.caption(
            f"Active headcount this month: **{headcount_fte:.1f} FTE**  •  "
            f"Approx. average cost per FTE: **{currency_symbol}{avg_cost_per_fte:,.0f}**"
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 1b) Decision panel + affordability context (rule-based)
    # ------------------------------------------------------------------
    settings = get_client_settings(selected_client_id)
    runway_min = float(settings.get("runway_min_months", 4.0)) if settings else 4.0

    severity, decisions = build_team_decision_panel(
        runway_months,
        runway_min,
        payroll_vs_rev_pct,
        delta_payroll_6m,
    )

    st.markdown("### 🧭 Team decision panel")

    msg = "\n".join([f"• {d}" for d in decisions]) if decisions else "• No decision signals available yet."
    if severity == "error":
        st.error(msg)
    elif severity == "warning":
        st.warning(msg)
    else:
        st.success(msg)

    afford_bits = []
    if payroll_vs_rev_pct is not None:
        afford_bits.append(f"Payroll is **{payroll_vs_rev_pct:.0f}% of revenue** this month")
    if runway_months is not None:
        afford_bits.append(f"Runway is **{runway_months:.1f} months**")
    if delta_payroll_6m != 0:
        direction = "increasing" if delta_payroll_6m > 0 else "decreasing"
        afford_bits.append(f"Team cost is **{direction}** by ~{currency_symbol}{abs(delta_payroll_6m):,.0f} over 6 months")

    if afford_bits:
        st.caption(" • ".join(afford_bits))

    # ------------------------------------------------------------------
    # 2) Team & payroll section
    # ------------------------------------------------------------------
    st.subheader("👥 Team & payroll (drives cashflow engine)")
    col_left, col_right = st.columns([2, 1])

    # ---------- LEFT ----------
    with col_left:
        st.markdown("#### Payroll cash-out (next 6 months)")

        if payroll_by_month_6:
            pay_df = pd.DataFrame({
                "month_date": month_index_6,
                "Payroll_cash": [float(payroll_by_month_6.get(m, 0.0)) for m in month_index_6],
            })

            chart = (
                alt.Chart(pay_df)
                .mark_line()
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("Payroll_cash:Q", title=f"Payroll cash-out ({currency_code})"),
                    tooltip=[
                        alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                        alt.Tooltip("Payroll_cash:Q", title="Payroll cash", format=",.0f"),
                    ],
                )
            )
            st.altair_chart(chart.interactive(), width="stretch")

            month_label = month_ts.strftime("%b %Y")
            story_lines, action_lines = build_team_spending_story(
                currency_symbol, month_label,
                month_payroll, month_ap, payroll_share, payroll_vs_rev_pct,
                delta_payroll_6m
            )

            st.markdown("### 🧠 What this means (founder view)")
            st.info(
                "\n".join([f"• {l}" for l in story_lines])
                + "\n\n**Recommended actions**\n"
                + "\n".join([f"• {a}" for a in action_lines])
            )
        else:
            st.caption("No payroll forecast yet — add roles on the right to populate this chart.")

        # Spend mix (labels improved)
        st.markdown("#### 💸 Spend mix this month (cash basis)")
        mix_df = pd.DataFrame({
            "Spend category": ["Payroll (team)", "Supplier bills (AP)"],
            "Cash out": [month_payroll, month_ap],
        }).set_index("Spend category")

        st.bar_chart(mix_df)

        st.caption(
            "Cash basis = money leaving the bank this month for payroll + supplier bills. "
            "This is not a full P&L."
        )

        # Data checks
        warnings = []
        if month_revenue == 0 and month_payroll > 0:
            warnings.append("Revenue is 0/blank for this month. Payroll % of revenue may be misleading.")
        if (df_positions is None or df_positions.empty) and month_payroll > 0:
            warnings.append("Payroll shows spend but no roles exist in Team list. Check payroll engine inputs.")
        if df_ap is None or df_ap.empty:
            warnings.append("No AP bills found. If you pay suppliers, add AP invoices to improve accuracy.")

        if warnings:
            st.warning("**Data checks**\n" + "\n".join([f"• {w}" for w in warnings]))

    # ---------- RIGHT (keep your existing role editor exactly) ----------
    with col_right:
        st.markdown("#### ➕ Add or update a role")

        # Build options for edit mode
        existing_options = []
        id_map = {}
        if df_positions is not None and not df_positions.empty:
            for _, r in df_positions.iterrows():
                label = f"{r.get('role_name', 'Role')} – {r.get('employee_name', 'Unassigned')}"
                existing_options.append(label)
                id_map[label] = r.get("id")

        edit_mode = st.radio(
            "What do you want to do?",
            options=["Add new role", "Edit existing role"],
            horizontal=True,
            key="role_edit_mode",
        )

        position_id = None
        existing_row = None
        if edit_mode == "Edit existing role" and existing_options:
            chosen = st.selectbox(
                "Pick a role to edit",
                ["-- Select --"] + existing_options,
                key="payroll_edit_select",
            )
            if chosen != "-- Select --":
                position_id = id_map.get(chosen)
                if position_id:
                    existing_row = df_positions[df_positions["id"] == position_id].iloc[0]
        elif edit_mode == "Edit existing role" and not existing_options:
            st.caption("No roles yet – switch to 'Add new role' to create one.")

        def _get(field, default=None):
            if existing_row is not None:
                return existing_row.get(field, default)
            return default

        with st.form(key="payroll_role_form"):
            role_name = st.text_input("Role title", value=_get("role_name", "Finance Manager"))
            employee_name = st.text_input(
                "Employee name (optional)",
                value=_get("employee_name", ""),
            )
            fte = st.number_input(
                "FTE",
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                value=float(_get("fte", 1.0) or 1.0),
            )
            base_salary_annual = st.number_input(
                "Base salary (annual, AUD)",
                min_value=0.0,
                step=5000.0,
                value=float(_get("base_salary_annual", 120000.0) or 0.0),
            )
            super_rate_pct = st.number_input(
                "Super rate (%)",
                min_value=0.0,
                max_value=30.0,
                step=0.5,
                value=float(_get("super_rate_pct", 11.0) or 0.0),
            )
            payroll_tax_pct = st.number_input(
                "Payroll tax & on-costs (%)",
                min_value=0.0,
                max_value=30.0,
                step=0.5,
                value=float(_get("payroll_tax_pct", 3.0) or 0.0),
            )

            start_default = _get("start_date", selected_month_start) or selected_month_start
            start_date = st.date_input(
                "Start date",
                value=start_default,
            )

            existing_end = _get("end_date", None)
            has_end_default = existing_end is not None
            has_end = st.checkbox(
                "This role has an end date",
                value=has_end_default,
            )
            if has_end:
                end_default = existing_end or start_default
                end_date = st.date_input(
                    "End date",
                    value=end_default,
                )
            else:
                end_date = None

            notes = st.text_area("Notes (optional)", value=_get("notes", ""))

            submitted = st.form_submit_button("Save role")

            if submitted:
                ok = upsert_payroll_position(
                    selected_client_id,
                    position_id=position_id
                    if (edit_mode == "Edit existing role" and position_id)
                    else None,
                    role_name=role_name,
                    employee_name=employee_name,
                    fte=fte,
                    base_salary_annual=base_salary_annual,
                    super_rate_pct=super_rate_pct,
                    payroll_tax_pct=payroll_tax_pct,
                    start_date=start_date,
                    end_date=end_date,
                    notes=notes,
                )
                if ok:
                    st.success("Role saved. Payroll engine will use this on the next cashflow rebuild.")
                    st.rerun()
                else:
                    st.error("Could not save role. Please try again.")

        # Delete role (still on the right panel)
        if df_positions is not None and not df_positions.empty:
            st.markdown("#### 🗑️ Delete a role")

            del_options = []
            del_map = {}
            for _, r in df_positions.iterrows():
                label = f"{r.get('role_name', 'Role')} – {r.get('employee_name', 'Unassigned')}"
                del_options.append(label)
                del_map[label] = r.get("id")

            chosen_del = st.selectbox(
                "Pick a role to delete",
                ["-- Select --"] + del_options,
                key="payroll_delete_select",
            )

            if chosen_del != "-- Select --":
                if st.button("Delete selected role", key="payroll_delete_btn"):
                    ok = delete_payroll_position(del_map[chosen_del])
                    if ok:
                        st.success("Role deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete that role.")



    # ------------------------------------------------------------------
    # 5) Collaboration: comments + tasks
    # ------------------------------------------------------------------
    st.markdown("---")
    col_comments, col_tasks = st.columns([3, 1])

    with col_comments:
        comments_block("team_spending")

    with col_tasks:
        with st.expander("✅ Tasks for this page", expanded=False):
            tasks_block("team_spending")



def build_team_decision_panel(
    runway_months: float | None,
    runway_min: float | None,
    payroll_vs_rev_pct: float | None,
    delta_payroll_6m: float,
):
    decisions = []
    severity = "info"  # info / warning / error

    # Runway rule
    if runway_months is not None and runway_min is not None:
        if runway_months < runway_min:
            decisions.append(
                f"Runway is **{runway_months:.1f} months**, below your safety threshold "
                f"of **{runway_min:.1f} months**."
            )
            decisions.append("Hiring freeze recommended until runway improves.")
            severity = "error"

    # Payroll vs revenue rule
    if payroll_vs_rev_pct is not None:
        if payroll_vs_rev_pct > 100:
            decisions.append(
                f"Payroll is **{payroll_vs_rev_pct:.0f}% of revenue**, which is not sustainable "
                f"without near-term revenue growth."
            )
            severity = max(severity, "warning")

    # Forward cost rule
    if delta_payroll_6m > 0:
        decisions.append(
            f"Payroll is expected to increase by **${delta_payroll_6m:,.0f}** over the next 6 months."
        )

    if not decisions:
        decisions.append(
            "Team cost is currently aligned with runway and revenue assumptions."
        )

    return severity, decisions

def build_team_spending_story(currency_symbol, month_label, month_payroll, month_ap, payroll_share, payroll_vs_rev_pct, delta_payroll_6m):
    lines = []
    lines.append(f"Payroll is {currency_symbol}{month_payroll:,.0f} in {month_label}.")
    if month_ap > 0:
        lines.append(f"Supplier bills expected this month are {currency_symbol}{month_ap:,.0f}.")
    lines.append(f"Payroll is {payroll_share:.0f}% of tracked cash-out on this page.")

    if payroll_vs_rev_pct is None:
        lines.append("Revenue for this month is not available yet, so payroll-to-revenue can’t be assessed.")
    else:
        if payroll_vs_rev_pct >= 150:
            lines.append(f"Payroll is {payroll_vs_rev_pct:.0f}% of revenue → **high burn risk** unless revenue ramps fast.")
        elif payroll_vs_rev_pct >= 80:
            lines.append(f"Payroll is {payroll_vs_rev_pct:.0f}% of revenue → monitor hiring pace and collections closely.")
        else:
            lines.append(f"Payroll is {payroll_vs_rev_pct:.0f}% of revenue → broadly healthy for an early-stage team (context matters).")

    # Forward trend
    if delta_payroll_6m > 0:
        lines.append(f"Forward view: payroll rises by {currency_symbol}{delta_payroll_6m:,.0f} over the next 6 months (new roles / raises / FTE changes).")
    elif delta_payroll_6m < 0:
        lines.append(f"Forward view: payroll falls by {currency_symbol}{abs(delta_payroll_6m):,.0f} over the next 6 months (roles ending / cost reduction).")
    else:
        lines.append("Forward view: payroll is broadly flat over the next 6 months.")

    # Actions
    actions = [
        "Confirm every planned role maps to a revenue milestone or delivery deadline.",
        "If runway is tight: pause non-critical hiring and review contractors.",
        "Check role end-dates and FTE assumptions are accurate (these drive the cash engine).",
    ]
    return lines, actions


@st.cache_data(ttl=60)
def build_14_week_cash_table(client_id, focus_month: date) -> pd.DataFrame:
    """
    Build a 14-week forward cash view starting from the focus month:

      - Week-by-week:
          * cash_in_ar        (customer receipts from AR.expected_date)
          * cash_out_ap       (supplier payments from AP.pay_expected_date / due_date)
          * cash_out_payroll  (approx. weekly payroll based on monthly payroll engine)
          * operating_cf      = cash_in_ar - cash_out_ap - cash_out_payroll
          * investing_cf      = 0 for now
          * financing_cf      = 0 for now
          * net_cash          = operating_cf + investing_cf + financing_cf
          * closing_cash      = prior closing + net_cash
          * danger_flag       = closing_cash <= 0

    Uses:
      - fetch_ar_ap_for_client
      - add_ar_aging / add_ap_aging
      - compute_payroll_by_month
      - fetch_kpis_for_client_month for starting cash
      - get_client_settings for AR/AP defaults
    """
    if client_id is None or focus_month is None:
        return pd.DataFrame()

    # ---- Client settings with safe fallbacks ----
    settings = get_client_settings(client_id) or {}
    ar_days = settings.get("ar_default_days", 30)
    ap_days = settings.get("ap_default_days", 30)

    # 1) Base date: first day of the focus month
    start_date = focus_month.replace(day=1)

    # 2) AR / AP with proper expected dates
    df_ar, df_ap = fetch_ar_ap_for_client(client_id)

    # ---------- AR (cash in) ----------
    if df_ar is not None and not df_ar.empty:
        df_ar = add_ar_aging(df_ar, as_of=start_date, ar_default_days=ar_days)

        # Normalise expected_date to plain date
        df_ar["expected_date"] = pd.to_datetime(
            df_ar["expected_date"], errors="coerce"
        ).dt.date

        # Exclude fully paid/closed
        if "status" in df_ar.columns:
            df_ar["status"] = df_ar["status"].astype(str)
            df_ar = df_ar[~df_ar["status"].str.lower().isin(["paid", "closed", "settled"])]

        df_ar["amount"] = pd.to_numeric(df_ar.get("amount"), errors="coerce").fillna(0.0)
    else:
        df_ar = pd.DataFrame()

    # ---------- AP (cash out) ----------
    if df_ap is not None and not df_ap.empty:
        df_ap = add_ap_aging(df_ap, as_of=start_date, ap_default_days=ap_days)

        df_ap["pay_expected_date"] = pd.to_datetime(
            df_ap["pay_expected_date"], errors="coerce"
        ).dt.date

        if "status" in df_ap.columns:
            df_ap["status"] = df_ap["status"].astype(str)
            df_ap = df_ap[~df_ap["status"].str.lower().isin(["paid", "closed", "settled"])]

        df_ap["amount"] = pd.to_numeric(df_ap.get("amount"), errors="coerce").fillna(0.0)
    else:
        df_ap = pd.DataFrame()

    # 3) Payroll by month (we'll spread monthly payroll evenly across days)
    month_index = pd.date_range(
        start=pd.to_datetime(start_date),
        periods=6,          # covers the 14-week window comfortably
        freq="MS",
    )
    payroll_by_month = compute_payroll_by_month(client_id, month_index)

    # 4) Opening cash: use KPI cash_balance for the focus month if possible
    opening_cash = None
    kpi_row = fetch_kpis_for_client_month(client_id, start_date)
    if kpi_row and "cash_balance" in kpi_row:
        try:
            opening_cash = float(kpi_row["cash_balance"])
        except Exception:
            opening_cash = None

    if opening_cash is None:
        opening_cash = 120_000.0  # fallback default

    # 5) Build 14 weekly buckets
    week_starts = [start_date + timedelta(weeks=i) for i in range(14)]

    current_cash = opening_cash
    rows = []

    for w_start in week_starts:
        w_end = w_start + timedelta(days=6)

        # --- Cash in from AR for this week ---
        if not df_ar.empty:
            ar_mask = df_ar["expected_date"].between(w_start, w_end)
            cash_in_ar = float(df_ar.loc[ar_mask, "amount"].sum())
        else:
            cash_in_ar = 0.0

        # --- Cash out from AP for this week ---
        if not df_ap.empty:
            ap_mask = df_ap["pay_expected_date"].between(w_start, w_end)
            cash_out_ap = float(df_ap.loc[ap_mask, "amount"].sum())
        else:
            cash_out_ap = 0.0

        # --- Payroll for this week (approximation) ---
        month_key = w_start.replace(day=1)
        month_ts = pd.to_datetime(month_key)

        month_payroll = float(payroll_by_month.get(month_ts, 0.0))

        days_in_month = calendar.monthrange(month_key.year, month_key.month)[1]
        payroll_per_day = month_payroll / days_in_month if days_in_month else 0.0
        cash_out_payroll = payroll_per_day * 7.0  # 7 days per week bucket

        # --- Operating, Investing, Financing ---
        operating_cf = cash_in_ar - cash_out_ap - cash_out_payroll
        investing_cf = 0.0
        financing_cf = 0.0

        net_cash = operating_cf + investing_cf + financing_cf
        closing_cash = current_cash + net_cash
        danger_flag = closing_cash <= 0

        rows.append(
            {
                "week_start": w_start,
                "week_label": f"Week of {w_start.strftime('%d %b %Y')}",
                "cash_in_ar": round(cash_in_ar, 2),
                "cash_out_ap": round(cash_out_ap, 2),
                "cash_out_payroll": round(cash_out_payroll, 2),
                "operating_cf": round(operating_cf, 2),
                "investing_cf": round(investing_cf, 2),
                "financing_cf": round(financing_cf, 2),
                "net_cash": round(net_cash, 2),
                "closing_cash": round(closing_cash, 2),
                "danger_flag": danger_flag,
            }
        )

        current_cash = closing_cash

    df_weeks = pd.DataFrame(rows)
    df_weeks["week_start"] = pd.to_datetime(df_weeks["week_start"])
    df_weeks = df_weeks.sort_values("week_start")

    return df_weeks



# ---------- Page: Cash & Bills ----------
def page_cash_bills():
    top_header("Cash & Bills")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    from datetime import date, timedelta

    # Currency for formatting
    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # Keep AR overdue alerts up to date (internally uses client settings)
    ensure_overdue_ar_alert(selected_client_id, as_of=date.today(), min_days_overdue=0)

    # Load AR/AP + 14-week cash table once
    with st.spinner("Loading cash, invoices and commitments..."):
        df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)
        week_df = build_14_week_cash_table(selected_client_id, selected_month_start)

    # If settings exist, reuse their AR/AP default days for ageing helpers
    settings = get_client_settings(selected_client_id) or {}
    ar_days = int(settings.get("ar_default_days", 30))
    ap_days = int(settings.get("ap_default_days", 30))

    today = date.today()
    horizon_4w = today + timedelta(days=28)

    # --- Add ageing info for AR / AP if we have data ---
    if df_ar is not None and not df_ar.empty:
        df_ar = add_ar_aging(df_ar, as_of=today, ar_default_days=ar_days)

    if df_ap is not None and not df_ap.empty:
        df_ap = add_ap_aging(df_ap, as_of=today, ap_default_days=ap_days)

    # ------------------------------------------------------------------
    # 1) Top KPIs – Cash + AR/AP next 4 weeks
    # ------------------------------------------------------------------
    st.subheader("📌 Cash & working capital snapshot")

    # Helper to get AR due next 4 weeks (unpaid)
    ar_next_4w = 0.0
    ar_overdue_amt = 0.0
    ar_overdue_count = 0

    if df_ar is not None and not df_ar.empty:
        ar_tmp = df_ar.copy()
        # Choose expected date column
        date_col = None
        if "expected_date" in ar_tmp.columns:
            date_col = "expected_date"
        elif "due_date" in ar_tmp.columns:
            date_col = "due_date"
        if date_col is not None:
            ar_tmp[date_col] = pd.to_datetime(ar_tmp[date_col], errors="coerce")
            mask_window = (ar_tmp[date_col].dt.date >= today) & (
                ar_tmp[date_col].dt.date <= horizon_4w
            )

            def _not_paid(row):
                s = str(row.get("status", "")).lower()
                return s not in ["paid", "closed", "settled"]

            mask_unpaid = ar_tmp.apply(_not_paid, axis=1)
            ar_next_4w = float(
                ar_tmp.loc[mask_window & mask_unpaid, "amount"].fillna(0.0).sum()
            )

        # Overdue AR
        if "days_past_expected" in ar_tmp.columns:
            mask_overdue = (ar_tmp["days_past_expected"] > 0) & ar_tmp.apply(
                _not_paid, axis=1
            )
            overdue_rows = ar_tmp[mask_overdue]
            ar_overdue_amt = float(overdue_rows["amount"].fillna(0.0).sum())
            ar_overdue_count = int(len(overdue_rows))

    # Helper to get AP due next 4 weeks (unpaid)
    ap_next_4w = 0.0
    ap_overdue_amt = 0.0
    ap_overdue_count = 0

    if df_ap is not None and not df_ap.empty:
        ap_tmp = df_ap.copy()
        # Choose expected pay date column
        pay_col = None
        if "pay_expected_date" in ap_tmp.columns:
            pay_col = "pay_expected_date"
        elif "expected_payment_date" in ap_tmp.columns:
            pay_col = "expected_payment_date"
        elif "due_date" in ap_tmp.columns:
            pay_col = "due_date"

        if pay_col is not None:
            ap_tmp[pay_col] = pd.to_datetime(ap_tmp[pay_col], errors="coerce")
            mask_window = (ap_tmp[pay_col].dt.date >= today) & (
                ap_tmp[pay_col].dt.date <= horizon_4w
            )

            def _not_paid_ap(row):
                s = str(row.get("status", "")).lower()
                return s not in ["paid", "closed", "settled"]

            mask_unpaid = ap_tmp.apply(_not_paid_ap, axis=1)
            ap_next_4w = float(
                ap_tmp.loc[mask_window & mask_unpaid, "amount"].fillna(0.0).sum()
            )

        # Overdue AP
        if "days_past_expected" in ap_tmp.columns:
            mask_overdue_ap = (ap_tmp["days_past_expected"] > 0) & ap_tmp.apply(
                _not_paid_ap, axis=1
            )
            overdue_rows_ap = ap_tmp[mask_overdue_ap]
            ap_overdue_amt = float(overdue_rows_ap["amount"].fillna(0.0).sum())
            ap_overdue_count = int(len(overdue_rows_ap))

    # Approximate "cash in bank now" from the 14-week table if available
    cash_now_label = "—"
    if week_df is not None and not week_df.empty:
        first_row = week_df.iloc[0]
        closing_first = float(first_row.get("closing_cash") or 0.0)
        net_first = float(first_row.get("net_cash") or 0.0)
        opening_first = closing_first - net_first
        cash_now_label = f"{currency_symbol}{opening_first:,.0f}"

    # Net position next 4 weeks
    net_4w = ar_next_4w - ap_next_4w

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Projected opening cash (week 1)",
            cash_now_label,
            help="Opening cash implied by week 1 of the 14-week cashflow projection (not a live bank feed).",
        )

    with k2:
        st.metric(
            "Invoices to collect (next 4 weeks)",
            f"{currency_symbol}{ar_next_4w:,.0f}",
            help="Customer invoices due / expected in the next 28 days.",
        )
    with k3:
        st.metric(
            "Bills to pay (next 4 weeks)",
            f"{currency_symbol}{ap_next_4w:,.0f}",
            help="Supplier bills expected to be paid in the next 28 days.",
        )
    with k4:
        st.metric(
            "Net position (next 4 weeks)",
            f"{currency_symbol}{net_4w:,.0f}",
            help="Invoices due in minus bills due out over the next 28 days.",
        )
    # ------------------------------------------------------------------
    # Cash danger summary strip (from 14-week view)
    # ------------------------------------------------------------------
    st.markdown("### 🚨 Cash danger summary (next 14 weeks)")

    if week_df is None or week_df.empty or "closing_cash" not in week_df.columns:
        st.info("14-week cashflow not available yet to assess cash risk.")
    else:
        w = week_df.copy()

        # Ensure datetime
        if "week_start" in w.columns:
            w["week_start"] = pd.to_datetime(w["week_start"], errors="coerce")

        min_cash = float(w["closing_cash"].min())
        idx_min = int(w["closing_cash"].idxmin())
        worst_week = w.loc[idx_min, "week_start"] if "week_start" in w.columns else None

        # Optional safety buffer from settings (add later if you want)
        cash_buffer = float((settings.get("min_cash_buffer") or 0.0))

        gap_to_buffer = max(0.0, cash_buffer - min_cash)

        # Severity label
        if min_cash <= 0:
            sev = "error"
            headline = "Cash goes negative in the next 14 weeks."
        elif gap_to_buffer > 0:
            sev = "warning"
            headline = "Cash stays positive, but falls below your safety buffer."
        else:
            sev = "success"
            headline = "Cash stays above your safety buffer."

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Lowest projected cash", f"{currency_symbol}{min_cash:,.0f}")
        cB.metric("Week of lowest cash", worst_week.strftime("%d %b %Y") if worst_week is not None else "—")
        cC.metric("Safety buffer", f"{currency_symbol}{cash_buffer:,.0f}")
        cD.metric("Gap to buffer", f"{currency_symbol}{gap_to_buffer:,.0f}")

        if sev == "error":
            st.error(f"• {headline}")
        elif sev == "warning":
            st.warning(f"• {headline}")
        else:
            st.success(f"• {headline}")

    # Short caption for overdue situation
    overdue_bits = []
    if ar_overdue_amt > 0:
        overdue_bits.append(
            f"{ar_overdue_count} invoice(s) overdue from customers "
            f"≈ {currency_symbol}{ar_overdue_amt:,.0f}"
        )
    if ap_overdue_amt > 0:
        overdue_bits.append(
            f"{ap_overdue_count} bill(s) overdue to suppliers "
            f"≈ {currency_symbol}{ap_overdue_amt:,.0f}"
        )

    if overdue_bits:
        st.caption("Overdue snapshot: " + " · ".join(overdue_bits))

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2) AR & AP ageing – side by side + drilldown
    # ------------------------------------------------------------------
    st.subheader("📊 AR & AP ageing – who pays late, who you pay")

    col_ar, col_ap = st.columns(2)

    # ---------- AR ageing summary ----------
    with col_ar:
        st.markdown("#### 📥 Invoices coming in (AR)")

        if df_ar is None or df_ar.empty:
            st.info("No AR records yet for this client.")
        else:
            if "aging_bucket" in df_ar.columns and "amount" in df_ar.columns:
                # Only non-paid invoices
                def _not_paid(row):
                    s = str(row.get("status", "")).lower()
                    return s not in ["paid", "closed", "settled"]

                df_not_paid = df_ar[df_ar.apply(_not_paid, axis=1)].copy()

                if df_not_paid.empty:
                    st.caption("All customer invoices appear to be paid or closed.")
                else:
                    ageing_summary = (
                        df_not_paid.groupby("aging_bucket", as_index=False)["amount"]
                        .sum()
                        .rename(
                            columns={
                                "aging_bucket": "Bucket",
                                "amount": "Total amount",
                            }
                        )
                    )
                    st.dataframe(ageing_summary, width="stretch")
                    st.caption("Unpaid customer invoices grouped by ageing bucket.")

            with st.expander("View AR invoice details (drilldown)", expanded=False):
                ar_view = df_ar.copy()
                cols = []
                if "counterparty" in ar_view.columns:
                    cols.append("counterparty")
                if "issued_date" in ar_view.columns:
                    cols.append("issued_date")
                if "expected_date" in ar_view.columns:
                    cols.append("expected_date")
                if "amount" in ar_view.columns:
                    cols.append("amount")
                if "due_date" in ar_view.columns:
                    cols.append("due_date")
                if "status" in ar_view.columns:
                    cols.append("status")
                if "days_past_expected" in ar_view.columns:
                    cols.append("days_past_expected")
                if "aging_bucket" in ar_view.columns:
                    cols.append("aging_bucket")
                if "is_over_default" in ar_view.columns:
                    cols.append("is_over_default")

                if cols:
                    ar_view = ar_view[cols].rename(
                        columns={
                            "counterparty": "Customer",
                            "issued_date": "Issued",
                            "expected_date": "Expected",
                            "amount": "Amount",
                            "due_date": "Original due",
                            "status": "Status",
                            "days_past_expected": "Days overdue",
                            "aging_bucket": "Aging bucket",
                            "is_over_default": "Over default terms?",
                        }
                    )

                    filter_text = st.text_input(
                        "Filter AR by customer or status",
                        value="",
                        key="ar_filter",
                    )
                    if filter_text:
                        ft = filter_text.lower()
                        mask = (
                            ar_view["Customer"]
                            .astype(str)
                            .str.lower()
                            .str.contains(ft)
                            | ar_view["Status"]
                            .astype(str)
                            .str.lower()
                            .str.contains(ft)
                        )
                        ar_view = ar_view[mask]

                    st.dataframe(ar_view, width="stretch")
                else:
                    st.caption("AR table missing expected columns for detailed view.")

    # ---------- AP ageing summary ----------
    with col_ap:
        st.markdown("#### 📤 Bills to pay (AP)")

        if df_ap is None or df_ap.empty:
            st.info("No AP records yet for this client.")
        else:
            if "aging_bucket" in df_ap.columns and "amount" in df_ap.columns:
                def _not_paid_ap(row):
                    s = str(row.get("status", "")).lower()
                    return s not in ["paid", "closed", "settled"]

                df_not_paid_ap = df_ap[df_ap.apply(_not_paid_ap, axis=1)].copy()

                if df_not_paid_ap.empty:
                    st.caption("All supplier bills appear to be paid or closed.")
                else:
                    ap_ageing_summary = (
                        df_not_paid_ap.groupby("aging_bucket", as_index=False)["amount"]
                        .sum()
                        .rename(
                            columns={
                                "aging_bucket": "Bucket",
                                "amount": "Total amount",
                            }
                        )
                    )
                    st.dataframe(ap_ageing_summary, width="stretch")
                    st.caption("Unpaid supplier bills grouped by ageing bucket.")

            with st.expander("View AP bill details (drilldown)", expanded=False):
                ap_view = df_ap.copy()
                cols = []
                if "counterparty" in ap_view.columns:
                    cols.append("counterparty")
                if "amount" in ap_view.columns:
                    cols.append("amount")
                if "due_date" in ap_view.columns:
                    cols.append("due_date")
                if "pay_expected_date" in ap_view.columns:
                    cols.append("pay_expected_date")
                elif "expected_payment_date" in ap_view.columns:
                    cols.append("expected_payment_date")
                if "status" in ap_view.columns:
                    cols.append("status")
                if "days_past_expected" in ap_view.columns:
                    cols.append("days_past_expected")
                if "aging_bucket" in ap_view.columns:
                    cols.append("aging_bucket")
                if "is_over_default" in ap_view.columns:
                    cols.append("is_over_default")

                if cols:
                    ap_view = ap_view[cols].rename(
                        columns={
                            "counterparty": "Supplier",
                            "amount": "Amount",
                            "due_date": "Original due",
                            "pay_expected_date": "Expected to pay",
                            "expected_payment_date": "Expected to pay",
                            "status": "Status",
                            "days_past_expected": "Days overdue",
                            "aging_bucket": "Aging bucket",
                            "is_over_default": "Over default terms?",
                        }
                    )

                    filter_text_ap = st.text_input(
                        "Filter AP by supplier or status",
                        value="",
                        key="ap_filter",
                    )
                    if filter_text_ap:
                        ft = filter_text_ap.lower()
                        mask_ap = (
                            ap_view["Supplier"]
                            .astype(str)
                            .str.lower()
                            .str.contains(ft)
                            | ap_view["Status"]
                            .astype(str)
                            .str.lower()
                            .str.contains(ft)
                        )
                        ap_view = ap_view[mask_ap]

                    st.dataframe(ap_view, width="stretch")
                else:
                    st.caption("AP table missing expected columns for detailed view.")

    # ------------------------------------------------------------------
    # 3) Cash commitments (near-term list)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Cash commitments coming up")

    commitments = build_cash_commitments(df_ar, df_ap, limit=7)

    if commitments.empty:
        st.info("No upcoming cash movements found yet.")
    else:
        st.dataframe(commitments, width="stretch")
        st.caption("Top upcoming customer receipts and supplier payments.")

    # ------------------------------------------------------------------
    # 4) 14-week cashflow – chart + table + narrative
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📆 14-week cashflow – can we sleep at night?")

    if week_df is None or week_df.empty:
        st.caption(
            "Not enough AR/AP + payroll data to build a 14-week view yet. "
            "Add invoices and team data to see near-term cash pressure."
        )
    else:
        view = week_df.copy()
        view["Week"] = view["week_start"].dt.strftime("%d %b %Y")

        cols = [
            "Week",
            "cash_in_ar",
            "cash_out_ap",
            "cash_out_payroll",
            "operating_cf",
            "investing_cf",
            "financing_cf",
            "net_cash",
            "closing_cash",
        ]
        cols = [c for c in cols if c in view.columns]

        view = view[cols].rename(
            columns={
                "cash_in_ar": "Cash in (customers)",
                "cash_out_ap": "Cash out (bills/AP)",
                "cash_out_payroll": "Cash out (payroll)",
                "operating_cf": "Operating CF",
                "investing_cf": "Investing CF",
                "financing_cf": "Financing CF",
                "net_cash": "Net cash",
                "closing_cash": "Closing cash",
            }
        )

        st.dataframe(view, width="stretch")
        st.caption(
            "Expected cash in/out by week for the next ~3 months, "
            "split by operating, investing and financing activity."
        )

        # Cash cliff highlight (first week where closing_cash <= 0)
        if "closing_cash" in week_df.columns:
            danger_weeks = week_df[week_df["closing_cash"] <= 0]
            if not danger_weeks.empty:
                first_danger = danger_weeks.iloc[0]
                st.error(
                    f"⚠️ Cash cliff around **week of "
                    f"{first_danger['week_start'].strftime('%d %b %Y')}** – "
                    f"closing cash projected at "
                    f"{currency_symbol}{first_danger['closing_cash']:,.0f}."
                )

        # Weekly closing cash chart
        chart = (
            alt.Chart(week_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "week_start:T",
                    title="Week starting",
                    axis=alt.Axis(format="%d %b"),
                ),
                y=alt.Y(
                    "closing_cash:Q",
                    title=f"Projected closing cash ({currency_code})",
                ),
                tooltip=[
                    alt.Tooltip(
                        "week_start:T", title="Week of", format="%d %b %Y"
                    ),
                    alt.Tooltip(
                        "closing_cash:Q", title="Closing cash", format=",.0f"
                    ),
                    alt.Tooltip(
                        "operating_cf:Q", title="Operating CF", format=",.0f"
                    ),
                    alt.Tooltip(
                        "investing_cf:Q", title="Investing CF", format=",.0f"
                    ),
                    alt.Tooltip(
                        "financing_cf:Q", title="Financing CF", format=",.0f"
                    ),
                ],
            )
        )

        st.altair_chart(chart.interactive(), width="stretch")

        # --- Narrative: why cash is going up/down and what to do ---
        st.markdown("#### 🧠 What this 14-week view is telling you")

        # Approximate starting and ending cash
        first_row = week_df.iloc[0]
        closing_first = float(first_row.get("closing_cash") or 0.0)
        net_first = float(first_row.get("net_cash") or 0.0)
        opening_first = closing_first - net_first

        last_row = week_df.iloc[-1]
        closing_last = float(last_row.get("closing_cash") or 0.0)

        delta_cash = closing_last - opening_first

        total_op = float(week_df.get("operating_cf", 0.0).sum()) if "operating_cf" in week_df.columns else 0.0
        total_inv = float(week_df.get("investing_cf", 0.0).sum()) if "investing_cf" in week_df.columns else 0.0
        total_fin = float(week_df.get("financing_cf", 0.0).sum()) if "financing_cf" in week_df.columns else 0.0

        # Identify main driver by absolute size
        driver_map = {
            "operating activity (customers & suppliers)": total_op,
            "investing moves (capex / asset sales)": total_inv,
            "financing moves (equity / loans / repayments)": total_fin,
        }
        main_driver_label = max(driver_map, key=lambda k: abs(driver_map[k]))
        main_driver_value = driver_map[main_driver_label]

        lines = []

        # Overall direction
        if abs(delta_cash) < max(10_000, abs(opening_first) * 0.05):
            lines.append(
                f"- Your cash stays **roughly flat** over the next 14 weeks "
                f"(change of about {currency_symbol}{delta_cash:,.0f})."
            )
        elif delta_cash > 0:
            lines.append(
                f"- Your cash **improves by about {currency_symbol}{delta_cash:,.0f}** "
                f"over the next 14 weeks."
            )
        else:
            lines.append(
                f"- Your cash **drops by about {currency_symbol}{abs(delta_cash):,.0f}** "
                f"over the next 14 weeks."
            )

        # Driver explanation
        lines.append(
            f"- The biggest driver of this change is **{main_driver_label}** "
            f"(net {currency_symbol}{main_driver_value:,.0f} over the period)."
        )

        # Operating / investing / financing hints
        if total_op < 0:
            lines.append(
                "- **Operating cash is negative** – team and supplier payments are "
                "higher than customer cash in. Consider tightening AR, reducing non-critical spend, "
                "or adjusting pricing/discounts."
            )
        elif total_op > 0:
            lines.append(
                "- **Operating cash is positive**, which is good – just keep an eye on "
                "large upcoming bills that could flip this."
            )

        if total_inv < 0:
            lines.append(
                "- You have **net investing cash out** (likely capex or asset purchases). "
                "Check if the timing/size of this spend matches your cash comfort level."
            )

        if total_fin < 0:
            lines.append(
                "- **Financing is net cash out** (debt repayments/dividends). If cash is tight, "
                "consider slowing repayments or pausing distributions where possible."
            )
        elif total_fin > 0:
            lines.append(
                "- **Financing is net cash in** – you’re relying on equity/loans/top-ups to "
                "support your cash. Make sure this is intentional and sustainable."
            )

        if "closing_cash" in week_df.columns and (week_df["closing_cash"] <= 0).any():
            lines.append(
                "- Because cash is projected to go **below zero**, you’ll likely need a mix of: "
                "faster collections, delayed non-essential bills, leaner spend, or external funding."
            )
        else:
            lines.append(
                "- Cash **does not go negative** in this 14-week window. Focus on keeping AR current "
                "and avoiding sudden large, unplanned payments."
            )

        st.markdown("\n".join(lines))
    # ------------------------------------------------------------------
    # Action box (Founder next steps)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("✅ Action plan (next 7 days)")

    actions = []

    # 1) Cash risk actions from 14-week view
    if week_df is not None and not week_df.empty and "closing_cash" in week_df.columns:
        min_cash = float(week_df["closing_cash"].min())
        if min_cash <= 0:
            actions.append("Trigger cash protection mode: freeze non-essential spend and re-check payroll + supplier schedule.")
            actions.append("Pull forward collections: chase overdue AR and ask top customers to pay early.")
            actions.append("Prepare funding path: bridge loan / investor update / overdraft discussions this week.")
        elif min_cash < 50_000:  # you can replace with settings buffer later
            actions.append("Cash is tight: renegotiate payment timing for non-critical AP and tighten AR follow-ups.")

    # 2) AR actions
    if ar_overdue_amt > 0:
        actions.append(f"Chase overdue customer invoices totaling **{currency_symbol}{ar_overdue_amt:,.0f}** (prioritise top 3 by value).")
    elif ar_next_4w > 0:
        actions.append(f"Confirm expected receipts of **{currency_symbol}{ar_next_4w:,.0f}** due in the next 4 weeks (lock in payment dates).")

    # 3) AP actions
    if ap_next_4w > 0:
        actions.append(f"Review supplier bills of **{currency_symbol}{ap_next_4w:,.0f}** due in next 4 weeks and delay non-critical payments if needed.")

    # 4) If there are no actions (very rare)
    if not actions:
        actions.append("No immediate cash risks detected. Keep AR current and avoid unplanned spending.")

    st.info("\n".join([f"• {a}" for a in actions]))

    # ------------------------------------------------------------------
    # 5) Collaboration: comments + tasks (bottom-right)
    # ------------------------------------------------------------------
    st.markdown("---")
    col_comments, col_tasks = st.columns([3, 1])

    with col_comments:
        comments_block("cash_bills")

    with col_tasks:
        with st.expander("✅ Tasks for this page", expanded=False):
            tasks_block("cash_bills")



def cash_flows_editor_block():
    st.markdown("---")
    st.subheader("💹 Investing & Financing cash moves")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(
            "Use this to record *big one-off moves* like buying equipment, "
            "raising equity, taking or repaying loans, or paying dividends. "
            "Positive = cash **in**, negative = cash **out**."
        )

    with col_btn:
        if st.button("Rebuild cashflow after changes", key="rebuild_after_inv_fin"):
            ok = recompute_cashflow_from_ar_ap(
                selected_client_id,
                base_month=selected_month_start,
                n_months=12,
                opening_cash_hint=None,
            )
            if ok:
                st.success("Cashflow rebuilt with latest investing/financing moves.")
            else:
                st.error("Could not rebuild cashflow. Check data and try again.")

    # Load existing flows
    df_inv = fetch_investing_flows_for_client(selected_client_id)
    df_fin = fetch_financing_flows_for_client(selected_client_id)

    tab_inv, tab_fin = st.tabs(["🏗 Investing flows", "🏦 Financing flows"])

    # ----------------- INVESTING TAB -----------------
    with tab_inv:
        st.markdown("#### Investing flows (capex, asset sales, etc.)")

        if df_inv is None or df_inv.empty:
            st.caption("No investing flows yet for this business.")
        else:
            df_show = df_inv.copy()
            df_show["month_date"] = pd.to_datetime(
                df_show["month_date"], errors="coerce"
            )
            df_show = df_show.sort_values("month_date")
            df_show["Month"] = df_show["month_date"].dt.strftime("%b %Y")
            st.dataframe(
                df_show[["id", "Month", "amount", "category", "notes"]],
                width="stretch",
            )

        st.markdown("##### ➕ Add a new investing flow")
        with st.form(key="new_investing_flow_form"):
            col1, col2 = st.columns(2)
            with col1:
                inv_date = st.date_input(
                    "Month (use any date in that month)",
                    value=selected_month_start,
                    key="inv_date",
                )
                inv_amount = st.number_input(
                    "Amount (positive = cash in, negative = cash out)",
                    value=0.0,
                    step=1000.0,
                    key="inv_amount",
                )
            with col2:
                inv_category = st.text_input(
                    "Category",
                    value="Capex",
                    key="inv_category",
                )
                inv_notes = st.text_area(
                    "Notes (optional)",
                    value="",
                    key="inv_notes",
                )

            submitted_inv = st.form_submit_button("Save investing flow")
            if submitted_inv:
                ok = create_investing_flow(
                    selected_client_id,
                    month_date=inv_date,
                    amount=inv_amount,
                    category=inv_category,
                    notes=inv_notes,
                )
                if ok:
                    st.success("Investing flow saved.")
                    st.rerun()
                else:
                    st.error("Could not save investing flow. Check inputs and try again.")

    # ----------------- FINANCING TAB -----------------
    with tab_fin:
        st.markdown("#### Financing flows (equity, loans, repayments, dividends)")

        if df_fin is None or df_fin.empty:
            st.caption("No financing flows yet for this business.")
        else:
            df_show = df_fin.copy()
            df_show["month_date"] = pd.to_datetime(
                df_show["month_date"], errors="coerce"
            )
            df_show = df_show.sort_values("month_date")
            df_show["Month"] = df_show["month_date"].dt.strftime("%b %Y")
            st.dataframe(
                df_show[["id", "Month", "amount", "category", "notes"]],
                width="stretch",
            )

        st.markdown("##### ➕ Add a new financing flow")
        with st.form(key="new_financing_flow_form"):
            col1, col2 = st.columns(2)
            with col1:
                fin_date = st.date_input(
                    "Month (use any date in that month)",
                    value=selected_month_start,
                    key="fin_date",
                )
                fin_amount = st.number_input(
                    "Amount (positive = cash in, negative = cash out)",
                    value=0.0,
                    step=1000.0,
                    key="fin_amount",
                )
            with col2:
                fin_category = st.text_input(
                    "Category",
                    value="Equity raise",
                    key="fin_category",
                )
                fin_notes = st.text_area(
                    "Notes (optional)",
                    value="",
                    key="fin_notes",
                )

            submitted_fin = st.form_submit_button("Save financing flow")
            if submitted_fin:
                ok = create_financing_flow(
                    selected_client_id,
                    month_date=fin_date,
                    amount=fin_amount,
                    category=fin_category,
                    notes=fin_notes,
                )
                if ok:
                    st.success("Financing flow saved.")
                    st.rerun()
                else:
                    st.error("Could not save financing flow. Check inputs and try again.")



def _resolve_dept_overspend_alerts_for_month(client_id, month_start: date):
    """
    Internal helper: resolve existing dept_overspend alerts for the given month.
    """
    try:
        supabase.table("alerts").update(
            {
                "is_active": False,
                "is_dismissed": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("client_id", str(client_id)) \
         .eq("alert_type", "dept_overspend") \
         .eq("month_date", month_start.isoformat()) \
         .eq("is_active", True) \
         .execute()
    except Exception:
        pass


def ensure_dept_overspend_alert(client_id, month_start: date):
    """
    Look at dept_monthly for this client+month and create/resolve a 'dept_overspend' alert.

    Uses settings:
      - overspend_warn_pct (start alerting if variance% > warn)
      - overspend_high_pct (mark as high/critical if variance% > high)
    """
    if client_id is None or month_start is None:
        return

    # Get settings
    settings = get_client_settings(client_id)
    warn_pct = float(settings["overspend_warn_pct"])
    high_pct = float(settings["overspend_high_pct"])

    df_dept = fetch_dept_monthly_for_client(client_id)
    if df_dept.empty:
        _resolve_dept_overspend_alerts_for_month(client_id, month_start)
        return

    # Filter to this month if we have month_date
    if "month_date" in df_dept.columns:
        df_dept["month_date"] = pd.to_datetime(df_dept["month_date"], errors="coerce")
        df_dept = df_dept[df_dept["month_date"] == pd.to_datetime(month_start)]
        if df_dept.empty:
            _resolve_dept_overspend_alerts_for_month(client_id, month_start)
            return

    # Identify budget/actual columns similar to page_team_spending
    budget_col = None
    actual_col = None

    for cand in ["budget_total", "budget", "budget_opex_total"]:
        if cand in df_dept.columns:
            budget_col = cand
            break

    for cand in ["actual_total", "actual", "actual_opex_total"]:
        if cand in df_dept.columns:
            actual_col = cand
            break

    if budget_col is None or actual_col is None or "department" not in df_dept.columns:
        # Can't compute overspend reliably – bail out
        _resolve_dept_overspend_alerts_for_month(client_id, month_start)
        return

    # Aggregate by department
    dept_view = (
        df_dept
        .groupby("department", as_index=False)
        .agg(
            {
                budget_col: "sum",
                actual_col: "sum",
            }
        )
    )
    dept_view = dept_view.rename(
        columns={
            "department": "Department",
            budget_col: "Plan",
            actual_col: "Actual",
        }
    )

    # Compute variance and variance %
    dept_view["Variance"] = dept_view["Actual"] - dept_view["Plan"]

    def _var_pct(row):
        if row["Plan"]:
            return (row["Variance"] / row["Plan"]) * 100.0
        return 0.0

    dept_view["Variance %"] = dept_view.apply(_var_pct, axis=1)

    # Focus on overspends only
    overspends = dept_view[dept_view["Variance"] > 0].copy()
    if overspends.empty:
        _resolve_dept_overspend_alerts_for_month(client_id, month_start)
        return

    # Only those above warning threshold
    overspends = overspends[overspends["Variance %"] >= warn_pct]
    if overspends.empty:
        _resolve_dept_overspend_alerts_for_month(client_id, month_start)
        return

    worst_row = overspends.sort_values("Variance %", ascending=False).iloc[0]
    worst_dept = worst_row["Department"]
    worst_pct = float(worst_row["Variance %"])
    total_overspend = float(overspends["Variance"].sum())
    count_depts = len(overspends)

    # Decide severity
    if worst_pct >= high_pct or total_overspend > 100_000:
        severity = "high"
    else:
        severity = "medium"

    # Avoid duplicates
    try:
        existing = (
            supabase
            .table("alerts")
            .select("id")
            .eq("client_id", str(client_id))
            .eq("alert_type", "dept_overspend")
            .eq("month_date", month_start.isoformat())
            .eq("is_active", True)
            .execute()
        )
        if existing.data:
            return
    except Exception:
        return

    msg = (
        f"{count_depts} team(s) are spending above plan. "
        f"Worst is **{worst_dept}**, about {worst_pct:.0f}% over budget "
        f"(roughly ${total_overspend:,.0f} over across these teams)."
    )

    data = {
        "client_id": str(client_id),
        "page_name": "team_spending",
        "alert_type": "dept_overspend",
        "severity": severity,
        "message": msg,
        "month_date": month_start.isoformat(),
        "context_type": "dept_monthly",
        "context_id": None,
        "is_active": True,
        "is_dismissed": False,
    }

    try:
        supabase.table("alerts").insert(data).execute()
    except Exception:
        pass



def _add_title_only_slide(prs, title_text: str):
    layout = prs.slide_layouts[5]  # Title Only
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title_text
    return slide


def _add_text_box(slide, left_in, top_in, width_in, height_in, text, font_size=14, bold=False):
    left = Inches(left_in)
    top = Inches(top_in)
    width = Inches(width_in)
    height = Inches(height_in)
    tx_box = slide.shapes.add_textbox(left, top, width, height)
    tf = tx_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    return tx_box


def _add_chart_image(slide, fig, left_in, top_in, width_in):
    """Save a matplotlib fig to PNG in-memory and insert as picture."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left_in), Inches(top_in), width=Inches(width_in))

def build_funding_strategy_snapshot(client_id: str, focus_month: datetime.date) -> dict:
    settings = get_client_settings(client_id) or {}
    currency_code, currency_symbol = get_client_currency(client_id)

    engine_df = fetch_cashflow_summary_for_client(client_id)

    runway, eff_burn = compute_runway_and_effective_burn_from_df(
        engine_df=engine_df,
        month_ref=focus_month,
    )

    # Closing cash at focus month (engine)
    closing_cash = 0.0
    if engine_df is not None and not engine_df.empty and "closing_cash" in engine_df.columns:
        tmp = engine_df.copy()
        tmp["month_date"] = pd.to_datetime(tmp["month_date"], errors="coerce").dt.date
        fm = pd.to_datetime(focus_month).to_period("M").to_timestamp().date()
        row = tmp[tmp["month_date"] == fm]
        if not row.empty:
            closing_cash = float(row["closing_cash"].iloc[0] or 0.0)

    # Strategy knobs (saved in client_settings or default)
    target_runway = float(settings.get("target_runway_months", 12) or 12)
    burn_change_pct = float(settings.get("funding_burn_change_pct", 0) or 0)

    adjusted_burn = (float(eff_burn) if eff_burn is not None else 0.0) * (1 + burn_change_pct / 100.0)

    required_cash = adjusted_burn * target_runway if adjusted_burn > 0 else 0.0
    funding_gap = max(0.0, required_cash - closing_cash)

    pre_money = float(settings.get("rough_pre_money", 0.0) or 0.0)
    dilution = None
    if pre_money > 0 and funding_gap > 0:
        dilution = funding_gap / (pre_money + funding_gap) * 100.0

    return {
        "currency_code": currency_code,
        "currency_symbol": currency_symbol,
        "runway_months": runway,
        "effective_burn": eff_burn,
        "target_runway": target_runway,
        "burn_change_pct": burn_change_pct,
        "adjusted_burn": adjusted_burn,
        "closing_cash": closing_cash,
        "funding_gap": funding_gap,
        "pre_money": pre_money,
        "dilution_pct": dilution,
        "engine_df": engine_df,  # keep for chart
    }


def build_funding_strategy_commentary(snapshot: dict) -> list[str]:
    cs = snapshot["currency_symbol"]
    runway = snapshot.get("runway_months")
    burn = snapshot.get("effective_burn")
    adj_burn = snapshot.get("adjusted_burn")
    burn_pct = snapshot.get("burn_change_pct")
    target = snapshot.get("target_runway")
    gap = snapshot.get("funding_gap")
    cash = snapshot.get("closing_cash")
    dilution = snapshot.get("dilution_pct")

    lines = []

    # What happened / where you stand
    if runway is None:
        lines.append("Runway does not hit zero in the forecast horizon (cash stays positive).")
    else:
        lines.append(f"Runway is ~{runway:.1f} months based on the forward cash forecast.")

    if burn is not None:
        lines.append(f"Effective burn is ~{cs}{float(burn):,.0f}/month (next 3-month average).")

    # What changes with the plan
    if burn_pct and burn is not None:
        lines.append(f"With planned burn change ({burn_pct:+.0f}%), adjusted burn is {cs}{float(adj_burn):,.0f}/month.")

    # What decision is required
    if gap and gap > 0:
        lines.append(f"To reach {target:.0f} months runway, estimated funding required is {cs}{float(gap):,.0f}.")
        if dilution is not None:
            lines.append(f"At current pre-money assumption, implied dilution is ~{float(dilution):.1f}% (equity-only).")
        lines.append("Recommendation: align spend to runway target and begin raise prep now (metrics pack + pipeline + milestones).")
    else:
        lines.append(f"No immediate funding gap to reach {target:.0f} months runway at current assumptions.")
        lines.append("Recommendation: raise opportunistically (better terms) and keep collections + hiring discipline tight.")

    return lines[:6]


def add_funding_strategy_slide(
    prs: Presentation,
    client_id: str,
    focus_month: datetime.date,
) -> None:
    snap = build_funding_strategy_snapshot(client_id, focus_month)
    currency_code = snap["currency_code"]
    cs = snap["currency_symbol"]
    engine_df = snap["engine_df"]

    slide = _add_title_only_slide(prs, "5. Funding Strategy – Investing & Financing")

    # ---------- KPI block ----------
    runway = snap.get("runway_months")
    eff_burn = snap.get("effective_burn")
    target = snap.get("target_runway")
    gap = snap.get("funding_gap")
    cash = snap.get("closing_cash")
    dilution = snap.get("dilution_pct")

    kpi_lines = [
        f"Current closing cash: {cs}{float(cash):,.0f}",
        f"Runway (forward): {('—' if runway is None else f'{float(runway):.1f} months')}",
        f"Effective burn: {('—' if eff_burn is None else f'{cs}{float(eff_burn):,.0f} / month')}",
        f"Target runway: {float(target):.0f} months",
        f"Funding required (gap): {cs}{float(gap):,.0f}",
    ]
    if dilution is not None:
        kpi_lines.append(f"Implied dilution (equity-only): ~{float(dilution):.1f}%")

    _add_text_box(
        slide,
        left_in=0.5,
        top_in=1.3,
        width_in=4.8,
        height_in=2.2,
        text="Funding snapshot\n" + "\n".join(f"• {x}" for x in kpi_lines),
        font_size=13,
    )

    # ---------- Chart: investing_cf vs financing_cf next 12 months ----------
    if engine_df is not None and not engine_df.empty and ("investing_cf" in engine_df.columns or "financing_cf" in engine_df.columns):
        df = engine_df.copy()
        df["month_date"] = pd.to_datetime(df.get("month_date"), errors="coerce")
        df = df[df["month_date"].notna()].sort_values("month_date")

        start = pd.to_datetime(focus_month).replace(day=1)
        end = start + pd.DateOffset(months=12)
        df = df[(df["month_date"] >= start) & (df["month_date"] < end)]

        if not df.empty:
            months = df["month_date"].dt.strftime("%b %y").tolist()
            inv = pd.to_numeric(df.get("investing_cf", 0), errors="coerce").fillna(0.0).tolist()
            fin = pd.to_numeric(df.get("financing_cf", 0), errors="coerce").fillna(0.0).tolist()

            x = np.arange(len(months))
            fig, ax = plt.subplots(figsize=(7.8, 3.0))
            ax.bar(x - 0.2, inv, width=0.4, label="Investing CF")
            ax.bar(x + 0.2, fin, width=0.4, label="Financing CF")
            ax.axhline(0, linewidth=1)
            ax.set_title("Investing vs Financing cashflows (next 12 months)")
            ax.set_ylabel(f"Cashflow ({currency_code})")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()

            _add_chart_image(slide, fig, left_in=5.4, top_in=1.3, width_in=4.7)
        else:
            _add_text_box(slide, 5.4, 1.3, 4.7, 2.0, "No engine rows in the next 12 months window.", font_size=13)
    else:
        _add_text_box(slide, 5.4, 1.3, 4.7, 2.0, "No investing/financing cashflow columns in cashflow_summary yet.", font_size=13)

    # ---------- Commentary ----------
    lines = build_funding_strategy_commentary(snap)
    _add_text_box(
        slide,
        left_in=0.5,
        top_in=3.8,
        width_in=9.6,
        height_in=2.4,
        text="Board commentary\n" + "\n".join(f"• {l}" for l in lines),
        font_size=13,
    )


def generate_board_pack_pptx(client_id: str, focus_month: datetime.date) -> str:
    settings = get_client_settings(client_id) or {}
    currency_code, currency_symbol = get_client_currency(client_id)

    kpis = fetch_kpis_for_client_month(client_id, focus_month) or {}
    kpi_change = build_kpi_change_explanations(client_id, focus_month)
    deltas = kpi_change.get("deltas", {})
    expl = kpi_change.get("explanations", {})

    month_label = pd.to_datetime(focus_month).strftime("%b %Y")

    rev_sched = build_revenue_schedule_for_client(client_id, base_month=focus_month, n_months=12)

    cash_df = fetch_cashflow_summary_for_client(client_id)
    df_pipeline = fetch_pipeline_for_client(client_id)
    df_positions = fetch_payroll_positions_for_client(client_id)
    df_dept = fetch_dept_monthly_for_client(client_id)
    df_ar, df_ap = fetch_ar_ap_for_client(client_id)

    # Runway + burn from engine (use function params!)
    engine_df = cash_df
    runway_from_engine, effective_burn = compute_runway_and_effective_burn_from_df(
        engine_df=engine_df,
        month_ref=focus_month,
    )

    month_summary = build_month_summary_insight(client_id, focus_month)

    prs = Presentation()

    # Cover
    cover_layout = prs.slide_layouts[0]
    cover = prs.slides.add_slide(cover_layout)
    cover.shapes.title.text = "Board Pack"
    subtitle = cover.placeholders[1].text_frame
    subtitle.text = f"Client: {client_id}\nPeriod: {month_label}"

    # 1) Exec Summary
    slide = _add_title_only_slide(prs, "1. Executive Summary")

    revenue_this = float(kpis.get("revenue") or 0.0)
    burn_this = float(kpis.get("burn") or 0.0)
    cash_balance = float(kpis.get("cash_balance") or 0.0)
    runway_months = runway_from_engine if runway_from_engine is not None else kpis.get("runway_months")

    kpi_text = [
        f"Revenue ({month_label}): {currency_symbol}{revenue_this:,.0f}",
        f"Burn ({month_label}): {currency_symbol}{burn_this:,.0f}",
        f"Cash in bank: {currency_symbol}{cash_balance:,.0f}",
    ]
    if runway_months is not None:
        kpi_text.append(f"Runway (forward-looking): {float(runway_months):.1f} months")
    if effective_burn is not None:
        kpi_text.append(f"Effective burn (avg): {currency_symbol}{float(effective_burn):,.0f} / month")

    _add_text_box(slide, 0.5, 1.3, 4.5, 2.0, "Key KPIs\n" + "\n".join(f"• {line}" for line in kpi_text), font_size=14)

    alerts = fetch_alerts_for_client(client_id, only_active=True, limit=50) or []
    key_risks, key_opps = [], []
    for a in alerts:
        sev = str(a.get("severity", "medium")).lower()
        msg = a.get("message", "")
        if sev in ["high", "critical"]:
            key_risks.append(msg)
        else:
            key_opps.append(msg)
    if not key_risks:
        key_risks.append("No high / critical risks currently flagged.")
    if not key_opps:
        key_opps.append("Focus on executing pipeline and improving collections.")

    _add_text_box(slide, 5.2, 1.3, 4.5, 1.5, "Key risks\n" + "\n".join(f"• {r}" for r in key_risks[:5]), font_size=13)
    _add_text_box(slide, 5.2, 3.0, 4.5, 1.5, "Key opportunities\n" + "\n".join(f"• {o}" for o in key_opps[:5]), font_size=13)

    exec_lines = []
    if expl.get("money_in"):
        exec_lines.append(f"Revenue: {expl['money_in']}")
    if expl.get("money_out"):
        exec_lines.append(f"Burn: {expl['money_out']}")
    if expl.get("cash"):
        exec_lines.append(f"Cash: {expl['cash']}")
    if expl.get("runway"):
        exec_lines.append(f"Runway: {expl['runway']}")
    exec_lines.append(month_summary)

    _add_text_box(
        slide,
        0.5,
        3.7,
        9.2,
        2.5,
        "Executive commentary\n" + "\n".join(f"• {line}" for line in exec_lines[:6]),
        font_size=13,
    )
# ------------------------------------------------------------------
    # 3) Revenue Summary slide – with chart
    # ------------------------------------------------------------------
    slide = _add_title_only_slide(prs, "2. Revenue Summary – Sales & Deals")

    if rev_sched is not None and not rev_sched.empty:
        rev_df = rev_sched.copy()
        rev_df["month_date"] = pd.to_datetime(rev_df["month_date"], errors="coerce")
        rev_df = rev_df[rev_df["month_date"].notna()]

        rev_col = None
        for cand in ["recognised_revenue", "revenue_amount", "amount"]:
            if cand in rev_df.columns:
                rev_col = cand
                break

        if rev_col:
            monthly = (
                rev_df.groupby("month_date", as_index=False)[rev_col]
                .sum()
                .sort_values("month_date")
            )

            # Split last 6 / next 6 around focus_month
            focus_start = pd.to_datetime(focus_month).replace(day=1)
            past = monthly[monthly["month_date"] <= focus_start].tail(6)
            future = monthly[monthly["month_date"] > focus_start].head(6)

            # Build line chart (12-month view)
            fig, ax = plt.subplots(figsize=(7.5, 3))
            ax.plot(monthly["month_date"], monthly[rev_col], marker="o")
            ax.axvline(focus_start, linestyle="--", linewidth=1)
            ax.set_title("Recognised revenue – last 6 & next 6 months")
            ax.set_ylabel(f"Revenue ({currency_code})")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            _add_chart_image(slide, fig, left_in=0.4, top_in=1.3, width_in=6.5)

            # Text summary on right
            rev_text_lines = []
            if not past.empty:
                last_val = float(past[rev_col].iloc[-1])
                rev_text_lines.append(
                    f"Latest month recognised revenue: {currency_symbol}{last_val:,.0f}"
                )
            if not future.empty:
                future_total = float(future[rev_col].sum())
                rev_text_lines.append(
                    f"Next 6 months forecast total: {currency_symbol}{future_total:,.0f}"
                )

            # Pipeline summary
            total_pipeline = 0.0
            weighted_pipeline = 0.0
            if df_pipeline is not None and not df_pipeline.empty:
                tmp = df_pipeline.copy()
                tmp["value_total"] = pd.to_numeric(
                    tmp.get("value_total"), errors="coerce"
                ).fillna(0.0)
                tmp["probability_pct"] = pd.to_numeric(
                    tmp.get("probability_pct"), errors="coerce"
                ).fillna(0.0)
                total_pipeline = float(tmp["value_total"].sum())
                weighted_pipeline = float(
                    (tmp["value_total"] * tmp["probability_pct"] / 100.0).sum()
                )
                rev_text_lines.append(
                    f"Total pipeline: {currency_symbol}{total_pipeline:,.0f}"
                )
                rev_text_lines.append(
                    f"Weighted pipeline: {currency_symbol}{weighted_pipeline:,.0f}"
                )

            if total_pipeline > 0:
                qual = (weighted_pipeline / total_pipeline) * 100.0
                rev_text_lines.append(
                    f"Pipeline quality (weighted/gross): {qual:.1f}%"
                )

            rev_commentary = [
                "Revenue commentary",
                "• Recent revenue plus weighted pipeline gives visibility into the next 3–6 months.",
            ]
            if future is not None and not future.empty:
                rev_commentary.append(
                    "• Check if the forecast is driven by a few large deals – monitor conversion risk."
                )
            if total_pipeline > 0:
                rev_commentary.append(
                    "• Consider whether pipeline coverage is at least 3× your target revenue."
                )

            right_block = "\n".join(
                ["Key numbers"] + [f"• {l}" for l in rev_text_lines] + [""] + rev_commentary
            )

            _add_text_box(slide, 7.2, 1.3, 3.0, 3.5, right_block, font_size=12)

        else:
            _add_text_box(
                slide,
                0.5,
                1.5,
                9.0,
                3.0,
                "No recognised revenue schedule yet – add deals and revenue methods to see a full revenue summary.",
                font_size=14,
            )

    # ------------------------------------------------------------------
    # 4) Team & Spending slide – payroll chart + commentary
    # ------------------------------------------------------------------
    slide = _add_title_only_slide(prs, "3. Team & Spending")

    # Payroll trend for next 6 months (same logic as dashboard)
    months_6 = pd.date_range(start=pd.to_datetime(focus_month), periods=6, freq="MS")
    payroll_by_month = compute_payroll_by_month(client_id, months_6)

    if payroll_by_month:
        pay_vals = [float(payroll_by_month.get(m, 0.0)) for m in months_6]

        fig, ax = plt.subplots(figsize=(7.5, 3))
        ax.plot(months_6, pay_vals, marker="o")
        ax.set_title("Payroll cash-out – next 6 months")
        ax.set_ylabel(f"Payroll ({currency_code})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        _add_chart_image(slide, fig, left_in=0.4, top_in=1.3, width_in=6.5)

        avg_payroll = sum(pay_vals) / len(pay_vals) if pay_vals else 0.0

        # Relative to current revenue
        revenue_this = revenue_this or 0.0
        payroll_vs_rev = (avg_payroll / revenue_this * 100.0) if revenue_this else None

        lines = [
            f"Average monthly payroll (next 6 months): {currency_symbol}{avg_payroll:,.0f}",
        ]
        if payroll_vs_rev is not None:
            lines.append(f"Payroll as % of current revenue: {payroll_vs_rev:.0f}%")

        # Headcount from positions
        headcount_fte = 0.0
        if df_positions is not None and not df_positions.empty:
            pos = df_positions.copy()
            pos["start_date"] = pd.to_datetime(pos.get("start_date"), errors="coerce")
            pos["end_date"] = pd.to_datetime(pos.get("end_date"), errors="coerce")
            month_start = pd.to_datetime(focus_month)

            active_mask = (pos["start_date"] <= month_start) & (
                pos["end_date"].isna() | (pos["end_date"] >= month_start)
            )
            pos_active = pos[active_mask]
            if not pos_active.empty:
                headcount_fte = float(pos_active.get("fte", 0.0).fillna(0.0).sum())

        if headcount_fte > 0:
            lines.append(f"Active headcount this month: {headcount_fte:.1f} FTE")

        commentary = [
            "Team & opex commentary",
            "• Check whether payroll growth is in line with revenue growth.",
            "• Identify any high-cost roles that are not clearly linked to revenue or product milestones.",
            "• Review departments that are consistently over budget.",
        ]

        text_block = "\n".join(
            ["Key numbers"] + [f"• {l}" for l in lines] + [""] + commentary
        )

        _add_text_box(slide, 7.2, 1.3, 3.0, 3.5, text_block, font_size=12)
    else:
        _add_text_box(
            slide,
            0.5,
            1.5,
            9.0,
            3.0,
            "No payroll data yet – add roles in the Team & Spending page to see headcount and payroll trends.",
            font_size=14,
        )

    # ------------------------------------------------------------------
    # 5) Cash Position & Forecast – cash engine chart + AR/AP snapshot
    # ------------------------------------------------------------------
    slide = _add_title_only_slide(prs, "4. Cash Position & Cash Forecast")

    if cash_df is not None and not cash_df.empty:
        df_cash = cash_df.copy()
        df_cash["month_date"] = pd.to_datetime(df_cash["month_date"], errors="coerce")
        df_cash = df_cash[df_cash["month_date"].notna()].sort_values("month_date")

        fig, ax = plt.subplots(figsize=(7.5, 3))
        if "closing_cash" in df_cash.columns:
            ax.plot(df_cash["month_date"], df_cash["closing_cash"], marker="o")
            ax.set_ylabel(f"Closing cash ({currency_code})")
        if "free_cash_flow" in df_cash.columns:
            ax.bar(
                df_cash["month_date"],
                df_cash["free_cash_flow"],
                alpha=0.3,
            )
        ax.set_title("Monthly cash forecast (engine)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        _add_chart_image(slide, fig, left_in=0.4, top_in=1.3, width_in=6.5)

        cash_today = cash_balance
        cliff = None
        if "closing_cash" in df_cash.columns:
            below_zero = df_cash[df_cash["closing_cash"] <= 0]
            if not below_zero.empty:
                cliff = below_zero["month_date"].iloc[0].date()

        wc_lines = [f"Cash today: {currency_symbol}{cash_today:,.0f}"]
        if cliff:
            wc_lines.append(f"Cash cliff (engine): {cliff.strftime('%d %b %Y')}")

        # Working capital from AR / AP
        ar_out = 0.0
        ap_out = 0.0
        if df_ar is not None and not df_ar.empty:
            df_ar2 = df_ar.copy()
            df_ar2["amount"] = pd.to_numeric(df_ar2.get("amount"), errors="coerce").fillna(0.0)
            ar_out = float(df_ar2["amount"].sum())
        if df_ap is not None and not df_ap.empty:
            df_ap2 = df_ap.copy()
            df_ap2["amount"] = pd.to_numeric(df_ap2.get("amount"), errors="coerce").fillna(0.0)
            ap_out = float(df_ap2["amount"].sum())

        wc = ar_out - ap_out
        wc_lines.append(f"AR outstanding: {currency_symbol}{ar_out:,.0f}")
        wc_lines.append(f"AP outstanding: {currency_symbol}{ap_out:,.0f}")
        wc_lines.append(f"Net working capital (AR – AP): {currency_symbol}{wc:,.0f}")

        cash_comment = [
            "Cash commentary",
            "• If cash cliff is within 6–9 months, begin funding or cost-reduction planning now.",
            "• Use AR and AP levers (collections, payment terms) before cutting critical growth spend.",
        ]
        if cliff:
            cash_comment.insert(
                1,
                f"• Cash cliff expected around {cliff.strftime('%b %Y')} if no changes are made.",
            )

        text_block = "\n".join(
            ["Key numbers"] + [f"• {l}" for l in wc_lines] + [""] + cash_comment
        )
        _add_text_box(slide, 7.2, 1.3, 3.0, 3.5, text_block, font_size=12)
    else:
        _add_text_box(
            slide,
            0.5,
            1.5,
            9.0,
            3.0,
            "No cashflow engine data yet – rebuild cashflow from the Business at a Glance page.",
            font_size=14,
        )



    # ✅ NEW 5) Funding Strategy – replaces Scenario + old I&F slide
    add_funding_strategy_slide(prs, client_id=client_id, focus_month=focus_month)

    # 6) Risks, Issues & Notes (renumbered)
    slide = _add_title_only_slide(prs, "6. Risks, Issues & Notes")

    risk_lines = []
    for a in alerts:
        sev = str(a.get("severity", "medium")).title()
        msg = a.get("message", "")
        owner = a.get("owner") or a.get("owner_name") or "Unassigned"
        risk_lines.append(f"• [{sev}] {msg}  (Owner: {owner})")

    if not risk_lines:
        risk_lines = ["• No major risks recorded – confirm this with the founder."]

    _add_text_box(slide, 0.5, 1.3, 9.0, 4.5, "Top risks & issues\n" + "\n".join(risk_lines[:8]), font_size=13)

    safe_month = pd.to_datetime(focus_month).strftime("%Y%m")
    filename = f"BoardPack_{client_id}_{safe_month}.pptx"
    file_path = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    prs.save(file_path)
    return file_path


# ---------- Page: Alerts & To-Dos ----------
def page_alerts_todos():
    """
    Collaboration Hub for the founder & team:
    - Snapshot of key alerts (runway, cash danger, overdue AR, overspend, etc)
    - One central task board (collaboration_hub) for cross-page work, with filters
    - Tabs to collaborate on each main page (comments + tasks together)
    """
    top_header("Collaboration hub")

    # -------------------------------
    # Basic guards
    # -------------------------------
    if not selected_client_id:
        st.info("Select a business from the sidebar to see collaboration items.")
        return

    if not selected_month_start:
        st.info("Pick a focus month in the navbar to see alerts and tasks for that period.")
        return

# --- Board pack download block (top of the page) ---
    if selected_client_id:
        if st.button("Download board pack (PowerPoint)", key="btn_board_pack"):
            file_path = generate_board_pack_pptx(selected_client_id, selected_month_start)
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Click to download board pack",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="dl_board_pack",
                )


    # -------------------------------
    # Keep automatic alerts in sync
    # -------------------------------
    try:
        ensure_runway_alert_for_month(selected_client_id, selected_month_start)
    except Exception as e:
        st.warning("Could not refresh runway alert for this month.")
        st.caption(str(e))

    try:
        ensure_cash_danger_alert_for_month(selected_client_id, selected_month_start)
    except Exception as e:
        st.warning("Could not refresh cash danger alert for this month.")
        st.caption(str(e))

    # Optionally keep AR + dept overspend alerts in sync too
    try:
        ensure_overdue_ar_alert(
            selected_client_id,
            as_of=date.today(),
            min_days_overdue=0,
        )
    except Exception:
        pass

    try:
        ensure_dept_overspend_alert(selected_client_id, selected_month_start)
    except Exception:
        pass

    # -------------------------------
    # 🔥 Key alerts snapshot
    # -------------------------------
    st.subheader("🚨 Key alerts (snapshot)")

    alerts = fetch_alerts_for_client(
        selected_client_id,
        only_active=True,
        limit=100,
    )

    if not alerts:
        st.caption("No open alerts right now. 🎉")
    else:
        # Try to focus on this month; if nothing, fall back to all
        alerts_for_month = []
        for a in alerts:
            month_value = a.get("month_date")
            if month_value and selected_month_start:
                try:
                    mdt = pd.to_datetime(month_value).date().replace(day=1)
                    if mdt == selected_month_start:
                        alerts_for_month.append(a)
                except Exception:
                    alerts_for_month.append(a)
            else:
                alerts_for_month.append(a)

        if not alerts_for_month:
            alerts_for_month = alerts

        alerts_sorted = sort_alerts_by_severity(alerts_for_month)

        # Show only the top 4 in detail to keep this clean
        top_alerts = alerts_sorted[:4]

        col_a1, col_a2 = st.columns(2)
        alert_cols = [col_a1, col_a2]
        idx = 0

        for alert in top_alerts:
            sev = str(alert.get("severity", "medium")).lower()
            atype = alert.get("alert_type", "alert")
            msg = alert.get("message", "")
            created = alert.get("created_at", "")
            page_name = alert.get("page_name") or "general"

            label = f"[{atype}] {msg}"
            meta = f"Page: {page_name} · Created: {created}"

            if sev in ["critical", "high"]:
                box = alert_cols[idx % 2].error
            elif sev == "medium":
                box = alert_cols[idx % 2].warning
            else:
                box = alert_cols[idx % 2].info

            box(label)
            alert_cols[idx % 2].caption(meta)
            idx += 1

        # Small meta summary
        total_crit = sum(
            1
            for a in alerts_for_month
            if str(a.get("severity", "")).lower() in ["critical", "high"]
        )
        total_med = sum(
            1
            for a in alerts_for_month
            if str(a.get("severity", "")).lower() == "medium"
        )
        total_low = len(alerts_for_month) - total_crit - total_med

        st.caption(
            f"Summary this period: {total_crit} high/critical · "
            f"{total_med} medium · {total_low} low/other alerts."
        )

    st.markdown("---")

    # -------------------------------
    # 🧩 Central workboard (Kanban-style)
    # -------------------------------
    st.subheader("✅ My workboard (cross-page tasks)")

    st.caption(
        "See tasks from all pages in one place. "
        "Filter by status, page or owner to focus your week."
    )

    # ----- Filters row -----
    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 1.6])

    with col_f1:
        status_filter = st.multiselect(
            "Status",
            options=["open", "in_progress", "done", "cancelled"],
            default=["open", "in_progress"],  # hide done/cancelled by default
            key="collab_status_filter",
        )

    with col_f2:
        page_options = get_task_page_options(selected_client_id)
        page_filter = st.multiselect(
            "Page",
            options=page_options,
            default=[],
            help="Limit to tasks from specific pages (or leave blank for all).",
            key="collab_page_filter",
        )

    with col_f3:
        owner_options = get_task_owner_options(selected_client_id)
        owner_filter = st.multiselect(
            "Owner",
            options=owner_options,
            default=[],
            help="Focus on tasks assigned to specific people.",
            key="collab_owner_filter",
        )

    search_text = st.text_input(
        "Search in titles / descriptions",
        value="",
        key="collab_search_text",
    )

    # ----- Fetch tasks from Supabase -----
    df_tasks = fetch_tasks_for_collab_hub(
        client_id=selected_client_id,
        status_filter=status_filter or None,
        page_filter=page_filter or None,
        owner_filter=owner_filter or None,
        search_text=search_text,
    )

    if df_tasks.empty:
        st.info("No tasks match the current filters.")
    else:
        # Map page_name to friendly labels
        page_label_map = {
            "business_overview": "Business overview",
            "sales_deals": "Sales & deals",
            "team_spending": "Team & spending",
            "cash_bills": "Cash & bills",
            "alerts_todos": "Alerts & to-dos",
            "collaboration_hub": "Collaboration Hub",
        }
        df_tasks = df_tasks.copy()
        df_tasks["page_label"] = df_tasks["page_name"].map(page_label_map).fillna("Other")

        # Priority label
        def _prio_label(p: str) -> str:
            p = (p or "").lower()
            if p == "low":
                return "Low"
            if p == "normal":
                return "Normal"
            if p == "high":
                return "High"
            if p == "critical":
                return "Critical"
            return "Normal"

        df_tasks["priority_label"] = df_tasks["priority"].apply(_prio_label)

        # Kanban columns by status
        statuses_order = ["open", "in_progress", "done", "cancelled"]
        status_headers = {
            "open": "📝 To-do",
            "in_progress": "🚧 In progress",
            "done": "✅ Done",
            "cancelled": "🗑️ Cancelled",
        }

        cols = st.columns(4)

        for idx, status in enumerate(statuses_order):
            col = cols[idx]
            subset = df_tasks[df_tasks["status"] == status]

            with col:
                col.markdown(f"**{status_headers.get(status, status.title())}**")

                if subset.empty:
                    col.caption("No tasks.")
                    continue

                # Sort by due date (nulls last), then priority, then created_at
                subset = subset.sort_values(
                    by=["due_date", "priority", "created_at"],
                    ascending=[True, False, True],
                )

                for _, row in subset.iterrows():
                    title = row.get("title", "Untitled task")
                    page_label = row.get("page_label", "Unknown")
                    owner = row.get("owner_name") or "Unassigned"
                    due = row.get("due_date")
                    priority = row.get("priority_label", "Normal")

                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        meta_parts = [page_label, owner, f"Priority: {priority}"]
                        if due:
                            meta_parts.append(f"Due: {due}")
                        st.caption(" · ".join(meta_parts))

                        # Later you can add buttons here (Mark done, etc.)

    st.markdown("---")

    # -------------------------------
    # 🧵 Page-by-page collaboration
    # -------------------------------
    st.subheader("📂 Page collaboration (comments + tasks together)")

    st.caption(
        "Drill into discussions and tasks for each page without leaving this hub. "
        "Use this when you want context-specific work but from one central place."
    )

    tab_bo, tab_sales, tab_team, tab_cash = st.tabs(
        [
            "Business overview",
            "Sales & deals",
            "Team & spending",
            "Cash & bills",
        ]
    )

    # ---- Business overview ----
    with tab_bo:
        col_c, col_t = st.columns([3, 1])
        with col_c:
            st.markdown("#### 💬 Comments – Business overview")
            comments_block("business_overview")
        with col_t:
            st.markdown("#### ✅ Tasks – Business overview")
            with st.expander("View & update tasks", expanded=False):
                tasks_block("business_overview")

    # ---- Sales & deals ----
    with tab_sales:
        col_c, col_t = st.columns([3, 1])
        with col_c:
            st.markdown("#### 💬 Comments – Sales & deals")
            comments_block("sales_deals")
        with col_t:
            st.markdown("#### ✅ Tasks – Sales & deals")
            with st.expander("View & update tasks", expanded=False):
                tasks_block("sales_deals")

    # ---- Team & spending ----
    with tab_team:
        col_c, col_t = st.columns([3, 1])
        with col_c:
            st.markdown("#### 💬 Comments – Team & spending")
            comments_block("team_spending")
        with col_t:
            st.markdown("#### ✅ Tasks – Team & spending")
            with st.expander("View & update tasks", expanded=False):
                tasks_block("team_spending")

    # ---- Cash & bills ----
    with tab_cash:
        col_c, col_t = st.columns([3, 1])
        with col_c:
            st.markdown("#### 💬 Comments – Cash & bills")
            comments_block("cash_bills")
        with col_t:
            st.markdown("#### ✅ Tasks – Cash & bills")
            with st.expander("View & update tasks", expanded=False):
                tasks_block("cash_bills")


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------


def run_app():


    page_names = [
        "Business at a Glance",
        "Investing & Financing Strategy",
        "Sales & Deals",
        "Team & Spending",
        "Cash & Bills",
        "Collaboration Hub",
        "Client Settings",
    ]

    page_choice = st.sidebar.radio("Go to page", page_names)

    if page_choice == "Business at a Glance":
        page_business_overview()
    elif page_choice == "Investing & Financing Strategy":
        page_investing_financing()
    elif page_choice == "Sales & Deals":
        page_sales_deals()
    elif page_choice == "Team & Spending":
        page_team_spending()
    elif page_choice == "Cash & Bills":
        page_cash_bills()    
    elif page_choice == "Collaboration Hub":
        page_alerts_todos()    
    elif page_choice == "Client Settings":
        page_client_settings()


if __name__ == "__main__":
    run_app()



import time
import traceback
import pandas as pd

def supabase_select_with_debug(table: str, client_id: str, order_col: str | None = None, max_retries: int = 3):
    """
    Returns (df, meta) where meta contains debug info about failures.
    """
    meta = {
        "table": table,
        "client_id": client_id,
        "attempts": [],
        "ok": False,
        "row_count": 0,
        "error": None,
    }

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            q = supabase.table(table).select("*").eq("client_id", str(client_id))
            if order_col:
                q = q.order(order_col)

            res = q.execute()
            rows = res.data or []
            df = pd.DataFrame(rows)

            meta["ok"] = True
            meta["row_count"] = len(df)
            meta["attempts"].append({
                "attempt": attempt,
                "ms": int((time.time() - t0) * 1000),
                "status": "ok",
                "rows": len(df),
            })
            return df, meta

        except Exception as e:
            meta["error"] = str(e)
            meta["attempts"].append({
                "attempt": attempt,
                "ms": int((time.time() - t0) * 1000),
                "status": "fail",
                "error": str(e),
            })

            print(f"[WARN] fetch {table} attempt {attempt}/{max_retries} failed:", e)
            # small backoff
            time.sleep(0.4 * attempt)

    # all attempts failed
    return pd.DataFrame(), meta

# ---------- Board Pack Helper ----------


def _safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _safe_pct(numerator, denominator):
    try:
        if not denominator:
            return None
        return float(numerator) / float(denominator) * 100.0
    except Exception:
        return None


def _month_start(d: date) -> date:
    return pd.to_datetime(d).to_period("M").to_timestamp().date()

