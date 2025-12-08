import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
from supabase import create_client
import altair as alt
import calendar


# ---------- Supabase config ----------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase_client()

st.set_page_config(page_title="FFIS – Founder Financial Intelligence", layout="wide")


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

    settings = get_client_settings(selected_client_id)

    st.subheader("💱 Currency Settings")
    render_currency_settings_section(selected_client_id)

    st.markdown("---")

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

@st.cache_data(ttl=60)
def compute_payroll_by_month(client_id, _month_index) -> dict:
    """
    Compute monthly payroll cash-out for each month in _month_index.

    Uses payroll_positions schema:

      - base_salary_annual is the base salary at 1.0 FTE
      - fte scales salary (0.5, 1.0, 1.2 etc.)
      - super_rate_pct and payroll_tax_pct are on top (if not contractor)
      - start_date / end_date control when the role is active

    Assumptions:
      - Payroll is spread evenly over 12 months (base_salary_annual / 12)
    """
    if client_id is None or _month_index is None:
        return {}

    df = fetch_payroll_positions_for_client(client_id)
    if df is None or df.empty:
        return {}

    # Normalise month index to first-of-month timestamps
    month_index = pd.to_datetime(_month_index).to_period("M").to_timestamp()
    payroll_by_month: dict = {m: 0.0 for m in month_index}

    for _, row in df.iterrows():
        base_annual = float(row.get("base_salary_annual") or 0.0)
        if base_annual <= 0:
            continue

        fte = float(row.get("fte") or 1.0)
        is_contractor = bool(row.get("is_contractor", False))

        super_pct = float(row.get("super_rate_pct") or 0.0) / 100.0
        tax_pct = float(row.get("payroll_tax_pct") or 0.0) / 100.0

        start = row.get("start_date")
        end = row.get("end_date")

        # 🔒 Normalise start/end: convert Timestamp -> date, NaT/NaN -> None
        if isinstance(start, pd.Timestamp):
            start = start.date()
        if isinstance(end, pd.Timestamp):
            end = end.date()

        if pd.isna(start):
            start = None
        if pd.isna(end):
            end = None

        # Monthly base salary at given FTE
        base_monthly = (base_annual / 12.0) * fte

        for m in month_index:
            month_start = m.date()
            month_end = (m + pd.offsets.MonthEnd(0)).date()

            # Active only if role overlaps this month
            if start is not None and start > month_end:
                continue
            if end is not None and end < month_start:
                continue

            if is_contractor:
                monthly_cash = base_monthly
            else:
                super_amt = base_monthly * super_pct
                tax_amt = base_monthly * tax_pct
                monthly_cash = base_monthly + super_amt + tax_amt

            payroll_by_month[m] = payroll_by_month.get(m, 0.0) + monthly_cash

    return payroll_by_month


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

def fetch_cashflow_summary_for_client(client_id):
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("cashflow_summary")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date")
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        print("Error fetching cashflow_summary:", e)
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
        df = df[df["month_date"].notna()].sort_values("month_date")

    return df

def fetch_kpis_for_client_month(
    client_id: str,
    month_date: date | datetime,
) -> dict | None:
    """
    Get a single KPI row for (client, month_date) from kpi_monthly.

    If nothing exists, derive a KPI row from:
      - AR (cash in)
      - AP + payroll (cash out)
      - cashflow_summary (closing cash + runway)
    and just RETURN it (no upsert, to avoid ON CONFLICT errors).
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

    # ---------- 2) Derive KPIs from AR/AP + payroll + engine ----------
    try:
        # Normalised month as Timestamp and plain date
        base_month_ts = pd.to_datetime(month_dt).replace(day=1)
        as_of_date = base_month_ts.date()

        # Settings + AR/AP
        settings = get_client_settings(client_id)
        ar_days = int(settings.get("ar_default_days", 30))
        ap_days = int(settings.get("ap_default_days", 30))

        df_ar, df_ap = fetch_ar_ap_for_client(client_id)

        # ----- AR: expected cash-in this month -----
        money_in = 0.0
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
            df_ap["amount"] = pd.to_numeric(
                df_ap.get("amount"), errors="coerce"
            ).fillna(0.0)

            if "status" in df_ap.columns:
                df_ap = df_ap[
                    ~df_ap["status"].str.lower().isin(["paid", "closed", "settled"])
                ]

            mask_ap = df_ap[pay_col].dt.to_period("M") == base_month_ts.to_period("M")
            ap_cash = float(df_ap.loc[mask_ap, "amount"].sum())

        # ----- Payroll this month -----
        payroll_by_month = compute_payroll_by_month(client_id, [base_month_ts])
        payroll_cash = float(payroll_by_month.get(base_month_ts, 0.0))

        # (Optional) extra opex table, if you’ve created it
        opex_cash = 0.0
        try:
            df_opex = fetch_opex_for_client(client_id)
            if df_opex is not None and not df_opex.empty:
                df_opex = df_opex.copy()
                df_opex["month_bucket"] = (
                    pd.to_datetime(df_opex["month_date"], errors="coerce")
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
                mask_ox = df_opex["month_bucket"] == base_month_ts
                opex_cash = float(df_opex.loc[mask_ox, "amount"].sum())
        except Exception:
            # If opex table not present yet, just treat as 0
            opex_cash = 0.0

        money_out = ap_cash + payroll_cash + opex_cash

        # ----- Closing cash from engine -----
        closing_cash = 0.0
        engine_df = fetch_cashflow_summary_for_client(client_id)
        if engine_df is not None and not engine_df.empty:
            df_eng = engine_df.copy()
            df_eng["month_date"] = pd.to_datetime(df_eng["month_date"], errors="coerce").dt.date
            row_eng = df_eng[df_eng["month_date"] == month_dt]
            if not row_eng.empty:
                closing_cash = float(row_eng.get("closing_cash", 0.0).iloc[0] or 0.0)

        # ----- Runway + effective burn from engine history -----
        runway_months, effective_burn = compute_runway_and_effective_burn(
            client_id=client_id,
            focus_month=month_dt,
        )

        kpi_row = {
            "client_id": client_id,
            "month_date": month_iso,
            "revenue": money_in,
            "burn": money_out,
            "cash_balance": closing_cash,
            "runway_months": runway_months,
        }

        print(
            f"Derived KPIs from cashflow_summary/AR/AP for client={client_id}, "
            f"month_date={month_iso}: {kpi_row}"
        )

        # 🟢 IMPORTANT: we just return, no upsert => no 42P10
        return kpi_row

    except Exception as e:
        print("Error deriving KPIs from cashflow_summary:", e)
        return None


@st.cache_data(ttl=60)
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

def page_sales_deals():
    top_header("Sales & Deals")

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

    # ---------- Pipeline overview ----------
    with st.spinner("Loading your pipeline..."):
        df_pipeline = fetch_pipeline_for_client(selected_client_id)

    st.subheader("📂 Pipeline overview")

    if df_pipeline.empty:
        st.info("No deals in the pipeline yet for this business.")
    else:
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
        st.dataframe(view, use_container_width=True)

        total_pipeline = df_pipeline["value_total"].fillna(0).sum()
        weighted_pipeline = (
            df_pipeline["value_total"].fillna(0)
            * df_pipeline["probability_pct"].fillna(0)
            / 100
        ).sum()
        open_deals = len(df_pipeline)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Total pipeline", f"${total_pipeline:,.0f}")
        with k2:
            st.metric("Weighted pipeline", f"${weighted_pipeline:,.0f}")
        with k3:
            st.metric("Open deals", open_deals)

        # ---------- Edit deal status / probability + milestones ----------
        st.markdown("---")
        st.subheader("✏️ Update deal status / probability")

        deal_options = []
        deal_id_map = {}
        for _, r in df_pipeline.iterrows():
            deal_id = r.get("id")
            name = r.get("deal_name", "Unnamed")
            customer = r.get("customer_name") or ""
            stage = r.get("stage", "")
            prob = r.get("probability_pct", 0)
            label = f"{name} – {customer}  [{stage}, {prob}%]"
            deal_options.append(label)
            deal_id_map[label] = deal_id

        if not deal_options:
            st.caption("No deals to update.")
        else:
            selected_label = st.selectbox(
                "Pick a deal to update",
                ["-- Select --"] + deal_options,
                key="pipeline_edit_deal",
            )

            if selected_label != "-- Select --":
                deal_id = deal_id_map[selected_label]
                current_row = df_pipeline[df_pipeline["id"] == deal_id].iloc[0]
                current_stage = str(current_row.get("stage", "proposal"))
                current_prob = float(current_row.get("probability_pct") or 0.0)

                col_ed1, col_ed2 = st.columns(2)
                with col_ed1:
                    new_stage = st.selectbox(
                        "New stage",
                        ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"],
                        index=[
                            "idea",
                            "proposal",
                            "demo",
                            "contract",
                            "closed_won",
                            "closed_lost",
                        ].index(
                            current_stage
                            if current_stage
                            in ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"]
                            else "proposal"
                        ),
                        key="pipeline_new_stage",
                    )
                with col_ed2:
                    new_prob = st.slider(
                        "Probability (%)",
                        min_value=0,
                        max_value=100,
                        value=int(current_prob),
                        key="pipeline_new_prob",
                    )

                if st.button("Save changes", key="pipeline_save_changes"):
                    ok = update_pipeline_deal_status(deal_id, new_stage, new_prob)
                    if ok:
                        st.success("Deal updated.")
                        st.rerun()
                    else:
                        st.error("Could not update deal. Please try again.")

                # ---------- Milestone editor (only if client uses milestone recognition) ----------
                if selected_method_code == "milestone":
                    st.markdown("#### 🧱 Milestones for this deal")

                    ms_df = fetch_milestones_for_deal(selected_client_id, deal_id)

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
                        key=f"milestones_editor_{deal_id}",
                        use_container_width=True,
                    )

                    st.caption(
                        "Add a row per milestone. "
                        "Percent of contract can total 100%, or we’ll normalise it automatically."
                    )

                    if st.button("Save milestones", key=f"save_milestones_{deal_id}"):
                        ok_ms = save_milestones_for_deal(
                            selected_client_id, deal_id, edited
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

    st.markdown("---")
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

                    # 🔹 Let the user choose which month they are updating
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
                        f"${this_rev:,.0f}",
                    )
                with c2:
                    st.metric(
                        "Next 3 months (total recognised)",
                        f"${next_3_rev:,.0f}",
                    )

                table_view = monthly[["Month", value_col]].rename(
                    columns={value_col: "Recognised revenue"}
                )
                st.dataframe(table_view, use_container_width=True)

    st.markdown("---")

    # ---------- Revenue recognition engine output ----------
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

            # 2) Breakdown by revenue_type (pivot table)
            pivot_by_type = (
                rev_df.pivot_table(
                    index=["month_date", "Month"],
                    columns="revenue_type",
                    values=total_col,
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reset_index()
            )

            display_cols = ["Month"]
            type_cols = [
                c for c in pivot_by_type.columns
                if c not in ["month_date", "Month"]
            ]
            display_cols.extend(type_cols)

            revenue_table = pivot_by_type[display_cols].copy()

            st.dataframe(revenue_table, use_container_width=True)
            st.caption(
                "Recognised revenue by month, split by revenue recognition method. "
                "Amounts are probability-weighted based on pipeline probability."
            )

            # 3) Chart – total revenue over time
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
            st.altair_chart(chart.interactive(), use_container_width=True)

            # 4) Optional stacked chart by type (nice visual)
            long_df = rev_df.copy()
            long_df["revenue_type"] = long_df["revenue_type"].astype(str)

            stacked = (
                alt.Chart(long_df)
                .mark_area()
                .encode(
                    x=alt.X(
                        "month_date:T",
                        title="Month",
                        axis=alt.Axis(format="%b %Y"),
                    ),
                    y=alt.Y(
                        total_col + ":Q",
                        stack="zero",
                        title="Revenue by type",
                    ),
                    color=alt.Color("revenue_type:N", title="Recognition method"),
                    tooltip=[
                        alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                        "revenue_type:N",
                        alt.Tooltip(
                            total_col + ":Q",
                            title="Revenue",
                            format=",.0f",
                        ),
                    ],
                )
            )
            st.altair_chart(stacked.interactive(), use_container_width=True)

    st.markdown("---")

        # ---------- Add a new deal ----------
    st.subheader("➕ Add a new deal")
    with st.form(key="new_deal_form"):
        col1, col2 = st.columns(2)
        with col1:
            deal_name = st.text_input("Deal name")
            customer_name = st.text_input("Customer (optional)")
            value_total = st.number_input(
                "Total value (AUD)",
                min_value=0.0,
                step=1000.0,
            )
        with col2:
            probability_pct = st.slider(
                "Win chance (%)",
                min_value=0,
                max_value=100,
                value=50,
            )
            stage = st.selectbox(
                "Stage",
                ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"],
            )

        st.markdown(
            f"**Revenue type for this business:** {current_method_display}  \n"
            "To change this, update the revenue recognition method at the top of this page."
        )
        method = current_method_display

        # Start month (we don’t ask for end month here)
        start_month_input = st.date_input(
            "Start / expected close month",
            value=selected_month_start,
        )

        # 🔹 Contract length in months (used when there is no end_date)
        contract_months = None
        if selected_method_code == "saas":
            contract_months = st.number_input(
                "Contract length (months)",
                min_value=1,
                max_value=60,
                value=int(get_client_settings(selected_client_id).get("saas_default_months", 12) or 12),
                step=1,
                help="How many months to spread the subscription over.",
            )
        elif selected_method_code in ["straight_line", "poc"]:
            contract_months = st.number_input(
                "Project duration (months)",
                min_value=1,
                max_value=60,
                value=int(get_client_settings(selected_client_id).get("project_default_months", 6) or 6),
                step=1,
                help="How many months to spread the project revenue over if no end date is set.",
            )
        # For milestone / point_in_time / usage we don’t need a contract length here.

        commentary = st.text_area("Notes (optional)")

        submitted = st.form_submit_button("Create deal")

        if submitted:
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
                st.rerun()
            else:
                st.error(
                    "Could not create deal. Please check inputs or try again later."
                )



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




def update_pipeline_deal_status(
    deal_id: int,
    new_stage: str,
    new_probability_pct: float | int,
):
    """
    Update a pipeline deal's stage and probability.
    Also keeps is_won / is_lost flags in sync.
    """
    if not deal_id:
        return False

    stage = (new_stage or "").lower()

    # Auto-set probability for won/lost
    if stage == "closed_won":
        prob = 100.0
    elif stage == "closed_lost":
        prob = 0.0
    else:
        prob = float(new_probability_pct or 0.0)

    data = {
        "stage": stage,
        "probability_pct": prob,
        "is_won": stage == "closed_won",
        "is_lost": stage == "closed_lost",
    }

    try:
        supabase.table("revenue_pipeline").update(data).eq("id", deal_id).execute()
        # 🔄 Invalidate cached data so pipeline + charts refresh immediately
        st.cache_data.clear()
        return True
    except Exception as e:
        print("Error updating pipeline deal:", e)
        return False

# ---------------------------------------------------------
# Investing & Financing flows – simple fetch helpers
# ---------------------------------------------------------

def fetch_investing_flows_for_client(client_id: str) -> pd.DataFrame:
    """
    Read investing_flows for this client.
    Expected table columns:
      - id (uuid)
      - client_id (uuid/text)
      - flow_date (date)
      - amount (numeric)   # +ve = cash in, -ve = cash out
      - description (text, optional)
    """
    if not client_id:
        return pd.DataFrame()

    try:
        resp = (
            supabase.table("investing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )

        data = getattr(resp, "data", None) if hasattr(resp, "data") else resp.get("data")
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print("Error fetching investing_flows:", e)
        return pd.DataFrame()


def fetch_financing_flows_for_client(client_id: str) -> pd.DataFrame:
    """
    Read financing_flows for this client.
    Expected table columns:
      - id (uuid)
      - client_id (uuid/text)
      - flow_date (date)
      - amount (numeric)   # +ve = cash in (funding), -ve = cash out (repayments)
      - description (text, optional)
    """
    if not client_id:
        return pd.DataFrame()

    try:
        resp = (
            supabase.table("financing_flows")
            .select("*")
            .eq("client_id", str(client_id))
            .execute()
        )

        data = getattr(resp, "data", None) if hasattr(resp, "data") else resp.get("data")
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print("Error fetching financing_flows:", e)
        return pd.DataFrame()


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

def compute_runway_and_effective_burn(
    client_id: str,
    focus_month: date,
) -> tuple[float | None, float | None]:
    """
    Uses cashflow_summary to compute:

      - runway_months: based on last 3 months' average negative free_cash_flow
      - effective_burn: this month's free_cash_flow expressed as a positive burn
                        when cash is going down (else 0).

    Returns (runway_months, effective_burn).
    """
    try:
        engine_df = fetch_cashflow_summary_for_client(client_id)
        if engine_df is None or engine_df.empty:
            return None, None

        df = engine_df.copy()
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.date
        df = df.sort_values("month_date")

        focus = pd.to_datetime(focus_month).date().replace(day=1)

        this_row = df[df["month_date"] == focus]
        if this_row.empty:
            return None, None

        closing_cash = float(this_row.get("closing_cash", 0.0).iloc[0] or 0.0)
        free_cf_this = float(this_row.get("free_cash_flow", 0.0).iloc[0] or 0.0)

        # Effective burn: positive number when free CF is negative
        effective_burn = -free_cf_this if free_cf_this < 0 else 0.0

        # Use last 3 months BEFORE the focus month to compute avg burn
        hist = df[df["month_date"] < focus].tail(3)
        if hist.empty or "free_cash_flow" not in hist.columns:
            return None, effective_burn

        # Only consider negative free cashflow as "burn"
        hist_burn = hist["free_cash_flow"].where(hist["free_cash_flow"] < 0)
        if hist_burn.isna().all():
            return None, effective_burn

        avg_burn = -hist_burn.mean()
        if avg_burn <= 0:
            return None, effective_burn

        runway_months = closing_cash / avg_burn
        return runway_months, effective_burn

    except Exception as e:
        print("Error computing runway/effective burn:", e)
        return None, None



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


def fetch_comments(client_id, page_name: str, month_start: date):
    """
    Fetch recent comments for this client + page + month.
    """
    if client_id is None or month_start is None:
        return []

    try:
        res = (
            supabase
            .table("comments")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("page_name", page_name)
            .eq("month_date", month_start.isoformat())
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


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


def get_client_currency(client_id: str | None) -> tuple[str, str]:
    """
    Returns (currency_code, symbol_prefix) for the client.
    E.g. ("AUD", "A$"), ("USD", "$"), etc.
    """
    settings = get_client_settings(client_id)
    code = settings.get("currency_code", "AUD")

    symbol_map = {
        "AUD": "A$",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "NZD": "NZ$",
        "CAD": "C$",
    }
    symbol = symbol_map.get(code, code + " ")

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



@st.cache_data(ttl=60)
def get_client_settings(client_id: str | None):
    """
    Pure data helper:
    Fetch AR/AP default days, alert thresholds, currency, and revenue-recognition
    for a client. Falls back to sensible defaults if no row exists yet.
    """
    defaults = {
        "ar_default_days": 30,
        "ap_default_days": 30,
        "runway_min_months": 4.0,
        "overspend_warn_pct": 10.0,
        "overspend_high_pct": 20.0,
        "revenue_recognition_method": "saas",
        "currency_code": "AUD",
    }

    if client_id is None:
        return defaults

    try:
        res = (
            supabase
            .table("client_settings")
            .select("*")
            .eq("client_id", str(client_id))
            .single()
            .execute()
        )
        data = res.data or {}
    except Exception:
        data = {}

    return {
        "ar_default_days": data.get("ar_default_days", defaults["ar_default_days"]),
        "ap_default_days": data.get("ap_default_days", defaults["ap_default_days"]),
        "runway_min_months": float(
            data.get("runway_min_months", defaults["runway_min_months"])
        ),
        "overspend_warn_pct": float(
            data.get("overspend_warn_pct", defaults["overspend_warn_pct"])
        ),
        "overspend_high_pct": float(
            data.get("overspend_high_pct", defaults["overspend_high_pct"])
        ),
        "revenue_recognition_method": data.get(
            "revenue_recognition_method",
            defaults["revenue_recognition_method"],
        ),
        "currency_code": data.get("currency_code", defaults["currency_code"]),
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


def save_client_settings(client_id, ar_days: int, ap_days: int):
    """
    Upsert AR/AP default days for this client.
    """
    if client_id is None:
        return False

    payload = {
        "client_id": str(client_id),
        "ar_default_days": int(ar_days),
        "ap_default_days": int(ap_days),
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        # upsert via on_conflict on primary key (client_id)
        supabase.table("client_settings").upsert(payload, on_conflict="client_id").execute()
        # Invalidate cached settings so the UI sees latest values
        st.cache_data.clear()
        return True
    except Exception:
        return False


def render_currency_settings_section(selected_client_id):
    settings = get_client_settings(selected_client_id)
    current_code = settings.get("currency_code", DEFAULT_CURRENCY_CODE)

    currency_options = ["AUD", "USD", "EUR", "GBP", "NZD", "CAD"]

    st.markdown("### 💱 Base currency")
    new_code = st.selectbox(
        "Base currency for this client",
        options=currency_options,
        index=currency_options.index(current_code) if current_code in currency_options else 0,
        key="currency_select_box",
    )

    if st.button("Save currency settings", key="save_currency_btn"):
        supabase.table("client_settings").upsert(
            {
                "client_id": str(selected_client_id),
                "currency_code": new_code,
            }
        ).execute()
        # clear cached settings so changes show immediately
        get_client_settings.clear()
        st.success("Currency updated for this client.")


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
                "resolved_at": datetime.utcnow().isoformat(),
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
                "resolved_at": datetime.utcnow().isoformat(),
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
                    "resolved_at": datetime.utcnow().isoformat(),
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
                "resolved_at": datetime.utcnow().isoformat(),
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
    """
    Comments UI backed by Supabase.
    """
    with st.expander("💬 Comments & notes", expanded=False):
        st.write("Notes for this page and month.")

        # Show existing comments
        existing = fetch_comments(selected_client_id, page_name, selected_month_start)
        if existing:
            # Show only the latest 10 comments
            existing = existing[:10]
            for c in existing:
                author = c.get("author_name") or "Unknown"
                body = c.get("body", "")
                created = c.get("created_at", "")[:16]
                st.markdown(f"**{author}** • _{created}_")
                st.write(body)
                st.markdown("---")
            st.caption("Showing latest 10 comments for this page and month.")
        else:
            st.caption("No comments yet for this page and month.")

        new_comment = st.text_area("Add a new comment", key=f"{page_name}_new_comment")
        if st.button("Save comment", key=f"{page_name}_save_comment"):
            ok = save_comment(selected_client_id, page_name, new_comment, selected_month_start)
            if ok:
                st.success("Comment saved.")
                st.rerun()
            else:
                st.error("Could not save comment. Please try again.")

def tasks_block(page_name: str):
    """
    Tasks UI backed by Supabase.
    """
    with st.expander("✅ Tasks & follow-ups", expanded=False):
        st.write("Action items for this page and month.")

        # Show existing tasks (all statuses)
        tasks = fetch_tasks_for_page(selected_client_id, page_name, selected_month_start)

        open_labels = []
        open_id_map = {}

        if tasks:
            st.markdown("**Tasks:**")
            for t in tasks:
                task_id = t.get("id")
                title = t.get("title", "")
                status = t.get("status", "open")
                created = t.get("created_at", "")[:16]

                if status == "done":
                    # Strike-through for done tasks
                    st.markdown(f"- ~~{title}~~  _(done, created {created})_")
                else:
                    st.markdown(f"- {title}  _(created {created})_")
                    # Only non-done tasks go into the "mark as done" list
                    label = f"{title}  (created {created})"
                    open_labels.append(label)
                    open_id_map[label] = task_id
        else:
            st.caption("No tasks yet for this page and month.")

        st.markdown("---")

        # Only allow marking OPEN tasks as done
        if open_labels:
            selected_label = st.selectbox(
                "Pick a task to mark as done",
                ["-- Select --"] + open_labels,
                key=f"{page_name}_task_to_close",
            )
            if selected_label != "-- Select --":
                if st.button("Mark selected task as done", key=f"{page_name}_close_task"):
                    task_id = open_id_map[selected_label]
                    ok = update_task_status(task_id, "done")
                    if ok:
                        st.success("Task marked as done.")
                        st.rerun()
                    else:
                        st.error("Could not update task. Please try again.")
        else:
            st.caption("No open tasks to close.")

        # New task input
        new_task = st.text_input("Add a new task", key=f"{page_name}_new_task")
        if st.button("Create task", key=f"{page_name}_create_task"):
            ok = save_task(selected_client_id, page_name, new_task, selected_month_start)
            if ok:
                st.success("Task created.")
                st.rerun()
            else:
                st.error("Could not create task. Please try again.")



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
        runway_months, effective_burn = compute_runway_and_effective_burn(
            client_id, focus
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



# ---------- Page: Business at a Glance ----------

def page_business_overview():
    top_header("Business at a Glance")


    # --- Currency for this client ---
    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # Pull KPIs from Supabase
    kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up runway alerts for this month (now reads client settings)
    ensure_runway_alert_for_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up cash danger alert based on cashflow_summary
    ensure_cash_danger_alert_for_month(selected_client_id, selected_month_start)


    # ---------- Runway & effective burn from engine ----------
    runway_from_engine, effective_burn = compute_runway_and_effective_burn(
        selected_client_id,
        selected_month_start,
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

    # Optional: second row for Effective Burn
    if effective_burn is not None:
        burn_col, _ = st.columns([1, 3])
        with burn_col:
            st.metric(
                "🔥 Effective burn (per month)",
                f"${effective_burn:,.0f}",
                help="Based on this month's free cashflow (negative cash movement).",
            )

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
        ok = recompute_cashflow_from_ar_ap(
            selected_client_id,
            base_month=selected_month_start,
            n_months=12,
            opening_cash_hint=None,
        )
        if ok:
            st.success("Cashflow updated from AR/AP, payroll, opex and investing/financing moves.")
            st.rerun()
        else:
            st.error("Could not rebuild cashflow. Check AR/AP data and try again.")

    # ---------- Cash curve + danger month ----------
    st.subheader("📉 Cash over time & danger month")

    cash_df = fetch_cash_curve_for_client(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if cash_df.empty:
        st.caption("No cash projection data yet for this business.")
    else:
        cash_df = cash_df.copy()
        cash_df["month_date"] = pd.to_datetime(cash_df["month_date"], errors="coerce")
        cash_df = cash_df.sort_values("month_date")

        danger_label, danger_date = compute_danger_month_from_cash(cash_df, "closing_cash")

        base = alt.Chart(cash_df).mark_line().encode(
            x=alt.X(
                "month_date:T",
                title="Month",
                axis=alt.Axis(format="%b %Y"),
            ),
            y=alt.Y("closing_cash:Q", title=f"Projected cash in bank ({currency_code})"),
            tooltip=[
                alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                alt.Tooltip("closing_cash:Q", title="Cash", format=",.0f"),
            ],
        )

        charts = [base]

        if danger_date is not None:
            danger_point = alt.Chart(
                cash_df[cash_df["month_date"] == pd.to_datetime(danger_date)]
            ).mark_point(size=80, color="red").encode(
                x=alt.X("month_date:T", axis=alt.Axis(format="%b %Y")),
                y="closing_cash:Q",
                tooltip=[
                    alt.Tooltip("month_date:T", title="Danger month", format="%b %Y"),
                    alt.Tooltip("closing_cash:Q", title="Cash", format=",.0f"),
                ],
            )
            charts.append(danger_point)

        st.altair_chart(alt.layer(*charts).interactive(), width="stretch")


        # ---------- Scenario: quick stress-test vs baseline ----------

        st.markdown("#### 🔮 Quick scenario: tweak Money in / Money out")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            rev_change_pct = st.slider(
                "Money in (revenue) change vs baseline",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Increase or decrease the positive part of operating cashflow.",
                key="scenario_rev_change",
            )
        with col_s2:
            burn_change_pct = st.slider(
                "Money out (burn) change vs baseline",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help="Increase or decrease the negative part of operating cashflow.",
                key="scenario_burn_change",
            )

        # Build baseline subset for the 12-month window
        engine_df_full = fetch_cashflow_summary_for_client(selected_client_id)
        scenario_df = pd.DataFrame()

        if engine_df_full is not None and not engine_df_full.empty:
            df_eng = engine_df_full.copy()
            df_eng["month_date"] = pd.to_datetime(
                df_eng["month_date"], errors="coerce"
            )
            df_eng = df_eng.sort_values("month_date")

            start = pd.to_datetime(selected_month_start).replace(day=1)
            end = start + pd.DateOffset(months=12)
            df_eng = df_eng[
                (df_eng["month_date"] >= start) & (df_eng["month_date"] < end)
            ].copy()

            if not df_eng.empty and "free_cash_flow" in df_eng.columns:
                df_eng["month_date"] = (
                    df_eng["month_date"].dt.to_period("M").dt.to_timestamp()
                )

                # Make sure numeric
                df_eng["closing_cash"] = pd.to_numeric(
                    df_eng.get("closing_cash"), errors="coerce"
                ).fillna(0.0)
                df_eng["free_cash_flow"] = pd.to_numeric(
                    df_eng.get("free_cash_flow"), errors="coerce"
                ).fillna(0.0)
                df_eng["operating_cf"] = pd.to_numeric(
                    df_eng.get("operating_cf"), errors="coerce"
                ).fillna(0.0)
                df_eng["investing_cf"] = pd.to_numeric(
                    df_eng.get("investing_cf"), errors="coerce"
                ).fillna(0.0)
                df_eng["financing_cf"] = pd.to_numeric(
                    df_eng.get("financing_cf"), errors="coerce"
                ).fillna(0.0)

                # Reconstruct baseline opening cash from closing & free CF
                openings = []
                prev_closing = None
                for _, row in df_eng.iterrows():
                    if prev_closing is None:
                        opening = row["closing_cash"] - row["free_cash_flow"]
                    else:
                        opening = prev_closing
                    openings.append(opening)
                    prev_closing = row["closing_cash"]
                df_eng["opening_cash"] = openings

                # Apply sliders:
                #   - rev slider to positive operating CF
                #   - burn slider to negative operating CF
                rev_factor = 1.0 + rev_change_pct / 100.0
                burn_factor = 1.0 + burn_change_pct / 100.0

                rev_part = df_eng["operating_cf"].clip(lower=0.0)
                burn_part = df_eng["operating_cf"].where(
                    df_eng["operating_cf"] < 0.0, 0.0
                )

                df_eng["operating_cf_scn"] = (
                    rev_part * rev_factor + burn_part * burn_factor
                )
                df_eng["free_cf_scn"] = (
                    df_eng["operating_cf_scn"]
                    + df_eng["investing_cf"]
                    + df_eng["financing_cf"]
                )

                # Roll forward scenario closing cash
                scenario_closing = []
                running = None
                for _, row in df_eng.iterrows():
                    if running is None:
                        running = row["opening_cash"] + row["free_cf_scn"]
                    else:
                        running = running + row["free_cf_scn"]
                    scenario_closing.append(running)
                df_eng["closing_cash_scn"] = scenario_closing

                scenario_df = df_eng[
                    ["month_date", "closing_cash", "closing_cash_scn"]
                ].copy()
                scenario_df["Month"] = scenario_df["month_date"].dt.strftime("%b %Y")
                scenario_df["Delta"] = (
                    scenario_df["closing_cash_scn"] - scenario_df["closing_cash"]
                )

                # Scenario chart overlay
                scn_long = scenario_df.melt(
                    id_vars="month_date",
                    value_vars=["closing_cash", "closing_cash_scn"],
                    var_name="Series",
                    value_name="Cash",
                )
                label_map = {
                    "closing_cash": "Baseline closing cash",
                    "closing_cash_scn": "Scenario closing cash",
                }
                scn_long["Series"] = scn_long["Series"].map(label_map)

                scn_chart = (
                    alt.Chart(scn_long)
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "month_date:T",
                            title="Month",
                            axis=alt.Axis(format="%b %Y"),
                        ),
                        y=alt.Y("Cash:Q", title="Cash in bank"),
                        color=alt.Color("Series:N", title=""),
                        tooltip=[
                            alt.Tooltip(
                                "month_date:T", title="Month", format="%b %Y"
                            ),
                            "Series:N",
                            alt.Tooltip("Cash:Q", title="Cash", format=",.0f"),
                        ],
                    )
                )

                st.altair_chart(scn_chart.interactive(), width="stretch")

                # Baseline vs Scenario comparison table
                st.markdown("##### Baseline vs Scenario (closing cash)")

                table_view = scenario_df[
                    ["Month", "closing_cash", "closing_cash_scn", "Delta"]
                ].rename(
                    columns={
                        "closing_cash": "Baseline closing",
                        "closing_cash_scn": "Scenario closing",
                        "Delta": "Δ Scenario - Baseline",
                    }
                )
                st.dataframe(table_view, width="stretch")
            else:
                st.caption("Not enough engine data to build a scenario yet.")
        else:
            st.caption("No cashflow engine data yet for scenarios.")


        # ------------------------------------------------------------------
        # Scenario slider: tweak operating cash and compare baseline vs scenario
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🧪 Scenario: What if operating cash changes?")

        # % change to operating cash (affects operating_cf only)
        scenario_pct = st.slider(
            "Change in operating cash each month (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            help=(
                "Positive = better operating cash (more cash in / less cash out). "
                "Negative = worse operating cash (less cash in / more cash out)."
            ),
            key="scenario_op_cf_pct",
        )

        # Load baseline engine rows for the same 12-month window
        engine_df = fetch_cashflow_summary_for_client(selected_client_id)
        if engine_df is None or engine_df.empty:
            st.caption("No baseline cashflow engine rows yet – rebuild cashflow first.")
        else:
            eng = engine_df.copy()
            eng["month_date"] = pd.to_datetime(eng["month_date"], errors="coerce")
            eng = eng.sort_values("month_date")

            start = pd.to_datetime(selected_month_start).replace(day=1)
            end = start + pd.DateOffset(months=12)

            eng = eng[(eng["month_date"] >= start) & (eng["month_date"] < end)].copy()

            if eng.empty:
                st.caption("No engine rows in this 12-month window yet.")
            else:
                # Ensure we have the needed columns
                for col in ["operating_cf", "investing_cf", "financing_cf", "free_cash_flow", "closing_cash"]:
                    if col not in eng.columns:
                        st.caption(f"Cashflow engine is missing '{col}' – cannot build scenario view yet.")
                        eng = None
                        break

            if eng is not None and not eng.empty:
                eng = eng.reset_index(drop=True)

                # Derive opening cash for first month from baseline numbers
                # closing_0 = opening_0 + free_cf_0  =>  opening_0 = closing_0 - free_cf_0
                closing_0 = float(eng.loc[0, "closing_cash"] or 0.0)
                free_cf_0 = float(eng.loc[0, "free_cash_flow"] or 0.0)
                opening_0 = closing_0 - free_cf_0

                pct_factor = 1.0 + (scenario_pct / 100.0)

                scenario_rows = []
                current_open = opening_0
                baseline_closing_series = []
                scenario_closing_series = []

                for _, r in eng.iterrows():
                    mdate = r["month_date"]
                    op_cf_base = float(r.get("operating_cf", 0.0) or 0.0)
                    inv_cf = float(r.get("investing_cf", 0.0) or 0.0)
                    fin_cf = float(r.get("financing_cf", 0.0) or 0.0)

                    # Baseline free CF & closing (recomputed, just to be explicit)
                    free_cf_base = op_cf_base + inv_cf + fin_cf
                    closing_base = current_open + free_cf_base

                    # Scenario: only operating_cf changes
                    op_cf_scen = op_cf_base * pct_factor
                    free_cf_scen = op_cf_scen + inv_cf + fin_cf
                    closing_scen = current_open + free_cf_scen

                    baseline_closing_series.append(
                        {"month_date": mdate, "closing_cash_baseline": closing_base}
                    )
                    scenario_closing_series.append(
                        {"month_date": mdate, "closing_cash_scenario": closing_scen}
                    )

                    scenario_rows.append(
                        {
                            "month_date": mdate,
                            "operating_cf_baseline": op_cf_base,
                            "operating_cf_scenario": op_cf_scen,
                            "free_cf_baseline": free_cf_base,
                            "free_cf_scenario": free_cf_scen,
                            "closing_cash_baseline": closing_base,
                            "closing_cash_scenario": closing_scen,
                        }
                    )

                    # Next month opens with baseline closing (we keep baseline path as "truth")
                    current_open = closing_base

                scenario_df = pd.DataFrame(scenario_rows)
                baseline_df = pd.DataFrame(baseline_closing_series)
                scen_df = pd.DataFrame(scenario_closing_series)

                # Build a combined df for chart
                chart_df = (
                    baseline_df.merge(scen_df, on="month_date", how="left")
                    .sort_values("month_date")
                )
                chart_df["Month"] = chart_df["month_date"].dt.strftime("%b %Y")

                long_chart = chart_df.melt(
                    id_vars=["month_date", "Month"],
                    value_vars=["closing_cash_baseline", "closing_cash_scenario"],
                    var_name="Series",
                    value_name="ClosingCash",
                )
                series_name_map = {
                    "closing_cash_baseline": "Baseline cash",
                    "closing_cash_scenario": "Scenario cash",
                }
                long_chart["Series"] = long_chart["Series"].map(series_name_map)

                scen_chart = (
                    alt.Chart(long_chart)
                    .mark_line()
                    .encode(
                        x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                        y=alt.Y("ClosingCash:Q", title="Closing cash"),
                        color=alt.Color("Series:N", title=""),
                        tooltip=[
                            alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                            "Series:N",
                            alt.Tooltip("ClosingCash:Q", title="Closing cash", format=",.0f"),
                        ],
                    )
                )

                st.altair_chart(scen_chart.interactive(), width="stretch")

                # Baseline vs scenario table
                table_view = chart_df[["Month", "closing_cash_baseline", "closing_cash_scenario"]].copy()
                table_view["Δ vs baseline"] = (
                    table_view["closing_cash_scenario"] - table_view["closing_cash_baseline"]
                )
                table_view = table_view.rename(
                    columns={
                        "closing_cash_baseline": "Baseline closing cash",
                        "closing_cash_scenario": "Scenario closing cash",
                    }
                )

                st.dataframe(table_view, width="stretch")

                # Compute scenario "danger month" (first month scenario cash < 0)
                danger_month_scen = None
                for _, r in chart_df.sort_values("month_date").iterrows():
                    if r["closing_cash_scenario"] < 0:
                        danger_month_scen = r["month_date"]
                        break

                if danger_month_scen is not None:
                    st.caption(
                        f"Scenario danger month (cash < 0) ≈ "
                        f"{pd.to_datetime(danger_month_scen).strftime('%b %Y')} "
                        f"at {scenario_pct:+d}% change to operating cash."
                    )
                else:
                    st.caption(
                        f"In this scenario ({scenario_pct:+d}% to operating cash), "
                        "cash does not go negative within the 12-month window."
                    )

    # ------------------------------------------------------------------
    # NEW: Investing & financing cash moves (UI lives here)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💹 Investing & Financing cash moves")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(
            "Use this to record *big one-off moves* like buying equipment, "
            "raising equity, taking/repaying loans or paying dividends. "
            "Positive = cash **in**, negative = cash **out**."
        )

    with col_btn:
        if st.button("Rebuild cashflow (after changes)", key="rebuild_after_inv_fin"):
            ok = recompute_cashflow_from_ar_ap(
                selected_client_id,
                base_month=selected_month_start,
                n_months=12,
                opening_cash_hint=None,
            )
            if ok:
                st.success("Cashflow rebuilt with latest investing/financing moves.")
                st.rerun()
            else:
                st.error("Could not rebuild cashflow. Check data and try again.")

    df_inv = fetch_investing_flows_for_client(selected_client_id)
    df_fin = fetch_financing_flows_for_client(selected_client_id)

    tab_inv, tab_fin = st.tabs(["🏗 Investing flows", "🏦 Financing flows"])

    # --- INVESTING TAB ---
    with tab_inv:
        st.markdown("#### Investing flows (capex, asset sales, etc.)")

        if df_inv is None or df_inv.empty:
            st.caption("No investing flows yet for this business.")
        else:
            df_show = df_inv.copy()
            df_show["month_date"] = pd.to_datetime(df_show["month_date"], errors="coerce")
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

    # --- FINANCING TAB ---
    with tab_fin:
        st.markdown("#### Financing flows (equity, loans, repayments, dividends)")

        if df_fin is None or df_fin.empty:
            st.caption("No financing flows yet for this business.")
        else:
            df_show = df_fin.copy()
            df_show["month_date"] = pd.to_datetime(df_show["month_date"], errors="coerce")
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

    if engine_df_all.empty:
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

        if not df_ap.empty:
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

    # ---------- Cashflow engine breakdown table ----------
    st.markdown("---")
    st.subheader("🔍 Cashflow engine – 12-month breakdown")

    engine_df = fetch_cashflow_summary_for_client(selected_client_id)
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
                "that feeds the chart and the cash danger alerts."
            )

    comments_block("business_overview")
    tasks_block("business_overview")


def page_scenarios():
    top_header("Scenarios & What-if")

    st.markdown(
        "Tweak a few simple levers and see how they change your cash curve, "
        "burn and danger month – without touching your actual data."
    )

    # ---------- Load baseline cashflow engine ----------
    engine_df_all = fetch_cashflow_summary_for_client(selected_client_id)

    if engine_df_all is None or engine_df_all.empty:
        st.warning(
            "No cashflow engine data yet for this business. "
            "Go to **Business at a Glance → Rebuild cashflow for next 12 months** first."
        )
        return

    engine_df_all = engine_df_all.copy()
    engine_df_all["month_date"] = pd.to_datetime(
        engine_df_all["month_date"], errors="coerce"
    )
    engine_df_all = engine_df_all.dropna(subset=["month_date"])

    focus_start = pd.to_datetime(selected_month_start).replace(day=1)
    focus_end = focus_start + pd.DateOffset(months=12)

    base_df = engine_df_all[
        (engine_df_all["month_date"] >= focus_start)
        & (engine_df_all["month_date"] < focus_end)
    ].copy()

    if base_df.empty:
        st.caption(
            "No cashflow rows for this 12-month window yet. "
            "Try rebuilding cashflow or changing the focus month."
        )
        return

    base_df = base_df.sort_values("month_date").reset_index(drop=True)

    # Ensure required columns exist
    for col in ["operating_cf", "investing_cf", "financing_cf", "free_cash_flow", "closing_cash"]:
        if col not in base_df.columns:
            base_df[col] = 0.0

    # ---------- Scenario controls ----------
    st.markdown("---")
    st.subheader("🎛 Scenario controls")

    month_opts = base_df["month_date"].dt.to_period("M").drop_duplicates()
    month_label_map = {
        p.strftime("%b %Y"): p.to_timestamp() for p in month_opts
    }
    month_labels = list(month_label_map.keys())

    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        scenario_start_label = st.selectbox(
            "Scenario starts from month",
            options=month_labels,
        )
        op_change_pct = st.slider(
            "Change **net operating cashflow** from that month (%)",
            min_value=-50,
            max_value=50,
            value=0,
            help="Approximation of revenue/spend change. +10% means ops cashflow improves by 10%.",
        )
        extra_capex = st.number_input(
            "Extra **investing cashflow** per month from that month (- = more capex)",
            value=0.0,
            step=1000.0,
            help="Use negative numbers for extra capex (cash out), positive for asset sales (cash in).",
        )

    with col_ctrl2:
        one_off_financing = st.number_input(
            "One-off **financing inflow** in the start month",
            value=0.0,
            step=1000.0,
            help="E.g. equity injection or new loan in the start month.",
        )
        recurring_financing = st.number_input(
            "Recurring **financing inflow** each month from start month",
            value=0.0,
            step=1000.0,
            help="E.g. monthly loan drawdowns or recurring investor top-ups.",
        )

    start_month_ts = month_label_map[scenario_start_label]

    # ---------- Build scenario dataframe ----------
    scen_df = base_df.copy()

    mask = scen_df["month_date"] >= start_month_ts

    # Operating CF scenario
    scen_df["operating_cf_scenario"] = scen_df["operating_cf"]
    scen_df.loc[mask, "operating_cf_scenario"] = (
        scen_df.loc[mask, "operating_cf"] * (1.0 + op_change_pct / 100.0)
    )

    # Investing CF scenario
    scen_df["investing_cf_scenario"] = scen_df["investing_cf"]
    scen_df.loc[mask, "investing_cf_scenario"] = (
        scen_df.loc[mask, "investing_cf_scenario"] + float(extra_capex)
    )

    # Financing CF scenario
    scen_df["financing_cf_scenario"] = scen_df["financing_cf"]

    # One-off at start month
    scen_df.loc[
        scen_df["month_date"] == start_month_ts, "financing_cf_scenario"
    ] = (
        scen_df.loc[scen_df["month_date"] == start_month_ts, "financing_cf_scenario"]
        + float(one_off_financing)
    )

    # Recurring from start month
    scen_df.loc[mask, "financing_cf_scenario"] = (
        scen_df.loc[mask, "financing_cf_scenario"] + float(recurring_financing)
    )

    # Recompute free CF + closing cash for scenario
    scen_df = scen_df.sort_values("month_date").reset_index(drop=True)

    scen_df["free_cash_flow_scenario"] = (
        scen_df["operating_cf_scenario"]
        + scen_df["investing_cf_scenario"]
        + scen_df["financing_cf_scenario"]
    )

    # Derive an opening cash based on baseline row 0
    first_row = scen_df.iloc[0]
    base_opening_0 = float(first_row["closing_cash"]) - float(first_row["free_cash_flow"])

    closing_vals = []
    opening = base_opening_0
    for _, r in scen_df.iterrows():
        free_cf_s = float(r["free_cash_flow_scenario"])
        closing = opening + free_cf_s
        closing_vals.append(closing)
        opening = closing

    scen_df["closing_cash_scenario"] = closing_vals

    # ---------- Danger month comparison ----------
    baseline_label, baseline_date = compute_danger_month_from_cash(
        base_df.copy(), "closing_cash"
    )
    scen_tmp = scen_df.rename(
        columns={"closing_cash_scenario": "closing_cash"}
    )
    scen_label, scen_date = compute_danger_month_from_cash(
        scen_tmp, "closing_cash"
    )

    # ---------- Metrics ----------
    st.markdown("---")
    st.subheader("📌 Scenario vs baseline – danger month")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            "Baseline danger month",
            baseline_label or "None in 12 months",
        )
    with col_m2:
        # Simple text delta (months difference) if both dates exist
        delta_txt = None
        if baseline_date is not None and scen_date is not None:
            # positive = pushed out (good), negative = brought forward (bad)
            month_diff = (scen_date.to_period("M") - baseline_date.to_period("M")).n
            if month_diff != 0:
                sign_word = "later" if month_diff > 0 else "earlier"
                delta_txt = f"{abs(month_diff)} month(s) {sign_word}"
        st.metric(
            "Scenario danger month",
            scen_label or "None in 12 months",
            delta=delta_txt,
        )

    # ---------- Chart: baseline vs scenario cash curve ----------
    st.markdown("---")
    st.subheader("📉 Cash curve – baseline vs scenario")

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
            y=alt.Y("Cash:Q", title="Closing cash"),
            color=alt.Color("Series:N", title=""),
            tooltip=[
                alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                "Series:N",
                alt.Tooltip("Cash:Q", title="Cash", format=",.0f"),
            ],
        )
    )

    st.altair_chart(cash_chart.interactive(), use_container_width=True)

    # ---------- Summary table ----------
    st.markdown("---")
    st.subheader("🔍 Scenario breakdown table")

    view = scen_df[["month_date",
                    "operating_cf",
                    "operating_cf_scenario",
                    "investing_cf",
                    "investing_cf_scenario",
                    "financing_cf",
                    "financing_cf_scenario",
                    "closing_cash",
                    "closing_cash_scenario",
                    ]].copy()

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

    st.dataframe(view, use_container_width=True)
    st.caption(
        "Baseline vs scenario cashflow by month. Scenario is purely simulated on top of the engine – "
        "it does not change your stored AR/AP, payroll or opex data."
    )



# ---------- Page: Team & Spending ----------

def page_team_spending():
    top_header("Team & Spending")

    # Dept overspend alerts
    ensure_dept_overspend_alert(selected_client_id, selected_month_start)
# Load data with a spinner
    with st.spinner("Loading team and department data..."):
        df_hiring = fetch_hiring_monthly_for_client(selected_client_id)
        df_dept = fetch_dept_monthly_for_client(selected_client_id)

        # ---------- Payroll summary for selected month ----------
    st.subheader("👥 Payroll snapshot for this month")

    # Build a one-month index for the selected month
    month_index = pd.date_range(
        start=pd.to_datetime(selected_month_start),
        periods=1,
        freq="MS",
    )
    payroll_by_month = compute_payroll_by_month(selected_client_id, month_index)
    month_ts = month_index[0]
    month_payroll = float(payroll_by_month.get(month_ts, 0.0))

    # Approximate "other operating cash out" from AP (bills) for the same month
    df_ar_ap = fetch_ar_ap_for_client(selected_client_id)
    if isinstance(df_ar_ap, tuple):
        df_ar, df_ap = df_ar_ap
    else:
        df_ar, df_ap = None, None

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

            month_ap = float(
                df_ap.loc[df_ap["bucket_month"] == month_ts, "amount"].sum()
            )

    total_cash_out = month_payroll + month_ap
    payroll_share = (month_payroll / total_cash_out * 100.0) if total_cash_out else 0.0

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric(
            "Payroll cash this month",
            f"${month_payroll:,.0f}",
            help="Total payroll cash out for this month from the payroll engine.",
        )
    with col_p2:
        st.metric(
            "Bills (AP) this month",
            f"${month_ap:,.0f}",
            help="Supplier bills expected to be paid this month.",
        )
    with col_p3:
        st.metric(
            "Payroll as % of total cash out",
            f"{payroll_share:.0f}%",
            help="Payroll divided by (payroll + supplier bills) for this month.",
        )

    st.markdown("---")

    # --------------------------------------------------
    # 1) Team & payroll – roles that drive the engine
    # --------------------------------------------------
    st.subheader("👥 Team & payroll (drives cashflow engine)")

    df_positions = fetch_payroll_positions_for_client(selected_client_id)

    if df_positions.empty:
        st.info("No roles added yet. Add your team below to drive payroll in the cashflow engine.")
    else:
        view = df_positions.copy()
        rename_map = {
            "role_name": "Role",
            "employee_name": "Employee",
            "fte": "FTE",
            "base_salary_annual": "Base salary (annual)",
            "super_rate_pct": "Super (%)",
            "payroll_tax_pct": "Payroll tax (%)",
            "start_date": "Start date",
            "end_date": "End date",
            "notes": "Notes",
        }
        cols = [c for c in rename_map.keys() if c in view.columns]
        view = view[cols].rename(columns=rename_map)

        st.dataframe(view, use_container_width=True)
        st.caption(
            "These roles feed your payroll cash-out in the core cashflow engine "
            "and the 'Payroll vs bills vs Net operating cash' chart."
        )

    # Small payroll forecast (next 6 months) using the same engine function
    st.markdown("")
    st.caption("Payroll cash-out (next 6 months)")

    month_index = pd.date_range(
        start=pd.to_datetime(selected_month_start),
        periods=6,
        freq="MS",
    )
    payroll_by_month = compute_payroll_by_month(selected_client_id, month_index)

    if payroll_by_month:
        pay_df = pd.DataFrame(
            {
                "month_date": month_index,
                "Payroll_cash": [float(payroll_by_month.get(m, 0.0)) for m in month_index],
            }
        )
        pay_df["Month"] = pay_df["month_date"].dt.strftime("%b %Y")
        st.line_chart(pay_df.set_index("Month")[["Payroll_cash"]])
    else:
        st.caption("No payroll yet – add roles below to see the forecast.")

    st.markdown("---")

    # --------------------------------------------------
    # 2) Add / edit a role
    # --------------------------------------------------
    st.subheader("➕ Add or update a role")

    # Build options for edit mode
    existing_options = []
    id_map = {}
    if not df_positions.empty:
        for _, r in df_positions.iterrows():
            label = f"{r.get('role_name', 'Role')} – {r.get('employee_name', 'Unassigned')}"
            existing_options.append(label)
            id_map[label] = r.get("id")

    edit_mode = st.radio(
        "What do you want to do?",
        options=["Add new role", "Edit existing role"],
        horizontal=True,
    )

    position_id = None
    existing_row = None
    if edit_mode == "Edit existing role" and existing_options:
        chosen = st.selectbox("Pick a role to edit", ["-- Select --"] + existing_options)
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
        c1, c2 = st.columns(2)
        with c1:
            role_name = st.text_input("Role title", value=_get("role_name", "Finance Manager"))
            employee_name = st.text_input("Employee name (optional)", value=_get("employee_name", ""))
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
        with c2:
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
                position_id=position_id if (edit_mode == "Edit existing role" and position_id) else None,
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

    # --------------------------------------------------
    # 3) Delete a role
    # --------------------------------------------------
    if not df_positions.empty:
        st.markdown("---")
        st.subheader("🗑️ Delete a role")

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

    st.markdown("---")

    # --------------------------------------------------
    # 4) Department spend vs plan (your existing logic)
    # --------------------------------------------------
    st.subheader("🏢 Department spend vs plan")

    with st.spinner("Loading department data..."):
        df_dept = fetch_dept_monthly_for_client(selected_client_id)

    if df_dept.empty:
        st.info("No department budget/actuals yet for this business.")
    else:
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
            st.error(
                "dept_monthly is missing expected columns (department, budget_*, actual_*). "
                "Please check the table structure."
            )
        else:
            if "month_date" in df_dept.columns and selected_month_start is not None:
                df_dept_month = df_dept[
                    df_dept["month_date"] == pd.to_datetime(selected_month_start)
                ]
                if df_dept_month.empty:
                    latest = df_dept["month_date"].max()
                    df_dept_month = df_dept[df_dept["month_date"] == latest]
            else:
                df_dept_month = df_dept.copy()

            dept_view = (
                df_dept_month
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
            dept_view["Variance"] = dept_view["Actual"] - dept_view["Plan"]
            dept_view["Variance %"] = dept_view.apply(
                lambda row: (row["Variance"] / row["Plan"] * 100) if row["Plan"] else 0,
                axis=1,
            )

            st.dataframe(dept_view, use_container_width=True)

            st.caption("Plan vs actual by department (selected / latest month)")
            chart_df = dept_view.set_index("Department")[["Plan", "Actual"]]
            st.bar_chart(chart_df)

    # --------------------------------------------------
    # 5) Internal CSV uploads (unchanged from your version)
    # --------------------------------------------------
    with st.expander("🧪 Upload test data (internal only)", expanded=False):
        st.caption(
            "Use this to quickly upload test data for this business. "
            "CSV columns must match the Supabase table columns (except client_id)."
        )

        col_u1, col_u2 = st.columns(2)

        with col_u1:
            st.markdown("**Upload hiring_monthly CSV**")
            hiring_csv = st.file_uploader(
                "Select CSV for hiring_monthly",
                type=["csv"],
                key="hiring_monthly_csv",
            )
            if hiring_csv is not None:
                if st.button("Upload to hiring_monthly", key="upload_hiring_monthly"):
                    ok, msg = upload_csv_to_table_for_client(
                        selected_client_id, hiring_csv, "hiring_monthly"
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with col_u2:
            st.markdown("**Upload dept_monthly CSV**")
            dept_csv = st.file_uploader(
                "Select CSV for dept_monthly",
                type=["csv"],
                key="dept_monthly_csv",
            )
            if dept_csv is not None:
                if st.button("Upload to dept_monthly", key="upload_dept_monthly"):
                    ok, msg = upload_csv_to_table_for_client(
                        selected_client_id, dept_csv, "dept_monthly"
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    comments_block("team_spending")
    tasks_block("team_spending")



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

    # -------------------------------
    # Load payment term settings
    # -------------------------------
    settings = get_client_settings(selected_client_id) or {}
    ar_term = int(settings.get("ar_default_days", 30))
    ap_term = int(settings.get("ap_default_days", 30))

    # -------------------------------
    # Settings UI
    # -------------------------------
    st.subheader("⚙️ Payment terms for this business")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        ar_input = st.number_input(
            "Default days to get paid (customers / AR)",
            min_value=1,
            max_value=180,
            value=ar_term,
            step=1,
        )
    with col_s2:
        ap_input = st.number_input(
            "Default days to pay bills (suppliers / AP)",
            min_value=1,
            max_value=180,
            value=ap_term,
            step=1,
        )

    if st.button("Save payment terms"):
        ok = save_client_settings(selected_client_id, ar_input, ap_input)
        if ok:
            st.success("Payment terms updated for this business.")
            st.rerun()
        else:
            st.error("Could not update payment terms. Please try again.")

    # Use the *current* UI values as "active" terms
    ar_days_display = ar_input
    ap_days_display = ap_input

    # Keep AR overdue alerts updated (uses these defaults internally now)
    ensure_overdue_ar_alert(selected_client_id, as_of=date.today(), min_days_overdue=0)

    st.markdown("---")

    # -------------------------------
    # Load AR / AP data with spinner
    # -------------------------------
    with st.spinner("Loading invoices and cash commitments..."):
        df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)

        # Add ageing info to AR using today's date and the current AR term
        if df_ar is not None and not df_ar.empty:
            df_ar = add_ar_aging(df_ar, as_of=date.today(), ar_default_days=ar_days_display)

        # Add ageing info to AP using today's date and the current AP term
        if df_ap is not None and not df_ap.empty:
            df_ap = add_ap_aging(df_ap, as_of=date.today(), ap_default_days=ap_days_display)

    # ------------------------------------------------------------------
    # 📥 AR – Invoices coming in
    # ------------------------------------------------------------------
    st.subheader("📥 Invoices coming in (AR)")
    st.caption(f"Default expectation: customers pay within {ar_days_display} day(s) of invoice.")
    st.write("Who owes you money and when it's expected in.")

    if df_ar is None or df_ar.empty:
        st.info("No AR records yet for this client.")
    else:
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
        st.dataframe(ar_view, use_container_width=True)

    # Optional: AR ageing summary (only for non-paid invoices)
    if df_ar is not None and not df_ar.empty and \
       "aging_bucket" in df_ar.columns and "amount" in df_ar.columns:

        st.markdown("#### AR ageing summary (by overdue bucket)")

        def _not_paid(row):
            s = str(row.get("status", "")).lower()
            return s not in ["paid", "closed", "settled"]

        df_not_paid = df_ar[df_ar.apply(_not_paid, axis=1)].copy()

        if df_not_paid.empty:
            st.caption("All customer invoices appear to be paid or closed.")
        else:
            ageing_summary = (
                df_not_paid
                .groupby("aging_bucket", as_index=False)["amount"]
                .sum()
                .rename(columns={"aging_bucket": "Bucket", "amount": "Total amount"})
            )
            st.dataframe(ageing_summary, use_container_width=True)

    # ------------------------------------------------------------------
    # 📤 AP – Bills to pay
    # ------------------------------------------------------------------
    st.subheader("📤 Bills to pay (AP)")
    st.caption(f"Default expectation: you pay suppliers within {ap_days_display} day(s) of invoice.")
    st.write("Your upcoming bills and payment plans.")

    if df_ap is None or df_ap.empty:
        st.info("No AP records yet for this client.")
    else:
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
        st.dataframe(ap_view, use_container_width=True)

    # Optional: AP ageing summary (only for non-paid bills)
    if df_ap is not None and not df_ap.empty and \
       "aging_bucket" in df_ap.columns and "amount" in df_ap.columns:

        st.markdown("#### AP ageing summary (by overdue bucket)")

        def _not_paid_ap(row):
            s = str(row.get("status", "")).lower()
            return s not in ["paid", "closed", "settled"]

        df_not_paid_ap = df_ap[df_ap.apply(_not_paid_ap, axis=1)].copy()

        if df_not_paid_ap.empty:
            st.caption("All supplier bills appear to be paid or closed.")
        else:
            ap_ageing_summary = (
                df_not_paid_ap
                .groupby("aging_bucket", as_index=False)["amount"]
                .sum()
                .rename(columns={"aging_bucket": "Bucket", "amount": "Total amount"})
            )
            st.dataframe(ap_ageing_summary, use_container_width=True)

    # ------------------------------------------------------------------
    # 🔍 Cash commitments (near-term list)
    # ------------------------------------------------------------------
    st.subheader("🔍 Cash commitments coming up")
    commitments = build_cash_commitments(df_ar, df_ap, limit=7)

    if commitments.empty:
        st.info("No upcoming cash movements found yet.")
    else:
        st.dataframe(commitments, use_container_width=True)

    # ------------------------------------------------------------------
    # 📆 14-week cashflow (operations view)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📆 14-week cashflow (operations view)")

    week_df = build_14_week_cash_table(selected_client_id, selected_month_start)

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

        st.dataframe(view, use_container_width=True)
        st.caption(
            "Shows your expected cash in/out by week for the next ~3 months, "
            "based on AR, AP and payroll assumptions."
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
                    f"${first_danger['closing_cash']:,.0f}."
                )

        # Small line chart of weekly closing cash
        chart = (
            alt.Chart(week_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "week_start:T",
                    title="Week starting",
                    axis=alt.Axis(format="%d %b"),
                ),
                y=alt.Y("closing_cash:Q", title="Projected closing cash"),
                tooltip=[
                    alt.Tooltip("week_start:T", title="Week of", format="%d %b %Y"),
                    alt.Tooltip("closing_cash:Q", title="Closing cash", format=",.0f"),
                    alt.Tooltip("operating_cf:Q", title="Operating CF", format=",.0f"),
                ],
            )
        )

        st.altair_chart(chart.interactive(), use_container_width=True)

    comments_block("cash_bills")
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
                "resolved_at": datetime.utcnow().isoformat(),
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

# ---------- Page: Alerts & To-Dos ----------

def page_alerts_todos():
    top_header("Alerts & To-Dos")

    # ---------- Alert settings for this business ----------
    st.subheader("⚙️ Alert settings for this business")

    settings = get_client_settings(selected_client_id)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        runway_input = st.number_input(
            "Alert when runway drops below (months)",
            min_value=1.0,
            max_value=24.0,
            value=float(settings["runway_min_months"]),
            step=0.5,
        )
    with col_a2:
        overspend_warn_input = st.number_input(
            "Dept overspend – warn at (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(settings["overspend_warn_pct"]),
            step=1.0,
        )
    with col_a3:
        overspend_high_input = st.number_input(
            "Dept overspend – high at (%)",
            min_value=0.0,
            max_value=200.0,
            value=float(settings["overspend_high_pct"]),
            step=1.0,
        )

    if st.button("Save alert settings"):
        ok = save_alert_thresholds(
            selected_client_id,
            runway_min_months=runway_input,
            overspend_warn_pct=overspend_warn_input,
            overspend_high_pct=overspend_high_input,
        )
        if ok:
            st.success("Alert settings updated.")
            st.rerun()
        else:
            st.error("Could not update alert settings. Please try again.")

    st.markdown("---")

    # ---------- Alerts ----------
    st.subheader("🚨 Alerts watching your business")

    view_mode = st.radio(
        "Which alerts do you want to see?",
        ["Only active", "All (active + resolved)"],
        horizontal=True,
    )
    only_active = view_mode == "Only active"

    with st.spinner("Loading alerts..."):
        alerts = fetch_alerts_for_client(selected_client_id, only_active=only_active, limit=50)

    if not alerts:
        if only_active:
            st.info("No active alerts right now. You're all clear based on the current rules.")
        else:
            st.info("No alerts stored yet for this business.")
    else:
        df_alerts = pd.DataFrame(alerts)

        cols = []
        if "alert_type" in df_alerts.columns:
            cols.append("alert_type")
        if "severity" in df_alerts.columns:
            cols.append("severity")
        if "title" in df_alerts.columns:
            cols.append("title")
        if "message" in df_alerts.columns:
            cols.append("message")
        if "month_date" in df_alerts.columns:
            cols.append("month_date")
        if "is_active" in df_alerts.columns:
            cols.append("is_active")
        if "created_at" in df_alerts.columns:
            cols.append("created_at")

        df_alerts = df_alerts[cols].rename(
            columns={
                "alert_type": "Type",
                "severity": "Severity",
                "title": "What’s happening",
                "message": "Details",
                "month_date": "Month",
                "is_active": "Active?",
                "created_at": "Created at",
            }
        )

        if "Details" in df_alerts.columns:
            df_alerts["Details"] = df_alerts["Details"].astype(str).str.slice(0, 160)

        st.dataframe(df_alerts, use_container_width=True)
        st.caption("Showing latest alerts for this business.")
                # --- Allow dismissing active alerts directly from here ---
        # Build a list of active alerts only (so we don't "dismiss" already resolved ones)
        active_alerts = [a for a in alerts if a.get("is_active")]

        if active_alerts:
            st.markdown("#### Dismiss an alert")

            alert_labels = []
            alert_id_map = {}

            for a in active_alerts:
                aid = a.get("id")
                atype = a.get("alert_type", "alert")
                sev = a.get("severity", "medium")
                msg = str(a.get("message", ""))[:80]  # short preview
                month_val = a.get("month_date") or ""
                label = f"[{sev}] {atype} – {msg} ({month_val})"
                alert_labels.append(label)
                alert_id_map[label] = aid

            selected_alert_label = st.selectbox(
                "Pick an alert to dismiss",
                ["-- Select --"] + alert_labels,
                key="alert_to_dismiss",
            )

            if selected_alert_label != "-- Select --":
                if st.button("Dismiss selected alert", key="dismiss_alert_button"):
                    alert_id = alert_id_map[selected_alert_label]
                    ok = dismiss_alert(alert_id)
                    if ok:
                        st.success("Alert dismissed.")
                        st.rerun()
                    else:
                        st.error("Could not dismiss alert. Please try again.")
        else:
            st.caption("No active alerts available to dismiss.")


    st.markdown("---")

    # ---------- Tasks for this business ----------
    st.subheader("✅ Tasks for this business")
    st.write("All open tasks across all pages and months.")

    tasks = fetch_all_open_tasks(selected_client_id)
    if not tasks:
        st.info("No open tasks yet for this client.")
    else:
        df_tasks = pd.DataFrame(tasks)
        cols = []
        if "page_name" in df_tasks.columns:
            cols.append("page_name")
        if "month_date" in df_tasks.columns:
            cols.append("month_date")
        if "title" in df_tasks.columns:
            cols.append("title")
        if "status" in df_tasks.columns:
            cols.append("status")
        if "created_at" in df_tasks.columns:
            cols.append("created_at")

        df_tasks = df_tasks[cols].rename(
            columns={
                "page_name": "Page",
                "month_date": "Month",
                "title": "Task",
                "status": "Status",
                "created_at": "Created at",
            }
        )
        st.dataframe(df_tasks, use_container_width=True)

        labels = []
        id_map = {}
        for t in tasks:
            task_id = t.get("id")
            title = t.get("title", "")
            page_name = t.get("page_name", "unknown")
            status = t.get("status", "open")
            label = f"[{status}] {title}  (page: {page_name})"
            labels.append(label)
            id_map[label] = task_id

        selected_label = st.selectbox(
            "Pick a task to mark as done (any page)",
            ["-- Select --"] + labels,
            key="global_task_to_close",
        )
        if selected_label != "-- Select --":
            if st.button("Mark selected task as done", key="global_close_task"):
                task_id = id_map[selected_label]
                ok = update_task_status(task_id, "done")
                if ok:
                    st.success("Task marked as done.")
                    st.rerun()
                else:
                    st.error("Could not update task. Please try again.")

    # Optional: still keep page-specific tasks for this page
    tasks_block("alerts_todos")




# ---------- Page: Issues & Notes ----------

def page_issues_notes():
    top_header("Issues & Notes")

    st.subheader("🧾 Open issues for this business")
    st.write(
        "This page brings together unresolved tasks, recent notes, and overdue invoices "
        "so you can see what still needs attention."
    )

    # ---------- Open tasks across all pages ----------
    st.markdown("### ✅ Open tasks")
    tasks = fetch_all_open_tasks(selected_client_id)
    if not tasks:
        st.caption("No open tasks yet for this client.")
    else:
        df_tasks = pd.DataFrame(tasks)
        cols = []
        if "page_name" in df_tasks.columns:
            cols.append("page_name")
        if "month_date" in df_tasks.columns:
            cols.append("month_date")
        if "title" in df_tasks.columns:
            cols.append("title")
        if "priority" in df_tasks.columns:
            cols.append("priority")
        if "status" in df_tasks.columns:
            cols.append("status")
        if "created_at" in df_tasks.columns:
            cols.append("created_at")

        df_tasks = df_tasks[cols].rename(
            columns={
                "page_name": "Page",
                "month_date": "Month",
                "title": "Task",
                "priority": "Priority",
                "status": "Status",
                "created_at": "Created at",
            }
        )
        st.dataframe(df_tasks, use_container_width=True)

    st.markdown("---")

    # ---------- Recent comments ----------
    st.markdown("### 💬 Recent notes & commentary")

    comments = fetch_recent_comments_for_client(selected_client_id, limit=30)
    if not comments:
        st.caption("No comments yet for this client.")
    else:
        df_comments = pd.DataFrame(comments)
        cols = []
        if "page_name" in df_comments.columns:
            cols.append("page_name")
        if "month_date" in df_comments.columns:
            cols.append("month_date")
        if "body" in df_comments.columns:
            cols.append("body")
        if "author_name" in df_comments.columns:
            cols.append("author_name")
        if "created_at" in df_comments.columns:
            cols.append("created_at")

        df_comments = df_comments[cols].rename(
            columns={
                "page_name": "Page",
                "month_date": "Month",
                "body": "Comment",
                "author_name": "Author",
                "created_at": "Created at",
            }
        )

        # Optionally truncate long comments for the table
        if "Comment" in df_comments.columns:
            df_comments["Comment"] = df_comments["Comment"].astype(str).str.slice(0, 160)

        st.dataframe(df_comments, use_container_width=True)

    st.markdown("---")

    # ---------- Overdue invoices (AR) ----------
    st.markdown("### 📥 Overdue invoices (customers who owe you)")

    overdue_ar = fetch_overdue_ar_for_client(selected_client_id)
    if overdue_ar.empty:
        st.caption("No overdue AR found for this client (based on due dates).")
    else:
        view = overdue_ar.copy()
        cols = []
        if "counterparty" in view.columns:
            cols.append("counterparty")
        if "amount" in view.columns:
            cols.append("amount")
        if "due_date" in view.columns:
            cols.append("due_date")
        if "status" in view.columns:
            cols.append("status")
        if "alert_flag" in view.columns:
            cols.append("alert_flag")

        view = view[cols].rename(
            columns={
                "counterparty": "Customer",
                "amount": "Amount",
                "due_date": "Due date",
                "status": "Status",
                "alert_flag": "Alert?",
            }
        )
        st.dataframe(view, use_container_width=True)

    st.markdown("---")

    # Page-specific notes & tasks (still useful here)
    comments_block("issues_notes")
    tasks_block("issues_notes")



# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------


def run_app():
    page_names = [
        "Business at a Glance",
        "Sales & Deals",
        "Team & Spending",
        "Cash & Bills",
        "Alerts & To-Dos",
        "Issues & Notes",
        "Client Settings",
    ]

    page_choice = st.sidebar.radio("Go to page", page_names)

    if page_choice == "Business at a Glance":
        page_business_overview()
    elif page_choice == "Sales & Deals":
        page_sales_deals()
    elif page_choice == "Team & Spending":
        page_team_spending()
    elif page_choice == "Cash & Bills":
        page_cash_bills()
    elif page_choice == "Alerts & To-Dos":
        page_alerts_todos    
    elif page_choice == "Issues & Notes":
        page_issues_notes()
    elif page_choice == "Client Settings":
        page_client_settings()


if __name__ == "__main__":
    run_app()


