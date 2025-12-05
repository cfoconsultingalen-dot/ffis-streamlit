import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
from supabase import create_client
import altair as alt

# ---------- Supabase config ----------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase_client()

st.set_page_config(page_title="FFIS – Founder Financial Intelligence", layout="wide")


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

@st.cache_data(ttl=60)
def fetch_cashflow_summary_for_client(client_id):
    """
    Fetch cashflow_summary rows for this client.
    """
    if client_id is None:
        return pd.DataFrame()

    try:
        res = (
            supabase
            .table("cashflow_summary")
            .select("*")
            .eq("client_id", str(client_id))
            .order("month_date", asc=True)
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def fetch_kpis_for_client_month(client_id, month_start: date):
    """
    Fetch a single KPI row from kpi_monthly for the given client and month.
    Returns a dict or None.
    """
    if client_id is None or month_start is None:
        return None

    try:
        response = (
            supabase
            .table("kpi_monthly")
            .select("*")
            .eq("client_id", str(client_id))
            .eq("month_date", month_start.isoformat())
            .single()
            .execute()
        )
        return response.data
    except Exception:
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
            "Month": [m.strftime("%b %Y") for m in month_index],
            "Expected revenue": [months[m] for m in month_index],
        }
    )
    return out_df


def create_pipeline_deal(
    client_id,
    deal_name: str,
    customer_name: str,
    value_total: float,
    probability_pct: float,
    method: str,
    stage: str,
    start_month: date,
    end_month: date | None = None,
    commentary: str | None = None,
):
    """
    Insert a new deal into revenue_pipeline.
    """
    if client_id is None or not deal_name.strip():
        return False

    data = {
        "client_id": str(client_id),
        "deal_name": deal_name.strip(),
        "customer_name": customer_name.strip() if customer_name else None,
        "value_total": float(value_total) if value_total is not None else None,
        "currency": "AUD",
        "probability_pct": float(probability_pct) if probability_pct is not None else None,
        "method": method or "SaaS",  # default for now
        "stage": stage or "proposal",
        "start_month": start_month.isoformat() if start_month else None,
        "end_month": end_month.isoformat() if end_month else None,
        "commentary": commentary,
        "is_won": stage == "closed_won",
        "is_lost": stage == "closed_lost",
    }

    try:
        supabase.table("revenue_pipeline").insert(data).execute()
        return True
    except Exception:
        return False

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

def recompute_cashflow_from_ar_ap(
    client_id,
    base_month: date,
    n_months: int = 12,
    opening_cash_hint: float | None = None,
):
    """
    Core cashflow engine (v2):
    Produces rows for cashflow_summary matching your required schema:
        operating_cf, investing_cf, financing_cf,
        free_cash_flow, closing_cash, cash_danger_flag

    Operating CF = AR expected receipts - AP expected payments
    Investing CF = 0 for now
    Financing CF = 0 for now

    closing_cash = prior_closing_cash + free_cash_flow
    """
    if client_id is None or base_month is None:
        return False

    # 1) Month index (normalise to first-of-month just in case)
    base_month_start = pd.to_datetime(base_month).replace(day=1)
    month_index = pd.date_range(
        start=base_month_start,
        periods=n_months,
        freq="MS",
    )

    # 2) Pull AR/AP
    df_ar, df_ap = fetch_ar_ap_for_client(client_id)

    if not df_ar.empty:
        df_ar["amount"] = pd.to_numeric(df_ar["amount"], errors="coerce").fillna(0.0)
        df_ar["expected_date"] = pd.to_datetime(df_ar.get("expected_date"), errors="coerce")
        df_ar["bucket_month"] = df_ar["expected_date"].dt.to_period("M").dt.to_timestamp()

    if not df_ap.empty:
        df_ap["amount"] = pd.to_numeric(df_ap["amount"], errors="coerce").fillna(0.0)
        pay_col = "expected_payment_date" if "expected_payment_date" in df_ap.columns else "due_date"
        df_ap[pay_col] = pd.to_datetime(df_ap[pay_col], errors="coerce")
        df_ap["bucket_month"] = df_ap[pay_col].dt.to_period("M").dt.to_timestamp()

    # 3) Opening cash
    opening_cash = None
    if opening_cash_hint is not None:
        opening_cash = float(opening_cash_hint)
    else:
        kpi_row = fetch_kpis_for_client_month(client_id, base_month_start.date())
        if kpi_row and "cash_balance" in kpi_row:
            try:
                opening_cash = float(kpi_row["cash_balance"])
            except Exception:
                opening_cash = None

    if opening_cash is None:
        opening_cash = 120_000.0  # fallback

    current_cash = opening_cash
    results = []

    # 4) Month-by-month cashflow rows
    for m in month_index:
        # --- Cash In ---
        if not df_ar.empty:
            cash_in = float(df_ar.loc[df_ar["bucket_month"] == m, "amount"].sum())
        else:
            cash_in = 0.0

        # --- Cash Out ---
        if not df_ap.empty:
            cash_out = float(df_ap.loc[df_ap["bucket_month"] == m, "amount"].sum())
        else:
            cash_out = 0.0

        # Operating CF
        operating_cf = cash_in - cash_out

        # Investing & Financing (placeholder)
        investing_cf = 0.0
        financing_cf = 0.0

        # ✅ Free Cash Flow (correct formula)
        free_cf = operating_cf + investing_cf + financing_cf

        # Closing Cash
        closing_cash = current_cash + free_cf

        # Danger flag
        danger = closing_cash <= 0

        # Add row
        results.append(
            {
                "client_id": str(client_id),
                "month_date": m.date().isoformat(),
                "operating_cf": round(operating_cf, 2),
                "investing_cf": round(investing_cf, 2),
                "financing_cf": round(financing_cf, 2),
                "free_cash_flow": round(free_cf, 2),
                "closing_cash": round(closing_cash, 2),
                "cash_danger_flag": danger,
                "updated_at": datetime.utcnow().isoformat(),
            }
        )

        current_cash = closing_cash  # next month opening = this month closing

    # 5) Replace rows in Supabase
    try:
        supabase.table("cashflow_summary").delete().eq("client_id", str(client_id)).execute()
    except Exception:
        pass

    try:
        supabase.table("cashflow_summary").insert(results).execute()
        return True
    except Exception as e:
        print("Error writing cashflow:", e)
        return False



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

# Page navigation
pages = [
    "Business at a Glance",
    "Sales & Deals",
    "Team & Spending",
    "Cash & Bills",
    "Alerts & To-Dos",
    "Issues & Notes",
]

page = st.sidebar.radio("Go to page", pages)


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

@st.cache_data(ttl=60)
def get_client_settings(client_id):
    """
    Fetch AR/AP default days and alert thresholds for this client.
    Falls back to sensible defaults if no row exists yet.
    """
    defaults = {
        "ar_default_days": 30,
        "ap_default_days": 30,
        "runway_min_months": 4.0,
        "overspend_warn_pct": 10.0,
        "overspend_high_pct": 20.0,
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
        return {
            "ar_default_days": data.get("ar_default_days", defaults["ar_default_days"]),
            "ap_default_days": data.get("ap_default_days", defaults["ap_default_days"]),
            "runway_min_months": float(data.get("runway_min_months", defaults["runway_min_months"])),
            "overspend_warn_pct": float(data.get("overspend_warn_pct", defaults["overspend_warn_pct"])),
            "overspend_high_pct": float(data.get("overspend_high_pct", defaults["overspend_high_pct"])),
        }
    except Exception:
        return defaults

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
    elif runway < 4:
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


# ---------- Page: Business at a Glance ----------

def page_business_overview():
    top_header("Business at a Glance")

    # Pull KPIs from Supabase
    kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up runway alerts for this month (now reads client settings)
    ensure_runway_alert_for_month(selected_client_id, selected_month_start)

    # Auto-generate / clean up cash danger alert based on cashflow_summary
    ensure_cash_danger_alert_for_month(selected_client_id, selected_month_start)

    money_in = "—"
    money_out = "—"
    cash_in_bank = "—"
    runway_months = "—"

    if kpis:
        # Adapt these keys to match your kpi_monthly columns
        money_in = kpis.get("revenue", "—")
        money_out = kpis.get("burn", "—")
        cash_in_bank = kpis.get("cash_balance", "—")
        runway_months = kpis.get("runway_months", "—")

    # Top KPI row (now using data if available)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("💰 Money in", money_in, help="Total cash in for the month")
    with kpi_col2:
        st.metric("💸 Money out", money_out, help="Total cash out for the month")
    with kpi_col3:
        st.metric("🏦 Cash in bank", cash_in_bank)
    with kpi_col4:
        st.metric("🧯 Runway (months)", runway_months)

    # ---------- Key alerts for this business ----------
    st.markdown("---")
    st.subheader("🚨 Key alerts for this month")

    all_alerts = fetch_alerts_for_client(selected_client_id, only_active=True, limit=50)
    if not all_alerts:
        st.caption("No critical alerts right now. Keep doing what you're doing. 🙌")
    else:
        # Optionally filter by month if month_date is present
        alerts_for_month = []
        for a in all_alerts:
            month_value = a.get("month_date")
            if month_value and selected_month_start:
                # Compare only year/month
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

        # Sort by severity and show top 3
        alerts_sorted = sort_alerts_by_severity(alerts_for_month)
        top_alerts = alerts_sorted[:3]

        for alert in top_alerts:
            sev = str(alert.get("severity", "medium")).lower()
            msg = alert.get("message", "")
            atype = alert.get("alert_type", "alert")
            page_name = alert.get("page_name") or ""
            label = f"[{atype}] {msg}"

            # Use different visual style by severity
            if sev in ["critical", "high"]:
                st.error(label)
            elif sev == "medium":
                st.warning(label)
            else:
                st.info(label)

    st.markdown("---")

    # This month in 20 seconds
    st.subheader("📌 This month in 20 seconds")
    st.info(
        "This will show a simple, founder-friendly summary of what happened this month "
        "(revenue, spend, cash, key issues). For now, it's a placeholder."
    )
    st.markdown("---")

    # Allow manual rebuild of the cashflow engine from AR/AP + KPIs
    st.subheader("🔁 Cashflow engine controls")

    if st.button("Rebuild cashflow for next 12 months"):
        ok = recompute_cashflow_from_ar_ap(
            selected_client_id,
            base_month=selected_month_start,
            n_months=12,
            opening_cash_hint=None,  # or set a starting cash manually
        )
        if ok:
            st.success("Cashflow updated from AR/AP and KPIs.")
            st.rerun()
        else:
            st.error("Could not rebuild cashflow. Check AR/AP data and try again.")

    # Cash curve + danger month (real curve + marker)
    st.subheader("📉 Cash over time & danger month")

    cash_df = fetch_cash_curve_for_client(
        selected_client_id,
        base_month=selected_month_start,
        n_months=12,
    )

    if cash_df.empty:
        st.caption("No cash projection data yet for this business.")
    else:
        # Compute danger month
        danger_label, danger_date = compute_danger_month_from_cash(cash_df, "closing_cash")

        # Build Altair chart
        base = alt.Chart(cash_df).mark_line().encode(
            x=alt.X("month_label:N", title="Month"),
            y=alt.Y("closing_cash:Q", title="Projected cash in bank"),
            tooltip=[
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip("closing_cash:Q", title="Cash", format=",.0f"),
            ],
        )

        charts = [base]

        # Add red point on danger month if any
        if danger_label is not None:
            danger_point = alt.Chart(
                cash_df[cash_df["month_label"] == danger_label]
            ).mark_point(size=80, color="red").encode(
                x="month_label:N",
                y="closing_cash:Q",
                tooltip=[
                    alt.Tooltip("month_label:N", title="Danger month"),
                    alt.Tooltip("closing_cash:Q", title="Cash", format=",.0f"),
                ],
            )
            charts.append(danger_point)

        st.altair_chart(alt.layer(*charts).interactive(), use_container_width=True)

        # Text explanation for founder
        if danger_label is not None:
            st.markdown(
                f"🔴 **Danger month:** Based on your current plan, cash is projected to fall to zero "
                f"or below around **{danger_label}**. This is your red line for decisions."
            )
        else:
            st.markdown(
                "🟢 **No danger month in this view:** Cash stays above zero for the next 12 months "
                "based on the data loaded into your engine."
            )

    # ------------------------------------------------------------------
    # Cashflow engine – month-by-month breakdown (from cashflow_summary)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Cashflow engine – 12-month breakdown")

    engine_df = fetch_cashflow_summary_for_client(selected_client_id)

    if engine_df.empty:
        st.caption(
            "No cashflow engine data yet. "
            "Hit **'Rebuild cashflow for next 12 months'** above to generate it."
        )
    else:
        # Filter to the 12-month window starting from selected_month_start
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

            st.dataframe(view, use_container_width=True)
            st.caption(
                "This is the raw output of your core cashflow engine "
                "that feeds the chart and the cash danger alerts."
            )

    comments_block("business_overview")
    tasks_block("business_overview")


# ---------- Page: Sales & Deals ----------

def page_sales_deals():
    top_header("Sales & Deals")

    # Load pipeline with spinner
    with st.spinner("Loading your pipeline..."):
        df_pipeline = fetch_pipeline_for_client(selected_client_id)

    st.subheader("📂 Pipeline overview")

    if df_pipeline.empty:
        st.info("No deals in the pipeline yet for this business.")
    else:
        # Nice compact view for founders
        display_cols = []
        for col in [
            "deal_name",
            "customer_name",
            "stage",
            "value_total",
            "probability_pct",
            "start_month",
            "end_month",
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
            }
        )
        st.dataframe(view, use_container_width=True)

        # Some quick stats for the founder
        total_pipeline = df_pipeline["value_total"].fillna(0).sum()
        weighted_pipeline = (df_pipeline["value_total"].fillna(0) *
                             df_pipeline["probability_pct"].fillna(0) / 100).sum()
        open_deals = len(df_pipeline)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Total pipeline", f"${total_pipeline:,.0f}")
        with k2:
            st.metric("Weighted pipeline", f"${weighted_pipeline:,.0f}")
        with k3:
            st.metric("Open deals", open_deals)

    st.markdown("---")

    st.subheader("📆 Expected revenue over time (simple model)")
    if df_pipeline.empty:
        st.caption("Add a few deals to see expected revenue by month.")
    else:
        curve_df = build_expected_revenue_curve(
            df_pipeline,
            base_month=selected_month_start,
            n_months=12,
        )
        st.line_chart(curve_df.set_index("Month"))

    st.markdown("---")

    st.subheader("➕ Add a new deal")
    with st.form(key="new_deal_form"):
        col1, col2 = st.columns(2)
        with col1:
            deal_name = st.text_input("Deal name")
            customer_name = st.text_input("Customer (optional)")
            value_total = st.number_input("Total value (AUD)", min_value=0.0, step=1000.0)
        with col2:
            probability_pct = st.slider("Win chance (%)", min_value=0, max_value=100, value=50)
            stage = st.selectbox(
                "Stage",
                ["idea", "proposal", "demo", "contract", "closed_won", "closed_lost"],
            )
            method = st.selectbox(
                "Revenue type",
                ["SaaS", "Milestone", "Accrual"],
            )

        start_month_input = st.date_input(
            "Start / expected close month",
            value=selected_month_start,
        )

        commentary = st.text_area("Notes (optional)")

        submitted = st.form_submit_button("Create deal")

        if submitted:
            ok = create_pipeline_deal(
                selected_client_id,
                deal_name=deal_name,
                customer_name=customer_name,
                value_total=value_total,
                probability_pct=probability_pct,
                method=method,
                stage=stage,
                start_month=start_month_input,
                end_month=None,
                commentary=commentary,
            )
            if ok:
                st.success("Deal added to pipeline.")
                st.rerun()
            else:
                st.error("Could not create deal. Please check inputs or try again later.")

    comments_block("sales_deals")
    tasks_block("sales_deals")



# ---------- Page: Team & Spending ----------

def page_team_spending():
    top_header("Team & Spending")

    # Auto-generate / clean up dept overspend alerts for this month
    ensure_dept_overspend_alert(selected_client_id, selected_month_start)
  # Load data with a spinner
    with st.spinner("Loading team and department data..."):
        df_hiring = fetch_hiring_monthly_for_client(selected_client_id)
        df_dept = fetch_dept_monthly_for_client(selected_client_id)

    # ---------- Team & payroll ----------
    st.subheader("👥 Team & payroll")

    if df_hiring.empty:
        st.info("No hiring / headcount data yet for this business.")
    else:
        # Build headcount + payroll by month (all departments)
        hiring_agg = (
            df_hiring
            .groupby("month_date", as_index=False)
            .agg(
                {
                    # fall back safely if some columns are missing
                    "cumulative_roles": "max" if "cumulative_roles" in df_hiring.columns else "sum",
                    "cumulative_payroll": "max" if "cumulative_payroll" in df_hiring.columns else "sum",
                }
            )
        )

        # Clean column names
        if "cumulative_roles" in hiring_agg.columns:
            hiring_agg = hiring_agg.rename(columns={"cumulative_roles": "Headcount"})
        else:
            hiring_agg["Headcount"] = 0

        if "cumulative_payroll" in hiring_agg.columns:
            hiring_agg = hiring_agg.rename(columns={"cumulative_payroll": "Payroll"})
        else:
            hiring_agg["Payroll"] = 0

        hiring_agg = hiring_agg.sort_values("month_date")
        hiring_agg["Month"] = hiring_agg["month_date"].dt.strftime("%b %Y")

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Total headcount over time")
            st.bar_chart(hiring_agg.set_index("Month")[["Headcount"]])
        with col2:
            st.caption("Total payroll over time")
            st.line_chart(hiring_agg.set_index("Month")[["Payroll"]])

    st.markdown("---")

    # ---------- Department spend vs plan ----------
    st.subheader("🏢 Department spend vs plan")

    if df_dept.empty:
        st.info("No department budget/actuals yet for this business.")
    else:
        # Decide which columns represent budget vs actual
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
            # Aggregate latest month (or filter by selected_month_start if you want)
            # For now, show the selected month only if present
            if "month_date" in df_dept.columns and selected_month_start is not None:
                df_dept_month = df_dept[df_dept["month_date"] == pd.to_datetime(selected_month_start)]
                if df_dept_month.empty:
                    # fallback: latest month available
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

            # Simple bar chart for plan vs actual
            st.caption("Plan vs actual by department (selected month)")
            chart_df = dept_view.set_index("Department")[["Plan", "Actual"]]
            st.bar_chart(chart_df)


    # ---------- Internal: CSV uploads for testing ----------
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


# ---------- Page: Cash & Bills ----------

def page_cash_bills():
    top_header("Cash & Bills")

    # Load current payment term settings for this client
    settings = get_client_settings(selected_client_id)
    ar_term = settings["ar_default_days"]
    ap_term = settings["ap_default_days"]

    # Settings UI
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

    # 🔹 Auto-generate / clean up AR overdue alerts based on current date
    # (uses these defaults inside the helper now)
    ensure_overdue_ar_alert(selected_client_id, as_of=date.today(), min_days_overdue=0)
    # Keep these values for captions below
    ar_days_display = ar_input
    ap_days_display = ap_input

    st.markdown("---")

    # 🔁 Keep your AR overdue alert logic here if you already had it
    # ensure_overdue_ar_alert(selected_client_id, as_of=date.today(), min_days_overdue=14

   # Use a spinner for the data-heavy part
    with st.spinner("Loading invoices and cash commitments..."):
        df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)

        # Add ageing info to AR using today's date and the current AR term
    if not df_ar.empty:
        df_ar = add_ar_aging(df_ar, as_of=date.today(), ar_default_days=ar_term)

        
    # ---------- AR ----------
    st.subheader("📥 Invoices coming in (AR)")
    st.caption(f"Default expectation: customers pay within {ar_input} day(s) of invoice.")
    st.write("Who owes you money and when it's expected in.")


    if df_ar.empty:
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
    if not df_ar.empty and "aging_bucket" in df_ar.columns and "amount" in df_ar.columns:
        st.markdown("#### AR ageing summary (by overdue bucket)")

        # Treat only invoices that are not fully paid/closed
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


    st.subheader("📤 Bills to pay (AP)")
    st.caption(f"Default expectation: you pay suppliers within {ap_input} day(s) of invoice.")
    st.write("Your upcoming bills and payment plans.")


    if df_ap.empty:
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
        # we now have pay_expected_date + maybe expected_payment_date
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
    if not df_ap.empty and "aging_bucket" in df_ap.columns and "amount" in df_ap.columns:
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


    st.subheader("🔍 Cash commitments coming up")
    commitments = build_cash_commitments(df_ar, df_ap, limit=7)

    if commitments.empty:
        st.info("No upcoming cash movements found yet.")
    else:
        st.dataframe(commitments, use_container_width=True)

    comments_block("cash_bills")
    tasks_block("cash_bills")

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
# ---------- Main router ----------

if page == "Business at a Glance":
    page_business_overview()
elif page == "Sales & Deals":
    page_sales_deals()
elif page == "Team & Spending":
    page_team_spending()
elif page == "Cash & Bills":
    page_cash_bills()
elif page == "Alerts & To-Dos":
    page_alerts_todos()
elif page == "Issues & Notes":
    page_issues_notes()
