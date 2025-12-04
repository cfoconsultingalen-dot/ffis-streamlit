import streamlit as st
from datetime import date, datetime
import pandas as pd
from supabase import create_client

# ---------- Supabase config ----------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="FFIS – Founder Financial Intelligence", layout="wide")


# ---------- Helpers (data + utilities) ----------

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
    for col in ["due_date", "expected_payment_date", "created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    df_ar = df[df["type"] == "AR"].copy()
    df_ap = df[df["type"] == "AP"].copy()

    return df_ar, df_ap


def build_cash_commitments(df_ar: pd.DataFrame, df_ap: pd.DataFrame, limit: int = 7):
    """
    Build a small table of upcoming cash in (AR) and cash out (AP).
    """
    frames = []

    if not df_ar.empty:
        tmp = df_ar.copy()
        tmp["direction"] = "Cash in"
        tmp["who"] = tmp.get("counterparty", "")
        tmp["amount"] = tmp.get("amount", 0)
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
    Fetch overdue AR rows for this client from ar_ap_tracker.
    Overdue = type = 'AR' AND due_date < as_of AND status != 'paid' (if status column exists).
    """
    if client_id is None:
        return pd.DataFrame()

    if as_of is None:
        as_of = date.today()

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

    if "due_date" in df.columns:
        df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce").dt.date
        df = df[df["due_date"].notna()]
        df = df[df["due_date"] < as_of]

    # Exclude fully paid if we have a status column
    if "status" in df.columns:
        df = df[~df["status"].str.lower().isin(["paid", "closed", "settled"])]

    return df


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

def fetch_alerts_for_client(client_id, only_active: bool = True, limit: int = 100):
    """
    Fetch alerts for this client from alerts table.
    By default returns only active alerts, most recent first.
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

        res = (
            query
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
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
            for c in existing:
                author = c.get("author_name") or "Unknown"
                body = c.get("body", "")
                created = c.get("created_at", "")[:16]  # trim timestamp
                st.markdown(f"**{author}** • _{created}_")
                st.write(body)
                st.markdown("---")
        else:
            st.caption("No comments yet for this page and month.")

        # New comment box
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

    st.markdown("---")

    # This month in 20 seconds
    st.subheader("📌 This month in 20 seconds")
    st.info(
        "This will show a simple, founder-friendly summary of what happened this month "
        "(revenue, spend, cash, key issues). For now, it's a placeholder."
    )

    # Cash curve + danger month (placeholder chart)
    st.subheader("📉 Cash over time & danger month")
    st.line_chart(pd.DataFrame({"Cash": [100, 90, 75, 60, 40, 20]}))

    st.markdown("Danger month will be highlighted here once the engine is wired in.")

    comments_block("business_overview")
    tasks_block("business_overview")


# ---------- Page: Sales & Deals ----------

def page_sales_deals():
    top_header("Sales & Deals")

    # 🔹 Load pipeline for this client
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

    # 🔹 Load data
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

    # Load AR/AP for this client
    df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)

    st.subheader("📥 Invoices coming in (AR)")
    st.write("Who owes you money and when it's expected in.")

    if df_ar.empty:
        st.info("No AR records yet for this client.")
    else:
        ar_view = df_ar.copy()
        cols = []
        if "counterparty" in ar_view.columns:
            cols.append("counterparty")
        if "amount" in ar_view.columns:
            cols.append("amount")
        if "due_date" in ar_view.columns:
            cols.append("due_date")
        if "expected_payment_date" in ar_view.columns:
            cols.append("expected_payment_date")
        if "status" in ar_view.columns:
            cols.append("status")
        if "alert_flag" in ar_view.columns:
            cols.append("alert_flag")

        ar_view = ar_view[cols].rename(
            columns={
                "counterparty": "Customer",
                "amount": "Amount",
                "due_date": "Due date",
                "expected_payment_date": "Expected date",
                "status": "Status",
                "alert_flag": "Alert?",
            }
        )
        st.dataframe(ar_view, use_container_width=True)

    st.subheader("📤 Bills to pay (AP)")
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
        if "expected_payment_date" in ap_view.columns:
            cols.append("expected_payment_date")
        if "status" in ap_view.columns:
            cols.append("status")
        if "alert_flag" in ap_view.columns:
            cols.append("alert_flag")

        ap_view = ap_view[cols].rename(
            columns={
                "counterparty": "Supplier",
                "amount": "Amount",
                "due_date": "Due date",
                "expected_payment_date": "Expected date",
                "status": "Status",
                "alert_flag": "Alert?",
            }
        )
        st.dataframe(ap_view, use_container_width=True)

    st.subheader("🔍 Cash commitments coming up")
    commitments = build_cash_commitments(df_ar, df_ap, limit=7)

    if commitments.empty:
        st.info("No upcoming cash movements found yet.")
    else:
        st.dataframe(commitments, use_container_width=True)

    comments_block("cash_bills")
    tasks_block("cash_bills")


# ---------- Page: Alerts & To-Dos ----------

def page_alerts_todos():
    top_header("Alerts & To-Dos")

    # ---------- Alerts ----------
    st.subheader("🚨 Alerts")
    st.write("Critical things your future self will thank you for noticing early.")

    alerts = fetch_alerts_for_client(selected_client_id, only_active=True, limit=100)

    if not alerts:
        st.info("No active alerts right now for this business.")
    else:
        df_alerts = pd.DataFrame(alerts)

        # Build a clean view for the founder
        cols = []
        if "page_name" in df_alerts.columns:
            cols.append("page_name")
        if "alert_type" in df_alerts.columns:
            cols.append("alert_type")
        if "severity" in df_alerts.columns:
            cols.append("severity")
        if "message" in df_alerts.columns:
            cols.append("message")
        if "month_date" in df_alerts.columns:
            cols.append("month_date")
        if "created_at" in df_alerts.columns:
            cols.append("created_at")

        df_view = df_alerts[cols].rename(
            columns={
                "page_name": "Page",
                "alert_type": "Type",
                "severity": "Severity",
                "message": "Message",
                "month_date": "Month",
                "created_at": "Created at",
            }
        )

        # Optional: shorten message in the table but keep full in hover
        if "Message" in df_view.columns:
            df_view["Message"] = df_view["Message"].astype(str).str.slice(0, 160)

        st.dataframe(df_view, use_container_width=True)

        # Simple dismiss UI
        st.markdown("#### Dismiss an alert")

        # Build labels for active alerts
        label_map = {}
        for a in alerts:
            aid = a.get("id")
            msg = a.get("message", "")
            sev = a.get("severity", "medium")
            atype = a.get("alert_type", "alert")
            label = f"[{sev}] {atype} – {msg[:60]}..."
            label_map[label] = aid

        alert_labels = list(label_map.keys())

        selected_label = st.selectbox(
            "Pick an alert to dismiss",
            ["-- Select --"] + alert_labels,
            key="alert_to_dismiss",
        )
        if selected_label != "-- Select --":
            if st.button("Dismiss selected alert", key="dismiss_alert_btn"):
                alert_id = label_map[selected_label]
                ok = dismiss_alert(alert_id)
                if ok:
                    st.success("Alert dismissed.")
                    st.rerun()
                else:
                    st.error("Could not dismiss alert. Please try again.")

    st.markdown("---")

    # ---------- Tasks ----------
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

        # Simple global "mark as done"
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

    # You can still keep page-specific tasks for this page if you like
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
