import streamlit as st
from datetime import date, timedelta
import pandas as pd
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
st.set_page_config(page_title="FFIS – Founder Financial Intelligence", layout="wide")


# ---------- Helpers ----------

def get_demo_clients():
    """
    TODO: Replace this with a Supabase query later.
    For now we just use dummy clients so the UI works.
    """
    return ["Demo Startup", "Client A", "Client B"]


def get_month_options(n_months: int = 18):
    """Return a list of month labels (most recent first)."""
    today = date.today().replace(day=1)
    months = []
    for i in range(n_months):
        m = today - pd.DateOffset(months=i)
        months.append(m.strftime("%b %Y"))  # e.g. "Dec 2025"
    return months


# ---------- Sidebar: global controls ----------

st.sidebar.title("FFIS")

# Client selector
clients = get_demo_clients()
selected_client = st.sidebar.selectbox("Select business", clients)

# Month selector
month_options = get_month_options(18)
selected_month_label = st.sidebar.selectbox("Focus month", month_options)

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


# ---------- Reusable UI pieces ----------

def comments_block(page_name: str):
    """
    Placeholder for comments UI.
    Later we will load/save comments from Supabase.
    """
    with st.expander("💬 Comments & notes", expanded=False):
        st.write("This is where you and your team will write notes for this page.")
        new_comment = st.text_area("Add a new comment", key=f"{page_name}_new_comment")
        if st.button("Save comment", key=f"{page_name}_save_comment"):
            # TODO: insert into Supabase comments table
            st.success("Comment saved (demo).")


def tasks_block(page_name: str):
    """
    Placeholder for tasks UI.
    Later we will load tasks + link to comments/alerts.
    """
    with st.expander("✅ Tasks & follow-ups", expanded=False):
        st.write("This is where action items for this page will show up.")
        new_task = st.text_input("Add a new task", key=f"{page_name}_new_task")
        if st.button("Create task", key=f"{page_name}_create_task"):
            # TODO: insert into Supabase tasks table
            st.success("Task created (demo).")


def top_header(title: str):
    st.title(title)
    st.caption(f"{selected_client} • Focus month: {selected_month_label}")


# ---------- Page: Business at a Glance ----------

def page_business_overview():
    top_header("Business at a Glance")

    # Top KPI row (placeholder values for now)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("💰 Money in", "—", help="Total cash in for the month")
    with kpi_col2:
        st.metric("💸 Money out", "—", help="Total cash out for the month")
    with kpi_col3:
        st.metric("🏦 Cash in bank", "—")
    with kpi_col4:
        st.metric("🧯 Runway (months)", "—")

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

    st.subheader("📂 Pipeline overview")
    st.write("This will show your live deals, their value, stage, and expected close month.")

    # Placeholder table
    demo_data = pd.DataFrame(
        {
            "Deal": ["Pilot with ABC", "Renewal with XYZ"],
            "Stage": ["Proposal", "Negotiation"],
            "Value": [25000, 40000],
            "Expected month": ["Jan 2026", "Feb 2026"],
        }
    )
    st.dataframe(demo_data, use_container_width=True)

    st.subheader("📆 Expected revenue over time")
    st.line_chart(pd.DataFrame({"Expected revenue": [0, 10, 25, 40, 55, 65]}))

    comments_block("sales_deals")
    tasks_block("sales_deals")


# ---------- Page: Team & Spending ----------

def page_team_spending():
    top_header("Team & Spending")

    st.subheader("👥 Team & payroll")
    st.write("This will show headcount, new hires, and payroll growth over time.")

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(pd.DataFrame({"Headcount": [3, 4, 6, 7, 8]}))
    with col2:
        st.line_chart(pd.DataFrame({"Payroll": [15, 18, 24, 28, 30]}))

    st.subheader("🏢 Department spend vs plan")
    st.write("This will show which departments are over or under budget.")
    st.dataframe(
        pd.DataFrame(
            {
                "Department": ["Product", "Sales", "Marketing"],
                "Plan": [20000, 15000, 12000],
                "Actual": [21000, 14000, 15000],
            }
        ),
        use_container_width=True,
    )

    comments_block("team_spending")
    tasks_block("team_spending")


# ---------- Page: Cash & Bills ----------

def page_cash_bills():
    top_header("Cash & Bills")

    st.subheader("📥 Invoices coming in (AR)")
    st.write("This will show who owes you money and when it's expected in.")
    st.dataframe(
        pd.DataFrame(
            {
                "Customer": ["Client A", "Client B"],
                "Amount": [12000, 8000],
                "Due date": ["15 Dec 2025", "25 Dec 2025"],
                "Status": ["Overdue", "Due soon"],
            }
        ),
        use_container_width=True,
    )

    st.subheader("📤 Bills to pay (AP)")
    st.write("This will show your upcoming bills and payment plans.")
    st.dataframe(
        pd.DataFrame(
            {
                "Supplier": ["AWS", "Contractor"],
                "Amount": [3000, 5000],
                "Due date": ["10 Dec 2025", "20 Dec 2025"],
                "Status": ["Paid", "Open"],
            }
        ),
        use_container_width=True,
    )

    st.subheader("🔍 Cash commitments coming up")
    st.write("This will highlight the next few big inflows and outflows.")

    comments_block("cash_bills")
    tasks_block("cash_bills")


# ---------- Page: Alerts & To-Dos ----------

def page_alerts_todos():
    top_header("Alerts & To-Dos")

    st.subheader("🚨 Alerts")
    st.write("Critical things your future self will thank you for noticing early.")

    # Placeholder alerts table
    st.dataframe(
        pd.DataFrame(
            {
                "Type": ["Runway", "AR overdue"],
                "Severity": ["High", "Medium"],
                "Message": [
                    "Runway under 4 months.",
                    "3 invoices overdue > 14 days.",
                ],
            }
        ),
        use_container_width=True,
    )

    st.subheader("✅ Tasks")
    st.write("This will show the main follow-ups created from comments, alerts, and your own notes.")

    tasks_block("alerts_todos")


# ---------- Page: Issues & Notes ----------

def page_issues_notes():
    top_header("Issues & Notes")

    st.subheader("🧾 All open issues")
    st.write(
        "This page will bring together unresolved comments and tasks from all other pages, "
        "so you have a single place to see what still needs attention."
    )

    # Placeholder content
    st.info("In the final version, this will be powered by the comments + tasks tables in Supabase.")

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
