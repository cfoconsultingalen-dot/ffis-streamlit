# ui/plotly_charts.py

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import plotly.graph_objects as go

# ---------- Minimal Plotly config (CFO-grade) ----------
PLOTLY_CONFIG_MINIMAL = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
}

# ---------- Theme ----------
def apply_nordic_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12, color="#1F2937"),
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color="#64748B"),
        ),
        hoverlabel=dict(font=dict(size=12)),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#64748B"),
        linecolor="#E5E7EB",
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#64748B"),
        linecolor="#E5E7EB",
        mirror=False,
    )
    return fig


def _month_labels(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    return dt.dt.strftime("%b %y")


def _money_fmt(currency_symbol: str) -> str:
    # Plotly uses d3 formatting inside hovertemplate
    # Example: "$%{y:,.0f}"
    return f"{currency_symbol}%{{y:,.0f}}"


# ---------- 12-month cashflow outlook ----------
def fig_cashflow_outlook_12m(
    engine_df: pd.DataFrame,
    selected_month_start,
    currency_symbol: str,
) -> go.Figure:
    if engine_df is None or engine_df.empty:
        return go.Figure()

    start = pd.to_datetime(selected_month_start).replace(day=1)
    month_index = pd.date_range(start=start, periods=12, freq="MS")

    df = engine_df.copy()
    df["month_date"] = pd.to_datetime(df.get("month_date"), errors="coerce")
    df = df[df["month_date"].notna()].copy()
    df["month_date"] = df["month_date"].dt.to_period("M").dt.to_timestamp()

    grid = pd.DataFrame({"month_date": month_index})
    merged = grid.merge(df, on="month_date", how="left")

    for c in ["operating_cf", "investing_cf", "financing_cf", "closing_cash"]:
        merged[c] = pd.to_numeric(merged.get(c), errors="coerce").fillna(0.0)

    x = _month_labels(merged["month_date"])
    hover_money = _money_fmt(currency_symbol)

    fig = go.Figure()

    # Stacked CF bars
    fig.add_trace(go.Bar(
        name="Operating CF",
        x=x,
        y=merged["operating_cf"],
        hovertemplate=f"Operating CF<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Investing CF",
        x=x,
        y=merged["investing_cf"],
        hovertemplate=f"Investing CF<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Financing CF",
        x=x,
        y=merged["financing_cf"],
        hovertemplate=f"Financing CF<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))

    # Closing cash line (secondary axis)
    fig.add_trace(go.Scatter(
        name="Closing cash",
        x=x,
        y=merged["closing_cash"],
        mode="lines+markers",
        yaxis="y2",
        hovertemplate=f"Closing cash<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))

    fig.update_layout(
        barmode="relative",
        yaxis=dict(title="Cashflow movement"),
        yaxis2=dict(
            title="Closing cash",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    return apply_nordic_plotly_theme(fig)


# ---------- Revenue trend (6 months) ----------
def fig_revenue_trend(
    rev_trend_df: pd.DataFrame,
    currency_symbol: str,
) -> go.Figure:
    if rev_trend_df is None or rev_trend_df.empty:
        return go.Figure()

    df = rev_trend_df.copy()
    df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
    df = df[df["month_date"].notna()].copy()
    df = df.sort_values("month_date")

    x = df["month_date"].dt.strftime("%b %y")
    y = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)

    hover_money = _money_fmt(currency_symbol)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Recognised revenue",
        x=x,
        y=y,
        mode="lines+markers",
        hovertemplate=f"Revenue<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(title="Revenue"))
    return apply_nordic_plotly_theme(fig)


# ---------- Payroll vs other cost (6 months) ----------
def fig_payroll_vs_other_cost(
    selected_client_id: str,
    selected_month_start,
    currency_symbol: str,
    fetch_cashflow_summary_for_client,
    fetch_ar_ap_for_client,
    compute_payroll_by_month,
) -> go.Figure:
    window_months = pd.date_range(start=pd.to_datetime(selected_month_start).replace(day=1), periods=6, freq="MS")

    engine_df_all = fetch_cashflow_summary_for_client(selected_client_id)
    df_ar, df_ap = fetch_ar_ap_for_client(selected_client_id)
    payroll_by_month = compute_payroll_by_month(selected_client_id, window_months)

    if engine_df_all is None or engine_df_all.empty:
        return go.Figure()

    engine_df_all = engine_df_all.copy()
    engine_df_all["month_date"] = pd.to_datetime(engine_df_all.get("month_date"), errors="coerce")
    engine_df_all = engine_df_all[engine_df_all["month_date"].notna()].copy()
    engine_df_all["month_date"] = engine_df_all["month_date"].dt.to_period("M").dt.to_timestamp()

    engine_window = engine_df_all[engine_df_all["month_date"].isin(window_months)][["month_date", "operating_cf"]].copy()

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

    ap_monthly["ap_cash_out"] = pd.to_numeric(ap_monthly.get("ap_cash_out"), errors="coerce").fillna(0.0)

    payroll_series = pd.DataFrame({
        "month_date": window_months,
        "payroll_cash": [float(payroll_by_month.get(m, 0.0)) for m in window_months],
    })

    merged = pd.DataFrame({"month_date": window_months})
    merged = merged.merge(engine_window, on="month_date", how="left")
    merged = merged.merge(ap_monthly, on="month_date", how="left")
    merged = merged.merge(payroll_series, on="month_date", how="left")

    merged["operating_cf"] = pd.to_numeric(merged.get("operating_cf"), errors="coerce").fillna(0.0)
    merged["ap_cash_out"] = pd.to_numeric(merged.get("ap_cash_out"), errors="coerce").fillna(0.0)
    merged["payroll_cash"] = pd.to_numeric(merged.get("payroll_cash"), errors="coerce").fillna(0.0)
    merged = merged.sort_values("month_date")

    x = merged["month_date"].dt.strftime("%b %y")
    hover_money = _money_fmt(currency_symbol)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name="Net operating CF",
        x=x,
        y=merged["operating_cf"],
        mode="lines+markers",
        hovertemplate=f"Net operating CF<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Payroll (cash out)",
        x=x,
        y=merged["payroll_cash"],
        mode="lines+markers",
        hovertemplate=f"Payroll<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Bills / AP (cash out)",
        x=x,
        y=merged["ap_cash_out"],
        mode="lines+markers",
        hovertemplate=f"Bills / AP<br>%{{x}}<br>{hover_money}<extra></extra>",
    ))

    fig.update_layout(yaxis=dict(title="Amount (cash)"))
    return apply_nordic_plotly_theme(fig)


