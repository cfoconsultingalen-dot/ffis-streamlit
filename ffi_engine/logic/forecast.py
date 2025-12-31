

def page_business_overview():
    top_header("Business at a Glance")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    # --- Client settings (for opening cash, thresholds, etc.) ---
    settings = get_client_settings(selected_client_id)

    # --- Currency for this client ---
    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # --- Recognised revenue (this month & last month) from Sales engine ---
    this_month_rev = get_recognised_revenue_for_month(selected_client_id, selected_month_start)

    prev_month_date = (
        pd.to_datetime(selected_month_start) - pd.DateOffset(months=1)
    ).to_period("M").to_timestamp().date()

    prev_month_rev = get_recognised_revenue_for_month(selected_client_id, prev_month_date)

    rev_delta_label = None
    if prev_month_rev is not None:
        diff = this_month_rev - prev_month_rev
        rev_delta_label = f"{diff:+,.0f}"

    # Pull KPIs from Supabase
    kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    # Alerts (these read from settings + engine/kpis)
    ensure_runway_alert_for_month(selected_client_id, selected_month_start)
    ensure_cash_danger_alert_for_month(selected_client_id, selected_month_start)

    # ------------------------------------------------------------------
    # Engine df + runway (FETCH ONCE, COMPUTE ONCE, REUSE EVERYWHERE)
    # ------------------------------------------------------------------
    engine_df = fetch_cashflow_summary_for_client_as_of(
        selected_client_id,
        selected_month_start,
    )

    runway_from_engine = None
    effective_burn = None
    if engine_df is not None and not engine_df.empty:
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
        if deltas.get("money_in") is not None:
            money_in_delta_label = f"{deltas['money_in']:+,.0f}"
        if deltas.get("money_out") is not None:
            money_out_delta_label = f"{deltas['money_out']:+,.0f}"
        if deltas.get("cash") is not None:
            cash_delta_label = f"{deltas['cash']:+,.0f}"
        if deltas.get("runway") is not None:
            runway_delta_label = f"{deltas['runway']:+.1f} mo"

    # Prefer engine runway if available
    if runway_from_engine is not None:
        runway_months_val = f"{runway_from_engine:.1f}"

    # ---------- Top KPI row ----------
    kpi_col1, kpi_col0, kpi_col2, kpi_col3, kpi_col4 = st.columns(5)

    with kpi_col1:
        st.metric(
            f"💰 Expect Cash in ({currency_code})",
            format_money(money_in_val, currency_symbol),
            delta=money_in_delta_label,
            help=(
                f"Expected cash **in** this month from customers "
                f"(unpaid AR with expected_date in this month, in {currency_code})."
            ),
        )
        st.caption(expl.get("money_in", ""))

    with kpi_col0:
        st.metric(
            f"📊 This month (recognised revenue, {currency_code})",
            f"{currency_symbol}{this_month_rev:,.0f}",
            delta=rev_delta_label,
            help=(
                "Recognised revenue for this month based on your Sales & Deals "
                "revenue recognition schedule (SaaS / milestone / POC / etc)."
            ),
        )

    with kpi_col2:
        st.metric(
            f"💸 Expect Cash out ({currency_code})",
            format_money(money_out_val, currency_symbol),
            delta=money_out_delta_label,
            help=(
                "Expected cash **out** this month: unpaid supplier bills due/expected this month "
                "+ payroll cash for this month + opex for this month "
                f"(in {currency_code})."
            ),
        )
        st.caption(expl.get("money_out", ""))

    with kpi_col3:
        st.metric(
            f"🏦 Cash in bank ({currency_code})",
            format_money(cash_in_bank_val, currency_symbol),
            delta=cash_delta_label,
        )
        st.caption(expl.get("cash", ""))

    with kpi_col4:
        st.metric(
            "🧯 Runway (months)",
            "—" if runway_months_val in (None, "—") else f"{float(runway_months_val):.1f}",
            delta=runway_delta_label,
        )
        st.caption(expl.get("runway", ""))

    # ---------- This month in 20 seconds ----------
    st.subheader("📌 This month in 20 seconds")
    summary_text = build_month_summary_insight(selected_client_id, selected_month_start)
    st.info(summary_text)

    st.markdown("---")

    # ---------- Health verdict ----------
    runway_num = None
    try:
        runway_num = float(runway_months_val) if runway_months_val not in (None, "—") else None
    except Exception:
        runway_num = None

    cash_num = None
    if kpis:
        cash_num = _safe_float(kpis.get("cash_balance"), default=None)

    rev_num = _safe_float(money_in_val, 0.0) if money_in_val != "—" else 0.0
    burn_num = _safe_float(money_out_val, 0.0) if money_out_val != "—" else 0.0

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

    # ---------- 6-month trends ----------
    st.markdown("---")
    st.subheader("📈 6-month trends – revenue, Operating Cash Flow, cash")

    rev_trend_df = _build_revenue_trend_df(selected_client_id, selected_month_start)

    # Use the same engine_df we fetched once above
    burn_trend_df, cash_trend_df = _build_burn_and_cash_trend_df(
        engine_df,
        selected_month_start,
    )

    c_rev, c_burn, c_cash = st.columns(3)

    with c_rev:
        st.caption("Recognised revenue (last 6 months)")
        if rev_trend_df is None or rev_trend_df.empty:
            st.caption("No recognised revenue data available for this 6-month window.")
        else:
            rev_chart = (
                alt.Chart(rev_trend_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %y")),
                    y=alt.Y("Revenue:Q", title=f"Revenue ({currency_code})"),
                    tooltip=[
                        alt.Tooltip("Month:N", title="Month"),
                        alt.Tooltip("Revenue:Q", title="Revenue", format=",.0f"),
                    ],
                )
                .properties(height=200)
            )
            st.altair_chart(rev_chart.interactive(), use_container_width=True)

    with c_burn:
        st.caption("Operating Cashflow (last 6 months)")
        if burn_trend_df is None or burn_trend_df.empty:
            st.caption("No operating cashflow data available for this 6-month window.")
        else:
            burn_chart = (
                alt.Chart(burn_trend_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %y")),
                    y=alt.Y("Burn:Q", title=f"Burn ({currency_code})"),
                    tooltip=[
                        alt.Tooltip("Month:N", title="Month"),
                        alt.Tooltip("Burn:Q", title="Burn", format=",.0f"),
                    ],
                )
                .properties(height=200)
            )
            st.altair_chart(burn_chart.interactive(), use_container_width=True)

    with c_cash:
        st.caption("Cash balance (last 6 months)")
        if cash_trend_df is None or cash_trend_df.empty:
            st.caption("No cash balance data available for this 6-month window.")
        else:
            cash_chart = (
                alt.Chart(cash_trend_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %y")),
                    y=alt.Y("Closing cash:Q", title=f"Closing cash ({currency_code})"),
                    tooltip=[
                        alt.Tooltip("Month:N", title="Month"),
                        alt.Tooltip("Closing cash:Q", title="Closing cash", format=",.0f"),
                    ],
                )
                .properties(height=200)
            )
            st.altair_chart(cash_chart.interactive(), use_container_width=True)

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

    # ---------------------------------------------
    # Cashflow reconciliation for Focus Month
    # ---------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Cashflow reconciliation – Focus Month")

    recon_df = build_cashflow_recon_for_month(
        selected_client_id,
        selected_month_start,
        opening_cash_hint=settings.get("opening_cash_start"),
    )
    if recon_df is None or recon_df.empty:
        st.caption("No reconciliation data available for this month yet.")
    else:
        st.dataframe(recon_df, width="stretch")
        st.caption(
            "This table reconciles AR/AP, payroll, opex, other income, investing, and financing "
            "into Operating CF, Free CF, and Closing cash – and compares them to the engine row "
            "in cashflow_summary and the KPI cash balance."
        )

    # ---------------------------------------------
    # Cashflow engine – 12-month breakdown (reuse engine_df)
    # ---------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Cashflow engine – 12-month breakdown")

    if engine_df is None or engine_df.empty:
        st.caption(
            "No cashflow engine data yet for this Focus Month. "
            "Hit **'Rebuild cashflow for next 12 months'** above to generate it."
        )
    else:
        start = pd.to_datetime(selected_month_start).replace(day=1)
        month_index = pd.date_range(start=start, periods=12, freq="MS")
        grid = pd.DataFrame({"month_date": month_index})

        df = engine_df.copy()
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce")
        df["month_date"] = df["month_date"].dt.to_period("M").dt.to_timestamp()

        engine_window = df[df["month_date"].isin(month_index)].copy()

        merged = grid.merge(engine_window, on="month_date", how="left", suffixes=("", "_raw"))

        if merged.empty:
            st.caption(
                "No cashflow rows in cashflow_summary for this window yet. "
                "Try rebuilding cashflow or picking a different focus month."
            )
        else:
            numeric_cols = [
                "operating_cf",
                "investing_cf",
                "financing_cf",
                "free_cash_flow",
                "closing_cash",
            ]
            for c in numeric_cols:
                if c in merged.columns:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

            if "cash_danger_flag" in merged.columns:
                merged["cash_danger_flag"] = merged["cash_danger_flag"].fillna(False)

            merged["Month"] = merged["month_date"].dt.strftime("%b %Y")

            cols = ["Month"]
            for c in [
                "operating_cf",
                "investing_cf",
                "financing_cf",
                "free_cash_flow",
                "closing_cash",
                "cash_danger_flag",
            ]:
                if c in merged.columns:
                    cols.append(c)

            view = merged[cols].rename(
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
                "that feeds the alerts and scenario views. "
                "Each row is anchored to your selected Focus Month."
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
        engine_df_all["month_date"] = pd.to_datetime(engine_df_all["month_date"], errors="coerce")

        engine_window = engine_df_all[
            engine_df_all["month_date"].isin(window_months)
        ][["month_date", "operating_cf"]].copy()

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

        payroll_series = (
            pd.Series(
                [float(payroll_by_month.get(m, 0.0)) for m in window_months],
                index=window_months,
                name="payroll_cash",
            )
            .reset_index()
            .rename(columns={"index": "month_date"})
        )

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
                x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
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
    # Ranked next actions (end-of-page action box)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("✅ Ranked next actions (do these first)")

    actions = []

    if runway_num is not None and runway_num <= 4.0:
        actions.append("Start funding prep: build your raise plan, timeline, and target amount (Funding Strategy page).")
        actions.append("Freeze / slow discretionary spend and hiring until runway improves.")
        actions.append("Aggressively chase overdue AR and pull forward collections (Cash & Bills page).")
    else:
        if burn_num > rev_num:
            actions.append("Reduce burn drivers (payroll + vendor commitments). Identify 1–2 cuts that move the needle.")
            actions.append("Improve collections: tighten terms, follow-ups, incentives for early payment (Cash & Bills page).")
            actions.append("Review pricing/discounting: ensure growth isn’t destroying cash.")
        else:
            actions.append("Keep burn controlled and invest only where ROI is clear (avoid unnecessary fixed commitments).")
            actions.append("Strengthen collection discipline so AR doesn’t become a hidden cash leak (Cash & Bills page).")
            actions.append("Run a quick scenario to stress-test the next 90 days (Scenarios section above).")

    actions = actions[:3]
    st.markdown("\n".join([f"{i+1}. {a}" for i, a in enumerate(actions)]))
    st.caption("This list is intentionally short. The goal is focus, not a to-do dump.")

        # ------------------------------------------------------------------
    # NEW: Business Health + Primary Focus + Main Actions + Governance
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Business governance summary")

    focus_ts = pd.to_datetime(selected_month_start).to_period("M").to_timestamp()

    # Inputs needed for hard rules
    prev_runway_num = None
    prev_burn_num = None
    prev_cash_num = None

    # We already have prev_month_date and prev_month_rev above
    # Attempt to get last month KPIs for deltas/governance (if available)
    try:
        prev_kpis = fetch_kpis_for_client_month(selected_client_id, prev_month_date)
        if prev_kpis:
            prev_cash_num = _safe_float(prev_kpis.get("cash_balance"), default=None)
            prev_runway_num = _safe_float(prev_kpis.get("runway_months"), default=None)
            prev_burn_num = _safe_float(prev_kpis.get("burn"), default=None)
    except Exception:
        prev_kpis = None

    # Cliff within 90d (prefer an alert if you already store it; else estimate from runway)
    cash_cliff_90d = False
    try:
        # If your alerts include something like "cash_danger" with a due date, use that.
        # Otherwise: approximate cliff risk from runway < 3 months (~90d)
        if runway_num is not None and runway_num < 3:
            cash_cliff_90d = True
    except Exception:
        cash_cliff_90d = False

    # “One-off cash outflow occurred this month” (mechanical proxy)
    # Use your investing+financing recon table as signal if you have it; otherwise keep conservative False.
    one_off_outflow_flag = False
    try:
        # If recon_df exists and has investing/financing spikes you can detect, do it.
        # Minimal safe proxy: large negative delta in cash with burn not increasing.
        if cash_delta_label is not None:
            # deltas['cash'] already exists earlier
            if deltas.get("cash") is not None and deltas.get("money_out") is not None:
                if float(deltas["cash"]) < 0 and float(deltas["money_out"]) <= 0:
                    one_off_outflow_flag = True
    except Exception:
        one_off_outflow_flag = False

    # Planned burn (optional) - use a setting if you store it; else None
    planned_burn = None
    try:
        planned_burn = _safe_float(settings.get("planned_burn_monthly"), default=None) if settings else None
    except Exception:
        planned_burn = None

    # Health state (mechanical)
    raw_state, health_msg = _compute_business_health_state(
        focus_month=focus_ts,
        runway_months=runway_num,
        cash_balance=cash_num,
        revenue_this_month=float(this_month_rev or 0.0),
        revenue_prev_month=float(prev_month_rev) if prev_month_rev is not None else None,
        burn_this_month=float(burn_num or 0.0),
        burn_prev_month=float(prev_burn_num) if prev_burn_num is not None else None,
        cash_delta_mom=deltas.get("cash"),
        burn_delta_mom=deltas.get("money_out"),
        rev_delta_mom=deltas.get("money_in"),
        planned_burn=planned_burn,
        cash_cliff_within_90d=cash_cliff_90d,
        one_off_outflow_flag=one_off_outflow_flag,
        forecast_runway_down_2m=False,  # set True only if you can compute it reliably
    )

    # Governance: one level change per month
    prev_state = None

    # Prefer DB if you have it; else session_state fallback
    # If you later add DB helpers:
    #   prev_state = fetch_business_health_state_for_month(selected_client_id, prev_month_date)
    #   save_business_health_state_for_month(selected_client_id, selected_month_start, final_state)
    try:
        prev_state = st.session_state.get(f"_health_state_prev_{selected_client_id}_{str(prev_month_date)}")
    except Exception:
        prev_state = None

    final_state = _clamp_one_level_change(prev_state, raw_state)

    # Persist current month state for next month governance (session fallback)
    try:
        st.session_state[f"_health_state_prev_{selected_client_id}_{str(selected_month_start)}"] = final_state
    except Exception:
        pass

    # Render Health section (ONLY 3 states)
    st.markdown("### 1) Business Health Overall Message")
    if final_state == "AT RISK":
        st.error(f"🔴 **AT RISK** — {health_msg}")
    elif final_state == "WATCH":
        st.warning(f"🟠 **WATCH** — {health_msg}")
    else:
        st.success(f"🟢 **STABLE** — {health_msg}")

    # Primary focus (single focus only, KPI-derived)
    primary_focus = _compute_primary_focus(
        health_state=final_state,
        runway_months=runway_num,
        cash_cliff_within_90d=cash_cliff_90d,
        cash_delta_mom=deltas.get("cash"),
        burn_delta_mom=deltas.get("money_out"),
        revenue_delta_mom=deltas.get("money_in"),
        revenue_this_month=float(this_month_rev or 0.0),
        burn_this_month=float(burn_num or 0.0),
        payroll_share_up=False,       # wire later if you compute it
        ar_collections_issue=False,   # wire later if you compute it
    )

    st.markdown("### 2) Primary Focus (single)")
    st.info(f"**{primary_focus}**")

    # Main actions (3–5 max) + KPI trace per action
    st.markdown("### 3) Main Actions (this month)")

    actions = _build_main_actions(primary_focus=primary_focus)

    # HARD RULE: If no actions can be justified by KPI traces, show none.
    if not actions:
        st.caption("No actions generated because no CFO-grade action could be defensibly traced to a KPI shown on this page.")
    else:
        for i, a in enumerate(actions[:5], start=1):
            st.markdown(f"**{i}. {a['action']}**")
            st.caption(f"Why: {a['why']}")
            st.caption("KPI trace: " + " • ".join(a["kpi_trace"]))

