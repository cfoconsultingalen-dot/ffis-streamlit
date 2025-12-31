
def page_sales_deals():
    """
    Sales & Deals — Updated exactly to your checklist:

    ✅ Pipeline overview KPIs
    ✅ Pipeline health panel
    ✅ Data checks warning
    ✅ “View full pipeline list” expander (kept)
    ✅ Waterfall chart (TABLE REMOVED)
    ✅ Revenue / Risk / What this means / Action plan (NO tables)
    ✅ Create / update deal (fully implemented)

    GUARANTEE:
    - Does NOT change any existing external function signatures.
    - Uses your existing functions if present; otherwise falls back safely to Supabase upsert.
    """

    top_header("Sales & Deals")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    # ---------------------------------------------------------------------
    # Revenue recognition settings (business default) — unchanged behaviour
    # ---------------------------------------------------------------------
    settings = get_client_settings(selected_client_id) or {}
    current_method = (settings.get("revenue_recognition_method") or "saas").lower()

    method_labels = {
        "saas": "SaaS subscription (spread over contract term)",
        "milestone": "Milestone / project (milestones by month)",
        "poc": "Percentage-of-completion (true-up by month)",
        "straight_line": "Straight-line service (start–end months)",
        "usage": "Usage-based (placeholder spreading)",
        "point_in_time": "Point-in-time goods delivery (lump on delivery month)",
    }

    method_display_map = {
        "saas": "SaaS subscription",
        "milestone": "Milestone project",
        "poc": "Percentage-of-completion",
        "straight_line": "Straight-line service",
        "usage": "Usage-based",
        "point_in_time": "Point-in-time goods",
    }

    method_options = list(method_labels.keys())
    default_index = method_options.index(current_method) if current_method in method_options else 0

    selected_method_label = st.selectbox(
        "Revenue recognition method (business default)",
        options=[method_labels[m] for m in method_options],
        index=default_index,
        help=(
            "Default method for this business. Deals can override this individually. "
            "The engine uses deal.method first, then falls back to this default."
        ),
    )
    reverse_map = {v: k for k, v in method_labels.items()}
    selected_method_code = reverse_map[selected_method_label]
    current_method_display = method_display_map.get(selected_method_code, "Straight-line service")

    apply_to_all = st.checkbox(
        "Apply this method to ALL existing deals (overwrite per-deal overrides)",
        value=False,
        help="Use this if you want business default to drive the numbers (Option A).",
    )

    if st.button("Save business default method"):
        ok = save_revenue_method(selected_client_id, selected_method_code)

        if ok and apply_to_all:
            ok2 = bulk_update_deal_methods_for_client(
                selected_client_id,
                method_display=method_display_map.get(selected_method_code, "Straight-line service"),
            )
            ok = ok and ok2

        if ok:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.success("Business default method updated.")
            st.rerun()
        else:
            st.error("Could not update method. Please try again.")

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Load pipeline — unchanged
    # ---------------------------------------------------------------------
    with st.spinner("Loading your pipeline..."):
        try:
            df_pipeline = fetch_pipeline_for_client(selected_client_id)
        except Exception as e:
            st.error("Could not load pipeline from Supabase (revenue_pipeline table).")
            st.caption(f"{type(e).__name__}: {e}")
            df_pipeline = pd.DataFrame()

    if df_pipeline is None:
        df_pipeline = pd.DataFrame()

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    with st.expander("🧪 Debug: contract months columns detected", expanded=False):
        if not df_pipeline.empty:
            cols = [
                c
                for c in [
                    "id",
                    "deal_name",
                    "customer_name",
                    "stage",
                    "value_total",
                    "probability_pct",
                    "method",
                    "contract_months",
                    "start_month",
                    "end_month",
                    "stage_entry_date",
                    "created_at",
                ]
                if c in df_pipeline.columns
            ]
            st.dataframe(df_pipeline[cols], width="stretch")

    # ---------------------------------------------------------------------
    # Normalise pipeline — unchanged (plus safe id/customer column handling)
    # ---------------------------------------------------------------------
    total_pipeline = 0.0
    open_deals_count = 0

    if not df_pipeline.empty:
        df_pipeline = df_pipeline.copy()
        df_pipeline["value_total"] = pd.to_numeric(df_pipeline.get("value_total"), errors="coerce").fillna(0.0)
        df_pipeline["probability_pct"] = pd.to_numeric(df_pipeline.get("probability_pct"), errors="coerce").fillna(0.0)
        df_pipeline["stage"] = df_pipeline.get("stage", "").astype(str).str.lower().fillna("")

        for col in ["start_month", "end_month", "stage_entry_date", "created_at"]:
            if col in df_pipeline.columns:
                df_pipeline[col] = pd.to_datetime(df_pipeline[col], errors="coerce")

        total_pipeline = float(df_pipeline["value_total"].sum())
        open_stages = ["idea", "proposal", "demo", "contract"]
        open_deals_count = int(df_pipeline["stage"].isin(open_stages).sum())

    # ---------------------------------------------------------------------
    # Helpers (local-only; no external APIs modified)
    # ---------------------------------------------------------------------
    def _safe_month_start(d) -> pd.Timestamp:
        return pd.to_datetime(d).to_period("M").to_timestamp()

    def _parse_month_input(maybe_date) -> pd.Timestamp | None:
        """
        Accepts:
        - date/datetime
        - string like '2025-12-01' or 'Dec 2025'
        Returns month-start timestamp or None.
        """
        if maybe_date is None or str(maybe_date).strip() == "":
            return None
        try:
            ts = pd.to_datetime(maybe_date, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.to_period("M").to_timestamp()
        except Exception:
            return None

    def _get_plan_for_month(client_id, focus_month) -> float:
        try:
            tdf = fetch_client_monthly_targets(client_id)
        except Exception:
            tdf = None
        if tdf is None or tdf.empty:
            return 0.0
        tdf = tdf.copy()
        tdf["month_date"] = pd.to_datetime(tdf["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        m = _safe_month_start(focus_month)
        row = tdf[tdf["month_date"] == m]
        if row.empty:
            return 0.0
        return float(pd.to_numeric(row.get("revenue_target"), errors="coerce").fillna(0.0).iloc[0])

    def _compute_weighted_pipeline(df_pipe: pd.DataFrame) -> float:
        if df_pipe is None or df_pipe.empty:
            return 0.0
        df = df_pipe.copy()
        df["stage"] = df.get("stage", "").astype(str).str.lower().fillna("")
        df["value_total"] = pd.to_numeric(df.get("value_total"), errors="coerce").fillna(0.0)
        df["probability_pct"] = pd.to_numeric(df.get("probability_pct"), errors="coerce").fillna(0.0)

        df = df[df["stage"] != "closed_lost"].copy()
        if df.empty:
            return 0.0

        def _pf(row) -> float:
            stg = str(row.get("stage", "")).lower()
            if stg == "closed_won":
                return 1.0
            p = float(row.get("probability_pct", 0.0) or 0.0)
            p = max(0.0, min(100.0, p))
            return p / 100.0

        df["weighted_value"] = df["value_total"] * df.apply(_pf, axis=1)
        return float(df["weighted_value"].sum())

    def _weighted_pipeline_last_month(df_pipe: pd.DataFrame, focus_month) -> float:
        if df_pipe is None or df_pipe.empty:
            return 0.0
        df = df_pipe.copy()
        if "created_at" not in df.columns:
            return _compute_weighted_pipeline(df)

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        cutoff = (
            pd.Timestamp(focus_month)
            .to_period("M")
            .to_timestamp()
            .tz_localize("UTC")
            + pd.DateOffset(months=1)
        )
        df = df[df["created_at"].notna() & (df["created_at"] < cutoff)].copy()
        return _compute_weighted_pipeline(df)

    def _get_actual_recognised_for_month_from_schedule_df(schedule_df: pd.DataFrame, focus_month) -> float:
        if schedule_df is None or schedule_df.empty:
            return 0.0
        df = schedule_df.copy()
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df = df[df["month_date"].notna()].copy()

        value_col = None
        for cand in ["recognised_revenue", "revenue_amount", "amount"]:
            if cand in df.columns:
                value_col = cand
                break
        if value_col is None:
            return 0.0

        m = _safe_month_start(focus_month)
        return float(pd.to_numeric(df.loc[df["month_date"] == m, value_col], errors="coerce").fillna(0.0).sum())

    def _get_next3_total_from_schedule_df(schedule_df: pd.DataFrame, focus_month) -> float:
        if schedule_df is None or schedule_df.empty:
            return 0.0
        df = schedule_df.copy()
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df = df[df["month_date"].notna()].copy()

        value_col = None
        for cand in ["recognised_revenue", "revenue_amount", "amount"]:
            if cand in df.columns:
                value_col = cand
                break
        if value_col is None:
            return 0.0

        focus_start = _safe_month_start(focus_month)
        end_4 = focus_start + pd.DateOffset(months=4)
        w = df[(df["month_date"] > focus_start) & (df["month_date"] < end_4)].copy()
        if w.empty:
            return 0.0
        return float(pd.to_numeric(w[value_col], errors="coerce").fillna(0.0).sum())

    def _avg_deal_size_active(df_pipe: pd.DataFrame) -> float:
        if df_pipe is None or df_pipe.empty:
            return 0.0
        active = df_pipe[df_pipe["stage"].isin(["proposal", "demo", "contract"])].copy()
        if active.empty:
            return 0.0
        active["value_total"] = pd.to_numeric(active.get("value_total"), errors="coerce").fillna(0.0)
        return float(active["value_total"].sum() / max(1, len(active)))

    def _cash_received_this_prev_month(client_id, focus_month) -> tuple[float, float]:
        df_ar, _ = fetch_ar_ap_for_client(client_id)
        if df_ar is None or df_ar.empty:
            return (0.0, 0.0)

        ar = df_ar.copy()
        if "payment_date" not in ar.columns or "amount" not in ar.columns:
            return (0.0, 0.0)

        ar["payment_date"] = pd.to_datetime(ar["payment_date"], errors="coerce")
        ar["amount"] = pd.to_numeric(ar["amount"], errors="coerce").fillna(0.0)
        ar["status"] = ar.get("status", "").astype(str).str.lower().fillna("")

        paid = ar[(ar["status"] == "paid") & ar["payment_date"].notna()].copy()
        if paid.empty:
            return (0.0, 0.0)

        focus_start = _safe_month_start(focus_month)
        prev_start = focus_start - pd.DateOffset(months=1)

        paid["pay_month"] = paid["payment_date"].dt.to_period("M").dt.to_timestamp()
        cash_this = float(paid.loc[paid["pay_month"] == focus_start, "amount"].sum())
        cash_prev = float(paid.loc[paid["pay_month"] == prev_start, "amount"].sum())
        return (cash_this, cash_prev)

    def _concentration_top2_weighted(df_pipe: pd.DataFrame) -> tuple[float, str]:
        if df_pipe is None or df_pipe.empty:
            return (0.0, "")
        df = df_pipe.copy()
        df["stage"] = df.get("stage", "").astype(str).str.lower().fillna("")
        df["value_total"] = pd.to_numeric(df.get("value_total"), errors="coerce").fillna(0.0)
        df["probability_pct"] = pd.to_numeric(df.get("probability_pct"), errors="coerce").fillna(0.0)

        df = df[df["stage"] != "closed_lost"].copy()
        if df.empty:
            return (0.0, "")

        def _pf(row) -> float:
            stg = str(row.get("stage", "")).lower()
            if stg == "closed_won":
                return 1.0
            p = float(row.get("probability_pct", 0.0) or 0.0)
            p = max(0.0, min(100.0, p))
            return p / 100.0

        df["weighted_value"] = df["value_total"] * df.apply(_pf, axis=1)
        total = float(df["weighted_value"].sum() or 0.0)
        if total <= 0:
            return (0.0, "")

        df = df.sort_values("weighted_value", ascending=False).copy()
        top2 = df.head(2)
        share = float(top2["weighted_value"].sum()) / total
        names = [str(r.get("deal_name") or "Unnamed") for _, r in top2.iterrows()]
        return (float(share), ", ".join(names[:2]))

    def _data_issues(df_pipe: pd.DataFrame) -> list[str]:
        issues = []
        if df_pipe is None or df_pipe.empty:
            return issues

        df = df_pipe.copy()
        df["stage"] = df.get("stage", "").astype(str).str.lower().fillna("")

        if "start_month" in df.columns:
            missing_close = pd.to_datetime(df["start_month"], errors="coerce").isna().sum()
            if missing_close > 0:
                issues.append(f"{missing_close} deal(s) missing expected close month")

        if "method" in df.columns:
            missing_method = df["method"].astype(str).str.strip().eq("").sum()
            if missing_method > 0:
                issues.append(f"{missing_method} deal(s) missing revenue recognition method")

        if "value_total" in df.columns:
            won = df[df["stage"] == "closed_won"].copy()
            if not won.empty:
                won_val = pd.to_numeric(won["value_total"], errors="coerce")
                bad = won_val.isna().sum() + int((won_val.fillna(0.0) <= 0.0).sum())
                if bad > 0:
                    issues.append(f"{bad} Closed_Won deal(s) missing/zero contract value")

        if (df["stage"] == "closed_lost").any():
            issues.append("Closed_Lost deals exist — ensure excluded from pipeline totals")

        return issues

    def _pipeline_ageing_risk_exists(df_pipe: pd.DataFrame, settings: dict) -> bool:
        if df_pipe is None or df_pipe.empty:
            return False

        max_proposal = int(settings.get("stage_max_days_proposal", 30) or 30)
        max_demo = int(settings.get("stage_max_days_demo", 45) or 45)
        max_contract = int(settings.get("stage_max_days_contract", 60) or 60)
        max_map = {"proposal": max_proposal, "demo": max_demo, "contract": max_contract}

        df = df_pipe.copy()
        df["stage"] = df.get("stage", "").astype(str).str.lower().fillna("")
        df = df[df["stage"].isin(["proposal", "demo", "contract"])].copy()
        if df.empty or "stage_entry_date" not in df.columns:
            return False

        df["stage_entry_date"] = pd.to_datetime(df["stage_entry_date"], errors="coerce")
        df = df[df["stage_entry_date"].notna()].copy()
        if df.empty:
            return False

        today = pd.Timestamp(datetime.now(timezone.utc).date())
        df["days_in_stage"] = (today - df["stage_entry_date"].dt.normalize()).dt.days.astype(int)
        df["max_days"] = df["stage"].map(max_map).fillna(0).astype(int)
        return bool((df["days_in_stage"] > df["max_days"]).any())

    def _pipeline_ageing_one_liner(df_pipe: pd.DataFrame, settings: dict, currency_symbol: str) -> str | None:
        if df_pipe is None or df_pipe.empty:
            return None

        max_proposal = int(settings.get("stage_max_days_proposal", 30) or 30)
        max_demo = int(settings.get("stage_max_days_demo", 45) or 45)
        max_contract = int(settings.get("stage_max_days_contract", 60) or 60)
        max_map = {"proposal": max_proposal, "demo": max_demo, "contract": max_contract}
        high_value_threshold = float(settings.get("pipeline_high_value_threshold", 100_000) or 100_000)

        active = df_pipe[df_pipe["stage"].isin(["proposal", "demo", "contract"])].copy()
        if active.empty:
            return None
        if "stage_entry_date" not in active.columns:
            return "⚠️ Pipeline ageing cannot be assessed (missing stage entry dates)."

        active["stage_entry_date"] = pd.to_datetime(active["stage_entry_date"], errors="coerce")
        if active["stage_entry_date"].isna().any():
            return "⚠️ Pipeline ageing data issue: missing stage entry date on one or more active deals."

        today = pd.Timestamp(datetime.now(timezone.utc).date())
        active["days_in_stage"] = (today - active["stage_entry_date"].dt.normalize()).dt.days.astype(int)
        active["max_days"] = active["stage"].map(max_map).fillna(0).astype(int)

        ageing = active[active["days_in_stage"] > active["max_days"]].copy()
        if ageing.empty:
            return None

        ageing["value_total"] = pd.to_numeric(ageing.get("value_total"), errors="coerce").fillna(0.0)
        late_stage_count = int((ageing["stage"].isin(["demo", "contract"])).sum())
        any_high_value = bool((ageing["value_total"] >= high_value_threshold).any())
        stalled_value = float(ageing["value_total"].sum())
        ageing_count = int(len(ageing))

        if late_stage_count >= 2:
            return f"🔴 {late_stage_count} late-stage deals ageing beyond expected stage duration"
        if any_high_value:
            return f"⚠️ High-value deal ageing beyond expected stage duration ({currency_symbol}{stalled_value:,.0f} impacted)"
        return f"ℹ️ {ageing_count} deal(s) ageing beyond expected stage duration ({currency_symbol}{stalled_value:,.0f})"

    def _rev_to_cash_one_liner(client_id, focus_month, settings: dict, secured_schedule_df: pd.DataFrame) -> str:
        focus_start = _safe_month_start(focus_month)
        prev_start = focus_start - pd.DateOffset(months=1)

        rev_this = 0.0
        rev_prev = 0.0
        if secured_schedule_df is not None and not secured_schedule_df.empty:
            sdf = secured_schedule_df.copy()
            sdf["month_date"] = pd.to_datetime(sdf["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            sdf = sdf[sdf["month_date"].notna()].copy()

            val_col = None
            for cand in ["recognised_revenue", "revenue_amount", "amount"]:
                if cand in sdf.columns:
                    val_col = cand
                    break

            if val_col is not None:
                rev_this = float(pd.to_numeric(sdf.loc[sdf["month_date"] == focus_start, val_col], errors="coerce").fillna(0.0).sum())
                rev_prev = float(pd.to_numeric(sdf.loc[sdf["month_date"] == prev_start, val_col], errors="coerce").fillna(0.0).sum())

        df_ar, _df_ap = fetch_ar_ap_for_client(client_id)
        if df_ar is None or df_ar.empty:
            ar_days = int(settings.get("ar_default_days", 30) or 30)
            return f"Revenue typically converts to cash in ~{ar_days} days (expected)."

        ar = df_ar.copy()
        for c in ["issued_date", "payment_date", "amount", "status"]:
            if c in ar.columns:
                if c in ["issued_date", "payment_date"]:
                    ar[c] = pd.to_datetime(ar[c], errors="coerce")
                if c == "amount":
                    ar[c] = pd.to_numeric(ar[c], errors="coerce").fillna(0.0)

        if "payment_date" not in ar.columns or "amount" not in ar.columns:
            ar_days = int(settings.get("ar_default_days", 30) or 30)
            return f"Revenue typically converts to cash in ~{ar_days} days (expected)."

        paid = ar[(ar.get("status", "").astype(str).str.lower() == "paid")].copy()
        paid = paid[paid["payment_date"].notna()].copy()

        cash_this = 0.0
        cash_prev = 0.0
        if not paid.empty:
            paid["pay_month"] = paid["payment_date"].dt.to_period("M").dt.to_timestamp()
            cash_this = float(paid.loc[paid["pay_month"] == focus_start, "amount"].sum())
            cash_prev = float(paid.loc[paid["pay_month"] == prev_start, "amount"].sum())

        if "issued_date" in paid.columns and paid["issued_date"].notna().any():
            paid2 = paid[paid["issued_date"].notna()].copy()
            if not paid2.empty:
                lookback_start = focus_start - pd.DateOffset(months=6)
                paid2 = paid2[(paid2["payment_date"] >= lookback_start) & (paid2["payment_date"] < (focus_start + pd.DateOffset(months=1)))].copy()

                if not paid2.empty:
                    paid2["lag_days"] = (paid2["payment_date"].dt.normalize() - paid2["issued_date"].dt.normalize()).dt.days
                    paid2 = paid2[pd.to_numeric(paid2["lag_days"], errors="coerce").notna()].copy()

                    if not paid2.empty:
                        avg_lag = float(pd.to_numeric(paid2["lag_days"], errors="coerce").fillna(0.0).mean())

                        mid = focus_start - pd.DateOffset(months=3)
                        last3 = paid2[(paid2["payment_date"] >= mid)].copy()
                        prev3 = paid2[(paid2["payment_date"] < mid)].copy()

                        last3_lag = float(pd.to_numeric(last3["lag_days"], errors="coerce").fillna(0.0).mean()) if not last3.empty else avg_lag
                        prev3_lag = float(pd.to_numeric(prev3["lag_days"], errors="coerce").fillna(0.0).mean()) if not prev3.empty else last3_lag

                        if rev_this > rev_prev and cash_this < cash_prev:
                            return "🔴 Revenue up but cash down — collection risk rising"

                        if last3_lag > prev3_lag + 7:
                            return "⚠️ Revenue-to-cash lag increasing — expect delayed cash impact"

                        return f"ℹ️ Revenue typically converts to cash in ~{int(round(avg_lag))} days"

        ar_days = int(settings.get("ar_default_days", 30) or 30)
        return f"Revenue typically converts to cash in ~{ar_days} days (expected)."

    def _build_revenue_risk_section(
        df_pipe: pd.DataFrame,
        settings: dict,
        client_id: str,
        focus_month,
        secured_schedule_df: pd.DataFrame,
        currency_symbol: str,
    ) -> dict:
        risks = []
        data_issues = _data_issues(df_pipe)

        if _pipeline_ageing_risk_exists(df_pipe, settings):
            risks.append({"code": "A", "message": "Pipeline ageing detected — expected conversion may be delayed."})

        weighted_now = _compute_weighted_pipeline(df_pipe)
        weighted_prev = _weighted_pipeline_last_month(df_pipe, focus_month)

        rec_this = _get_actual_recognised_for_month_from_schedule_df(secured_schedule_df, focus_month)
        prev_month = (pd.to_datetime(focus_month).to_period("M").to_timestamp() - pd.DateOffset(months=1)).date()
        rec_prev = _get_actual_recognised_for_month_from_schedule_df(secured_schedule_df, prev_month)

        if weighted_now > weighted_prev and rec_this <= rec_prev:
            risks.append({"code": "B", "message": "Pipeline growth not converting into recognised revenue."})

        cash_this, cash_prev = _cash_received_this_prev_month(client_id, focus_month)
        rev_to_cash_line = _rev_to_cash_one_liner(client_id, focus_month, settings, secured_schedule_df)
        lag_deteriorating = ("lag increasing" in rev_to_cash_line.lower()) or ("collection risk" in rev_to_cash_line.lower())

        if (rec_this > rec_prev and cash_this < cash_prev) or lag_deteriorating:
            risks.append({"code": "C", "message": "Revenue recognised, but cash conversion is slower than expected."})

        top2_share, top2_names = _concentration_top2_weighted(df_pipe)
        conc_threshold = float(settings.get("pipeline_concentration_threshold", 0.50) or 0.50)
        if top2_share >= conc_threshold and top2_share > 0:
            risks.append({"code": "D", "message": f"Revenue outlook dependent on a small number of deals (Top 2 ~{top2_share*100:,.0f}%: {top2_names})."})

        bullets = []
        if rec_this >= rec_prev:
            bullets.append("Recognised revenue is stable vs last month, but forward growth depends on late-stage conversion.")
        else:
            bullets.append("Recognised revenue is down vs last month, increasing reliance on pipeline conversion to recover.")

        if any(r["code"] == "A" for r in risks):
            bullets.append("Aging pipeline suggests near-term revenue may slip if deals do not progress.")

        if any(r["code"] == "C" for r in risks):
            bullets.append("Cash impact from recognised revenue is deteriorating, implying higher working-capital pressure.")
        else:
            bullets.append(rev_to_cash_line.replace("ℹ️ ", "").replace("⚠️ ", "").replace("🔴 ", ""))

        if any(r["code"] == "D" for r in risks):
            bullets.append("Revenue outcomes are concentrated; slippage in one major deal will materially move results.")

        actions = []

        if any(r["code"] == "A" for r in risks):
            actions += [
                "Push stalled deals to decision or exit (book a buyer next-step within 72 hours).",
                "Revalidate deal status with the buyer and update stage_entry_date if it changed.",
            ]
        if any(r["code"] == "B" for r in risks):
            actions += [
                "Focus sales effort on late-stage (contract/demo) deals; deprioritise early-stage outreach for 7 days.",
                "Reassess conversion assumptions (win rate + cycle time) and adjust forecast expectations.",
            ]
        if any(r["code"] == "C" for r in risks):
            actions += [
                "Review payment terms on recent wins; tighten terms on any new contracts signed this week.",
                "Accelerate invoicing milestones and escalate collections on overdue invoices.",
            ]
        if any(r["code"] == "D" for r in risks):
            actions += ["Create a contingency plan: identify 3 next-best deals to advance if a top deal slips."]

        if data_issues:
            actions += [
                "Fix missing deal metadata (close month, method, value) and enforce pipeline hygiene immediately.",
                "Add a simple rule: no stage move without stage_entry_date update (and reason if moving backwards).",
            ]

        seen = set()
        deduped = []
        for a in actions:
            if a not in seen:
                deduped.append(a)
                seen.add(a)

        if len(deduped) < 3:
            deduped += [
                "Confirm next 7-day sales target is tied to contract-stage progression (not new leads).",
                "Remove dead deals from pipeline to avoid false confidence in weighted totals.",
            ]
            seen = set()
            tmp = []
            for a in deduped:
                if a not in seen:
                    tmp.append(a)
                    seen.add(a)
            deduped = tmp

        return {
            "risks": risks,
            "data_issues": data_issues,
            "what_this_means": bullets[:4],
            "actions": deduped[:5],
        }

    def _render_pipeline_waterfall(df_pipe: pd.DataFrame, currency_symbol: str):
        """
        Waterfall chart (WEIGHTED). ✅ TABLE REMOVED.
        Stages: Idea -> Demo -> Proposal -> Contract -> Closed Won -> Total Estimated
        """
        import matplotlib.pyplot as plt

        if df_pipe is None or df_pipe.empty:
            st.caption("No pipeline data to display.")
            return

        st.subheader("🧩 Pipeline Waterfall (Weighted)")

        dfw = df_pipe.copy()
        dfw["stage"] = dfw.get("stage", "").astype(str).str.lower().fillna("")
        dfw["value_total"] = pd.to_numeric(dfw.get("value_total"), errors="coerce").fillna(0.0)
        dfw["probability_pct"] = pd.to_numeric(dfw.get("probability_pct"), errors="coerce").fillna(0.0)

        stage_order = ["idea", "demo", "proposal", "contract", "closed_won"]
        dfw = dfw[dfw["stage"].isin(stage_order)].copy()
        if dfw.empty:
            st.caption("No staged pipeline rows to display.")
            return

        def _prob_factor(row) -> float:
            stg = str(row.get("stage", "")).lower()
            if stg == "closed_won":
                return 1.0
            p = float(row.get("probability_pct", 0.0) or 0.0)
            p = max(0.0, min(100.0, p))
            return p / 100.0

        dfw["weighted_value"] = dfw["value_total"] * dfw.apply(_prob_factor, axis=1)

        stage_sum = (
            dfw.groupby("stage", as_index=False)[["weighted_value"]]
            .sum()
            .set_index("stage")
            .reindex(stage_order)
            .fillna(0.0)
            .reset_index()
        )
        stage_sum["Stage"] = stage_sum["stage"].str.replace("_", " ").str.title()

        labels = stage_sum["Stage"].tolist()
        values = stage_sum["weighted_value"].astype(float).tolist()
        total_estimated = float(sum(values))

        labels_total = labels + ["Total Estimated"]
        step_heights = values + [total_estimated]

        bases = []
        cum = 0.0
        for v in values:
            bases.append(cum)
            cum += float(v)
        bases_total = bases + [0.0]

        fig, ax = plt.subplots(figsize=(9, 4))
        x = list(range(len(labels_total)))
        ax.bar(x, step_heights, bottom=bases_total)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_total, rotation=0)
        ax.set_ylabel("Weighted pipeline value")
        ax.set_title("Pipeline Waterfall (Weighted)")

        for i, (b, h) in enumerate(zip(bases_total, step_heights)):
            ax.text(i, b + h, f"{currency_symbol}{h:,.0f}", ha="center", va="bottom", fontsize=9)

        ax.axhline(0, linewidth=1)
        ax.margins(x=0.02)
        st.pyplot(fig)

    # ---------------------------------------------------------------------
    # Save/Upsert helpers for deal editor (NO breaking changes)
    # ---------------------------------------------------------------------
    def _supabase_upsert_revenue_pipeline(row: dict) -> bool:
        """
        Fallback path (only used if your project doesn't already expose a save/upsert function).
        """
        try:
            # Prefer authed client (RLS safe)
            sb = None
            if "get_supabase_authed" in globals():
                sb = get_supabase_authed()
            if sb is None and "get_supabase_public" in globals():
                sb = get_supabase_public()
            if sb is None:
                raise RuntimeError("No Supabase client available (get_supabase_authed/public missing).")

            # Supabase-py v2 style
            sb.table("revenue_pipeline").upsert(row).execute()
            return True
        except Exception as e:
            st.error("Save failed (Supabase upsert).")
            st.caption(f"{type(e).__name__}: {e}")
            return False

    def _save_deal(row: dict) -> bool:
        """
        Uses your existing save function if available, else falls back to Supabase upsert.
        """
        # If you already have a preferred function in your codebase, it will be used automatically.
        for fn_name in [
            "save_pipeline_deal",
            "upsert_pipeline_deal",
            "save_deal_to_pipeline",
            "upsert_revenue_pipeline_row",
        ]:
            fn = globals().get(fn_name)
            if callable(fn):
                try:
                    out = fn(row)
                    return bool(out) if out is not None else True
                except Exception as e:
                    st.error(f"Save failed ({fn_name}).")
                    st.caption(f"{type(e).__name__}: {e}")
                    return False

        return _supabase_upsert_revenue_pipeline(row)

    def _delete_deal(deal_id: str) -> bool:
        """
        Optional delete. Uses existing delete function if present; else tries Supabase delete.
        """
        if not deal_id:
            return False

        for fn_name in [
            "delete_pipeline_deal",
            "delete_deal_from_pipeline",
            "delete_revenue_pipeline_row",
        ]:
            fn = globals().get(fn_name)
            if callable(fn):
                try:
                    out = fn(deal_id, selected_client_id) if fn.__code__.co_argcount >= 2 else fn(deal_id)
                    return bool(out) if out is not None else True
                except Exception as e:
                    st.error(f"Delete failed ({fn_name}).")
                    st.caption(f"{type(e).__name__}: {e}")
                    return False

        try:
            sb = get_supabase_authed() if "get_supabase_authed" in globals() else None
            if sb is None and "get_supabase_public" in globals():
                sb = get_supabase_public()
            if sb is None:
                raise RuntimeError("No Supabase client available (get_supabase_authed/public missing).")
            sb.table("revenue_pipeline").delete().eq("id", deal_id).eq("client_id", selected_client_id).execute()
            return True
        except Exception as e:
            st.error("Delete failed (Supabase delete).")
            st.caption(f"{type(e).__name__}: {e}")
            return False

    # ---------------------------------------------------------------------
    # Layout: LEFT (overview) / RIGHT (deal editor)
    # ---------------------------------------------------------------------
    col_main, col_side = st.columns([2, 1])

    # ============================================================
    # RIGHT: Create / update deal (FULLY IMPLEMENTED)
    # ============================================================
    with col_side:
        st.subheader("✍️ Create / update a deal")

        # Choose deal to edit (optional)
        deal_id_col = "id" if "id" in df_pipeline.columns else None
        deal_choices = []
        deal_map = {}

        if not df_pipeline.empty and deal_id_col:
            tmp = df_pipeline.copy()
            tmp["deal_name"] = tmp.get("deal_name", "").astype(str).fillna("")
            tmp["customer_name"] = tmp.get("customer_name", "").astype(str).fillna("")
            tmp["label"] = tmp.apply(
                lambda r: f"{r.get('deal_name','').strip() or 'Unnamed'}  —  {r.get('customer_name','').strip() or 'Unknown'}",
                axis=1,
            )
            tmp = tmp.sort_values(["label"])
            deal_choices = ["➕ New deal"] + tmp["label"].tolist()
            deal_map = {row["label"]: row for _, row in tmp.iterrows()}
        else:
            deal_choices = ["➕ New deal"]

        selected_label = st.selectbox("Select deal to edit", deal_choices, index=0)
        editing_existing = selected_label != "➕ New deal"
        row0 = deal_map.get(selected_label) if editing_existing else {}

        # Defaults
        def _d(key, fallback=""):
            return row0.get(key, fallback) if isinstance(row0, dict) else fallback

        stage_options = ["idea", "demo", "proposal", "contract", "closed_won", "closed_lost"]
        default_stage = str(_d("stage", "idea") or "idea").lower()
        if default_stage not in stage_options:
            default_stage = "idea"

        method_options_display = list(method_display_map.values())
        default_method_display = str(_d("method", "") or "").strip()
        if default_method_display not in method_options_display:
            default_method_display = current_method_display

        # Form
        with st.form("deal_form", clear_on_submit=False):
            deal_name = st.text_input("Deal name", value=str(_d("deal_name", "") or ""))
            customer_name = st.text_input("Customer", value=str(_d("customer_name", "") or ""))

            c1, c2 = st.columns(2)
            with c1:
                stage = st.selectbox("Stage", stage_options, index=stage_options.index(default_stage))
            with c2:
                probability_pct = st.number_input(
                    "Win chance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(pd.to_numeric(_d("probability_pct", 0.0), errors="coerce") or 0.0),
                    step=5.0,
                )

            value_total = st.number_input(
                "Contract value (total)",
                min_value=0.0,
                value=float(pd.to_numeric(_d("value_total", 0.0), errors="coerce") or 0.0),
                step=1000.0,
            )

            method_display = st.selectbox(
                "Revenue recognition (deal override)",
                options=method_options_display,
                index=method_options_display.index(default_method_display),
                help="This overrides the business default above for this deal.",
            )

            c3, c4 = st.columns(2)
            with c3:
                start_month_in = st.text_input(
                    "Expected close / start month (e.g., 2025-12-01 or Dec 2025)",
                    value=str(_d("start_month", "") or ""),
                )
            with c4:
                end_month_in = st.text_input(
                    "End month (optional; for service term)",
                    value=str(_d("end_month", "") or ""),
                )

            contract_months_default = pd.to_numeric(_d("contract_months", 0), errors="coerce")
            contract_months_default = int(contract_months_default) if pd.notna(contract_months_default) else 0
            contract_months = st.number_input(
                "Contract months (optional)",
                min_value=0,
                value=int(contract_months_default),
                step=1,
            )

            stage_entry_default = _d("stage_entry_date", None)
            stage_entry_default = pd.to_datetime(stage_entry_default, errors="coerce")
            stage_entry_date = st.date_input(
                "Stage entry date",
                value=stage_entry_default.date() if pd.notna(stage_entry_default) else datetime.now(timezone.utc).date(),
                help="Used for ageing rules. Keep updated when stage changes.",
            )

            notes = st.text_area("Notes (optional)", value=str(_d("notes", "") or ""), height=120)

            submitted = st.form_submit_button("Save deal")

        # Save action (outside form submit button)
        if submitted:
            # Minimal validation
            if not str(deal_name).strip():
                st.error("Deal name is required.")
            else:
                start_ts = _parse_month_input(start_month_in)
                end_ts = _parse_month_input(end_month_in)

                payload = {
                    "client_id": selected_client_id,
                    "deal_name": str(deal_name).strip(),
                    "customer_name": str(customer_name).strip() if customer_name is not None else "",
                    "stage": str(stage).lower().strip(),
                    "probability_pct": float(probability_pct or 0.0),
                    "value_total": float(value_total or 0.0),
                    "method": str(method_display).strip(),
                    "contract_months": int(contract_months or 0),
                    "stage_entry_date": pd.to_datetime(stage_entry_date).strftime("%Y-%m-%d") if stage_entry_date else None,
                    "notes": str(notes).strip() if notes is not None else "",
                }

                # Keep your existing column names start_month/end_month if used
                payload["start_month"] = start_ts.strftime("%Y-%m-%d") if start_ts is not None else None
                payload["end_month"] = end_ts.strftime("%Y-%m-%d") if end_ts is not None else None

                # If editing existing, preserve id so upsert updates
                if editing_existing and deal_id_col and isinstance(row0, dict) and row0.get(deal_id_col):
                    payload["id"] = row0.get(deal_id_col)

                ok = _save_deal(payload)
                if ok:
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    st.success("Deal saved.")
                    st.rerun()

        # Optional delete (only if editing existing + has id)
        if editing_existing and deal_id_col and isinstance(row0, dict) and row0.get(deal_id_col):
            st.markdown("---")
            if st.button("🗑️ Delete deal", type="secondary"):
                did = str(row0.get(deal_id_col))
                if _delete_deal(did):
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    st.success("Deal deleted.")
                    st.rerun()

    # ============================================================
    # LEFT: Pipeline overview + KPIs + Health panel + Data checks
    # ============================================================
    with col_main:
        st.subheader("📂 Pipeline overview")

        if df_pipeline.empty:
            st.info("No deals in the pipeline yet for this business.")
        else:
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Total pipeline", f"{currency_symbol}{total_pipeline:,.0f}")
            with k2:
                st.metric("Open deals", open_deals_count)

            # Build secured schedule ONCE
            snap_df = build_revenue_schedule_for_client(
                selected_client_id,
                base_month=selected_month_start,
                n_months=12,
                mode="secured",
            )

            rec_this = _get_actual_recognised_for_month_from_schedule_df(snap_df, selected_month_start)
            rec_next3 = _get_next3_total_from_schedule_df(snap_df, selected_month_start)

            rk1, rk2 = st.columns(2)
            with rk1:
                st.metric("Recognised revenue (this month)", f"{currency_symbol}{rec_this:,.0f}")
            with rk2:
                st.metric("Recognised revenue (next 3 months)", f"{currency_symbol}{rec_next3:,.0f}")

            plan_rev = _get_plan_for_month(selected_client_id, selected_month_start)
            rev_vs_plan_pct = (rec_this - plan_rev) / plan_rev if plan_rev else None
            avg_deal = _avg_deal_size_active(df_pipeline)

            kk1, kk2 = st.columns(2)
            with kk1:
                if plan_rev <= 0:
                    st.metric("Revenue vs Plan", "Plan not set")
                else:
                    st.metric("Revenue vs Plan", f"{rev_vs_plan_pct*100:,.1f}%")
            with kk2:
                st.metric("Average deal size (Active only)", f"{currency_symbol}{avg_deal:,.0f}")

            ageing_line = _pipeline_ageing_one_liner(df_pipeline, settings, currency_symbol)
            if ageing_line:
                st.caption(ageing_line)

            cash_line = _rev_to_cash_one_liner(selected_client_id, selected_month_start, settings, snap_df)
            st.caption(cash_line)

            horizon_start = pd.to_datetime(selected_month_start).replace(day=1)
            horizon_end = horizon_start + pd.DateOffset(months=12)
            st.caption(
                f"Views below use deals closing between "
                f"{horizon_start.strftime('%b %Y')} and "
                f"{(horizon_end - pd.DateOffset(days=1)).strftime('%b %Y')} "
                "(next 12 months window)."
            )

            # Health panel (external; unchanged)
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

            # Data checks warning (external; unchanged)
            issues = build_pipeline_data_checks(df_pipeline)
            if issues:
                st.warning("**Data checks**\n" + "\n".join([f"• {i}" for i in issues]))

            # View full pipeline list expander (kept)
            display_cols = [
                c
                for c in [
                    "deal_name",
                    "customer_name",
                    "stage",
                    "value_total",
                    "probability_pct",
                    "start_month",
                    "end_month",
                    "method",
                    "contract_months",
                    "stage_entry_date",
                ]
                if c in df_pipeline.columns
            ]
            if display_cols:
                view = df_pipeline[display_cols].copy().rename(
                    columns={
                        "deal_name": "Deal",
                        "customer_name": "Customer",
                        "stage": "Stage",
                        "value_total": "Value",
                        "probability_pct": "Win chance (%)",
                        "start_month": "Start month",
                        "end_month": "End month",
                        "method": "Revenue type",
                        "contract_months": "Contract months",
                        "stage_entry_date": "Stage entry date",
                    }
                )
                with st.expander("View full pipeline list", expanded=False):
                    st.dataframe(view, width="stretch")

    # ============================================================
    # Waterfall chart (TABLE REMOVED)
    # ============================================================
    _render_pipeline_waterfall(df_pipeline, currency_symbol)

    # ============================================================
    # Revenue / Risk / What this means / Action plan (NO tables)
    # ============================================================
    st.markdown("---")
    st.subheader("🚦 Revenue, Risk & data issues")

    try:
        snap_df
    except NameError:
        snap_df = build_revenue_schedule_for_client(
            selected_client_id,
            base_month=selected_month_start,
            n_months=12,
            mode="secured",
        )

    bundle = _build_revenue_risk_section(
        df_pipe=df_pipeline,
        settings=settings,
        client_id=selected_client_id,
        focus_month=selected_month_start,
        secured_schedule_df=snap_df,
        currency_symbol=currency_symbol,
    )

    if not bundle["risks"]:
        st.success("No revenue risk triggers fired for this month based on current data.")
    else:
        for r in bundle["risks"]:
            st.error(f"🔴 Risk {r['code']} — {r['message']}")

    if bundle["data_issues"]:
        st.warning("⚠️ Data issues detected (hygiene):")
        for di in bundle["data_issues"]:
            st.write(f"• {di}")
    else:
        st.caption("No data hygiene issues detected based on available fields.")

    st.markdown("#### What this means")
    for b in bundle["what_this_means"][:4]:
        st.write(f"• {b}")

    st.markdown("#### Action plan (next 7 days)")
    for i, a in enumerate(bundle["actions"][:5], start=1):
        st.write(f"{i}. {a}")


def page_team_spending():
    top_header("Team & Spending")

    if not selected_client_id:
        st.info("Select a business from the sidebar to view this page.")
        return

    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Loading team and spending data..."):
        df_positions = fetch_payroll_positions_for_client(selected_client_id)
        df_ar_ap = fetch_ar_ap_for_client(selected_client_id)
        kpis = fetch_kpis_for_client_month(selected_client_id, selected_month_start)

    if isinstance(df_ar_ap, tuple):
        df_ar, df_ap = df_ar_ap
    else:
        df_ar, df_ap = None, None

    # ------------------------------------------------------------------
    # Month setup
    # ------------------------------------------------------------------
    month_ts = pd.to_datetime(selected_month_start).to_period("M").to_timestamp()

    # ------------------------------------------------------------------
    # Payroll cash engine (source of truth)
    # ------------------------------------------------------------------
    month_index_1 = pd.date_range(start=month_ts, periods=1, freq="MS")
    payroll_by_month = compute_payroll_by_month(selected_client_id, month_index_1)
    month_payroll_cash_engine = float(payroll_by_month.get(month_ts, 0.0))

    # ------------------------------------------------------------------
    # Payroll components + economic payroll cost
    # ------------------------------------------------------------------
    comp_df = compute_payroll_cash_components_by_month(selected_client_id, month_index_1)
    comp = comp_df.iloc[0].to_dict() if comp_df is not None and not comp_df.empty else {}

    headcount_fte_eom = float(comp.get("headcount_fte_eom", 0.0) or 0.0)

    # Cash KPIs
    salaries_paid_cash = float(comp.get("net_salary_cash", 0.0) or 0.0)
    super_paid_cash = float(comp.get("super_cash", 0.0) or 0.0)
    tax_paid_cash = float(comp.get("payg_cash", 0.0) or 0.0)
    total_payroll_cash_components = float(comp.get("total_payroll_cash", 0.0) or 0.0)

    # Economic payroll cost (work month)
    monthly_payroll_cost = float(comp.get("monthly_payroll_cost", 0.0) or 0.0)
    avg_cost_per_fte = (monthly_payroll_cost / headcount_fte_eom) if headcount_fte_eom else 0.0

    # Employer on-costs
    workers_comp_generated = float(comp.get("workers_comp_generated", 0.0) or 0.0)
    payroll_tax_generated = float(comp.get("payroll_tax_generated", 0.0) or 0.0)
    annualised_wages_eom = float(comp.get("annualised_wages_eom", 0.0) or 0.0)
    payroll_tax_applicable = bool(comp.get("payroll_tax_applicable", False))

    # Tie-out
    tie_diff = float(total_payroll_cash_components - month_payroll_cash_engine)

    # ------------------------------------------------------------------
    # Headcount split for the month (focus month)
    # ------------------------------------------------------------------
    hc = compute_headcount_month_split(df_positions, month_ts)
    hires_fte_in_month = float(hc.get("hires_fte_in_month", 0.0))
    exits_fte_in_month = float(hc.get("exits_fte_in_month", 0.0))
    net_change_in_month = float(hc.get("net_change_fte_in_month", 0.0))

    # ------------------------------------------------------------------
    # Headcount trend (past 6 months)  ✅ NEW CHART
    # ------------------------------------------------------------------
    hc_trend_df = None
    try:
        # Past 6 months incl current focus month
        month_index_6_past = pd.date_range(end=month_ts, periods=6, freq="MS")
        comp_6_df = compute_payroll_cash_components_by_month(selected_client_id, month_index_6_past)

        if comp_6_df is not None and not comp_6_df.empty:
            hc_trend_df = comp_6_df[["month_date", "headcount_fte_eom"]].copy()
            hc_trend_df["month_date"] = pd.to_datetime(hc_trend_df["month_date"], errors="coerce")
            hc_trend_df["headcount_fte_eom"] = pd.to_numeric(
                hc_trend_df["headcount_fte_eom"], errors="coerce"
            ).fillna(0.0)
            hc_trend_df = hc_trend_df.sort_values("month_date")
    except Exception:
        hc_trend_df = None

    # ------------------------------------------------------------------
    # Payroll target (client_monthly_targets)
    # ------------------------------------------------------------------
    targets = get_monthly_targets_for_month(selected_client_id, month_ts)
    payroll_target = float(targets.get("payroll_target", 0.0) or 0.0)
    payroll_var = monthly_payroll_cost - payroll_target
    payroll_var_pct = (payroll_var / payroll_target * 100.0) if payroll_target else None

    # ------------------------------------------------------------------
    # Revenue (from KPIs)
    # ------------------------------------------------------------------
    month_revenue = 0.0
    if kpis:
        try:
            month_revenue = float(kpis.get("revenue") or 0.0)
        except Exception:
            month_revenue = 0.0

    # ------------------------------------------------------------------
    # 3 new KPIs: burn + next hire + sensitivity (source of truth)
    # ------------------------------------------------------------------
    hire_kpis = compute_next_hire_and_burn_kpis(
        selected_client_id,
        month_ts=month_ts,
        monthly_payroll_cost=monthly_payroll_cost,
    )
    effective_burn_monthly = float(hire_kpis.get("effective_burn_monthly", 0.0) or 0.0)
    payroll_pct_of_burn = hire_kpis.get("payroll_pct_of_burn", None)
    payroll_burn_band = hire_kpis.get("payroll_burn_band", None)
    next_hire_monthly_cost = float(hire_kpis.get("next_hire_monthly_cost", 0.0) or 0.0)
    next_hire_annual_cost = float(hire_kpis.get("next_hire_annual_cost", 0.0) or 0.0)
    runway_impact_months = hire_kpis.get("runway_impact_months", None)

    # ------------------------------------------------------------------
    # KPI HEADER (row-style sections)
    # ------------------------------------------------------------------
    st.subheader("👥 Team & Payroll KPIs (focus month)")

    # -------------------- Headcount (ROW) --------------------
    st.markdown("### 👥 Headcount")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
    with c1:
        st.metric("Total HC (FTE)", f"{headcount_fte_eom:,.1f}")
    with c2:
        st.metric("New hires", f"+{hires_fte_in_month:,.1f}")
    with c3:
        st.metric("Exits", f"-{exits_fte_in_month:,.1f}")
    with c4:
        st.metric("Net change", f"{net_change_in_month:+.1f} FTE")

    # ✅ NEW: headcount trend line (past 6 months) under headcount KPIs
    if hc_trend_df is not None and not hc_trend_df.empty:
        st.markdown("#### Headcount trend (past 6 months)")
        try:
            import altair as alt

            hc_chart = (
                alt.Chart(hc_trend_df)
                .mark_line()
                .encode(
                    x=alt.X("month_date:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("headcount_fte_eom:Q", title="FTE (EOM)"),
                    tooltip=[
                        alt.Tooltip("month_date:T", title="Month", format="%b %Y"),
                        alt.Tooltip("headcount_fte_eom:Q", title="Headcount (FTE)", format=",.1f"),
                    ],
                )
            )
            st.altair_chart(hc_chart, width="stretch")
        except Exception:
            # fallback to Streamlit line_chart if Altair fails
            tmp = hc_trend_df.set_index("month_date")[["headcount_fte_eom"]]
            st.line_chart(tmp)
    else:
        st.caption("Headcount trend unavailable (no payroll roles / insufficient history).")

    st.markdown("---")

    # -------------------- Payroll (work month) (ROW) --------------------
    st.markdown("### 💰 Payroll (work month)")
    p1, p2, p3, p4 = st.columns([1.3, 1.0, 1.1, 1.4])
    with p1:
        st.metric("Monthly payroll cost", f"{currency_symbol}{monthly_payroll_cost:,.0f}")
    with p2:
        st.metric("Avg cost per FTE", f"{currency_symbol}{avg_cost_per_fte:,.0f}")
    with p3:
        st.metric("Workers comp (gen.)", f"{currency_symbol}{workers_comp_generated:,.0f}")
    with p4:
        st.metric("Payroll tax (gen.)", f"{currency_symbol}{payroll_tax_generated:,.0f}")

    st.caption(
        f"Payroll tax applicable: **{'Yes' if payroll_tax_applicable else 'No'}** · "
        f"Annualised wages (EOM): **{currency_symbol}{annualised_wages_eom:,.0f}**"
    )

    st.markdown("#### 🎯 Payroll vs target")
    t1, t2, t3 = st.columns([1.2, 1.2, 1.6])
    with t1:
        if payroll_target:
            st.metric("Payroll target", f"{currency_symbol}{payroll_target:,.0f}")
        else:
            st.info("No payroll target set for this month in client_monthly_targets.")
    with t2:
        if payroll_target:
            delta_txt = f"{currency_symbol}{payroll_var:,.0f}"
            if payroll_var_pct is not None:
                delta_txt += f" ({payroll_var_pct:+.0f}%)"
            st.metric("Variance vs target", delta_txt)
        else:
            st.metric("Variance vs target", "—")
    with t3:
        st.caption("Target is compared to **work-month payroll cost** (economic), not cash timing.")

    st.markdown("#### 🧪 Hiring sensitivity & burn mix")
    h1, h2, h3, h4 = st.columns([1.2, 1.6, 1.2, 1.2])
    with h1:
        if effective_burn_monthly > 0:
            st.metric("Effective burn (monthly)", f"{currency_symbol}{effective_burn_monthly:,.0f}")
        else:
            st.info("Effective burn not available.")
    with h2:
        st.metric(
            "Fully loaded cost of next hire",
            f"{currency_symbol}{next_hire_monthly_cost:,.0f} / month",
            help="Incremental cost: salary + super + workers comp + payroll tax (if applicable) + benefits + onboarding (if recurring).",
        )
        st.caption(f"Annualised: **{currency_symbol}{next_hire_annual_cost:,.0f} / year**")
    with h3:
        st.metric(
            "Runway sensitivity (1 hire)",
            f"-{float(runway_impact_months):.2f} months" if runway_impact_months is not None else "—",
        )
    with h4:
        st.metric(
            "Payroll % of burn",
            f"{float(payroll_pct_of_burn):.0f}%" if payroll_pct_of_burn is not None else "—",
        )

    if payroll_pct_of_burn is not None:
        if payroll_burn_band == "High risk":
            st.error("Payroll % of burn: **High risk** (>65%)")
        elif payroll_burn_band == "Watch":
            st.warning("Payroll % of burn: **Watch** (50–65%)")
        else:
            st.success("Payroll % of burn: **Healthy** (<50%)")

    st.markdown("---")

    # -------------------- Payroll (cash paid) (ROW) --------------------
    st.markdown("### 💳 Payroll (cash paid)")
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with cc1:
        st.metric("Base salaries (net)", f"{currency_symbol}{salaries_paid_cash:,.0f}")
    with cc2:
        st.metric("Super paid (lagged)", f"{currency_symbol}{super_paid_cash:,.0f}")
    with cc3:
        st.metric("PAYG paid (lagged)", f"{currency_symbol}{tax_paid_cash:,.0f}")
    with cc4:
        st.metric(
            "Total payroll cash",
            f"{currency_symbol}{month_payroll_cash_engine:,.0f}",
            delta=f"{currency_symbol}{tie_diff:,.0f}",
        )

    if abs(tie_diff) > 0.01:
        st.warning(
            f"Payroll engine tie-out mismatch: components={currency_symbol}{total_payroll_cash_components:,.2f} "
            f"vs engine={currency_symbol}{month_payroll_cash_engine:,.2f} "
            f"(diff={currency_symbol}{tie_diff:,.2f})."
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # ROW-STYLE PANELS (no 3 columns)
    # ------------------------------------------------------------------
    panel = compute_team_risk_panels(
        selected_client_id,
        month_ts,
        df_positions=df_positions,
        monthly_payroll_cost=monthly_payroll_cost,
        payroll_pct_of_burn=payroll_pct_of_burn,
        runway_impact_months=runway_impact_months,
        next_hire_monthly_cost=next_hire_monthly_cost,
        month_revenue=month_revenue,
    )

    st.markdown("## 🧾 Payroll risk & compliance")
    for title, msg in panel.get("risks", []):
        if title.startswith("🔴"):
            st.error(f"**{title}**\n\n{msg}")
        elif title.startswith("🟢"):
            st.success(f"**{title}**\n\n{msg}")
        else:
            st.info(f"**{title}**\n\n{msg}")

    st.markdown("## 📅 Upcoming payroll obligations (next month)")
    for ob in panel.get("obligations", [])[:3]:
        st.markdown(f"- {ob}")

    if df_positions is not None and not df_positions.empty:
        if ("worker_type" not in df_positions.columns) and ("employment_type" not in df_positions.columns):
            st.caption("Silent hiring detection note: add `worker_type` or `employment_type` to classify contractors.")

    st.markdown("---")

    st.markdown("## 🧠 What this means")
    for b in panel.get("what_this_means", [])[:5]:
        st.markdown(f"- {b}")

    st.markdown("---")

    st.markdown("## ✅ Next 7 days action plan")
    for a in panel.get("actions_7d", [])[:2]:
        st.markdown(f"- {a}")

    # Manage roles section intentionally removed.

