


import streamlit as st
from ui.boot import boot_screen_start

def run_app():
    render_boot, boot_close = boot_screen_start("Starting Matfina…")

    # Step 0: Initialising workspace
    render_boot(0, "Warming caches and preparing UI…")

    # ✅ Step 1: Loading client profile
    render_boot(1, "Loading client list…")
    clients_data = load_clients()  # ideally cached
    client_names = [c["name"] for c in clients_data]

    selected_client_name = st.sidebar.selectbox("Select business", client_names)
    selected_client_id = next((c["id"] for c in clients_data if c["name"] == selected_client_name), None)

    # ✅ Step 2: Fetching cashflow summary (or other key data)
    render_boot(2, "Fetching engine + KPIs…")
    month_options = get_client_focus_month_options(selected_client_id)
    selected_month_label = st.sidebar.selectbox("Focus month", month_options, index=len(month_options)-1)
    selected_month_start = parse_month_label_to_date(selected_month_label)

    # This is the slow bit for many apps — do it here so boot screen is meaningful
    as_of_month = get_active_as_of_month(selected_client_id, selected_month_start)
    engine_df = fetch_cashflow_summary_for_client_as_of(selected_client_id, as_of_month)

    # ✅ Step 3: Preparing charts (or any precomputation)
    render_boot(3, "Preparing charts and summaries…")
    # Example: precompute expensive transformations once
    # chart_payload = build_chart_payload(engine_df)

    # Done -> show app
    boot_close()

    # Now render navigation + pages (normal app)
    page_names = [
        "Business at a Glance",
        "Investing & Financing Strategy",
        "Sales & Deals",
        "Team & Spending",
        "Cash & Bills",
        "Collaboration Hub",
        "Configuration",
        "Client Settings",
    ]
    page_choice = st.sidebar.radio("Go to page", page_names)

    if page_choice == "Business at a Glance":
        page_business_overview()
    elif page_choice == "Funding Strategy":
        page_investing_financing()
    elif page_choice == "Sales & Deals":
        page_sales_deals()
    elif page_choice == "Team & Spending":
        page_team_spending()
    elif page_choice == "Cash & Bills":
        page_cash_bills()    
    elif page_choice == "Collaboration Hub":
        page_alerts_todos()    
    elif page_choice == "Configuration":
        page_configuration()
    elif page_choice == "Client Settings":
        page_client_settings()



if __name__ == "__main__":
    run_app()



def run_app():


    page_names = [
        "Business at a Glance",
        "Funding Strategy",
        "Sales & Deals",
        "Team & Spending",
        "Cash & Bills",
        "Collaboration Hub",
        "Configuration",
        "Client Settings",
    ]

    page_choice = st.sidebar.radio("Go to page", page_names)

    if page_choice == "Business at a Glance":
        page_business_overview()
    elif page_choice == "Funding Strategy":
        page_investing_financing()
    elif page_choice == "Sales & Deals":
        page_sales_deals()
    elif page_choice == "Team & Spending":
        page_team_spending()
    elif page_choice == "Cash & Bills":
        page_cash_bills()    
    elif page_choice == "Collaboration Hub":
        page_alerts_todos()    
    elif page_choice == "Configuration":
        page_configuration()
    elif page_choice == "Client Settings":
        page_client_settings()





def run_app():
    render_boot, boot_close = boot_screen_start("Starting Matfina…")

    render_boot(0, "Initialising workspace…")

    render_boot(1, "Loading client profile…")
    selected_client_id, selected_month_start = build_sidebar_state()

    render_boot(2, "Fetching cashflow summary…")
    # your engine fetch here...

    render_boot(3, "Preparing charts…")
    # precompute if needed...

    boot_close()

    # navigation radio (also needs a key)
    page_choice = st.sidebar.radio(
        "Go to page",
        [
            "Business at a Glance",
            "Investing & Financing Strategy",
            "Sales & Deals",
            "Team & Spending",
            "Cash & Bills",
            "Collaboration Hub",
            "Configuration",
            "Client Settings",
        ],
        key="sb_page_nav",
    )

    if page_choice == "Business at a Glance":
        page_business_overview()
    elif page_choice == "Funding Strategy":
        page_investing_financing()
    elif page_choice == "Sales & Deals":
        page_sales_deals()
    elif page_choice == "Team & Spending":
        page_team_spending()
    elif page_choice == "Cash & Bills":
        page_cash_bills()    
    elif page_choice == "Collaboration Hub":
        page_alerts_todos()    
    elif page_choice == "Configuration":
        page_configuration()
    elif page_choice == "Client Settings":
        page_client_settings()



def page_configuration():
    """
    Configuration page (NEW)

    Contains:
    1) Revenue recognition business default + Save (+ optional apply to all deals)
    2) Create / update deal editor (moved from Sales & Deals)

    Does NOT change any external function signatures.
    Reuses your existing save/bulk-update helpers.
    """

    top_header("Configuration")

    if not selected_client_id:
        st.info("Select a client from the sidebar to view this page.")
        return

    settings = get_client_settings(selected_client_id) or {}
    currency_code, currency_symbol = get_client_currency(selected_client_id)

    # ============================================================
    # 1) Revenue recognition method (business default)
    # ============================================================
    st.subheader("🧾 Revenue recognition settings")

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
        "Business default method",
        options=[method_labels[m] for m in method_options],
        index=default_index,
        help=(
            "Default method for this business. Deals can override this individually. "
            "The engine uses deal.method first, then falls back to this business default."
        ),
        key="cfg_rev_method_select",
    )

    reverse_map = {v: k for k, v in method_labels.items()}
    selected_method_code = reverse_map[selected_method_label]

    apply_to_all = st.checkbox(
        "Apply this method to ALL existing deals (overwrite per-deal overrides)",
        value=False,
        help="Use this if you want business default to drive the numbers across all deals.",
        key="cfg_apply_to_all",
    )

    cA, cB = st.columns([1, 3])
    with cA:
        if st.button("Save method", key="cfg_save_method"):
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
                st.success("Revenue recognition method updated.")
                st.rerun()
            else:
                st.error("Could not update method. Please try again.")

    st.markdown("---")

    # ============================================================
    # 2) Create / update deal editor (moved here)
    # ============================================================
    st.subheader("✍️ Create / update a deal")

    # Load pipeline for editing list
    try:
        df_pipeline = fetch_pipeline_for_client(selected_client_id)
    except Exception as e:
        st.error("Could not load pipeline from Supabase (revenue_pipeline table).")
        st.caption(f"{type(e).__name__}: {e}")
        df_pipeline = pd.DataFrame()

    if df_pipeline is None:
        df_pipeline = pd.DataFrame()

    # ---- Helpers (same ones you had) ----
    def _parse_month_input(maybe_date) -> pd.Timestamp | None:
        if maybe_date is None or str(maybe_date).strip() == "":
            return None
        ts = pd.to_datetime(maybe_date, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_period("M").to_timestamp()

    def _supabase_upsert_revenue_pipeline(row: dict) -> bool:
        try:
            sb = supabase  # ✅ uses your global
            sb.table("revenue_pipeline").upsert(row).execute()
            return True
        except Exception as e:
            st.error("Save failed (Supabase upsert).")
            st.caption(f"{type(e).__name__}: {e}")
            return False

    def _save_deal(row: dict) -> bool:
        try:
            supabase.table("revenue_pipeline").upsert(row).execute()
            return True
        except Exception as e:
            st.error("Save failed (Supabase upsert).")
            st.caption(f"{type(e).__name__}: {e}")
            return False


    def _delete_deal(deal_id: str) -> bool:
        if not deal_id:
            return False

        # If you truly have no delete helper, skip searching globals and delete directly
        try:
            supabase.table("revenue_pipeline") \
                .delete() \
                .eq("id", deal_id) \
                .eq("client_id", selected_client_id) \
                .execute()
            return True
        except Exception as e:
            st.error("Delete failed (Supabase delete).")
            st.caption(f"{type(e).__name__}: {e}")
            return False


    # ---- Deal selector ----
    deal_id_col = "id" if ("id" in df_pipeline.columns) else None
    deal_choices = ["➕ New deal"]
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

    selected_label = st.selectbox("Select deal", deal_choices, index=0, key="cfg_deal_select")
    editing_existing = selected_label != "➕ New deal"
    row0 = deal_map.get(selected_label) if editing_existing else {}

    def _d(key, fallback=""):
        return row0.get(key, fallback) if isinstance(row0, dict) else fallback

    stage_options = ["idea", "demo", "proposal", "contract", "closed_won", "closed_lost"]
    default_stage = str(_d("stage", "idea") or "idea").lower()
    if default_stage not in stage_options:
        default_stage = "idea"

    # Use same display map as Sales page
    current_method_display = method_display_map.get(selected_method_code, "SaaS subscription")
    method_options_display = list(method_display_map.values())
    default_method_display = str(_d("method", "") or "").strip()
    if default_method_display not in method_options_display:
        default_method_display = current_method_display

    with st.form("cfg_deal_form", clear_on_submit=False):
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
            f"Contract value (total) ({currency_code})",
            min_value=0.0,
            value=float(pd.to_numeric(_d("value_total", 0.0), errors="coerce") or 0.0),
            step=1000.0,
        )

        method_display = st.selectbox(
            "Revenue recognition (deal override)",
            options=method_options_display,
            index=method_options_display.index(default_method_display),
            help="Overrides the business default above for this deal.",
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

    if submitted:
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
                "start_month": start_ts.strftime("%Y-%m-%d") if start_ts is not None else None,
                "end_month": end_ts.strftime("%Y-%m-%d") if end_ts is not None else None,
            }

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

    if editing_existing and deal_id_col and isinstance(row0, dict) and row0.get(deal_id_col):
        st.markdown("---")
        if st.button("🗑️ Delete deal", type="secondary", key="cfg_delete_deal"):
            did = str(row0.get(deal_id_col))
            if _delete_deal(did):
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.success("Deal deleted.")
                st.rerun()

        # ============================================================
    # 1B) AR/AP: Record payments (moved from Cash & Bills)
    # ============================================================
    st.markdown("---")
    st.subheader("💳 AR/AP — record payments")

    # Load AR/AP (normalised) so the payment widgets have clean data
    try:
        df_ar_cfg, df_ap_cfg = fetch_ar_ap_for_client(selected_client_id)
    except Exception as e:
        st.error(f"Failed to load AR/AP: {e}")
        df_ar_cfg, df_ap_cfg = pd.DataFrame(), pd.DataFrame()

    # Normalise (safe)
    try:
        df_ar_cfg = normalise_ar_ap_for_cash(df_ar_cfg)
    except Exception:
        pass
    try:
        df_ap_cfg = normalise_ar_ap_for_cash(df_ap_cfg)
    except Exception:
        pass

    # Use Focus Month month-end as default payment date
    focus_start_cfg = selected_month_start
    focus_end_cfg = (pd.to_datetime(focus_start_cfg) + MonthEnd(0)).date()

    currency_code_cfg, currency_symbol_cfg = get_client_currency(selected_client_id)

    # ----------------------------
    # AR Payment
    # ----------------------------
    with st.expander("💳 Record a payment on an AR invoice", expanded=False):
        ar_open = (
            df_ar_cfg[df_ar_cfg.get("is_cash_relevant", True) & (df_ar_cfg.get("effective_amount", 0) > 0)].copy()
            if df_ar_cfg is not None and not df_ar_cfg.empty
            else pd.DataFrame()
        )

        if ar_open.empty:
            st.caption("No open customer invoices to record payments against.")
        else:
            AR_TABLE_NAME = "ar_ap_tracker"  # 🔁 adjust if needed

            label_map = {}
            for _, r in ar_open.iterrows():
                inv_id = r.get("id")
                inv_no = r.get("invoice_number", inv_id)
                cust = r.get("counterparty", "")
                due = r.get("due_date", "")
                bal = float(r.get("effective_amount") or 0.0)

                try:
                    due_txt = pd.to_datetime(due).strftime("%d %b %Y")
                except Exception:
                    due_txt = str(due)

                label_map[inv_id] = f"{inv_no} – {cust} – due {due_txt} – bal {currency_symbol_cfg}{bal:,.0f}"

            selected_id = st.selectbox(
                "Select invoice to update",
                options=list(label_map.keys()),
                format_func=lambda x: label_map.get(x, str(x)),
                key="cfg_ar_payment_invoice",
            )

            payment_type = st.radio(
                "Payment type",
                ["Full payment", "Partial payment"],
                horizontal=True,
                key="cfg_ar_payment_type",
            )

            current_row = ar_open[ar_open["id"] == selected_id].iloc[0]
            current_balance = float(current_row.get("effective_amount") or 0.0)

            default_amount = current_balance if payment_type == "Full payment" else 0.0

            payment_amount = st.number_input(
                "Amount paid",
                min_value=0.0,
                value=float(default_amount),
                step=100.0,
                key="cfg_ar_payment_amount",
            )

            payment_date = st.date_input(
                "Payment date",
                value=focus_end_cfg,
                key="cfg_ar_payment_date",
            )

            if st.button("Record AR payment", key="cfg_ar_payment_button"):
                try:
                    # Prefer your existing authed client if available
                    sb = None
                    if "get_supabase_authed" in globals():
                        sb = get_supabase_authed()
                    if sb is None and "get_supabase_client" in globals():
                        sb = get_supabase_client()
                    if sb is None:
                        raise RuntimeError("No Supabase client available.")

                    prev_paid = float(current_row.get("amount_paid") or 0.0)
                    new_paid_total = prev_paid + float(payment_amount)

                    total_amt = float(current_row.get("amount") or 0.0)
                    if new_paid_total >= total_amt - 0.01:
                        new_status = "paid"
                        partially_paid_flag = False
                    else:
                        new_status = "open"
                        partially_paid_flag = True

                    update_data = {
                        "amount_paid": new_paid_total,
                        "partially_paid": partially_paid_flag,
                        "status": new_status,
                        "payment_date": payment_date.isoformat(),
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                    }

                    sb.table(AR_TABLE_NAME).update(update_data).eq("id", selected_id).execute()

                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass

                    st.success("AR payment recorded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to record payment: {e}")

    # ----------------------------
    # AP Payment
    # ----------------------------
    with st.expander("💳 Record a payment on an AP bill", expanded=False):
        ap_open = (
            df_ap_cfg[df_ap_cfg.get("is_cash_relevant", True) & (df_ap_cfg.get("effective_amount", 0) > 0)].copy()
            if df_ap_cfg is not None and not df_ap_cfg.empty
            else pd.DataFrame()
        )

        if ap_open.empty:
            st.caption("No open supplier bills to record payments against.")
        else:
            AP_TABLE_NAME = "ar_ap_tracker"  # 🔁 adjust if AP lives in a different table

            label_map_ap = {}
            for _, r in ap_open.iterrows():
                bill_id = r.get("id")
                bill_no = r.get("invoice_number", bill_id)
                supp = r.get("counterparty", "")
                due = r.get("due_date", "")
                bal = float(r.get("effective_amount") or 0.0)

                try:
                    due_txt = pd.to_datetime(due).strftime("%d %b %Y")
                except Exception:
                    due_txt = str(due)

                label_map_ap[bill_id] = f"{bill_no} – {supp} – due {due_txt} – bal {currency_symbol_cfg}{bal:,.0f}"

            selected_bill_id = st.selectbox(
                "Select bill to update",
                options=list(label_map_ap.keys()),
                format_func=lambda x: label_map_ap.get(x, str(x)),
                key="cfg_ap_payment_bill",
            )

            payment_type_ap = st.radio(
                "Payment type",
                ["Full payment", "Partial payment"],
                horizontal=True,
                key="cfg_ap_payment_type",
            )

            current_ap_row = ap_open[ap_open["id"] == selected_bill_id].iloc[0]
            current_balance_ap = float(current_ap_row.get("effective_amount") or 0.0)

            default_amount_ap = current_balance_ap if payment_type_ap == "Full payment" else 0.0

            payment_amount_ap = st.number_input(
                "Amount paid",
                min_value=0.0,
                value=float(default_amount_ap),
                step=100.0,
                key="cfg_ap_payment_amount",
            )

            payment_date_ap = st.date_input(
                "Payment date",
                value=focus_end_cfg,
                key="cfg_ap_payment_date",
            )

            if st.button("Record AP payment", key="cfg_ap_payment_button"):
                try:
                    sb = None
                    if "get_supabase_authed" in globals():
                        sb = get_supabase_authed()
                    if sb is None and "get_supabase_client" in globals():
                        sb = get_supabase_client()
                    if sb is None:
                        raise RuntimeError("No Supabase client available.")

                    prev_paid_ap = float(current_ap_row.get("amount_paid") or 0.0)
                    new_paid_total_ap = prev_paid_ap + float(payment_amount_ap)

                    total_amt_ap = float(current_ap_row.get("amount") or 0.0)
                    if new_paid_total_ap >= total_amt_ap - 0.01:
                        new_status_ap = "paid"
                        partially_paid_flag_ap = False
                    else:
                        new_status_ap = "open"
                        partially_paid_flag_ap = True

                    update_data_ap = {
                        "amount_paid": new_paid_total_ap,
                        "partially_paid": partially_paid_flag_ap,
                        "status": new_status_ap,
                        "payment_date": payment_date_ap.isoformat(),
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                    }

                    sb.table(AP_TABLE_NAME).update(update_data_ap).eq("id", selected_bill_id).execute()

                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass

                    st.success("AP payment recorded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to record payment: {e}")

    st.markdown("---")
    render_monthly_targets_section(
        selected_client_id,
        selected_month_start,
        page_key_prefix="cfg",  # prevents key collisions
    )

     # Load Investing and Financing Entres
    st.markdown("---")
    load_admin = st.toggle("Load detailed investing & financing entries (admin)", value=False, key="fund_admin_load")

    if load_admin:
        with st.expander("📚 Detailed investing & financing entries (admin)", expanded=True):
            # with perf_step("fetch: investing + financing flows (admin only)"):  # ❌ removed
            df_inv = _cached_fetch_investing_flows(selected_client_id)
            if df_inv is None:
                df_inv = pd.DataFrame()

            df_fin = _cached_fetch_financing_flows(selected_client_id)
            if df_fin is None:
                df_fin = pd.DataFrame()

            col_left_admin, col_right_admin = st.columns([2, 1])

            with col_left_admin:
                st.markdown("### Investing flows")
                if df_inv.empty:
                    st.caption("No investing flows yet.")
                else:
                    inv_view = df_inv.copy()
                    inv_view["month_date"] = pd.to_datetime(inv_view["month_date"], errors="coerce")
                    inv_view = inv_view.sort_values("month_date")
                    inv_view["Month"] = inv_view["month_date"].dt.strftime("%b %Y")
                    cols = ["id", "Month"] + [c for c in ["amount", "category", "notes"] if c in inv_view.columns]
                    st.dataframe(inv_view[cols], width="stretch")

                st.markdown("### Financing flows")
                if df_fin.empty:
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
                        ok = create_investing_flow(
                            selected_client_id, month_date=inv_date, amount=inv_amount, category=inv_category, notes=inv_notes
                        )
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
                        ok = create_financing_flow(
                            selected_client_id, month_date=fin_date, amount=fin_amount, category=fin_category, notes=fin_notes
                        )
                        if ok:
                            st.success("Saved. Rebuild cashflow to refresh projections.")
                            st.rerun()
                        else:
                            st.error("Could not save financing flow.")

