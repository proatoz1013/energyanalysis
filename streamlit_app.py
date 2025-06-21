import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tnb_tariff_comparison import show as show_tnb_tariff_comparison

st.set_page_config(page_title="Energy Analysis", layout="wide")

tabs = st.tabs(["Load Profile Analysis", "TNB New Tariff Comparison"])

with tabs[0]:
    st.title("Energy Analysis Dashboard")
    st.subheader("Tariff Setup")

    industry = st.selectbox("Select Industry Type", ["Industrial", "Commercial", "Residential"])
    tariff_options = {
        "Industrial": [
            "E1 - Medium Voltage General",
            "E2 - Medium Voltage Peak/Off-Peak",
            "E3 - High Voltage Peak/Off-Peak",
            "D - Low Voltage Industrial"
        ],
        "Commercial": ["C1 - Low Voltage Commercial", "C2 - Medium Voltage Commercial"],
        "Residential": ["D - Domestic Tariff"]
    }
    tariff_rate = st.selectbox("Select Tariff Rate", tariff_options[industry])

    charging_rates = {
        "E1 - Medium Voltage General": "Base: RM 0.337/kWh, MD: RM 29.60/kW",
        "E2 - Medium Voltage Peak/Off-Peak": "Peak: RM 0.355/kWh, Off-Peak: RM 0.219/kWh, MD: RM 37.00/kW",
        "E3 - High Voltage Peak/Off-Peak": "Peak: RM 0.337/kWh, Off-Peak: RM 0.202/kWh, MD: RM 35.50/kW",
        "D - Low Voltage Industrial": "Tiered: RM 0.38/kWh (1-200 kWh), RM 0.441/kWh (>200 kWh)",
        "C1 - Low Voltage Commercial": "Flat: RM 0.435/kWh, ICPT: RM 0.027/kWh",
        "C2 - Medium Voltage Commercial": "Base: RM 0.385/kWh, MD: RM 25/kW, ICPT: RM 0.16/kWh",
        "D - Domestic Tariff": "Tiered: RM 0.218–0.571/kWh, ICPT: -RM 0.02 to RM 0.10/kWh depending on usage tier"
    }
    st.markdown(f"**Charging Rate:** {charging_rates.get(tariff_rate, 'N/A')}")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("File uploaded and read successfully!")

            st.subheader("Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.subheader("Column Selection")
            timestamp_col = st.selectbox("Select timestamp column", df.columns, key="timestamp_col_selector")
            power_col = st.selectbox("Select power (kW) column", df.select_dtypes(include='number').columns, key="power_col_selector")

            df["Parsed Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.dropna(subset=["Parsed Timestamp"])
            df = df.set_index("Parsed Timestamp")

            def is_tnb_peak_time(timestamp):
                if timestamp.weekday() < 5:
                    if 8 <= timestamp.hour < 22:
                        return True
                return False

            st.subheader("0. Cost Comparison by Tariff")
            voltage_level = st.selectbox("Select Voltage Level", ["Low Voltage", "Medium Voltage", "High Voltage"], key="voltage_level_selector_cost_comp")

            if power_col not in df.columns:
                st.error(f"Selected power column '{power_col}' not found. Please check selection.")
            else:
                df_energy_for_cost_kwh = df[[power_col]].copy()
                df_energy_for_cost_kwh["Energy (kWh)"] = df_energy_for_cost_kwh[power_col] * (1/60)
                peak_energy_cost_calc = df_energy_for_cost_kwh.between_time("08:00", "21:59")["Energy (kWh)"].sum()
                offpeak_energy_cost_calc = df_energy_for_cost_kwh.between_time("00:00", "07:59")["Energy (kWh)"].sum() + df_energy_for_cost_kwh.between_time("22:00", "23:59")["Energy (kWh)"].sum()
                total_energy_cost_calc = peak_energy_cost_calc + offpeak_energy_cost_calc
                max_demand = df[power_col].rolling('30T', min_periods=1).mean().max()

                tariff_data = {
                    "Industrial": [
                        {"Tariff": "E1 - Medium Voltage General", "Voltage": "Medium Voltage", "Base Rate": 0.337, "MD Rate": 29.60, "ICPT": 0, "Split": False, "Tiered": False},
                        {"Tariff": "E2 - Medium Voltage Peak/Off-Peak", "Voltage": "Medium Voltage", "Peak Rate": 0.355, "OffPeak Rate": 0.219, "MD Rate": 37.00, "ICPT": 0, "Split": True, "Tiered": False},
                        {"Tariff": "E3 - High Voltage Peak/Off-Peak", "Voltage": "High Voltage", "Peak Rate": 0.337, "OffPeak Rate": 0.202, "MD Rate": 35.50, "ICPT": 0, "Split": True, "Tiered": False},
                        {"Tariff": "D - Low Voltage Industrial", "Voltage": "Low Voltage", "Tier1 Rate": 0.38, "Tier1 Limit": 200, "Tier2 Rate": 0.441, "MD Rate": 0, "ICPT": 0, "Split": False, "Tiered": True}
                    ],
                    "Commercial": [
                        {"Tariff": "C1 - Low Voltage Commercial", "Voltage": "Low Voltage", "Base Rate": 0.435, "MD Rate": 0, "ICPT": 0.027, "Split": False, "Tiered": False},
                        {"Tariff": "C2 - Medium Voltage Commercial", "Voltage": "Medium Voltage", "Base Rate": 0.385, "MD Rate": 25, "ICPT": 0.16, "Split": False, "Tiered": False}
                    ]
                }
                filtered_tariffs = [t for t in tariff_data.get(industry, []) if t["Voltage"] == voltage_level]
                if filtered_tariffs:
                    cost_table_data = []
                    for t_info in filtered_tariffs:
                        energy_cost = 0
                        if t_info.get("Split"):
                            energy_cost = (peak_energy_cost_calc * t_info["Peak Rate"]) + (offpeak_energy_cost_calc * t_info["OffPeak Rate"])
                        elif t_info.get("Tiered"):
                            if total_energy_cost_calc <= t_info["Tier1 Limit"]:
                                energy_cost = total_energy_cost_calc * t_info["Tier1 Rate"]
                            else:
                                energy_cost = (t_info["Tier1 Limit"] * t_info["Tier1 Rate"]) + ((total_energy_cost_calc - t_info["Tier1 Limit"]) * t_info["Tier2 Rate"])
                        else:
                            energy_cost = total_energy_cost_calc * t_info.get("Base Rate", 0)
                        md_cost = max_demand * t_info.get("MD Rate", 0)
                        icpt_cost = total_energy_cost_calc * t_info.get("ICPT", 0)
                        total_bill = energy_cost + md_cost + icpt_cost
                        cost_table_data.append({
                            "Tariff": t_info["Tariff"],
                            "Peak Energy (kWh)": peak_energy_cost_calc,
                            "Off Peak Energy (kWh)": offpeak_energy_cost_calc,
                            "Total Energy (kWh)": total_energy_cost_calc,
                            "MD (kW)": max_demand,
                            "Energy Cost (RM)": energy_cost,
                            "MD Cost (RM)": md_cost,
                            "ICPT (RM)": icpt_cost,
                            "Total Estimated Bill (RM)": total_bill
                        })
                    if cost_table_data:
                        cost_table = pd.DataFrame(cost_table_data)
                        min_cost = cost_table["Total Estimated Bill (RM)"].min()
                        cost_table["Best Option"] = cost_table["Total Estimated Bill (RM)"].apply(lambda x: "✅ Lowest" if x == min_cost else "")
                        display_cols = [
                            "Tariff", "Peak Energy (kWh)", "Off Peak Energy (kWh)", "Total Energy (kWh)", "MD (kW)",
                            "Energy Cost (RM)", "MD Cost (RM)", "ICPT (RM)", "Total Estimated Bill (RM)", "Best Option"
                        ]
                        cost_table_display = cost_table[display_cols].copy()
                        st.dataframe(cost_table_display.style.format({
                            "Peak Energy (kWh)": "{:,.2f}", "Off Peak Energy (kWh)": "{:,.2f}", "Total Energy (kWh)": "{:,.2f}",
                            "MD (kW)": "{:,.2f}", "Energy Cost (RM)": "{:,.2f}", "MD Cost (RM)": "{:,.2f}",
                            "ICPT (RM)": "{:,.2f}", "Total Estimated Bill (RM)": "{:,.2f}"
                        }), use_container_width=True)
                    else:
                        st.info("No applicable tariffs found for cost calculation.")
                else:
                    st.info(f"No tariffs for Industry: '{industry}', Voltage: '{voltage_level}'.")

            # -----------------------------
            # SECTION: Energy Consumption Over Time
            # Main Title: 1. Energy Consumption Over Time
            # - Shows reference kW lines, daily energy charts, and peak/off-peak share.
            # -----------------------------
            st.subheader("1. Energy Consumption Over Time")
            if power_col not in df.columns:
                st.error(f"Selected power column '{power_col}' not found.")
            else:
                # --- Add Reference kW Lines Section ---
                st.markdown("#### Add Reference kW Lines")
                if "ref_kw_lines" not in st.session_state:
                    st.session_state.ref_kw_lines = []

                if st.button("Add kW Line"):
                    st.session_state.ref_kw_lines.append(0.0)  # Default value

                # Render input fields for each reference kW line
                for i, val in enumerate(st.session_state.ref_kw_lines):
                    new_val = st.number_input(
                        f"Reference kW Line {i+1}",
                        min_value=0.0,
                        value=val,
                        step=0.1,
                        key=f"ref_kw_{i}"
                    )
                    st.session_state.ref_kw_lines[i] = new_val

                # kWh Chart (Daily Energy Consumption by Peak and Off-Peak)
                df_energy_kwh_viz = df[[power_col]].copy()
                df_energy_kwh_viz["Energy (kWh)"] = df_energy_kwh_viz[power_col] * (1/60)
                df_peak_viz = df_energy_kwh_viz.between_time("08:00", "21:59").copy(); df_peak_viz["Period"] = "Peak"
                df_offpeak_viz = pd.concat([df_energy_kwh_viz.between_time("00:00", "07:59"), df_energy_kwh_viz.between_time("22:00", "23:59")]).copy(); df_offpeak_viz["Period"] = "Off Peak"
                df_energy_period_viz = pd.concat([df_peak_viz, df_offpeak_viz])
                df_daily_period_viz = df_energy_period_viz.groupby([pd.Grouper(freq="D"), "Period"])["Energy (kWh)"].sum().reset_index()
                date_col_for_plot = 'Parsed Timestamp'; df_daily_period_viz_plot = df_daily_period_viz
                if 'Parsed Timestamp' not in df_daily_period_viz.columns and isinstance(df_daily_period_viz.index, pd.DatetimeIndex):
                     df_daily_period_viz_plot = df_daily_period_viz.reset_index(); date_col_for_plot = df_daily_period_viz_plot.columns[0]
                elif 'level_0' in df_daily_period_viz.columns and pd.api.types.is_datetime64_any_dtype(df_daily_period_viz['level_0']): date_col_for_plot = 'level_0'
                
                fig_kwh = px.area(df_daily_period_viz_plot, x=date_col_for_plot, y="Energy (kWh)", color="Period", labels={date_col_for_plot: "Date", "Energy (kWh)": "Daily Energy Consumption (kWh)"}, title="Daily Energy Consumption by Period (kWh)")
                # Add horizontal lines for each reference kW value
                for ref_kw in st.session_state.ref_kw_lines:
                    if ref_kw is not None and ref_kw > 0:
                        fig_kwh.add_hline(
                            y=ref_kw,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"{ref_kw:.2f} kW",
                            annotation_position="top left"
                        )
                st.plotly_chart(fig_kwh, use_container_width=True)
                
                st.markdown("#### Daily Peak vs Off-Peak Energy Share (%)")
                df_daily_pivot_viz = df_daily_period_viz_plot.pivot(index=date_col_for_plot, columns="Period", values="Energy (kWh)").fillna(0)
                if not df_daily_pivot_viz.empty:
                    df_daily_pivot_viz["Total"] = df_daily_pivot_viz.sum(axis=1)
                    df_daily_pivot_viz["Peak %"] = (df_daily_pivot_viz.get("Peak", 0) / df_daily_pivot_viz["Total"].replace(0, np.nan) * 100).round(2)
                    df_daily_pivot_viz["Off Peak %"] = (df_daily_pivot_viz.get("Off Peak", 0) / df_daily_pivot_viz["Total"].replace(0, np.nan) * 100).round(2)
                    st.dataframe(df_daily_pivot_viz[["Peak %", "Off Peak %"]].fillna(0).style.format({"Peak %": "{:.2f}%", "Off Peak %": "{:.2f}%"}), use_container_width=True)
                else: st.info("Not enough data for daily peak/off-peak share.")

                # Filterable Power Consumption Trend
                st.markdown("---") 
                st.markdown("#### Detailed Power Consumption Trend (Filterable)")

                trend_filter_option = st.radio(
                    "Filter Trend by Period:",
                    options=["All", "Peak Only", "Off-Peak Only"],
                    index=0, # Default to "All"
                    horizontal=True,
                    key="power_trend_filter_detailed"
                )

                df_for_detailed_trend = df.copy() 
                title_suffix = " (All Periods)"

                if trend_filter_option == "Peak Only":
                    df_for_detailed_trend = df_for_detailed_trend.between_time("08:00", "21:59")
                    title_suffix = " (Peak Period Only)"
                elif trend_filter_option == "Off-Peak Only":
                    df_offpeak_part1 = df_for_detailed_trend.between_time("00:00", "07:59")
                    df_offpeak_part2 = df_for_detailed_trend.between_time("22:00", "23:59")
                    df_for_detailed_trend = pd.concat([df_offpeak_part1, df_offpeak_part2]).sort_index()
                    title_suffix = " (Off-Peak Period Only)"
                
                # --- Maximum Demand Line (Trend) ---
                max_demand_val = df_for_detailed_trend[power_col].max() if not df_for_detailed_trend.empty else 0
                st.markdown(f"**Maximum Demand (Trend):** {max_demand_val:.2f} kW")

                # --- Add Reference kW Lines Section for Detailed Power Consumption Trend ---
                st.markdown("#### Add Reference kW Lines (Trend Chart)")
                if "ref_kw_lines_trend" not in st.session_state:
                    st.session_state.ref_kw_lines_trend = []

                if st.button("Add kW Line (Trend Chart)", key="add_kw_line_trend"):
                    st.session_state.ref_kw_lines_trend.append(0.0)

                # Render input fields and delete buttons for each reference kW line (side by side, using % of max demand)
                indices_to_delete = []
                for i, val in enumerate(st.session_state.ref_kw_lines_trend):
                    cols = st.columns([5, 2, 1])
                    # Show as percent of max demand
                    percent_val = 0.0
                    if max_demand_val > 0:
                        percent_val = (val / max_demand_val) * 100
                    new_percent = cols[0].number_input(
                        f"Reference kW Line (Trend) {i+1} (% of Max Demand)",
                        min_value=-100.0,
                        max_value=200.0,
                        value=percent_val,
                        step=1.0,
                        key=f"ref_kw_trend_percent_{i}"
                    )
                    # Update the kW value based on percent
                    new_val = (new_percent / 100.0) * max_demand_val if max_demand_val > 0 else 0.0
                    st.session_state.ref_kw_lines_trend[i] = new_val
                    # Show the actual kW value next to the percent
                    cols[1].markdown(f"**{new_val:.2f} kW**")
                    # Delete button aligned
                    if cols[2].button("Delete", key=f"delete_ref_kw_trend_{i}"):
                        indices_to_delete.append(i)
                for idx in sorted(indices_to_delete, reverse=True):
                    st.session_state.ref_kw_lines_trend.pop(idx)

                # --- Peak Event Detection Logic (define variables for all blocks below) ---
                # Use the first reference kW line (Trend) as the target max demand if set and >0
                target_max_demand = None
                if st.session_state.ref_kw_lines_trend and st.session_state.ref_kw_lines_trend[0] > 0:
                    target_max_demand = st.session_state.ref_kw_lines_trend[0]
                else:
                    target_max_demand = max_demand_val
                PEAK_THRESHOLD = target_max_demand
                # Get MD Rate for selected tariff
                md_rate_for_selected_tariff = 0
                current_industry_tariffs_list = tariff_data.get(industry, [])
                for t_info_detail in current_industry_tariffs_list:
                    if t_info_detail["Tariff"] == tariff_rate:
                        md_rate_for_selected_tariff = t_info_detail.get("MD Rate", 0)
                        break
                if not md_rate_for_selected_tariff:
                    for ind_key_fallback in tariff_data:
                        for t_info_detail_fallback in tariff_data[ind_key_fallback]:
                            if t_info_detail_fallback["Tariff"] == tariff_rate:
                                md_rate_for_selected_tariff = t_info_detail_fallback.get("MD Rate", 0)
                                break
                        if md_rate_for_selected_tariff:
                            break
                df_peak_events = df_for_detailed_trend[[power_col]].copy()
                df_peak_events['Above_Threshold'] = df_peak_events[power_col] > PEAK_THRESHOLD
                df_peak_events['Event_ID'] = (df_peak_events['Above_Threshold'] != df_peak_events['Above_Threshold'].shift()).cumsum()
                event_summaries = []
                for event_id, group in df_peak_events.groupby('Event_ID'):
                    if not group['Above_Threshold'].iloc[0]:
                        continue  # Only interested in blocks above threshold
                    start_time = group.index[0]
                    end_time = group.index[-1]
                    peak_load = group[power_col].max()
                    excess = peak_load - PEAK_THRESHOLD
                    duration = (end_time - start_time).total_seconds() / 60  # minutes
                    md_cost = excess * md_rate_for_selected_tariff if excess > 0 else 0
                    # Calculate total kWh required to shave the maximum demand for this event
                    # Only consider points above threshold
                    group_above = group[group[power_col] > PEAK_THRESHOLD]
                    # Energy above threshold: sum((load - threshold) * (interval in hours))
                    # Assume data is at 1-min intervals (1/60 hr), but check actual interval
                    if len(group_above) > 1:
                        interval_minutes = (group_above.index[1] - group_above.index[0]).total_seconds() / 60
                    else:
                        interval_minutes = 1  # fallback
                    interval_hours = interval_minutes / 60
                    total_kwh_to_shave = ((group_above[power_col] - PEAK_THRESHOLD) * interval_hours).sum()
                    event_summaries.append({
                        'Date': start_time.date(),
                        'Start Time': start_time.strftime('%H:%M'),
                        'End Time': end_time.strftime('%H:%M'),
                        'Peak Load (kW)': peak_load,
                        f'Excess over {PEAK_THRESHOLD:,.2f} (kW)': excess,
                        'Total kWh to Shave MD': total_kwh_to_shave,
                        'Duration (minutes)': duration,
                        'Maximum Demand Cost (RM)': md_cost
                    })
                # -----------------------------
                # SECTION: Power Trend with Peak Events Highlighted

                # -----------------------------
                # SECTION: Power Consumption Trend (Area Chart)
                # [1] Power Consumption Trend (Chart SECOND)
                # -----------------------------
                if not df_for_detailed_trend.empty:
                    fig_energy_trend = px.area(df_for_detailed_trend.reset_index(), x="Parsed Timestamp", y=power_col, labels={"Parsed Timestamp": "Time", power_col: f"Power ({power_col})"}, title=f"Power Consumption Trend (kW){title_suffix}")
                    # Always add the max demand line (blue)
                    if max_demand_val > 0:
                        fig_energy_trend.add_hline(
                            y=max_demand_val,
                            line_dash="dash",
                            line_color="blue",
                            annotation_text=f"Max Demand: {max_demand_val:.2f} kW",
                            annotation_position="top left"
                        )
                    # Add user reference lines (red)
                    for ref_kw in st.session_state.ref_kw_lines_trend:
                        if ref_kw is not None and ref_kw > 0:
                            fig_energy_trend.add_hline(
                                y=ref_kw,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"{ref_kw:.2f} kW",
                                annotation_position="top left"
                            )
                    st.plotly_chart(fig_energy_trend, use_container_width=True)
                else:
                    st.info(f"No data available for the selected period: {trend_filter_option}")

                # [2] Power Trend with Peak Events Highlighted (Chart FIRST)
                # -----------------------------
                if not df_for_detailed_trend.empty:
                    fig_peak_highlight = go.Figure()
                    fig_peak_highlight.add_trace(go.Scatter(
                        x=df_for_detailed_trend.index,
                        y=df_for_detailed_trend[power_col],
                        mode='lines',
                        name='Power (kW)',
                        line=dict(color='gray')
                    ))
                    # Highlight peak regions
                    for event in event_summaries:
                        mask = (df_for_detailed_trend.index.date == event['Date']) & \
                               (df_for_detailed_trend.index.strftime('%H:%M') >= event['Start Time']) & \
                               (df_for_detailed_trend.index.strftime('%H:%M') <= event['End Time'])
                        if mask.any():
                            fig_peak_highlight.add_trace(go.Scatter(
                                x=df_for_detailed_trend.index[mask],
                                y=df_for_detailed_trend[power_col][mask],
                                mode='lines',
                                name=f"Peak Event ({event['Date']})",
                                line=dict(color='red', width=3),
                                showlegend=False,
                                opacity=0.5
                            ))
                    # Add threshold line
                    fig_peak_highlight.add_hline(y=PEAK_THRESHOLD, line_dash="dot", line_color="orange", annotation_text=f"Target Max Demand: {PEAK_THRESHOLD:,.2f} kW", annotation_position="top left")
                    # Add max demand line
                    if max_demand_val > 0:
                        fig_peak_highlight.add_hline(y=max_demand_val, line_dash="dash", line_color="blue", annotation_text=f"Max Demand: {max_demand_val:.2f} kW", annotation_position="top left")
                    # Add reference lines
                    for ref_kw in st.session_state.ref_kw_lines_trend:
                        if ref_kw is not None and ref_kw > 0:
                            fig_peak_highlight.add_hline(y=ref_kw, line_dash="dash", line_color="red", annotation_text=f"{ref_kw:.2f} kW", annotation_position="top left")
                    fig_peak_highlight.update_layout(title="Power Trend with Peak Events Highlighted", xaxis_title="Time", yaxis_title=f"Power ({power_col})", height=500)
                    st.plotly_chart(fig_peak_highlight, use_container_width=True)


                # -----------------------------
                # SECTION: Peak Event Detection and Summary Table
                # [3] Peak Event Detection (Table and Cost Summary THIRD)
                # -----------------------------
                st.markdown("---")
                st.markdown(f"#### Peak Event Detection (Load > Target Max Demand: {target_max_demand:.2f} kW)")
                if event_summaries:
                    df_events_summary = pd.DataFrame(event_summaries)
                    if 'Maximum Demand Cost (RM)' in df_events_summary.columns:
                        cols = [c for c in df_events_summary.columns if c != 'Maximum Demand Cost (RM)'] + ['Maximum Demand Cost (RM)']
                        df_events_summary = df_events_summary[cols]
                    st.dataframe(df_events_summary.style.format({
                        'Peak Load (kW)': '{:,.2f}',
                        f'Excess over {PEAK_THRESHOLD:,.2f} (kW)': '{:,.2f}',
                        'Duration (minutes)': '{:,.1f}',
                        'Maximum Demand Cost (RM)': 'RM {:,.2f}'
                    }), use_container_width=True)
                    max_md_cost = df_events_summary['Maximum Demand Cost (RM)'].max()
                    st.markdown(f"**Maximum of Maximum Demand Cost (RM): RM {max_md_cost:,.2f}**")
                else:
                    st.info(f"No peak events (load > {PEAK_THRESHOLD:,.2f} kW) detected in the selected period.")

                # -----------------------------
            # 2. Load Duration Curve
            st.subheader("2. Load Duration Curve")
            if power_col not in df.columns:
                st.error(f"Selected power column '{power_col}' not found.")
            else:
                ldc_period = st.radio("Select Time Range for LDC", ["All", "Peak Only", "Off Peak Only"], horizontal=True, key="ldc_period_selection")
                top_percentage_filter = st.number_input("Filter LDC: Show Top % of Highest Demand Readings", 0.1, 100.0, 100.0, 0.1, "%.1f", key="ldc_top_percentage_filter", help="Enter % (e.g., 1 for top 1%). Shows LDC for highest X% of demand readings.")
                st.caption(f"LDC shows top {top_percentage_filter:.1f}% demand readings for '{ldc_period}' period.")

                if ldc_period == "Peak Only": ldc_df_base = df.between_time("08:00", "21:59")[[power_col]].copy()
                elif ldc_period == "Off Peak Only": ldc_df_base = pd.concat([df.between_time("00:00", "07:59")[[power_col]], df.between_time("22:00", "23:59")[[power_col]]]).copy()
                else: ldc_df_base = df[[power_col]].copy()
                
                ldc_df_sorted = ldc_df_base.dropna(subset=[power_col]).sort_values(by=power_col, ascending=False).reset_index(drop=True)

                if not ldc_df_sorted.empty:
                    plot_df_ldc = ldc_df_sorted.copy()
                    if top_percentage_filter < 100.0:
                        num_points_to_show = int(np.ceil(len(ldc_df_sorted) * (top_percentage_filter / 100.0)))
                        if num_points_to_show == 0 and len(ldc_df_sorted) > 0: num_points_to_show = 1
                        plot_df_ldc = ldc_df_sorted.head(num_points_to_show).copy()

                    if not plot_df_ldc.empty:
                        plot_df_ldc["Percentage Time"] = (plot_df_ldc.index + 1) / len(plot_df_ldc) * 100
                        chart_title = f"Load Duration Curve - {ldc_period} ({'Top ' + str(top_percentage_filter) + '%' if top_percentage_filter < 100.0 else 'All'} Readings)"
                        xaxis_label = f"Cumulative % of these {('Top ' + str(top_percentage_filter) + '%' if top_percentage_filter < 100.0 else '')} Readings" if top_percentage_filter < 100.0 else "% of Time (Load is at or above this level)"
                        yaxis_label = f"Power ({power_col})"
                        fig_ldc = px.line(plot_df_ldc, x="Percentage Time", y=power_col, labels={"Percentage Time": xaxis_label, power_col: yaxis_label}, title=chart_title)
                        fig_ldc.update_traces(mode="lines+markers"); fig_ldc.update_layout(xaxis_title=xaxis_label, yaxis_title=yaxis_label, height=500)
                        st.plotly_chart(fig_ldc, use_container_width=True)

                        # --- AUTOMATED LDC ANALYSIS FOR DEMAND SHAVING ---
                        analysis_message_parts = []
                        analysis_conclusion_style = st.warning 
                        min_points_for_analysis = 5
                        if len(plot_df_ldc) >= min_points_for_analysis:
                            p_peak = plot_df_ldc[power_col].iloc[0]
                            shoulder_index_target_percentage = 0.05 
                            shoulder_index = min(max(1, int(shoulder_index_target_percentage * len(plot_df_ldc))), len(plot_df_ldc) - 1)
                            p_shoulder = plot_df_ldc[power_col].iloc[shoulder_index]
                            percentage_time_at_shoulder = plot_df_ldc['Percentage Time'].iloc[shoulder_index]

                            if p_peak > (p_shoulder + 1.0): 
                                shave_potential_kw = p_peak - p_shoulder
                                relative_shave_potential = shave_potential_kw / p_peak if p_peak > 0 else 0
                                is_relatively_deep_peak = relative_shave_potential > 0.15
                                is_absolutely_significant_shave = shave_potential_kw > 10

                                if is_relatively_deep_peak and is_absolutely_significant_shave:
                                    analysis_conclusion_style = st.success
                                    analysis_message_parts.append(f"**LDC Analysis: GOOD for Demand Shaving.**")
                                    analysis_message_parts.append(
                                        (f"- **Observation:** Distinct peak. Max demand on this LDC: **{p_peak:,.2f} kW**. "
                                         f"Drops to ~**{p_shoulder:,.2f} kW** by **{percentage_time_at_shoulder:.1f}%** of displayed duration.")
                                    )
                                    analysis_message_parts.append(f"- **Estimated Shaving Potential:** Roughly **{shave_potential_kw:,.2f} kW**.")

                                    # Find original timestamps for the P_peak value from the LDC's base data (before sorting for LDC plot)
                                    peak_value_timestamps = ldc_df_base[np.isclose(ldc_df_base[power_col], p_peak)].index.to_list()
                                    if peak_value_timestamps:
                                        timestamps_str_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in peak_value_timestamps[:5]] 
                                        timestamps_display_str = "; ".join(timestamps_str_list)
                                        if len(peak_value_timestamps) > 5: timestamps_display_str += f"; ... ({len(peak_value_timestamps) - 5} more)"
                                        analysis_message_parts.append(f"- **Time(s) of this {p_peak:,.2f} kW peak:** {timestamps_display_str}")
                                        
                                        tnb_peak_hour_count = sum(1 for ts in peak_value_timestamps if is_tnb_peak_time(ts))
                                        total_found_timestamps = len(peak_value_timestamps)
                                        time_context_summary = ""
                                        if tnb_peak_hour_count > 0:
                                            if tnb_peak_hour_count == total_found_timestamps: time_context_summary = "All these peaks are within typical TNB peak periods (Mon-Fri, 08:00-21:59)."
                                            elif tnb_peak_hour_count >= total_found_timestamps / 2: time_context_summary = f"A majority ({tnb_peak_hour_count}/{total_found_timestamps}) of these peaks are within TNB peak periods."
                                            else: time_context_summary = f"Some ({tnb_peak_hour_count}/{total_found_timestamps}) of these peaks are within TNB peak periods."
                                            analysis_message_parts.append(f"- **Time Frame Context:** {time_context_summary} This makes them critical targets.")
                                        else:
                                            analysis_message_parts.append("- **Time Frame Context:** These peaks occur mainly outside TNB peak periods.")
                                    else:
                                        analysis_message_parts.append(f"- *Could not pinpoint exact timestamps for the peak value {p_peak:,.2f} kW from source data for this LDC period.*")
                                    
                                    md_rate_for_selected_tariff = 0
                                    current_industry_tariffs_list = tariff_data.get(industry, [])
                                    for t_info_detail in current_industry_tariffs_list:
                                        if t_info_detail["Tariff"] == tariff_rate: md_rate_for_selected_tariff = t_info_detail.get("MD Rate", 0); break
                                    if not md_rate_for_selected_tariff:
                                         for ind_key_fallback in tariff_data:
                                            for t_info_detail_fallback in tariff_data[ind_key_fallback]:
                                                if t_info_detail_fallback["Tariff"] == tariff_rate: md_rate_for_selected_tariff = t_info_detail_fallback.get("MD Rate", 0); break
                                            if md_rate_for_selected_tariff: break
                                    if md_rate_for_selected_tariff > 0:
                                        financial_saving_estimate = shave_potential_kw * md_rate_for_selected_tariff
                                        analysis_message_parts.append(
                                            (f"- **Potential Monthly Saving (tariff '{tariff_rate}'):** "
                                             f"~**RM {financial_saving_estimate:,.2f}** (MD rate: RM {md_rate_for_selected_tariff:.2f}/kW).")
                                        )
                                    else: analysis_message_parts.append(f"- *Financial saving not estimated: selected tariff '{tariff_rate}' has no MD rate or MD rate is zero.*")
                                else: 
                                    analysis_message_parts.append(f"**LDC Analysis: LESS IDEAL for Simple Demand Shaving.**")
                                    analysis_message_parts.append(
                                        (f"- **Observation:** Peak ({p_peak:,.2f} kW) not significantly pronounced over shoulder ({p_shoulder:,.2f} kW at {percentage_time_at_shoulder:.1f}% duration), "
                                         f"or kW reduction potential ({shave_potential_kw:,.2f} kW) is small.")
                                    )
                            else: 
                                analysis_message_parts.append(f"**LDC Analysis: NOT GOOD for Demand Shaving.**")
                                analysis_message_parts.append(f"- **Observation:** Curve is relatively flat or no clear sharp peak. Max demand: {p_peak:,.2f} kW.")
                        else: 
                            analysis_message_parts.append(f"**LDC Analysis: Insufficient Data for Automated Assessment.**")
                            analysis_message_parts.append(
                                (f"- At least {min_points_for_analysis} data points are needed on the LDC (after filtering) for analysis. "
                                 "Try different LDC filters or ensure adequate data in the selected period.")
                            )
                        if analysis_message_parts: analysis_conclusion_style("\n\n".join(analysis_message_parts))
                        # --- END OF AUTOMATED LDC ANALYSIS ---
                    else: st.info(f"No data for LDC: Filter (Top {top_percentage_filter:.1f}%) resulted in zero points, or no data in '{ldc_period}'.")
                else: st.info(f"No data to generate LDC for period: {ldc_period}.")

            # -----------------------------
            # SECTION: Processed 30-Min Average Data Table & Heatmap
            # Main Title: 3. Processed 30-Min Average Data & Heatmap
            # - Shows 30-min average table and heatmap for power data.
            # -----------------------------
            st.subheader("3. Processed 30-Min Average Data & Heatmap")
            if power_col not in df.columns:
                st.error(f"Selected power column '{power_col}' not found.")
            else:
                df_30min_avg = df[[power_col]].resample("30T").mean().dropna()
                df_30min_avg_renamed = df_30min_avg.rename(columns={power_col: "Average Power (kW)"}) # Renamed for clarity
                df_30min_avg_display = df_30min_avg_renamed.reset_index().rename(columns={"Parsed Timestamp": "Timestamp"})
                st.dataframe(df_30min_avg_display.head(50).style.format({"Average Power (kW)": "{:,.2f}"}), use_container_width=True)

                df_30min_avg_heatmap_prep = df_30min_avg_display.copy()
                df_30min_avg_heatmap_prep["Date"] = pd.to_datetime(df_30min_avg_heatmap_prep["Timestamp"]).dt.date
                df_30min_avg_heatmap_prep["30min_Interval"] = pd.to_datetime(df_30min_avg_heatmap_prep["Timestamp"]).dt.strftime("%H:%M")
                pivot_proc = df_30min_avg_heatmap_prep.pivot(index="30min_Interval", columns="Date", values="Average Power (kW)")
                
                if not pivot_proc.empty:
                    min_val_default = float(df_30min_avg_heatmap_prep["Average Power (kW)"].min()) if not df_30min_avg_heatmap_prep["Average Power (kW)"].empty else 0
                    max_val_default = float(df_30min_avg_heatmap_prep["Average Power (kW)"].max()) if not df_30min_avg_heatmap_prep["Average Power (kW)"].empty else 100
                    min_val = st.number_input("Minimum kW for heatmap color scale", value=min_val_default, step=10.0, key="proc_min_heatmap")
                    max_val = st.number_input("Maximum kW for heatmap color scale", value=max_val_default, step=10.0, key="proc_max_heatmap")
                    fig_proc_heatmap = px.imshow(pivot_proc.sort_index(), labels=dict(x="Date", y="30-Minute Interval", color="Average Power (kW)"), aspect="auto",
                                                 color_continuous_scale=[[0, "#ffffcc"], [0.3, "#ffeda0"], [0.6, "#feb24c"], [0.9, "#f03b20"], [1, "#bd0026"]],
                                                 title="Heatmap of Averaged 30-Minute Power Data")
                    fig_proc_heatmap.update_coloraxes(colorbar_title="Average Power (kW)", cmin=min_val, cmax=max_val)
                    fig_proc_heatmap.update_layout(height=600, yaxis={'title': '30-Minute Interval Starting'})
                    st.plotly_chart(fig_proc_heatmap, use_container_width=True)
                else: st.info("Not enough data for heatmap from 30-min averages.")

            # -----------------------------
            # SECTION: Energy Consumption Statistics
            # Main Title: 4. Energy Consumption Statistics
            # - Shows total energy, peak demand, average power, and detailed charts.
            # -----------------------------
            st.subheader("4. Energy Consumption Statistics")
            if power_col not in df.columns:
                st.error(f"Selected power column '{power_col}' not found.")
            else:
                total_energy_mwh = (df[power_col].sum() * (1/60)) / 1000 
                average_power_kw = df[power_col].mean()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Energy Consumed (MWh)", f"{total_energy_mwh:.2f}")
                # max_demand is defined in section 0.
                col2.metric("Peak Demand (kW)", f"{max_demand:.2f}") 
                col3.metric("Average Power (kW)", f"{average_power_kw:.2f}")
                with st.expander("See detailed consumption statistics charts"):
                    st.write("Hourly Average Power (kW):")
                    df_hourly_avg_power = df[power_col].resample('H').mean()
                    st.line_chart(df_hourly_avg_power)
                    st.write("Daily Total Energy Consumption (kWh):")
                    df_daily_total_energy = df[power_col].resample('D').apply(lambda x: x.sum() * (1/60) if not x.empty else 0)
                    st.bar_chart(df_daily_total_energy)
                    st.write("Weekly Total Energy Consumption (kWh):")
                    df_weekly_total_energy = df[power_col].resample('W').apply(lambda x: x.sum() * (1/60) if not x.empty else 0)
                    st.bar_chart(df_weekly_total_energy)

        except pd.errors.EmptyDataError: st.error("Uploaded Excel file is empty or unreadable.")
        except KeyError as e: st.error(f"Column key error: {e}. Check column selection/Excel structure.")
        except ValueError as e: st.error(f"Value error: {e}. Check data types/parsing.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error("Ensure Excel file is correctly formatted and columns are selected.")
            # import traceback; st.error(traceback.format_exc()) # For debugging

