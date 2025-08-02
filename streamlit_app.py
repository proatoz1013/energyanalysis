import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from tnb_tariff_comparison import show as show_tnb_tariff_comparison
from advanced_energy_analysis import show as show_advanced_energy_analysis
from md_shaving_solution import show as show_md_shaving_solution
import sys
import os

# Add the chiller dashboard directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'energyanalaysis', 'chiller-energy-dashboard', 'src'))

st.set_page_config(page_title="Load Profile Analysis", layout="wide")

# Load custom CSS for global styling (including increased font sizes)
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")

# Sidebar Configuration
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# AFA Rate Configuration (Global setting)
st.sidebar.markdown("### AFA Rate Setting")
st.sidebar.caption("Alternative Fuel Agent rate for RP4 calculations")

if 'global_afa_rate' not in st.session_state:
    st.session_state.global_afa_rate = 3.0

global_afa_rate_cent = st.sidebar.number_input(
    "AFA Rate (cent/kWh)", 
    min_value=-10.0, 
    max_value=10.0, 
    value=st.session_state.global_afa_rate, 
    step=0.1,
    help="Current AFA rate in cents per kWh. Used for RP4 tariff calculations."
)

# Update session state
st.session_state.global_afa_rate = global_afa_rate_cent
global_afa_rate = global_afa_rate_cent / 100

st.sidebar.markdown("---")
st.sidebar.info(f"**Current AFA Rate:** {global_afa_rate_cent:+.1f} cent/kWh")
if global_afa_rate_cent >= 0:
    st.sidebar.success("✅ AFA adds to electricity cost")
else:
    st.sidebar.warning("⚠️ AFA reduces electricity cost")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.markdown("""
**Energy Analysis Dashboard**

This tool provides comprehensive analysis of:
- TNB tariff comparisons
- Load profile analysis  
- Advanced RP4 integration
- Monthly rate impact analysis

**RP4 Features:**
- Holiday-aware peak/off-peak logic
- Accurate capacity + network MD rates
- AFA rate integration
- Demand shaving analysis
""")

tabs = st.tabs(["TNB New Tariff Comparison", "Load Profile Analysis", "Advanced Energy Analysis", "Monthly Rate Impact Analysis", "MD Shaving Solution", "🔋 Advanced MD Shaving", "❄️ Chiller Energy Dashboard"])

with tabs[1]:
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

    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xls", "xlsx"])

    # Helper function to read different file formats
    def read_uploaded_file(file):
        """Read uploaded file based on its extension"""
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")

    # Add preprocessing logic to handle any type of timestamp format
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            st.success("File uploaded and read successfully!")

            st.subheader("Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Ensure unique keys for Streamlit widgets
            # Update the keys for timestamp and power column selection

            # Column Selection
            st.subheader("Column Selection")
            timestamp_col = st.selectbox("Select timestamp column", df.columns, key="unique_timestamp_col_selector")
            power_col = st.selectbox("Select power (kW) column", df.select_dtypes(include='number').columns, key="unique_power_col_selector")

            # Preprocess timestamp column to handle various formats
            def preprocess_timestamp_column(column):
                # Remove leading/trailing spaces
                column = column.str.strip()
                # Replace text-based months with numeric equivalents (e.g., Jan -> 01)
                column = column.str.replace(r'\bJan\b', '01', regex=True)
                column = column.str.replace(r'\bFeb\b', '02', regex=True)
                column = column.str.replace(r'\bMar\b', '03', regex=True)
                column = column.str.replace(r'\bApr\b', '04', regex=True)
                column = column.str.replace(r'\bMay\b', '05', regex=True)
                column = column.str.replace(r'\bJun\b', '06', regex=True)
                column = column.str.replace(r'\bJul\b', '07', regex=True)
                column = column.str.replace(r'\bAug\b', '08', regex=True)
                column = column.str.replace(r'\bSep\b', '09', regex=True)
                column = column.str.replace(r'\bOct\b', '10', regex=True)
                column = column.str.replace(r'\bNov\b', '11', regex=True)
                column = column.str.replace(r'\bDec\b', '12', regex=True)
                return column
                column = column.str.replace(r'\bSep\b', '09', regex=True)
                column = column.str.replace(r'\bOct\b', '10', regex=True)
                column = column.str.replace(r'\bNov\b', '11', regex=True)
                column = column.str.replace(r'\bDec\b', '12', regex=True)
                return column

            # Process data with automatic interval detection
            df["Parsed Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.dropna(subset=["Parsed Timestamp"])
            df = df.set_index("Parsed Timestamp")
            
            # Store processed data in session state
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['processed_df'] = df
            st.session_state['power_column'] = power_col
            st.session_state['timestamp_column'] = timestamp_col

            # === DATA INTERVAL DETECTION ===
            st.subheader("📊 Data Interval Detection")
            
            # Detect data interval from the entire dataset
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    # Get the most common time interval (mode)
                    most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
                    interval_minutes = most_common_interval.total_seconds() / 60
                    interval_hours = most_common_interval.total_seconds() / 3600
                    
                    # Check for consistency
                    unique_intervals = time_diffs.value_counts()
                    consistency_percentage = (unique_intervals.iloc[0] / len(time_diffs)) * 100 if len(unique_intervals) > 0 else 100
                    
                    # Display interval information
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if interval_minutes < 60:
                            st.metric("Detected Interval", f"{interval_minutes:.0f} minutes")
                        else:
                            hours = interval_minutes / 60
                            st.metric("Detected Interval", f"{hours:.1f} hours")
                    
                    with col2:
                        st.metric("Data Consistency", f"{consistency_percentage:.1f}%")
                    
                    with col3:
                        readings_per_hour = 60 / interval_minutes if interval_minutes > 0 else 0
                        st.metric("Readings/Hour", f"{readings_per_hour:.0f}")
                    
                    with col4:
                        daily_readings = readings_per_hour * 24
                        st.metric("Readings/Day", f"{daily_readings:.0f}")
                    
                    # Show interval quality indicator
                    if consistency_percentage >= 95:
                        st.success(f"✅ **Excellent data quality**: {consistency_percentage:.1f}% of readings have consistent {interval_minutes:.0f}-minute intervals")
                    elif consistency_percentage >= 85:
                        st.warning(f"⚠️ **Good data quality**: {consistency_percentage:.1f}% consistent intervals. Some missing data points detected.")
                    else:
                        st.error(f"❌ **Poor data quality**: Only {consistency_percentage:.1f}% consistent intervals. Results may be less accurate.")
                    
                    # Show interval breakdown if there are inconsistencies
                    if consistency_percentage < 95:
                        with st.expander("🔍 View Interval Analysis Details"):
                            st.markdown("**Top 5 Time Intervals Found:**")
                            top_intervals = unique_intervals.head()
                            for interval, count in top_intervals.items():
                                percentage = (count / len(time_diffs)) * 100
                                interval_mins = interval.total_seconds() / 60
                                st.write(f"• **{interval_mins:.0f} minutes**: {count:,} occurrences ({percentage:.1f}%)")
                            
                            st.info("💡 **Tip**: Inconsistent intervals may be due to missing data points or different data collection periods. The analysis will use the most common interval for calculations.")
                else:
                    # Fallback to 15 minutes if we can't determine interval
                    interval_hours = 0.25
                    interval_minutes = 15
                    st.warning("⚠️ Could not detect data interval automatically. Using default 15-minute intervals.")
            else:
                interval_hours = 0.25
                interval_minutes = 15
                st.warning("⚠️ Insufficient data to detect interval. Using default 15-minute intervals.")

            def is_tnb_peak_time(timestamp):
                if timestamp.weekday() < 5:
                    if 8 <= timestamp.hour < 22:
                        return True
                return False

            st.subheader("0. Cost Comparison by Tariff")
            voltage_level = st.selectbox("Select Voltage Level", ["Low Voltage", "Medium Voltage", "High Voltage"], key="voltage_level_selector_cost_comp")

            # Debugging statements to verify data structures
            st.write("Uploaded DataFrame:", df.head())
            st.write("Selected Timestamp Column:", timestamp_col)
            st.write("Selected Power Column:", power_col)

            # Cost calculation logic using detected interval
            st.info(f"ℹ️ **Using detected interval**: {interval_minutes:.0f} minutes ({interval_hours:.3f} hours) for all energy calculations")

            # Energy calculation based on detected interval
            df_energy_for_cost_kwh = df[[power_col]].copy()
            df_energy_for_cost_kwh["Energy (kWh)"] = df_energy_for_cost_kwh[power_col] * interval_hours

            # Cost calculation logic using the detected time interval
            peak_energy_cost_calc = df_energy_for_cost_kwh.between_time("08:00", "21:59")["Energy (kWh)"].sum()
            offpeak_energy_cost_calc = df_energy_for_cost_kwh.between_time("00:00", "07:59")["Energy (kWh)"].sum() + df_energy_for_cost_kwh.between_time("22:00", "23:59")["Energy (kWh)"].sum()
            total_energy_cost_calc = peak_energy_cost_calc + offpeak_energy_cost_calc
            max_demand = df[power_col].rolling('30T', min_periods=1).mean().max()

            tariff_data = {
                "Industrial": [
                    {"Tariff": "E1 - Medium Voltage General", "Voltage": "Medium Voltage", "Base Rate": 0.337, "MD Rate": 29.60, "ICPT": 0.16, "Split": False, "Tiered": False},
                    {"Tariff": "E2 - Medium Voltage Peak/Off-Peak", "Voltage": "Medium Voltage", "Peak Rate": 0.355, "OffPeak Rate": 0.219, "MD Rate": 37.00, "ICPT": 0.16, "Split": True, "Tiered": False},
                    {"Tariff": "E3 - High Voltage Peak/Off-Peak", "Voltage": "High Voltage", "Peak Rate": 0.337, "OffPeak Rate": 0.202, "MD Rate": 35.50, "ICPT": 0.16, "Split": True, "Tiered": False},
                    {"Tariff": "D - Low Voltage Industrial", "Voltage": "Low Voltage", "Tier1 Rate": 0.38, "Tier1 Limit": 200, "Tier2 Rate": 0.441, "MD Rate": 0, "ICPT": 0.027, "Split": False, "Tiered": True}
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

                # kWh Chart (Daily Energy Consumption by Peak and Off-Peak) using detected interval
                df_energy_kwh_viz = df[[power_col]].copy()
                df_energy_kwh_viz["Energy (kWh)"] = df_energy_kwh_viz[power_col] * interval_hours
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
                    # Fallback: search all tariffs for MD Rate if not found above
                    for ind_key_fallback in tariff_data:
                        for t_info_detail_fallback in tariff_data[ind_key_fallback]:
                            if t_info_detail_fallback["Tariff"] == tariff_rate:
                                md_rate_for_selected_tariff = t_info_detail_fallback.get("MD Rate", 0)
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
                    # Use globally detected interval for consistency
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
                else:
                    st.info(f"No peak events (load > {PEAK_THRESHOLD:,.2f} kW) detected in the selected period.")

            # -----------------------------
            # SECTION: Load Duration Curve
            # Main Title: 2. Load Duration Curve
            # - Shows the distribution of load over time, highlighting peak periods.
            # -----------------------------
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
                                                 color_continuous_scale="Viridis",  # Better for accessibility and both light/dark modes
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

        except pd.errors.EmptyDataError:
            st.error("Uploaded Excel file is empty or unreadable.")
        except KeyError as e:
            st.error(f"Column key error: {e}. Check column selection/Excel structure.")
        except ValueError as e:
            st.error(f"Value error: {e}. Check data types/parsing.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error("Ensure Excel file is correctly formatted and columns are selected.")

with tabs[0]:
    show_tnb_tariff_comparison()

with tabs[2]:
    show_advanced_energy_analysis()

with tabs[3]:
    st.title("Monthly Rate Impact Analysis")
    st.markdown("""
    Compare the financial impact of old TNB tariffs vs new RP4 tariffs on a **monthly basis**. 
    This analysis is perfect for understanding cost differences and identifying optimization opportunities 
    when switching from legacy tariffs to the new RP4 structure.
    """)
    
    # Import required modules
    from tariffs.rp4_tariffs import get_tariff_data
    from tariffs.peak_logic import is_peak_rp4
    from utils.cost_calculator import calculate_cost
    from utils.old_cost_calculator import calculate_old_cost
    from old_rate import charging_rates, old_to_new_tariff_map
    
    uploaded_file = st.file_uploader("Upload your data file for monthly analysis", type=["csv", "xls", "xlsx"], key="monthly_file_uploader")
    
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Data Configuration
            st.subheader("Data Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp_col = st.selectbox("Select timestamp column", df.columns, key="monthly_timestamp_col")
                power_col = st.selectbox("Select power (kW) column", 
                                       df.select_dtypes(include='number').columns, key="monthly_power_col")
            
            with col2:
                # Holiday selection for RP4 calculations
                st.markdown("**Public Holidays (for RP4 peak logic)**")
                holidays = set()  # Default empty set
                
                # Simple holiday input - user can add holidays manually
                holiday_input = st.text_area(
                    "Enter holidays (one per line, YYYY-MM-DD format):",
                    placeholder="2024-01-01\n2024-02-10\n2024-04-10",
                    key="monthly_holidays_input"
                )
                
                if holiday_input.strip():
                    try:
                        holiday_lines = [line.strip() for line in holiday_input.split('\n') if line.strip()]
                        holidays = set()
                        for date_str in holiday_lines:
                            holidays.add(pd.to_datetime(date_str).date())
                        st.success(f"Added {len(holidays)} holidays")
                    except Exception as e:
                        st.warning(f"Some holiday dates couldn't be parsed: {e}")
            
            # Process data
            df["Parsed Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.dropna(subset=["Parsed Timestamp"]).set_index("Parsed Timestamp")
            
            if not df.empty and power_col in df.columns:
                
                # === DATA INTERVAL DETECTION ===
                st.subheader("📊 Data Interval Detection")
                
                # Detect data interval from the entire dataset
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        # Get the most common time interval (mode)
                        most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
                        interval_minutes = most_common_interval.total_seconds() / 60
                        interval_hours = most_common_interval.total_seconds() / 3600
                        
                        # Check for consistency
                        unique_intervals = time_diffs.value_counts()
                        consistency_percentage = (unique_intervals.iloc[0] / len(time_diffs)) * 100 if len(unique_intervals) > 0 else 100
                        
                        # Display interval information
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if interval_minutes < 60:
                                st.metric("Detected Interval", f"{interval_minutes:.0f} minutes")
                            else:
                                hours = interval_minutes / 60
                                st.metric("Detected Interval", f"{hours:.1f} hours")
                        
                        with col2:
                            st.metric("Data Consistency", f"{consistency_percentage:.1f}%")
                        
                        with col3:
                            readings_per_hour = 60 / interval_minutes if interval_minutes > 0 else 0
                            st.metric("Readings/Hour", f"{readings_per_hour:.0f}")
                        
                        with col4:
                            daily_readings = readings_per_hour * 24
                            st.metric("Readings/Day", f"{daily_readings:.0f}")
                        
                        # Show interval quality indicator
                        if consistency_percentage >= 95:
                            st.success(f"✅ **Excellent data quality**: {consistency_percentage:.1f}% of readings have consistent {interval_minutes:.0f}-minute intervals")
                        elif consistency_percentage >= 85:
                            st.warning(f"⚠️ **Good data quality**: {consistency_percentage:.1f}% consistent intervals. Some missing data points detected.")
                        else:
                            st.error(f"❌ **Poor data quality**: Only {consistency_percentage:.1f}% consistent intervals. Results may be less accurate.")
                        
                        # Show interval breakdown if there are inconsistencies
                        if consistency_percentage < 95:
                            with st.expander("🔍 View Interval Analysis Details"):
                                st.markdown("**Top 5 Time Intervals Found:**")
                                top_intervals = unique_intervals.head()
                                for interval, count in top_intervals.items():
                                    percentage = (count / len(time_diffs)) * 100
                                    interval_mins = interval.total_seconds() / 60
                                    st.write(f"• **{interval_mins:.0f} minutes**: {count:,} occurrences ({percentage:.1f}%)")
                                
                                st.info("💡 **Tip**: Inconsistent intervals may be due to missing data points or different data collection periods. The analysis will use the most common interval for calculations.")
                    else:
                        # Fallback to 15 minutes if we can't determine interval
                        interval_hours = 0.25
                        interval_minutes = 15
                        st.warning("⚠️ Could not detect data interval automatically. Using default 15-minute intervals.")
                else:
                    interval_hours = 0.25
                    interval_minutes = 15
                    st.warning("⚠️ Insufficient data to detect interval. Using default 15-minute intervals.")
                
                # ===============================
                # DEBUG SECTION FOR ENERGY CALCULATION VERIFICATION
                # ===============================
                with st.expander("🐛 DEBUG: Energy Calculation Verification", expanded=False):
                    st.markdown("### Energy Calculation Debug Information")
                    st.markdown("Use this section to verify that energy calculations are correct for your data interval.")
                    
                    # Sample the first few hours of data for debugging
                    debug_sample_size = min(48, len(df))  # First 48 readings or less
                    df_debug = df.head(debug_sample_size).copy()
                    
                    st.markdown(f"**Analyzing first {len(df_debug)} data points:**")
                    
                    # Debug: Show raw timestamps and calculated intervals
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🕐 Raw Timestamp Analysis:**")
                        debug_timestamps = df_debug.index[:10]  # First 10 timestamps
                        
                        debug_info = []
                        for i, ts in enumerate(debug_timestamps):
                            if i > 0:
                                prev_ts = debug_timestamps[i-1]
                                time_diff = ts - prev_ts
                                diff_minutes = time_diff.total_seconds() / 60
                                debug_info.append({
                                    'Reading #': i+1,
                                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Time Diff (min)': f"{diff_minutes:.1f}" if i > 0 else "N/A",
                                    'Power (kW)': f"{df_debug[power_col].iloc[i]:.2f}"
                                })
                            else:
                                debug_info.append({
                                    'Reading #': i+1,
                                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Time Diff (min)': "N/A",
                                    'Power (kW)': f"{df_debug[power_col].iloc[i]:.2f}"
                                })
                        
                        df_debug_info = pd.DataFrame(debug_info)
                        st.dataframe(df_debug_info, use_container_width=True)
                    
                    with col2:
                        st.markdown("**⚡ Energy Calculation Methods:**")
                        
                        # Method 1: Using detected interval (CORRECT METHOD)
                        energy_method1 = df_debug[power_col] * interval_hours
                        total_energy_method1 = energy_method1.sum()
                        
                        # Method 2: Using diff() method (INCORRECT - what was causing the bug)
                        time_deltas_diff = df_debug.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)
                        energy_method2 = (df_debug[power_col] * time_deltas_diff).sum()
                        
                        # Method 3: Manual calculation for first few readings
                        manual_energy = 0
                        for i in range(1, min(6, len(df_debug))):
                            time_diff_hours = (df_debug.index[i] - df_debug.index[i-1]).total_seconds() / 3600
                            power_avg = (df_debug[power_col].iloc[i] + df_debug[power_col].iloc[i-1]) / 2
                            manual_energy += power_avg * time_diff_hours
                        
                        st.write(f"**Method 1 (CORRECT - Detected Interval):**")
                        st.write(f"• Interval: {interval_minutes:.0f} minutes ({interval_hours:.4f} hours)")
                        st.write(f"• Total Energy: {total_energy_method1:.2f} kWh")
                        st.write(f"• Formula: Power × {interval_hours:.4f}h")
                        
                        st.write(f"**Method 2 (OLD - diff() method):**")
                        st.write(f"• Total Energy: {energy_method2:.2f} kWh")
                        st.write(f"• Formula: Power × diff().hours")
                        
                        st.write(f"**Manual Check (first 5 intervals):**")
                        st.write(f"• Manual Energy: {manual_energy:.2f} kWh")
                        
                        # Show the difference
                        if abs(total_energy_method1 - energy_method2) > 0.01:
                            difference_percentage = abs(total_energy_method1 - energy_method2) / total_energy_method1 * 100
                            st.error(f"⚠️ **Methods differ by {difference_percentage:.1f}%**")
                            st.error(f"Difference: {abs(total_energy_method1 - energy_method2):.2f} kWh")
                        else:
                            st.success("✅ Methods agree within tolerance")
                    
                    # Debug: Show calculation details
                    st.markdown("**📊 Detailed Calculation Breakdown:**")
                    
                    # Create detailed calculation table
                    debug_calc = df_debug.head(10).copy()
                    debug_calc['Energy_Method1'] = debug_calc[power_col] * interval_hours
                    
                    # Calculate time differences for method 2
                    debug_calc['Time_Diff_Hours'] = debug_calc.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)
                    debug_calc['Energy_Method2'] = debug_calc[power_col] * debug_calc['Time_Diff_Hours']
                    
                    debug_display = debug_calc[[power_col, 'Energy_Method1', 'Time_Diff_Hours', 'Energy_Method2']].copy()
                    debug_display.columns = ['Power (kW)', f'Energy Method1 (kWh)', 'Time Diff (h)', 'Energy Method2 (kWh)']
                    debug_display = debug_display.reset_index()
                    debug_display['Timestamp'] = debug_display['Parsed Timestamp'].dt.strftime('%H:%M:%S')
                    debug_display = debug_display[['Timestamp', 'Power (kW)', 'Energy Method1 (kWh)', 'Time Diff (h)', 'Energy Method2 (kWh)']]
                    
                    formatted_debug = debug_display.style.format({
                        'Power (kW)': '{:.2f}',
                        'Energy Method1 (kWh)': '{:.4f}',
                        'Time Diff (h)': '{:.4f}',
                        'Energy Method2 (kWh)': '{:.4f}'
                    })
                    
                    st.dataframe(formatted_debug, use_container_width=True)
                    
                    # Debug: Full dataset comparison
                    st.markdown("**🔬 Full Dataset Energy Comparison:**")
                    
                    # Calculate for entire dataset
                    full_energy_method1 = (df[power_col] * interval_hours).sum()
                    
                    # Old method (with diff())
                    full_time_deltas = df.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)
                    full_energy_method2 = (df[power_col] * full_time_deltas).sum()
                    
                    # Theoretical maximum energy (if all intervals were detected correctly)
                    total_readings = len(df)
                    theoretical_total_hours = total_readings * interval_hours
                    avg_power = df[power_col].mean()
                    theoretical_energy = avg_power * theoretical_total_hours
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Method 1 (Correct)", f"{full_energy_method1:,.2f} kWh")
                        st.caption(f"Using {interval_minutes:.0f}-min intervals")
                    
                    with col2:
                        st.metric("Method 2 (Old diff())", f"{full_energy_method2:,.2f} kWh")
                        difference = full_energy_method1 - full_energy_method2
                        st.caption(f"Difference: {difference:+,.2f} kWh")
                    
                    with col3:
                        st.metric("Theoretical Max", f"{theoretical_energy:,.2f} kWh")
                        st.caption(f"Avg power × total time")
                    
                    # Show the percentage difference
                    if full_energy_method1 > 0:
                        percentage_diff = (full_energy_method2 - full_energy_method1) / full_energy_method1 * 100
                        
                        if abs(percentage_diff) > 5:
                            st.error(f"🚨 **SIGNIFICANT DIFFERENCE**: Method 2 shows {percentage_diff:+.1f}% compared to Method 1")
                            if percentage_diff < -40:
                                st.error("**This explains why 30-minute data shows ~50% less energy!**")
                                st.error("The diff() method creates NaN values and incorrect intervals.")
                        else:
                            st.success(f"✅ **GOOD**: Methods agree within {abs(percentage_diff):.1f}%")
                    
                    # Debug: Show where NaN values occur in diff() method
                    nan_count = full_time_deltas.isna().sum()
                    zero_count = (full_time_deltas == 0).sum()
                    
                    st.markdown("**🐞 Data Quality Analysis:**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Readings", f"{total_readings:,}")
                    col2.metric("NaN Time Diffs", f"{nan_count:,}")
                    col3.metric("Zero Time Diffs", f"{zero_count:,}")
                    col4.metric("Valid Time Diffs", f"{total_readings - nan_count - zero_count:,}")
                    
                    if nan_count > 0 or zero_count > 0:
                        st.warning(f"⚠️ {nan_count + zero_count} readings have invalid time differences (NaN or zero)")
                        st.info("💡 This is why Method 1 (using detected interval) is more reliable than Method 2 (diff())")
                    
                    # Debug: Show summary
                    st.markdown("**📋 Debug Summary:**")
                    
                    if abs(percentage_diff) > 5:
                        st.error(f"""
                        **ISSUE DETECTED:**
                        - Your data has {interval_minutes:.0f}-minute intervals
                        - Method 1 (correct): {full_energy_method1:,.2f} kWh
                        - Method 2 (old diff): {full_energy_method2:,.2f} kWh
                        - Difference: {percentage_diff:+.1f}%
                        
                        **ROOT CAUSE:** The diff() method loses the first reading (becomes NaN) and doesn't handle irregular intervals properly.
                        
                        **SOLUTION:** The app now uses Method 1 everywhere for consistent energy calculations.
                        """)
                    else:
                        st.success(f"""
                        **CALCULATIONS ARE CORRECT:**
                        - Data interval: {interval_minutes:.0f} minutes
                        - Energy calculation: {full_energy_method1:,.2f} kWh
                        - Methods agree within {abs(percentage_diff):.1f}%
                        
                        **STATUS:** Energy calculations are working properly for your data.
                        """)
                
                # ===============================
                # TARIFF SELECTION
                # ===============================
                st.subheader("Tariff Selection")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Old Tariff (Legacy)**")
                    old_tariff_options = list(charging_rates.keys())
                    selected_old_tariff = st.selectbox("Select your current tariff:", old_tariff_options, key="monthly_old_tariff")
                    
                    # Show old tariff details
                    st.info(f"**Rate:** {charging_rates[selected_old_tariff]}")
                
                with col2:
                    st.markdown("**New Tariff (RP4)**")
                    
                    # Get suggested new tariff based on mapping
                    suggested_new = old_to_new_tariff_map.get(selected_old_tariff, "Medium Voltage General")
                    
                    # Load RP4 tariff data
                    rp4_data = get_tariff_data()
                    
                    # User Type Selection
                    user_types = list(rp4_data.keys())
                    selected_user_type = st.selectbox("Select User Type", user_types, 
                                                    index=1 if "Business" in user_types else 0, key="monthly_user_type")
                    
                    # Tariff Group Selection  
                    tariff_groups = list(rp4_data[selected_user_type]["Tariff Groups"].keys())
                    selected_tariff_group = st.selectbox("Select Tariff Group", tariff_groups, key="monthly_tariff_group")
                    
                    # Specific Tariff Selection
                    tariffs = rp4_data[selected_user_type]["Tariff Groups"][selected_tariff_group]["Tariffs"]
                    tariff_names = [t["Tariff"] for t in tariffs]
                    
                    # Try to find suggested tariff in the list
                    suggested_index = 0
                    for i, name in enumerate(tariff_names):
                        if suggested_new.lower() in name.lower():
                            suggested_index = i
                            break
                    
                    selected_new_tariff_name = st.selectbox("Select Specific Tariff", tariff_names, 
                                                          index=suggested_index, key="monthly_new_tariff")
                    
                    # Get the selected new tariff object
                    selected_new_tariff = next((t for t in tariffs if t["Tariff"] == selected_new_tariff_name), None)
                    
                    if selected_new_tariff:
                        # Show new tariff details
                        rates = selected_new_tariff.get('Rates', {})
                        rules = selected_new_tariff.get('Rules', {})
                        tariff_type = selected_new_tariff.get('Type', 'Unknown')
                        
                        # Show rate information with tariff type indicator
                        if rates.get('Peak Rate') and rates.get('OffPeak Rate'):
                            st.info(f"**🕐 TOU Tariff - Peak/Off-Peak Split:**  \n**Peak:** RM {rates['Peak Rate']:.4f}/kWh  \n**Off-Peak:** RM {rates['OffPeak Rate']:.4f}/kWh  \n**Capacity:** RM {rates.get('Capacity Rate', 0):.2f}/kW  \n**Network:** RM {rates.get('Network Rate', 0):.2f}/kW")
                        else:
                            st.info(f"**⚡ General Tariff - Single Rate:**  \n**Energy:** RM {rates.get('Energy Rate', 0):.4f}/kWh  \n**Capacity:** RM {rates.get('Capacity Rate', 0):.2f}/kW  \n**Network:** RM {rates.get('Network Rate', 0):.2f}/kW")
                        
                        # Add explanation about TOU vs General tariffs
                        if tariff_type == "General":
                            st.warning("""
                            ⚠️ **Note**: This is a **General tariff** (single rate for all periods). 
                            In the breakdown tables, all energy will appear as "Peak Energy", 
                            and "Off-Peak Energy" will show as 0. This represents that all energy 
                            is treated as primary/main energy with a single rate.
                            
                            💡 **To see time-based peak/off-peak breakdown**: Select a **TOU (Time-of-Use)** tariff instead.
                            """)
                        elif tariff_type == "TOU":
                            st.success("""
                            ✅ **TOU Tariff Selected**: This tariff uses different rates for peak and off-peak periods. 
                            You will see separate peak/off-peak energy and cost breakdowns in the analysis.
                            """)
                
                # ===============================
                # MONTHLY TARIFF IMPACT ANALYSIS
                # ===============================
                st.subheader("Monthly Rate Impact Analysis")
                st.caption("Comprehensive comparison between legacy tariff and new RP4 tariff structure")
                
                # Brief info about calculation methodology
                with st.expander("ℹ️ About the Calculation", expanded=False):
                    st.markdown("""
                    **Energy Calculation Method:**
                    - **Auto-detects data interval** (15min, 30min, 1hr, etc.) from your timestamps
                    - **Accurate energy calculation**: Power (kW) × Time Interval (hours) = Energy (kWh)
                    - **Peak period logic**: Monday-Friday 8AM-10PM for legacy tariffs
                    - **RP4 logic**: Holiday-aware peak/off-peak classification for new tariffs
                    - **Validation**: Ensures Peak + Off-Peak = Total Energy for accuracy
                    """)
                
                # Check data timespan
                min_date = df.index.min()
                max_date = df.index.max()
                total_days = (max_date - min_date).days
                
                # Display data period info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Start", min_date.strftime('%Y-%m-%d'))
                with col2:
                    st.metric("Data End", max_date.strftime('%Y-%m-%d'))
                with col3:
                    st.metric("Total Days", f"{total_days} days")
                
                # Group data by month for analysis
                df['Year_Month'] = df.index.to_period('M')
                monthly_groups = df.groupby('Year_Month')
                
                # Initialize monthly analysis results
                monthly_analysis = []
                
                # Progress indicator for monthly calculations
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_months = len(monthly_groups)
                
                for idx, (month_period, month_data) in enumerate(monthly_groups):
                    # Update progress
                    progress_percentage = (idx + 1) / total_months
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Processing {month_period}... ({idx + 1}/{total_months})")
                    
                    # Skip months with insufficient data
                    if len(month_data) < 24:
                        continue
                    
                    # === ENERGY CALCULATION ===
                    month_str = str(month_period)
                    
                    # Calculate energy consumption using the globally detected interval
                    if len(month_data) > 1:
                        # Use the globally detected interval_hours for consistent calculations
                        # This ensures all months use the same time interval
                        
                        # Calculate energy: Power (kW) × Time (hours) = Energy (kWh)
                        # Each power reading represents the average power for that interval
                        energy_per_reading = month_data[power_col] * interval_hours
                        
                        # Basic energy metrics
                        total_energy_kwh = energy_per_reading.sum()
                        max_demand_kw = month_data[power_col].max()
                        avg_demand_kw = month_data[power_col].mean()
                        
                        # === OLD TARIFF PEAK/OFF-PEAK LOGIC ===
                        # Traditional TNB logic: Mon-Fri 8AM-10PM = Peak
                        old_peak_mask = month_data.index.to_series().apply(
                            lambda ts: ts.weekday() < 5 and 8 <= ts.hour < 22
                        )
                        old_peak_energy = energy_per_reading[old_peak_mask].sum()
                        offpeak_energy = energy_per_reading[~old_peak_mask].sum()
                        
                        # Validation check
                        energy_total_check = old_peak_energy + offpeak_energy
                        if abs(energy_total_check - total_energy_kwh) > 0.01:
                            st.error(f"Energy calculation error for {month_str}: {energy_total_check:.2f} vs {total_energy_kwh:.2f}")
                            continue
                        

                        
                        # === OLD TARIFF COST CALCULATION ===
                        old_tariff_result = calculate_old_cost(
                            tariff_name=selected_old_tariff,
                            total_kwh=total_energy_kwh,
                            max_demand_kw=max_demand_kw,
                            peak_kwh=old_peak_energy,
                            offpeak_kwh=offpeak_energy,
                            icpt=0.16
                        )
                        
                        # === NEW TARIFF (RP4) COST CALCULATION ===
                        month_data_for_rp4 = month_data.reset_index()
                        rp4_result = calculate_cost(
                            df=month_data_for_rp4,
                            tariff=selected_new_tariff,
                            power_col=power_col,
                            holidays=holidays,
                            afa_rate=global_afa_rate
                        )
                        
                        # === EXTRACT COST COMPONENTS ===
                        # Old Tariff Components
                        old_total_cost = old_tariff_result.get('Total Cost', 0)
                        old_peak_cost = old_tariff_result.get('Peak Energy Cost', 0)
                        old_offpeak_cost = old_tariff_result.get('Off-Peak Energy Cost', 0)
                        old_energy_cost = old_peak_cost + old_offpeak_cost
                        old_icpt_cost = old_tariff_result.get('ICPT Cost', 0)
                        old_md_cost = old_tariff_result.get('MD Cost', 0)
                        
                        # RP4 Tariff Components
                        rp4_total_cost = rp4_result.get('Total Cost', 0)
                        
                        # Handle TOU vs General RP4 tariffs
                        if 'Peak kWh' in rp4_result and selected_new_tariff.get('Rules', {}).get('has_peak_split', False):
                            # Time-of-Use RP4 tariff (true peak/off-peak split)
                            rp4_peak_energy = rp4_result.get('Peak kWh', 0)
                            rp4_offpeak_energy = rp4_result.get('Off-Peak kWh', 0)
                            rp4_peak_cost = rp4_result.get('Peak Energy Cost', 0)
                            rp4_offpeak_cost = rp4_result.get('Off-Peak Energy Cost', 0)
                            rp4_energy_cost = rp4_peak_cost + rp4_offpeak_cost
                        else:
                            # General RP4 tariff - now shows all energy as "Peak Energy" for better UX
                            # All energy is treated as primary/main energy rather than "off-peak"
                            rp4_peak_energy = rp4_result.get('Peak kWh', total_energy_kwh)
                            rp4_offpeak_energy = rp4_result.get('Off-Peak kWh', 0)
                            rp4_peak_cost = rp4_result.get('Peak Energy Cost', rp4_result.get('Energy Cost (RM)', 0))
                            rp4_offpeak_cost = rp4_result.get('Off-Peak Energy Cost', 0)
                            rp4_energy_cost = rp4_peak_cost + rp4_offpeak_cost
                        
                        rp4_afa_cost = rp4_result.get('AFA Adjustment', 0)
                        rp4_capacity_cost = rp4_result.get('Capacity Cost', rp4_result.get('Capacity Cost (RM)', 0))
                        rp4_network_cost = rp4_result.get('Network Cost', rp4_result.get('Network Cost (RM)', 0))
                        rp4_demand_cost = rp4_capacity_cost + rp4_network_cost
                        
                        # === COST COMPARISON METRICS ===
                        cost_difference = rp4_total_cost - old_total_cost
                        percentage_change = (cost_difference / old_total_cost * 100) if old_total_cost > 0 else 0
                        
                        # Determine status
                        if cost_difference < -5:
                            status = "🟢 Significant Savings"
                        elif cost_difference < 0:
                            status = "🟡 Minor Savings"
                        elif cost_difference < 5:
                            status = "🟠 Roughly Equal"
                        else:
                            status = "🔴 Higher Cost"
                        
                        # === EFFICIENCY METRICS ===
                        old_cost_per_kwh = old_total_cost / total_energy_kwh if total_energy_kwh > 0 else 0
                        rp4_cost_per_kwh = rp4_total_cost / total_energy_kwh if total_energy_kwh > 0 else 0
                        
                        old_peak_percentage = (old_peak_energy / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
                        rp4_peak_percentage = (rp4_peak_energy / total_energy_kwh * 100) if total_energy_kwh > 0 else 0
                        
                        # === COMPILE MONTHLY RESULT ===
                        monthly_analysis.append({
                            # Basic Information
                            'Month': month_str,
                            'Total Energy (kWh)': total_energy_kwh,
                            'Max Demand (kW)': max_demand_kw,
                            'Avg Demand (kW)': avg_demand_kw,
                            
                            # Old Tariff Structure
                            'Old Peak Energy (kWh)': old_peak_energy,
                            'Old Off-Peak Energy (kWh)': offpeak_energy,
                            'Old Peak %': old_peak_percentage,
                            'Old Peak Cost (RM)': old_peak_cost,
                            'Old Off-Peak Cost (RM)': old_offpeak_cost,
                            'Old Energy Cost (RM)': old_energy_cost,
                            'Old ICPT Cost (RM)': old_icpt_cost,
                            'Old MD Cost (RM)': old_md_cost,
                            'Old Total Cost (RM)': old_total_cost,
                            'Old Cost/kWh (RM)': old_cost_per_kwh,
                            
                            # RP4 Tariff Structure
                            'RP4 Peak Energy (kWh)': rp4_peak_energy,
                            'RP4 Off-Peak Energy (kWh)': rp4_offpeak_energy,
                            'RP4 Peak %': rp4_peak_percentage,
                            'RP4 Peak Cost (RM)': rp4_peak_cost,
                            'RP4 Off-Peak Cost (RM)': rp4_offpeak_cost,
                            'RP4 Energy Cost (RM)': rp4_energy_cost,
                            'RP4 AFA Cost (RM)': rp4_afa_cost,
                            'RP4 Demand Cost (RM)': rp4_demand_cost,
                            'RP4 Total Cost (RM)': rp4_total_cost,
                            'RP4 Cost/kWh (RM)': rp4_cost_per_kwh,
                            
                            # Comparison Metrics
                            'Cost Difference (RM)': cost_difference,
                            'Change (%)': percentage_change,
                            'Status': status,
                            'Energy Cost Difference (RM)': rp4_energy_cost - old_energy_cost,
                            'Demand Cost Difference (RM)': rp4_demand_cost - (old_icpt_cost + old_md_cost),
                            'Peak Classification Difference (%)': rp4_peak_percentage - old_peak_percentage
                        })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # ===============================
                # DEBUG SECTION FOR RP4 CALCULATION ISSUES
                # ===============================
                with st.expander("🐛 DEBUG: RP4 Peak Energy & Cost Calculation", expanded=False):
                    st.markdown("### RP4 Calculation Debug Information")
                    st.markdown("Use this section to debug why RP4 peak energy/cost values might be missing or zero.")
                    
                    if monthly_analysis:
                        debug_month = st.selectbox("Select month to debug:", [m['Month'] for m in monthly_analysis])
                        debug_data = next((m for m in monthly_analysis if m['Month'] == debug_month), None)
                        
                        if debug_data:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📊 Raw Values from Calculation:**")
                                st.write(f"RP4 Peak Energy: {debug_data['RP4 Peak Energy (kWh)']:.2f} kWh")
                                st.write(f"RP4 Off-Peak Energy: {debug_data['RP4 Off-Peak Energy (kWh)']:.2f} kWh")
                                st.write(f"RP4 Peak Cost: RM {debug_data['RP4 Peak Cost (RM)']:.2f}")
                                st.write(f"RP4 Off-Peak Cost: RM {debug_data['RP4 Off-Peak Cost (RM)']:.2f}")
                                st.write(f"Total Energy: {debug_data['Total Energy (kWh)']:.2f} kWh")
                            
                            with col2:
                                st.markdown("**🔍 Selected Tariff Details:**")
                                st.write(f"Selected Tariff: {selected_new_tariff_name}")
                                st.write(f"Tariff Type: {selected_new_tariff.get('Type', 'Unknown')}")
                                st.write(f"Has Peak Split: {selected_new_tariff.get('Rules', {}).get('has_peak_split', False)}")
                                
                                rates = selected_new_tariff.get('Rates', {})
                                st.write(f"Peak Rate: RM {rates.get('Peak Rate', 'N/A')}/kWh")
                                st.write(f"OffPeak Rate: RM {rates.get('OffPeak Rate', 'N/A')}/kWh")
                                st.write(f"Energy Rate: RM {rates.get('Energy Rate', 'N/A')}/kWh")
                            
                            # Test RP4 calculation for debug month
                            month_period = pd.Period(debug_month)
                            month_data = df[df['Year_Month'] == month_period]
                            
                            if not month_data.empty:
                                st.markdown("**🧪 Test RP4 Calculation:**")
                                month_data_for_test = month_data.reset_index()
                                test_result = calculate_cost(
                                    df=month_data_for_test,
                                    tariff=selected_new_tariff,
                                    power_col=power_col,
                                    holidays=holidays,
                                    afa_rate=global_afa_rate
                                )
                                
                                st.write("**Raw RP4 calculation result:**")
                                for key, value in test_result.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"- {key}: {value:.4f}")
                                    else:
                                        st.write(f"- {key}: {value}")
                                
                                # Check if this is a TOU tariff
                                if 'Peak kWh' in test_result:
                                    st.success("✅ This is a TOU tariff - peak/off-peak values should be present")
                                    st.write(f"Peak kWh detected: {test_result.get('Peak kWh', 0):.2f}")
                                    st.write(f"Off-Peak kWh detected: {test_result.get('Off-Peak kWh', 0):.2f}")
                                else:
                                    st.warning("⚠️ This appears to be a General tariff - no peak/off-peak split")
                                
                                # Show peak/off-peak logic test
                                st.markdown("**⏰ Peak/Off-Peak Logic Test:**")
                                sample_timestamps = month_data.index[:10]
                                peak_test_results = []
                                for ts in sample_timestamps:
                                    is_peak = is_peak_rp4(ts, holidays)
                                    peak_test_results.append({
                                        'Timestamp': ts.strftime('%Y-%m-%d %H:%M'),
                                        'Weekday': ts.strftime('%A'),
                                        'Hour': ts.hour,
                                        'Is Peak': is_peak
                                    })
                                
                                df_peak_test = pd.DataFrame(peak_test_results)
                                st.dataframe(df_peak_test, use_container_width=True)
                    else:
                        st.warning("No monthly analysis data available for debugging.")
                
                if monthly_analysis:
                    # ===============================
                    # SUMMARY METRICS
                    # ===============================
                    df_monthly = pd.DataFrame(monthly_analysis)
                    
                    total_old_cost = df_monthly['Old Total Cost (RM)'].sum()
                    total_new_cost = df_monthly['RP4 Total Cost (RM)'].sum()
                    total_difference = total_new_cost - total_old_cost
                    avg_percentage_change = df_monthly['Change (%)'].mean()
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Old Cost", f"RM {total_old_cost:,.2f}")
                    col2.metric("Total New Cost", f"RM {total_new_cost:,.2f}")
                    col3.metric("Total Difference", f"RM {total_difference:,.2f}", f"{avg_percentage_change:+.1f}%")
                    
                    if total_difference < 0:
                        col4.success(f"💰 Annual Savings: RM {abs(total_difference):,.2f}")
                    elif total_difference > 0:
                        col4.error(f"📈 Annual Increase: RM {total_difference:,.2f}")
                    else:
                        col4.info("➖ No Change")
                    
                    # ===============================
                    # DETAILED COST BREAKDOWN
                    # ===============================
                    st.subheader("Detailed Cost Breakdown")
                    
                    # Create tabs for detailed breakdown
                    breakdown_tabs = st.tabs(["📊 Summary Comparison", "🔴 Old Tariff Details", "🔵 New Tariff (RP4) Details"])
                    
                    with breakdown_tabs[0]:
                        st.markdown("#### Summary Cost Comparison")
                        
                        # Summary comparison table
                        summary_columns = ['Month', 'Total Energy (kWh)', 'Max Demand (kW)', 
                                         'Old Total Cost (RM)', 'RP4 Total Cost (RM)', 
                                         'Cost Difference (RM)', 'Change (%)', 'Status']
                        df_summary = df_monthly[summary_columns].copy()
                        
                        formatted_summary = df_summary.style.format({
                            'Total Energy (kWh)': '{:,.0f}',
                            'Max Demand (kW)': '{:,.2f}',
                            'Old Total Cost (RM)': 'RM {:,.2f}',
                            'RP4 Total Cost (RM)': 'RM {:,.2f}',
                            'Cost Difference (RM)': 'RM {:+,.2f}',
                            'Change (%)': '{:+.1f}%'
                        }).apply(lambda x: ['background-color: rgba(40, 167, 69, 0.2)' if v < 0 else 'background-color: rgba(220, 53, 69, 0.2)' if v > 0 else '' 
                                         for v in df_summary['Cost Difference (RM)']], axis=0)
                        
                        st.dataframe(formatted_summary, use_container_width=True)
                    
                    with breakdown_tabs[1]:
                        st.markdown("#### Old Tariff Structure Breakdown")
                        st.caption("Legacy TNB tariff with traditional peak/off-peak logic")
                        
                        # Old tariff breakdown table
                        old_columns = ['Month', 'Old Peak Energy (kWh)', 'Old Off-Peak Energy (kWh)',
                                     'Old Peak Cost (RM)', 'Old Off-Peak Cost (RM)', 
                                     'Old ICPT Cost (RM)', 'Old MD Cost (RM)', 'Old Total Cost (RM)']
                        df_old = df_monthly[old_columns].copy()
                        
                        formatted_old = df_old.style.format({
                            'Old Peak Energy (kWh)': '{:,.0f}',
                            'Old Off-Peak Energy (kWh)': '{:,.0f}',
                            'Old Peak Cost (RM)': 'RM {:,.2f}',
                            'Old Off-Peak Cost (RM)': 'RM {:,.2f}',
                            'Old ICPT Cost (RM)': 'RM {:,.2f}',
                            'Old MD Cost (RM)': 'RM {:,.2f}',
                            'Old Total Cost (RM)': 'RM {:,.2f}'
                        })
                        
                        st.dataframe(formatted_old, use_container_width=True)
                        
                        # Old tariff metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Peak Energy", f"{df_monthly['Old Peak Energy (kWh)'].sum():,.0f} kWh")
                        col2.metric("Total Off-Peak Energy", f"{df_monthly['Old Off-Peak Energy (kWh)'].sum():,.0f} kWh")
                        col3.metric("Total Energy Cost", f"RM {(df_monthly['Old Peak Cost (RM)'].sum() + df_monthly['Old Off-Peak Cost (RM)'].sum()):,.2f}")
                        col4.metric("Total ICPT + MD", f"RM {(df_monthly['Old ICPT Cost (RM)'].sum() + df_monthly['Old MD Cost (RM)'].sum()):,.2f}")
                    
                    with breakdown_tabs[2]:
                        st.markdown("#### New RP4 Tariff Structure Breakdown")
                        tariff_type = selected_new_tariff.get('Type', 'Unknown')
                        if tariff_type == "General":
                            st.caption("⚡ General RP4 tariff - Single rate for all periods (All energy shows as Peak Energy for clarity)")
                        else:
                            st.caption("🕐 TOU RP4 tariff with holiday-aware peak/off-peak logic and separate capacity/network rates")
                        
                        # New tariff breakdown table
                        new_columns = ['Month', 'RP4 Peak Energy (kWh)', 'RP4 Off-Peak Energy (kWh)',
                                     'RP4 Peak Cost (RM)', 'RP4 Off-Peak Cost (RM)', 
                                     'RP4 AFA Cost (RM)', 'RP4 Demand Cost (RM)', 'RP4 Total Cost (RM)']
                        df_new = df_monthly[new_columns].copy()
                        
                        formatted_new = df_new.style.format({
                            'RP4 Peak Energy (kWh)': '{:,.0f}',
                            'RP4 Off-Peak Energy (kWh)': '{:,.0f}',
                            'RP4 Peak Cost (RM)': 'RM {:,.2f}',
                            'RP4 Off-Peak Cost (RM)': 'RM {:,.2f}',
                            'RP4 AFA Cost (RM)': 'RM {:,.2f}',
                            'RP4 Demand Cost (RM)': 'RM {:,.2f}',
                            'RP4 Total Cost (RM)': 'RM {:,.2f}'
                        })
                        
                        st.dataframe(formatted_new, use_container_width=True)
                        
                        # New tariff metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Peak Energy", f"{df_monthly['RP4 Peak Energy (kWh)'].sum():,.0f} kWh")
                        col2.metric("Total Off-Peak Energy", f"{df_monthly['RP4 Off-Peak Energy (kWh)'].sum():,.0f} kWh")
                        col3.metric("Total Energy Cost", f"RM {(df_monthly['RP4 Peak Cost (RM)'].sum() + df_monthly['RP4 Off-Peak Cost (RM)'].sum()):,.2f}")
                        col4.metric("Total AFA + MD", f"RM {(df_monthly['RP4 AFA Cost (RM)'].sum() + df_monthly['RP4 Demand Cost (RM)'].sum()):,.2f}")
                    
                    # ===============================
                    # SIDE-BY-SIDE DETAILED BREAKDOWN (NEW)
                    # ===============================
                    st.subheader("📋 Side-by-Side Monthly Breakdown")
                    st.caption("Compare old and new tariff details month by month")
                    
                    # Create side-by-side comparison for each month
                    for _, row in df_monthly.iterrows():
                        month = row['Month']
                        
                        with st.expander(f"📅 {month} - Detailed Side-by-Side Comparison", expanded=False):
                            # Month summary at the top
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Energy", f"{row['Total Energy (kWh)']:,.0f} kWh")
                            col2.metric("Max Demand", f"{row['Max Demand (kW)']:,.2f} kW")
                            col3.metric("Cost Difference", f"RM {row['Cost Difference (RM)']:+,.2f}")
                            col4.metric("Change", f"{row['Change (%)']:+.1f}%")
                            
                            st.markdown("---")
                            
                            # Side-by-side tariff comparison
                            col_old, col_new = st.columns(2)
                            
                            with col_old:
                                st.markdown("### 🔴 Old Tariff Details")
                                st.markdown(f"**{selected_old_tariff}**")
                                st.markdown(f"*{charging_rates[selected_old_tariff]}*")
                                
                                # Old tariff breakdown table
                                old_breakdown = pd.DataFrame({
                                    'Component': [
                                        'Peak Energy (kWh)',
                                        'Off-Peak Energy (kWh)', 
                                        'Peak Energy Cost (RM)',
                                        'Off-Peak Energy Cost (RM)',
                                        'ICPT Cost (RM)',
                                        'MD Cost (RM)',
                                        'Total Cost (RM)'
                                    ],
                                    'Value': [
                                        f"{row['Old Peak Energy (kWh)']:,.0f}",
                                        f"{row['Old Off-Peak Energy (kWh)']:,.0f}",
                                        f"RM {row['Old Peak Cost (RM)']:,.2f}",
                                        f"RM {row['Old Off-Peak Cost (RM)']:,.2f}",
                                        f"RM {row['Old ICPT Cost (RM)']:,.2f}",
                                        f"RM {row['Old MD Cost (RM)']:,.2f}",
                                        f"RM {row['Old Total Cost (RM)']:,.2f}"
                                    ]
                                })
                                
                                # Style the old tariff table
                                styled_old = old_breakdown.style.apply(
                                    lambda x: ['background-color: rgba(255, 107, 107, 0.1)' if 'Total Cost' in x['Component'] 
                                             else 'background-color: rgba(255, 107, 107, 0.05)' for _ in x], axis=1
                                )
                                st.dataframe(styled_old, use_container_width=True, hide_index=True)
                                
                                # Old tariff metrics
                                st.metric("Cost per kWh", f"RM {row['Old Cost/kWh (RM)']:.4f}")
                                st.metric("Peak Energy %", f"{row['Old Peak %']:.1f}%")
                            
                            with col_new:
                                st.markdown("### 🔵 New Tariff (RP4) Details")
                                st.markdown(f"**{selected_new_tariff_name}**")
                                
                                # Show tariff type and rates
                                tariff_type = selected_new_tariff.get('Type', 'Unknown')
                                rates = selected_new_tariff.get('Rates', {})
                                
                                if tariff_type == "TOU":
                                    rate_info = f"*Peak: RM {rates.get('Peak Rate', 0):.4f}/kWh, Off-Peak: RM {rates.get('OffPeak Rate', 0):.4f}/kWh, Capacity: RM {rates.get('Capacity Rate', 0):.2f}/kW, Network: RM {rates.get('Network Rate', 0):.2f}/kW*"
                                else:
                                    rate_info = f"*Energy: RM {rates.get('Energy Rate', 0):.4f}/kWh (General tariff), Capacity: RM {rates.get('Capacity Rate', 0):.2f}/kW, Network: RM {rates.get('Network Rate', 0):.2f}/kW*"
                                
                                st.markdown(rate_info)
                                
                                # New tariff breakdown table
                                new_breakdown = pd.DataFrame({
                                    'Component': [
                                        'Peak Energy (kWh)',
                                        'Off-Peak Energy (kWh)',
                                        'Peak Energy Cost (RM)',
                                        'Off-Peak Energy Cost (RM)',
                                        'AFA Cost (RM)',
                                        'Demand Cost (RM)',
                                        'Total Cost (RM)'
                                    ],
                                    'Value': [
                                        f"{row['RP4 Peak Energy (kWh)']:,.0f}",
                                        f"{row['RP4 Off-Peak Energy (kWh)']:,.0f}",
                                        f"RM {row['RP4 Peak Cost (RM)']:,.2f}",
                                        f"RM {row['RP4 Off-Peak Cost (RM)']:,.2f}",
                                        f"RM {row['RP4 AFA Cost (RM)']:,.2f}",
                                        f"RM {row['RP4 Demand Cost (RM)']:,.2f}",
                                        f"RM {row['RP4 Total Cost (RM)']:,.2f}"
                                    ]
                                })
                                
                                # Style the new tariff table
                                styled_new = new_breakdown.style.apply(
                                    lambda x: ['background-color: rgba(78, 205, 196, 0.1)' if 'Total Cost' in x['Component'] 
                                             else 'background-color: rgba(78, 205, 196, 0.05)' for _ in x], axis=1
                                )
                                st.dataframe(styled_new, use_container_width=True, hide_index=True)
                                
                                # New tariff metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Peak Energy", f"{df_monthly['RP4 Peak Energy (kWh)'].sum():,.0f} kWh")
                                col2.metric("Total Off-Peak Energy", f"{df_monthly['RP4 Off-Peak Energy (kWh)'].sum():,.0f} kWh")
                                col3.metric("Total Energy Cost", f"RM {(df_monthly['RP4 Peak Cost (RM)'].sum() + df_monthly['RP4 Off-Peak Cost (RM)'].sum()):,.2f}")
                                col4.metric("Total AFA + MD", f"RM {(df_monthly['RP4 AFA Cost (RM)'].sum() + df_monthly['RP4 Demand Cost (RM)'].sum()):,.2f}")
                    
                    # ===============================
                    # COMPONENT-WISE COMPARISON CHARTS
                    # ===============================
                    st.subheader("Component-wise Cost Analysis")
                    
                    # Create comparison charts
                    chart_tabs = st.tabs(["Energy Costs", "Additional Charges", "Peak vs Off-Peak Energy"])
                    
                    with chart_tabs[0]:
                        # Energy cost comparison
                        energy_comparison_data = []
                        for _, row in df_monthly.iterrows():
                            energy_comparison_data.extend([
                                {'Month': row['Month'], 'Tariff': 'Old', 'Component': 'Peak Energy', 'Cost': row['Old Peak Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'Old', 'Component': 'Off-Peak Energy', 'Cost': row['Old Off-Peak Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'New (RP4)', 'Component': 'Peak Energy', 'Cost': row['RP4 Peak Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'New (RP4)', 'Component': 'Off-Peak Energy', 'Cost': row['RP4 Off-Peak Cost (RM)']}
                            ])
                        
                        df_energy_comp = pd.DataFrame(energy_comparison_data)
                        fig_energy = px.bar(df_energy_comp, x='Month', y='Cost', color='Component', 
                                          facet_col='Tariff', title='Energy Cost Comparison: Peak vs Off-Peak',
                                          labels={'Cost': 'Cost (RM)'})
                        fig_energy.update_layout(height=500)
                        st.plotly_chart(fig_energy, use_container_width=True)
                    
                    with chart_tabs[1]:
                        # Additional charges comparison (ICPT/AFA + MD)
                        charges_comparison_data = []
                        for _, row in df_monthly.iterrows():
                            charges_comparison_data.extend([
                                {'Month': row['Month'], 'Tariff': 'Old', 'Component': 'ICPT', 'Cost': row['Old ICPT Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'Old', 'Component': 'MD', 'Cost': row['Old MD Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'New (RP4)', 'Component': 'AFA', 'Cost': row['RP4 AFA Cost (RM)']},
                                {'Month': row['Month'], 'Tariff': 'New (RP4)', 'Component': 'MD (Capacity+Network)', 'Cost': row['RP4 Demand Cost (RM)']}
                            ])
                        
                        df_charges_comp = pd.DataFrame(charges_comparison_data)
                        fig_charges = px.bar(df_charges_comp, x='Month', y='Cost', color='Component', 
                                           facet_col='Tariff', title='Additional Charges Comparison: ICPT/AFA and MD',
                                           labels={'Cost': 'Cost (RM)'})
                        fig_charges.update_layout(height=500)
                        st.plotly_chart(fig_charges, use_container_width=True)
                    
                    with chart_tabs[2]:
                        # Peak vs Off-Peak energy consumption comparison
                        energy_kwh_data = []
                        for _, row in df_monthly.iterrows():
                            energy_kwh_data.extend([
                                {'Month': row['Month'], 'Tariff': 'Old Logic', 'Period': 'Peak', 'Energy (kWh)': row['Old Peak Energy (kWh)']},
                                {'Month': row['Month'], 'Tariff': 'Old Logic', 'Period': 'Off-Peak', 'Energy (kWh)': row['Old Off-Peak Energy (kWh)']},
                                {'Month': row['Month'], 'Tariff': 'RP4 Logic', 'Period': 'Peak', 'Energy (kWh)': row['RP4 Peak Energy (kWh)']},
                                {'Month': row['Month'], 'Tariff': 'RP4 Logic', 'Period': 'Off-Peak', 'Energy (kWh)': row['RP4 Off-Peak Energy (kWh)']}
                            ])
                        
                        df_energy_kwh = pd.DataFrame(energy_kwh_data)
                        fig_energy_kwh = px.bar(df_energy_kwh, x='Month', y='Energy (kWh)', color='Period', 
                                              facet_col='Tariff', title='Energy Consumption: Old vs RP4 Peak/Off-Peak Logic',
                                              labels={'Energy (kWh)': 'Energy Consumption (kWh)'})
                        fig_energy_kwh.update_layout(height=500)
                        st.plotly_chart(fig_energy_kwh, use_container_width=True)
                        
                        # Show the difference in peak/off-peak classification
                        st.markdown("#### Peak/Off-Peak Classification Differences")
                        classification_data = []
                        for _, row in df_monthly.iterrows():
                            peak_diff = row['RP4 Peak Energy (kWh)'] - row['Old Peak Energy (kWh)']
                            offpeak_diff = row['RP4 Off-Peak Energy (kWh)'] - row['Old Off-Peak Energy (kWh)']
                            classification_data.append({
                                'Month': row['Month'],
                                'Peak Energy Difference (kWh)': peak_diff,
                                'Off-Peak Energy Difference (kWh)': offpeak_diff,
                                'Net Classification Change': '⬆️ More Peak' if peak_diff > 0 else '⬇️ More Off-Peak' if peak_diff < 0 else '➖ Same'
                            })
                        
                        df_classification = pd.DataFrame(classification_data)
                        formatted_classification = df_classification.style.format({
                            'Peak Energy Difference (kWh)': '{:+,.0f}',
                            'Off-Peak Energy Difference (kWh)': '{:+,.0f}'
                        })
                        st.dataframe(formatted_classification, use_container_width=True)
                        
                        st.info("""
                        **Note:** Differences in peak/off-peak energy classification between old and RP4 logic are due to:
                        - **Holiday consideration**: RP4 treats public holidays as off-peak periods
                        - **Weekend classification**: Different treatment of Saturday/Sunday periods
                        - **Time boundary differences**: Slight variations in peak period definitions
                        """)

                    # ===============================
                    # MONTHLY BREAKDOWN TABLE (ORIGINAL)
                    # ===============================
                    st.subheader("Monthly Summary Table")
                    
                    # Format the dataframe for display (simplified summary)
                    summary_display_columns = ['Month', 'Total Energy (kWh)', 'Max Demand (kW)', 
                                             'Old Total Cost (RM)', 'RP4 Total Cost (RM)', 
                                             'Cost Difference (RM)', 'Change (%)', 'Status']
                    df_display = df_monthly[summary_display_columns].copy()
                    formatted_df = df_display.style.format({
                        'Total Energy (kWh)': '{:,.0f}',
                        'Max Demand (kW)': '{:,.2f}',
                        'Old Total Cost (RM)': 'RM {:,.2f}',
                        'RP4 Total Cost (RM)': 'RM {:,.2f}',
                        'Cost Difference (RM)': 'RM {:+,.2f}',
                        'Change (%)': '{:+.1f}%'
                    }).apply(lambda x: ['background-color: rgba(40, 167, 69, 0.2)' if v < 0 else 'background-color: rgba(220, 53, 69, 0.2)' if v > 0 else '' 
                                     for v in df_display['Cost Difference (RM)']], axis=0)
                    
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # ===============================
                    # VISUALIZATIONS
                    # ===============================
                    st.subheader("Visual Analysis")
                    
                    # Cost comparison chart
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        name='Old Tariff',
                        x=df_monthly['Month'],
                        y=df_monthly['Old Total Cost (RM)'],
                        marker_color='#FF6B6B'  # Modern coral - works in both light/dark
                    ))
                    fig_cost.add_trace(go.Bar(
                        name='New Tariff (RP4)',
                        x=df_monthly['Month'],
                        y=df_monthly['RP4 Total Cost (RM)'],
                        marker_color='#4ECDC4'  # Modern teal - works in both light/dark
                    ))
                    
                    fig_cost.update_layout(
                        title='Monthly Cost Comparison: Old vs New Tariff',
                        xaxis_title='Month',
                        yaxis_title='Cost (RM)',
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                    
                    # Difference trend chart
                    fig_diff = px.line(df_monthly, x='Month', y='Cost Difference (RM)', 
                                     title='Monthly Cost Difference Trend',
                                     labels={'Cost Difference (RM)': 'Cost Difference (RM)'})
                    fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", 
                                     annotation_text="Break-even")
                    fig_diff.update_traces(mode="markers+lines", marker=dict(size=8))
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # ===============================
                    # INSIGHTS & RECOMMENDATIONS
                    # ===============================
                    st.subheader("Key Insights & Recommendations")
                    
                    # Calculate insights
                    months_with_savings = len(df_monthly[df_monthly['Cost Difference (RM)'] < 0])
                    months_with_increase = len(df_monthly[df_monthly['Cost Difference (RM)'] > 0])
                    total_months = len(df_monthly)
                    avg_monthly_difference = df_monthly['Cost Difference (RM)'].mean()
                    
                    # Display insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Performance Summary:**")
                        st.write(f"• **{months_with_savings}/{total_months}** months show savings")
                        st.write(f"• **{months_with_increase}/{total_months}** months show cost increase")
                        st.write(f"• **Average monthly difference:** RM {avg_monthly_difference:+,.2f}")
                        
                        if avg_monthly_difference < -10:
                            st.success("🎯 **Strong case for switching to RP4 tariff**")
                        elif avg_monthly_difference < 0:
                            st.info("💡 **Moderate savings with RP4 tariff**")
                        elif avg_monthly_difference > 10:
                            st.warning("⚠️ **RP4 may be more expensive - review strategy**")
                        else:
                            st.info("📊 **Costs are roughly equivalent**")
                    
                    with col2:
                        st.markdown("**🔧 Optimization Recommendations:**")
                        
                        # Peak demand analysis
                        highest_md_month = df_monthly.loc[df_monthly['Max Demand (kW)'].idxmax()]
                        st.write(f"• **Highest demand month:** {highest_md_month['Month']} ({highest_md_month['Max Demand (kW)']:.1f} kW)")
                        
                        # Cost variation analysis
                        cost_std = df_monthly['Cost Difference (RM)'].std()
                        if cost_std > 50:
                            st.write("• **High cost variation** - focus on demand management")
                        
                        # Seasonal insights
                        if total_months >= 6:
                            seasonal_pattern = df_monthly['Cost Difference (RM)'].rolling(3).mean().std()
                            if seasonal_pattern > 20:
                                st.write("• **Seasonal patterns detected** - consider seasonal strategies")
                        
                        # General recommendations
                        if avg_percentage_change > 5:
                            st.write("• Consider **demand shaving** during peak periods")
                            st.write("• Review **load scheduling** opportunities")
                        elif avg_percentage_change < -5:
                            st.write("• **Proceed with RP4 migration** - clear benefits")
                            st.write("• Monitor **peak period** consumption patterns")
                
                else:
                    st.warning("No sufficient monthly data found for analysis. Please ensure your data spans at least one full month.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your Excel file has proper timestamp and power columns.")
    
    else:
        st.info("👆 Upload an Excel file with load profile data to start the monthly analysis.")
        
        # Show example of expected data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your Excel file should contain:
            
            **Required Columns:**
            - **Timestamp**: Date and time (e.g., "2024-01-01 00:00:00")
            - **Power (kW)**: Power consumption values
            
            **Example:**
            ```
            Timestamp               | Power (kW)
            2024-01-01 00:00:00    | 150.5
            2024-01-01 00:15:00    | 145.2
            2024-01-01 00:30:00    | 148.7
            ...
            ```
            
            **Tips:**
            - Data can be at any interval (15min, 30min, 1hr, etc.)
            - Multiple months of data will provide better insights
            - Include public holidays for accurate RP4 peak/off-peak calculation
            """)

with tabs[4]:
    show_md_shaving_solution()

with tabs[5]:
    # 🔋 Advanced MD Shaving Tab
    st.title("🔋 Advanced MD Shaving")
    st.markdown("""
    **Advanced Maximum Demand (MD) shaving analysis with real battery degradation modeling.**
    
    This tool integrates actual WEIHENG TIANWU series degradation data to provide:
    - **Real degradation curves** (not linear approximations)
    - **20-year battery lifecycle analysis** with non-linear patterns
    - **MD target vs capacity modeling** over extended timeframes
    - **Investment planning** with accurate performance projections
    """)
    
    # Add information box about real degradation data
    with st.expander("🔬 About WEIHENG TIANWU Real Degradation Data", expanded=False):
        st.markdown("""
        **This analysis uses actual WEIHENG TIANWU series test data:**
        
        ✅ **Real Performance Data:**
        - 21 data points over 20-year period (Year 0-20)
        - State of Health (SOH) measurements from laboratory testing
        - Non-linear degradation pattern with initial steep drop then gradual decline
        - End-of-life defined at 80% SOH (typically achieved around year 15)
        
        📊 **Key Degradation Characteristics:**
        - **Year 0:** 100.00% SOH (new battery)
        - **Year 1:** 93.78% SOH (6.22% initial loss - typical for Li-ion)
        - **Years 1-15:** Gradual linear decline (~0.93% per year)
        - **Year 15:** 79.95% SOH (warranty end-of-life)
        - **Year 20:** 60.48% SOH (calendar life end)
        
        🎯 **Advantages over Linear Models:**
        - More accurate capacity predictions
        - Better financial planning capabilities  
        - Realistic performance expectations
        - Validated against real test data
        
        ⚠️ **Important Notes:**
        - Data represents laboratory conditions
        - Real-world performance may vary with operating conditions
        - Temperature, charge/discharge patterns affect actual degradation
        - Regular monitoring recommended for validation
        """)
    
    st.markdown("---")
    
    # Hardcoded battery database (from WEIHENG specs)
    battery_db = {
        "TIANWU-50-233-0.25C": {
            "company": "WEIHENG",
            "model": "WH-TIANWU-50-233B",
            "c_rate": 0.25,
            "power_kW": 50,
            "energy_kWh": 233,
            "voltage_V": 832,
            "lifespan_years": 15,
            "eol_capacity_pct": 80,
            "cycles_per_day": 1.0,
            "cooling": "Liquid (Battery), Air (PCS)",
            "weight_kg": 2700,
            "dimensions_mm": [1400, 1350, 2100]
        },
        "TIANWU-100-233-0.5C": {
            "company": "WEIHENG",
            "model": "WH-TIANWU-100-233B",
            "c_rate": 0.5,
            "power_kW": 100,
            "energy_kWh": 233,
            "voltage_V": 832,
            "lifespan_years": 15,
            "eol_capacity_pct": 80,
            "cycles_per_day": 1.0,
            "cooling": "Liquid (Battery + PCS)",
            "weight_kg": 2700,
            "dimensions_mm": [1400, 1350, 2100]
        },
        "TIANWU-250-233-1C": {
            "company": "WEIHENG",
            "model": "WH-TIANWU-250-A",
            "c_rate": 1.0,
            "power_kW": 250,
            "energy_kWh": 233,
            "voltage_V": 832,
            "lifespan_years": 15,
            "eol_capacity_pct": 80,
            "cycles_per_day": 1.0,
            "cooling": "Liquid (Battery), Air (PCS)",
            "weight_kg": 2600,
            "dimensions_mm": [1400, 1350, 2100]
        }
    }
    
    # Section 1: Upload Load Profile
    st.header("📊 Section 1: Upload Load Profile")
    st.markdown("Upload a CSV file containing peak event data with the following columns:")
    
    expected_columns = [
        "Start Date", "Start Time", "End Date", "End Time",
        "Peak Load (kW)", "Excess (kW)", "Duration (min)",
        "Energy to Shave (kWh)", "Energy to Shave (Peak Period Only)",
        "MD Cost Impact (RM)"
    ]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Expected columns:**")
        for col in expected_columns[:5]:
            st.write(f"• {col}")
    with col2:
        st.markdown("**Additional columns:**")
        for col in expected_columns[5:]:
            st.write(f"• {col}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file with load profile data",
        type=["csv"],
        help="Upload your peak events data from MD shaving analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} peak events.")
            
            # Display file preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate required columns
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
                st.info("The analysis will continue with available columns.")
            
            # Critical Event Analysis Sections
            st.markdown("---")
            st.header("🎯 Critical Event Analysis")
            
            # Section 1: Event with Maximum MD Excess
            st.subheader("💰 Section 1: Highest MD Cost Event")
            
            try:
                if "MD Cost Impact (RM)" in df.columns and not df.empty:
                    # Find event with maximum MD cost impact
                    max_md_cost_idx = df["MD Cost Impact (RM)"].idxmax()
                    max_md_event = df.loc[max_md_cost_idx]
                    
                    # Display event details in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MD Excess", f"{max_md_event.get('MD Excess (kW)', 'N/A'):.2f} kW" if pd.notna(max_md_event.get('MD Excess (kW)')) else "N/A")
                        st.metric("Energy to Shave", f"{max_md_event.get('Energy to Shave (kWh)', 'N/A'):.1f} kWh" if pd.notna(max_md_event.get('Energy to Shave (kWh)')) else "N/A")
                    
                    with col2:
                        st.metric("MD Cost Impact", f"RM {max_md_event.get('MD Cost Impact (RM)', 0):.2f}")
                        st.metric("Duration", f"{max_md_event.get('Duration (min)', 'N/A'):.1f} min" if pd.notna(max_md_event.get('Duration (min)')) else "N/A")
                    
                    with col3:
                        st.metric("Date", f"{max_md_event.get('Start Date', 'N/A')}")
                        st.metric("Time", f"{max_md_event.get('Start Time', 'N/A')} - {max_md_event.get('End Time', 'N/A')}")
                    
                    # Additional details in expander
                    with st.expander("📊 Detailed Event Information"):
                        event_details = {}
                        for col in df.columns:
                            if col in max_md_event.index:
                                value = max_md_event[col]
                                if pd.notna(value):
                                    event_details[col] = value
                        
                        # Create a single-row DataFrame for better display
                        event_df = pd.DataFrame([event_details])
                        st.dataframe(event_df, use_container_width=True)
                        
                        st.info("💡 **This is the most financially impactful event for MD charges.** Focus on preventing this level of excess demand.")
                        
                else:
                    st.warning("No MD Cost Impact data available for critical event analysis.")
                    
            except Exception as e:
                st.error(f"Error analyzing maximum MD cost event: {str(e)}")
            
            # Section 2: Event with Maximum Energy to Shave
            st.subheader("⚡ Section 2: Highest Energy Demand Event")
            
            try:
                if "Energy to Shave (kWh)" in df.columns and not df.empty:
                    # Find event with maximum energy to shave
                    max_energy_idx = df["Energy to Shave (kWh)"].idxmax()
                    max_energy_event = df.loc[max_energy_idx]
                    
                    # Display event details in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Energy to Shave", f"{max_energy_event.get('Energy to Shave (kWh)', 0):.1f} kWh")
                        st.metric("Peak Load", f"{max_energy_event.get('Peak Load (kW)', 'N/A'):.2f} kW" if pd.notna(max_energy_event.get('Peak Load (kW)')) else "N/A")
                    
                    with col2:
                        st.metric("Duration", f"{max_energy_event.get('Duration (min)', 'N/A'):.1f} min" if pd.notna(max_energy_event.get('Duration (min)')) else "N/A")
                        st.metric("Excess Power", f"{max_energy_event.get('Excess (kW)', 'N/A'):.2f} kW" if pd.notna(max_energy_event.get('Excess (kW)')) else "N/A")
                    
                    with col3:
                        st.metric("Date", f"{max_energy_event.get('Start Date', 'N/A')}")
                        st.metric("Time", f"{max_energy_event.get('Start Time', 'N/A')} - {max_energy_event.get('End Time', 'N/A')}")
                    
                    # Additional details in expander
                    with st.expander("📊 Detailed Event Information"):
                        event_details = {}
                        for col in df.columns:
                            if col in max_energy_event.index:
                                value = max_energy_event[col]
                                if pd.notna(value):
                                    event_details[col] = value
                        
                        # Create a single-row DataFrame for better display
                        event_df = pd.DataFrame([event_details])
                        st.dataframe(event_df, use_container_width=True)
                        
                        st.info("💡 **This event requires the most battery capacity.** Size your battery system to handle this energy requirement.")
                        
                        # Calculate battery sizing recommendation for this specific event
                        energy_required = max_energy_event.get('Energy to Shave (kWh)', 0)
                        if energy_required > 0:
                            # Assume 85% DoD and 20% safety factor
                            recommended_capacity = energy_required / 0.85 * 1.2
                            st.success(f"🔋 **Battery Sizing Recommendation**: {recommended_capacity:.1f} kWh capacity needed for this event (85% DoD + 20% safety)")
                        
                else:
                    st.warning("No Energy to Shave data available for critical event analysis.")
                    
            except Exception as e:
                st.error(f"Error analyzing maximum energy event: {str(e)}")
            
            # Comparison insight
            try:
                if ("MD Cost Impact (RM)" in df.columns and "Energy to Shave (kWh)" in df.columns and 
                    not df.empty):
                    
                    max_md_cost_idx = df["MD Cost Impact (RM)"].idxmax()
                    max_energy_idx = df["Energy to Shave (kWh)"].idxmax()
                    
                    if max_md_cost_idx != max_energy_idx:
                        st.info("📋 **Key Insight**: The highest MD cost event and highest energy event are different. Consider both when sizing your battery system.")
                    else:
                        st.success("✅ **Key Insight**: The highest MD cost event and highest energy event are the same. This simplifies battery sizing requirements.")
                        
            except Exception as e:
                st.warning("Unable to compare critical events.")
            
            # Section 2: Battery Selection and Degradation Analysis
            st.header("🔋 Section 2: Battery Selection & Degradation Analysis")
            
            # Extract requirements from uploaded data
            max_excess_kw = df["Excess (kW)"].max() if "Excess (kW)" in df.columns else 0
            total_energy_kwh = df["Energy to Shave (kWh)"].sum() if "Energy to Shave (kWh)" in df.columns else 0
            max_event_energy = df["Energy to Shave (kWh)"].max() if "Energy to Shave (kWh)" in df.columns else 0
            
            # Smart battery selection algorithm
            st.subheader("🤖 Smart Battery Selection Algorithm")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📊 Load Requirements (from uploaded data):**")
                st.write(f"• **Max Excess Power:** {max_excess_kw:.1f} kW")
                st.write(f"• **Max Event Energy:** {max_event_energy:.1f} kWh")
                
                # Battery quantity calculation parameters
                st.subheader("🔢 Battery Quantity Calculation")
                
                # Depth of Discharge for usable capacity calculation
                depth_of_discharge = st.slider(
                    "Depth of Discharge (%)",
                    min_value=70,
                    max_value=95,
                    value=85,
                    step=5,
                    help="Usable capacity percentage for battery longevity"
                )
                
                # Safety factor for capacity
                capacity_safety_factor = st.slider(
                    "Capacity Safety Factor (%)",
                    min_value=0,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Additional capacity buffer for performance variations"
                )
                
                # Discharge efficiency for usable energy calculation
                discharge_efficiency = st.slider(
                    "Discharge Efficiency (%)",
                    min_value=85,
                    max_value=98,
                    value=94,
                    step=1,
                    help="Energy delivered to load during discharge (affects battery quantity)"
                )
                
                # Add explanation box for efficiency parameters
                with st.expander("ℹ️ Understanding Efficiency Parameters", expanded=False):
                    st.markdown("""
                    **Discharge Efficiency** vs **Round-trip Efficiency**:
                    
                    🔋 **Discharge Efficiency (94%)**: 
                    - Energy delivered to the load during discharge
                    - Used in battery **quantity calculation** 
                    - Accounts for conversion losses from DC battery to AC load
                    - Example: 100 kWh stored → 94 kWh delivered to load
                    
                    🔄 **Round-trip Efficiency (92%)**:
                    - Energy recovered compared to energy stored (charge + discharge losses)
                    - Used in battery **simulation** and operation modeling
                    - Accounts for both charging and discharging losses
                    - Example: 100 kWh from grid → 92 kWh recovered to load
                    
                    ⚡ **Why Both Matter**:
                    - **Discharge efficiency** ensures we size enough batteries to deliver required energy
                    - **Round-trip efficiency** models realistic energy flows during operation
                    - Using both provides more accurate battery sizing and simulation
                    """)
                
                
                # Add explanation about discharge efficiency
                with st.expander("ℹ️ About Discharge Efficiency vs Round-trip Efficiency", expanded=False):
                    st.markdown("""
                    **🔋 Discharge Efficiency (94%):** Used in battery quantity calculation
                    - Energy actually delivered to the load during discharge
                    - Accounts for losses in inverter, wiring, and power conversion
                    - Directly impacts how much usable energy each battery unit provides
                    
                    **🔄 Round-trip Efficiency (92%):** Used in operation simulation
                    - Energy efficiency of complete charge-discharge cycle
                    - Includes both charging and discharging losses
                    - Used to simulate actual battery performance over time
                    
                    **📊 Why This Matters:**
                    - Previous calculations assumed 100% discharge efficiency
                    - Real batteries lose 6-15% energy during discharge due to inverter losses
                    - This update provides more accurate battery quantity requirements
                    """)
                
                # MD Target input
                st.subheader("MD Target Configuration")
                target_type = st.radio(
                    "Target Type",
                    ["kW (Power)", "RM (Cost)"],
                    help="Set target as power limit or cost limit"
                )
                
                # Check if uploaded data with MD cost impact is available
                max_md_cost_impact = None
                if uploaded_file is not None and 'df' in locals() and "MD Cost Impact (RM)" in df.columns and not df.empty:
                    try:
                        max_md_cost_impact = df["MD Cost Impact (RM)"].max()
                        st.success(f"✅ Using actual MD cost impact from data: RM {max_md_cost_impact:.2f}")
                    except:
                        max_md_cost_impact = None
                
                if target_type == "kW (Power)":
                    md_target_kw = st.number_input(
                        "MD Target (kW)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=max(100.0, max_excess_kw * 0.8),
                        step=1.0,
                        help="Maximum demand target in kW"
                    )
                    
                    # Use actual MD cost impact if available, otherwise fallback to assumption
                    if max_md_cost_impact is not None:
                        # Calculate based on actual data - this shows the maximum achievable MD cost reduction
                        md_target_rm = max_md_cost_impact
                        st.info(f"💡 **Maximum MD cost impact from data:** RM {md_target_rm:.2f}")
                        st.caption("This represents the maximum monthly MD cost reduction achievable with your uploaded data")
                    else:
                        # Fallback to assumption when no data is available
                        md_target_rm = md_target_kw * 35.0  # Assume RM 35/kW rate
                        st.info(f"Equivalent cost target (estimated): RM {md_target_rm:.2f}")
                        st.caption("⚠️ Using estimated MD rate. Upload data for precise calculations.")
                else:
                    md_target_rm = st.number_input(
                        "MD Target (RM)",
                        min_value=0.0,
                        max_value=50000.0,
                        value=max_md_cost_impact if max_md_cost_impact is not None else 3500.0,
                        step=100.0,
                        help="Maximum demand cost target in RM"
                    )
                    
                    # Use actual data relationship if available, otherwise fallback to assumption
                    if max_md_cost_impact is not None:
                        # When we have actual data, show the power equivalent based on the data
                        if max_excess_kw > 0:
                            # Calculate implied MD rate from the data
                            implied_md_rate = max_md_cost_impact / max_excess_kw
                            md_target_kw = md_target_rm / implied_md_rate if implied_md_rate > 0 else md_target_rm / 35.0
                            st.info(f"Equivalent power target (from data): {md_target_kw:.1f} kW")
                            st.caption(f"Based on actual MD rate from data: RM {implied_md_rate:.2f}/kW")
                        else:
                            md_target_kw = md_target_rm / 35.0  # Fallback
                            st.info(f"Equivalent power target (estimated): {md_target_kw:.1f} kW")
                    else:
                        # Fallback to assumption when no data is available
                        md_target_kw = md_target_rm / 35.0  # Assume RM 35/kW rate
                        st.info(f"Equivalent power target (estimated): {md_target_kw:.1f} kW")
                        st.caption("⚠️ Using estimated MD rate. Upload data for precise calculations.")
                
                # Target year for degradation analysis
                st.subheader("Analysis Parameters")
                target_year = st.slider(
                    "Target Analysis Year",
                    min_value=1,
                    max_value=20,
                    value=15,
                    help="Year to evaluate battery performance (typically 15 for warranty end)"
                )
                
                min_capacity_ratio = st.slider(
                    "Minimum Capacity Ratio",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.2,
                    step=0.1,
                    help="Minimum ratio of degraded capacity to energy requirement"
                )
            
            with col2:
                st.subheader("🔋 Battery Selection Analysis")
                
                # Real WEIHENG TIANWU series degradation data (SOH %)
                real_degradation_data = {
                    0: 100.00, 1: 93.78, 2: 90.85, 3: 88.69, 4: 87.04, 5: 85.73,
                    6: 84.67, 7: 83.79, 8: 83.05, 9: 82.42, 10: 81.88, 11: 81.40,
                    12: 80.98, 13: 80.60, 14: 80.26, 15: 79.95, 16: 79.67, 17: 79.41,
                    18: 79.17, 19: 78.95, 20: 60.48
                }
                
                # Battery selection algorithm
                suitable_batteries = []
                
                for battery_key, battery_specs in battery_db.items():
                    # Get degraded capacity at target year
                    soh_at_target_year = real_degradation_data.get(target_year, 80)
                    degraded_capacity = battery_specs["energy_kWh"] * (soh_at_target_year / 100)
                    
                    # Calculate usable capacity with DOD, discharge efficiency, and safety factor
                    usable_capacity_per_unit = degraded_capacity * (depth_of_discharge / 100) * (discharge_efficiency / 100)
                    usable_capacity_with_safety = usable_capacity_per_unit / (1 + capacity_safety_factor / 100)
                    
                    # Battery quantity calculation using ceiling function
                    import math
                    qty_required_energy = math.ceil(max_event_energy / usable_capacity_with_safety) if usable_capacity_with_safety > 0 else float('inf')
                    qty_required_power = math.ceil(max_excess_kw / battery_specs["power_kW"]) if battery_specs["power_kW"] > 0 else float('inf')
                    
                    # Final quantity is the maximum of energy and power requirements
                    qty_required = max(qty_required_energy, qty_required_power)
                    
                    # Total system specifications
                    total_power_rating = qty_required * battery_specs["power_kW"]
                    total_energy_capacity = qty_required * battery_specs["energy_kWh"]
                    total_usable_capacity = qty_required * usable_capacity_per_unit
                    
                    # Check power requirement
                    power_adequate = total_power_rating >= max_excess_kw
                    
                    # Check energy requirement (use max event energy as critical requirement)
                    energy_adequate = total_usable_capacity >= max_event_energy
                    
                    # Calculate capacity ratio
                    capacity_ratio = total_usable_capacity / max_event_energy if max_event_energy > 0 else 0
                    
                    # Check if battery meets minimum criteria
                    meets_criteria = (power_adequate and 
                                    energy_adequate and 
                                    capacity_ratio >= min_capacity_ratio and
                                    qty_required < 20)  # Practical limit
                    
                    suitable_batteries.append({
                        "Model": battery_key,
                        "Initial Capacity (kWh)": battery_specs["energy_kWh"],
                        "Qty Required": qty_required,
                        "Total System Capacity (kWh)": total_energy_capacity,
                        f"Degraded Capacity Year {target_year} (kWh)": degraded_capacity,
                        "Unit Power (kW)": battery_specs["power_kW"],
                        "Total Power (kW)": total_power_rating,
                        "Total Usable (kWh)": total_usable_capacity,
                        f"SOH Year {target_year} (%)": soh_at_target_year,
                        "Power Adequate": "✅" if power_adequate else "❌",
                        "Energy Adequate": "✅" if energy_adequate else "❌",
                        "Capacity Ratio": capacity_ratio,
                        "Meets Criteria": "✅ Suitable" if meets_criteria else "❌ Inadequate"
                    })
                
                # Display battery comparison table
                selection_df = pd.DataFrame(suitable_batteries)
                
                st.markdown("**Battery Comparison Table with Quantity Calculation:**")
                
                # Color coding function
                def color_cells(val):
                    if isinstance(val, str):
                        if "✅" in val:
                            return 'background-color: lightgreen'
                        elif "❌" in val:
                            return 'background-color: lightcoral'
                    elif isinstance(val, (int, float)):
                        if val >= min_capacity_ratio:
                            return 'background-color: lightgreen'
                        elif val >= 1.0:
                            return 'background-color: lightyellow'
                        else:
                            return 'background-color: lightcoral'
                    return ''
                
                # Apply styling
                styled_selection = selection_df.style.applymap(
                    color_cells, 
                    subset=["Power Adequate", "Energy Adequate", "Capacity Ratio", "Meets Criteria"]
                ).format({
                    "Initial Capacity (kWh)": "{:.0f} kWh",
                    "Qty Required": "{:.0f} units",
                    "Total System Capacity (kWh)": "{:.0f} kWh",
                    f"Degraded Capacity Year {target_year} (kWh)": "{:.1f} kWh",
                    "Unit Power (kW)": "{:.0f} kW",
                    "Total Power (kW)": "{:.0f} kW",
                    "Total Usable (kWh)": "{:.1f} kWh",
                    f"SOH Year {target_year} (%)": "{:.2f}%",
                    "Capacity Ratio": "{:.2f}x"
                })
                
                st.dataframe(styled_selection, use_container_width=True)
                
                # Display quantity calculation details
                st.markdown("**🔢 Quantity Calculation Logic:**")
                st.info(f"""
                **Formula:** `qty_required = max(ceil(energy_req / usable_kWh_per_unit), ceil(power_req / power_kW_per_unit))`
                
                **Parameters:**
                - Energy Requirement: {max_event_energy:.1f} kWh (worst single event)
                - Power Requirement: {max_excess_kw:.1f} kW (maximum excess demand)
                - Depth of Discharge: {depth_of_discharge}% (usable capacity)
                - Discharge Efficiency: {discharge_efficiency}% (energy delivered to load)
                - Safety Factor: {capacity_safety_factor}% (additional margin)
                
                **Example for first battery:**
                - Usable capacity per unit = {battery_db[list(battery_db.keys())[0]]["energy_kWh"]:.0f} kWh × {real_degradation_data.get(target_year, 80):.1f}% × {depth_of_discharge}% × {discharge_efficiency}% = {battery_db[list(battery_db.keys())[0]]["energy_kWh"] * (real_degradation_data.get(target_year, 80)/100) * (depth_of_discharge/100) * (discharge_efficiency/100):.1f} kWh
                - With safety factor = {battery_db[list(battery_db.keys())[0]]["energy_kWh"] * (real_degradation_data.get(target_year, 80)/100) * (depth_of_discharge/100) * (discharge_efficiency/100) / (1 + capacity_safety_factor/100):.1f} kWh per unit
                - Quantity needed = ceil({max_event_energy:.1f} / {battery_db[list(battery_db.keys())[0]]["energy_kWh"] * (real_degradation_data.get(target_year, 80)/100) * (depth_of_discharge/100) * (discharge_efficiency/100) / (1 + capacity_safety_factor/100):.1f}) = {math.ceil(max_event_energy / (battery_db[list(battery_db.keys())[0]]["energy_kWh"] * (real_degradation_data.get(target_year, 80)/100) * (depth_of_discharge/100) * (discharge_efficiency/100) / (1 + capacity_safety_factor/100))) if (battery_db[list(battery_db.keys())[0]]["energy_kWh"] * (real_degradation_data.get(target_year, 80)/100) * (depth_of_discharge/100) * (discharge_efficiency/100) / (1 + capacity_safety_factor/100)) > 0 else 'inf'} units
                """)
                
                # Show recommendations
                suitable_models = [battery for battery in suitable_batteries if "✅ Suitable" in battery["Meets Criteria"]]
                
                if suitable_models:
                    st.success(f"✅ Found {len(suitable_models)} suitable battery configuration(s)")
                    
                    # Recommend the most efficient option (lowest total energy capacity)
                    best_battery = min(suitable_models, key=lambda x: x["Total System Capacity (kWh)"])
                    st.info(f"🎯 **Recommended:** {best_battery['Qty Required']} × {best_battery['Model']} (most cost-effective)")
                    st.write(f"   **Total System:** {best_battery['Total Power (kW)']} kW / {best_battery['Total System Capacity (kWh)']} kWh")
                    
                    # Enhanced selection interface with optional filtering
                    with st.expander("🔍 **Advanced Selection Options**", expanded=False):
                        st.markdown("**Filter suitable configurations:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Brand filtering for suitable models (WEIHENG only)
                            suitable_brands = list(set([battery_db[battery["Model"]]["company"] for battery in suitable_models]))
                            selected_brand_suitable = st.selectbox(
                                "Filter by Brand",
                                ["All WEIHENG Models"] + suitable_brands,
                                help="Filter suitable batteries by manufacturer (Currently: WEIHENG TIANWU series only)",
                                key="suitable_brand_filter"
                            )
                        
                        with col2:
                            # Capacity ratio filtering for suitable models
                            capacity_ratio_filter = st.selectbox(
                                "Filter by Capacity Ratio",
                                ["All Ratios", "High Performance (≥2.0x)", "Good Performance (≥1.5x)", "Adequate (≥1.2x)"],
                                help="Filter by capacity ratio performance",
                                key="suitable_ratio_filter"
                            )
                        
                        # Apply filtering to suitable models
                        filtered_suitable = []
                        for battery in suitable_models:
                            # Brand filter (WEIHENG only)
                            if selected_brand_suitable != "All WEIHENG Models":
                                battery_company = battery_db[battery["Model"]]["company"]
                                if battery_company != selected_brand_suitable:
                                    continue
                            
                            # Capacity ratio filter
                            if capacity_ratio_filter == "High Performance (≥2.0x)":
                                if battery["Capacity Ratio"] < 2.0:
                                    continue
                            elif capacity_ratio_filter == "Good Performance (≥1.5x)":
                                if battery["Capacity Ratio"] < 1.5:
                                    continue
                            elif capacity_ratio_filter == "Adequate (≥1.2x)":
                                if battery["Capacity Ratio"] < 1.2:
                                    continue
                            
                            filtered_suitable.append(battery)
                        
                        if filtered_suitable:
                            st.info(f"📊 **Filtered Results:** {len(filtered_suitable)} suitable configurations match your criteria")
                        else:
                            st.warning("⚠️ No suitable configurations match the selected filters. Showing all suitable options.")
                            filtered_suitable = suitable_models
                    
                    # Use filtered results if filters were applied, otherwise use all suitable models
                    if 'filtered_suitable' in locals() and filtered_suitable and len(filtered_suitable) < len(suitable_models):
                        display_models = filtered_suitable
                        st.info(f"🔍 **Showing {len(display_models)} filtered suitable configurations**")
                    else:
                        display_models = suitable_models
                    
                    # Allow manual selection
                    recommended_battery = st.selectbox(
                        "Select Battery Configuration for Detailed Analysis",
                        [f"{b['Qty Required']} × {b['Model']}" for b in display_models],
                        index=0,
                        help="Choose from suitable configurations for detailed degradation analysis"
                    )
                    
                    # Extract selected battery details
                    selected_config = recommended_battery.split(" × ")
                    selected_qty = int(selected_config[0])
                    selected_battery = selected_config[1]
                    
                else:
                    st.error("❌ No battery configurations meet the current criteria")
                    st.warning("Consider:")
                    st.write("• Reducing minimum capacity ratio")
                    st.write("• Increasing depth of discharge")
                    st.write("• Reducing capacity safety factor")
                    st.write("• Selecting earlier target year")
                    st.write("• Exploring custom battery configurations")
                    
                    # Enhanced selection interface with filtering options
                    st.markdown("### 🔍 **Advanced Battery Selection (Override)**")
                    
                    # Create filtering options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Brand filtering (WEIHENG only)
                        available_brands = list(set([battery_specs["company"] for battery_specs in battery_db.values()]))
                        selected_brand = st.selectbox(
                            "Filter by Brand",
                            ["All WEIHENG Models"] + available_brands,
                            help="Filter batteries by manufacturer (Currently: WEIHENG TIANWU series only)"
                        )
                    
                    with col2:
                        # Performance criteria filtering
                        performance_filter = st.selectbox(
                            "Filter by Performance Criteria",
                            ["All Options", "Energy Adequate Only", "Power Adequate Only", "Both Adequate", "High Capacity Ratio (≥1.5x)"],
                            help="Filter based on performance criteria from the comparison table"
                        )
                    
                    # Apply filtering logic
                    filtered_batteries = []
                    
                    for battery in suitable_batteries:
                        # Brand filter
                        if selected_brand != "All Brands":
                            battery_company = battery_db[battery["Model"]]["company"]
                            if battery_company != selected_brand:
                                continue
                        
                        # Performance filter
                        if performance_filter == "Energy Adequate Only":
                            if "✅" not in battery["Energy Adequate"]:
                                continue
                        elif performance_filter == "Power Adequate Only":
                            if "✅" not in battery["Power Adequate"]:
                                continue
                        elif performance_filter == "Both Adequate":
                            if "✅" not in battery["Energy Adequate"] or "✅" not in battery["Power Adequate"]:
                                continue
                        elif performance_filter == "High Capacity Ratio (≥1.5x)":
                            if battery["Capacity Ratio"] < 1.5:
                                continue
                        
                        filtered_batteries.append(battery)
                    
                    # Display filtering results
                    if filtered_batteries:
                        st.info(f"📊 **Filtered Results:** {len(filtered_batteries)} configurations match your criteria")
                        
                        # Create selection options from filtered results
                        filtered_options = [f"{b['Qty Required']} × {b['Model']}" for b in filtered_batteries]
                        
                        selected_config_override = st.selectbox(
                            "Select Configuration for Analysis",
                            filtered_options,
                            help="Choose from filtered battery configurations"
                        )
                        
                        # Show details of filtered selection
                        selected_battery_detail = next(b for b in filtered_batteries if f"{b['Qty Required']} × {b['Model']}" == selected_config_override)
                        
                        # Display selected battery summary
                        with st.expander("📋 Selected Configuration Details", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Power", f"{selected_battery_detail['Total Power (kW)']:.0f} kW")
                                st.metric("Power Adequate", selected_battery_detail["Power Adequate"])
                            with col2:
                                st.metric("Total Usable Capacity", f"{selected_battery_detail['Total Usable (kWh)']:.1f} kWh")
                                st.metric("Energy Adequate", selected_battery_detail["Energy Adequate"])
                            with col3:
                                st.metric("Capacity Ratio", f"{selected_battery_detail['Capacity Ratio']:.2f}x")
                                st.metric("Overall Status", selected_battery_detail["Meets Criteria"])
                    
                    else:
                        # Fallback: show all options if no filtered results
                        st.warning(f"⚠️ No batteries match the selected filters. Showing all available options.")
                        
                        # Show all battery options
                        all_battery_options = [f"1 × {key}" for key in battery_db.keys()]
                        
                        # Add brand information to options for clarity
                        detailed_options = []
                        for key in battery_db.keys():
                            company = battery_db[key]["company"]
                            power = battery_db[key]["power_kW"]
                            energy = battery_db[key]["energy_kWh"]
                            detailed_options.append(f"1 × {key} ({company} - {power}kW/{energy}kWh)")
                        
                        selected_detailed_option = st.selectbox(
                            "Select Configuration for Analysis (All Available)",
                            detailed_options,
                            help="All available battery configurations"
                        )
                        
                        # Extract the battery key from detailed option
                        selected_config_override = selected_detailed_option.split(" (")[0]  # Extract "1 × TIANWU-..." part
                    
                    # Extract selected battery details
                    selected_config = selected_config_override.split(" × ")
                    selected_qty = int(selected_config[0])
                    selected_battery = selected_config[1]
                    
                    # Show selection confirmation
                    selected_specs = battery_db[selected_battery]
                    st.success(f"✅ **Selected:** {selected_qty} × {selected_specs['model']} ({selected_specs['company']})")
                    st.write(f"   **Configuration:** {selected_qty * selected_specs['power_kW']} kW / {selected_qty * selected_specs['energy_kWh']} kWh total system")
            
            # Section 3: Detailed Battery Analysis & Degradation Modeling
            st.header("📉 Section 3: Detailed Battery Analysis")
            
            battery_specs = battery_db[selected_battery]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Selected Battery Configuration")
                
                # Display battery specs in a nice format
                specs_data = {
                    "Configuration": f"{selected_qty} × {battery_specs['model']}",
                    "Company": battery_specs["company"],
                    "Model (Single Unit)": battery_specs["model"],
                    "Quantity": f"{selected_qty} units",
                    "C-Rate": f"{battery_specs['c_rate']}C",
                    "Unit Power Rating": f"{battery_specs['power_kW']} kW",
                    "Unit Energy Capacity": f"{battery_specs['energy_kWh']} kWh",
                    "Total Power Rating": f"{selected_qty * battery_specs['power_kW']} kW",
                    "Total Energy Capacity": f"{selected_qty * battery_specs['energy_kWh']} kWh",
                    "Voltage (per unit)": f"{battery_specs['voltage_V']} V",
                    "Lifespan": f"{battery_specs['lifespan_years']} years",
                    "End of Life Capacity": f"{battery_specs['eol_capacity_pct']}%",
                    "Cycles per Day": battery_specs["cycles_per_day"],
                    "Cooling": battery_specs["cooling"],
                    "Unit Weight": f"{battery_specs['weight_kg']} kg",
                    "Total Weight": f"{selected_qty * battery_specs['weight_kg']} kg",
                    "Unit Dimensions (L×W×H)": f"{battery_specs['dimensions_mm'][0]}×{battery_specs['dimensions_mm'][1]}×{battery_specs['dimensions_mm'][2]} mm"
                }
                
                for key, value in specs_data.items():
                    st.write(f"**{key}:** {value}")
                    
                # Performance vs Requirements
                st.subheader("🎯 Performance vs Requirements")
                
                # Calculate total system specs
                total_power_rating = selected_qty * battery_specs["power_kW"]
                total_energy_capacity = selected_qty * battery_specs["energy_kWh"]
                
                # Power comparison
                power_margin = total_power_rating - max_excess_kw
                if power_margin >= 0:
                    st.success(f"✅ Power: {total_power_rating} kW (Margin: +{power_margin:.1f} kW)")
                else:
                    st.error(f"❌ Power: {total_power_rating} kW (Shortfall: {power_margin:.1f} kW)")
                
                # Energy comparison at target year
                target_soh = real_degradation_data.get(target_year, 80)
                total_degraded_capacity_target = total_energy_capacity * (target_soh / 100)
                total_usable_capacity_target = total_degraded_capacity_target * (depth_of_discharge / 100)
                energy_margin = total_usable_capacity_target - max_event_energy
                
                if energy_margin >= 0:
                    st.success(f"✅ Usable Energy (Year {target_year}): {total_usable_capacity_target:.1f} kWh (Margin: +{energy_margin:.1f} kWh)")
                else:
                    st.error(f"❌ Usable Energy (Year {target_year}): {total_usable_capacity_target:.1f} kWh (Shortfall: {energy_margin:.1f} kWh)")
                    
                # Capacity ratio
                capacity_ratio_target = total_usable_capacity_target / max_event_energy if max_event_energy > 0 else 0
                if capacity_ratio_target >= min_capacity_ratio:
                    st.success(f"✅ Capacity Ratio: {capacity_ratio_target:.2f}x (Target: {min_capacity_ratio:.1f}x)")
                else:
                    st.warning(f"⚠️ Capacity Ratio: {capacity_ratio_target:.2f}x (Target: {min_capacity_ratio:.1f}x)")
                
                # System-level metrics
                st.subheader("🏗️ System Configuration")
                col1_sys, col2_sys = st.columns(2)
                with col1_sys:
                    st.metric("Battery Units", f"{selected_qty}")
                    st.metric("Total Footprint", f"{selected_qty * battery_specs['weight_kg'] / 1000:.1f} tonnes")
                with col2_sys:
                    st.metric("Power Density", f"{total_power_rating / total_energy_capacity:.2f} kW/kWh")
                    total_floor_area = selected_qty * (battery_specs['dimensions_mm'][0] * battery_specs['dimensions_mm'][1]) / 1_000_000
                    st.metric("Floor Area", f"{total_floor_area:.1f} m²")
            
            with col2:
                st.subheader("🔍 Advanced Analysis Options")
                
                # Real WEIHENG TIANWU series degradation data (SOH %) - Define early for use throughout
                real_degradation_data = {
                    0: 100.00, 1: 93.78, 2: 90.85, 3: 88.69, 4: 87.04, 5: 85.73,
                    6: 84.67, 7: 83.79, 8: 83.05, 9: 82.42, 10: 81.88, 11: 81.40,
                    12: 80.98, 13: 80.60, 14: 80.26, 15: 79.95, 16: 79.67, 17: 79.41,
                    18: 79.17, 19: 78.95, 20: 60.48
                }
                
                # Analysis options
                show_comparison = st.checkbox("Compare with Linear Degradation", value=True)
                show_cost_analysis = st.checkbox("Include Cost Analysis", value=False)
                show_sensitivity = st.checkbox("Sensitivity Analysis", value=False)
                
                if show_cost_analysis:
                    st.markdown("**Cost Parameters:**")
                    cost_per_kwh = st.number_input("Battery Cost (RM/kWh)", value=1200, step=50)
                    pcs_cost_per_kw = st.number_input("PCS Cost (RM/kW)", value=400, step=25)
                    installation_factor = st.slider("Installation Factor", 1.1, 2.0, 1.4, 0.1)
                    
                    # Calculate costs
                    battery_cost = battery_specs["energy_kWh"] * cost_per_kwh
                    pcs_cost = battery_specs["power_kW"] * pcs_cost_per_kw
                    total_cost = (battery_cost + pcs_cost) * installation_factor
                    
                    st.metric("Total System Cost", f"RM {total_cost:,.0f}")
                    st.metric("Cost per kWh", f"RM {total_cost / battery_specs['energy_kWh']:,.0f}")
                    st.metric("Cost per kW", f"RM {total_cost / battery_specs['power_kW']:,.0f}")
                
                if show_sensitivity:
                    st.markdown("**Sensitivity Analysis:**")
                    degradation_factor = st.slider("Degradation Multiplier", 0.8, 1.5, 1.0, 0.1, 
                                                  help="1.0 = normal, >1.0 = faster degradation")
                else:
                    degradation_factor = 1.0
                st.subheader("Selected Battery Specifications")
                
                battery_specs = battery_db[selected_battery]
                
                # Display battery specs in a nice format
                specs_data = {
                    "Company": battery_specs["company"],
                    "Model": battery_specs["model"],
                    "C-Rate": f"{battery_specs['c_rate']}C",
                    "Power Rating": f"{battery_specs['power_kW']} kW",
                    "Energy Capacity": f"{battery_specs['energy_kWh']} kWh",
                    "Voltage": f"{battery_specs['voltage_V']} V",
                    "Lifespan": f"{battery_specs['lifespan_years']} years",
                    "End of Life Capacity": f"{battery_specs['eol_capacity_pct']}%",
                    "Cycles per Day": battery_specs["cycles_per_day"],
                    "Cooling": battery_specs["cooling"],
                    "Weight": f"{battery_specs['weight_kg']} kg",
                    "Dimensions (L×W×H)": f"{battery_specs['dimensions_mm'][0]}×{battery_specs['dimensions_mm'][1]}×{battery_specs['dimensions_mm'][2]} mm"
                }
                
                for key, value in specs_data.items():
                    st.write(f"**{key}:** {value}")
            
            # Calculate and display battery degradation
            st.subheader("📉 20-Year Battery Degradation Analysis")
            st.markdown(f"**Real WEIHENG TIANWU Series Degradation Curve - {selected_qty} Unit System**")
            
            initial_capacity = selected_qty * battery_specs["energy_kWh"]  # Total system capacity
            
            # Generate yearly data using real degradation curve with sensitivity
            years = list(range(0, 21))  # 0 to 20 years
            capacities = []
            soh_percentages = []
            linear_capacities = []  # For comparison
            
            for year in years:
                # Real degradation data
                base_soh_pct = real_degradation_data[year]
                
                # Apply sensitivity factor (accelerate/decelerate degradation)
                if year == 0:
                    adjusted_soh_pct = 100.0  # Always start at 100%
                else:
                    degradation_loss = 100 - base_soh_pct
                    adjusted_loss = degradation_loss * degradation_factor
                    adjusted_soh_pct = max(0, 100 - adjusted_loss)
                
                capacity_kwh = initial_capacity * (adjusted_soh_pct / 100)
                capacities.append(capacity_kwh)
                soh_percentages.append(adjusted_soh_pct)
                
                # Linear degradation for comparison (80% at 15 years)
                linear_soh = max(0, 100 - (20 / 15 * year))  # 20% loss over 15 years
                linear_capacity = initial_capacity * (linear_soh / 100)
                linear_capacities.append(linear_capacity)
            
            # Create degradation DataFrame
            degradation_df = pd.DataFrame({
                'Year': years,
                'SOH (%)': soh_percentages,
                'Total Capacity (kWh)': capacities,
                'Total Usable (kWh)': [cap * (depth_of_discharge / 100) * (discharge_efficiency / 100) for cap in capacities],
                'Linear SOH (%)': [max(0, 100 - (20 / 15 * year)) for year in years],
                'Linear Capacity (kWh)': linear_capacities,
                'Max Event Energy (kWh)': [max_event_energy] * len(years),
                'MD Target (kW)': [md_target_kw] * len(years),
                'Capacity Ratio': [cap * (depth_of_discharge / 100) * (discharge_efficiency / 100) / max_event_energy if max_event_energy > 0 else 0 for cap in capacities]
            })
            
            # Create enhanced dual-axis chart for MD shaving analysis
            fig = go.Figure()
            
            # Add MD Target Power line (left Y-axis)
            if md_target_kw > 0:
                fig.add_trace(go.Scatter(
                    x=degradation_df['Year'],
                    y=degradation_df['MD Target (kW)'],
                    mode='lines',
                    name='MD Target Power',
                    line=dict(color='green', width=3, dash='dot'),
                    yaxis='y',
                    hovertemplate='<b>MD Target</b><br>Year: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
                ))
            
            # Add Required Energy with Safety Factor (right Y-axis)
            required_energy_with_safety = max_event_energy * 1.10  # 10% safety factor
            fig.add_trace(go.Scatter(
                x=degradation_df['Year'],
                y=[required_energy_with_safety] * len(years),
                mode='lines',
                name='Required Energy (with 10% safety)',
                line=dict(color='red', width=3, dash='dash'),
                yaxis='y2',
                hovertemplate='<b>Required Energy</b><br>Year: %{x}<br>Energy: %{y:.1f} kWh<extra></extra>'
            ))
            
            # Add Battery Usable Capacity line (right Y-axis) - Including discharge efficiency
            fig.add_trace(go.Scatter(
                x=degradation_df['Year'],
                y=degradation_df['Total Usable (kWh)'],
                mode='lines+markers',
                name=f'Battery Usable Capacity ({selected_qty} × {battery_specs["model"]})',
                line=dict(color='blue', width=4),
                marker=dict(size=8, color='blue'),
                yaxis='y2',
                hovertemplate='<b>Usable Capacity (w/ Discharge Eff.)</b><br>Year: %{x}<br>Usable: %{y:.1f} kWh<br>SOH: %{customdata:.1f}%<br>DoD: ' + f'{depth_of_discharge}%<br>Discharge Eff: {discharge_efficiency}%<extra></extra>',
                customdata=degradation_df['SOH (%)']
            ))
            
            # Add zone fill where battery becomes insufficient
            insufficient_mask = [usable < required_energy_with_safety for usable in degradation_df['Total Usable (kWh)']]
            if any(insufficient_mask):
                # Find the first year where battery becomes insufficient
                first_insufficient_year = next((i for i, insufficient in enumerate(insufficient_mask) if insufficient), None)
                if first_insufficient_year is not None:
                    fig.add_vrect(
                        x0=first_insufficient_year, x1=20,
                        fillcolor="red", opacity=0.2,
                        layer="below", line_width=0,
                        annotation_text=f"Insufficient Capacity Zone (Year {first_insufficient_year}+)",
                        annotation_position="top left",
                    )
            
            # Add Warranty EOL marker
            eol_year = 15
            fig.add_vline(
                x=eol_year, 
                line_dash="dot", 
                line_color="orange",
                annotation_text="Warranty EOL",
                annotation_position="top"
            )
            # Add target year marker
            if target_year != eol_year:
                fig.add_vline(
                    x=target_year,
                    line_dash="dash",
                    line_color="purple",
                    annotation_text=f"Target Analysis Year: {target_year}",
                    annotation_position="top"
                )
            
            # Add year labels on the usable capacity line for key years
            key_years = [1, 5, 10, 15, 20]
            for year in key_years:
                if year < len(degradation_df):
                    usable_capacity = degradation_df.iloc[year]['Total Usable (kWh)']
                    fig.add_annotation(
                        x=year,
                        y=usable_capacity,
                        text=f"{usable_capacity:.0f} kWh",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="blue",
                        ax=20,
                        ay=-20,
                        font=dict(size=10, color="blue"),
                        yref="y2"
                    )
            
            # Update layout for dual y-axes optimized for MD shaving analysis
            fig.update_layout(
                title=f'📊 Advanced MD Shaving Analysis – {battery_specs["model"]} ({selected_qty} Units)<br><sub>MD Power Target vs Battery Usable Capacity Over Time (includes {discharge_efficiency}% discharge efficiency)</sub>',
                xaxis=dict(
                    title='Years',
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[0, 20],
                    dtick=1
                ),
                yaxis=dict(
                    title='MD Target Power (kW)',
                    side='left',
                    color='green',
                    showgrid=False,
                    range=[0, md_target_kw * 1.2] if md_target_kw > 0 else [0, 100]
                ),
                yaxis2=dict(
                    title='Battery Usable Capacity (kWh)',
                    side='right',
                    overlaying='y',
                    color='blue',
                    showgrid=True,
                    gridcolor='lightblue',
                    range=[0, max(max(degradation_df['Total Usable (kWh)']), required_energy_with_safety) * 1.1]
                ),
                legend=dict(
                    x=0.02, 
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                ),
                height=700,
                hovermode='x unified',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display enhanced degradation table
            st.subheader("📊 Enhanced Degradation Timeline Analysis")
            
            # Create comprehensive display table
            display_df = degradation_df.copy()
            display_df['SOH (%)'] = display_df['SOH (%)'].round(2)
            display_df['Capacity (kWh)'] = display_df['Total Capacity (kWh)'].round(1)
            display_df['Capacity Ratio'] = display_df['Capacity Ratio'].round(2)
            display_df['Years from EOL'] = 15 - display_df['Year']
            display_df['Years from EOL'] = display_df['Years from EOL'].apply(lambda x: f"+{abs(x)}" if x < 0 else str(x))
            
            # Add performance status
            def get_performance_status(ratio):
                if ratio >= 2.0:
                    return "🟢 Excellent"
                elif ratio >= min_capacity_ratio:
                    return "🟡 Adequate"
                elif ratio >= 1.0:
                    return "🟠 Marginal"
                else:
                    return "🔴 Insufficient"
            
            display_df['Performance Status'] = display_df['Capacity Ratio'].apply(get_performance_status)
            
            # Add comparison with linear model if selected
            if show_comparison:
                display_df['Linear SOH (%)'] = display_df['Linear SOH (%)'].round(2)
                display_df['Linear Capacity (kWh)'] = display_df['Linear Capacity (kWh)'].round(1)
                display_df['Real vs Linear'] = (display_df['Capacity (kWh)'] - display_df['Linear Capacity (kWh)']).round(1)
            
            # Color coding functions
            def color_capacity_ratio(val):
                if val >= 2.0:
                    return 'background-color: #28a745; color: white'  # Dark green
                elif val >= min_capacity_ratio:
                    return 'background-color: #90EE90'  # Light green
                elif val >= 1.0:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #F08080'  # Light coral
            
            def color_soh(val):
                if val >= 90:
                    return 'background-color: #90EE90'
                elif val >= 80:
                    return 'background-color: #FFFFE0'
                elif val >= 70:
                    return 'background-color: #FFA500'
                else:
                    return 'background-color: #F08080'
            
            def color_performance_status(val):
                if "🟢" in val:
                    return 'background-color: #28a745; color: white'
                elif "🟡" in val:
                    return 'background-color: #ffc107; color: black'
                elif "🟠" in val:
                    return 'background-color: #fd7e14; color: white'
                else:
                    return 'background-color: #dc3545; color: white'
            
            # Apply styling
            columns_to_format = {
                'SOH (%)': '{:.2f}%',
                'Capacity (kWh)': '{:.1f} kWh',
                'Capacity Ratio': '{:.2f}x'
            }
            
            if show_comparison:
                columns_to_format.update({
                    'Linear SOH (%)': '{:.2f}%',
                    'Linear Capacity (kWh)': '{:.1f} kWh',
                    'Real vs Linear': '{:+.1f} kWh'
                })
            
            styled_df = display_df.style.applymap(
                color_capacity_ratio, 
                subset=['Capacity Ratio']
            ).applymap(
                color_soh,
                subset=['SOH (%)']
            ).applymap(
                color_performance_status,
                subset=['Performance Status']
            ).format(columns_to_format)
            
            # Show warranty period (0-15 years) by default
            warranty_period_df = styled_df.data.iloc[:16]  # Years 0-15
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Warranty Period Analysis (Years 0-15):**")
                st.dataframe(warranty_period_df, use_container_width=True, height=400)
            
            with col2:
                # Key metrics for warranty period
                warranty_data = degradation_df.iloc[:16]
                years_adequate = sum(warranty_data['Capacity Ratio'] >= min_capacity_ratio)
                years_excellent = sum(warranty_data['Capacity Ratio'] >= 2.0)
                first_inadequate_year = None
                
                for idx, ratio in enumerate(warranty_data['Capacity Ratio']):
                    if ratio < min_capacity_ratio:
                        first_inadequate_year = idx
                        break
                
                st.markdown("**Warranty Period Summary:**")
                st.metric("Years Meeting Target", f"{years_adequate}/16", f"{(years_adequate/16*100):.0f}%")
                st.metric("Years Excellent", f"{years_excellent}/16", f"{(years_excellent/16*100):.0f}%")
                
                if first_inadequate_year is not None:
                    st.metric("First Inadequate Year", first_inadequate_year, delta="⚠️ Early failure", delta_color="inverse")
                else:
                    st.metric("Performance", "All years adequate", delta="✅ Excellent", delta_color="normal")
                
                # EOL metrics
                eol_soh = warranty_data.iloc[-1]['SOH (%)']
                eol_capacity = warranty_data.iloc[-1]['Total Capacity (kWh)']
                eol_ratio = warranty_data.iloc[-1]['Capacity Ratio']
                
                st.markdown("**15-Year EOL Metrics:**")
                st.write(f"• SOH: **{eol_soh:.2f}%**")
                st.write(f"• Capacity: **{eol_capacity:.1f} kWh**")
                st.write(f"• Ratio: **{eol_ratio:.2f}x**")
            
            # Show extended data in expander
            with st.expander("🔍 View Extended Timeline (16-20 years) - Calendar Life"):
                extended_df = styled_df.data.iloc[16:]
                st.dataframe(extended_df, use_container_width=True)
                st.caption("⚠️ Years 16-20 represent calendar life extension beyond typical warranty period")
                
                # Extended period warnings
                calendar_data = degradation_df.iloc[16:]
                if len(calendar_data) > 0:
                    min_calendar_ratio = calendar_data['Capacity Ratio'].min()
                    final_soh = calendar_data.iloc[-1]['SOH (%)']
                    
                    if min_calendar_ratio < 1.0:
                        st.error(f"⚠️ **Extended Period Risk:** Capacity ratio drops to {min_calendar_ratio:.2f}x by year 20")
                    if final_soh < 70:
                        st.warning(f"⚠️ **Calendar Life End:** Final SOH of {final_soh:.1f}% indicates end of useful life")
            
            # Enhanced analysis summary
            st.subheader("🎯 Comprehensive Analysis Summary & Investment Guidance")
            
            
            # Calculate comprehensive metrics using real degradation data
            eol_15_year = degradation_df[degradation_df['Year'] == 15]
            final_capacity_15 = eol_15_year['Total Capacity (kWh)'].iloc[0]
            final_soh_15 = eol_15_year['SOH (%)'].iloc[0]
            final_ratio_15 = eol_15_year['Capacity Ratio'].iloc[0]
            
            # Calendar life metrics (20 years)
            calendar_life_capacity = degradation_df[degradation_df['Year'] == 20]['Total Capacity (kWh)'].iloc[0]
            calendar_life_soh = degradation_df[degradation_df['Year'] == 20]['SOH (%)'].iloc[0]
            calendar_life_ratio = degradation_df[degradation_df['Year'] == 20]['Capacity Ratio'].iloc[0]
            
            # Target year metrics
            target_year_data = degradation_df[degradation_df['Year'] == target_year]
            target_capacity = target_year_data['Total Capacity (kWh)'].iloc[0]
            target_soh = target_year_data['SOH (%)'].iloc[0]
            target_ratio = target_year_data['Capacity Ratio'].iloc[0]
            
            capacity_loss_15 = initial_capacity - final_capacity_15
            years_above_target = sum(1 for ratio in degradation_df['Capacity Ratio'][:16] if ratio >= min_capacity_ratio)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_text = f"-{(100 - final_soh_15):.2f}%"
                st.metric(
                    "SOH at 15 Years (EOL)",
                    f"{final_soh_15:.2f}%",
                    delta_text
                )
                st.caption("End of Life warranty period")
            
            with col2:
                st.metric(
                    "Capacity at 15 Years",
                    f"{final_capacity_15:.1f} kWh",
                    f"-{capacity_loss_15:.1f} kWh"
                )
                st.caption("Usable capacity remaining")
            
            with col3:
                st.metric(
                    "Years Above Target Ratio",
                    f"{years_above_target}/16",
                    f"{(years_above_target/16*100):.0f}%"
                )
                st.caption("Within warranty period")
            
            with col4:
                if final_ratio_15 >= 2.0:
                    status = "🟢 Excellent"
                    color = "normal"
                elif final_ratio_15 >= min_capacity_ratio:
                    status = "🟡 Adequate"
                    color = "normal"
                else:
                    status = "🔴 Insufficient"
                    color = "inverse"
                    
                st.metric(
                    "15-Year Capacity Ratio",
                    f"{final_ratio_15:.2f}x",
                    status,
                    delta_color=color
                )
                st.caption(f"vs {min_capacity_ratio:.1f}x target")
            
            # Target year specific analysis
            if target_year != 15:
                st.markdown(f"### 📅 Target Year {target_year} Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"SOH at Year {target_year}", f"{target_soh:.2f}%")
                with col2:
                    st.metric(f"Capacity at Year {target_year}", f"{target_capacity:.1f} kWh")
                with col3:
                    if target_ratio >= min_capacity_ratio:
                        status = "✅ Meets Target"
                        color = "normal"
                    else:
                        status = "❌ Below Target"
                        color = "inverse"
                    st.metric(f"Capacity Ratio Year {target_year}", f"{target_ratio:.2f}x", status, delta_color=color)
            
            # Detailed degradation pattern analysis
            st.markdown("### 📈 Advanced Degradation Pattern Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Early Years Performance (0-5 years):**")
                year_1_loss = 100 - real_degradation_data[1] * degradation_factor if degradation_factor != 1.0 else 100 - real_degradation_data[1]
                year_5_data = degradation_df[degradation_df['Year'] == 5]
                year_5_loss = 100 - year_5_data['SOH (%)'].iloc[0]
                year_5_ratio = year_5_data['Capacity Ratio'].iloc[0]
                
                st.write(f"• Year 1: {year_1_loss:.2f}% capacity loss")
                st.write(f"• Year 5: {year_5_loss:.2f}% total loss")
                st.write(f"• Year 5 ratio: {year_5_ratio:.2f}x")
                
                st.markdown("**Mid-Life Performance (5-15 years):**")
                mid_life_loss = final_soh_15 - year_5_data['SOH (%)'].iloc[0]
                avg_annual_loss = abs(mid_life_loss) / 10
                st.write(f"• Years 5-15: {abs(mid_life_loss):.2f}% additional loss")
                st.write(f"• Average: {avg_annual_loss:.2f}%/year")
                st.write(f"• Final ratio: {final_ratio_15:.2f}x")
            
            with col2:
                st.markdown("**Extended Life Analysis (15-20 years):**")
                extended_loss = calendar_life_soh - final_soh_15
                st.write(f"• Years 15-20: {abs(extended_loss):.2f}% additional loss")
                st.write(f"• Final SOH: {calendar_life_soh:.2f}%")
                st.write(f"• Final ratio: {calendar_life_ratio:.2f}x")
                
                st.markdown("**Comparison with Linear Model:**")
                if show_comparison:
                    linear_15_soh = max(0, 100 - (20 / 15 * 15))
                    real_vs_linear = final_soh_15 - linear_15_soh
                    st.write(f"• Linear model at 15y: {linear_15_soh:.1f}%")
                    st.write(f"• Real WEIHENG data: {final_soh_15:.2f}%")
                    comparison_text = "(better)" if real_vs_linear > 0 else "(worse)"
                    st.write(f"• Difference: {real_vs_linear:+.2f}% {comparison_text}")
                else:
                    st.write("• Enable comparison to see linear model data")
            
            # Investment decision framework
            st.markdown("### 💰 Investment Decision Framework")
            
            if final_ratio_15 < 1.0:
                st.error(f"""
                🚨 **Critical Investment Risk - Not Recommended**
                
                **Issues Identified:**
                - Battery will fail to meet energy requirements by year {[i for i, r in enumerate(degradation_df['Capacity Ratio'][:16]) if r < 1.0][0] if any(r < 1.0 for r in degradation_df['Capacity Ratio'][:16]) else 'unknown'}
                - 15-year capacity ratio: {final_ratio_15:.2f}x (below 1.0x minimum)
                - Insufficient capacity for worst-case MD events
                
                **Required Actions:**
                - Choose higher capacity battery model (consider TIANWU-250-233-1C)
                - Increase system sizing by {((1.2 / final_ratio_15) - 1) * 100:.0f}% minimum
                - Consider hybrid MD management approach
                - Plan for mid-life battery replacement
                """)
                
            elif final_ratio_15 < min_capacity_ratio:
                st.warning(f"""
                ⚠️ **Marginal Investment - High Risk**
                
                **Performance Concerns:**
                - 15-year capacity ratio: {final_ratio_15:.2f}x (below {min_capacity_ratio:.1f}x target)
                - Limited safety margin for performance variations
                - {16 - years_above_target} years may not meet target ratio
                
                **Risk Mitigation:**
                - Implement strict battery management protocols
                - Plan preventive maintenance optimization
                - Monitor real degradation vs predictions
                - Prepare contingency plans for year {16 - years_above_target + 1}+
                """)
                
            elif final_ratio_15 < 2.0:
                st.info(f"""
                ✅ **Viable Investment - Adequate Performance**
                
                **Performance Profile:**
                - 15-year capacity ratio: {final_ratio_15:.2f}x (meets {min_capacity_ratio:.1f}x target)
                - {years_above_target}/16 years meet target ratio
                - Suitable for primary MD shaving application
                
                **Optimization Opportunities:**
                - Monitor real degradation trends
                - Optimize charging/discharging cycles
                - Plan post-warranty operation strategy
                - Consider capacity for load growth
                """)
                
            else:
                st.success(f"""
                🎯 **Excellent Investment - Superior Performance**
                
                **Outstanding Characteristics:**
                - 15-year capacity ratio: {final_ratio_15:.2f}x (well above {min_capacity_ratio:.1f}x target)
                - All warranty years meet target with significant margin
                - Oversized for current application requirements
                
                **Strategic Considerations:**
                - Evaluate smaller, more cost-effective models
                - Explore additional revenue streams (grid services, peak shaving expansion)
                - Conservative operation for life extension
                - Accommodate future load growth
                """)
            
            # Cost analysis section
            if show_cost_analysis:
                st.markdown("### 💵 Financial Analysis Summary")
                
                # Calculate total system costs
                battery_cost = battery_specs["energy_kWh"] * cost_per_kwh
                pcs_cost = battery_specs["power_kW"] * pcs_cost_per_kw
                total_system_cost = (battery_cost + pcs_cost) * installation_factor
                
                # Financial metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Investment", f"RM {total_system_cost:,.0f}")
                with col2:
                    cost_per_kwh_total = total_system_cost / battery_specs["energy_kWh"]
                    st.metric("Cost per kWh", f"RM {cost_per_kwh_total:,.0f}")
                with col3:
                    cost_per_kw_total = total_system_cost / battery_specs["power_kW"]
                    st.metric("Cost per kW", f"RM {cost_per_kw_total:,.0f}")
                with col4:
                    # Estimate MD savings potential
                    monthly_md_savings = max_excess_kw * 35  # Assume RM 35/kW
                    annual_savings = monthly_md_savings * 12
                    simple_payback = total_system_cost / annual_savings if annual_savings > 0 else float('inf')
                    st.metric("Simple Payback", f"{simple_payback:.1f} years" if simple_payback < 50 else ">50 years")
                
                # ROI analysis
                if annual_savings > 0:
                    lifecycle_savings = annual_savings * 15  # 15-year warranty
                    net_benefit = lifecycle_savings - total_system_cost
                    roi_percent = (net_benefit / total_system_cost) * 100
                    
                    if roi_percent > 20:
                        st.success(f"🎯 **Strong ROI:** {roi_percent:.1f}% over 15 years (RM {net_benefit:,.0f} net benefit)")
                    elif roi_percent > 0:
                        st.info(f"📊 **Positive ROI:** {roi_percent:.1f}% over 15 years (RM {net_benefit:,.0f} net benefit)")
                    else:
                        st.error(f"📉 **Negative ROI:** {roi_percent:.1f}% over 15 years (RM {net_benefit:,.0f} net loss)")
            
            # Data Export Section
            st.markdown("### 📊 Data Export & Download")
            st.markdown("**Export the complete 20-year degradation analysis data for external analysis or reporting.**")
            
            # Prepare export DataFrame with all relevant data
            export_df = degradation_df.copy()
            
            # Add additional calculated fields for comprehensive export
            export_df['System_Configuration'] = f"{selected_qty} x {battery_specs['model']}"
            export_df['Initial_System_Capacity_kWh'] = initial_capacity
            export_df['Depth_of_Discharge_Percent'] = depth_of_discharge
            export_df['Discharge_Efficiency_Percent'] = discharge_efficiency
            export_df['Max_Event_Energy_Requirement_kWh'] = max_event_energy
            export_df['Degradation_Factor'] = degradation_factor
            export_df['Min_Capacity_Ratio_Target'] = min_capacity_ratio
            
            # Add performance status columns
            export_df['Meets_Minimum_Ratio'] = export_df['Capacity Ratio'] >= min_capacity_ratio
            export_df['Performance_Category'] = export_df['Capacity Ratio'].apply(
                lambda x: 'Excellent' if x >= 2.0 else 'Adequate' if x >= min_capacity_ratio else 'Marginal' if x >= 1.0 else 'Insufficient'
            )
            
            # Add year-over-year changes
            export_df['SOH_Change_From_Previous_Year'] = export_df['SOH (%)'].diff()
            export_df['Capacity_Change_From_Previous_Year_kWh'] = export_df['Total Capacity (kWh)'].diff()
            export_df['Capacity_Loss_From_Year_0_kWh'] = initial_capacity - export_df['Total Capacity (kWh)']
            export_df['Capacity_Loss_From_Year_0_Percent'] = ((initial_capacity - export_df['Total Capacity (kWh)']) / initial_capacity * 100)
            
            # Reorder columns for better export readability
            export_columns = [
                'Year', 'System_Configuration', 'Initial_System_Capacity_kWh',
                'SOH (%)', 'Total Capacity (kWh)', 'Total Usable (kWh)', 'Capacity Ratio',
                'Depth_of_Discharge_Percent', 'Discharge_Efficiency_Percent',
                'Max_Event_Energy_Requirement_kWh', 'MD Target (kW)',
                'Performance_Category', 'Meets_Minimum_Ratio',
                'SOH_Change_From_Previous_Year', 'Capacity_Change_From_Previous_Year_kWh',
                'Capacity_Loss_From_Year_0_kWh', 'Capacity_Loss_From_Year_0_Percent',
                'Degradation_Factor', 'Min_Capacity_Ratio_Target'
            ]
            
            # Add linear comparison data if enabled
            if show_comparison:
                export_columns.extend(['Linear SOH (%)', 'Linear Capacity (kWh)', 'Max Event Energy (kWh)'])
            
            export_df_final = export_df[export_columns].copy()
            
            # Display preview of export data
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Preview of Export Data:**")
                # Show key columns in preview
                preview_columns = ['Year', 'SOH (%)', 'Total Capacity (kWh)', 'Total Usable (kWh)', 
                                 'Capacity Ratio', 'Performance_Category', 'Meets_Minimum_Ratio']
                preview_df = export_df_final[preview_columns].head(10)
                st.dataframe(preview_df, use_container_width=True)
                
                if len(export_df_final) > 10:
                    st.caption(f"Showing first 10 rows of {len(export_df_final)} total rows. Full data available in export.")
            
            with col2:
                st.markdown("**Export Options:**")
                
                # Export metadata
                export_metadata = {
                    'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Battery_Model': battery_specs['model'],
                    'System_Quantity': selected_qty,
                    'Total_System_Capacity_kWh': initial_capacity,
                    'Analysis_Target_Year': target_year,
                    'Degradation_Factor_Applied': degradation_factor,
                    'Total_Rows': len(export_df_final),
                    'Data_Source': 'Real WEIHENG TIANWU Series Test Data'
                }
                
                st.json(export_metadata)
            
            # Export buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                csv_data = export_df_final.to_csv(index=False)
                filename_csv = f"WEIHENG_Battery_Degradation_{battery_specs['model']}_{selected_qty}units_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
                
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=filename_csv,
                    mime="text/csv",
                    help="Download complete degradation data as CSV file for Excel analysis"
                )
            
            with col2:
                # Excel Export (using CSV format but with .xlsx extension for compatibility)
                filename_excel = f"WEIHENG_Battery_Degradation_{battery_specs['model']}_{selected_qty}units_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx"
                
                # Create Excel-compatible CSV
                excel_data = export_df_final.to_csv(index=False, sep=',')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=filename_excel,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download as Excel-compatible file for advanced analysis"
                )
            
            with col3:
                # JSON Export for API/programming use
                json_export = {
                    'metadata': export_metadata,
                    'degradation_data': export_df_final.to_dict('records')
                }
                
                json_data = json.dumps(json_export, indent=2, default=str)
                filename_json = f"WEIHENG_Battery_Degradation_{battery_specs['model']}_{selected_qty}units_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json"
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=filename_json,
                    mime="application/json",
                    help="Download as JSON for programming/API integration"
                )
            
            # Additional export information
            with st.expander("ℹ️ Export Data Documentation"):
                st.markdown("""
                **Column Descriptions:**
                
                **Basic Data:**
                - `Year`: Analysis year (0-20)
                - `SOH (%)`: State of Health percentage
                - `Total Capacity (kWh)`: Total system capacity after degradation
                - `Total Usable (kWh)`: Usable capacity considering DoD and discharge efficiency
                - `Capacity Ratio`: Ratio of usable capacity to energy requirement
                - `MD Target (kW)`: Maximum demand target power
                
                **Performance Analysis:**
                - `Performance_Category`: Excellent/Adequate/Marginal/Insufficient
                - `Meets_Minimum_Ratio`: Boolean indicator if meets minimum capacity ratio
                
                **Degradation Tracking:**
                - `SOH_Change_From_Previous_Year`: Year-over-year SOH change
                - `Capacity_Loss_From_Year_0_kWh`: Cumulative capacity loss from new
                - `Capacity_Loss_From_Year_0_Percent`: Cumulative capacity loss percentage
                
                **Configuration Parameters:**
                - `System_Configuration`: Battery model and quantity
                - `Depth_of_Discharge_Percent`: Applied DoD setting
                - `Discharge_Efficiency_Percent`: Applied discharge efficiency
                - `Degradation_Factor`: Sensitivity factor applied
                
                **Data Source:**
                This export contains real WEIHENG TIANWU series degradation data from laboratory testing,
                not theoretical linear degradation models. Use this data for accurate long-term planning
                and financial analysis.
                """)
            
            st.markdown("---")
            
            # Action items and next steps
            st.markdown("### 📋 Recommended Next Steps")
            
            next_steps = []
            
            if final_ratio_15 >= min_capacity_ratio:
                next_steps.extend([
                    "✅ **Proceed with detailed engineering design**",
                    "📋 **Obtain formal quotations from WEIHENG or certified distributors**",
                    "🔧 **Conduct site survey for installation requirements**",
                    "📄 **Prepare regulatory applications and grid connection permits**"
                ])
            else:
                next_steps.extend([
                    "🔄 **Re-evaluate battery sizing requirements**",
                    "🔍 **Consider alternative battery models or configurations**",
                    "📊 **Analyze hybrid MD management strategies**",
                    "💼 **Reassess financial justification**"
                ])
            
            next_steps.extend([
                "📈 **Develop battery monitoring and maintenance plan**",
                "🎯 **Establish performance tracking KPIs**",
                "⚡ **Design integration with existing electrical systems**",
                "📚 **Prepare operator training and documentation**"
            ])
            
            for i, step in enumerate(next_steps, 1):
                st.write(f"{i}. {step}")
            
            # Add technical considerations
            with st.expander("🔬 Technical Implementation Considerations"):
                st.markdown("""
                **System Integration Requirements:**
                - Grid connection studies and protection coordination
                - Power conversion system (PCS) specification and sizing
                - SCADA/monitoring system integration with existing infrastructure
                - Cooling system design and integration (liquid cooling for battery, air cooling for PCS)
                
                **Operational Considerations:**
                - Daily cycling pattern optimization for MD shaving
                - SOC management strategies to maximize lifespan
                - Preventive maintenance scheduling and spare parts inventory
                - Emergency response procedures and safety protocols
                
                **Performance Monitoring:**
                - Real-time degradation tracking vs predicted curve
                - Energy throughput and cycle counting
                - Temperature monitoring and thermal management
                - Power quality monitoring and grid compliance
                
                **Future Expansion Options:**
                - Modular design for capacity expansion
                - Grid services revenue opportunities (frequency regulation, voltage support)
                - Integration with renewable energy systems
                - Smart grid and demand response participation
                """)
                
        except Exception as e:
            st.error(f"❌ Error in advanced degradation analysis: {str(e)}")
            st.info("Falling back to simplified analysis...")
            
            # Simplified fallback analysis
            simple_15_year_soh = 80  # Conservative estimate
            simple_capacity_15 = battery_specs["energy_kWh"] * 0.8
            simple_ratio = simple_capacity_15 / max_event_energy if max_event_energy > 0 else 0
            
            st.metric("15-Year Capacity (Est.)", f"{simple_capacity_15:.1f} kWh")
            st.metric("Capacity Ratio (Est.)", f"{simple_ratio:.2f}x")
            
            if simple_ratio >= 1.2:
                st.success("✅ Battery appears suitable based on simplified analysis")
            else:
                st.warning("⚠️ Battery may be inadequate based on simplified analysis")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the expected columns and data format.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis.")
        
        # Show example data format
        with st.expander("📋 Example Data Format"):
            example_data = {
                "Start Date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "Start Time": ["14:30", "15:45", "16:15"],
                "End Date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "End Time": ["15:15", "16:30", "17:00"],
                "Peak Load (kW)": [145.2, 167.8, 134.5],
                "Excess (kW)": [25.2, 47.8, 14.5],
                "Duration (min)": [45, 45, 45],
                "Energy to Shave (kWh)": [18.9, 35.85, 10.875],
                "Energy to Shave (Peak Period Only)": [18.9, 35.85, 10.875],
                "MD Cost Impact (RM)": [882.0, 1673.0, 507.5]
            }
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            st.caption("Sample format showing expected columns and data types")

with tabs[6]:
    # ❄️ Chiller Energy Dashboard Tab
    st.title("❄️ Chiller Plant Energy Dashboard")
    st.markdown("""
    Comprehensive analysis of chiller plant energy efficiency, including data upload, 
    column mapping, efficiency calculations, and visualizations.
    """)
    
    # Initialize session state for chiller dashboard
    if 'chiller_uploaded_data' not in st.session_state:
        st.session_state.chiller_uploaded_data = None
    if 'chiller_column_mapping' not in st.session_state:
        st.session_state.chiller_column_mapping = {}
    if 'chiller_processed_data' not in st.session_state:
        st.session_state.chiller_processed_data = None
    
    # Try to import and use the chiller dashboard app_new.py main function
    try:
        # Import the chiller dashboard main function
        import sys
        chiller_path = os.path.join(os.path.dirname(__file__), 'energyanalaysis', 'chiller-energy-dashboard', 'src')
        if chiller_path not in sys.path:
            sys.path.insert(0, chiller_path)
        
        # Import the main function from app_new.py
        from app_new import main as chiller_main
        
        # Call the chiller dashboard main function
        # Note: We need to modify the main function to not call st.set_page_config
        # since it's already set in the main app
        
        # Temporarily store the original session state keys
        original_keys = list(st.session_state.keys())
        
        # Custom CSS for chiller dashboard
        st.markdown("""
        <style>
        .chiller-header {
            font-size: 2rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .chiller-section-header {
            font-size: 1.3rem;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .chiller-metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Import components individually to avoid conflicts
        try:
            from components.data_upload import render_data_upload
            from components.data_preview import render_data_preview
            from components.column_mapper import render_column_mapper, calculate_derived_metrics
            from components.metrics_calculator import render_metrics_display, calculate_efficiency_metrics
            from components.visualizations import render_visualizations
            from components.equipment_performance import render_equipment_performance
            
            # Sidebar for chiller dashboard navigation
            with st.sidebar:
                st.markdown("---")
                st.markdown("### ❄️ Chiller Dashboard")
                chiller_steps = [
                    "📁 Data Upload",
                    "👀 Data Preview", 
                    "🔗 Column Mapping",
                    "📊 Analysis & Results",
                    "🛠️ Equipment Performance"
                ]
                
                # Determine current step based on session state
                chiller_current_step = 0
                if st.session_state.chiller_uploaded_data is not None:
                    chiller_current_step = max(chiller_current_step, 1)
                if st.session_state.chiller_column_mapping:
                    chiller_current_step = max(chiller_current_step, 2)
                if st.session_state.chiller_processed_data is not None:
                    chiller_current_step = max(chiller_current_step, 3)
                    
                chiller_selected_step = st.radio("Chiller Steps:", chiller_steps, index=chiller_current_step, key="chiller_steps")
                
                # Progress indicator
                st.markdown("#### 📊 Progress")
                chiller_progress_value = (chiller_current_step) / (len(chiller_steps) - 1) if len(chiller_steps) > 1 else 0
                st.progress(chiller_progress_value)
                st.caption(f"Step {chiller_current_step + 1} of {len(chiller_steps)}")
                
                # Add some helpful information
                st.markdown("#### 💡 Tips")
                st.info("""
                **Supported Formats:**
                - Excel (.xlsx, .xls)
                - CSV (.csv)
                
                **Required Data:**
                - Timestamp column
                - Power consumption data
                - Cooling load information
                
                **Calculations:**
                - Total Power = Sum of components
                - kW/TR = Total Power / Cooling Load
                - COP = (Cooling Load × 3.51685) / Total Power
                """)
            
            # Main content area based on selected step
            if chiller_selected_step == "📁 Data Upload":
                st.session_state.chiller_uploaded_data = render_data_upload()
                
            elif chiller_selected_step == "👀 Data Preview":
                if st.session_state.chiller_uploaded_data is not None:
                    render_data_preview(st.session_state.chiller_uploaded_data)
                else:
                    st.warning("Please upload data first.")
                    
            elif chiller_selected_step == "🔗 Column Mapping":
                if st.session_state.chiller_uploaded_data is not None:
                    mapping = render_column_mapper(st.session_state.chiller_uploaded_data)
                    if mapping:
                        st.session_state.chiller_column_mapping = mapping
                        
                        # Calculate derived metrics
                        st.session_state.chiller_processed_data = calculate_derived_metrics(
                            st.session_state.chiller_uploaded_data, 
                            mapping
                        )
                        
                        # Show metrics
                        render_metrics_display(st.session_state.chiller_uploaded_data, mapping)
                else:
                    st.warning("Please upload data first.")
                    
            elif chiller_selected_step == "📊 Analysis & Results":
                if st.session_state.chiller_column_mapping and st.session_state.chiller_processed_data is not None:
                    # Show final results and visualizations
                    render_visualizations(st.session_state.chiller_processed_data, st.session_state.chiller_column_mapping)
                    
                    # Export options
                    st.markdown("---")
                    st.subheader("📤 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📊 Download Processed Data", key="chiller_download_data"):
                            csv = st.session_state.chiller_processed_data.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="chiller_analysis_results.csv",
                                mime="text/csv",
                                key="chiller_csv_download"
                            )
                    
                    with col2:
                        if st.button("📋 Download Analysis Report", key="chiller_download_report"):
                            # Generate a simple text report
                            if st.session_state.chiller_uploaded_data is not None:
                                metrics = calculate_efficiency_metrics(
                                    st.session_state.chiller_uploaded_data, 
                                    st.session_state.chiller_column_mapping
                                )
                                
                                report = f"""
Chiller Plant Energy Analysis Report
===================================

Data Overview:
- Total Records: {len(st.session_state.chiller_uploaded_data)}
- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Average kW/TR: {metrics.get('avg_kw_tr', 0):.3f}
- Average COP: {metrics.get('avg_cop', 0):.2f}
- Total kW/TR: {metrics.get('total_kw_tr', 0):.3f}
- Total COP: {metrics.get('total_cop', 0):.2f}

Column Mapping Used:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in st.session_state.chiller_column_mapping.items() if v])}
                                """
                                
                                st.download_button(
                                    label="Download Report",
                                    data=report,
                                    file_name="chiller_analysis_report.txt",
                                    mime="text/plain",
                                    key="chiller_report_download"
                                )
                else:
                    st.warning("Please complete the column mapping step first.")
            
            elif chiller_selected_step == "🛠️ Equipment Performance":
                if st.session_state.chiller_column_mapping and st.session_state.chiller_processed_data is not None:
                    # Show equipment performance analysis
                    render_equipment_performance(
                        st.session_state.chiller_uploaded_data,
                        st.session_state.chiller_processed_data, 
                        st.session_state.chiller_column_mapping
                    )
                else:
                    st.warning("Please complete the column mapping step first to see equipment performance.")
        
        except ImportError as e:
            st.error(f"❌ Chiller Dashboard components not found: {str(e)}")
            st.info("Running in basic mode with file upload only.")
            
            # Basic file upload as fallback
            st.markdown("### 📁 Upload Your Chiller Plant Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your chiller plant data in CSV or Excel format",
                key="basic_chiller_upload"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Rows", f"{len(df):,}")
                    with col2:
                        st.metric("📋 Columns", f"{len(df.columns)}")
                    with col3:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        st.metric("🔢 Numeric Columns", f"{len(numeric_cols)}")
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    
    except Exception as e:
        st.error(f"❌ Error loading chiller dashboard: {str(e)}")
        st.info("The chiller dashboard will be available once the components are properly installed.")

