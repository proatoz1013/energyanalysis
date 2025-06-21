import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tariffs.rp4_tariffs import get_tariff_data


def show():
    st.title("TNB New Tariff Comparison")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("""
    This tool allows you to compare the new TNB tariffs for different consumer categories. Select your industry and tariff schedule to see a breakdown and comparison of costs under the new tariff structure.
    """)

    # 1. Select Non Domestic (only one option for now)
    industry_options = ["Non Domestic"]
    selected_industry = st.selectbox("Select Non Domestic / Domestic", industry_options, index=0)

    # 2. Tariff Industry (keys from rp4_tariffs, e.g. Non Domestic, Specific Agriculture, ...)
    tariff_data = get_tariff_data()
    tariff_industries = list(tariff_data.keys())
    selected_tariff_industry = st.selectbox("Select Tariff Industry", tariff_industries, index=0)

    # 3. Voltage and Tariff Type (Tariff names for selected industry)
    tariff_types = [t["Tariff"] for t in tariff_data[selected_tariff_industry]["Tariffs"]]
    selected_tariff_type = st.selectbox("Select Voltage and Tariff Type", tariff_types, index=0)

    # Find the selected tariff object
    selected_tariff_obj = next((t for t in tariff_data[selected_tariff_industry]["Tariffs"] if t["Tariff"] == selected_tariff_type), None)
    if not selected_tariff_obj:
        st.error("Selected tariff details not found.")
        return

    # --- File uploader and column selection ---
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"], key="tariff_file_uploader")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded and read successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.subheader("Column Selection")
        timestamp_col = st.selectbox("Select timestamp column", df.columns, key="timestamp_col_selector")
        power_col = st.selectbox("Select power (kW) column", df.select_dtypes(include='number').columns, key="power_col_selector")

        # --- Calculate period of start time and end time ---
        df["Parsed Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=["Parsed Timestamp"])
        # --- Calculate peak/off-peak mask before metrics ---
        from tariffs.peak_logic import is_peak_rp4
        # For initial metrics, treat all holidays as empty set (will be updated after manual input)
        is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, set()))
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            start_time = df["Parsed Timestamp"].min()
            end_time = df["Parsed Timestamp"].max()
            col1.metric("Start Time", start_time.strftime("%Y-%m-%d %H:%M"))
            col2.metric("End Time", end_time.strftime("%Y-%m-%d %H:%M"))
            # Calculate total kWh
            # Assume power_col is in kW, and data is at regular intervals (e.g., 15-min, 30-min, 1-hour)
            df = df.sort_values("Parsed Timestamp")
            if len(df) > 1:
                # Calculate time delta in hours between rows
                time_deltas = df["Parsed Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
                # Energy per interval = kW * hours
                interval_kwh = df[power_col] * time_deltas
                total_kwh = interval_kwh.sum()
            else:
                total_kwh = 0
            col3.metric("Total Energy (kWh)", f"{total_kwh:,.2f}")
            # Calculate maximum demand for peak and off-peak periods
            peak_max_demand = df.loc[is_peak, power_col].max() if not df.loc[is_peak].empty else 0
            offpeak_max_demand = df.loc[~is_peak, power_col].max() if not df.loc[~is_peak].empty else 0
            col3.metric("Maximum Demand (Peak) kW", f"{peak_max_demand:,.2f}")
            col3.metric("Maximum Demand (Off-Peak) kW", f"{offpeak_max_demand:,.2f}")

        # --- Manual input for number of public holidays ---
        st.subheader("Manual Public Holiday Count")
        manual_holiday_count = st.number_input(
            "Enter number of public holidays in the period:",
            min_value=0, value=0, step=1, key="manual_holiday_count_input"
        )
        # Select the first N unique dates as holidays
        unique_dates = df["Parsed Timestamp"].dt.date.unique()
        holidays = set(unique_dates[:manual_holiday_count])

        # --- Calculate % of peak and off-peak period and show as bar chart ---
        from tariffs.peak_logic import is_peak_rp4
        import plotly.express as px
        is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays))
        # Calculate kWh for peak and off-peak
        df = df.sort_values("Parsed Timestamp")
        time_deltas = df["Parsed Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
        interval_kwh = df[power_col] * time_deltas
        peak_kwh = interval_kwh[is_peak].sum()
        offpeak_kwh = interval_kwh[~is_peak].sum()
        total_kwh = interval_kwh.sum()
        peak_pct = (peak_kwh / total_kwh) * 100 if total_kwh else 0
        offpeak_pct = (offpeak_kwh / total_kwh) * 100 if total_kwh else 0
        pie_df = pd.DataFrame({
            'Period': ['Peak', 'Off-Peak'],
            'kWh': [peak_kwh, offpeak_kwh],
            'Percentage': [peak_pct, offpeak_pct],
            'Label': [
                f"Peak: {peak_kwh:,.2f} kWh ({peak_pct:.1f}%)",
                f"Off-Peak: {offpeak_kwh:,.2f} kWh ({offpeak_pct:.1f}%)"
            ]
        })
        fig = px.bar(
            pie_df,
            x='Period',
            y='kWh',
            text='Label',
            color='Period',
            color_discrete_map={'Peak': 'orange', 'Off-Peak': 'blue'},
            title='Peak vs Off-Peak Energy (kWh)'
        )
        fig.update_traces(textposition='outside', textfont_size=28)  # Double the default font size (usually 14)
        # Add 15% margin above the tallest bar for clarity
        max_kwh = pie_df['kWh'].max()
        fig.update_yaxes(range=[0, max_kwh * 1.15])
        fig.update_layout(yaxis_title='Energy (kWh)', xaxis_title='', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Peak: {peak_kwh:,.2f} kWh ({peak_pct:.1f}%) | Off-Peak: {offpeak_kwh:,.2f} kWh ({offpeak_pct:.1f}%)")

        # --- Show number of days: weekday, weekend, and holidays ---
        if not df.empty:
            unique_dates = df["Parsed Timestamp"].dt.date.unique()
            years = pd.Series(unique_dates).apply(lambda d: d.year).unique()
            weekday_count = 0
            weekend_count = 0
            for d in unique_dates:
                if pd.Timestamp(d).weekday() >= 5:
                    weekend_count += 1
                else:
                    weekday_count += 1
            st.markdown(f"**Number of Days:**  ")
            st.markdown(f"- Weekdays: **{weekday_count}**  ")
            st.markdown(f"- Weekends: **{weekend_count}**  ")
            st.markdown(f"- Holidays: **{manual_holiday_count}**  ")

        # --- Cost Calculation and Display ---
        st.subheader("Cost Breakdown for Selected Tariff")
        # --- AFA Input ---
        afa_rate_cents = st.number_input(
            "AFA Rate Adjustment (sen/kWh)",
            min_value=-100.0, max_value=100.0, value=3.0, step=0.1,
            help="Enter the AFA adjustment in sen/kWh. Default is +3.0. Can be negative."
        )
        afa_rate_rm = afa_rate_cents / 100.0
        st.caption("Maximum AFA that TNB can charge without government approval is RM 0.03/kWh (or 3.0 sen/kWh).")
        
        

        # Get holidays for all years in data
        years = df["Parsed Timestamp"].dt.year.unique()
        from utils.holiday_api import get_malaysia_public_holidays
        holidays = set()
        for year in years:
            holidays.update(get_malaysia_public_holidays(year))
        from utils.cost_calculator import calculate_cost
        # --- Call cost calculation rules ---
        cost_breakdown = calculate_cost(df, selected_tariff_obj, power_col, holidays, afa_rate=afa_rate_rm)
        voltage = selected_tariff_obj.get("Voltage", "Low Voltage")
        # --- Table display logic ---
        # Always build the table with all rows for the most detailed (Medium Voltage TOU) structure
        def safe_get(d, key, default="–"):
            return d.get(key, default) if isinstance(d, dict) else default
        def safe_val(val, fmt="{:,}"):
            try:
                if val is None or val == "–":
                    return "–"
                if isinstance(val, str):
                    val = val.replace(",", "")
                val = float(val)
                return fmt.format(val)
            except Exception:
                return "–"
        is_tou = selected_tariff_obj.get("Split", False)
        voltage = selected_tariff_obj.get("Voltage", "Low Voltage")
        # Prepare all values, default to zero or "–" if not present
        # For non-TOU, auto-fill unit rate at row 0 from Standard Rate or Energy Rate
        std_rate = safe_val(
            safe_get(cost_breakdown, 'Standard Rate', safe_get(cost_breakdown, 'Energy Rate', safe_get(cost_breakdown, 'Peak Rate', 0))),
            "{:,.4f}"
        ) if not is_tou else "–"
        # Prepare all values, default to zero or "–" if not present
        energy_kwh = safe_val(safe_get(cost_breakdown, 'Total kWh', 0), "{:,.2f}")
        peak_kwh = safe_val(safe_get(cost_breakdown, 'Peak kWh', 0), "{:,.2f}") if is_tou else "–"
        peak_rate = safe_val(safe_get(cost_breakdown, 'Peak Rate', 0), "{:,.4f}") if is_tou else "–"
        peak_cost = safe_val(safe_get(cost_breakdown, 'Peak Energy Cost', 0), "{:,.2f}") if is_tou else "–"
        offpeak_kwh = safe_val(safe_get(cost_breakdown, 'Off-Peak kWh', 0), "{:,.2f}") if is_tou else "–"
        offpeak_rate = safe_val(safe_get(cost_breakdown, 'Off-Peak Rate', 0), "{:,.4f}") if is_tou else "–"
        offpeak_cost = safe_val(safe_get(cost_breakdown, 'Off-Peak Energy Cost', 0), "{:,.2f}") if is_tou else "–"
        std_cost = safe_val(safe_get(cost_breakdown, 'Energy Cost', 0), "{:,.2f}") if not is_tou else "–"
        afa_kwh = safe_val(safe_get(cost_breakdown, 'AFA kWh', 0), "{:,.2f}")
        afa_cost = safe_val(safe_get(cost_breakdown, 'AFA Adjustment', 0), "{:,.2f}")
        afa_rate_disp = f"{afa_rate_rm:,.4f}"
        max_demand = safe_val(safe_get(cost_breakdown, 'Max Demand (kW)', 0), "{:,.2f}")
        max_demand_rate = safe_val(safe_get(cost_breakdown, 'Max Demand Rate', '–'))
        max_demand_cost = safe_val(safe_get(cost_breakdown, 'Max Demand Cost', 0), "{:,.2f}")
        cap_unit = "kWh" if voltage == "Low Voltage" else "kW"
        cap_value = energy_kwh if voltage == "Low Voltage" else max_demand
        cap_rate = safe_val(safe_get(cost_breakdown, 'Capacity Rate', 0), "{:,.4f}")
        cap_cost = safe_val(safe_get(cost_breakdown, 'Capacity Cost', 0), "{:,.2f}")
        net_value = energy_kwh if voltage == "Low Voltage" else max_demand
        net_rate = safe_val(safe_get(cost_breakdown, 'Network Rate', 0), "{:,.4f}")
        net_cost = safe_val(safe_get(cost_breakdown, 'Network Cost', 0), "{:,.2f}")
        # Build the table rows in the same order always
        table_rows = [
            {"Description": "A. Energy Consumption kWh", "Unit": "kWh", "Value": energy_kwh, "Unit Rate (RM)": std_rate if not is_tou else "–", "Total Cost (RM)": std_cost if not is_tou else None},
            {"Description": "Peak Period Consumption", "Unit": "kWh", "Value": peak_kwh, "Unit Rate (RM)": peak_rate, "Total Cost (RM)": peak_cost},
            {"Description": "Off-Peak Consumption", "Unit": "kWh", "Value": offpeak_kwh, "Unit Rate (RM)": offpeak_rate, "Total Cost (RM)": offpeak_cost},
            {"Description": "AFA Adjustment", "Unit": "kWh", "Value": afa_kwh, "Unit Rate (RM)": afa_rate_disp, "Total Cost (RM)": afa_cost},
            {"Description": "", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": ""},
            {"Description": "B. Maximum Demand (Peak Period)", "Unit": "kW", "Value": max_demand, "Unit Rate (RM)": max_demand_rate, "Total Cost (RM)": max_demand_cost},
            {"Description": "Capacity Charge", "Unit": cap_unit, "Value": cap_value, "Unit Rate (RM)": cap_rate, "Total Cost (RM)": cap_cost},
            {"Description": "Network Charge", "Unit": cap_unit, "Value": net_value, "Unit Rate (RM)": net_rate, "Total Cost (RM)": net_cost},
            {"Description": "", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": ""},
            {"Description": "💰 Total Estimated Cost", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": safe_val(safe_get(cost_breakdown, 'Total Cost', 0), "{:,.2f}")}
        ]
        # If TOU, update row 0's total cost to be the sum of rows 1, 2, 3
        if is_tou:
            try:
                total = 0
                for i in [1, 2, 3]:
                    val = table_rows[i]["Total Cost (RM)"]
                    if isinstance(val, str) and val != "–":
                        val = val.replace(",", "")
                        total += float(val)
                table_rows[0]["Total Cost (RM)"] = f"{total:,.2f}" if total else "–"
            except Exception:
                table_rows[0]["Total Cost (RM)"] = "–"
        # Set row 0's value to the sum of rows 1 and 2, or to row 3 if present and nonzero (TOU only)
        if is_tou:
            try:
                def parse_num(val):
                    if isinstance(val, str) and val != "–":
                        return float(val.replace(",", ""))
                    return float(val) if val not in (None, "–") else 0
                kwh1 = parse_num(table_rows[1]["Value"])
                kwh2 = parse_num(table_rows[2]["Value"])
                kwh3 = parse_num(table_rows[3]["Value"])
                if kwh3 > 0:
                    table_rows[0]["Value"] = f"{kwh3:,.2f}"
                else:
                    table_rows[0]["Value"] = f"{kwh1 + kwh2:,.2f}"
            except Exception:
                pass
        # Now update row 5's total cost to be the sum of rows 6 and 7
        try:
            val6 = table_rows[6]["Total Cost (RM)"]
            val7 = table_rows[7]["Total Cost (RM)"]
            total = 0
            for val in [val6, val7]:
                if isinstance(val, str) and val != "–":
                    val = val.replace(",", "")
                    total += float(val)
            table_rows[5]["Total Cost (RM)"] = f"{total:,.2f}" if total else "–"
        except Exception:
            table_rows[5]["Total Cost (RM)"] = "–"
        # --- Blank row ---
        table_rows.append({"Description": "", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": ""})
        # --- Total row ---
        total_cost = safe_get(cost_breakdown, 'Total Cost', 0)
        table_rows.append({
            "Description": "💰 Total Estimated Cost",
            "Unit": "",
            "Value": "",
            "Unit Rate (RM)": "",
            "Total Cost (RM)": f"{total_cost:,.2f}"
        })
        # --- After table_rows is fully built, add percentage column ---
        # Row indices: 1,2,3 (Peak, Off-Peak, AFA), 6,7 (Capacity, Network), 9 (Total)
        # Only add % if there are at least 10 rows (row 9 exists)
        percent_col = ["" for _ in table_rows]
        if len(table_rows) >= 10:
            try:
                total_cost_row9 = table_rows[9]["Total Cost (RM)"]
                if isinstance(total_cost_row9, str):
                    total_cost_row9 = total_cost_row9.replace(",", "")
                total_cost_row9 = float(total_cost_row9)
                for idx in [1,2,3,6,7]:
                    val = table_rows[idx]["Total Cost (RM)"]
                    if isinstance(val, str):
                        val = val.replace(",", "")
                    try:
                        percent = (float(val) / total_cost_row9 * 100) if total_cost_row9 else 0
                        percent_col[idx] = f"{percent:.1f}%"
                    except Exception:
                        percent_col[idx] = "–"
            except Exception:
                pass
        # Add the new column to the DataFrame
        cost_df = pd.DataFrame(table_rows, columns=["Description", "Unit", "Value", "Unit Rate (RM)", "Total Cost (RM)"])
        cost_df["% of Total"] = percent_col
        st.dataframe(cost_df, use_container_width=True)

        # --- Pie Chart: Breakdown of row 1, 2, 3, and row 5 ---
        try:
            # Get the relevant rows (1, 2, 3, 5)
            pie_rows = [1, 2, 3, 5]
            pie_labels = []
            pie_values = []
            for idx in pie_rows:
                desc = table_rows[idx]["Description"]
                val = table_rows[idx]["Total Cost (RM)"]
                if isinstance(val, str):
                    val = val.replace(",", "")
                try:
                    val = float(val)
                except Exception:
                    val = 0
                # Only include if value is positive and description is not blank
                if val > 0 and desc.strip():
                    pie_labels.append(desc)
                    pie_values.append(val)
            if pie_values and sum(pie_values) > 0:
                pie_fig = px.pie(
                    names=pie_labels,
                    values=pie_values,
                    title="Cost Breakdown: Energy, AFA, and Demand Charges",
                    hole=0.35
                )
                pie_fig.update_traces(textinfo='label+percent', textfont_size=18)
                pie_fig.update_layout(showlegend=True)
                st.plotly_chart(pie_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display cost breakdown pie chart: {e}")

        # --- Calculate and display RM/kWh for Total Estimated Cost ---
        try:
            total_kwh = float(cost_df.loc[cost_df['Description'] == 'A. Energy Consumption kWh', 'Value'].values[0].replace(',', ''))
            rm_per_kwh = total_cost / total_kwh if total_kwh else 0
            st.metric("RM/kWh: Total Estimated Cost", f"{rm_per_kwh:,.4f} RM/kWh")
        except Exception as e:
            st.warning(f"Could not calculate RM/kWh: {e}")

        # --- Pie Chart for Cost Breakdown (Rows 1, 2, 3, and 5) ---
        try:
            # Extract labels and values for rows 1, 2, 3, and 5
            pie_labels = []
            pie_values = []
            pie_colors = ["#FFA726", "#42A5F5", "#66BB6A", "#AB47BC"]  # orange, blue, green, purple
            for idx in [1, 2, 3, 5]:
                desc = cost_df.iloc[idx]["Description"]
                val = cost_df.iloc[idx]["Total Cost (RM)"]
                if isinstance(val, str):
                    val = val.replace(",", "")
                try:
                    val = float(val)
                except Exception:
                    val = 0
                pie_labels.append(desc)
                pie_values.append(val)
            # Only plot if at least one value is > 0
            if any(v > 0 for v in pie_values):
                fig = px.pie(
                    names=pie_labels,
                    values=pie_values,
                    color=pie_labels,
                    color_discrete_sequence=pie_colors,
                    title="Cost Breakdown: Energy, AFA, and Maximum Demand Charges"
                )
                fig.update_traces(textinfo='label+percent+value', textfont_size=18)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not plot cost breakdown pie chart: {e}")

        # --- Regression Formula Display for Key Cost Components ---
        st.subheader("Regression Formulae for Cost Components")
        regression_rows = [1, 2, 3, 5]  # Peak, Off-Peak, AFA, Max Demand (now Capacity+Network)
        regression_labels = [
            "Peak Period Consumption (RM) = Peak kWh × Peak Rate (RM/kWh)",
            "Off-Peak Consumption (RM) = Off-Peak kWh × Off-Peak Rate (RM/kWh)",
            "AFA Adjustment (RM) = AFA kWh × AFA Rate (RM/kWh)",
            "Maximum Demand (RM) = Capacity Charge (RM) + Network Charge (RM)"
        ]
        regression_values = []
        for idx, label in zip(regression_rows, regression_labels):
            try:
                if idx == 5:
                    # Maximum Demand = Capacity + Network
                    cap_row = cost_df.iloc[6]
                    net_row = cost_df.iloc[7]
                    cap_val = cap_row["Total Cost (RM)"]
                    net_val = net_row["Total Cost (RM)"]
                    if isinstance(cap_val, str):
                        cap_val = cap_val.replace(",", "")
                    if isinstance(net_val, str):
                        net_val = net_val.replace(",", "")
                    cap_val = float(cap_val) if cap_val else 0
                    net_val = float(net_val) if net_val else 0
                    regression_values.append(f"**Maximum Demand (RM)**: {cap_val:,.2f} (Capacity) + {net_val:,.2f} (Network) = **{cap_val + net_val:,.2f} RM**")
                else:
                    row = cost_df.iloc[idx]
                    desc = row["Description"]
                    val = row["Total Cost (RM)"]
                    value = row["Value"]
                    rate = row["Unit Rate (RM)"]
                    unit = row["Unit"]
                    if isinstance(val, str):
                        val = val.replace(",", "")
                    if isinstance(value, str):
                        value = value.replace(",", "")
                    if isinstance(rate, str):
                        rate = rate.replace(",", "")
                    regression_values.append(f"**{desc}**: {value} {unit} × {rate} = **{float(value) * float(rate):,.2f} RM**")
            except Exception:
                regression_values.append(f"{label}: –")
        for reg in regression_values:
            st.markdown(reg)
