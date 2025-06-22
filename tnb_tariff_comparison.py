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

    # Step 1: Select User Type
    tariff_data = get_tariff_data()
    user_types = list(tariff_data.keys())
    selected_user_type = st.selectbox("Select User Type", user_types, index=0)

    # Step 2: Select Tariff Group (under selected User Type)
    tariff_groups = list(tariff_data[selected_user_type]["Tariff Groups"].keys())
    selected_tariff_group = st.selectbox("Select Tariff Group", tariff_groups, index=0)

    # Step 3: Select Voltage and Tariff Type
    tariffs = tariff_data[selected_user_type]["Tariff Groups"][selected_tariff_group]["Tariffs"]
    tariff_types = [t["Tariff"] for t in tariffs]
    selected_tariff_type = st.selectbox("Select Voltage and Tariff Type", tariff_types, index=0)

    # Find the selected tariff object
    selected_tariff_obj = next((t for t in tariffs if t["Tariff"] == selected_tariff_type), None)
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
            col3.metric("Maximum Demand kW", f"{df[power_col].max():,.2f}")

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

        # --- AFA Input ---
        st.markdown("**AFA (Additional Fuel Adjustment) Rate**")
        st.caption("Maximum allowable AFA is 3 cents (0.03 RM/kWh). Any value above requires government approval.")
        afa_rate_cent = st.number_input(
            "Enter AFA Rate (cent/kWh, optional)",
            min_value=-10.0, max_value=10.0, value=3.0, step=0.1, format="%.1f", key="afa_rate_input_cent"
        )
        afa_rate = afa_rate_cent / 100  # Convert cent to RM

        # --- Cost Calculation and Display ---
        from utils.cost_calculator import calculate_cost
        st.subheader("Cost Breakdown for Selected Tariff")
        cost_breakdown = calculate_cost(df, selected_tariff_obj, power_col, holidays, afa_rate=afa_rate)
        if "error" in cost_breakdown:
            st.error(cost_breakdown["error"])
        else:
            # Build a DataFrame with columns: Description, Unit, Value, Unit Rate (RM), Total Cost (RM)
            table_rows = []
            def safe_get(d, key, default="–"):
                return d.get(key, default) if isinstance(d, dict) else default

            # --- Energy Section ---
            table_rows.append({
                "Description": "A. Energy Consumption kWh",
                "Unit": "kWh",
                "Value": f"{safe_get(cost_breakdown, 'Total kWh', 0):,.2f}",
                "Unit Rate (RM)": "–",
                "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Energy Cost', 0):,.2f}"
            })
            if "Peak kWh" in cost_breakdown:
                table_rows.append({
                    "Description": "Peak Period Consumption",
                    "Unit": "kWh",
                    "Value": f"{safe_get(cost_breakdown, 'Peak kWh', 0):,.2f}",
                    "Unit Rate (RM)": safe_get(cost_breakdown, 'Peak Rate', '–'),
                    "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Peak Energy Cost', 0):,.2f}"
                })
            if "Off-Peak kWh" in cost_breakdown:
                table_rows.append({
                    "Description": "Off-Peak Consumption",
                    "Unit": "kWh",
                    "Value": f"{safe_get(cost_breakdown, 'Off-Peak kWh', 0):,.2f}",
                    "Unit Rate (RM)": safe_get(cost_breakdown, 'Off-Peak Rate', '–'),
                    "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Off-Peak Energy Cost', 0):,.2f}"
                })

            # --- AFA Section (if present) ---
            if "AFA kWh" in cost_breakdown:
                table_rows.append({
                    "Description": "AFA Consumption",
                    "Unit": "kWh",
                    "Value": f"{safe_get(cost_breakdown, 'AFA kWh', 0):,.2f}",
                    "Unit Rate (RM)": safe_get(cost_breakdown, 'AFA Rate', '–'),
                    "Total Cost (RM)": f"{safe_get(cost_breakdown, 'AFA Adjustment', 0):,.2f}"
                })

            # --- Maximum Demand Section ---
            table_rows.append({
                "Description": "B. Maximum Demand",
                "Unit": "kW",
                "Value": f"{safe_get(cost_breakdown, 'Max Demand', 0):,.2f}",
                "Unit Rate (RM)": "–",
                "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Capacity Cost', 0):,.2f}"
            })
            if "Network Cost" in cost_breakdown:
                table_rows.append({
                    "Description": "Network Charge",
                    "Unit": "kW",
                    "Value": f"{safe_get(cost_breakdown, 'Max Demand', 0):,.2f}",
                    "Unit Rate (RM)": f"{safe_get(cost_breakdown, 'Network Rate', '–')}",
                    "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Network Cost', 0):,.2f}"
                })
            if "Retail Cost" in cost_breakdown:
                table_rows.append({
                    "Description": "Retail Charge",
                    "Unit": "–",
                    "Value": "–",
                    "Unit Rate (RM)": "–",
                    "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Retail Cost', 0):,.2f}"
                })

            # --- Blank row for spacing ---
            table_rows.append({"Description": "", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": ""})

            # --- Total row ---
            table_rows.append({
                "Description": "💰 Total Estimated Cost",
                "Unit": "",
                "Value": "",
                "Unit Rate (RM)": "",
                "Total Cost (RM)": f"{safe_get(cost_breakdown, 'Total Cost', 0):,.2f}"
            })

            cost_df = pd.DataFrame(table_rows, columns=["Description", "Unit", "Value", "Unit Rate (RM)", "Total Cost (RM)"])
            st.dataframe(cost_df, use_container_width=True)

            # --- Pie Chart for Cost Breakdown ---
            pie_labels = []
            pie_values = []
            pie_colors = []
            # Only include rows with cost and not blank/total rows
            for row in table_rows:
                desc = row["Description"]
                cost = row["Total Cost (RM)"]
                if desc and desc not in ["", "💰 Total Estimated Cost"] and cost not in ["", "–"]:
                    try:
                        val = float(str(cost).replace(",", ""))
                        if val > 0:
                            pie_labels.append(desc)
                            pie_values.append(val)
                            # Assign color by type
                            if "Peak" in desc:
                                pie_colors.append("orange")
                            elif "Off-Peak" in desc:
                                pie_colors.append("blue")
                            elif "AFA" in desc:
                                pie_colors.append("green")
                            elif "Maximum Demand" in desc or "Network" in desc:
                                pie_colors.append("red")
                            else:
                                pie_colors.append("grey")
                    except Exception:
                        continue
            if pie_labels and pie_values:
                fig = px.pie(
                    names=pie_labels,
                    values=pie_values,
                    color=pie_labels,
                    color_discrete_sequence=pie_colors,
                    title="Cost Breakdown Pie Chart"
                )
                fig.update_traces(textinfo='label+percent', textfont_size=18)
                st.plotly_chart(fig, use_container_width=True)

            # --- Regression Formula Display ---
            st.subheader("Cost Calculation Formulae")
            formulae = []
            if "Peak kWh" in cost_breakdown:
                formulae.append(f"Peak Energy Cost = Peak kWh × Peak Rate = {safe_get(cost_breakdown, 'Peak kWh', 0):,.2f} × {safe_get(cost_breakdown, 'Peak Rate', '–')} = {safe_get(cost_breakdown, 'Peak Energy Cost', 0):,.2f}")
            if "Off-Peak kWh" in cost_breakdown:
                formulae.append(f"Off-Peak Energy Cost = Off-Peak kWh × Off-Peak Rate = {safe_get(cost_breakdown, 'Off-Peak kWh', 0):,.2f} × {safe_get(cost_breakdown, 'Off-Peak Rate', '–')} = {safe_get(cost_breakdown, 'Off-Peak Energy Cost', 0):,.2f}")
            if "AFA kWh" in cost_breakdown:
                formulae.append(f"AFA Adjustment = AFA kWh × AFA Rate = {safe_get(cost_breakdown, 'AFA kWh', 0):,.2f} × {safe_get(cost_breakdown, 'AFA Rate', '–')} = {safe_get(cost_breakdown, 'AFA Adjustment', 0):,.2f}")
            if "Max Demand" in cost_breakdown:
                formulae.append(f"Maximum Demand Cost = Max Demand × Capacity Rate = {safe_get(cost_breakdown, 'Max Demand', 0):,.2f} × {safe_get(cost_breakdown, 'Capacity Rate', '–')} = {safe_get(cost_breakdown, 'Capacity Cost', 0):,.2f}")
            if "Network Cost" in cost_breakdown:
                formulae.append(f"Network Cost = Max Demand × Network Rate = {safe_get(cost_breakdown, 'Max Demand', 0):,.2f} × {safe_get(cost_breakdown, 'Network Rate', '–')} = {safe_get(cost_breakdown, 'Network Cost', 0):,.2f}")
            if "Retail Cost" in cost_breakdown:
                formulae.append(f"Retail Cost = {safe_get(cost_breakdown, 'Retail Cost', 0):,.2f}")
            for f in formulae:
                st.markdown(f"- {f}")
