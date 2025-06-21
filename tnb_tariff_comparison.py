import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tariffs.rp4_tariffs import get_tariff_data


def show():
    st.title("TNB New Tariff Comparison")
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

    st.info(f"Selected Industry: {selected_industry} | Tariff Industry: {selected_tariff_industry} | Tariff Type: {selected_tariff_type}")
    st.json(selected_tariff_obj)

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
        from utils.cost_calculator import calculate_cost
        st.subheader("Cost Breakdown for Selected Tariff")
        # Get holidays for all years in data
        years = df["Parsed Timestamp"].dt.year.unique()
        from utils.holiday_api import get_malaysia_public_holidays
        holidays = set()
        for year in years:
            holidays.update(get_malaysia_public_holidays(year))
        cost_breakdown = calculate_cost(df, selected_tariff_obj, power_col, holidays)
        if "error" in cost_breakdown:
            st.error(cost_breakdown["error"])
        else:
            st.json(cost_breakdown)
            # Optionally, show as a table
            cost_df = pd.DataFrame(list(cost_breakdown.items()), columns=["Item", "Value"])
            st.dataframe(cost_df, use_container_width=True)
    # ...existing code after file upload...

    if selected_tariff_industry == "Non Domestic":
        st.subheader("Non-Domestic Tariff Comparison")
        monthly_kwh = st.number_input("Enter Monthly Energy Consumption (kWh)", min_value=0.0, value=10000.0, step=100.0)
        max_demand_kw = st.number_input("Enter Maximum Demand (kW)", min_value=0.0, value=500.0, step=10.0)
        # Example rates (replace with actual new tariff rates)
        e1_rate = 0.337
        e2_peak_rate = 0.355
        e2_offpeak_rate = 0.219
        e2_peak_kwh = st.number_input("E2: Peak Energy (kWh)", min_value=0.0, value=6000.0, step=100.0)
        e2_offpeak_kwh = monthly_kwh - e2_peak_kwh
        e3_peak_rate = 0.337
        e3_offpeak_rate = 0.202
        d_tier1_limit = 200
        d_tier1_rate = 0.38
        d_tier2_rate = 0.441
        md_rate_e1 = 29.60
        md_rate_e2 = 37.00
        md_rate_e3 = 35.50
        # Calculate costs
        e1_cost = monthly_kwh * e1_rate + max_demand_kw * md_rate_e1
        e2_cost = e2_peak_kwh * e2_peak_rate + e2_offpeak_kwh * e2_offpeak_rate + max_demand_kw * md_rate_e2
        e3_cost = e2_peak_kwh * e3_peak_rate + e2_offpeak_kwh * e3_offpeak_rate + max_demand_kw * md_rate_e3
        if monthly_kwh <= d_tier1_limit:
            d_cost = monthly_kwh * d_tier1_rate
        else:
            d_cost = d_tier1_limit * d_tier1_rate + (monthly_kwh - d_tier1_limit) * d_tier2_rate
        # Display table
        df = pd.DataFrame({
            "Tariff": ["E1", "E2", "E3", "D (Low Voltage Industrial)"],
            "Total Cost (RM)": [e1_cost, e2_cost, e3_cost, d_cost]
        })
        st.dataframe(df, use_container_width=True)
        st.success("Comparison complete. Adjust the values above to see updated results.")

    else:
        st.warning(f"Tariff comparison for '{selected_tariff_industry}' is not yet implemented.")
