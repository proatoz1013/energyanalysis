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

    # 3. Tariff Type (Tariff names for selected industry)
    tariff_types = [t["Tariff"] for t in tariff_data[selected_tariff_industry]["Tariffs"]]
    selected_tariff_type = st.selectbox("Select Voltage and Tariff Type", tariff_types, index=0)

    # Find the selected tariff object
    selected_tariff_obj = next((t for t in tariff_data[selected_tariff_industry]["Tariffs"] if t["Tariff"] == selected_tariff_type), None)
    if not selected_tariff_obj:
        st.error("Selected tariff details not found.")
        return

    st.info(f"Selected Industry: {selected_industry} | Tariff Industry: {selected_tariff_industry} | Tariff Type: {selected_tariff_type}")
    st.json(selected_tariff_obj)
    # ...existing or new calculation logic can follow here...

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
