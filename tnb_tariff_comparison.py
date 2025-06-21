import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def show():
    st.title("TNB New Tariff Comparison")
    st.markdown("""
    This tool allows you to compare the new TNB tariffs for different consumer categories. Select your category and input your consumption details to see a breakdown and comparison of costs under the new tariff structure.
    """)

    # Example categories and tariffs (customize as needed)
    categories = [
        "Industrial (E1, E2, E3, D)",
        "Commercial (C1, C2)",
        "Residential (D)"
    ]
    category = st.selectbox("Select Consumer Category", categories)

    if category == "Industrial (E1, E2, E3, D)":
        st.subheader("Industrial Tariff Comparison")
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

    elif category == "Commercial (C1, C2)":
        st.subheader("Commercial Tariff Comparison")
        monthly_kwh = st.number_input("Enter Monthly Energy Consumption (kWh)", min_value=0.0, value=8000.0, step=100.0)
        max_demand_kw = st.number_input("Enter Maximum Demand (kW)", min_value=0.0, value=300.0, step=10.0)
        c1_rate = 0.435
        c1_icpt = 0.027
        c2_rate = 0.385
        c2_md_rate = 25.0
        c2_icpt = 0.16
        c1_cost = monthly_kwh * (c1_rate + c1_icpt)
        c2_cost = monthly_kwh * (c2_rate + c2_icpt) + max_demand_kw * c2_md_rate
        df = pd.DataFrame({
            "Tariff": ["C1 (Low Voltage Commercial)", "C2 (Medium Voltage Commercial)"],
            "Total Cost (RM)": [c1_cost, c2_cost]
        })
        st.dataframe(df, use_container_width=True)
        st.success("Comparison complete. Adjust the values above to see updated results.")

    elif category == "Residential (D)":
        st.subheader("Residential Tariff Comparison")
        monthly_kwh = st.number_input("Enter Monthly Energy Consumption (kWh)", min_value=0.0, value=500.0, step=10.0)
        # Example tiered rates (replace with actual new tariff rates)
        tiers = [
            (200, 0.218),
            (100, 0.334),
            (300, 0.516),
            (float('inf'), 0.571)
        ]
        remaining = monthly_kwh
        total_cost = 0
        for limit, rate in tiers:
            if remaining > 0:
                kwh_this_tier = min(remaining, limit)
                total_cost += kwh_this_tier * rate
                remaining -= kwh_this_tier
        st.write(f"Estimated Monthly Bill: RM {total_cost:,.2f}")
        st.success("Comparison complete. Adjust the value above to see updated results.")
    else:
        st.info("Please select a valid category.")
