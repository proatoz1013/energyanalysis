import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4


def show():
    st.title("TNB New Tariff Comparison")
    # Inject custom CSS for table styling at the very top
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("""
    This tool allows you to compare the new TNB tariffs for different consumer categories. Select your industry and tariff schedule to see a breakdown and comparison of costs under the new tariff structure.
    """)

    # Step 1: Select User Type
    # This sets the top-level category of the tariff structure.
    # Options:
    # - "Business"    # (also known as Non-Domestic)
    # - "Residential" # (tariff data may be added later)
    tariff_data = get_tariff_data()
    user_types = list(tariff_data.keys())
    # Robust default: 'Business' if present, else first
    default_user_type = 'Business' if 'Business' in user_types else user_types[0]
    user_type_index = user_types.index(default_user_type)
    selected_user_type = st.selectbox(
        "Select User Type",
        user_types,
        index=user_type_index
    )

    # Step 2: Select Tariff Group (under selected User Type)
    # These are specific industry or supply categories within the User Type.
    # Available options under "Business":
    # - "Non Domestic"
    # - "Specific Agriculture"
    # - "Water & Sewerage Operator"
    # - "Street Lighting"
    # - "Co-Generation"
    # - "Traction"
    # - "Bulk"
    # - "Thermal Energy Storage (TES)"
    # - "Backfeed"
    tariff_groups = list(tariff_data[selected_user_type]["Tariff Groups"].keys())
    # Robust default: 'Non Domestic' if present, else tariff_groups[0]
    default_tariff_group = 'Non Domestic' if 'Non Domestic' in tariff_groups else tariff_groups[0]
    tariff_group_index = tariff_groups.index(default_tariff_group)
    selected_tariff_group = st.selectbox(
        "Select Tariff Group",
        tariff_groups,
        index=tariff_group_index
    )

    # Step 3: Select Voltage and Tariff Type
    # These are full tariff definitions under the selected Tariff Group.
    # Format: "<Voltage> <Tariff Type>"
    # Example options under "Business" → "Non Domestic":
    # - "Low Voltage General"
    # - "Low Voltage TOU"
    # - "Medium Voltage General"
    # - "Medium Voltage TOU"
    # - "High Voltage General"
    # - "High Voltage TOU"
    # These dropdowns map to: tariff_data[user_type]["Tariff Groups"][group]["Tariffs"]
    tariffs = tariff_data[selected_user_type]["Tariff Groups"][selected_tariff_group]["Tariffs"]
    tariff_types = [t["Tariff"] for t in tariffs]
    # Robust default: 'Medium Voltage TOU' if present, else first
    default_tariff_type = 'Medium Voltage TOU' if 'Medium Voltage TOU' in tariff_types else tariff_types[0]
    tariff_type_index = tariff_types.index(default_tariff_type)
    selected_tariff_type = st.selectbox(
        "Select Voltage and Tariff Type",
        tariff_types,
        index=tariff_type_index
    )
    

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
       # --- Manual input for public holidays using a multi-select dropdown ---
        st.subheader("Manual Public Holiday Selection")
        st.caption("Select any combination of days as public holidays. The dropdown supports non-consecutive dates.")
        # Suggest a default range based on the data
        if not df.empty:
            min_date = df[timestamp_col].min().date() if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]) else pd.to_datetime(df[timestamp_col], errors="coerce").min().date()
            max_date = df[timestamp_col].max().date() if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]) else pd.to_datetime(df[timestamp_col], errors="coerce").max().date()
            unique_dates = pd.date_range(min_date, max_date).date
        else:
            unique_dates = []
        holiday_options = [d.strftime('%A, %d %B %Y') for d in unique_dates]
        selected_labels = st.multiselect(
            "Select public holidays in the period:",
            options=holiday_options,
            default=[],
            help="Pick all public holidays in the data period. You can select multiple, non-consecutive days."
        )
        # Map back to date objects
        label_to_date = {d.strftime('%A, %d %B %Y'): d for d in unique_dates}
        selected_holidays = [label_to_date[label] for label in selected_labels]
        holidays = set(selected_holidays)
        manual_holiday_count = len(holidays)
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
            # Show Peak Demand (maximum demand during peak periods only)
            peak_demand = get_peak_demand(df, power_col, holidays)
            if peak_demand is not None:
                col3.metric("Peak Demand (kW, Peak Period Only)", f"{peak_demand:,.2f}")
            else:
                col3.metric("Peak Demand (kW, Peak Period Only)", "N/A")
        # --- Calculate % of peak and off-peak period and show as bar chart ---
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
            holidays_set = set(holidays)
            weekday_count = 0
            weekend_count = 0
            holiday_count = 0
            for d in unique_dates:
                if d in holidays_set:
                    holiday_count += 1
                elif pd.Timestamp(d).weekday() >= 5:
                    weekend_count += 1
                else:
                    weekday_count += 1
            total_days = len(unique_dates)
            st.markdown("**Number of Days:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Weekdays", weekday_count)
            col2.metric("Weekends", weekend_count)
            col3.metric("Holidays", holiday_count)
            col4.metric("Total Days", total_days)
            st.caption("Total = Weekdays + Weekends + Holidays")

        # --- AFA Input ---
        st.markdown("**AFA (Additional Fuel Adjustment) Rate**")
        st.caption("Maximum allowable AFA is 3 cents (0.03 RM/kWh). Any value above requires government approval.")
        afa_rate_cent = st.number_input(
            "Enter AFA Rate (cent/kWh, optional)",
            min_value=-10.0, max_value=10.0, value=3.0, step=0.1, format="%.1f", key="afa_rate_input_cent"
        )
        afa_rate = afa_rate_cent / 100  # Convert cent to RM

        # --- Cost Calculation and Display ---
        from utils.cost_calculator import calculate_cost, format_cost_breakdown
        st.subheader("Cost Breakdown for Selected Tariff")
        cost_breakdown = calculate_cost(df, selected_tariff_obj, power_col, holidays, afa_rate=afa_rate)
        if "error" in cost_breakdown:
            st.error(cost_breakdown["error"])
        else:

            # --- HTML Table for Cost Breakdown (before Streamlit table) ---
            def html_cost_table(breakdown):
                def fmt(val):
                    if val is None or val == "":
                        return ""
                    if isinstance(val, (int, float)):
                        return f"{val:,.2f}"
                    return val
                html = """
                <table class=\"cost-table\">
                    <tr>
                        <th>No</th>
                        <th>Description</th>
                        <th>Unit</th>
                        <th>Value</th>
                        <th>Unit Rate (RM)</th>
                        <th>Total Cost (RM)</th>
                    </tr>
                    <tr class=\"section\">
                        <td>A</td>
                        <td class=\"left\"><b>A. Energy Consumption kWh</b></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td class=\"left\">Peak Period Consumption</td>
                        <td>kWh</td>
                        <td>{peak_kwh}</td>
                        <td>{peak_rate}</td>
                        <td>{peak_cost}</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td class=\"left\">Off-Peak Consumption</td>
                        <td>kWh</td>
                        <td>{offpeak_kwh}</td>
                        <td>{offpeak_rate}</td>
                        <td>{offpeak_cost}</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td class=\"left\">AFA Consumption</td>
                        <td>kWh</td>
                        <td>{afa_kwh}</td>
                        <td>{afa_rate}</td>
                        <td>{afa_cost}</td>
                    </tr>
                    <tr class=\"section\">
                        <td>B</td>
                        <td class=\"left\"><b>B. Maximum Demand (Peak Demand)</b></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td class="left">Capacity Charge</td>
                        <td>kW</td>
                        <td>{peak_demand}</td>
                        <td>{capacity_rate}</td>
                        <td>{capacity_cost}</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td class="left">Network Charge</td>
                        <td>kW</td>
                        <td>{peak_demand}</td>
                        <td>{network_rate}</td>
                        <td>{network_cost}</td>
                    </tr>
                </table>
                """.format(
                    peak_kwh=fmt(breakdown.get("Peak kWh", "")),
                    peak_rate=fmt(breakdown.get("Peak Rate", "")),
                    peak_cost=fmt(breakdown.get("Peak Energy Cost", "")),
                    offpeak_kwh=fmt(breakdown.get("Off-Peak kWh", "")),
                    offpeak_rate=fmt(breakdown.get("Off-Peak Rate", "")),
                    offpeak_cost=fmt(breakdown.get("Off-Peak Energy Cost", "")),
                    afa_kwh=fmt(breakdown.get("AFA kWh", "")),
                    afa_rate=fmt(breakdown.get("AFA Rate", "")),
                    afa_cost=fmt(breakdown.get("AFA Adjustment", "")),
                    peak_demand=fmt(breakdown.get("Peak Demand (kW, Peak Period Only)", "")),
                    network_rate=fmt(breakdown.get("Network Rate", "")),
                    network_cost=fmt(breakdown.get("Network Cost", "")),
                    capacity_rate=fmt(breakdown.get("Capacity Rate", "")),
                    capacity_cost=fmt(breakdown.get("Capacity Cost", "")),
                    others_charges=fmt(breakdown.get("Others Charges", 0)),
                    total_cost=fmt(breakdown.get("Total Cost", "")),
                )
                return html

            st.write("DEBUG: cost_breakdown", cost_breakdown)
            st.markdown(html_cost_table(cost_breakdown), unsafe_allow_html=True)

            # --- Pie Chart for Cost Breakdown ---
            pie_labels = []
            pie_values = []
            pie_colors = []
            # Use cost_breakdown dict directly for pie chart
            if "Peak Energy Cost" in cost_breakdown and cost_breakdown.get("Peak Energy Cost", 0):
                pie_labels.append("Peak Period Consumption")
                pie_values.append(cost_breakdown.get("Peak Energy Cost", 0))
                pie_colors.append("orange")
            if "Off-Peak Energy Cost" in cost_breakdown and cost_breakdown.get("Off-Peak Energy Cost", 0):
                pie_labels.append("Off-Peak Consumption")
                pie_values.append(cost_breakdown.get("Off-Peak Energy Cost", 0))
                pie_colors.append("blue")
            if "AFA Adjustment" in cost_breakdown and cost_breakdown.get("AFA Adjustment", 0):
                pie_labels.append("AFA Consumption")
                pie_values.append(cost_breakdown.get("AFA Adjustment", 0))
                pie_colors.append("green")
            if "Capacity Cost" in cost_breakdown and cost_breakdown.get("Capacity Cost", 0):
                pie_labels.append("Maximum Demand (Peak Demand)")
                pie_values.append(cost_breakdown.get("Capacity Cost", 0))
                pie_colors.append("red")
            if "Network Cost" in cost_breakdown and cost_breakdown.get("Network Cost", 0):
                pie_labels.append("Network Charge")
                pie_values.append(cost_breakdown.get("Network Cost", 0))
                pie_colors.append("red")
            if "Retail Cost" in cost_breakdown and cost_breakdown.get("Retail Cost", 0):
                pie_labels.append("Retail Charge")
                pie_values.append(cost_breakdown.get("Retail Cost", 0))
                pie_colors.append("grey")
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
                formulae.append(f"Peak Energy Cost = Peak kWh × Peak Rate = {cost_breakdown.get('Peak kWh', 0):,.2f} × {cost_breakdown.get('Peak Rate', '–')} = {cost_breakdown.get('Peak Energy Cost', 0):,.2f}")
            if "Off-Peak kWh" in cost_breakdown:
                formulae.append(f"Off-Peak Energy Cost = Off-Peak kWh × Off-Peak Rate = {cost_breakdown.get('Off-Peak kWh', 0):,.2f} × {cost_breakdown.get('Off-Peak Rate', '–')} = {cost_breakdown.get('Off-Peak Energy Cost', 0):,.2f}")
            if "AFA kWh" in cost_breakdown:
                formulae.append(f"AFA Adjustment = AFA kWh × AFA Rate = {cost_breakdown.get('AFA kWh', 0):,.2f} × {cost_breakdown.get('AFA Rate', '–')} = {cost_breakdown.get('AFA Adjustment', 0):,.2f}")
            if "Max Demand (kW)" in cost_breakdown:
                formulae.append(f"Maximum Demand Cost = Max Demand × Capacity Rate = {cost_breakdown.get('Max Demand (kW)', 0):,.2f} × {cost_breakdown.get('Capacity Rate', '–')} = {cost_breakdown.get('Capacity Cost', 0):,.2f}")
            if "Network Cost" in cost_breakdown:
                formulae.append(f"Network Cost = Max Demand × Network Rate = {cost_breakdown.get('Max Demand (kW)', 0):,.2f} × {cost_breakdown.get('Network Rate', '–')} = {cost_breakdown.get('Network Cost', 0):,.2f}")
            if "Retail Cost" in cost_breakdown:
                formulae.append(f"Retail Cost = {cost_breakdown.get('Retail Cost', 0):,.2f}")
            for f in formulae:
                st.markdown(f"- {f}")

def get_peak_demand(df, power_col, holidays):
    """
    Returns the maximum demand (kW) during peak periods only.
    """
    if df.empty:
        return None
    is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays))
    if is_peak.any():
        return df.loc[is_peak, power_col].max()
    return None
