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

        # --- AFA (Additional Fuel Adjustment) Rate ---
        st.markdown("**AFA (Additional Fuel Adjustment) Rate**")
        st.caption("Maximum allowable AFA is 3 cents (0.03 RM/kWh). Any value above requires government approval.")
        afa_rate_cent = st.number_input(
            "Enter AFA Rate (cent/kWh, optional)",
            min_value=-10.0, max_value=10.0, value=3.0, step=0.1, format="%.1f", key="afa_rate_input_cent"
        )
        afa_rate = afa_rate_cent / 100  # Convert cent to RM

        # --- Cost Calculation and Display ---
        from utils.cost_calculator import calculate_cost, format_cost_breakdown
        # Add dynamic title reflecting the selected tariff
        st.markdown(f"### {selected_user_type} > {selected_tariff_group} > {selected_tariff_type}")
        st.subheader("Cost Breakdown for Selected Tariff")
        cost_breakdown = calculate_cost(df, selected_tariff_obj, power_col, holidays, afa_rate=afa_rate)
        if "error" in cost_breakdown:
            st.error(cost_breakdown["error"])
        else:

            # --- HTML Table for Cost Breakdown (before Streamlit table) ---
            def html_cost_table(breakdown):
                # Update the formatting function to ensure numbers are displayed with commas for thousands separators.
                def fmt(val):
                    if val is None or val == "":
                        return ""
                    if isinstance(val, (int, float)):
                        return f"{val:,.0f}"  # Format numbers with commas for thousands separators.
                    return val

                def get_cost(*keys):
                    for k in keys:
                        if k in breakdown:
                            return breakdown[k]
                    return None
                is_tou = "Peak kWh" in breakdown or "Off-Peak kWh" in breakdown
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
                """
                # Section A: Energy Consumption
                html += "<tr class=\"section\"> <td>A</td> <td class=\"left\"><b>A. Energy Consumption kWh</b></td> <td></td><td></td><td></td><td></td> </tr>"
                # Determine if AFA is applicable (not for Low Voltage General)
                selected_tariff_name = selected_tariff_obj.get('Tariff', '').lower() if selected_tariff_obj else ''
                afa_applicable = not (selected_tariff_name.startswith('low voltage') and 'general' in selected_tariff_name)
                # Check if AFA row should be shown (if applicable and either kWh or adjustment value exists)
                afa_kwh = breakdown.get('AFA kWh', None)
                afa_rate = breakdown.get('AFA Rate', None)
                afa_cost = get_cost('AFA Adjustment', 'AFA Adjustment (RM)')
                if is_tou:
                    if breakdown.get('Peak kWh', None) is not None:
                        html += f"<tr> <td>1</td> <td class=\"left\">Peak Period Consumption</td> <td>kWh</td> <td>{fmt(breakdown.get('Peak kWh'))}</td> <td>{fmt(breakdown.get('Peak Rate'))}</td> <td>{fmt(get_cost('Peak Energy Cost', 'Peak Energy Cost (RM)'))}</td> </tr>"
                    if breakdown.get('Off-Peak kWh', None) is not None:
                        html += f"<tr> <td>2</td> <td class=\"left\">Off-Peak Consumption</td> <td>kWh</td> <td>{fmt(breakdown.get('Off-Peak kWh'))}</td> <td>{fmt(breakdown.get('Off-Peak Rate'))}</td> <td>{fmt(get_cost('Off-Peak Energy Cost', 'Off-Peak Energy Cost (RM)'))}</td> </tr>"
                else:
                    if breakdown.get('Total kWh', None) is not None:
                        html += f"<tr> <td>1</td> <td class=\"left\">Total Consumption</td> <td>kWh</td> <td>{fmt(breakdown.get('Total kWh'))}</td> <td>{fmt(breakdown.get('Energy Rate'))}</td> <td>{fmt(get_cost('Energy Cost', 'Energy Cost (RM)'))}</td> </tr>"
                # AFA row if present and applicable (show if either kWh or adjustment value exists)
                if afa_applicable and (afa_kwh is not None or afa_cost is not None):
                    html += f"<tr> <td></td> <td class=\"left\">AFA Consumption</td> <td>kWh</td> <td>{fmt(afa_kwh)}</td> <td>{fmt(afa_rate)}</td> <td>{fmt(afa_cost)}</td> </tr>"
                # Section B: Demand/Capacity/Network charges
                html += "<tr class=\"section\"> <td>B</td> <td class=\"left\"><b>B. Maximum Demand (Peak Demand)</b></td> <td></td><td></td><td></td><td></td> </tr>"
                # Capacity Charge
                cap_val = breakdown.get('Peak Demand (kW, Peak Period Only)', breakdown.get('Max Demand (kW)', None))
                cap_rate = breakdown.get('Capacity Rate', None)
                cap_cost = get_cost('Capacity Cost', 'Capacity Cost (RM)')
                if cap_val is not None or cap_rate is not None or cap_cost is not None:
                    html += f"<tr> <td>1</td> <td class=\"left\">Capacity Charge</td> <td>kW</td> <td>{fmt(cap_val)}</td> <td>{fmt(cap_rate)}</td> <td>{fmt(cap_cost)}</td> </tr>"
                # Network Charge
                net_val = breakdown.get('Peak Demand (kW, Peak Period Only)', breakdown.get('Max Demand (kW)', None))
                net_rate = breakdown.get('Network Rate', None)
                net_cost = get_cost('Network Cost', 'Network Cost (RM)')
                if net_val is not None or net_rate is not None or net_cost is not None:
                    html += f"<tr> <td>2</td> <td class=\"left\">Network Charge</td> <td>kW</td> <td>{fmt(net_val)}</td> <td>{fmt(net_rate)}</td> <td>{fmt(net_cost)}</td> </tr>"
                # Retail Charge (if present)
                if breakdown.get('Retail Cost', None) is not None:
                    html += f"<tr> <td>3</td> <td class=\"left\">Retail Charge</td> <td></td> <td></td> <td></td> <td>{fmt(breakdown.get('Retail Cost'))}</td> </tr>"
                # Add any other cost rows if present (future extensibility)
                # Total Cost row (if present)
                if get_cost('Total Cost', 'Total Cost (RM)') is not None:
                    html += f"<tr class=\"section\"><td colspan=5 class=\"left\"><b>Total Cost</b></td><td><b>{fmt(get_cost('Total Cost', 'Total Cost (RM)'))}</b></td></tr>"
                html += "</table>"
                return html

            # If either value is missing, do not show the section at all

            # --- Debug section (disabled, enable for future debugging) ---
            # st.write("DEBUG: cost_breakdown", cost_breakdown)
            # afa_kwh = cost_breakdown.get('AFA kwh', cost_breakdown.get('Total kWh', None))
            # afa_rate = cost_breakdown.get('AFA Rate', None)
            # if afa_kwh is not None:
            #     st.write("DEBUG: AFA Value (kWh)", afa_kwh)
            # if afa_rate is not None:
            #     st.write("DEBUG: AFA Rate (RM/kWh)", afa_rate)
            # ...existing code...

            # --- Side-by-side Tariff Comparison Section ---
            st.markdown("---")
            st.subheader("Compare with Selected Tariff and Old Tariff")
            colA, colB = st.columns(2)

            # Ensure `selected_old_tariff` is defined before use
            from utils.old_cost_calculator import calculate_old_cost
            from old_rate import charging_rates

            
            # Display selected tariff breakdown
            with colA:
                st.markdown(f"#### {selected_user_type} > {selected_tariff_group} > {selected_tariff_type}")
                st.markdown("<b>Cost per kWh (Total Cost / Total kWh):</b> " + (
                    f"{(cost_breakdown.get('Total Cost',0)/cost_breakdown.get('Total kWh',1)) if cost_breakdown.get('Total kWh',0) else 'N/A'} RM/kWh"
                ), unsafe_allow_html=True)
                st.markdown(html_cost_table(cost_breakdown), unsafe_allow_html=True)

            # Move the old tariff selection and cost calculation into colB
            with colB:
                # Let user select old tariff
                old_tariff_names = list(charging_rates.keys())
                default_old_tariff = old_tariff_names[0]
                selected_old_tariff = st.selectbox(
                    "Select Old Tariff for Comparison",
                    old_tariff_names,
                    index=old_tariff_names.index(default_old_tariff),
                    key="old_tariff_selector"
                )

                # Calculate old cost
                old_cost_breakdown = calculate_old_cost(
                    selected_old_tariff,
                    total_kwh=total_kwh,
                    max_demand_kw=peak_demand if peak_demand is not None else 0,
                    peak_kwh=peak_kwh,
                    offpeak_kwh=offpeak_kwh
                )

                # Display old tariff breakdown
                with colB:
                    st.markdown(f"#### Old Tariff: {selected_old_tariff}")
                    if "error" in old_cost_breakdown:
                        st.error(old_cost_breakdown["error"])
                    else:
                        st.markdown(
                            f"<b>Cost per kWh (Total Cost / Total kWh):</b> "
                            f"{(old_cost_breakdown.get('Total Cost',0)/total_kwh) if total_kwh else 'N/A'} RM/kWh",
                            unsafe_allow_html=True
                        )
                        # Update the formatting function to ensure numbers are displayed with commas for thousands separators.
                        def fmt(val):
                            if val is None or val == "":
                                return ""
                            if isinstance(val, (int, float)):
                                return f"{val:,.0f}"  # Format numbers with commas for thousands separators.
                            return val

                        # Apply the updated formatting function to the Old Tariff breakdown table.
                        old_cost_breakdown_formatted = [
                            {"Description": k, "Value": fmt(v)}
                            for k, v in old_cost_breakdown.items() if k != "Tariff"
                        ]
                        st.write(pd.DataFrame(old_cost_breakdown_formatted))
            # --- Regression Formula Display (disabled, enable for future debugging) ---
            # st.subheader("Cost Calculation Formulae")
            # formulae = []
            # if "Peak kWh" in cost_breakdown:
            #     formulae.append(f"Peak Energy Cost = Peak kWh × Peak Rate = {cost_breakdown.get('Peak kWh', 0):,.2f} × {cost_breakdown.get('Peak Rate', '–')} = {cost_breakdown.get('Peak Energy Cost', 0):,.2f}")
            # if "Off-Peak kWh" in cost_breakdown:
            #     formulae.append(f"Off-Peak Energy Cost = Off-Peak kWh × Off-Peak Rate = {cost_breakdown.get('Off-Peak kWh', 0):,.2f} × {cost_breakdown.get('Off-Peak Rate', '–')} = {cost_breakdown.get('Off-Peak Energy Cost', 0):,.2f}")
            # afa_kwh = cost_breakdown.get('AFA kWh', cost_breakdown.get('Total kWh', 0))
            # afa_rate = cost_breakdown.get('AFA Rate', '–')
            # afa_cost = cost_breakdown.get('AFA Adjustment', 0)
            # if afa_kwh and afa_rate != '–':
            #     formulae.append(f"AFA Adjustment = AFA Value × AFA Rate = {afa_kwh:,.2f} × {afa_rate} = {afa_cost:,.2f}")
            # if "Max Demand (kW)" in cost_breakdown:
            #     formulae.append(f"Maximum Demand Cost = Max Demand × Capacity Rate = {cost_breakdown.get('Max Demand (kW)', 0):,.2f} × {cost_breakdown.get('Capacity Rate', '–')} = {cost_breakdown.get('Capacity Cost', 0):,.2f}")
            # if "Network Cost" in cost_breakdown:
            #     formulae.append(f"Network Cost = Max Demand × Network Rate = {cost_breakdown.get('Max Demand (kW)', 0):,.2f} × {cost_breakdown.get('Network Rate', '–')} = {cost_breakdown.get('Network Cost', 0):,.2f}")
            # if "Retail Cost" in cost_breakdown:
            #     formulae.append(f"Retail Cost = {cost_breakdown.get('Retail Cost', 0):,.2f}")
            # for f in formulae:
            #     st.markdown(f"- {f}")

            
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
