"""
Advanced Energy Analysis Module

This module provides comprehensive advanced energy analysis functionality
with RP4 tariff integration, including:
- Peak/Off-Peak Analysis with RP4 Logic
- Cost Analysis with Current RP4 Rates
- Advanced Peak Event Detection
- Load Duration Curve Analysis

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import RP4 and utility modules
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4
from utils.cost_calculator import calculate_cost


# Helper function to read different file formats
def read_uploaded_file(file):
    """Read uploaded file based on its extension"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLS, or XLSX files.")


def fmt(val):
    """
    Format numbers with specific rules:
    - Values below RM1: 4 decimal places (0.1111)
    - Values RM1 and above: comma-separated with 2 decimal places (RM1,000,000.00)
    """
    if val is None or val == "":
        return ""
    if isinstance(val, (int, float)):
        if val < 1:
            return f"{val:.4f}"
        return f"{val:,.2f}"
    return val


def show():
    """
    Main function to display the Advanced Energy Analysis interface.
    This function handles the entire Advanced Energy Analysis workflow.
    """
    st.title("Advanced Energy Analysis with RP4 Integration")
    st.markdown("""
    This advanced analysis uses the latest RP4 tariff structure with accurate peak/off-peak logic 
    and current MD rates for sophisticated energy management insights.
    """)
    
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xls", "xlsx"], key="advanced_file_uploader")
    
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            
            # Additional safety check for dataframe validity
            if df is None or df.empty:
                st.error("The uploaded file appears to be empty or invalid.")
                return
            
            if not hasattr(df, 'columns') or df.columns is None or len(df.columns) == 0:
                st.error("The uploaded file doesn't have valid column headers.")
                return
                
            st.success("File uploaded successfully!")
            
            # Column Selection and Holiday Configuration
            timestamp_col, power_col, holidays = _configure_data_inputs(df)
            
            # Simple validation - just check that columns exist
            if timestamp_col and power_col and timestamp_col in df.columns and power_col in df.columns:
                # Process data
                df = _process_dataframe(df, timestamp_col)
                
                # Final validation after processing
                if df.empty:
                    st.error("❌ No data remaining after processing. Please check your timestamp column format.")
                    return
                    
                if power_col not in df.columns:
                    st.error(f"❌ Power column '{power_col}' not found after data processing.")
                    return
                
                if not df.empty and power_col in df.columns:
                    # Tariff Selection
                    selected_tariff = _configure_tariff_selection()
                    
                    if selected_tariff:
                        # Execute all analysis sections
                        df_peak_analysis, overall_max_demand, interval_hours = _perform_peak_offpeak_analysis(df, power_col, holidays)
                        _perform_cost_analysis(df, selected_tariff, power_col, holidays)
                        _perform_peak_event_detection(df, power_col, selected_tariff, interval_hours)
                        _perform_load_duration_analysis(df, power_col, holidays)
                else:
                    st.warning("Please check your data. The selected power column may not exist after processing.")
            else:
                st.info("👆 Please select both timestamp and power columns to start the analysis.")
                    
        except TypeError as e:
            if "argument of type 'NoneType' is not iterable" in str(e):
                st.error("❌ Column selection error: Please ensure both timestamp and power columns are properly selected.")
                st.info("💡 This usually happens when your data has no numeric columns or invalid column names.")
            else:
                st.error(f"❌ Type error: {str(e)}")
        except KeyError as e:
            st.error(f"❌ Column not found: {str(e)}")
            st.info("💡 Please check that the selected columns exist in your data.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please ensure your Excel file has proper timestamp and power columns.")


def _configure_data_inputs(df):
    """Configure data inputs including column selection and holiday setup."""
    st.subheader("Data Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        timestamp_col = st.selectbox("Select timestamp column", df.columns, key="adv_timestamp_col")
        power_col = st.selectbox("Select power (kW) column", 
                               df.select_dtypes(include='number').columns, key="adv_power_col")
    
    with col2:
        holidays = _configure_holidays(df, timestamp_col)
    
    return timestamp_col, power_col, holidays


def _configure_holidays(df, timestamp_col):
    """Configure holiday selection for RP4 peak logic."""
    st.markdown("**Public Holidays (for RP4 peak logic)**")
    holidays = set()  # Default empty set
    
    # Simple holiday input - user can add holidays manually
    holiday_input = st.text_area(
        "Enter holidays (one per line, YYYY-MM-DD format):",
        placeholder="2024-01-01\n2024-02-10\n2024-04-10",
        key="adv_holidays_input"
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

    return holidays


def _process_dataframe(df, timestamp_col):
    """Process the dataframe with timestamp parsing, sorting validation, and indexing."""
    if not timestamp_col:
        return df
    
    try:
        df["Parsed Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=["Parsed Timestamp"])
        
        # Check if data is sorted
        is_sorted = df["Parsed Timestamp"].is_monotonic_increasing
        if not is_sorted:
            st.warning("⚠️ **Data not chronologically sorted** - Sorting timestamps for accurate interval detection")
            df = df.sort_values("Parsed Timestamp")
            st.success("✅ Data successfully sorted by timestamp")
        else:
            st.success("✅ Data is already chronologically sorted")
        
        # Set index
        df = df.set_index("Parsed Timestamp")
        
        # Validate for gaps and display data quality info
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
                interval_minutes = most_common_interval.total_seconds() / 60
                
                # Check for data gaps
                max_gap = time_diffs.max()
                expected_readings = (df.index.max() - df.index.min()) / most_common_interval + 1
                actual_readings = len(df)
                completeness = (actual_readings / expected_readings) * 100 if expected_readings > 0 else 100
                
                # Display data quality metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Period", f"{(df.index.max() - df.index.min()).days} days")
                with col2:
                    st.metric("Data Completeness", f"{completeness:.1f}%")
                with col3:
                    max_gap_hours = max_gap.total_seconds() / 3600
                    if max_gap_hours > interval_minutes / 60 * 2:  # Gap more than 2 intervals
                        st.metric("Max Gap", f"{max_gap_hours:.1f}h", delta="⚠️")
                    else:
                        st.metric("Max Gap", f"{max_gap_hours:.1f}h")
        
        return df
    except Exception as e:
        st.error(f"Error processing timestamp column: {e}")
        return pd.DataFrame()  # Return empty dataframe on error


def _configure_tariff_selection():
    """Configure RP4 tariff selection interface."""
    st.subheader("Tariff Configuration")
    tariff_data = get_tariff_data()
    
    # User Type Selection (Default: Business)
    user_types = list(tariff_data.keys())
    default_user_type = 'Business' if 'Business' in user_types else user_types[0]
    user_type_index = user_types.index(default_user_type)
    selected_user_type = st.selectbox("Select User Type", user_types, 
                                    index=user_type_index, key="adv_user_type")
    
    # Tariff Group Selection (Default: Non Domestic)
    tariff_groups = list(tariff_data[selected_user_type]["Tariff Groups"].keys())
    default_tariff_group = 'Non Domestic' if 'Non Domestic' in tariff_groups else tariff_groups[0]
    tariff_group_index = tariff_groups.index(default_tariff_group)
    selected_tariff_group = st.selectbox("Select Tariff Group", tariff_groups, 
                                       index=tariff_group_index, key="adv_tariff_group")
    
    # Specific Tariff Selection (Default: Medium Voltage TOU)
    tariffs = tariff_data[selected_user_type]["Tariff Groups"][selected_tariff_group]["Tariffs"]
    tariff_names = [t["Tariff"] for t in tariffs]
    default_tariff_name = 'Medium Voltage TOU' if 'Medium Voltage TOU' in tariff_names else tariff_names[0]
    tariff_name_index = tariff_names.index(default_tariff_name)
    selected_tariff_name = st.selectbox("Select Specific Tariff", tariff_names, 
                                      index=tariff_name_index, key="adv_specific_tariff")
    
    # Get the selected tariff object
    selected_tariff = next((t for t in tariffs if t["Tariff"] == selected_tariff_name), None)
    
    if selected_tariff:
        # Display tariff info
        st.info(f"**Selected:** {selected_user_type} > {selected_tariff_group} > {selected_tariff_name}")
    
    return selected_tariff


def _perform_peak_offpeak_analysis(df, power_col, holidays):
    """Perform Peak/Off-Peak Analysis with RP4 Logic."""
    st.subheader("1. Peak/Off-Peak Analysis (RP4 Logic)")
    
    # Display RP4 MD Peak Hours Information
    st.info("""
    **RP4 Maximum Demand (MD) Peak Hours:**
    - **Peak Period:** Monday to Friday, **2:00 PM to 10:00 PM** (14:00-22:00)
    - **Off-Peak Period:** All other times including weekends and public holidays
    - **MD Calculation:** Maximum demand recorded during peak periods only
    """)
    
    # === DATA INTERVAL DETECTION ===
    # Detect data interval from the entire dataset
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            # Get the most common time interval (mode)
            most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
            interval_hours = most_common_interval.total_seconds() / 3600
            interval_minutes = most_common_interval.total_seconds() / 60
            
            # Display interval detection results
            col1, col2, col3 = st.columns(3)
            with col1:
                if interval_minutes < 60:
                    st.metric("Detected Interval", f"{interval_minutes:.0f} minutes")
                else:
                    hours = interval_minutes / 60
                    st.metric("Detected Interval", f"{hours:.1f} hours")
            
            with col2:
                # Check for consistency
                unique_intervals = time_diffs.value_counts()
                consistency_percentage = (unique_intervals.iloc[0] / len(time_diffs)) * 100 if len(unique_intervals) > 0 else 100
                st.metric("Data Consistency", f"{consistency_percentage:.1f}%")
            
            with col3:
                # Handle last row: The last reading represents consumption for the detected interval
                total_intervals = len(df)  # Each reading represents one interval period
                st.metric("Total Intervals", f"{total_intervals:,}")
            
            # Show quality indicator
            if consistency_percentage >= 95:
                st.success(f"✅ **Excellent data quality**: {consistency_percentage:.1f}% consistent intervals")
            elif consistency_percentage >= 85:
                st.warning(f"⚠️ **Good data quality**: {consistency_percentage:.1f}% consistent intervals")
            else:
                st.error(f"❌ **Poor data quality**: Only {consistency_percentage:.1f}% consistent intervals")
                
            st.info(f"ℹ️ **Energy calculation**: Each power reading (kW) × {interval_hours:.3f} hours = Energy per interval (kWh)")
        else:
            # Fallback to 15 minutes if we can't determine interval
            interval_hours = 0.25
            st.warning("⚠️ Could not detect data interval. Using default 15-minute intervals.")
    else:
        interval_hours = 0.25
        st.warning("⚠️ Insufficient data for interval detection. Using default 15-minute intervals.")
    
    # Calculate peak/off-peak using RP4 logic
    is_peak_series = df.index.to_series().apply(lambda ts: is_peak_rp4(ts, holidays))
    df_peak_analysis = df[[power_col]].copy()
    df_peak_analysis['Is_Peak'] = is_peak_series
    
    # Calculate energy consumption using detected interval
    df_peak_analysis['Energy_kWh'] = df_peak_analysis[power_col] * interval_hours
    
    peak_kwh = df_peak_analysis[df_peak_analysis['Is_Peak']]['Energy_kWh'].sum()
    offpeak_kwh = df_peak_analysis[~df_peak_analysis['Is_Peak']]['Energy_kWh'].sum()
    total_kwh = peak_kwh + offpeak_kwh
    
    # Get peak demand (max during peak periods only)
    peak_demand_kw = df_peak_analysis[df_peak_analysis['Is_Peak']][power_col].max() if df_peak_analysis['Is_Peak'].any() else 0
    overall_max_demand = df_peak_analysis[power_col].max()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Energy", f"{fmt(peak_kwh)} kWh")
    col2.metric("Off-Peak Energy", f"{fmt(offpeak_kwh)} kWh") 
    col3.metric("Peak Demand (Peak Periods)", f"{fmt(peak_demand_kw)} kW")
    col4.metric("Overall Max Demand", f"{fmt(overall_max_demand)} kW")
    
    # Visualization
    period_df = pd.DataFrame({
        'Period': ['Peak', 'Off-Peak'],
        'Energy (kWh)': [peak_kwh, offpeak_kwh],
        'Percentage': [peak_kwh/total_kwh*100 if total_kwh > 0 else 0, 
                     offpeak_kwh/total_kwh*100 if total_kwh > 0 else 0]
    })
    
    fig_period = px.bar(period_df, x='Period', y='Energy (kWh)', 
                      text=[f"{row['Energy (kWh)']:,.0f} kWh ({row['Percentage']:.1f}%)" 
                            for _, row in period_df.iterrows()],
                      color='Period',
                      color_discrete_map={'Peak': 'orange', 'Off-Peak': 'blue'},
                      title='Energy Consumption by Period (RP4 Logic)')
    fig_period.update_traces(textposition='outside')
    st.plotly_chart(fig_period, use_container_width=True)
    
    return df_peak_analysis, overall_max_demand, interval_hours


def _perform_cost_analysis(df, selected_tariff, power_col, holidays):
    """Perform Cost Analysis with Current RP4 Rates."""
    st.subheader("2. Cost Analysis with Current RP4 Rates")
    
    # Display RP4 Rate Structure Information
    st.info("""
    **RP4 Rate Structure:**
    - **Peak Energy Rate:** Applied during 2:00 PM to 10:00 PM (Mon-Fri, excluding holidays)
    - **Off-Peak Energy Rate:** Applied during all other times
    - **Maximum Demand (MD):** Recorded during peak periods only (2:00 PM to 10:00 PM)
    """)
    
    # AFA Rate (use global setting from sidebar)
    global_afa_rate = st.session_state.get('global_afa_rate', 3.0) / 100
    global_afa_rate_cent = st.session_state.get('global_afa_rate', 3.0)
    st.info(f"Using AFA Rate: {global_afa_rate_cent:+.1f} cent/kWh (configured in sidebar)")
    
    # Calculate cost using the integrated cost calculator
    cost_breakdown = calculate_cost(df.reset_index(), selected_tariff, power_col, holidays, afa_rate=global_afa_rate)
    
    if "error" not in cost_breakdown:
        # Display cost breakdown
        col1, col2, col3 = st.columns(3)
        
        total_cost = cost_breakdown.get('Total Cost', 0)
        energy_cost = cost_breakdown.get('Peak Energy Cost', 0) + cost_breakdown.get('Off-Peak Energy Cost', 0) if 'Peak Energy Cost' in cost_breakdown else cost_breakdown.get('Energy Cost', 0)
        
        # Calculate demand cost - handle both RP4 style (Capacity + Network) and general tariff style (Demand Cost)
        demand_cost = 0
        demand_cost += cost_breakdown.get('Capacity Cost', 0)  # RP4 tariffs
        demand_cost += cost_breakdown.get('Network Cost', 0)   # RP4 tariffs
        demand_cost += cost_breakdown.get('Demand Cost', 0)   # General tariffs
        demand_cost += cost_breakdown.get('Maximum Demand Cost', 0)  # Alternative naming
        
        col1.metric("Total Cost", f"RM {fmt(total_cost)}")
        col2.metric("Energy Cost", f"RM {fmt(energy_cost)}")
        col3.metric("Demand Cost", f"RM {fmt(demand_cost)}")
        
        # Cost breakdown chart
        _display_cost_breakdown_chart(cost_breakdown)
    else:
        st.error(f"Cost calculation error: {cost_breakdown['error']}")


def _display_cost_breakdown_chart(cost_breakdown):
    """Display cost breakdown pie chart."""
    cost_categories = []
    cost_values = []
    
    if 'Peak Energy Cost' in cost_breakdown:
        cost_categories.extend(['Peak Energy', 'Off-Peak Energy'])
        cost_values.extend([cost_breakdown.get('Peak Energy Cost', 0), 
                          cost_breakdown.get('Off-Peak Energy Cost', 0)])
    else:
        cost_categories.append('Energy')
        cost_values.append(cost_breakdown.get('Energy Cost', 0))
    
    if cost_breakdown.get('Capacity Cost', 0) > 0:
        cost_categories.append('Capacity')
        cost_values.append(cost_breakdown.get('Capacity Cost', 0))
    
    if cost_breakdown.get('Network Cost', 0) > 0:
        cost_categories.append('Network')
        cost_values.append(cost_breakdown.get('Network Cost', 0))
    
    # Add general tariff demand costs
    if cost_breakdown.get('Demand Cost', 0) > 0:
        cost_categories.append('Demand')
        cost_values.append(cost_breakdown.get('Demand Cost', 0))
    
    if cost_breakdown.get('Maximum Demand Cost', 0) > 0:
        cost_categories.append('Maximum Demand')
        cost_values.append(cost_breakdown.get('Maximum Demand Cost', 0))
    
    if cost_breakdown.get('ICPT Cost', 0) != 0:
        cost_categories.append('ICPT')
        cost_values.append(cost_breakdown.get('ICPT Cost', 0))
    
    if cost_categories and cost_values:
        fig_cost = px.pie(values=cost_values, names=cost_categories, 
                        title='Cost Breakdown by Component')
        st.plotly_chart(fig_cost, use_container_width=True)


def _perform_peak_event_detection(df, power_col, selected_tariff, interval_hours):
    """Perform Advanced Peak Event Detection."""
    st.subheader("3. Advanced Peak Event Detection")
    
    st.info("""
    **Peak Event Detection for RP4 MD Management:**
    - **Focus Period:** 2:00 PM to 10:00 PM (Monday to Friday, excluding holidays)
    - **Objective:** Identify demand spikes during MD recording periods
    - **Impact:** Only peak period demand affects monthly MD charges
    """)
    
    overall_max_demand = df[power_col].max()
    
    # Target demand setting
    col1, col2 = st.columns(2)
    with col1:
        target_demand_percent = st.slider("Target Max Demand (% of current max)", 
                                        50, 100, 90, 1, key="adv_target_percent")
        target_demand = overall_max_demand * (target_demand_percent / 100)
        st.info(f"Target: {fmt(target_demand)} kW ({target_demand_percent}% of {fmt(overall_max_demand)} kW)")
    
    with col2:
        # Get MD rate from tariff (Capacity + Network rates)
        capacity_rate = selected_tariff.get('Rates', {}).get('Capacity Rate', 0)
        network_rate = selected_tariff.get('Rates', {}).get('Network Rate', 0)
        total_md_rate = capacity_rate + network_rate
        
        if total_md_rate > 0:
            st.metric("MD Rate (Capacity + Network)", f"RM {fmt(total_md_rate)}/kW")
            st.caption(f"Capacity: RM {fmt(capacity_rate)}/kW + Network: RM {fmt(network_rate)}/kW")
            potential_saving = (overall_max_demand - target_demand) * total_md_rate
            st.success(f"Potential Monthly Saving: RM {fmt(potential_saving)}")
        else:
            st.warning("No MD rate available for this tariff")
    
    # Detect and analyze peak events
    event_summaries = _detect_peak_events(df, power_col, target_demand, total_md_rate, interval_hours)
    
    if event_summaries:
        _display_peak_event_results(df, power_col, event_summaries, target_demand, total_md_rate, overall_max_demand, interval_hours)
    else:
        _display_no_peak_events(target_demand, overall_max_demand, df, power_col)


def _detect_peak_events(df, power_col, target_demand, total_md_rate, interval_hours):
    """Detect peak events above target demand."""
    df_events = df[[power_col]].copy()
    df_events['Above_Target'] = df_events[power_col] > target_demand
    df_events['Event_ID'] = (df_events['Above_Target'] != df_events['Above_Target'].shift()).cumsum()
    
    event_summaries = []
    for event_id, group in df_events.groupby('Event_ID'):
        if not group['Above_Target'].iloc[0]:
            continue
        
        start_time = group.index[0]
        end_time = group.index[-1]
        peak_load = group[power_col].max()
        excess = peak_load - target_demand
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Calculate energy to shave for entire event duration using global interval
        group_above = group[group[power_col] > target_demand]
        total_energy_to_shave = ((group_above[power_col] - target_demand) * interval_hours).sum()
        
        # Calculate energy to shave during MD peak period only (2 PM to 10 PM) - IMPROVED HIERARCHY
        # Filter for MD peak hours (14:00-22:00, weekdays only, excluding holidays)
        md_peak_mask = group_above.index.to_series().apply(
            lambda ts: (ts.weekday() < 5) and (14 <= ts.hour < 22)  # Holiday check would require holidays parameter
        )
        group_md_peak = group_above[md_peak_mask]
        md_peak_energy_to_shave = ((group_md_peak[power_col] - target_demand) * interval_hours).sum() if not group_md_peak.empty else 0
        
        # MD cost impact: excess MD during peak period × MD Rate (Capacity + Network)
        # Only calculate if event occurs during MD recording hours (2 PM-10 PM, weekdays)
        md_excess_during_peak = 0
        if not group_md_peak.empty:
            # Get the maximum excess during MD peak period for this event
            md_excess_during_peak = group_md_peak[power_col].max() - target_demand
        md_cost_impact = md_excess_during_peak * total_md_rate if md_excess_during_peak > 0 and total_md_rate > 0 else 0
        
        event_summaries.append({
            'Start Date': start_time.date(),
            'Start Time': start_time.strftime('%H:%M'),
            'End Date': end_time.date(),
            'End Time': end_time.strftime('%H:%M'),
            'Peak Load (kW)': peak_load,
            'Excess (kW)': excess,
            'Duration (min)': duration_minutes,
            'Energy to Shave (kWh)': total_energy_to_shave,
            'Energy to Shave (Peak Period Only)': md_peak_energy_to_shave,
            'MD Cost Impact (RM)': md_cost_impact
        })
    
    return event_summaries


def _filter_events_by_period(event_summaries, filter_type):
    """Filter events based on whether they occur during peak periods."""
    if filter_type == "All":
        return event_summaries
    
    filtered_events = []
    for event in event_summaries:
        start_date = event['Start Date']
        start_time_str = event['Start Time']
        
        # Parse the start time to check if it's in peak period
        start_hour = int(start_time_str.split(':')[0])
        start_weekday = start_date.weekday()  # 0=Monday, 6=Sunday
        
        # Check if event starts during RP4 MD peak hours (2 PM-10 PM, weekdays)
        is_peak_period_event = (start_weekday < 5) and (14 <= start_hour < 22)
        
        if filter_type == "Peak Period Only" and is_peak_period_event:
            filtered_events.append(event)
        elif filter_type == "Off-Peak Period Only" and not is_peak_period_event:
            filtered_events.append(event)
    
    return filtered_events


def _display_peak_event_results(df, power_col, event_summaries, target_demand, total_md_rate, overall_max_demand, interval_hours):
    """Display peak event detection results and analysis."""
    
    # Safety check for event_summaries
    if event_summaries is None:
        st.error("Error: Peak event data is not available.")
        return
    
    # Add filtering options
    st.markdown("#### Peak Event Filtering")
    event_filter = st.radio(
        "Select which events to display:",
        options=["All", "Peak Period Only", "Off-Peak Period Only"],
        index=0,
        horizontal=True,
        key="event_filter_radio",
        help="Filter events based on when they occur relative to RP4 MD peak hours (2 PM-10 PM, weekdays)"
    )
    
    # Filter events based on selection
    filtered_events = _filter_events_by_period(event_summaries, event_filter)
    
    if not filtered_events:
        st.warning(f"No events found for '{event_filter}' filter.")
        return
    
    st.markdown(f"**Showing {len(filtered_events)} of {len(event_summaries)} total events ({event_filter})**")
    
    # Display events table
    df_events_summary = pd.DataFrame(filtered_events)
    st.dataframe(df_events_summary.style.format({
        'Peak Load (kW)': lambda x: fmt(x),
        'Excess (kW)': lambda x: fmt(x),
        'Duration (min)': '{:.1f}',
        'Energy to Shave (kWh)': lambda x: fmt(x),
        'Energy to Shave (Peak Period Only)': lambda x: fmt(x),
        'MD Cost Impact (RM)': lambda x: f'RM {fmt(x)}' if x is not None else 'RM 0.0000'
    }), use_container_width=True)
    
    # Display explanation for the new columns
    st.info("""
    **Table Column Explanations:**
    - **Energy to Shave (kWh)**: Total energy above target for entire event duration
    - **Energy to Shave (Peak Period Only)**: Energy above target during MD recording hours only (2 PM-10 PM, weekdays)
    - **MD Cost Impact**: Excess MD during peak period × MD Rate (Capacity + Network) - only applies to events during 2 PM-10 PM weekdays
    """)
    
    # Visualization of events
    _display_peak_events_chart(df, power_col, event_summaries, target_demand)
    
    # Peak Event Summary & Analysis
    _display_peak_event_analysis(event_summaries, total_md_rate)
    
    # Threshold sensitivity analysis
    _display_threshold_analysis(df, power_col, overall_max_demand, total_md_rate, interval_hours)


def _display_peak_events_chart(df, power_col, event_summaries, target_demand):
    """Display peak events visualization chart with color-coded filled areas."""
    # Safety check for DataFrame and index
    if df is None or df.empty or not hasattr(df, 'index') or df.index is None:
        st.error("Unable to create chart: Invalid data structure.")
        return
    
    # Check if index is datetime
    if not hasattr(df.index, 'strftime'):
        st.error("Unable to create chart: Data index is not in datetime format.")
        return
    
    fig_events = go.Figure()
    
    # Add the main power consumption line
    fig_events.add_trace(go.Scatter(
        x=df.index, y=df[power_col],
        mode='lines', name='Power Consumption',
        line=dict(color='blue', width=2)
    ))
    
    # Track events by type for legend
    has_peak_period_events = False
    has_offpeak_period_events = False
    
    # Process events to create filled areas and event highlights
    for event in event_summaries:
        start_date = event['Start Date']
        end_date = event['End Date']
        start_time_str = event['Start Time']
        
        # Determine if this is a peak period event (2 PM-10 PM, weekdays)
        start_hour = int(start_time_str.split(':')[0])
        start_weekday = start_date.weekday()  # 0=Monday, 6=Sunday
        is_peak_period_event = (start_weekday < 5) and (14 <= start_hour < 22)
        
        # Choose colors based on period
        if is_peak_period_event:
            line_color = 'red'
            fill_color = 'rgba(255, 0, 0, 0.2)'  # Semi-transparent red
            event_type = 'Peak Period Event'
            has_peak_period_events = True
        else:
            line_color = 'green'
            fill_color = 'rgba(0, 128, 0, 0.2)'  # Semi-transparent green
            event_type = 'Off-Peak Period Event'
            has_offpeak_period_events = True
        
        # Create mask for event period (handle multi-day events)
        if start_date == end_date:
            # Single day event - add safety checks
            try:
                event_mask = (df.index.date == start_date) & \
                            (df.index.strftime('%H:%M') >= event['Start Time']) & \
                            (df.index.strftime('%H:%M') <= event['End Time'])
            except (AttributeError, TypeError) as e:
                st.warning(f"Error creating event mask for {start_date}: {e}")
                continue
        else:
            # Multi-day event - add safety checks
            try:
                event_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
                # For start day, filter by start time
                start_day_mask = (df.index.date == start_date) & (df.index.strftime('%H:%M') >= event['Start Time'])
                # For end day, filter by end time
                end_day_mask = (df.index.date == end_date) & (df.index.strftime('%H:%M') <= event['End Time'])
                # For middle days, include all times
                middle_days_mask = (df.index.date > start_date) & (df.index.date < end_date)
                
                event_mask = start_day_mask | end_day_mask | middle_days_mask
            except (AttributeError, TypeError) as e:
                st.warning(f"Error creating multi-day event mask for {start_date} to {end_date}: {e}")
                continue
        
        if event_mask.any():
            event_data = df[event_mask]
            event_label = f"{event_type} ({start_date})" if start_date == end_date else f"{event_type} ({start_date} to {end_date})"
            
            # Only show in legend for the first occurrence of each type
            show_in_legend = False
            if is_peak_period_event and not any('Peak Period Event' in trace.name for trace in fig_events.data):
                show_in_legend = True
            elif not is_peak_period_event and not any('Off-Peak Period Event' in trace.name for trace in fig_events.data):
                show_in_legend = True
            
            # Add filled area between power consumption and target line
            # Create arrays for the fill
            x_coords = list(event_data.index) + list(reversed(event_data.index))
            y_coords = list(event_data[power_col]) + [target_demand] * len(event_data)
            
            fig_events.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill='toself',
                fillcolor=fill_color,
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                mode='none',
                name=f'{event_type} - Excess Area' if show_in_legend else None,
                showlegend=show_in_legend,
                hoverinfo='skip'
            ))
            
            # Add event highlight line
            fig_events.add_trace(go.Scatter(
                x=event_data.index,
                y=event_data[power_col],
                mode='lines', 
                name=event_type if show_in_legend else event_label,
                line=dict(color=line_color, width=3),
                showlegend=False,  # Don't show line in legend since we have the fill
                hovertemplate=f'<b>{event_label}</b><br>' +
                             'Time: %{x}<br>' +
                             'Power: %{y:.2f} kW<br>' +
                             f'Excess: %{{y:.2f}} - {fmt(target_demand)} = %{{customdata:.2f}} kW<extra></extra>',
                customdata=[max(0, p - target_demand) for p in event_data[power_col]]
            ))
    
    # Add target line
    fig_events.add_hline(y=target_demand, line_dash="dot", line_color="orange", line_width=2,
                       annotation_text=f"Target: {fmt(target_demand)} kW")
    
    # Update title to reflect color coding and filled areas
    title_text = "Power Consumption with Shaded Excess Areas"
    subtitle_parts = []
    if has_peak_period_events:
        subtitle_parts.append("Red Shaded: Peak Period Excess (2PM-10PM, Weekdays)")
    if has_offpeak_period_events:
        subtitle_parts.append("Green Shaded: Off-Peak Period Excess")
    
    if subtitle_parts:
        title_text += "<br><sub>" + " | ".join(subtitle_parts) + "</sub>"
    
    fig_events.update_layout(
        title=title_text,
        xaxis_title="Time", 
        yaxis_title="Power (kW)", 
        height=500,
        hovermode='closest'
    )
    st.plotly_chart(fig_events, use_container_width=True)


def _display_peak_event_analysis(event_summaries, total_md_rate):
    """Display enhanced peak event summary and analysis."""
    st.markdown("---")
    st.markdown("#### Peak Event Summary & Threshold Analysis")
    
    # Safety check for event_summaries
    if event_summaries is None:
        st.error("Error: Event summaries data is not available.")
        return
    
    total_events = len(event_summaries)
    
    if total_events > 0:
        # Group events by day to get better statistics
        daily_events = {}
        daily_kwh_ranges = []
        daily_md_kwh_ranges = []
        
        for event in event_summaries:
            start_date = event['Start Date']  # Updated to use Start Date
            if start_date not in daily_events:
                daily_events[start_date] = []
            daily_events[start_date].append(event)
        
        # Calculate daily kWh ranges and total demand cost impact
        total_md_cost_monthly = 0
        max_md_excess_during_peak = 0
        
        for date, day_events in daily_events.items():
            daily_kwh_total = sum(e['Energy to Shave (kWh)'] for e in day_events)
            daily_md_kwh_total = sum(e['Energy to Shave (Peak Period Only)'] for e in day_events)
            daily_kwh_ranges.append(daily_kwh_total)
            daily_md_kwh_ranges.append(daily_md_kwh_total)
            
            # For MD cost calculation: find highest MD excess during peak periods
            for event in day_events:
                if event['Energy to Shave (Peak Period Only)'] > 0:  # Event has peak period component
                    # Calculate MD excess for this event during peak period
                    event_md_excess = event['MD Cost Impact (RM)'] / total_md_rate if total_md_rate > 0 else 0
                    max_md_excess_during_peak = max(max_md_excess_during_peak, event_md_excess)
        
        # Proper MD cost calculation: only the highest MD excess during peak periods
        total_md_cost_monthly = max_md_excess_during_peak * total_md_rate if total_md_rate > 0 else 0
        
        # Statistics for daily kWh ranges
        min_daily_kwh = min(daily_kwh_ranges) if daily_kwh_ranges else 0
        max_daily_kwh = max(daily_kwh_ranges) if daily_kwh_ranges else 0
        min_daily_md_kwh = min(daily_md_kwh_ranges) if daily_md_kwh_ranges else 0
        max_daily_md_kwh = max(daily_md_kwh_ranges) if daily_md_kwh_ranges else 0
        avg_events_per_day = total_events / len(daily_events) if daily_events else 0
        days_with_events = len(daily_events);
        
        # Display enhanced summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Days with Peak Events", f"{days_with_events}")
        col2.metric("Max MD Impact (Monthly)", f"RM {fmt(total_md_cost_monthly)}")
        col3.metric("Avg Events/Day", f"{avg_events_per_day:.1f}")
        col4.metric("Daily MD kWh Range", f"{fmt(min_daily_md_kwh)} - {fmt(max_daily_md_kwh)}")
        
        # Additional insights in expandable section
        _display_detailed_insights(total_events, days_with_events, max_md_excess_during_peak, 
                                 total_md_cost_monthly, max_daily_md_kwh)
    else:
        # No events detected
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Days with Peak Events", "0")
        col2.metric("MD Cost Impact", "RM 0.00")
        col3.metric("Target Status", "✅ Met")
        col4.metric("Optimization Needed", "None")


def _display_detailed_insights(total_events, days_with_events, max_md_excess_during_peak, 
                             total_md_cost_monthly, max_daily_kwh):
    """Display detailed MD management insights."""
    with st.expander("📊 Detailed MD Management Insights"):
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("**🎯 Peak Events Analysis:**")
            st.write(f"• Total events detected: {total_events}")
            st.write(f"• Events span across: {days_with_events} days")
            st.write(f"• Highest MD excess (peak periods): {fmt(max_md_excess_during_peak)} kW")
            st.write(f"• Peak intervals (30-min blocks): {total_events}")
        
        with insight_col2:
            st.markdown("**💰 MD Cost Strategy:**")
            st.write(f"• MD charges only highest demand")
            st.write(f"• Monthly impact: RM {fmt(total_md_cost_monthly)}")
            
            if days_with_events > 0:
                st.write(f"• Focus on worst day saves: RM {fmt(total_md_cost_monthly)}")
                st.write(f"• Multiple events/day = same MD cost")
            
            # Efficiency insight
            if max_daily_kwh > 0:
                efficiency_ratio = total_md_cost_monthly / max_daily_kwh if max_daily_kwh > 0 else 0
                st.write(f"• Cost per kWh shaved: RM {fmt(efficiency_ratio)}")


def _display_threshold_analysis(df, power_col, overall_max_demand, total_md_rate, interval_hours):
    """Display threshold sensitivity analysis."""
    st.markdown("#### Threshold Sensitivity Analysis")
    st.markdown("*How changing the target threshold affects the number of peak events and shaving requirements*")
    
    # Create analysis for different threshold percentages
    threshold_analysis = []
    test_percentages = [70, 75, 80, 85, 90, 95]
    
    for pct in test_percentages:
        test_target = overall_max_demand * (pct / 100)
        test_events = []
        
        # Recalculate events for this threshold
        df_test = df[[power_col]].copy()
        df_test['Above_Target'] = df_test[power_col] > test_target
        df_test['Event_ID'] = (df_test['Above_Target'] != df_test['Above_Target'].shift()).cumsum()
        
        for event_id, group in df_test.groupby('Event_ID'):
            if not group['Above_Target'].iloc[0]:
                continue
            
            peak_load = group[power_col].max()
            excess = peak_load - test_target
            
            # Calculate energy to shave for this threshold - total event using global interval
            group_above = group[group[power_col] > test_target]
            total_energy_to_shave = ((group_above[power_col] - test_target) * interval_hours).sum()
            
            # Calculate energy to shave during MD peak period only (2 PM to 10 PM) using global interval
            # IMPROVED HIERARCHY: MD peak period logic
            md_peak_mask = group_above.index.to_series().apply(
                lambda ts: (ts.weekday() < 5) and (14 <= ts.hour < 22)  # Holiday check would require holidays parameter
            )
            group_md_peak = group_above[md_peak_mask]
            md_peak_energy_to_shave = ((group_md_peak[power_col] - test_target) * interval_hours).sum() if not group_md_peak.empty else 0
            
            # Calculate MD excess during peak period for cost calculation
            md_excess_during_peak = 0
            if not group_md_peak.empty:
                md_excess_during_peak = group_md_peak[power_col].max() - test_target
            
            test_events.append({
                'excess': excess,
                'energy': total_energy_to_shave,
                'md_energy': md_peak_energy_to_shave,
                'md_excess': md_excess_during_peak
            })
        
        # Calculate totals for this threshold
        total_test_events = len(test_events)
        total_test_energy = sum(e['energy'] for e in test_events)
        total_md_energy = sum(e['md_energy'] for e in test_events)
        
        # CORRECTED MD cost calculation: only highest MD excess during peak periods matters
        max_md_excess_for_month = max(e['md_excess'] for e in test_events) if test_events else 0
        monthly_md_cost = max_md_excess_for_month * total_md_rate if total_md_rate > 0 else 0
        potential_monthly_saving = (overall_max_demand - test_target) * total_md_rate if total_md_rate > 0 else 0
        
        threshold_analysis.append({
            'Target (% of Max)': f"{pct}%",
            'Target (kW)': test_target,
            'Peak Events Count': total_test_events,
            'Total Energy to Shave (kWh)': total_test_energy,
            'MD Energy to Shave (kWh)': total_md_energy,
            'Monthly MD Cost (RM)': monthly_md_cost,
            'Monthly MD Saving (RM)': potential_monthly_saving,
            'Difficulty Level': 'Easy' if pct >= 90 else 'Medium' if pct >= 80 else 'Hard'
        })
    
    # Display threshold analysis results
    _display_threshold_results(threshold_analysis)


def _display_threshold_results(threshold_analysis):
    """Display threshold analysis results and insights."""
    df_threshold_analysis = pd.DataFrame(threshold_analysis)
    
    # Format and style the table
    styled_analysis = df_threshold_analysis.style.format({
        'Target (kW)': lambda x: fmt(x),
        'Total Energy to Shave (kWh)': lambda x: fmt(x),
        'MD Energy to Shave (kWh)': lambda x: fmt(x),
        'Monthly MD Cost (RM)': lambda x: f'RM {fmt(x)}',
        'Monthly MD Saving (RM)': lambda x: f'RM {fmt(x)}'
    }).apply(lambda x: ['background-color: rgba(40, 167, 69, 0.1)' if 'Easy' in str(v) 
                     else 'background-color: rgba(255, 193, 7, 0.1)' if 'Medium' in str(v)
                     else 'background-color: rgba(220, 53, 69, 0.1)' if 'Hard' in str(v)
                     else '' for v in x], axis=0)
    
    st.dataframe(styled_analysis, use_container_width=True)
    
    # Key insights
    _display_threshold_insights(df_threshold_analysis)
    
    # Visualization
    _display_threshold_chart(df_threshold_analysis)
    
    # MD calculation explanation
    _display_md_methodology_info()


def _display_threshold_insights(df_threshold_analysis):
    """Display key insights from threshold analysis."""
    st.markdown("#### Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Threshold-Event Relationship:**")
        lowest_threshold = df_threshold_analysis.iloc[-1]  # 70%
        highest_threshold = df_threshold_analysis.iloc[0]   # 95%
        
        event_increase = lowest_threshold['Peak Events Count'] - highest_threshold['Peak Events Count']
        energy_increase = lowest_threshold['Total Energy to Shave (kWh)'] - highest_threshold['Total Energy to Shave (kWh)']
        
        if event_increase > 0:
            st.success(f"✅ **Lower thresholds = More events**: Reducing from 95% to 70% increases events by {event_increase}")
        
        if energy_increase > 0:
            st.info(f"📈 **Energy impact**: {fmt(energy_increase)} kWh more energy needs shaving at lower thresholds")
    
    with col2:
        st.markdown("**💡 Recommended Strategy:**")
        
        # Find optimal balance (around 85-90%)
        optimal_row = df_threshold_analysis[df_threshold_analysis['Target (% of Max)'].isin(['85%', '90%'])]
        if not optimal_row.empty:
            best_option = optimal_row.iloc[0]
            st.success(f"🎯 **Recommended**: {best_option['Target (% of Max)']} target")
            st.write(f"• Events: {best_option['Peak Events Count']}")
            st.write(f"• Monthly saving: RM {fmt(best_option['Monthly MD Saving (RM)'])}")
            st.write(f"• Difficulty: {best_option['Difficulty Level']}")


def _display_threshold_chart(df_threshold_analysis):
    """Display threshold analysis visualization chart."""
    fig_threshold = go.Figure()
    
    # Add events count line
    fig_threshold.add_trace(go.Scatter(
        x=[int(x.rstrip('%')) for x in df_threshold_analysis['Target (% of Max)']],
        y=df_threshold_analysis['Peak Events Count'],
        mode='lines+markers',
        name='Peak Events Count',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    # Add MD cost on secondary y-axis
    fig_threshold.add_trace(go.Scatter(
        x=[int(x.rstrip('%')) for x in df_threshold_analysis['Target (% of Max)']],
        y=df_threshold_analysis['Monthly MD Cost (RM)'],
        mode='lines+markers',
        name='Monthly MD Cost (RM)',
        line=dict(color='#FFA500', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig_threshold.update_layout(
        title='Peak Events vs Target Threshold Analysis',
        xaxis_title='Target Threshold (% of Max Demand)',
        yaxis=dict(title='Number of Peak Events', side='left'),
        yaxis2=dict(title='Monthly MD Cost (RM)', side='right', overlaying='y'),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_threshold, use_container_width=True)


def _display_md_methodology_info():
    """Display MD calculation methodology information."""
    st.info("""
    **💡 MD Cost Calculation Methodology:**
    
    The Monthly MD Cost shown represents the **correct** way MD charges are calculated:
    • TNB records maximum demand every **30 minutes** during peak periods
    • Only the **single highest** 30-minute reading of the month is charged
    • Multiple peak events on the same day = same MD cost (if highest event is the same)
    • Focus should be on reducing the **worst single event** rather than total event count
    """)


def _display_no_peak_events(target_demand, overall_max_demand, df, power_col):
    """Display interface when no peak events are detected."""
    st.info(f"🎉 No peak events detected above target demand of {fmt(target_demand)} kW")
    st.success("✅ Current demand profile is already within target limits!")
    
    # Still show what would happen with lower thresholds
    st.markdown("#### What if we set a lower target?")
    lower_target = overall_max_demand * 0.8  # 80% threshold
    
    df_test = df[[power_col]].copy()
    df_test['Above_Target'] = df_test[power_col] > lower_target
    test_events_count = len(df_test[df_test['Above_Target'] == True].groupby((df_test['Above_Target'] != df_test['Above_Target'].shift()).cumsum()).filter(lambda x: x['Above_Target'].iloc[0]))
    
    st.info(f"At 80% threshold ({fmt(lower_target)} kW): {test_events_count} events would be detected")


def _perform_load_duration_analysis(df, power_col, holidays):
    """Perform Load Duration Curve Analysis."""
    st.subheader("4. Load Duration Curve Analysis")
    
    # Get RP4 peak analysis for filtering
    is_peak_series = df.index.to_series().apply(lambda ts: is_peak_rp4(ts, holidays))
    df_peak_analysis = df[[power_col]].copy()
    df_peak_analysis['Is_Peak'] = is_peak_series
    
    col1, col2 = st.columns(2)
    with col1:
        ldc_period = st.radio("Time Range for LDC", 
                            ["All", "Peak Only", "Off-Peak Only"], 
                            horizontal=True, key="adv_ldc_period")
    with col2:
        top_percent = st.number_input("Show Top % of Readings", 
                                    0.1, 100.0, 100.0, 0.1, 
                                    key="adv_ldc_percent")
    
    # Prepare LDC data
    if ldc_period == "Peak Only":
        ldc_df = df[df_peak_analysis['Is_Peak']][[power_col]].copy()
    elif ldc_period == "Off-Peak Only":
        ldc_df = df[~df_peak_analysis['Is_Peak']][[power_col]].copy()
    else:
        ldc_df = df[[power_col]].copy()
    
    if not ldc_df.empty:
        _display_load_duration_curve(ldc_df, power_col, ldc_period, top_percent)
    else:
        st.warning(f"No data available for {ldc_period} period")


def _display_load_duration_curve(ldc_df, power_col, ldc_period, top_percent):
    """Display Load Duration Curve and analysis."""
    # Sort for LDC
    ldc_sorted = ldc_df.sort_values(by=power_col, ascending=False).reset_index(drop=True)
    
    # Apply percentage filter
    if top_percent < 100.0:
        num_points = max(1, int(np.ceil(len(ldc_sorted) * (top_percent / 100.0))))
        ldc_plot = ldc_sorted.head(num_points).copy()
    else:
        ldc_plot = ldc_sorted.copy()
    
    ldc_plot["Percentage Time"] = (ldc_plot.index + 1) / len(ldc_plot) * 100
    
    # Create LDC plot
    fig_ldc = px.line(ldc_plot, x="Percentage Time", y=power_col,
                    title=f"Load Duration Curve - {ldc_period} (Top {top_percent}%)",
                    labels={"Percentage Time": "% of Time", power_col: "Power (kW)"})
    fig_ldc.update_traces(mode="lines+markers")
    fig_ldc.update_layout(height=500)
    st.plotly_chart(fig_ldc, use_container_width=True)
    
    # LDC Analysis
    if len(ldc_plot) >= 5:
        _analyze_ldc_for_demand_shaving(ldc_plot, power_col)
    else:
        st.warning("Insufficient data points for LDC analysis")


def _analyze_ldc_for_demand_shaving(ldc_plot, power_col):
    """Analyze LDC for demand shaving potential."""
    p_peak = ldc_plot[power_col].iloc[0]
    p_shoulder_idx = min(max(1, int(0.05 * len(ldc_plot))), len(ldc_plot) - 1)
    p_shoulder = ldc_plot[power_col].iloc[p_shoulder_idx]
    
    shave_potential = p_peak - p_shoulder
    relative_potential = shave_potential / p_peak if p_peak > 0 else 0
    
    if shave_potential > 10 and relative_potential > 0.15:
        st.success(f"""
        **LDC Analysis: EXCELLENT for Demand Shaving**
        - Peak: {fmt(p_peak)} kW
        - Shoulder (5%): {fmt(p_shoulder)} kW  
        - Shaving Potential: {fmt(shave_potential)} kW ({relative_potential:.1%})
        """)
    else:
        st.warning("LDC Analysis: Limited demand shaving potential")
