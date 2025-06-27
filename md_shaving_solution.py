"""
MD Shaving Solution Module

This module provides MD (Maximum Demand) shaving analysis functionality
reusing components from Advanced Energy Analysis with additional features:
- File upload using existing Advanced Energy Analysis logic
- Peak event filtering functionality  
- Right sidebar selectors
- RP4 tariff integration for MD cost calculations

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import RP4 and utility modules
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4
from utils.cost_calculator import calculate_cost


def fmt(val):
    """Format values for display with proper decimal places."""
    if val is None or val == "":
        return ""
    if isinstance(val, (int, float)):
        if val < 1:
            return f"{val:,.4f}"
        return f"{val:,.2f}"
    return val


def show():
    """
    Main function to display the MD Shaving Solution interface.
    This function handles the entire MD Shaving Solution workflow.
    """
    st.title("🔋 MD Shaving Solution")
    st.markdown("""
    **Advanced Maximum Demand (MD) shaving analysis** using RP4 tariff structure for accurate cost savings calculation.
    Upload your load profile to identify peak events and optimize MD cost reduction strategies.
    
    💡 **Tip:** Use the sidebar configuration to set your preferred default values for shaving percentages!
    """)
    
    # Quick info box about configurable defaults
    with st.expander("ℹ️ How to Use Configurable Defaults"):
        st.markdown("""
        **Step 1:** Open the "⚙️ Configure Default Values" section in the sidebar
        
        **Step 2:** Set your preferred default values:
        - **Default Shave %**: Your preferred percentage to reduce from peak (e.g., 15%)
        - **Default Target %**: Your preferred target as percentage of current max (e.g., 85%)
        - **Default Manual kW**: Your preferred manual target value
        
        **Step 3:** Use Quick Presets for common scenarios:
        - **Conservative**: 5% shaving (95% target)
        - **Moderate**: 10% shaving (90% target)  
        - **Aggressive**: 20% shaving (80% target)
        
        **Step 4:** Your configured values will be used as defaults for all new analyses!
        
        **Example:** If you set "Default Shave %" to 15%, then when you select "Percentage to Shave", 
        the slider will default to 15% instead of the factory default of 10%.
        """)
    
    
    # Sidebar configuration for MD Shaving Solution
    with st.sidebar:
        st.markdown("---")
        st.markdown("### MD Shaving Configuration")
        
        # Configuration section for default values
        with st.expander("⚙️ Configure Default Values", expanded=False):
            st.markdown("**Customize your default shaving parameters:**")
            
            col1, col2 = st.columns(2)
            with col1:
                default_shave_percent = st.number_input(
                    "Default Shave %",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key="config_default_shave",
                    help="Default percentage to shave from peak"
                )
                
                default_target_percent = st.number_input(
                    "Default Target %",
                    min_value=50,
                    max_value=100,
                    value=90,
                    step=1,
                    key="config_default_target",
                    help="Default target as % of current max"
                )
            
            with col2:
                default_manual_kw = st.number_input(
                    "Default Manual kW",
                    min_value=10.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0,
                    key="config_default_manual",
                    help="Default manual target in kW"
                )
                
                # Quick preset buttons
                st.markdown("**Quick Presets:**")
                if st.button("Conservative (5% shave)", key="preset_conservative"):
                    st.session_state.config_default_shave = 5
                    st.session_state.config_default_target = 95
                    st.rerun()
                    
                if st.button("Moderate (10% shave)", key="preset_moderate"):
                    st.session_state.config_default_shave = 10
                    st.session_state.config_default_target = 90
                    st.rerun()
                    
                if st.button("Aggressive (20% shave)", key="preset_aggressive"):
                    st.session_state.config_default_shave = 20
                    st.session_state.config_default_target = 80
                    st.rerun()
            
            # Display current configuration
            st.markdown("---")
            st.markdown("**Current Config:**")
            st.caption(f"• Shave: {default_shave_percent}% | Target: {default_target_percent}% | Manual: {default_manual_kw:.0f}kW")
            
            # Reset to factory defaults
            if st.button("🔄 Reset to Factory Defaults", key="reset_config"):
                st.session_state.config_default_shave = 10
                st.session_state.config_default_target = 90
                st.session_state.config_default_manual = 100.0
                st.success("✅ Reset to factory defaults!")
                st.rerun()
        
        st.markdown("### MD Shaving Controls")
        
        # Target demand setting options
        target_method = st.radio(
            "Target Setting Method:",
            options=["Percentage to Shave", "Percentage of Current Max", "Manual Target (kW)"],
            index=0,
            key="md_target_method",
            help="Choose how to set your target maximum demand"
        )
        
        if target_method == "Percentage to Shave":
            shave_percent = st.slider(
                "Percentage to Shave (%)", 
                min_value=1, 
                max_value=50, 
                value=default_shave_percent, 
                step=1,
                key="md_shave_percent",
                help="Percentage to reduce from current peak (e.g., 10% shaving reduces 200kW peak to 180kW)"
            )
            target_percent = None
            target_manual_kw = None
        elif target_method == "Percentage of Current Max":
            target_percent = st.slider(
                "Target MD (% of current max)", 
                min_value=50, 
                max_value=100, 
                value=default_target_percent, 
                step=1,
                key="md_target_percent",
                help="Set the target maximum demand as percentage of current peak"
            )
            shave_percent = None
            target_manual_kw = None
        else:
            target_manual_kw = st.number_input(
                "Target MD (kW)",
                min_value=0.0,
                max_value=10000.0,
                value=default_manual_kw,
                step=1.0,
                key="md_target_manual",
                help="Enter your desired target maximum demand in kW"
            )
            target_percent = None
            shave_percent = None
        
        # Peak event filter
        event_filter = st.radio(
            "Event Filter:",
            options=["All Events", "Peak Period Only", "Off-Peak Period Only"],
            index=0,
            key="md_event_filter",
            help="Filter events based on RP4 MD peak hours (2 PM-10 PM, weekdays)"
        )
        
        # Analysis options
        st.markdown("### Analysis Options")
        show_detailed_analysis = st.checkbox(
            "Show Detailed Analysis", 
            value=True,
            key="md_detailed_analysis"
        )
        
        show_threshold_sensitivity = st.checkbox(
            "Show Threshold Sensitivity", 
            value=True,
            key="md_threshold_sensitivity"
        )

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"], key="md_shaving_file_uploader")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
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
            
            # Only proceed if both columns are detected and valid
            if (timestamp_col and power_col and 
                hasattr(df, 'columns') and df.columns is not None and
                timestamp_col in df.columns and power_col in df.columns):
                
                # Process data
                df = _process_dataframe(df, timestamp_col)
                
                if not df.empty and power_col in df.columns:
                    # Tariff Selection
                    selected_tariff = _configure_tariff_selection()
                    
                    if selected_tariff:
                        # Calculate target demand based on selected method
                        overall_max_demand = df[power_col].max()
                        
                        if target_method == "Percentage to Shave":
                            target_demand = overall_max_demand * (1 - shave_percent / 100)
                            target_description = f"{shave_percent}% shaving ({fmt(target_demand)} kW, {100-shave_percent}% of current max)"
                        elif target_method == "Percentage of Current Max":
                            target_demand = overall_max_demand * (target_percent / 100)
                            target_description = f"{target_percent}% of current max ({fmt(target_demand)} kW)"
                        else:
                            target_demand = target_manual_kw
                            target_percent_actual = (target_demand / overall_max_demand * 100) if overall_max_demand > 0 else 0
                            target_description = f"{fmt(target_demand)} kW ({target_percent_actual:.1f}% of current max)"
                        
                        # Validate target demand
                        if target_demand <= 0:
                            st.error("❌ Target demand must be greater than 0 kW")
                            return
                        elif target_demand >= overall_max_demand:
                            st.warning(f"⚠️ Target demand ({fmt(target_demand)} kW) is equal to or higher than current max ({fmt(overall_max_demand)} kW). No peak shaving needed.")
                            st.info("💡 Consider setting a lower target to identify shaving opportunities.")
                            return
                        
                        # Display target information
                        st.info(f"🎯 **Target:** {target_description}")
                        
                        # Execute MD shaving analysis
                        interval_hours = _perform_md_shaving_analysis(
                            df, power_col, selected_tariff, holidays, 
                            target_demand, overall_max_demand, event_filter,
                            show_detailed_analysis, show_threshold_sensitivity
                        )
                else:
                    st.warning("Please check your data. The selected power column may not exist after processing.")
            else:
                if not timestamp_col:
                    st.error("❌ Could not auto-detect timestamp column. Please ensure your file has a date/time column.")
                if not power_col:
                    st.error("❌ Could not auto-detect power column. Please ensure your file has a numeric power/demand column.")
                if timestamp_col and power_col:
                    st.info("✅ Columns detected. Processing will begin automatically.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure your Excel file has proper timestamp and power columns.")


def _auto_detect_columns(df):
    """
    Auto-detect timestamp and power columns based on common patterns.
    Returns tuple of (timestamp_col, power_col)
    """
    timestamp_col = None
    power_col = None
    
    # Auto-detect timestamp column
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for common timestamp column names
        timestamp_keywords = ['date', 'time', 'timestamp', 'datetime', 'dt', 'period']
        if any(keyword in col_lower for keyword in timestamp_keywords):
            # Verify it can be parsed as datetime
            try:
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Try to parse a sample
                    pd.to_datetime(sample_values.iloc[0], errors='raise')
                    timestamp_col = col
                    break  # Use first valid datetime column found
            except:
                continue
        
        # If no keyword match, check if column contains datetime-like values
        if timestamp_col is None:
            try:
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Try to parse multiple samples to be sure
                    parsed_count = 0
                    for val in sample_values:
                        try:
                            pd.to_datetime(val, errors='raise')
                            parsed_count += 1
                        except:
                            break
                    
                    # If most samples parse successfully, it's likely a timestamp column
                    if parsed_count >= len(sample_values) * 0.8:  # 80% success rate
                        timestamp_col = col
                        break
            except:
                continue
    
    # Auto-detect power column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        col_lower = col.lower()
        
        # Check for common power/demand column names
        power_keywords = ['power', 'kw', 'kilowatt', 'demand', 'load', 'consumption', 'kwh']
        if any(keyword in col_lower for keyword in power_keywords):
            # Prefer columns with 'kw' or 'power' over 'kwh'
            if 'kwh' in col_lower:
                # Store as backup but keep looking for better match
                if power_col is None:
                    power_col = col
            else:
                power_col = col
                break  # Found a good match, use it
    
    # If no keyword match, use first numeric column as fallback
    if power_col is None and numeric_cols:
        power_col = numeric_cols[0]
    
    return timestamp_col, power_col


def _configure_data_inputs(df):
    """Configure data inputs including column selection and holiday setup."""
    st.subheader("Data Configuration")
    
    # Auto-detect columns
    auto_timestamp_col, auto_power_col = _auto_detect_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Column Selection**")
        
        # Auto-selected timestamp column with option to override
        timestamp_options = list(df.columns)
        
        if auto_timestamp_col:
            try:
                timestamp_index = timestamp_options.index(auto_timestamp_col)
                st.success(f"✅ Auto-detected timestamp column: **{auto_timestamp_col}**")
            except ValueError:
                timestamp_index = 0
                st.warning("⚠️ Could not auto-detect timestamp column")
        else:
            timestamp_index = 0
            st.warning("⚠️ Could not auto-detect timestamp column")
        
        timestamp_col = st.selectbox(
            "Timestamp column (auto-detected):", 
            timestamp_options, 
            index=timestamp_index,
            key="md_timestamp_col",
            help="Auto-detected based on datetime patterns. Change if incorrect."
        )
        
        # Auto-selected power column with option to override
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if auto_power_col and auto_power_col in numeric_cols:
            try:
                power_index = numeric_cols.index(auto_power_col)
                st.success(f"✅ Auto-detected power column: **{auto_power_col}**")
            except ValueError:
                power_index = 0
                st.warning("⚠️ Could not auto-detect power column")
        else:
            power_index = 0
            st.warning("⚠️ Could not auto-detect power column")
        
        power_col = st.selectbox(
            "Power (kW) column (auto-detected):", 
            numeric_cols, 
            index=power_index,
            key="md_power_col",
            help="Auto-detected based on column names containing 'power', 'kw', 'demand', etc."
        )
    
    with col2:
        st.markdown("**Holiday Configuration**")
        holidays = _configure_holidays(df, timestamp_col)
    
    return timestamp_col, power_col, holidays


def _configure_holidays(df, timestamp_col):
    """Configure holiday selection for RP4 peak logic."""
    if timestamp_col:
        try:
            # Parse timestamps to get date range
            df_temp = df.copy()
            df_temp["Parsed Timestamp"] = pd.to_datetime(df_temp[timestamp_col], errors="coerce")
            df_temp = df_temp.dropna(subset=["Parsed Timestamp"])
            
            if not df_temp.empty:
                min_date = df_temp["Parsed Timestamp"].min().date()
                max_date = df_temp["Parsed Timestamp"].max().date()
                unique_dates = pd.date_range(min_date, max_date).date
                
                holiday_options = [d.strftime('%A, %d %B %Y') for d in unique_dates]
                selected_labels = st.multiselect(
                    "Select public holidays:",
                    options=holiday_options,
                    default=[],
                    help="Pick all public holidays in the data period",
                    key="md_holidays"
                )
                
                # Map back to date objects
                label_to_date = {d.strftime('%A, %d %B %Y'): d for d in unique_dates}
                selected_holidays = [label_to_date[label] for label in selected_labels]
                holidays = set(selected_holidays)
                
                st.info(f"Selected {len(holidays)} holidays")
                return holidays
        except Exception as e:
            st.warning(f"Error processing dates: {e}")
    
    return set()


def _process_dataframe(df, timestamp_col):
    """Process the dataframe with timestamp parsing, sorting validation, and indexing."""
    df_processed = df.copy()
    
    # Parse timestamp column
    df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
    
    # Remove rows with invalid timestamps
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=[timestamp_col])
    final_rows = len(df_processed)
    
    if final_rows < initial_rows:
        st.warning(f"Removed {initial_rows - final_rows} rows with invalid timestamps")
    
    # Sort by timestamp
    df_processed = df_processed.sort_values(timestamp_col)
    
    # Set timestamp as index
    df_processed.set_index(timestamp_col, inplace=True)
    
    return df_processed


def _configure_tariff_selection():
    """Configure RP4 tariff selection interface."""
    st.subheader("RP4 Tariff Configuration")
    tariff_data = get_tariff_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # User Type Selection (Default: Business)
        user_types = list(tariff_data.keys())
        default_user_type = 'Business' if 'Business' in user_types else user_types[0]
        user_type_index = user_types.index(default_user_type)
        selected_user_type = st.selectbox("User Type", user_types, 
                                        index=user_type_index, key="md_user_type")
    
    with col2:
        # Tariff Group Selection (Default: Non Domestic)
        tariff_groups = list(tariff_data[selected_user_type]["Tariff Groups"].keys())
        default_tariff_group = 'Non Domestic' if 'Non Domestic' in tariff_groups else tariff_groups[0]
        tariff_group_index = tariff_groups.index(default_tariff_group)
        selected_tariff_group = st.selectbox("Tariff Group", tariff_groups, 
                                           index=tariff_group_index, key="md_tariff_group")
    
    with col3:
        # Specific Tariff Selection (Default: Medium Voltage TOU)
        tariffs = tariff_data[selected_user_type]["Tariff Groups"][selected_tariff_group]["Tariffs"]
        tariff_names = [t["Tariff"] for t in tariffs]
        default_tariff_name = 'Medium Voltage TOU' if 'Medium Voltage TOU' in tariff_names else tariff_names[0]
        tariff_name_index = tariff_names.index(default_tariff_name)
        selected_tariff_name = st.selectbox("Specific Tariff", tariff_names, 
                                          index=tariff_name_index, key="md_specific_tariff")
    
    # Get the selected tariff object
    selected_tariff = next((t for t in tariffs if t["Tariff"] == selected_tariff_name), None)
    
    if selected_tariff:
        # Display tariff info
        st.info(f"**Selected:** {selected_user_type} > {selected_tariff_group} > {selected_tariff_name}")
        
        # Show MD rates
        capacity_rate = selected_tariff.get('Rates', {}).get('Capacity Rate', 0)
        network_rate = selected_tariff.get('Rates', {}).get('Network Rate', 0)
        total_md_rate = capacity_rate + network_rate
        
        # Debug information - show exact tariff being used
        with st.expander("🔍 Debug: Tariff Rate Details", expanded=False):
            st.write("**Selected Tariff Object:**")
            st.json(selected_tariff)
            st.write(f"**Extracted Rates:**")
            st.write(f"- Capacity Rate: {capacity_rate}")
            st.write(f"- Network Rate: {network_rate}")
            st.write(f"- Total MD Rate: {total_md_rate}")
        
        if total_md_rate > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Capacity Rate", f"RM {fmt(capacity_rate)}/kW")
            col2.metric("Network Rate", f"RM {fmt(network_rate)}/kW")
            col3.metric("Total MD Rate", f"RM {fmt(total_md_rate)}/kW")
            
            # Expected vs Actual verification for Medium Voltage TOU
            if selected_tariff_name == "Medium Voltage TOU":
                expected_capacity = 30.19
                expected_network = 66.87
                expected_total = 97.06
                
                if abs(capacity_rate - expected_capacity) > 0.01 or abs(network_rate - expected_network) > 0.01:
                    st.error(f"⚠️ **Rate Mismatch Detected!**")
                    st.error(f"Expected: Capacity={expected_capacity}, Network={expected_network}, Total={expected_total}")
                    st.error(f"Actual: Capacity={capacity_rate}, Network={network_rate}, Total={total_md_rate}")
                else:
                    st.success(f"✅ **Rates Verified**: Medium Voltage TOU rates match expected values")
        else:
            st.warning("⚠️ This tariff has no MD charges - MD shaving will not provide savings")
    
    return selected_tariff


def _perform_md_shaving_analysis(df, power_col, selected_tariff, holidays, target_demand, 
                                overall_max_demand, event_filter, show_detailed_analysis, 
                                show_threshold_sensitivity):
    """Perform comprehensive MD shaving analysis."""
    
    # Detect data interval
    interval_hours = _detect_data_interval(df)
    
    # Display MD peak hours information
    st.subheader("🎯 MD Shaving Analysis")
    st.info("""
    **RP4 Maximum Demand (MD) Peak Hours:**
    - **Peak Period:** Monday to Friday, **2:00 PM to 10:00 PM** (14:00-22:00)
    - **Off-Peak Period:** All other times including weekends and public holidays
    - **MD Calculation:** Maximum demand recorded during peak periods only
    """)
    
    # Get MD rate from tariff
    capacity_rate = selected_tariff.get('Rates', {}).get('Capacity Rate', 0)
    network_rate = selected_tariff.get('Rates', {}).get('Network Rate', 0)
    total_md_rate = capacity_rate + network_rate
    
    # Display target and potential savings
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Max Demand", f"{fmt(overall_max_demand)} kW")
    col2.metric("Target Max Demand", f"{fmt(target_demand)} kW")
    if total_md_rate > 0:
        potential_saving = (overall_max_demand - target_demand) * total_md_rate
        col3.metric("Potential Monthly Saving", f"RM {fmt(potential_saving)}")
    else:
        col3.metric("MD Rate", "RM 0.00/kW")
        st.warning("No MD savings possible with this tariff")
        return interval_hours
    
    # Detect peak events
    event_summaries = _detect_peak_events(df, power_col, target_demand, total_md_rate, interval_hours)
    
    if event_summaries:
        # Display peak event results
        _display_peak_event_results(df, power_col, event_summaries, target_demand, 
                                   total_md_rate, overall_max_demand, interval_hours, 
                                   event_filter, show_detailed_analysis)
        
        if show_threshold_sensitivity:
            # Display threshold sensitivity analysis
            _display_threshold_analysis(df, power_col, overall_max_demand, total_md_rate, interval_hours)
    else:
        st.success("🎉 No peak events detected above target demand!")
        st.info(f"Current demand profile is already within target limit of {fmt(target_demand)} kW")
    
    return interval_hours


def _detect_data_interval(df):
    """Detect data interval from the dataframe."""
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
            interval_hours = most_common_interval.total_seconds() / 3600
            interval_minutes = most_common_interval.total_seconds() / 60
            
            st.info(f"📊 **Data interval detected:** {interval_minutes:.0f} minutes")
            return interval_hours
    
    # Fallback
    st.warning("⚠️ Could not detect data interval, assuming 15 minutes")
    return 0.25


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
        
        # Calculate energy to shave for entire event duration
        group_above = group[group[power_col] > target_demand]
        total_energy_to_shave = ((group_above[power_col] - target_demand) * interval_hours).sum()
        
        # Calculate energy to shave during MD peak period only (2 PM to 10 PM)
        md_peak_mask = group_above.index.to_series().apply(
            lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22
        )
        group_md_peak = group_above[md_peak_mask]
        md_peak_energy_to_shave = ((group_md_peak[power_col] - target_demand) * interval_hours).sum() if not group_md_peak.empty else 0
        
        # MD cost impact: excess MD during peak period × MD Rate
        md_excess_during_peak = 0
        md_peak_load_during_event = 0
        md_peak_time = None
        
        if not group_md_peak.empty:
            # Find the exact peak load and time during MD recording hours
            md_peak_load_during_event = group_md_peak[power_col].max()
            md_peak_time = group_md_peak[group_md_peak[power_col] == md_peak_load_during_event].index[0]
            md_excess_during_peak = md_peak_load_during_event - target_demand
        
        md_cost_impact = md_excess_during_peak * total_md_rate if md_excess_during_peak > 0 and total_md_rate > 0 else 0
        
        event_summaries.append({
            'Start Date': start_time.date(),
            'Start Time': start_time.strftime('%H:%M'),
            'End Date': end_time.date(),
            'End Time': end_time.strftime('%H:%M'),
            'Peak Load (kW)': peak_load,
            'Excess (kW)': excess,
            'MD Peak Load (kW)': md_peak_load_during_event,
            'MD Excess (kW)': md_excess_during_peak,
            'MD Peak Time': md_peak_time.strftime('%H:%M') if md_peak_time else 'N/A',
            'Duration (min)': duration_minutes,
            'Energy to Shave (kWh)': total_energy_to_shave,
            'Energy to Shave (Peak Period Only)': md_peak_energy_to_shave,
            'MD Cost Impact (RM)': md_cost_impact
        })
    
    return event_summaries


def _filter_events_by_period(event_summaries, filter_type):
    """Filter events based on whether they occur during peak periods."""
    if filter_type == "All Events":
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


def _display_peak_event_results(df, power_col, event_summaries, target_demand, total_md_rate, 
                               overall_max_demand, interval_hours, event_filter, show_detailed_analysis):
    """Display peak event detection results and analysis."""
    
    st.subheader("⚡ Peak Event Detection Results")
    
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
        'MD Peak Load (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
        'MD Excess (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
        'Duration (min)': '{:.1f}',
        'Energy to Shave (kWh)': lambda x: fmt(x),
        'Energy to Shave (Peak Period Only)': lambda x: fmt(x),
        'MD Cost Impact (RM)': lambda x: f'RM {fmt(x)}' if x is not None else 'RM 0.0000'
    }), use_container_width=True)
    
    # Display explanation
    st.info("""
    **Column Explanations:**
    - **Peak Load (kW)**: Highest demand during entire event period (may include off-peak hours)
    - **Excess (kW)**: Overall event peak minus target (for reference only)
    - **MD Peak Load (kW)**: Highest demand during MD recording hours only (2 PM-10 PM, weekdays)
    - **MD Excess (kW)**: MD peak load minus target - this determines MD cost impact
    - **MD Peak Time**: Exact time when MD peak occurred (for MD cost calculation)
    - **Energy to Shave (kWh)**: Total energy above target for entire event duration
    - **Energy to Shave (Peak Period Only)**: Energy above target during MD recording hours only
    - **MD Cost Impact**: MD Excess (kW) × MD Rate - based on MD peak, not overall event peak
    """)
    
    # Visualization of events
    _display_peak_events_chart(df, power_col, filtered_events, target_demand)
    
    if show_detailed_analysis:
        # Peak Event Summary & Analysis
        _display_peak_event_analysis(filtered_events, total_md_rate)


def _display_peak_events_chart(df, power_col, event_summaries, target_demand):
    """Display peak events visualization chart."""
    st.subheader("📈 Peak Events Timeline")
    
    # Create the main power consumption chart
    fig_events = go.Figure()
    
    # Add main power consumption line
    fig_events.add_trace(go.Scatter(
        x=df.index,
        y=df[power_col],
        mode='lines',
        name='Power Consumption',
        line=dict(color='blue', width=1),
        hovertemplate='%{x}<br>Power: %{y:.2f} kW<extra></extra>'
    ))
    
    # Add target demand line
    fig_events.add_hline(
        y=target_demand,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: {fmt(target_demand)} kW"
    )
    
    # Highlight peak events
    has_peak_period_events = False
    has_offpeak_period_events = False
    
    for event in event_summaries:
        start_date = event['Start Date']
        start_time_str = event['Start Time']
        end_date = event['End Date']
        
        # Determine if this is a peak period event
        start_hour = int(start_time_str.split(':')[0])
        start_weekday = start_date.weekday()
        is_peak_period_event = (start_weekday < 5) and (14 <= start_hour < 22)
        
        # Choose colors based on period
        if is_peak_period_event:
            fill_color = 'rgba(255, 0, 0, 0.2)'  # Semi-transparent red
            event_type = 'Peak Period Event'
            has_peak_period_events = True
        else:
            fill_color = 'rgba(0, 128, 0, 0.2)'  # Semi-transparent green
            event_type = 'Off-Peak Period Event'
            has_offpeak_period_events = True
        
        # Create mask for event period
        if start_date == end_date:
            event_mask = (df.index.date == start_date) & \
                        (df.index.strftime('%H:%M') >= event['Start Time']) & \
                        (df.index.strftime('%H:%M') <= event['End Time'])
        else:
            event_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        
        if event_mask.any():
            event_data = df[event_mask]
            event_label = f"{event_type} ({start_date})" if start_date == end_date else f"{event_type} ({start_date} to {end_date})"
            
            # Add filled area between power consumption and target line
            x_coords = list(event_data.index) + list(reversed(event_data.index))
            y_coords = list(event_data[power_col]) + [target_demand] * len(event_data)
            
            fig_events.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill='toself',
                fillcolor=fill_color,
                line=dict(color='rgba(0,0,0,0)'),
                name=event_label,
                hoverinfo='skip',
                showlegend=True
            ))
    
    fig_events.update_layout(
        title='Power Consumption with Peak Events Highlighted',
        xaxis_title='Time',
        yaxis_title='Power (kW)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_events, use_container_width=True)


def _display_peak_event_analysis(event_summaries, total_md_rate):
    """Display enhanced peak event summary and analysis."""
    st.subheader("📊 Peak Event Analysis Summary")
    
    total_events = len(event_summaries)
    
    if total_events > 0:
        # Group events by day to get better statistics
        daily_events = {}
        daily_kwh_ranges = []
        daily_md_kwh_ranges = []
        
        for event in event_summaries:
            start_date = event['Start Date']
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
                if event['Energy to Shave (Peak Period Only)'] > 0:
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
        days_with_events = len(daily_events)
        
        # Display enhanced summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Days with Peak Events", f"{days_with_events}")
        col2.metric("Max MD Impact (Monthly)", f"RM {fmt(total_md_cost_monthly)}")
        col3.metric("Avg Events/Day", f"{avg_events_per_day:.1f}")
        col4.metric("Daily MD kWh Range", f"{fmt(min_daily_md_kwh)} - {fmt(max_daily_md_kwh)}")
        
        # Additional insights in expandable section
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
                if max_daily_md_kwh > 0:
                    efficiency_ratio = total_md_cost_monthly / max_daily_md_kwh if max_daily_md_kwh > 0 else 0
                    st.write(f"• Cost per kWh shaved: RM {fmt(efficiency_ratio)}")


def _display_threshold_analysis(df, power_col, overall_max_demand, total_md_rate, interval_hours):
    """Display threshold sensitivity analysis."""
    st.subheader("📈 Threshold Sensitivity Analysis")
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
            
            # Calculate energy to shave for this threshold
            group_above = group[group[power_col] > test_target]
            total_energy_to_shave = ((group_above[power_col] - test_target) * interval_hours).sum()
            
            # Calculate energy to shave during MD peak period only
            md_peak_mask = group_above.index.to_series().apply(
                lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22
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
        
        # Potential monthly saving if target is achieved
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
    df_threshold_analysis = pd.DataFrame(threshold_analysis)
    
    st.markdown("#### Threshold Analysis Results")
    st.dataframe(df_threshold_analysis.style.format({
        'Target (kW)': lambda x: fmt(x),
        'Total Energy to Shave (kWh)': lambda x: fmt(x),
        'MD Energy to Shave (kWh)': lambda x: fmt(x),
        'Monthly MD Cost (RM)': lambda x: f'RM {fmt(x)}',
        'Monthly MD Saving (RM)': lambda x: f'RM {fmt(x)}'
    }), use_container_width=True)
    
    # Display threshold analysis chart
    fig_threshold = go.Figure()
    
    # Add bar chart for number of events
    fig_threshold.add_trace(go.Bar(
        x=df_threshold_analysis['Target (% of Max)'],
        y=df_threshold_analysis['Peak Events Count'],
        name='Peak Events Count',
        yaxis='y',
        marker_color='lightblue'
    ))
    
    # Add line chart for MD cost
    fig_threshold.add_trace(go.Scatter(
        x=df_threshold_analysis['Target (% of Max)'],
        y=df_threshold_analysis['Monthly MD Cost (RM)'],
        mode='lines+markers',
        name='Monthly MD Cost (RM)',
        yaxis='y2',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Update layout for dual y-axes
    fig_threshold.update_layout(
        title='Threshold Sensitivity Analysis: Events vs MD Cost',
        xaxis_title='Target Threshold (% of Max Demand)',
        yaxis=dict(
            title='Number of Peak Events',
            side='left'
        ),
        yaxis2=dict(
            title='Monthly MD Cost (RM)',
            side='right',
            overlaying='y'
        ),
        height=500
    )
    
    st.plotly_chart(fig_threshold, use_container_width=True)
    
    # Display insights
    st.markdown("#### Key Insights")
    
    if len(df_threshold_analysis) > 0:
        # Find the sweet spot (balance between savings and difficulty)
        best_row = df_threshold_analysis[df_threshold_analysis['Difficulty Level'] == 'Easy']
        if best_row.empty:
            best_row = df_threshold_analysis[df_threshold_analysis['Difficulty Level'] == 'Medium']
        if not best_row.empty:
            best_row = best_row.iloc[-1]  # Get the most aggressive target within the easy/medium range
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Recommended Target:** {best_row['Target (% of Max)']} ({fmt(best_row['Target (kW)'])} kW)")
                st.info(f"• {best_row['Peak Events Count']} events to manage")
                st.info(f"• {fmt(best_row['MD Energy to Shave (kWh)'])} kWh to shave (MD periods)")
            
            with col2:
                st.success(f"**Potential Savings:** RM {fmt(best_row['Monthly MD Saving (RM)'])}/month")
                st.info(f"• Difficulty level: {best_row['Difficulty Level']}")
                st.info(f"• Annual savings: RM {fmt(best_row['Monthly MD Saving (RM)'] * 12)}")
