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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import RP4 and utility modules
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4, get_period_classification
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

# Import battery algorithms
from battery_algorithms import (
    get_battery_parameters_ui, 
    perform_comprehensive_battery_analysis,
    create_battery_algorithms
)


def create_conditional_demand_line_with_peak_logic(fig, df, power_col, target_demand, selected_tariff=None, holidays=None, trace_name="Original Demand"):
    """
    Enhanced conditional coloring logic for Original Demand line with dynamic RP4 peak period logic.
    Creates continuous line segments with different colors based on conditions.
    
    Color Logic:
    - Red: Above target during Peak Periods (based on selected tariff) - Direct MD cost impact
    - Green: Above target during Off-Peak Periods - No MD cost impact  
    - Blue: Below target (any time) - Within acceptable limits
    """
    # Convert index to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df.index)
    else:
        df_copy = df
    
    # Create a series with color classifications
    df_copy = df_copy.copy()
    df_copy['color_class'] = ''
    
    for i in range(len(df_copy)):
        timestamp = df_copy.index[i]
        demand_value = df_copy.iloc[i][power_col]
        
        # Get peak period classification based on selected tariff
        if selected_tariff:
            period_type = get_tariff_period_classification(timestamp, selected_tariff, holidays)
        else:
            # Fallback to default RP4 logic
            period_type = get_period_classification(timestamp, holidays)
        
        if demand_value > target_demand:
            if period_type == 'Peak':
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'red'
            else:
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'green'
        else:
            df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'blue'
    
    # Create a single continuous line with color-coded segments
    # First, add all data as individual colored traces for proper line continuity
    x_data = df_copy.index
    y_data = df_copy[power_col]
    colors = df_copy['color_class']
    
    # Track legend status
    legend_added = {'red': False, 'green': False, 'blue': False}
    
    # Create continuous line segments by color groups with bridge points
    i = 0
    while i < len(df_copy):
        current_color = colors.iloc[i]
        
        # Find the end of current color segment
        j = i
        while j < len(colors) and colors.iloc[j] == current_color:
            j += 1
        
        # Extract segment data
        segment_x = list(x_data[i:j])
        segment_y = list(y_data[i:j])
        
        # Add bridge points for better continuity (connect to adjacent segments)
        if i > 0:  # Add connection point from previous segment
            segment_x.insert(0, x_data[i-1])
            segment_y.insert(0, y_data[i-1])
        
        if j < len(colors):  # Add connection point to next segment
            segment_x.append(x_data[j])
            segment_y.append(y_data[j])
        
        # Determine trace name based on color and tariff type
        tariff_description = _get_tariff_description(selected_tariff) if selected_tariff else "RP4 Peak Period"
        
        # Check if it's a TOU tariff for enhanced hover info
        is_tou = False
        if selected_tariff:
            tariff_type = selected_tariff.get('Type', '').lower()
            tariff_name = selected_tariff.get('Tariff', '').lower()
            is_tou = tariff_type == 'tou' or 'tou' in tariff_name
        
        if current_color == 'red':
            segment_name = f'{trace_name} (Above Target - {tariff_description})'
            if is_tou:
                hover_info = f'<b>Above Target - TOU Peak Rate Period</b><br><i>High Energy Cost + MD Cost Impact</i>'
            else:
                hover_info = f'<b>Above Target - General Tariff</b><br><i>MD Cost Impact Only (Flat Energy Rate)</i>'
        elif current_color == 'green':
            segment_name = f'{trace_name} (Above Target - Off-Peak)'
            if is_tou:
                hover_info = '<b>Above Target - TOU Off-Peak</b><br><i>Low Energy Cost, No MD Impact</i>'
            else:
                hover_info = '<b>Above Target - General Tariff</b><br><i>This should not appear for General tariffs</i>'
        else:  # blue
            segment_name = f'{trace_name} (Below Target)'
            hover_info = '<b>Below Target</b><br><i>Within Acceptable Limits</i>'
        
        # Only show legend for the first occurrence of each color
        show_legend = not legend_added[current_color]
        legend_added[current_color] = True
        
        # Add line segment
        fig.add_trace(go.Scatter(
            x=segment_x,
            y=segment_y,
            mode='lines',
            line=dict(color=current_color, width=2),
            name=segment_name,
            hovertemplate=f'{trace_name}: %{{y:.2f}} kW<br>%{{x}}<br>{hover_info}<extra></extra>',
            showlegend=show_legend,
            legendgroup=current_color,
            connectgaps=True  # Connect gaps within segments
        ))
        
        i = j
    
    return fig


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
        
        # Peak event filter with tariff-specific options
        event_filter_options = ["All Events", "Peak Period Only", "Off-Peak Period Only"]
        
        # Get current tariff selection to determine filter options
        tariff_selection = st.session_state.get('selected_tariff', {})
        tariff_type = tariff_selection.get('Type', '').lower()
        tariff_name = tariff_selection.get('Tariff', '').lower()
        is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
        
        # Add tariff-specific filter options
        if is_tou_tariff:
            event_filter_options.extend(["MD Cost Impact Events", "No MD Cost Impact Events"])
            help_text = "Filter events based on RP4 MD peak hours (2PM-10PM weekdays) and TOU tariff MD cost impact"
        else:
            event_filter_options.extend(["MD Cost Impact Events"])
            help_text = "Filter events based on RP4 MD peak hours (2PM-10PM weekdays) - General tariffs have MD cost impact 24/7"
        
        event_filter = st.radio(
            "Event Filter:",
            options=event_filter_options,
            index=0,
            key="md_event_filter",
            help=help_text
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

    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xls", "xlsx"], key="md_shaving_file_uploader")
    
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
            
            # Only proceed if both columns are detected and valid
            if (timestamp_col and power_col and 
                hasattr(df, 'columns') and df.columns is not None and
                timestamp_col in df.columns and power_col in df.columns):
                
                # Process data
                df = _process_dataframe(df, timestamp_col)
                
                if not df.empty and power_col in df.columns:
                    # Tariff Selection
                    selected_tariff = _configure_tariff_selection()
                    
                    # Store tariff selection in session state for sidebar access
                    st.session_state['selected_tariff'] = selected_tariff
                    
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
                
                # Add TNB holidays option
                holiday_mode = st.radio(
                    "Holiday Configuration:",
                    ["Auto: TNB Official Holidays", "Manual: Select Dates"],
                    index=0,
                    help="Choose TNB's 15 official holidays or manually select",
                    key="md_holiday_mode"
                )
                
                if holiday_mode == "Auto: TNB Official Holidays":
                    tnb_holidays = get_tnb_holidays_2024_2025()
                    # Filter to data range
                    tnb_holidays = {h for h in tnb_holidays if min_date <= h <= max_date}
                    st.success(f"✅ **TNB holidays**: {len(tnb_holidays)} official holidays applied")
                    
                    # Show the holidays in an expander
                    with st.expander("📅 View TNB Official Holidays", expanded=False):
                        if tnb_holidays:
                            holiday_list = sorted(list(tnb_holidays))
                            for holiday in holiday_list:
                                st.write(f"• {holiday.strftime('%A, %d %B %Y')}")
                        else:
                            st.write("No TNB holidays found in your data period")
                    
                    return tnb_holidays
                else:
                    # Original manual selection code
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
                    st.info(f"Selected {len(holidays)} holidays manually")
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
        tariff_type = selected_tariff.get('Type', '').lower()
        is_tou = tariff_type == 'tou' or 'tou' in selected_tariff.get('Tariff', '').lower()
        
        st.info(f"**Selected:** {selected_user_type} > {selected_tariff_group} > {selected_tariff_name}")
        
        # Show tariff type and peak period logic
        if is_tou:
            st.success("🎯 **TOU Tariff Detected**: Chart colors will reflect TOU peak periods (2PM-10PM weekdays)")
        else:
            st.info("📊 **General Tariff Detected**: Chart colors will reflect MD recording periods (2PM-10PM weekdays)")
        
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
    
    # Display MD peak hours information with tariff-specific color logic
    st.subheader("🎯 MD Shaving Analysis")
    
    # Get tariff type for color explanation
    tariff_type = selected_tariff.get('Type', '').lower()
    tariff_name = selected_tariff.get('Tariff', '')
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name.lower()
    
    if is_tou_tariff:
        color_explanation = """
    **📈 Chart Color Legend (TOU Tariff):**
    - 🔴 **Red**: Above target during TOU peak rate hours (2PM-10PM weekdays) → **High energy cost + MD charges**
    - 🟢 **Green**: Above target during TOU off-peak hours → **Low energy cost, no MD charges**  
    - 🔵 **Blue**: Below target (any time) → **Within acceptable limits**
    
    **TOU Impact:** Red periods have DOUBLE financial impact (energy + MD)
        """
    else:
        color_explanation = """
    **📈 Chart Color Legend (General Tariff):**
    - 🔴 **Red**: Above target (any time) → **MD charges apply (flat energy rate 24/7)**
    - 🟢 **Green**: N/A (no off-peak concept in general tariffs)
    - 🔵 **Blue**: Below target (any time) → **Within acceptable limits**
    
    **General Tariff:** Red periods affect MD charges only (flat energy rate applies 24/7)
        """
    
    st.info(f"""
    **RP4 Maximum Demand (MD) Peak Hours & Chart Color Logic:**
    - **Peak Period:** Monday to Friday, **2:00 PM to 10:00 PM** (14:00-22:00)
    - **Off-Peak Period:** All other times including weekends and public holidays
    - **MD Calculation:** Maximum demand recorded during peak periods only

{color_explanation}
    """)
    
    # Get MD rate from tariff
    capacity_rate = selected_tariff.get('Rates', {}).get('Capacity Rate', 0)
    network_rate = selected_tariff.get('Rates', {}).get('Network Rate', 0)
    total_md_rate = capacity_rate + network_rate
    
    # Detect peak events first to get actual MD cost impact
    event_summaries = _detect_peak_events_tou_aware(df, power_col, target_demand, total_md_rate, interval_hours, selected_tariff, holidays)
    
    # Calculate actual potential saving from maximum MD cost impact
    max_md_cost_impact = 0
    max_energy_to_shave_peak_only = 0
    max_md_excess = 0
    
    if event_summaries:
        max_md_cost_impact = max(event['MD Cost Impact (RM)'] for event in event_summaries)
        max_energy_to_shave_peak_only = max(event['TOU Required Energy (kWh)'] for event in event_summaries)
        max_md_excess = max(event['TOU Excess (kW)'] for event in event_summaries if event['TOU Excess (kW)'] > 0)
    
    # Display target and potential savings
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Max Demand", f"{fmt(overall_max_demand)} kW")
    col2.metric("Target Max Demand", f"{fmt(target_demand)} kW")
    if total_md_rate > 0:
        col3.metric("Potential Monthly Saving", f"RM {fmt(max_md_cost_impact)}")
    else:
        col3.metric("MD Rate", "RM 0.00/kW")
        st.warning("No MD savings possible with this tariff")
        return interval_hours
    
    # Display battery sizing recommendations
    if event_summaries:
        st.markdown("### 🔋 Recommended Battery Sizing")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Battery Capacity (Peak Period)", 
                f"{fmt(max_energy_to_shave_peak_only)} kWh",
                help="Maximum energy to shave during peak period only (2PM-10PM weekdays)"
            )
        with col2:
            st.metric(
                "Battery Power Rating", 
                f"{fmt(max_md_excess)} kW",
                help="Maximum MD excess power that needs to be shaved"
            )
        
        st.info(f"""
        💡 **Battery Sizing Logic:**
        - **Capacity**: Based on worst single event energy during MD peak hours ({fmt(max_energy_to_shave_peak_only)} kWh)
        - **Power Rating**: Based on maximum MD excess demand ({fmt(max_md_excess)} kW)
        - These values represent the minimum battery specifications needed for effective MD shaving
        """)
    else:
        st.info("No peak events detected - no battery sizing recommendations available")
    
    # Use the already detected peak events for analysis
    if event_summaries:
        # Display peak event results
        _display_peak_event_results(df, power_col, event_summaries, target_demand, 
                                   total_md_rate, overall_max_demand, interval_hours, 
                                   event_filter, show_detailed_analysis, selected_tariff, holidays)
        
        if show_threshold_sensitivity:
            # Display threshold sensitivity analysis
            _display_threshold_analysis(df, power_col, overall_max_demand, total_md_rate, interval_hours)
        
        # Battery Analysis Section
        st.markdown("---")
        st.markdown("## 🔋 Battery Energy Storage System (BESS) Analysis")
        
        # Get battery analysis parameters from sidebar
        battery_params = get_battery_parameters_ui(event_summaries)
        
        # Perform battery sizing and analysis
        battery_analysis = perform_comprehensive_battery_analysis(
            df, power_col, event_summaries, target_demand, 
            interval_hours, battery_params, total_md_rate, selected_tariff, holidays
        )
        
        # Display battery results
        _display_battery_analysis(battery_analysis, battery_params, target_demand, max_md_cost_impact, selected_tariff, holidays)
        
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


def _detect_peak_events(df, power_col, target_demand, total_md_rate, interval_hours, selected_tariff=None):
    """Detect peak events above target demand with tariff-specific MD cost impact calculation."""
    df_events = df[[power_col]].copy()
    df_events['Above_Target'] = df_events[power_col] > target_demand
    df_events['Event_ID'] = (df_events['Above_Target'] != df_events['Above_Target'].shift()).cumsum()
    
    # Determine tariff type for MD cost impact logic
    tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
    tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
    
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
        
        # Calculate energy to shave during MD peak period only (2 PM to 10 PM) - IMPROVED HIERARCHY
        md_peak_mask = group_above.index.to_series().apply(
            lambda ts: (ts.weekday() < 5) and (14 <= ts.hour < 22)  # Holiday check would require holidays parameter
        )
        group_md_peak = group_above[md_peak_mask]
        md_peak_energy_to_shave = ((group_md_peak[power_col] - target_demand) * interval_hours).sum() if not group_md_peak.empty else 0
        
        # Tariff-specific MD cost impact calculation
        md_excess_during_peak = 0
        md_peak_load_during_event = 0
        md_peak_time = None
        has_md_cost_impact = False
        
        if is_tou_tariff:
            # TOU tariffs: Only events during 2PM-10PM weekdays have MD cost impact
            if not group_md_peak.empty:
                md_peak_load_during_event = group_md_peak[power_col].max()
                md_peak_time = group_md_peak[group_md_peak[power_col] == md_peak_load_during_event].index[0]
                md_excess_during_peak = md_peak_load_during_event - target_demand
                has_md_cost_impact = True
        else:
            # General tariffs: ALL events above target have MD cost impact 24/7
            md_peak_load_during_event = peak_load
            md_peak_time = group[group[power_col] == peak_load].index[0]
            md_excess_during_peak = excess
            has_md_cost_impact = True
        
        md_cost_impact = md_excess_during_peak * total_md_rate if md_excess_during_peak > 0 and total_md_rate > 0 else 0
        
        event_summaries.append({
            'Start Date': start_time.date(),
            'Start Time': start_time.strftime('%H:%M'),
            'End Date': end_time.date(),
            'End Time': end_time.strftime('%H:%M'),
            'General Peak Load (kW)': peak_load,
            'General Excess (kW)': excess,
            'TOU Peak Load (kW)': md_peak_load_during_event,
            'TOU Excess (kW)': md_excess_during_peak,
            'TOU Peak Time': md_peak_time.strftime('%H:%M') if md_peak_time else 'N/A',
            'Duration (min)': duration_minutes,
            'General Required Energy (kWh)': total_energy_to_shave,
            'TOU Required Energy (kWh)': md_peak_energy_to_shave,
            'MD Cost Impact (RM)': md_cost_impact,
            'Has MD Cost Impact': has_md_cost_impact,
            'Tariff Type': 'TOU' if is_tou_tariff else 'General'
        })
    
    return event_summaries


def _detect_peak_events_tou_aware(df, power_col, target_demand, total_md_rate, interval_hours, selected_tariff=None, holidays=None):
    """
    Enhanced peak event detection with TOU-aware MD recording window logic.
    
    TOU Tariffs: Peak events only start during MD recording periods (2PM-10PM weekdays, excluding holidays)
    General Tariffs: Peak events can start anytime (24/7 MD recording)
    
    Args:
        df: DataFrame with power data
        power_col: Column name for power values
        target_demand: Target demand threshold
        total_md_rate: Total MD rate for cost calculation
        interval_hours: Data sampling interval in hours
        selected_tariff: Tariff configuration dict
        holidays: Set of holiday dates
    
    Returns:
        List of peak event summaries with enhanced TOU-aware logic
    """
    from tariffs.peak_logic import is_peak_rp4, get_malaysia_holidays
    
    # Import detection function for MD window validation
    def is_md_window(timestamp, holidays_set):
        """Check if timestamp falls within MD recording window based on tariff type"""
        return is_peak_rp4(timestamp, holidays_set if holidays_set else set())
    
    # Determine tariff type for event start logic
    tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
    tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
    
    # Use provided holidays or auto-detect from data years
    if holidays is None:
        years = df.index.year.unique()
        holidays = set()
        for year in years:
            holidays.update(get_malaysia_holidays(year))
    
    # Enhanced peak event detection with TOU-aware logic
    event_summaries = []
    in_event = False
    event_start = None
    event_data = []
    
    for timestamp, row in df.iterrows():
        power_value = row[power_col]
        
        # Determine if this timestamp can be a peak event start based on tariff type
        if is_tou_tariff:
            # TOU Tariff: Only during MD recording periods (2PM-10PM weekdays, excluding holidays)
            can_be_peak_event = is_md_window(timestamp, holidays)
        else:
            # General Tariff: Anytime above target (24/7 MD recording)
            can_be_peak_event = True
        
        # Peak event logic with TOU awareness
        if not in_event and power_value > target_demand and can_be_peak_event:
            # Start new peak event (only if within valid window for tariff type)
            in_event = True
            event_start = timestamp
            event_data = [(timestamp, power_value)]
            
        elif in_event and power_value > target_demand and (not is_tou_tariff or can_be_peak_event):
            # Continue peak event only if:
            # - For General tariffs: always continue while above target
            # - For TOU tariffs: continue only while still in MD window (14:00-22:00)
            event_data.append((timestamp, power_value))
            
        elif in_event and (power_value <= target_demand or 
                          (is_tou_tariff and not can_be_peak_event)):
            # End peak event when:
            # 1. Power drops below target, OR
            # 2. For TOU tariffs: leaving MD recording window (at 22:00)
            in_event = False
            
            # Process completed event
            if event_data:
                event_df = pd.DataFrame(event_data, columns=['timestamp', 'power'])
                event_df.set_index('timestamp', inplace=True)
                
                start_time = event_df.index[0]
                end_time = event_df.index[-1]
                peak_load = event_df['power'].max()
                excess = peak_load - target_demand
                duration_minutes = (end_time - start_time).total_seconds() / 60
                
                # Calculate energy to shave for entire event duration
                total_energy_to_shave = ((event_df['power'] - target_demand) * interval_hours).sum()
                
                # Calculate energy to shave during MD peak period only (2PM-10PM weekdays)
                md_peak_mask = event_df.index.to_series().apply(
                    lambda ts: is_md_window(ts, holidays)
                )
                event_md_peak = event_df[md_peak_mask]
                md_peak_energy_to_shave = ((event_md_peak['power'] - target_demand) * interval_hours).sum() if not event_md_peak.empty else 0
                
                # Tariff-specific MD cost impact calculation
                md_excess_during_peak = 0
                md_peak_load_during_event = 0
                md_peak_time = None
                has_md_cost_impact = False
                
                if is_tou_tariff:
                    # TOU tariffs: Only events during MD recording periods have MD cost impact
                    if not event_md_peak.empty:
                        md_peak_load_during_event = event_md_peak['power'].max()
                        md_peak_time = event_md_peak[event_md_peak['power'] == md_peak_load_during_event].index[0]
                        md_excess_during_peak = md_peak_load_during_event - target_demand
                        has_md_cost_impact = True
                else:
                    # General tariffs: ALL events above target have MD cost impact 24/7
                    md_peak_load_during_event = peak_load
                    md_peak_time = event_df[event_df['power'] == peak_load].index[0]
                    md_excess_during_peak = excess
                    has_md_cost_impact = True
                
                md_cost_impact = md_excess_during_peak * total_md_rate if md_excess_during_peak > 0 and total_md_rate > 0 else 0
                
                event_summaries.append({
                    'Start Date': start_time.date(),
                    'Start Time': start_time.strftime('%H:%M'),
                    'End Date': end_time.date(),
                    'End Time': end_time.strftime('%H:%M'),
                    'General Peak Load (kW)': peak_load,
                    'General Excess (kW)': excess,
                    'TOU Peak Load (kW)': md_peak_load_during_event,
                    'TOU Excess (kW)': md_excess_during_peak,
                    'TOU Peak Time': md_peak_time.strftime('%H:%M') if md_peak_time else 'N/A',
                    'Duration (min)': duration_minutes,
                    'General Required Energy (kWh)': total_energy_to_shave,
                    'TOU Required Energy (kWh)': md_peak_energy_to_shave,
                    'MD Cost Impact (RM)': md_cost_impact,
                    'Has MD Cost Impact': has_md_cost_impact,
                    'Tariff Type': 'TOU' if is_tou_tariff else 'General'
                })
            
            # Reset for next potential event
            event_data = []
    
    # Handle case where event is still ongoing at end of data
    if in_event and event_data:
        event_df = pd.DataFrame(event_data, columns=['timestamp', 'power'])
        event_df.set_index('timestamp', inplace=True)
        
        start_time = event_df.index[0]
        end_time = event_df.index[-1]
        peak_load = event_df['power'].max()
        excess = peak_load - target_demand
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Calculate energy to shave for entire event duration
        total_energy_to_shave = ((event_df['power'] - target_demand) * interval_hours).sum()
        
        # Calculate energy to shave during MD peak period only
        md_peak_mask = event_df.index.to_series().apply(
            lambda ts: is_md_window(ts, holidays)
        )
        event_md_peak = event_df[md_peak_mask]
        md_peak_energy_to_shave = ((event_md_peak['power'] - target_demand) * interval_hours).sum() if not event_md_peak.empty else 0
        
        # Tariff-specific MD cost impact calculation
        md_excess_during_peak = 0
        md_peak_load_during_event = 0
        md_peak_time = None
        has_md_cost_impact = False
        
        if is_tou_tariff:
            if not event_md_peak.empty:
                md_peak_load_during_event = event_md_peak['power'].max()
                md_peak_time = event_md_peak[event_md_peak['power'] == md_peak_load_during_event].index[0]
                md_excess_during_peak = md_peak_load_during_event - target_demand
                has_md_cost_impact = True
        else:
            md_peak_load_during_event = peak_load
            md_peak_time = event_df[event_df['power'] == peak_load].index[0]
            md_excess_during_peak = excess
            has_md_cost_impact = True
        
        md_cost_impact = md_excess_during_peak * total_md_rate if md_excess_during_peak > 0 and total_md_rate > 0 else 0
        
        event_summaries.append({
            'Start Date': start_time.date(),
            'Start Time': start_time.strftime('%H:%M'),
            'End Date': end_time.date(),
            'End Time': end_time.strftime('%H:%M'),
            'General Peak Load (kW)': peak_load,
            'General Excess (kW)': excess,
            'TOU Peak Load (kW)': md_peak_load_during_event,
            'TOU Excess (kW)': md_excess_during_peak,
            'TOU Peak Time': md_peak_time.strftime('%H:%M') if md_peak_time else 'N/A',
            'Duration (min)': duration_minutes,
            'General Required Energy (kWh)': total_energy_to_shave,
            'TOU Required Energy (kWh)': md_peak_energy_to_shave,
            'MD Cost Impact (RM)': md_cost_impact,
            'Has MD Cost Impact': has_md_cost_impact,
            'Tariff Type': 'TOU' if is_tou_tariff else 'General'
        })
    
    return event_summaries


def _filter_events_by_period(event_summaries, filter_type, selected_tariff=None):
    """Filter events based on whether they occur during peak periods or have MD cost impact."""
    if filter_type == "All Events":
        return event_summaries
    
    # Determine tariff type for filter options
    tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
    tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
    
    filtered_events = []
    for event in event_summaries:
        start_date = event['Start Date']
        start_time_str = event['Start Time']
        
        # Parse the start time to check if it's in peak period
        start_hour = int(start_time_str.split(':')[0])
        start_weekday = start_date.weekday()  # 0=Monday, 6=Sunday
        
        # Check if event starts during RP4 MD peak hours (2 PM-10 PM, weekdays)
        is_peak_period_event = (start_weekday < 5) and (14 <= start_hour < 22)
        
        # Check if event has MD cost impact (different logic for TOU vs General tariffs)
        has_md_cost_impact = event.get('Has MD Cost Impact', False)
        
        # Apply filters based on type
        if filter_type == "Peak Period Only" and is_peak_period_event:
            filtered_events.append(event)
        elif filter_type == "Off-Peak Period Only" and not is_peak_period_event:
            filtered_events.append(event)
        elif filter_type == "MD Cost Impact Events" and has_md_cost_impact:
            filtered_events.append(event)
        elif filter_type == "No MD Cost Impact Events" and not has_md_cost_impact:
            filtered_events.append(event)
    
    return filtered_events


def _display_peak_event_results(df, power_col, event_summaries, target_demand, total_md_rate, 
                               overall_max_demand, interval_hours, event_filter, show_detailed_analysis, 
                               selected_tariff=None, holidays=None):
    """Display peak event detection results and analysis with tariff-specific enhancements."""
    
    st.subheader("⚡ Peak Event Detection Results")
    
    # Determine tariff type for display enhancements
    tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
    tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
    
    # Filter events based on selection
    filtered_events = _filter_events_by_period(event_summaries, event_filter, selected_tariff)
    
    if not filtered_events:
        st.warning(f"No events found for '{event_filter}' filter.")
        return
    
    # Enhanced summary with tariff context
    total_events = len(event_summaries)
    filtered_count = len(filtered_events)
    
    if is_tou_tariff:
        md_impact_events = len([e for e in event_summaries if e.get('Has MD Cost Impact', False)])
        no_md_impact_events = total_events - md_impact_events
        summary_text = f"**Showing {filtered_count} of {total_events} total events ({event_filter})**\n"
        summary_text += f"📊 **TOU Tariff Summary:** {md_impact_events} events with MD cost impact, {no_md_impact_events} events without MD impact"
    else:
        summary_text = f"**Showing {filtered_count} of {total_events} total events ({event_filter})**\n"
        summary_text += f"📊 **General Tariff:** All {total_events} events have MD cost impact (24/7 MD charges)"
    
    st.markdown(summary_text)
    
    # Prepare dataframe with color-coding information
    df_events_summary = pd.DataFrame(filtered_events)
    
    # Create styled dataframe with color-coded rows
    def apply_row_colors(row):
        """Apply color coding based on MD cost impact."""
        has_impact = row.get('Has MD Cost Impact', False)
        if has_impact:
            return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)  # Light red for MD cost impact
        else:
            return ['background-color: rgba(0, 128, 0, 0.1)'] * len(row)  # Light green for no MD cost impact
    
    # Apply styling and formatting
    styled_df = df_events_summary.style.apply(apply_row_colors, axis=1).format({
        'General Peak Load (kW)': lambda x: fmt(x),
        'General Excess (kW)': lambda x: fmt(x),
        'TOU Peak Load (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
        'TOU Excess (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
        'Duration (min)': '{:.1f}',
        'General Required Energy (kWh)': lambda x: fmt(x),
        'TOU Required Energy (kWh)': lambda x: fmt(x),
        'MD Cost Impact (RM)': lambda x: f'RM {fmt(x)}' if x is not None else 'RM 0.0000'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced explanation with tariff-specific context
    if is_tou_tariff:
        explanation = """
    **Column Explanations (TOU Tariff):**
    - **General Peak Load (kW)**: Highest demand during entire event period (may include off-peak hours)
    - **General Excess (kW)**: Overall event peak minus target (for reference only)
    - **TOU Peak Load (kW)**: Highest demand during MD recording hours only (2PM-10PM, weekdays)
    - **TOU Excess (kW)**: MD peak load minus target - determines MD cost impact
    - **TOU Peak Time**: Exact time when MD peak occurred (for MD cost calculation)
    - **General Required Energy (kWh)**: Total energy above target for entire event duration
    - **TOU Required Energy (kWh)**: Energy above target during MD recording hours only
    - **MD Cost Impact**: MD Excess (kW) × MD Rate - **ONLY for events during 2PM-10PM weekdays**
    
    **🎨 Row Colors:**
    - 🔴 **Red background**: Events with MD cost impact (occur during 2PM-10PM weekdays)
    - 🟢 **Green background**: Events without MD cost impact (occur during off-peak periods)
        """
    else:
        explanation = """
    **Column Explanations (General Tariff):**
    - **General Peak Load (kW)**: Highest demand during entire event period
    - **General Excess (kW)**: Event peak minus target
    - **TOU Peak Load (kW)**: Same as Peak Load (General tariffs have 24/7 MD impact)
    - **TOU Excess (kW)**: Same as Excess (all events affect MD charges)
    - **TOU Peak Time**: Time when peak occurred
    - **General Required Energy (kWh)**: Total energy above target for entire event duration
    - **TOU Required Energy (kWh)**: Energy above target during MD recording hours only
    - **MD Cost Impact**: MD Excess (kW) × MD Rate - **ALL events have MD cost impact 24/7**
    
    **🎨 Row Colors:**
    - 🔴 **Red background**: All events have MD cost impact (General tariffs charge MD 24/7)
        """
    
    st.info(explanation)
    
    # Visualization of events
    _display_peak_events_chart(df, power_col, filtered_events, target_demand, selected_tariff, holidays)
    
    if show_detailed_analysis:
        # Peak Event Summary & Analysis
        _display_peak_event_analysis(filtered_events, total_md_rate)


def _display_peak_events_chart(df, power_col, event_summaries, target_demand, selected_tariff=None, holidays=None):
    """Display peak events visualization chart."""
    st.subheader("📈 Peak Events Timeline")
    
    # Create the main power consumption chart
    fig_events = go.Figure()
    
    # Add enhanced conditional coloring for power consumption line with tariff-specific logic
    fig_events = create_conditional_demand_line_with_peak_logic(
        fig_events, df, power_col, target_demand, selected_tariff, holidays, trace_name="Power Consumption"
    )
    
    # Add target demand line
    fig_events.add_hline(
        y=target_demand,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: {fmt(target_demand)} kW"
    )
    
    # Highlight peak events using tariff/holiday-aware classification
    has_peak_period_events = False
    has_offpeak_period_events = False
    
    for event in event_summaries:
        start_date = event['Start Date']
        start_time_str = event['Start Time']
        end_date = event['End Date']
        
        # Determine the event mask
        if start_date == end_date:
            event_mask = (df.index.date == start_date) & \
                         (df.index.strftime('%H:%M') >= event['Start Time']) & \
                         (df.index.strftime('%H:%M') <= event['End Time'])
        else:
            event_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        
        if event_mask.any():
            event_data = df[event_mask]
            
            # Classify each timestamp according to tariff/holiday logic
            if selected_tariff:
                classifications = [get_tariff_period_classification(ts, selected_tariff, holidays) for ts in event_data.index]
            else:
                classifications = [get_period_classification(ts, holidays) for ts in event_data.index]
            
            # Decide event type by majority of samples
            peak_count = sum(1 for c in classifications if c == 'Peak')
            offpeak_count = len(classifications) - peak_count
            is_peak_period_event = peak_count >= offpeak_count
            
            # Choose colors based on period
            if is_peak_period_event:
                fill_color = 'rgba(255, 0, 0, 0.2)'  # Semi-transparent red
                event_type = 'Peak Period Event'
                has_peak_period_events = True
            else:
                fill_color = 'rgba(0, 128, 0, 0.2)'  # Semi-transparent green
                event_type = 'Off-Peak Period Event'
                has_offpeak_period_events = True
            
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
            daily_kwh_total = sum(e['General Required Energy (kWh)'] for e in day_events)
            daily_md_kwh_total = sum(e['TOU Required Energy (kWh)'] for e in day_events)
            daily_kwh_ranges.append(daily_kwh_total)
            daily_md_kwh_ranges.append(daily_md_kwh_total)
            
            # For MD cost calculation: find highest MD excess during peak periods
            for event in day_events:
                if event['TOU Required Energy (kWh)'] > 0:
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
            
            # Calculate energy to shave during MD peak period only - IMPROVED HIERARCHY
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
                

# ============================================================================
# BATTERY ENERGY STORAGE SYSTEM (BESS) ANALYSIS FUNCTIONS
# ============================================================================

def _get_battery_parameters(event_summaries=None):
    """Get battery system parameters from sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔋 Battery System Parameters")
    
    # Calculate defaults from event summaries
    default_capacity = 500
    default_power = 250
    
    if event_summaries:
        # Get maximum energy to shave (peak period only) and maximum MD excess
        max_energy_peak_only = max(event.get('TOU Required Energy (kWh)', 0) for event in event_summaries)
        max_md_excess = max(event.get('TOU Excess (kW)', 0) for event in event_summaries if event.get('TOU Excess (kW)', 0) > 0)
        
        if max_energy_peak_only > 0:
            default_capacity = max_energy_peak_only
        if max_md_excess > 0:
            default_power = max_md_excess
    
    with st.sidebar.expander("⚙️ BESS Configuration", expanded=False):
        battery_params = {}
        
        # Battery Technology
        battery_params['technology'] = st.selectbox(
            "Battery Technology",
            ["Lithium-ion (Li-ion)", "Lithium Iron Phosphate (LiFePO4)", "Sodium-ion"],
            index=1,  # Default to LiFePO4
            help="Different battery technologies have different costs and characteristics"
        )
        
        # System Sizing Approach
        battery_params['sizing_approach'] = st.selectbox(
            "Sizing Approach",
            ["Auto-size for Peak Events", "Manual Capacity", "Energy Duration-based"],
            help="Choose how to determine the battery capacity"
        )
        
        if battery_params['sizing_approach'] == "Manual Capacity":
            st.markdown("**Manual Battery Sizing with Safety Factors**")
            
            # Capacity Safety Factor
            capacity_safety_factor = st.slider(
                "Capacity Safety Factor (%)", 
                min_value=0, 
                max_value=100, 
                value=20, 
                step=5,
                help="Additional capacity buffer above minimum requirement"
            )
            
            # Power Safety Factor
            power_safety_factor = st.slider(
                "Power Rating Safety Factor (%)", 
                min_value=0, 
                max_value=100, 
                value=15, 
                step=5,
                help="Additional power rating buffer above minimum requirement"
            )
            
            # Calculate suggested values with safety factors
            suggested_capacity = default_capacity * (1 + capacity_safety_factor / 100)
            suggested_power = default_power * (1 + power_safety_factor / 100)
            
            # Display recommendations
            st.info(f"""
            **📊 Sizing Recommendations:**
            - **Base Capacity**: {default_capacity:.0f} kWh (from peak events)
            - **With {capacity_safety_factor}% safety**: {suggested_capacity:.0f} kWh
            - **Base Power**: {default_power:.0f} kW (from MD excess)
            - **With {power_safety_factor}% safety**: {suggested_power:.0f} kW
            """)
            
            battery_params['manual_capacity_kwh'] = st.number_input(
                "Battery Capacity (kWh)", 
                min_value=10, 
                max_value=10000, 
                value=int(suggested_capacity), 
                step=10,
                help=f"Suggested: {suggested_capacity:.0f} kWh (includes {capacity_safety_factor}% safety factor)"
            )
            battery_params['manual_power_kw'] = st.number_input(
                "Battery Power Rating (kW)", 
                min_value=10, 
                max_value=5000, 
                value=int(suggested_power), 
                step=10,
                help=f"Suggested: {suggested_power:.0f} kW (includes {power_safety_factor}% safety factor)"
            )
            
            # Store safety factors for reference
            battery_params['capacity_safety_factor'] = capacity_safety_factor
            battery_params['power_safety_factor'] = power_safety_factor
            
        elif battery_params['sizing_approach'] == "Energy Duration-based":
            # Duration-based with safety factor
            st.markdown("**Duration-based Sizing with Safety Factor**")
            
            duration_safety_factor = st.slider(
                "Duration Safety Factor (%)", 
                min_value=0, 
                max_value=100, 
                value=25, 
                step=5,
                help="Additional duration buffer for extended peak events"
            )
            
            battery_params['duration_hours'] = st.number_input(
                "Discharge Duration (hours)", 
                min_value=0.5, 
                max_value=8.0, 
                value=2.0, 
                step=0.5,
                help="How many hours the battery should provide peak shaving"
            )
            
            battery_params['duration_safety_factor'] = duration_safety_factor
            
        else:  # Auto-size for Peak Events
            st.markdown("**Auto-sizing Safety Factors**")
            
            auto_capacity_safety = st.slider(
                "Auto-sizing Capacity Safety (%)", 
                min_value=10, 
                max_value=50, 
                value=20, 
                step=5,
                help="Safety margin for auto-calculated battery capacity"
            )
            
            auto_power_safety = st.slider(
                "Auto-sizing Power Safety (%)", 
                min_value=10, 
                max_value=50, 
                value=15, 
                step=5,
                help="Safety margin for auto-calculated power rating"
            )
            
            battery_params['auto_capacity_safety'] = auto_capacity_safety
            battery_params['auto_power_safety'] = auto_power_safety
        
        # Battery System Parameters
        st.markdown("**System Specifications**")
        battery_params['depth_of_discharge'] = st.slider(
            "Depth of Discharge (%)", 
            min_value=70, 
            max_value=95, 
            value=85, 
            step=5,
            help="Maximum usable capacity (affects battery life)"
        )
        
        battery_params['round_trip_efficiency'] = st.slider(
            "Round-trip Efficiency (%)", 
            min_value=85, 
            max_value=98, 
            value=92, 
            step=1,
            help="Energy efficiency of charge/discharge cycle (used in simulation)"
        )
        
        battery_params['discharge_efficiency'] = st.slider(
            "Discharge Efficiency (%)", 
            min_value=85, 
            max_value=98, 
            value=94, 
            step=1,
            help="Energy delivered to load during discharge (used in battery quantity calculation)"
        )
        
        battery_params['c_rate'] = st.slider(
            "C-Rate (Charge/Discharge)", 
            min_value=0.2, 
            max_value=2.0, 
            value=0.5, 
            step=0.1,
            help="Maximum charge/discharge rate relative to capacity"
        )
        
        # Financial Parameters
        st.markdown("**Financial Parameters**")
        battery_params['capex_per_kwh'] = st.number_input(
            "Battery Cost (RM/kWh)", 
            min_value=500, 
            max_value=3000, 
            value=1200, 
            step=50,
            help="Capital cost per kWh of battery capacity"
        )
        
        battery_params['pcs_cost_per_kw'] = st.number_input(
            "Power Conversion System (RM/kW)", 
            min_value=200, 
            max_value=1000, 
            value=400, 
            step=25,
            help="Cost of inverter/PCS per kW of power rating"
        )
        
        battery_params['installation_factor'] = st.slider(
            "Installation & Integration Factor", 
            min_value=1.1, 
            max_value=2.0, 
            value=1.4, 
            step=0.1,
            help="Multiplier for installation, civil works, and system integration"
        )
        
        battery_params['opex_percent'] = st.slider(
            "Annual O&M (% of CAPEX)", 
            min_value=1.0, 
            max_value=8.0, 
            value=3.0, 
            step=0.5,
            help="Annual operation and maintenance cost"
        )
        
        battery_params['battery_life_years'] = st.number_input(
            "Battery Life (years)", 
            min_value=5, 
            max_value=25, 
            value=15, 
            step=1,
            help="Expected operational life of the battery system"
        )
        
        battery_params['discount_rate'] = st.slider(
            "Discount Rate (%)", 
            min_value=3.0, 
            max_value=15.0, 
            value=8.0, 
            step=0.5,
            help="Discount rate for NPV calculations"
        )
    
    return battery_params


def _perform_battery_analysis(df, power_col, event_summaries, target_demand, 
                             interval_hours, battery_params, total_md_rate, selected_tariff=None, holidays=None):
    """Perform comprehensive battery analysis with TOU tariff awareness."""
    
    # Calculate required battery capacity and power
    battery_sizing = _calculate_battery_sizing(
        event_summaries, target_demand, interval_hours, battery_params
    )
    
    # Calculate battery costs
    battery_costs = _calculate_battery_costs(battery_sizing, battery_params)
    
    # Simulate battery operation
    battery_simulation = _simulate_battery_operation(
        df, power_col, target_demand, battery_sizing, battery_params, interval_hours, selected_tariff, holidays
    )
    
    # Calculate financial metrics
    financial_analysis = _calculate_financial_metrics(
        battery_costs, event_summaries, total_md_rate, battery_params
    )
    
    return {
        'sizing': battery_sizing,
        'costs': battery_costs,
        'simulation': battery_simulation,
        'financial': financial_analysis,
        'params': battery_params
    }


def _calculate_battery_sizing(event_summaries, target_demand, interval_hours, battery_params):
    """Calculate optimal battery sizing basedon peak events."""
    
    if battery_params['sizing_approach'] == "Manual Capacity":
        return {
            'capacity_kwh': battery_params['manual_capacity_kwh'],
            'power_rating_kw': battery_params['manual_power_kw'],
            'sizing_method': f"Manual Configuration (Capacity: +{battery_params.get('capacity_safety_factor', 0)}% safety, Power: +{battery_params.get('power_safety_factor', 0)}% safety)",
            'safety_factors_applied': True
        }
    
    # Calculate energy requirements from peak events using the correct event data
    total_energy_to_shave = 0
    max_power_requirement = 0
    worst_event_energy_peak_only = 0
    max_md_excess = 0
    
    for event in event_summaries:
        # Use TOU Required Energy (kWh) for capacity sizing
        energy_kwh_peak_only = event.get('TOU Required Energy (kWh)', 0)
        # Use TOU Excess (kW) for power sizing
        md_excess_power = event.get('TOU Excess (kW)', 0)
        
        total_energy_to_shave += energy_kwh_peak_only
        worst_event_energy_peak_only = max(worst_event_energy_peak_only, energy_kwh_peak_only)
        max_md_excess = max(max_md_excess, md_excess_power)
    
    if battery_params['sizing_approach'] == "Auto-size for Peak Events":
        # Size based on worst-case event during peak periods only
        if event_summaries and worst_event_energy_peak_only > 0:
            required_capacity = worst_event_energy_peak_only / (battery_params['depth_of_discharge'] / 100)
            required_power = max_md_excess
            
            # Apply auto-sizing safety factors
            capacity_safety = battery_params.get('auto_capacity_safety', 20) / 100
            power_safety = battery_params.get('auto_power_safety', 15) / 100
            
            required_capacity *= (1 + capacity_safety)
            required_power *= (1 + power_safety)
            
            sizing_method = f"Auto-sized for worst MD peak event ({worst_event_energy_peak_only:.1f} kWh + {battery_params.get('auto_capacity_safety', 20)}% safety)"
        else:
            required_capacity = 100  # Minimum
            required_power = 50
            sizing_method = "Default minimum sizing (no peak events detected)"
            
    else:  # Energy Duration-based
        required_power = max_md_excess
        required_capacity = required_power * battery_params['duration_hours']
        required_capacity = required_capacity / (battery_params['depth_of_discharge'] / 100)
        
        # Apply duration safety factor
        duration_safety = battery_params.get('duration_safety_factor', 25) / 100
        required_capacity *= (1 + duration_safety)
        
        sizing_method = f"Duration-based ({battery_params['duration_hours']} hours + {battery_params.get('duration_safety_factor', 25)}% safety)"
    
    # Apply C-rate constraints
    c_rate_capacity = required_power / battery_params.get('c_rate', 0.5)
    final_capacity = max(required_capacity, c_rate_capacity)
    
    return {
        'capacity_kwh': final_capacity,
        'power_rating_kw': required_power,
        'required_energy_kwh': total_energy_to_shave,
        'worst_event_energy_peak_only': worst_event_energy_peak_only,
        'max_md_excess': max_md_excess,
        'sizing_method': sizing_method,
        'c_rate_limited': final_capacity > required_capacity,
        'safety_factors_applied': True
    }


def _calculate_battery_costs(battery_sizing, battery_params):
    """Calculate comprehensive battery system costs."""
    
    capacity_kwh = battery_sizing['capacity_kwh']
    power_kw = battery_sizing['power_rating_kw']
    
    # CAPEX Components
    battery_cost = capacity_kwh * battery_params['capex_per_kwh']
    pcs_cost = power_kw * battery_params['pcs_cost_per_kw']
    
    # Base system cost
    base_system_cost = battery_cost + pcs_cost
    
    # Total installed cost (including installation, civil works, etc.)
    total_capex = base_system_cost * battery_params['installation_factor']
    
    # Annual OPEX
    annual_opex = total_capex * (battery_params['opex_percent'] / 100)
    
    # Total lifecycle cost
    total_lifecycle_opex = annual_opex * battery_params['battery_life_years']
    total_lifecycle_cost = total_capex + total_lifecycle_opex
    
    return {
        'battery_cost': battery_cost,
        'pcs_cost': pcs_cost,
        'base_system_cost': base_system_cost,
        'total_capex': total_capex,
        'annual_opex': annual_opex,
        'total_lifecycle_opex': total_lifecycle_opex,
        'total_lifecycle_cost': total_lifecycle_cost,
        'cost_per_kwh': total_capex / capacity_kwh,
        'cost_per_kw': total_capex / power_kw
    }


def _simulate_battery_operation(df, power_col, target_demand, battery_sizing, battery_params, interval_hours, selected_tariff=None, holidays=None):
    """Simulate battery charge/discharge operation with TOU tariff awareness."""
    
    # Create simulation dataframe
    df_sim = df[[power_col]].copy()
    df_sim['Original_Demand'] = df_sim[power_col]
    df_sim['Target_Demand'] = target_demand
    df_sim['Excess_Demand'] = (df_sim[power_col] - target_demand).clip(lower=0)
    
    # Battery state variables
    battery_capacity = battery_sizing['capacity_kwh']
    usable_capacity = battery_capacity * (battery_params['depth_of_discharge'] / 100)
    max_power = battery_sizing['power_rating_kw']
    efficiency = battery_params['round_trip_efficiency'] / 100
    
    # Initialize battery state
    soc = np.zeros(len(df_sim))  # State of Charge in kWh
    soc_percent = np.zeros(len(df_sim))  # SOC as percentage
    battery_power = np.zeros(len(df_sim))  # Positive = discharge, Negative = charge
    net_demand = df_sim[power_col].copy()
    
    # Simple charging strategy: charge during low demand, discharge during high demand
    # For this simulation, we'll focus on peak shaving
    
    for i in range(len(df_sim)):
        current_demand = df_sim[power_col].iloc[i]
        excess = max(0, current_demand - target_demand)
        current_timestamp = df_sim.index[i]
        
        # Determine if discharge is allowed based on tariff type
        should_discharge = excess > 0
        
        if selected_tariff and should_discharge:
            # Apply TOU logic for discharge decisions
            tariff_type = selected_tariff.get('Type', '').lower()
            tariff_name = selected_tariff.get('Tariff', '').lower()
            is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
            
            if is_tou_tariff:
                # TOU tariffs: Only discharge during peak periods (2PM-10PM weekdays)
                period_classification = get_tariff_period_classification(current_timestamp, selected_tariff, holidays)
                should_discharge = (excess > 0) and (period_classification == 'Peak')
            # For General tariffs, discharge anytime above target (original behavior)
        
        if should_discharge:  # Discharge battery with TOU awareness
            # Calculate required discharge power
            required_discharge = min(excess, max_power)
            # Check if battery has enough energy
            available_energy = soc[i-1] if i > 0 else usable_capacity * 0.8  # Start at 80% SOC
            max_discharge_energy = available_energy
            max_discharge_power = min(max_discharge_energy / interval_hours, required_discharge)
            
            actual_discharge = max_discharge_power
            battery_power[i] = actual_discharge
            soc[i] = (soc[i-1] if i > 0 else usable_capacity * 0.8) - actual_discharge * interval_hours
            net_demand.iloc[i] = current_demand - actual_discharge
            
        else:  # Can charge battery if there's room and low demand
            if i > 0:
                soc[i] = soc[i-1]
            else:
                soc[i] = usable_capacity * 0.8
            
            # Enhanced charging logic with better conditions and SOC awareness
            current_time = df_sim.index[i]

            hour = current_time.hour
            soc_percentage = (soc[i] / usable_capacity) * 100
            
            # Calculate dynamic demand thresholds based on recent patterns
            lookback_periods = min(96, len(df_sim))  # 24 hours of 15-min data or available
            start_idx = max(0, i - lookback_periods)
            recent_demand = df_sim[power_col].iloc[start_idx:i+1]
            
            if len(recent_demand) > 0:
                avg_demand = recent_demand.mean()
                demand_25th = recent_demand.quantile(0.25)
            else:
                avg_demand = df_sim[power_col].mean()
                demand_25th = avg_demand * 0.6
            
            # Determine charging conditions based on SOC level and time
            should_charge = False
            charge_rate_factor = 0.3  # Default conservative rate
            
            # Very low SOC - charge aggressively (updated to 5% safety limit)
            if soc_percentage < 10:  # Updated from 30% to 10% for emergency charging only
                should_charge = current_demand < avg_demand * 0.9  # Lenient threshold
                charge_rate_factor = 0.8  # Higher charge rate
            # Low SOC - moderate charging
            elif soc_percentage < 60:
                if hour >= 22 or hour < 8:  # Off-peak hours
                    should_charge = current_demand < avg_demand * 0.8
                    charge_rate_factor = 0.6
                else:  # Peak hours - more selective
                    should_charge = current_demand < demand_25th * 1.2
                    charge_rate_factor = 0.4
            # Normal SOC - conservative charging (standardized to 95% max SOC)
            elif soc_percentage < 95:  # Standardized 95% max SOC for both TOU and General tariffs
                if hour >= 22 or hour < 8:  # Off-peak hours
                    should_charge = current_demand < avg_demand * 0.7
                    charge_rate_factor = 0.5
                else:  # Peak hours - very selective
                    should_charge = current_demand < demand_25th
                    charge_rate_factor = 0.3
            
            # Execute charging if conditions are met
            if should_charge and soc[i] < usable_capacity * 0.95:
                # Calculate charge power with improved logic
                remaining_capacity = usable_capacity * 0.95 - soc[i]
                max_charge_energy = remaining_capacity / efficiency
                
                charge_power = min(
                    max_power * charge_rate_factor,  # Dynamic charging rate
                    max_charge_energy / interval_hours,  # Energy constraint
                    remaining_capacity / interval_hours / efficiency  # Don't exceed 95% SOC
                )
                
                # Apply charging
                battery_power[i] = -charge_power  # Negative for charging
                soc[i] = soc[i] + charge_power * interval_hours * efficiency
                net_demand.iloc[i] = current_demand + charge_power
        
        # Ensure SOC stays within limits
        soc[i] = max(0, min(soc[i], usable_capacity))
        soc_percent[i] = (soc[i] / usable_capacity) * 100
    
    # Add simulation results to dataframe
    df_sim['Battery_Power_kW'] = battery_power
    df_sim['Battery_SOC_kWh'] = soc
    df_sim['Battery_SOC_Percent'] = soc_percent
    df_sim['Net_Demand_kW'] = net_demand
    df_sim['Peak_Shaved'] = df_sim['Original_Demand'] - df_sim['Net_Demand_kW']
    
    # Calculate performance metrics
    total_energy_discharged = sum([p * interval_hours for p in battery_power if p > 0])
    total_energy_charged = sum([abs(p) * interval_hours for p in battery_power if p < 0])
    
    # Calculate peak reduction using maximum value from MD peak periods (same as successful peak shaving table)
    # Filter for MD peak periods only (2 PM-10 PM, weekdays) to match the table calculation
    def is_md_peak_period_for_reduction(timestamp):
        return timestamp.weekday() < 5 and 14 <= timestamp.hour < 22
    
    df_md_peak_for_reduction = df_sim[df_sim.index.to_series().apply(is_md_peak_period_for_reduction)]
    
    if len(df_md_peak_for_reduction) > 0:
        # Calculate daily peak reduction using same logic as the successful peak shaving table
        daily_reduction_analysis = df_md_peak_for_reduction.groupby(df_md_peak_for_reduction.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max'
        }).reset_index()
        daily_reduction_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD']
        daily_reduction_analysis['Peak_Reduction'] = daily_reduction_analysis['Original_Peak_MD'] - daily_reduction_analysis['Net_Peak_MD']
        
        # Use maximum value from Peak_Reduction column (same as displayed in successful peak shaving table)
        peak_reduction = daily_reduction_analysis['Peak_Reduction'].max()
    else:
        # Fallback to original calculation if no MD peak data
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
    
    # Calculate MD-focused success rate (consistent with Daily Peak Shave Effectiveness)
    # Filter for MD peak periods only (2 PM-10 PM, weekdays)
    def is_md_peak_period(timestamp):
        return timestamp.weekday() < 5 and 14 <= timestamp.hour < 22
    
    df_md_peak = df_sim[df_sim.index.to_series().apply(is_md_peak_period)]
    
    # Store debug information for better user feedback
    debug_info = {
        'total_points': len(df_sim),
        'md_peak_points': len(df_md_peak),
        'sample_timestamps': df_sim.index[:3].tolist() if len(df_sim) > 0 else [],
        'weekdays_present': sorted(df_sim.index.to_series().apply(lambda x: x.weekday()).unique()) if len(df_sim) > 0 else [],
        'hours_present': sorted(df_sim.index.to_series().apply(lambda x: x.hour).unique()) if len(df_sim) > 0 else []
    }
    
    if len(df_md_peak) > 0:
        # Calculate daily MD success rate (EXACT same logic as Daily Peak Shave Effectiveness)
        daily_md_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max'
        }).reset_index()
        daily_md_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD']
        # Use EXACT same success criteria as Daily Peak Shave Effectiveness
        daily_md_analysis['Success'] = daily_md_analysis['Net_Peak_MD'] <= target_demand  # No tolerance
        
        successful_days = sum(daily_md_analysis['Success'])
        total_days = len(daily_md_analysis)
        success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
        md_focused_calculation = True
        
        # Store debug info about the MD calculation
        debug_info['md_calculation_details'] = {
            'successful_days': successful_days,
            'total_days': total_days,
            'calculation_method': 'MD-focused (identical to Daily Peak Shave Effectiveness)'
        }
    else:
        # Fallback to original calculation if no MD peak data
        successful_shaves = len(df_sim[
            (df_sim['Original_Demand'] > target_demand) & 
            (df_sim['Net_Demand_kW'] <= target_demand)  # Exact target match
        ])
        total_peak_events = len(df_sim[df_sim['Original_Demand'] > target_demand])
        success_rate = (successful_shaves / total_peak_events * 100) if total_peak_events > 0 else 0
        successful_days = successful_shaves
        total_days = total_peak_events
        md_focused_calculation = False
        
        # Store debug info about the fallback calculation
        debug_info['md_calculation_details'] = {
            'successful_intervals': successful_shaves,
            'total_intervals': total_peak_events,
            'calculation_method': 'Fallback 24/7 calculation (no MD data found)'
        }
    
    return {
        'df_simulation': df_sim,
        'total_energy_discharged': total_energy_discharged,
        'total_energy_charged': total_energy_charged,
        'peak_reduction_kw': peak_reduction,
        'success_rate_percent': success_rate,
        'successful_shaves': successful_days,
        'total_peak_events': total_days,
        'average_soc': np.mean(soc_percent),
        'min_soc': np.min(soc_percent),
        'max_soc': np.max(soc_percent),
        'md_focused_calculation': md_focused_calculation,
        'debug_info': debug_info  # Include debug info for better user feedback
    }


def _calculate_financial_metrics(battery_costs, event_summaries, total_md_rate, battery_params):
    """Calculate ROI, IRR, and other financial metrics."""
    
    # Calculate annual MD savings
    if event_summaries and total_md_rate > 0:
        max_monthly_md_saving = max(event['MD Cost Impact (RM)'] for event in event_summaries)
        annual_md_savings = max_monthly_md_saving * 12
    else:
        annual_md_savings = 0
    
    # Additional potential savings (simplified - could include energy arbitrage, etc.)
    # For now, focus on MD savings only
    total_annual_savings = annual_md_savings
    
    # Calculate simple payback
    if total_annual_savings > battery_costs['annual_opex']:
        net_annual_savings = total_annual_savings - battery_costs['annual_opex']
        simple_payback_years = battery_costs['total_capex'] / net_annual_savings
    else:
        simple_payback_years = float('inf')
    
    # Calculate NPV and IRR
    discount_rate = battery_params['discount_rate'] / 100
    project_years = battery_params['battery_life_years']
    
    # Cash flows: Initial investment (negative), then annual net savings
    cash_flows = [-battery_costs['total_capex']]  # Initial investment
    for year in range(1, project_years + 1):
        annual_net_cash_flow = total_annual_savings - battery_costs['annual_opex']
        cash_flows.append(annual_net_cash_flow)
    
    # Calculate NPV
    npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
    
    # Calculate IRR (simplified approximation)
    irr = _calculate_irr_approximation(cash_flows)
    
    # Calculate profitability metrics
    total_lifecycle_savings = total_annual_savings * project_years
    total_lifecycle_costs = battery_costs['total_capex'] + battery_costs['total_lifecycle_opex']
    benefit_cost_ratio = total_lifecycle_savings / total_lifecycle_costs if total_lifecycle_costs > 0 else 0
    
    # Calculate simple annual ROI based on net annual savings
    annual_roi_percent = ((total_annual_savings - battery_costs['annual_opex']) / battery_costs['total_capex'] * 100) if battery_costs['total_capex'] > 0 else 0
    
    return {
        'annual_md_savings': annual_md_savings,
        'total_annual_savings': total_annual_savings,
        'net_annual_savings': total_annual_savings - battery_costs['annual_opex'],
        'simple_payback_years': simple_payback_years,
        'npv': npv,
        'irr_percent': irr * 100 if irr is not None else None,
        'benefit_cost_ratio': benefit_cost_ratio,
        'total_lifecycle_savings': total_lifecycle_savings,
        'roi_percent': annual_roi_percent,
        'cash_flows': cash_flows
    }


def _calculate_irr_approximation(cash_flows):
    """Calculate IRR using simple approximation method."""
    try:
        # Simple Newton-Raphson approximation for IRR
        def npv_at_rate(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
        
        # Try to find IRR between 0% and 100%
        for rate in np.arange(0.01, 1.0, 0.01):
            if npv_at_rate(rate) <= 0:
                return rate
        
        return None  # IRR > 100% or not found
    except:
        return None


def _display_battery_analysis(battery_analysis, battery_params, target_demand, max_md_cost_impact=None, selected_tariff=None, holidays=None):
    """Display comprehensive battery analysis results."""
    
    sizing = battery_analysis['sizing']
    costs = battery_analysis['costs']
    simulation = battery_analysis['simulation']
    financial = battery_analysis['financial']
    
    # Battery Sizing Results
    st.markdown("### 📏 Battery System Sizing")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Battery Capacity", f"{sizing['capacity_kwh']:.0f} kWh")
    with col2:
        st.metric("Power Rating", f"{sizing['power_rating_kw']:.0f} kW")
    with col3:
        st.metric("Technology", battery_params['technology'].split(' ')[0])
    with col4:
        if 'c_rate_limited' in sizing and sizing['c_rate_limited']:
            st.metric("Sizing", "C-rate Limited", delta="⚠️")
        else:
            st.metric("Sizing", "Energy Limited", delta="✅")
    
    st.info(f"**Sizing Method:** {sizing['sizing_method']}")
    
    # Cost Analysis
    st.markdown("### 💰 Cost Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total CAPEX", f"RM {costs['total_capex']:,.0f}")
        st.caption(f"• Battery: RM {costs['battery_cost']:,.0f}\n• PCS: RM {costs['pcs_cost']:,.0f}\n• Installation: {(battery_params['installation_factor']-1)*100:.0f}%")
    with col2:
        st.metric("Annual OPEX", f"RM {costs['annual_opex']:,.0f}")
        st.caption(f"{battery_params['opex_percent']}% of CAPEX")
    with col3:
        st.metric("Total Lifecycle Cost", f"RM {costs['total_lifecycle_cost']:,.0f}")
        st.caption(f"Over {battery_params['battery_life_years']} years")
    
    # ROI and Potential MD Savings Analysis
    st.markdown("### 📈 ROI & Potential MD Savings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        roi_percent = financial.get('roi_percent', 0)
        if roi_percent > 0:
            st.metric("ROI", f"{roi_percent:.1f}%", delta="Positive", delta_color="normal")
        else:
            st.metric("ROI", f"{roi_percent:.1f}%", delta="Negative", delta_color="inverse")
        st.caption("Return on Investment")
    
    with col2:
        if max_md_cost_impact is not None:
            monthly_savings = max_md_cost_impact
            st.metric("Max Monthly MD Savings", f"RM {monthly_savings:,.0f}")
            st.caption("Peak month potential savings")
        else:
            st.metric("Monthly MD Savings", "N/A")
            st.caption("Data not available")
    
    with col3:
        if max_md_cost_impact is not None:
            annual_savings = max_md_cost_impact * 12
            st.metric("Annual MD Savings Potential", f"RM {annual_savings:,.0f}")
            st.caption("Maximum annual savings estimate")
        else:
            st.metric("Annual MD Savings", "N/A")
            st.caption("Data not available")
    
    # Additional financial insights
    if max_md_cost_impact is not None and max_md_cost_impact > 0:
        payback_period = costs['total_capex'] / (max_md_cost_impact * 12)
        if payback_period > 0:
            st.info(f"💡 **Simple Payback Period:** {payback_period:.1f} years (based on maximum MD savings)")
        
        # ROI context
        if roi_percent > 15:
            st.success("🎯 **Excellent ROI** - This battery investment shows strong financial returns!")
        elif roi_percent > 8:
            st.info("👍 **Good ROI** - This battery investment is financially viable.")
        elif roi_percent > 0:
            st.warning("⚠️ **Marginal ROI** - Consider optimizing battery sizing or exploring alternatives.")
        else:
            st.error("❌ **Negative ROI** - Current configuration may not be financially justified.")

    
    # Battery Operation Visualization
    st.markdown("#### 🔋 Battery Operation Simulation")
    _display_battery_simulation_chart(simulation['df_simulation'], target_demand, sizing, selected_tariff, holidays)


def _display_battery_simulation_chart(df_sim, target_demand=None, sizing=None, selected_tariff=None, holidays=None):
    """Display enhanced battery operation simulation chart with integrated charge/discharge visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Handle None parameters with safe defaults
    if target_demand is None:
        target_demand = df_sim['Original_Demand'].quantile(0.9) if 'Original_Demand' in df_sim.columns else 0
    if sizing is None:
        sizing = {'power_rating_kw': 100}  # Default power rating for calculations
    
    # Resolve Net Demand column name flexibly
    net_candidates = ['Net_Demand_kW', 'Net_Demand_KW', 'Net_Demand']
    net_col = next((c for c in net_candidates if c in df_sim.columns), None)
    
    # Validate required columns exist
    required_base = ['Original_Demand', 'Battery_Power_kW', 'Battery_SOC_Percent']
    missing_columns = [col for col in required_base if col not in df_sim.columns]
    if net_col is None:
        missing_columns.append('Net_Demand_kW')
    
    if missing_columns:
        st.error(f"❌ Missing required columns in simulation data: {missing_columns}")
        st.info("Available columns: " + ", ".join(df_sim.columns.tolist()))
        return
    
    # Add Target_Demand column if it doesn't exist
    if 'Target_Demand' not in df_sim.columns:
        df_sim['Target_Demand'] = target_demand
    
    # Panel 1: Main Demand Profile Chart
    st.markdown("##### 1️⃣ MD Shaving Effectiveness: Demand vs Battery vs Target")
    
    fig = go.Figure()
    
    # Other demand lines
    fig.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim[net_col], 
                  name='Net Demand (with Battery)', line=dict(color='#00BFFF', width=2),
                  hovertemplate='Net: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    fig.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim['Target_Demand'], 
                  name='Target Demand', line=dict(color='green', dash='dash', width=2),
                  hovertemplate='Target: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    
    # Replace area fills with bar charts for battery discharge/charge
    # Prepare series for bars (positive = discharge, negative = charge)
    discharge_series = df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] > 0, other=0)
    charge_series = df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] < 0, other=0)
    
    # Discharge bars
    fig.add_trace(go.Bar(
        x=df_sim.index,
        y=discharge_series,
        name='Battery Discharge (kW)',
        marker=dict(color='orange'),
        opacity=0.6,
        hovertemplate='Discharge: %{y:.1f} kW<br>%{x}<extra></extra>',
        yaxis='y2'
    ))
    
    # Charge bars (negative values)
    fig.add_trace(go.Bar(
        x=df_sim.index,
        y=charge_series,
        name='Battery Charge (kW)',
        marker=dict(color='green'),
        opacity=0.6,
        hovertemplate='Charge: %{y:.1f} kW<br>%{x}<extra></extra>',
        yaxis='y2'
    ))
    
    # Add enhanced conditional coloring for Original Demand line LAST so it renders on top
    fig = create_conditional_demand_line_with_peak_logic(
        fig, df_sim, 'Original_Demand', target_demand, selected_tariff, holidays, "Original Demand"
    )
    
    # Compute symmetric range for y2 to show positive/negative bars
    try:
        max_abs_power = float(df_sim['Battery_Power_kW'].abs().max())
    except Exception:
        max_abs_power = float(sizing.get('power_rating_kw', 100))
    y2_limit = max(max_abs_power * 1.1, sizing.get('power_rating_kw', 100) * 0.5)
    
    fig.update_layout(
        title='🎯 MD Shaving Effectiveness: Demand vs Battery vs Target',
        xaxis_title='Time',
        yaxis_title='Power Demand (kW)',
        yaxis2=dict(
            title='Battery Power (kW) [+ discharge | - charge]',
            overlaying='y',
            side='right',
            range=[-y2_limit, y2_limit],
            zeroline=True,
            zerolinecolor='gray'
        ),
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Success/Failure Analysis (MD Peak Periods Only) - IMPROVED HIERARCHY
    df_md_peak_sim = df_sim[df_sim.index.to_series().apply(lambda ts: (ts.weekday() < 5) and (14 <= ts.hour < 22))]  # Holiday check would require holidays parameter
    
    if len(df_md_peak_sim) > 0:
        success_intervals = len(df_md_peak_sim[
            (df_md_peak_sim['Original_Demand'] > target_demand) & 
            (df_md_peak_sim[net_col] <= target_demand)
        ])
        total_peak_intervals = len(df_md_peak_sim[df_md_peak_sim['Original_Demand'] > target_demand])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak Intervals Above Target (MD Periods)", f"{total_peak_intervals}")
        col2.metric("Successfully Shaved (MD Periods)", f"{success_intervals}")
        col3.metric("MD Success Rate", f"{(success_intervals/total_peak_intervals*100) if total_peak_intervals > 0 else 0:.1f}%")
    else:
        success_intervals = 0
        total_peak_intervals = 0
        st.warning("⚠️ No MD peak period data available for interval-based analysis")
    
    # Panel 2: Combined SOC and Battery Power Chart
    st.markdown("##### 2️⃣ Combined SOC and Battery Power Chart")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # SOC line (left y-axis)
    fig2.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim['Battery_SOC_Percent'],
                  name='SOC (%)', line=dict(color='purple', width=2),
                  hovertemplate='SOC: %{y:.1f}%<br>%{x}<extra></extra>'),
        secondary_y=False
    )
    
    # Battery power line (right y-axis) 
    fig2.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim['Battery_Power_kW'],
                  name='Battery Power', line=dict(color='orange', width=2),
                  hovertemplate='Power: %{y:.1f} kW<br>%{x}<extra></extra>'),
        secondary_y=True
    )
    
    # Add horizontal line for minimum SOC warning
    fig2.add_hline(y=20, line_dash="dot", line_color="red", 
                   annotation_text="Low SOC Warning (20%)", secondary_y=False)
    
    # Update axes
    fig2.update_xaxes(title_text="Time")
    fig2.update_yaxes(title_text="State of Charge (%)", secondary_y=False, range=[0, 100])
    fig2.update_yaxes(title_text="Battery Discharge Power (kW)", secondary_y=True)
    
    fig2.update_layout(
        title='⚡ SOC vs Battery Power: Timing Analysis',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Panel 3: Battery Power Utilization Heatmap
    st.markdown("##### 3️⃣ Battery Power Utilization Heatmap")
    
    # Prepare data for heatmap
    df_heatmap = df_sim.copy()
    df_heatmap['Date'] = df_heatmap.index.date
    df_heatmap['Hour'] = df_heatmap.index.hour
    df_heatmap['Battery_Utilization_%'] = (df_heatmap['Battery_Power_kW'] / sizing['power_rating_kw'] * 100).clip(0, 100)
    
    # Create pivot table for heatmap
    heatmap_data = df_heatmap.pivot_table(
        values='Battery_Utilization_%', 
        index='Hour', 
        columns='Date', 
        aggfunc='mean',
        fill_value=0
    )
    
    # Create heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(d) for d in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Date: %{x}<br>Hour: %{y}<br>Utilization: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Battery Utilization (%)")
    ))
    
    fig3.update_layout(
        title='🔥 Battery Power Utilization Heatmap (% of Rated Power)',
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Panel 4: Daily Peak Shave Effectiveness Analysis (MD Peak Periods Only)
    st.markdown("##### 4️⃣ Daily Peak Shave Effectiveness & Success Analysis (MD Peak Periods Only)")
    
    # Filter data for MD peak periods only (2 PM-10 PM, weekdays)
    def is_md_peak_period_for_effectiveness(timestamp):
        return timestamp.weekday() < 5 and 14 <= timestamp.hour < 22
    
    df_md_peak = df_sim[df_sim.index.to_series().apply(is_md_peak_period_for_effectiveness)]
    
    # Calculate daily analysis using MD peak periods only
    if len(df_md_peak) > 0:
        daily_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Battery_Power_kW': 'max',
            'Battery_SOC_Percent': ['min', 'mean']
        }).reset_index()
        
        # Flatten column names
        daily_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Max_Battery_Power', 'Min_SOC', 'Avg_SOC']
        
        # Calculate detailed metrics based on MD peak periods only
        md_rate_estimate = 97.06  # RM/kW from Medium Voltage TOU
        daily_analysis['Peak_Reduction'] = daily_analysis['Original_Peak_MD'] - daily_analysis['Net_Peak_MD']
        daily_analysis['Est_Monthly_Saving'] = daily_analysis['Peak_Reduction'] * md_rate_estimate
        daily_analysis['Success'] = daily_analysis['Net_Peak_MD'] <= target_demand  # No tolerance
        daily_analysis['Peak_Shortfall'] = (daily_analysis['Net_Peak_MD'] - target_demand).clip(lower=0)
        daily_analysis['Required_Additional_Power'] = daily_analysis['Peak_Shortfall']
        
        # Add informational note about MD-focused analysis
        st.info("""
        📋 **MD-Focused Analysis Note:**
        This analysis considers only **MD peak periods (2-10 PM weekdays)** for success/failure calculation.
        Success rate reflects effectiveness during actual MD recording hours, not 24/7 performance.
        """)
    else:
        st.warning("⚠️ No MD peak period data found (weekdays 2-10 PM). Cannot calculate MD-focused effectiveness.")
        return
    
    # Categorize failure reasons
    def categorize_failure_reason(row):
        if row['Success']:
            return 'Success'
        elif row['Min_SOC'] < 10:
            return 'Low SOC (Battery Depleted)'
        elif row['Max_Battery_Power'] < sizing['power_rating_kw'] * 0.9:
            return 'Insufficient Battery Power'
        elif row['Peak_Shortfall'] > sizing['power_rating_kw']:
            return 'Demand Exceeds Battery Capacity'
        else:
            return 'Other (Algorithm/Timing)'
    
    daily_analysis['Failure_Reason'] = daily_analysis.apply(categorize_failure_reason, axis=1)
    
    # Create enhanced visualization
    fig4 = go.Figure()
    
    # Target line
    fig4.add_hline(y=target_demand, line_dash="dash", line_color="green", line_width=2,
                   annotation_text=f"MD Target: {target_demand:.0f} kW")
    
    # Color code bars based on success/failure
    bar_colors = ['green' if success else 'red' for success in daily_analysis['Success']]
    
    # Original peaks (MD peak periods only)
    fig4.add_trace(go.Bar(
        x=daily_analysis['Date'], y=daily_analysis['Original_Peak_MD'],
        name='Original Peak (MD Periods)', marker_color='lightcoral', opacity=0.6,
        hovertemplate='Original MD Peak: %{y:.0f} kW<br>Date: %{x}<extra></extra>'
    ))
    
    # Net peaks (after battery) - color coded by success
    fig4.add_trace(go.Bar(
        x=daily_analysis['Date'], y=daily_analysis['Net_Peak_MD'],
        name='Net Peak (MD Periods with Battery)', 
        marker_color=bar_colors, opacity=0.8,
        hovertemplate='Net MD Peak: %{y:.0f} kW<br>Status: %{customdata}<br>Date: %{x}<extra></extra>',
        customdata=['SUCCESS' if s else 'FAILED' for s in daily_analysis['Success']]
    ))
    
    fig4.update_layout(
        title='📊 Daily Peak Shaving Effectiveness - MD Periods Only (Green=Success, Red=Failed)',
        xaxis_title='Date',
        yaxis_title='Peak Demand during MD Hours (kW)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Summary stats
    total_days = len(daily_analysis)
    successful_days = sum(daily_analysis['Success'])
    failed_days = total_days - successful_days
    success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", f"{total_days}")
    col2.metric("Successful Days", f"{successful_days}", delta=f"{success_rate:.1f}%")
    col3.metric("Failed Days", f"{failed_days}", delta=f"{100-success_rate:.1f}%")
    col4.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Detailed Success/Failure Analysis
    st.markdown("#### 🔍 Detailed Success/Failure Analysis")
    
    # Separate successful and failed events
    successful_events = daily_analysis[daily_analysis['Success']].copy()
    failed_events = daily_analysis[~daily_analysis['Success']].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### ✅ **Successful Peak Shaving Days (MD Periods)**")
        if len(successful_events) > 0:
            st.dataframe(successful_events[['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Peak_Reduction', 'Min_SOC']].style.format({
                'Original_Peak_MD': '{:.1f} kW',
                'Net_Peak_MD': '{:.1f} kW', 
                'Peak_Reduction': '{:.1f} kW',
                'Min_SOC': '{:.1f}%'
            }), use_container_width=True)
            
            avg_reduction = successful_events['Peak_Reduction'].mean()
            avg_min_soc = successful_events['Min_SOC'].mean()
            st.info(f"📊 **Success Patterns (MD Periods):**\n- Average Peak Reduction: {avg_reduction:.1f} kW\n- Average Minimum SOC: {avg_min_soc:.1f}%")
        else:
            st.warning("No successful peak shaving days found during MD periods.")
    
    with col2:
        st.markdown("##### ❌ **Failed Peak Shaving Days (MD Periods)**")
        if len(failed_events) > 0:
            st.dataframe(failed_events[['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Peak_Shortfall', 'Required_Additional_Power', 'Failure_Reason', 'Min_SOC']].style.format({
                'Original_Peak_MD': '{:.1f} kW',
                'Net_Peak_MD': '{:.1f} kW',
                'Peak_Shortfall': '{:.1f} kW',
                'Required_Additional_Power': '{:.1f} kW',
                'Min_SOC': '{:.1f}%'
            }), use_container_width=True)
            
            # Analyze failure patterns
            failure_reasons = failed_events['Failure_Reason'].value_counts()
            st.error("🚫 **Failure Analysis (MD Periods):**")
            for reason, count in failure_reasons.items():
                st.write(f"- {reason}: {count} days ({count/len(failed_events)*100:.1f}%)")
        else:
            st.success("🎉 All MD peak periods were successfully managed!")
    
    # Recommendations for 100% Success Rate
    if failed_days > 0:
        st.markdown("#### 🎯 **Recommendations to Achieve 100% Success Rate**")
        
        # Analyze what's needed
        max_shortfall = daily_analysis['Peak_Shortfall'].max()
        avg_shortfall = daily_analysis[daily_analysis['Peak_Shortfall'] > 0]['Peak_Shortfall'].mean()
        low_soc_days = sum(daily_analysis['Min_SOC'] < 20)
        
        recommendations = []
        
        if max_shortfall > 0:
            current_power = sizing['power_rating_kw']
            recommended_power = current_power + max_shortfall * 1.1  # 10% safety margin
            recommendations.append(f"🔋 **Increase Battery Power Rating**: From {current_power:.0f} kW to {recommended_power:.0f} kW (+{max_shortfall*1.1:.0f} kW)")
        
        if low_soc_days > 0:
            current_capacity = sizing['capacity_kwh']
            recommended_capacity = current_capacity * 1.3  # 30% increase
            recommendations.append(f"⚡ **Increase Battery Capacity**: From {current_capacity:.0f} kWh to {recommended_capacity:.0f} kWh (+30%)")
        
        # Check for timing issues
        power_limited_days = sum((daily_analysis['Max_Battery_Power'] < sizing['power_rating_kw'] * 0.9) & (~daily_analysis['Success']))
        if power_limited_days > 0:
            recommendations.append(f"⏰ **Improve Charging Strategy**: {power_limited_days} days had insufficient charging before peak events")
        
        # Algorithm improvements
        if any(daily_analysis['Failure_Reason'] == 'Other (Algorithm/Timing)'):
            recommendations.append("🤖 **Optimize Battery Algorithm**: Consider predictive charging based on historical peak patterns")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Cost-benefit analysis for improvements
        if recommendations:
            st.markdown("##### 💰 **Investment Analysis for 100% Success Rate**")
            
            potential_additional_savings = failed_events['Peak_Shortfall'].sum() * md_rate_estimate
            st.write(f"**Additional Monthly MD Savings Potential**: RM {potential_additional_savings:.0f}")
            st.write(f"**Annual Additional Savings**: RM {potential_additional_savings * 12:.0f}")
            
            if max_shortfall > 0:
                additional_power_cost = max_shortfall * 1.1 * 400  # RM 400/kW for PCS
                additional_capacity_cost = sizing['capacity_kwh'] * 0.3 * 1200  # 30% more at RM 1200/kWh
                total_upgrade_cost = (additional_power_cost + additional_capacity_cost) * 1.4  # Installation factor
                
                payback_years = total_upgrade_cost / (potential_additional_savings * 12)
                
                st.write(f"**Estimated Upgrade Cost**: RM {total_upgrade_cost:,.0f}")
                st.write(f"**Simple Payback Period**: {payback_years:.1f} years")
    else:
        st.success("🎉 **Congratulations!** You've already achieved 100% success rate in peak shaving!")
    
    # Panel 5: Cumulative Energy Analysis - ALIGNED WITH DETAILED SUCCESS/FAILURE ANALYSIS
    st.markdown("##### 5️⃣ Cumulative Energy Discharged vs Required (MD Peak Periods Only)")
    st.markdown("**📊 Data Source:** Same as Detailed Success/Failure Analysis (daily aggregation during MD recording hours)")
    
    # Use the same daily analysis data that was calculated for the Detailed Success/Failure Analysis
    if len(daily_analysis) > 0:
        # Calculate energy requirements using the same daily aggregation approach
        daily_analysis_energy = daily_analysis.copy()
        
        # Energy Required: Calculate based on daily peak reduction needs during MD peak periods
        # This matches the approach used in the success/failure tables
        daily_analysis_energy['Daily_Energy_Required_kWh'] = 0.0
        
        # For each day, calculate energy required based on peak reduction needs
        for idx, row in daily_analysis_energy.iterrows():
            original_peak = row['Original_Peak_MD']
            net_peak = row['Net_Peak_MD']
            
            if original_peak > target_demand:
                # Calculate energy required to shave this day's peak to target
                if net_peak <= target_demand * 1.05:  # Successful day
                    # Energy that was successfully shaved (based on actual peak reduction)
                    energy_shaved = row['Peak_Reduction'] * 0.25  # Convert kW to kWh (15-min intervals)
                else:  # Failed day
                    # Energy that would be needed to reach target
                    energy_needed = (original_peak - target_demand) * 0.25
                    energy_shaved = energy_needed
                
                daily_analysis_energy.loc[idx, 'Daily_Energy_Required_kWh'] = energy_shaved
        
        # Calculate energy discharged from battery during MD peak periods for each day
        daily_analysis_energy['Daily_Energy_Discharged_kWh'] = 0.0
        
        # Group simulation data by date and sum battery discharge during MD peak periods
        df_sim_md_peak = df_sim[df_sim.index.to_series().apply(is_md_peak_period_for_effectiveness)]
        if len(df_sim_md_peak) > 0:
            daily_battery_discharge = df_sim_md_peak.groupby(df_sim_md_peak.index.date).agg({
                'Battery_Power_kW': lambda x: (x.clip(lower=0) * 0.25).sum()  # Only positive (discharge) * 15-min intervals
            }).reset_index()
            daily_battery_discharge.columns = ['Date', 'Daily_Battery_Discharge_kWh']
            
            # Merge with daily analysis
            daily_analysis_energy['Date'] = pd.to_datetime(daily_analysis_energy['Date'])
            daily_battery_discharge['Date'] = pd.to_datetime(daily_battery_discharge['Date'])
            daily_analysis_energy = daily_analysis_energy.merge(
                daily_battery_discharge, on='Date', how='left'
            ).fillna(0)
            
            daily_analysis_energy['Daily_Energy_Discharged_kWh'] = daily_analysis_energy['Daily_Battery_Discharge_kWh']
        else:
            st.warning("No MD peak period data available for energy analysis.")
            return
    
        # Sort by date for cumulative calculation
        daily_analysis_energy = daily_analysis_energy.sort_values('Date').reset_index(drop=True)
        
        # Calculate cumulative values
        daily_analysis_energy['Cumulative_Energy_Required'] = daily_analysis_energy['Daily_Energy_Required_kWh'].cumsum()
        daily_analysis_energy['Cumulative_Energy_Discharged'] = daily_analysis_energy['Daily_Energy_Discharged_kWh'].cumsum()
        daily_analysis_energy['Cumulative_Energy_Shortfall'] = daily_analysis_energy['Cumulative_Energy_Required'] - daily_analysis_energy['Cumulative_Energy_Discharged']
        
        # Create the chart using the daily aggregated data
        if len(daily_analysis_energy) > 0:
            fig5 = go.Figure()
            
            # Energy Discharged line (from daily analysis)
            fig5.add_trace(go.Scatter(
                x=daily_analysis_energy['Date'],
                y=daily_analysis_energy['Cumulative_Energy_Discharged'],
                mode='lines+markers',
                name='Cumulative Energy Discharged (MD Periods)',
                line=dict(color='blue', width=2),
                hovertemplate='Discharged: %{y:.1f} kWh<br>Date: %{x}<extra></extra>'
            ))
            
            # Energy Required line (from daily analysis)
            fig5.add_trace(go.Scatter(
                x=daily_analysis_energy['Date'],
                y=daily_analysis_energy['Cumulative_Energy_Required'],
                mode='lines+markers',
                name='Cumulative Energy Required (MD Periods)',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate='Required: %{y:.1f} kWh<br>Date: %{x}<extra></extra>'
            ))
            
            # Add area fill for energy shortfall
            fig5.add_trace(go.Scatter(
                x=daily_analysis_energy['Date'],
                y=daily_analysis_energy['Cumulative_Energy_Shortfall'].clip(lower=0),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='Cumulative Energy Shortfall',
                hovertemplate='Shortfall: %{y:.1f} kWh<br>Date: %{x}<extra></extra>'
            ))
            
            fig5.update_layout(
                title='📈 Cumulative Energy Analysis: Daily Aggregation (Same Source as Success/Failure Analysis)',
                xaxis_title='Date',
                yaxis_title='Cumulative Energy (kWh)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # Display metrics using daily aggregated data
            total_energy_required = daily_analysis_energy['Daily_Energy_Required_kWh'].sum()
            total_energy_discharged = daily_analysis_energy['Daily_Energy_Discharged_kWh'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Energy Required (MD Periods)", f"{total_energy_required:.1f} kWh")
            col2.metric("Total Energy Discharged (MD Periods)", f"{total_energy_discharged:.1f} kWh")
            
            if total_energy_required > 0:
                fulfillment_rate = (total_energy_discharged / total_energy_required) * 100
                col3.metric("MD Energy Fulfillment Rate", f"{fulfillment_rate:.1f}%")
            else:
                col3.metric("MD Energy Fulfillment Rate", "100%")
            
            # Add detailed breakdown table
            with st.expander("📊 Daily Energy Breakdown (Same Source as Success/Failure Analysis)"):
                display_columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Peak_Reduction', 
                                 'Daily_Energy_Required_kWh', 'Daily_Energy_Discharged_kWh', 'Success']
                
                if all(col in daily_analysis_energy.columns for col in display_columns):
                    daily_display = daily_analysis_energy[display_columns].copy()
                    daily_display.columns = ['Date', 'Original Peak (kW)', 'Net Peak (kW)', 'Peak Reduction (kW)',
                                           'Energy Required (kWh)', 'Energy Discharged (kWh)', 'Success']
                    
                    formatted_daily = daily_display.style.format({
                        'Original Peak (kW)': '{:.1f}',
                        'Net Peak (kW)': '{:.1f}',
                        'Peak Reduction (kW)': '{:.1f}',
                        'Energy Required (kWh)': '{:.2f}',
                        'Energy Discharged (kWh)': '{:.2f}'
                    })
                    
                    st.dataframe(formatted_daily, use_container_width=True)
                else:
                    st.warning("Some columns missing from daily analysis data.")
            
            # Add information box explaining the alignment
            st.info(f"""
            **📋 Data Source Alignment Confirmation:**
            - **Energy Required**: Calculated from daily peak reduction needs during MD recording hours (2-10 PM weekdays)
            - **Energy Discharged**: Sum of battery discharge energy during MD recording hours per day  
            - **Calculation Method**: Same daily aggregation approach as used in Detailed Success/Failure Analysis
            - **Target Demand**: {target_demand:.1f} kW (matches success/failure analysis)
            - **Total Days Analyzed**: {len(daily_analysis_energy)} days with MD peak period data
            - **Success Rate**: {(daily_analysis_energy['Success'].sum() / len(daily_analysis_energy) * 100):.1f}% (same as detailed analysis)
            
            ✅ **Consistency Check**: This chart now uses the same data source and methodology as the 🔍 Detailed Success/Failure Analysis tables.
            """)
            
        else:
            st.warning("No daily analysis data available for cumulative energy chart.")
    else:
        st.warning("No MD peak period data available for energy analysis.")
    
    # Original energy efficiency calculation for comparison (if needed)
    final_discharged = total_energy_discharged if 'total_energy_discharged' in locals() else 0
    final_required = total_energy_required if 'total_energy_required' in locals() else 0
    energy_efficiency = (final_discharged / final_required * 100) if final_required > 0 else 100
    
    # Key insights
    st.markdown("##### 🔍 Key Insights from Enhanced Analysis")
    
    insights = []
    
    # Use the new energy efficiency calculation
    if energy_efficiency < 80:
        insights.append("⚠️ **MD Energy Shortfall**: Battery capacity may be insufficient for complete MD peak shaving during 2-10 PM periods")
    elif energy_efficiency >= 95:
        insights.append("✅ **Excellent MD Coverage**: Battery effectively handles all MD peak period energy requirements")
    
    # Check if success intervals are available
    if 'success_intervals' in locals() and 'total_peak_intervals' in locals() and total_peak_intervals > 0:
        success_rate = success_intervals / total_peak_intervals
        if success_rate > 0.9:
            insights.append("✅ **High MD Success Rate**: Battery effectively manages most peak events during MD recording hours")
        elif success_rate < 0.6:
            insights.append("❌ **Low MD Success Rate**: Consider increasing battery power rating or capacity for better MD management")
    
    # Check battery utilization if heatmap data is available
    if 'df_heatmap' in locals() and len(df_heatmap) > 0:
        avg_utilization = df_heatmap['Battery_Utilization_%'].mean()
        if avg_utilization < 30:
            insights.append("📊 **Under-utilized**: Battery power rating may be oversized")
        elif avg_utilization > 80:
            insights.append("🔥 **High Utilization**: Battery operating near maximum capacity")
    
    # Check for low SOC events
    low_soc_events = len(df_sim[df_sim['Battery_SOC_Percent'] < 20])
    if low_soc_events > 0:
        insights.append(f"🔋 **Low SOC Warning**: {low_soc_events} intervals with SOC below 10%")
    
    # Add insight about data source alignment
    if len(daily_analysis) > 0:
        insights.append(f"📊 **Data Consistency**: Chart 5️⃣ now uses the same daily aggregation methodology as the Success/Failure Analysis ({len(daily_analysis)} days analyzed)")
    
    if not insights:
        insights.append("✅ **Optimal Performance**: Battery system operating within acceptable parameters")
    
    for insight in insights:
        st.info(insight)


def get_tnb_holidays_2024_2025():
    """Get TNB's 15 official public holidays for 2024-2025."""
    import datetime
    
    holidays_2024 = [
        datetime.date(2024, 1, 1),   # New Year
        datetime.date(2024, 2, 10),  # CNY Day 1
        datetime.date(2024, 2, 11),  # CNY Day 2
        datetime.date(2024, 4, 10),  # Aidil Fitri 1
        datetime.date(2024, 4, 11),  # Aidil Fitri 2
        datetime.date(2024, 5, 1),   # Labour Day
        datetime.date(2024, 5, 22),  # Vesak Day
        datetime.date(2024, 6, 3),   # Agong Birthday
        datetime.date(2024, 6, 17),  # Aidil Adha
        datetime.date(2024, 7, 7),   # Awal Muharram
        datetime.date(2024, 8, 31),  # National Day
        datetime.date(2024, 9, 15),  # Maulidur Rasul
        datetime.date(2024, 9, 16),  # Malaysia Day
        datetime.date(2024, 11, 1),  # Deepavali
        datetime.date(2024, 12, 25), # Christmas
    ]
    
    holidays_2025 = [
        datetime.date(2025, 1, 1),   # New Year
        datetime.date(2025, 1, 29),  # CNY Day 1
        datetime.date(2025, 1, 30),  # CNY Day 2
        datetime.date(2025, 3, 31),  # Aidil Fitri 1 (est)
        datetime.date(2025, 4, 1),   # Aidil Fitri 2 (est)
        datetime.date(2025, 5, 1),   # Labour Day
        datetime.date(2025, 5, 12),  # Vesak Day (est)
        datetime.date(2025, 6, 2),   # Agong Birthday (est)
        datetime.date(2025, 6, 7),   # Aidil Adha (est)
        datetime.date(2025, 6, 27),  # Awal Muharram (est)
        datetime.date(2025, 8, 31),  # National Day
        datetime.date(2025, 9, 5),   # Maulidur Rasul (est)
        datetime.date(2025, 9, 16),  # Malaysia Day
        datetime.date(2025, 10, 20), # Deepavali (est)
        datetime.date(2025, 12, 25), # Christmas
    ]
    
    return set(holidays_2024 + holidays_2025)


def get_tariff_period_classification(timestamp, selected_tariff, holidays=None):
    """
    Get period classification (Peak/Off-Peak) based on selected tariff configuration.
    Returns 'Peak' or 'Off-Peak' based on tariff-specific time bands.
    """
    # Handle holidays first
    if holidays and timestamp.date() in holidays:
        # For holidays, behavior depends on tariff type
        tariff_type = selected_tariff.get('Type', '').lower()
        tariff_name = selected_tariff.get('Tariff', '').lower()
        is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
        
        if is_tou_tariff:
            return 'Off-Peak'  # TOU tariffs have off-peak rates on holidays
        else:
            return 'Peak'  # General tariffs: always Peak (MD charges apply)
    
    # Get tariff type
    tariff_name = selected_tariff.get('Tariff', '')
    tariff_type = selected_tariff.get('Type', '').lower()
    
    # Check if it's a TOU (Time of Use) tariff
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name.lower()
    
    if is_tou_tariff:
        # For TOU tariffs, use time-based classification (peak vs off-peak rates)
        return _classify_tou_tariff_periods(timestamp)
    else:
        # For General tariffs, there's no peak/off-peak pricing
        # MD charges apply 24/7, so everything is "Peak" for visualization purposes
        return _classify_general_tariff_periods(timestamp)


def _classify_tou_tariff_periods(timestamp):
    """
    TOU Tariff Logic: Based on electricity pricing periods
    - Peak = High electricity rate periods (2PM-10PM weekdays)
    - Off-Peak = Low electricity rate periods (all other times)
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    # TOU Peak Hours: When electricity rates are highest - IMPROVED HIERARCHY APPLIED
    # Standard RP4 TOU: 2PM-10PM weekdays (holidays already handled by parent function)
    # Hierarchy: Holiday Check (✅ done by parent) → Weekday Check → Hour Check
    if weekday < 5 and 14 <= hour < 22:
        return 'Peak'  # High energy rate + MD recording
    else:
        return 'Off-Peak'  # Low energy rate


def _classify_general_tariff_periods(timestamp):
    """
    General Tariff Logic: No time-based pricing, but MD still applies
    - Peak = Always (MD charges apply 24/7 on flat rate)
    - Off-Peak = Never (no concept of off-peak in general tariffs)
    
    For visualization: everything above target should be red since MD charges
    apply regardless of time under flat rate pricing.
    """
    # For General tariffs, there's no time-based pricing distinction
    # MD charges apply whenever demand exceeds previous maximum
    # So for visualization purposes, everything is "Peak" 
    return 'Peak'


def _get_tariff_description(selected_tariff):
    """
    Get a descriptive text for the tariff peak period.
    """
    if not selected_tariff:
        return "Peak Period"
    
    tariff_name = selected_tariff.get('Tariff', '')
    tariff_type = selected_tariff.get('Type', '').lower()
    
    if tariff_type == 'tou' or 'tou' in tariff_name.lower():
        return "TOU Peak Rate Period (2PM-10PM weekdays)"
    else:
        return "General Tariff - MD Applies 24/7"
