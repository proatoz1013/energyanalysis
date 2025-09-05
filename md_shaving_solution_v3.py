"""
Author: Advanced MD Shaving Team
Version: 3.0
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Import V1 components for reuse
from md_shaving_solution import (
    read_uploaded_file,
    _configure_data_inputs,
    _process_dataframe,
    _configure_tariff_selection,
    _detect_peak_events,
    _display_battery_simulation_chart,
    _simulate_battery_operation,
    create_conditional_demand_line_with_peak_logic,
    _display_peak_event_results,
    get_tariff_period_classification
)
from tariffs.peak_logic import is_peak_rp4, get_period_classification

# ============================================================================
# TARIFF-AWARE SCAFFOLDING AND LOGIC INFRASTRUCTURE
# ============================================================================

def get_tariff_period_classification_v3(timestamp, selected_tariff, holidays=None):
    """
    V3 Enhanced tariff period classification with detailed logic.
    Returns 'Peak' or 'Off-Peak' based on tariff configuration.
    """
    # Handle holidays first
    if holidays and timestamp.date() in holidays:
        tariff_type = selected_tariff.get('Type', '').lower()
        tariff_name = selected_tariff.get('Tariff', '').lower()
        is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
        
        if is_tou_tariff:
            return 'Off-Peak'  # TOU tariffs have off-peak rates on holidays
        else:
            return 'Peak'  # General tariffs: always Peak (MD charges apply)
    
    # Get tariff configuration
    tariff_name = selected_tariff.get('Tariff', '')
    tariff_type = selected_tariff.get('Type', '').lower()
    voltage_level = selected_tariff.get('Voltage', '').lower()
    
    # Determine if it's a TOU tariff
    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name.lower()
    
    if is_tou_tariff:
        return _classify_tou_periods_v3(timestamp, voltage_level)
    else:
        return _classify_general_periods_v3(timestamp)

def _classify_tou_periods_v3(timestamp, voltage_level):
    """
    Enhanced TOU period classification for different voltage levels.
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    # Standard TOU Peak Hours: 2PM-10PM weekdays for all voltage levels
    if weekday < 5 and 14 <= hour < 22:
        return 'Peak'
    else:
        return 'Off-Peak'

def _classify_general_periods_v3(timestamp):
    """
    General tariff classification - MD applies 24/7.
    """
    return 'Peak'  # MD charges apply regardless of time

def get_tariff_md_rate_v3(selected_tariff):
    """
    Get MD rate (RM/kW/month) based on selected tariff.
    Returns the appropriate MD charge rate for cost calculations.
    """
    if not selected_tariff:
        return 35.0  # Default fallback rate
    
    tariff_name = selected_tariff.get('Tariff', '').upper()
    voltage_level = selected_tariff.get('Voltage', '').lower()
    tariff_type = selected_tariff.get('Type', '').lower()
    
    # MD rates based on tariff structure (RM/kW/month)
    md_rates = {
        # Low Voltage
        'C2-L TOU': 35.0,
        'C2-L GENERAL': 35.0,
        
        # Medium Voltage  
        'C2-M TOU': 97.06,
        'C2-M GENERAL': 97.06,
        
        # High Voltage
        'C1-H TOU': 143.54,
        'C1-H GENERAL': 143.54,
    }
    
    # Try to match exact tariff name first
    if tariff_name in md_rates:
        return md_rates[tariff_name]
    
    # Fallback to voltage-based matching
    if 'medium' in voltage_level:
        return 97.06
    elif 'high' in voltage_level:
        return 143.54
    else:  # Low voltage or unknown
        return 35.0

def get_tariff_description_v3(selected_tariff):
    """
    Get enhanced tariff description for V3 analysis.
    """
    if not selected_tariff:
        return "Unknown Tariff"
    
    tariff_name = selected_tariff.get('Tariff', 'Unknown')
    tariff_type = selected_tariff.get('Type', '').lower()
    voltage_level = selected_tariff.get('Voltage', '')
    
    if tariff_type == 'tou' or 'tou' in tariff_name.lower():
        return f"{tariff_name} - Peak: 2PM-10PM weekdays, Off-Peak: all other times"
    else:
        return f"{tariff_name} - Flat rate, MD charges apply 24/7"

def analyze_tariff_impact_v3(df_processed, power_col, target_kw, selected_tariff, holidays=None):
    """
    Analyze the impact of tariff structure on peak events and costs.
    This is scaffolding for future detailed analysis.
    """
    # Initialize analysis structure
    analysis = {
        'tariff_config': {
            'name': selected_tariff.get('Tariff', 'Unknown') if selected_tariff else 'Unknown',
            'type': selected_tariff.get('Type', 'Unknown') if selected_tariff else 'Unknown',
            'voltage': selected_tariff.get('Voltage', 'Unknown') if selected_tariff else 'Unknown',
            'md_rate': get_tariff_md_rate_v3(selected_tariff)
        },
        'peak_events': {
            'total_events': 0,
            'peak_period_events': 0,
            'off_peak_period_events': 0,
            'events_list': []
        },
        'cost_analysis': {
            'max_monthly_md_cost': 0,
            'potential_savings': 0,
            'events_by_period': {}
        },
        'timing_analysis': {
            'peak_hours_distribution': {},
            'weekday_distribution': {},
            'monthly_distribution': {}
        }
    }
    
    # Detect all peak events above target
    peak_events = df_processed[df_processed[power_col] > target_kw]
    analysis['peak_events']['total_events'] = len(peak_events)
    
    if not peak_events.empty and selected_tariff:
        # Classify each peak event by tariff period
        for timestamp, row in peak_events.iterrows():
            period = get_tariff_period_classification_v3(timestamp, selected_tariff, holidays)
            excess_kw = row[power_col] - target_kw
            
            event_data = {
                'timestamp': timestamp,
                'power_kw': row[power_col],
                'excess_kw': excess_kw,
                'period': period,
                'hour': timestamp.hour,
                'weekday': timestamp.weekday(),
                'month': timestamp.month
            }
            
            analysis['peak_events']['events_list'].append(event_data)
            
            # Count by period
            if period == 'Peak':
                analysis['peak_events']['peak_period_events'] += 1
            else:
                analysis['peak_events']['off_peak_period_events'] += 1
        
        # Calculate maximum monthly MD impact
        max_excess = peak_events[power_col].max() - target_kw
        if max_excess > 0:
            analysis['cost_analysis']['max_monthly_md_cost'] = max_excess * analysis['tariff_config']['md_rate']
            analysis['cost_analysis']['potential_savings'] = analysis['cost_analysis']['max_monthly_md_cost']
    
    return analysis

def _render_v3_analysis_infrastructure(df_processed, power_col, target_kw, selected_tariff, holidays, 
                                     overall_max_demand, avg_demand, load_factor, user_shave_pct=10):
    """
    Main V3 analysis infrastructure that coordinates all tariff-aware analysis.
    This sets up the scaffolding for future feature implementation.
    """
    
    # Perform tariff-aware analysis
    if selected_tariff:
        tariff_analysis = analyze_tariff_impact_v3(df_processed, power_col, target_kw, selected_tariff, holidays)
        
        # Display tariff configuration summary
        st.subheader("⚡ Tariff Analysis Summary")
        _display_tariff_summary_v3(tariff_analysis['tariff_config'])
        
        # Display enhanced peak events chart with colour-coded line and detailed table
        _display_v3_peak_events_chart(df_processed, power_col, target_kw, selected_tariff, holidays, tariff_analysis, user_shave_pct)
        
    else:
        st.warning("⚠️ Tariff not configured. Using simplified analysis.")
        _display_simplified_analysis_v3(df_processed, power_col, target_kw)

def _display_tariff_summary_v3(tariff_config):
    """
    Display tariff configuration summary with MD rates.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tariff", tariff_config['name'])
    with col2:
        st.metric("Type", tariff_config['type'].upper())
    with col3:
        st.metric("Voltage Level", tariff_config['voltage'])
    with col4:
        st.metric("MD Rate", f"RM {tariff_config['md_rate']:.2f}/kW/month")
    
    # Placeholder for future tariff details visualization
    st.info("📋 **Tariff Structure:** Peak period classification and MD rates configured for analysis")

def _display_peak_events_summary_v3(peak_events_data, target_kw):
    """
    Display peak events summary with tariff-aware breakdown.
    """
    total_events = peak_events_data['total_events']
    peak_period_events = peak_events_data['peak_period_events']
    off_peak_events = peak_events_data['off_peak_period_events']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Peak Events", f"{total_events:,}")
        st.caption(f"Above {target_kw:.1f} kW target")
    with col2:
        st.metric("Peak Period Events", f"{peak_period_events:,}")
        st.caption("During tariff peak hours")
    with col3:
        st.metric("Off-Peak Period Events", f"{off_peak_events:,}")
        st.caption("During off-peak hours")
    
    # Placeholder for future detailed peak events visualization
    if total_events > 0:
        peak_percentage = (peak_period_events / total_events) * 100
        st.info(f"📊 **Peak Period Impact:** {peak_percentage:.1f}% of events occur during peak tariff periods")
    else:
        st.success("✅ No peak events detected above target level!")

def _display_cost_analysis_summary_v3(cost_analysis):
    """
    Display cost analysis summary with MD impact calculations.
    """
    max_monthly_cost = cost_analysis['max_monthly_md_cost']
    potential_savings = cost_analysis['potential_savings']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Max Monthly MD Cost", f"RM {max_monthly_cost:,.0f}")
        st.caption("Maximum potential MD impact")
    with col2:
        st.metric("Annual Cost Potential", f"RM {potential_savings * 12:,.0f}")
        st.caption("If current peak is maintained")
    
    # Placeholder for future detailed cost breakdown
    if max_monthly_cost > 0:
        st.info("💡 **Cost Analysis:** MD costs calculated based on tariff-specific rates and peak event timing")
    else:
        st.success("✅ No additional MD costs above target level!")

def _display_battery_sizing_scaffolding_v3(tariff_analysis, target_kw, overall_max_demand):
    """
    Display battery sizing scaffolding with tariff-aware considerations.
    """
    peak_events = tariff_analysis['peak_events']['events_list']
    md_rate = tariff_analysis['tariff_config']['md_rate']
    
    if peak_events:
        # Calculate basic sizing requirements
        max_excess_power = max(event['excess_kw'] for event in peak_events)
        total_excess_energy = sum(event['excess_kw'] * 0.25 for event in peak_events)  # 15-min intervals
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recommended Power", f"{max_excess_power * 1.1:.1f} kW")
            st.caption("Max excess + 10% safety margin")
        with col2:
            st.metric("Recommended Capacity", f"{total_excess_energy * 1.2:.1f} kWh")
            st.caption("Total excess energy + 20% safety")
        
        # Cost-benefit preview
        potential_monthly_savings = max_excess_power * md_rate
        st.info(f"💰 **Potential Monthly Savings:** RM {potential_monthly_savings:,.0f} (based on max excess power reduction)")
        
        # Placeholder for future detailed battery analysis
        st.info("🔋 **Battery Analysis:** Tariff-aware sizing considers peak timing and MD rate impacts")
    else:
        st.success("✅ No battery required - demand already within target!")

def _calculate_v3_monthly_targets(df, power_col, base_target_kw, selected_tariff, holidays, user_shave_pct=10):
    """
    Calculate monthly targets for V3 stepped target line.
    Creates month-specific targets based on monthly peak patterns and user shaving percentage.
    Formula: monthly_target = (1 - user_shave_pct/100) × monthly_peak
    
    Args:
        df: DataFrame with power consumption data
        power_col: Name of power consumption column
        base_target_kw: Fallback target (used only in extreme fallback scenarios)
        selected_tariff: Tariff configuration dictionary
        holidays: List of holiday dates
        user_shave_pct: User-defined shaving percentage (e.g., 10 for 10% shaving)
        
    Returns:
        Dict mapping Period('M') to monthly target values
    """
    monthly_targets = {}
    fallback_logs = []
    
    try:
        if df.empty:
            return {}
            
        # Ensure timezone alignment with df.index
        df_tz = df.copy()
        if df.index.tz is None:
            df_tz.index = pd.to_datetime(df.index)
        
        # Get all months between min and max dates
        min_date = df_tz.index.min()
        max_date = df_tz.index.max()
        all_months = pd.period_range(
            start=min_date.to_period('M'),
            end=max_date.to_period('M'),
            freq='M'
        )
        
        # Group data by month
        df_monthly = df_tz.groupby(df_tz.index.to_period('M'))
        
        # Calculate global 24/7 peak for ultimate fallback
        global_24_7_peak = df_tz[power_col].max()
        global_fallback_target = global_24_7_peak * (1 - user_shave_pct/100)
        
        # Calculate targets for each month
        for month_period in all_months:
            if month_period in df_monthly.groups:
                month_data = df_monthly.get_group(month_period)
                
                # Build the monthly reference slice based on tariff type
                monthly_peak = None
                if selected_tariff:
                    tariff_type = selected_tariff.get('Type', '').lower()
                    tariff_name = selected_tariff.get('Tariff', '').lower()
                    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
                    
                    if is_tou_tariff:
                        # For TOU tariffs, use peak during peak periods (2PM-10PM weekdays)
                        monthly_peak = _calculate_monthly_tou_peak(month_data, power_col, holidays)
                else:
                    # For General/no tariff, use 24/7 monthly peak
                    monthly_peak = month_data[power_col].max()
                    
                # Apply fallback chain if monthly_peak is None or NaN
                if monthly_peak is None or pd.isna(monthly_peak) or not np.isfinite(monthly_peak):
                    # Fallback A: Use same month's 24/7 peak
                    monthly_24_7_peak = month_data[power_col].max()
                    if not pd.isna(monthly_24_7_peak) and np.isfinite(monthly_24_7_peak):
                        monthly_peak = monthly_24_7_peak
                        fallback_logs.append(f"{month_period}: Fallback A - 24/7 peak")
                    else:
                        # Fallback B: Carry nearest neighbor month's target (handled below after all months calculated)
                        monthly_peak = None
                        fallback_logs.append(f"{month_period}: Fallback B - neighbor needed")
                
                # Calculate monthly target if we have a valid peak
                if monthly_peak is not None and np.isfinite(monthly_peak):
                    monthly_target = monthly_peak * (1 - user_shave_pct/100)
                    monthly_targets[month_period] = monthly_target
                else:
                    # Mark for neighbor fallback
                    monthly_targets[month_period] = None
                    
            else:
                # Month has no data - skip this month
                monthly_targets[month_period] = None
                fallback_logs.append(f"{month_period}: No data - skipping month")
        
        # Handle Fallback A only: Skip Fallback B (neighbor) and C (global)
        # Simply ignore NaN/None months instead of trying to fill them
        final_monthly_targets = {}
        
        for month_period in all_months:
            if monthly_targets.get(month_period) is not None:
                target_value = monthly_targets[month_period]
                if not pd.isna(target_value) and np.isfinite(target_value):
                    final_monthly_targets[month_period] = target_value
                    continue
            
            # If we reach here, the month has NaN/None/invalid target
            # Skip this month entirely instead of using neighbor/global fallbacks
            fallback_logs.append(f"{month_period}: Skipped - no valid monthly data (ignoring NaN)")
        
        # Update monthly_targets to only contain valid months
        monthly_targets = final_monthly_targets
        
        # Integrity checks and logging
        expected_count = len(all_months)
        actual_count = len(monthly_targets)
        skipped_count = expected_count - actual_count
        
        print(f"✅ Month verification: {expected_count} months in data, {actual_count} targets calculated, {skipped_count} skipped")
        print(f"✅ NaN check: {actual_count} targets, 0 NaNs (NaN months ignored)")
        
        if fallback_logs:
            print(f"📋 Fallback usage: {', '.join(fallback_logs)}")
        
        return monthly_targets
        
    except Exception as e:
        print(f"❌ Error in monthly targets calculation: {str(e)}")
        # Return global fallback for all months if calculation fails
        if not df.empty:
            min_date = df.index.min()
            max_date = df.index.max()
            all_months = pd.period_range(
                start=min_date.to_period('M'),
                end=max_date.to_period('M'),
                freq='M'
            )
            return {month_period: global_fallback_target for month_period in all_months}
        return {}

def _calculate_monthly_tou_peak(month_data, power_col, holidays):
    """
    Calculate peak demand during TOU peak periods (2PM-10PM weekdays) for a given month.
    Returns None if TOU slice is empty to let caller pick fallback.
    """
    tou_peak_data = []
    
    for timestamp, row in month_data.iterrows():
        # Check if this timestamp is during TOU peak period
        is_weekday = timestamp.weekday() < 5  # Monday = 0, Sunday = 6
        is_peak_hour = 14 <= timestamp.hour < 22  # 2PM to 10PM
        is_holiday = holidays and timestamp.date() in holidays
        
        # TOU peak period: 2PM-10PM weekdays (excluding holidays)
        if is_weekday and is_peak_hour and not is_holiday:
            tou_peak_data.append(row[power_col])
    
    return max(tou_peak_data) if tou_peak_data else None

def _create_per_timestamp_target_series(df, monthly_targets):
    """
    Create a pandas Series with per-timestamp targets aligned to df.index.
    No NaNs - ensures every timestamp has a target value.
    """
    if not monthly_targets or df.empty:
        return None
    
    # Create target series aligned to df index
    target_series = pd.Series(index=df.index, dtype=float)
    
    # Map each timestamp to its monthly target
    for timestamp in df.index:
        month_period = timestamp.to_period('M')
        
        # Find target for this month (handle missing months with fallback)
        if month_period in monthly_targets:
            target_series.loc[timestamp] = monthly_targets[month_period]
        else:
            # Fallback to nearest available monthly target
            available_periods = list(monthly_targets.keys())
            if available_periods:
                # Use the closest month's target
                closest_period = min(available_periods, key=lambda x: abs((x.start_time - timestamp).days))
                target_series.loc[timestamp] = monthly_targets[closest_period]
            else:
                # Ultimate fallback - should not happen with fixed _calculate_v3_monthly_targets
                target_series.loc[timestamp] = 1000.0  # Safe default
    
    # Ensure no NaNs by forward-filling any missing values
    target_series = target_series.ffill().bfill()
    
    # Final safety check - if still NaN, use the first available monthly target
    if target_series.isna().any():
        default_target = list(monthly_targets.values())[0] if monthly_targets else 1000.0
        target_series = target_series.fillna(default_target)
    
    return target_series

def create_conditional_demand_line_with_stepped_targets(fig, df, power_col, target_demand, selected_tariff=None, holidays=None, trace_name="Power Consumption"):
    """
    Enhanced version of V1 function that accepts either scalar or Series targets.
    Creates continuous line segments with different colors based on per-timestamp targets.
    
    Args:
        target_demand: Either a scalar value or a pandas Series aligned to df.index
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
    
    # Check if target_demand is a Series or scalar
    is_target_series = isinstance(target_demand, pd.Series)
    
    for i in range(len(df_copy)):
        timestamp = df_copy.index[i]
        demand_value = df_copy.iloc[i][power_col]
        
        # Get the appropriate target for this timestamp
        if is_target_series:
            current_target = target_demand.loc[timestamp] if timestamp in target_demand.index else target_demand.iloc[0]
        else:
            current_target = target_demand
        
        # Get peak period classification based on selected tariff
        if selected_tariff:
            period_type = get_tariff_period_classification(timestamp, selected_tariff, holidays)
        else:
            # Fallback to default RP4 logic
            period_type = get_period_classification(timestamp, holidays)
        
        if demand_value > current_target:
            if period_type == 'Peak':
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'red'
            else:
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'green'
        else:
            df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'blue'
    
    # Create a single continuous line with color-coded segments
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
        
        # Color mapping
        color_map = {'red': 'red', 'green': 'green', 'blue': 'blue'}
        
        # Add the trace
        fig.add_trace(go.Scatter(
            x=segment_x,
            y=segment_y,
            mode='lines',
            line=dict(color=color_map[current_color], width=2),
            name=segment_name,
            showlegend=show_legend,
            legendgroup=current_color,
            hovertemplate=f'{hover_info}<br>Time: %{{x}}<br>Power: %{{y:.1f}} kW<extra></extra>',
        ))
        
        i = j
    
    return fig

def _get_tariff_description(selected_tariff):
    """Helper function to get tariff description for legend."""
    if not selected_tariff:
        return "RP4 Peak Period"
    
    tariff_type = selected_tariff.get('Type', '').lower()
    tariff_name = selected_tariff.get('Tariff', '').lower()
    
    if tariff_type == 'tou' or 'tou' in tariff_name:
        return "TOU Peak Period"
    else:
        return "General Tariff Peak"

def _add_v3_stepped_target_line(fig, df, monthly_targets, selected_tariff):
    """
    Add truly stepped monthly target line to the figure.
    Each month shows as a distinct horizontal step, with gaps allowed for missing months.
    """
    if not monthly_targets or df.empty:
        return
    
    # Create stepped target line for visualization
    target_line_data = []
    target_line_timestamps = []
    
    # Sort monthly targets by period to ensure chronological order
    sorted_months = sorted(monthly_targets.keys())
    
    # Get overall data time range
    data_start = df.index.min()
    data_end = df.index.max()
    
    # Get all possible months in the data range for gap detection
    all_possible_months = pd.period_range(
        start=data_start.to_period('M'),
        end=data_end.to_period('M'),
        freq='M'
    )
    
    # Create truly stepped line - each month gets its own horizontal segment
    # Allow gaps for months that don't have valid targets
    for i, month_period in enumerate(sorted_months):
        target_value = monthly_targets[month_period]
        
        # Determine month boundaries within the data range
        month_start = max(month_period.start_time, data_start)
        month_end = min(month_period.end_time, data_end)
        
        # Only add points if this month intersects with our data range
        if month_start <= data_end and month_end >= data_start:
            # If this is not the first valid month and there's a gap, add None to create a line break
            if i > 0:
                prev_month = sorted_months[i-1]
                # Check if there are missing months between prev_month and current month_period
                months_between = pd.period_range(start=prev_month, end=month_period, freq='M')
                if len(months_between) > 2:  # More than just prev and current month
                    # There's a gap - add None values to break the line
                    target_line_timestamps.append(month_start)
                    target_line_data.append(None)
            
            # Add step: horizontal line for this month's target
            target_line_timestamps.append(month_start)
            target_line_data.append(target_value)
            target_line_timestamps.append(month_end)
            target_line_data.append(target_value)
            
            # Debug logging for April specifically
            if month_period.month == 4:  # April
                print(f"🔍 April stepped line debug: {month_period} = {target_value:.1f} kW, range: {month_start} to {month_end}")
    
    # Add stepped monthly target line
    if target_line_data and target_line_timestamps:
        # Determine tariff description for legend
        tariff_description = "General Tariff"
        if selected_tariff:
            tariff_type = selected_tariff.get('Type', '').lower()
            tariff_name = selected_tariff.get('Tariff', '').lower()
            if tariff_type == 'tou' or 'tou' in tariff_name:
                tariff_description = "TOU Tariff"
        
        legend_label = f"Monthly Target - {tariff_description}"
        
        fig.add_trace(go.Scatter(
            x=target_line_timestamps,
            y=target_line_data,
            mode='lines',
            name=legend_label,
            line=dict(color='red', width=2, dash='dash'),
            opacity=0.9,
            showlegend=True,
            connectgaps=False  # Don't connect across None values (gaps)
        ))
        
        valid_months = len([v for v in target_line_data if v is not None])
        print(f"✅ Stepped line added: {len(target_line_timestamps)} points covering {valid_months} valid months (gaps allowed)")

def _display_v3_monthly_targets_summary(monthly_targets, selected_tariff):
    """
    Display summary of monthly targets calculation for V3.
    """
    st.markdown("#### 📋 Monthly Target Calculation Summary")
    
    # Determine tariff type for description
    tariff_type = "General Tariff"
    calculation_method = "24/7 Peak"
    if selected_tariff:
        tariff_type_raw = selected_tariff.get('Type', '').lower()
        tariff_name = selected_tariff.get('Tariff', '').lower()
        if tariff_type_raw == 'tou' or 'tou' in tariff_name:
            tariff_type = "TOU Tariff"
            calculation_method = "TOU Peak Period (2PM-10PM weekdays)"
    
    # Create summary table
    summary_data = []
    for month_period, target_value in monthly_targets.items():
        summary_data.append({
            'Month': str(month_period),
            'Target MD': f"{target_value:.1f} kW",
            'Calculation Method': calculation_method,
            'Tariff Type': tariff_type
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Display info about calculation
    total_months = len(monthly_targets)
    min_target = min(monthly_targets.values())
    max_target = max(monthly_targets.values())
    avg_target = sum(monthly_targets.values()) / len(monthly_targets)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Months Processed", total_months)
    with col2:
        st.metric("Target Range", f"{min_target:.1f} - {max_target:.1f} kW")
    with col3:
        st.metric("Average Target", f"{avg_target:.1f} kW")
    
    # Show detailed monthly breakdown
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.info(f"""
    **🎯 Target Calculation Method:**
    - **Tariff Type**: {tariff_type}
    - **Peak Reference**: {calculation_method}
    - **Shaving Percentage**: 10% (target = 90% of monthly peak)
    - **Stepped Line**: Each month uses its specific calculated target
    """)

def _display_v3_peak_events_chart(df_processed, power_col, target_kw, selected_tariff, holidays, tariff_analysis, user_shave_pct=10):
    """
    V3 Enhanced peak events visualization with tariff-aware coloring and analysis.
    """
    st.markdown("### 📈 Power Consumption with Peak Events Highlighted")
    
    # Get peak events from analysis
    peak_events_list = tariff_analysis['peak_events']['events_list']
    
    # Create main figure
    fig = go.Figure()
    
    # Calculate monthly targets for stepped target line
    monthly_targets = _calculate_v3_monthly_targets(df_processed, power_col, target_kw, selected_tariff, holidays, user_shave_pct)
    
    # Create per-timestamp target series for enhanced coloring
    target_series = _create_per_timestamp_target_series(df_processed, monthly_targets)
    
    # Display monthly targets summary if available
    if monthly_targets and len(monthly_targets) > 1:
        _display_v3_monthly_targets_summary(monthly_targets, selected_tariff)
        
        # Log verification for acceptance checks
        unique_months_in_data = len(df_processed.index.to_period('M').unique())
        unique_months_in_targets = len(monthly_targets)
        print(f"✅ Month verification: {unique_months_in_data} months in data, {unique_months_in_targets} targets calculated")
        
        if target_series is not None:
            nan_count = target_series.isna().sum()
            print(f"✅ Target series verification: {len(target_series)} values, {nan_count} NaNs")
    
    # Add stepped monthly target line first (before power consumption line)
    if monthly_targets:
        _add_v3_stepped_target_line(fig, df_processed, monthly_targets, selected_tariff)
    else:
        # Fallback to simple horizontal line if monthly calculation fails
        fig.add_hline(
            y=target_kw,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {target_kw:.1f} kW",
            annotation_position="top right"
        )
    
    # Create color-coded continuous line using enhanced function with per-timestamp targets
    if target_series is not None:
        fig = create_conditional_demand_line_with_stepped_targets(
            fig, df_processed, power_col, target_series, selected_tariff, holidays, "Power Consumption"
        )
    else:
        # Fallback to original V1 function with scalar target
        fig = create_conditional_demand_line_with_peak_logic(
            fig, df_processed, power_col, target_kw, selected_tariff, holidays, "Power Consumption"
        )
    
    # Highlight individual peak events with filled areas using enhanced function
    if peak_events_list:
        if target_series is not None:
            _add_v3_peak_event_highlights(fig, df_processed, power_col, target_series, peak_events_list)
        else:
            _add_v3_peak_event_highlights(fig, df_processed, power_col, target_kw, peak_events_list)
    
    # Update layout with V3 styling
    fig.update_layout(
        title="Power Consumption with Monthly Peak Events Highlighted",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional analysis if peak events exist
    if peak_events_list:
        _display_v3_peak_events_analysis_summary(peak_events_list, tariff_analysis)
        
        # Display detailed peak events table using V1 reused function
        _display_v3_peak_events_table(df_processed, power_col, target_kw, selected_tariff, holidays)

def _create_v3_conditional_demand_line(fig, df, power_col, monthly_targets, selected_tariff, holidays, trace_name):
    """
    V3 conditional coloring logic - exact copy from V2 with V3 tariff classification.
    Creates continuous line segments with different colors based on monthly targets.
    """
    # Process data chronologically to create continuous segments
    all_timestamps = sorted(df.index)
    
    # Create segments for continuous colored lines
    segments = []
    current_segment = {'type': None, 'x': [], 'y': []}
    
    for timestamp in all_timestamps:
        power_value = df.loc[timestamp, power_col]
        
        # Get the monthly target for this timestamp (V2 logic)
        month_period = timestamp.to_period('M')
        if month_period in monthly_targets:
            target_value = monthly_targets[month_period]
            
            # Determine the color category for this point
            if power_value <= target_value:
                segment_type = 'below_target'
            else:
                # Use V3 tariff classification instead of V2's is_peak_rp4
                if selected_tariff:
                    period_type = get_tariff_period_classification_v3(timestamp, selected_tariff, holidays)
                else:
                    # Fallback to V2's original logic for compatibility
                    is_peak = is_peak_rp4(timestamp, holidays if holidays else set())
                    period_type = 'Peak' if is_peak else 'Off-Peak'
                
                if period_type == 'Peak':
                    segment_type = 'above_target_peak'
                else:
                    segment_type = 'above_target_offpeak'
            
            # If this is the start or the segment type changed, finalize previous and start new
            if current_segment['type'] != segment_type:
                # Finalize the previous segment if it has data
                if current_segment['type'] is not None and len(current_segment['x']) > 0:
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    'type': segment_type, 
                    'x': [timestamp], 
                    'y': [power_value]
                }
            else:
                # Continue current segment
                current_segment['x'].append(timestamp)
                current_segment['y'].append(power_value)
    
    # Don't forget the last segment
    if current_segment['type'] is not None and len(current_segment['x']) > 0:
        segments.append(current_segment)
    
    # Plot the colored segments with proper continuity (exact V2 approach)
    color_map = {
        'below_target': {'color': 'blue', 'name': f'{trace_name} (Below Monthly Target)'},
        'above_target_offpeak': {'color': 'green', 'name': f'{trace_name} (Above Monthly Target - Off-Peak Period)'},
        'above_target_peak': {'color': 'red', 'name': f'{trace_name} (Above Monthly Target - Peak Period)'}
    }
    
    # Track legend status
    legend_added = {'below_target': False, 'above_target_offpeak': False, 'above_target_peak': False}
    
    # Create continuous line segments by color groups with bridge points (exact V2 approach)
    i = 0
    while i < len(segments):
        current_segment = segments[i]
        current_type = current_segment['type']
        
        # Extract segment data
        segment_x = list(current_segment['x'])
        segment_y = list(current_segment['y'])
        
        # Add bridge points for better continuity (connect to adjacent segments)
        if i > 0:  # Add connection point from previous segment
            prev_segment = segments[i-1]
            if len(prev_segment['x']) > 0:
                segment_x.insert(0, prev_segment['x'][-1])
                segment_y.insert(0, prev_segment['y'][-1])
        
        if i < len(segments) - 1:  # Add connection point to next segment
            next_segment = segments[i+1]
            if len(next_segment['x']) > 0:
                segment_x.append(next_segment['x'][0])
                segment_y.append(next_segment['y'][0])
        
        # Get color info
        color_info = color_map[current_type]
        
        # Only show legend for the first occurrence of each type
        show_legend = not legend_added[current_type]
        legend_added[current_type] = True
        
        # Add line segment
        fig.add_trace(go.Scatter(
            x=segment_x,
            y=segment_y,
            mode='lines',
            line=dict(color=color_info['color'], width=1),
            name=color_info['name'],
            opacity=0.8,
            showlegend=show_legend,
            legendgroup=current_type,
            connectgaps=True  # Connect gaps within segments
        ))
        
        i += 1
    
    return fig

def _add_v3_peak_event_highlights(fig, df, power_col, target_demand, peak_events_list):
    """
    Add filled areas to highlight individual peak events.
    Enhanced to accept either scalar or Series targets.
    
    Args:
        target_demand: Either a scalar value or a pandas Series aligned to df.index
    """
    # Check if target_demand is a Series or scalar
    is_target_series = isinstance(target_demand, pd.Series)
    
    # Group events by period type for better visualization
    peak_period_events = [e for e in peak_events_list if e['period'] == 'Peak']
    offpeak_period_events = [e for e in peak_events_list if e['period'] == 'Off-Peak']
    
    # Highlight peak period events with red fill
    for i, event in enumerate(peak_period_events):
        timestamp = event['timestamp']
        power_kw = event['power_kw']
        
        # Get the appropriate target for this timestamp
        if is_target_series:
            current_target = target_demand.loc[timestamp] if timestamp in target_demand.index else target_demand.iloc[0]
        else:
            current_target = target_demand
        
        # Create a small area around this event (± 15 minutes for visibility)
        start_time = timestamp - timedelta(minutes=7.5)
        end_time = timestamp + timedelta(minutes=7.5)
        
        # Create filled area
        x_coords = [start_time, end_time, end_time, start_time]
        y_coords = [current_target, current_target, power_kw, power_kw]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Peak Period Event' if i == 0 else '',
            showlegend=(i == 0),
            legendgroup='peak_events',
            hoverinfo='skip'
        ))
    
    # Highlight off-peak period events with green fill
    for i, event in enumerate(offpeak_period_events):
        timestamp = event['timestamp']
        power_kw = event['power_kw']
        
        # Get the appropriate target for this timestamp
        if is_target_series:
            current_target = target_demand.loc[timestamp] if timestamp in target_demand.index else target_demand.iloc[0]
        else:
            current_target = target_demand
        
        # Create a small area around this event
        start_time = timestamp - timedelta(minutes=7.5)
        end_time = timestamp + timedelta(minutes=7.5)
        
        # Create filled area
        x_coords = [start_time, end_time, end_time, start_time]
        y_coords = [current_target, current_target, power_kw, power_kw]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Off-Peak Period Event' if i == 0 else '',
            showlegend=(i == 0),
            legendgroup='offpeak_events',
            hoverinfo='skip'
        ))

def _display_v3_peak_events_table(df_processed, power_col, target_kw, selected_tariff, holidays):
    """
    Display detailed peak events table using V1 reused function with lifted parameters.
    """
    # Calculate parameters needed for V1 function (lifting globals to parameters)
    overall_max_demand = df_processed[power_col].max()
    
    # Calculate interval hours
    if len(df_processed) > 1:
        time_diff = df_processed.index[1] - df_processed.index[0]
        interval_hours = time_diff.total_seconds() / 3600
    else:
        interval_hours = 0.25  # Default 15-minute intervals
    
    # Get MD rate from tariff
    total_md_rate = get_tariff_md_rate_v3(selected_tariff) if selected_tariff else 30.0
    
    # Detect peak events using V1 function
    event_summaries = _detect_peak_events(
        df_processed, power_col, target_kw, total_md_rate, interval_hours, selected_tariff
    )
    
    # Note: Peak events table and chart are now handled by _display_v3_peak_events_chart in _render_v3_analysis_infrastructure
    if not event_summaries:
        st.info("✅ No peak events detected above target demand!")


def _display_v3_peak_events_analysis_summary(peak_events_list, tariff_analysis):
    """
    Display detailed analysis summary of detected peak events.
    """
    st.markdown("#### ⚡ Peak Event Detection Results")
    
    # Calculate additional statistics
    total_events = len(peak_events_list)
    peak_period_events = [e for e in peak_events_list if e['period'] == 'Peak']
    offpeak_period_events = [e for e in peak_events_list if e['period'] == 'Off-Peak']
    
    # Time-based analysis
    hourly_distribution = {}
    weekday_distribution = {}
    monthly_distribution = {}
    
    for event in peak_events_list:
        hour = event['hour']
        weekday = event['weekday']
        month = event['month']
        
        hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        weekday_distribution[weekday] = weekday_distribution.get(weekday, 0) + 1
        monthly_distribution[month] = monthly_distribution.get(month, 0) + 1
    
    # Display timing insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🕒 Peak Events by Time Pattern:**")
        if hourly_distribution:
            most_common_hour = max(hourly_distribution, key=hourly_distribution.get)
            st.write(f"• Most common hour: {most_common_hour:02d}:00 ({hourly_distribution[most_common_hour]} events)")
        
        if weekday_distribution:
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            most_common_weekday = max(weekday_distribution, key=weekday_distribution.get)
            st.write(f"• Most common day: {weekday_names[most_common_weekday]} ({weekday_distribution[most_common_weekday]} events)")
    
    with col2:
        st.markdown("**📊 Event Impact Distribution:**")
        if peak_period_events:
            avg_peak_excess = sum(e['excess_kw'] for e in peak_period_events) / len(peak_period_events)
            st.write(f"• Avg peak period excess: {avg_peak_excess:.1f} kW")
        
        if offpeak_period_events:
            avg_offpeak_excess = sum(e['excess_kw'] for e in offpeak_period_events) / len(offpeak_period_events)
            st.write(f"• Avg off-peak excess: {avg_offpeak_excess:.1f} kW")
    
    # Cost impact summary
    md_rate = tariff_analysis['tariff_config']['md_rate']
    max_monthly_cost = tariff_analysis['cost_analysis']['max_monthly_md_cost']
    
    if max_monthly_cost > 0:
        st.info(f"""
        💰 **Cost Impact Analysis:**
        - Maximum monthly MD cost impact: **RM {max_monthly_cost:,.0f}**
        - Annual cost potential: **RM {max_monthly_cost * 12:,.0f}**
        - MD rate: RM {md_rate:.2f}/kW/month
        """)

def _display_simplified_analysis_v3(df_processed, power_col, target_kw):
    """
    Simplified analysis when tariff is not configured.
    """
    # Basic peak events analysis
    peak_events = df_processed[df_processed[power_col] > target_kw]
    
    if not peak_events.empty:
        total_events = len(peak_events)
        max_excess = (peak_events[power_col] - target_kw).max()
        total_excess_energy = ((peak_events[power_col] - target_kw) * 0.25).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Events", f"{total_events:,}")
        with col2:
            st.metric("Max Excess Power", f"{max_excess:.1f} kW")
        with col3:
            st.metric("Total Excess Energy", f"{total_excess_energy:.1f} kWh")
        
        st.info("⚠️ Configure tariff for detailed cost analysis and tariff-aware peak event classification")
    else:
        st.success("✅ No peak events detected above target level!")

def render_md_shaving_v3():
    """
    Main function for MD Shaving V3 - Enhanced MD Shaving Analysis
    """
    st.title("🚀 MD Shaving (v3) - Enhanced Analysis")
    
    st.markdown("""
    **Enhanced Maximum Demand shaving analysis with advanced features:**
    
    - 📊 **Advanced Data Processing** with improved validation
    - 🎯 **Flexible Target Setting** with multiple methods
    - 🔋 **Enhanced Battery Analysis** with detailed metrics
    - 📈 **Improved Visualizations** and reporting
    - 💰 **Cost-Benefit Analysis** with ROI calculations
    """)

    # File upload section
    st.subheader("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your energy consumption data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with timestamp and power consumption data",
        key="v3_file_upload"
    )
    
    if uploaded_file:
        try:
            # Reuse V1 file reading logic
            df = read_uploaded_file(uploaded_file)
            
            if df is None or df.empty:
                st.error("The uploaded file appears to be empty or invalid.")
                return
            
            if not hasattr(df, 'columns') or df.columns is None or len(df.columns) == 0:
                st.error("The uploaded file doesn't have valid column headers.")
                return
                
            st.success("✅ File uploaded successfully!")
            
            # Reuse V1 data configuration
            st.subheader("📋 Data Configuration")
            
            # Column Selection and Holiday Configuration
            timestamp_col, power_col, holidays = _configure_data_inputs(df)
            
            # Only proceed if both columns are detected and valid
            if (timestamp_col and power_col and 
                hasattr(df, 'columns') and df.columns is not None and
                timestamp_col in df.columns and power_col in df.columns):
                
                # Process data
                df_processed = _process_dataframe(df, timestamp_col)
                
                if not df_processed.empty and power_col in df_processed.columns:
                    # Display tariff selection
                    st.subheader("⚡ Tariff Configuration")
                    
                    # Get tariff selection
                    try:
                        selected_tariff = _configure_tariff_selection()
                        if selected_tariff:
                            st.success(f"✅ Tariff configured: **{selected_tariff.get('Tariff', 'Unknown')}**")
                    except Exception as e:
                        st.warning(f"⚠️ Tariff configuration error: {str(e)}")
                        selected_tariff = None
                    
                    # V3 Target Setting Configuration
                    st.subheader("🎯 Target Setting")
                    
                    # Get overall max demand for calculations
                    overall_max_demand = df_processed[power_col].max()
                    avg_demand = df_processed[power_col].mean()
                    load_factor = (avg_demand / overall_max_demand) * 100
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Demand", f"{overall_max_demand:.1f} kW")
                    with col2:
                        st.metric("Average Demand", f"{avg_demand:.1f} kW")
                    with col3:
                        st.metric("Load Factor", f"{load_factor:.1f}%")
                    
                    # Target setting method selection
                    target_method = st.radio(
                        "Target Setting Method:",
                        options=["Percentage to Shave", "Percentage of Current Max", "Manual Target (kW)"],
                        index=0,
                        key="v3_target_method",
                        help="Choose how to set your maximum demand target"
                    )
                    
                    # Configure target based on selected method
                    if target_method == "Percentage to Shave":
                        shave_percent = st.slider(
                            "Percentage to Shave (%)", 
                            min_value=1, 
                            max_value=50, 
                            value=10, 
                            step=1,
                            key="v3_shave_percent",
                            help="Percentage to reduce from monthly peak"
                        )
                        target_kw = overall_max_demand * (1 - shave_percent/100)
                        target_description = f"{shave_percent}% shaving (Target: {target_kw:.1f} kW)"
                        user_shave_pct = shave_percent
                        
                    elif target_method == "Percentage of Current Max":
                        target_percent = st.slider(
                            "Target MD (% of monthly max)", 
                            min_value=50, 
                            max_value=100, 
                            value=90, 
                            step=1,
                            key="v3_target_percent",
                            help="Set target as percentage of monthly peak"
                        )
                        target_kw = overall_max_demand * (target_percent/100)
                        target_description = f"{target_percent}% of max (Target: {target_kw:.1f} kW)"
                        user_shave_pct = 100 - target_percent  # Convert to shaving percentage
                        
                    else:  # Manual Target
                        target_kw = st.number_input(
                            "Target MD (kW)",
                            min_value=0.0,
                            max_value=overall_max_demand,
                            value=overall_max_demand * 0.8,
                            step=10.0,
                            key="v3_target_manual",
                            help="Enter your desired target maximum demand in kW"
                        )
                        target_description = f"Manual target: {target_kw:.1f} kW"
                        # Calculate equivalent shaving percentage based on manual target
                        user_shave_pct = ((overall_max_demand - target_kw) / overall_max_demand) * 100
                    
                    # Display target information
                    st.info(f"🎯 **Target Configuration:** {target_description}")
                    
                    # V3 Tariff-Aware Analysis Infrastructure
                    _render_v3_analysis_infrastructure(
                        df_processed, power_col, target_kw, selected_tariff, holidays, 
                        overall_max_demand, avg_demand, load_factor, user_shave_pct
                    )
                    
                else:
                    st.error("❌ Failed to process the uploaded data")
            else:
                st.warning("⚠️ Please ensure your file has proper timestamp and power columns")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 **Upload your energy data file to begin V3 analysis**")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Your CSV/Excel file should contain:**
            - **Timestamp column**: Date and time information
            - **Power column**: Power consumption in kW
            
            **Supported formats:**
            - CSV files (.csv)
            - Excel files (.xlsx, .xls)
            
            **Example data structure:**
            ```
            Timestamp,Power_kW
            2024-01-01 00:00:00,125.5
            2024-01-01 00:15:00,130.2
            2024-01-01 00:30:00,128.8
            ...
            ```
            """)
            
            # Show sample data
            sample_data = {
                "Timestamp": ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30"],
                "Power_kW": [125.5, 130.2, 128.8],
                "Description": ["Normal load", "Slight increase", "Stable load"]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)

