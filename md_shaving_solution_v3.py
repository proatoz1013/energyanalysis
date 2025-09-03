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
    _simulate_battery_operation
)
from tariffs.peak_logic import is_peak_rp4

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
                                     overall_max_demand, avg_demand, load_factor):
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
        
        # Peak Events Analysis (Scaffolding)
        st.subheader("📊 Peak Events Analysis")
        _display_peak_events_summary_v3(tariff_analysis['peak_events'], target_kw)
        
        # V3 Peak Events Timeline visualization
        _display_v3_peak_events_chart(df_processed, power_col, target_kw, selected_tariff, holidays, tariff_analysis)
        
        # Cost Analysis (Scaffolding)
        st.subheader("💰 Cost Impact Analysis")
        _display_cost_analysis_summary_v3(tariff_analysis['cost_analysis'])
        
        # Battery Sizing Scaffolding
        st.subheader("🔋 Battery Sizing Analysis")
        _display_battery_sizing_scaffolding_v3(tariff_analysis, target_kw, overall_max_demand)
        
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

def _calculate_v3_monthly_targets(df, power_col, base_target_kw, selected_tariff, holidays):
    """
    Calculate monthly targets for V3 stepped target line.
    Creates month-specific targets based on monthly peak patterns and tariff type.
    """
    monthly_targets = {}
    
    try:
        # Group data by month and calculate monthly peaks
        df_monthly = df.groupby(df.index.to_period('M'))
        
        for month_period, month_data in df_monthly:
            if not month_data.empty:
                # Calculate monthly peak based on tariff type
                if selected_tariff:
                    tariff_type = selected_tariff.get('Type', '').lower()
                    tariff_name = selected_tariff.get('Tariff', '').lower()
                    is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
                    
                    if is_tou_tariff:
                        # For TOU tariffs, use peak during peak periods (2PM-10PM weekdays)
                        monthly_peak = _calculate_monthly_tou_peak(month_data, power_col, holidays)
                    else:
                        # For General tariffs, use overall monthly peak
                        monthly_peak = month_data[power_col].max()
                else:
                    # Fallback to overall monthly peak
                    monthly_peak = month_data[power_col].max()
                
                # Calculate target as percentage of monthly peak (default 90%)
                # This creates varying targets that adapt to monthly demand patterns
                target_percentage = 0.9  # 10% shaving
                monthly_target = monthly_peak * target_percentage
                
                # Ensure target doesn't go below the base target (safety measure)
                monthly_targets[month_period] = max(monthly_target, base_target_kw * 0.8)
                
        return monthly_targets
    except Exception as e:
        # Return simple monthly targets if calculation fails
        return {df.index[0].to_period('M'): base_target_kw} if len(df) > 0 else {}

def _calculate_monthly_tou_peak(month_data, power_col, holidays):
    """
    Calculate peak demand during TOU peak periods (2PM-10PM weekdays) for a given month.
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
    
    return max(tou_peak_data) if tou_peak_data else month_data[power_col].max()

def _add_v3_stepped_target_line(fig, df, monthly_targets, selected_tariff):
    """
    Add stepped monthly target line to the figure (V2 style implementation).
    """
    # Create stepped target line for visualization
    target_line_data = []
    target_line_timestamps = []
    
    # Create a stepped line that changes at month boundaries
    for month_period, target_value in monthly_targets.items():
        # Get start and end of month
        month_start = month_period.start_time
        month_end = month_period.end_time
        
        # Filter data for this month
        month_mask = (df.index >= month_start) & (df.index <= month_end)
        month_data = df[month_mask]
        
        if not month_data.empty:
            # Add target value for each timestamp in this month
            for timestamp in month_data.index:
                target_line_timestamps.append(timestamp)
                target_line_data.append(target_value)
    
    # Add stepped monthly target line first
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
            showlegend=True
        ))

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

def _display_v3_peak_events_chart(df_processed, power_col, target_kw, selected_tariff, holidays, tariff_analysis):
    """
    V3 Enhanced peak events visualization with tariff-aware coloring and analysis.
    """
    st.markdown("### 📈 Power Consumption with Peak Events Highlighted")
    
    # Get peak events from analysis
    peak_events_list = tariff_analysis['peak_events']['events_list']
    
    # Create main figure
    fig = go.Figure()
    
    # Calculate monthly targets for stepped target line
    monthly_targets = _calculate_v3_monthly_targets(df_processed, power_col, target_kw, selected_tariff, holidays)
    
    # Display monthly targets summary if available
    if monthly_targets and len(monthly_targets) > 1:
        _display_v3_monthly_targets_summary(monthly_targets, selected_tariff)
    
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
    
    # Create color-coded continuous line using V3 tariff logic
    fig = _create_v3_conditional_demand_line(
        fig, df_processed, power_col, target_kw, selected_tariff, holidays, "Power Consumption"
    )
    
    # Highlight individual peak events with filled areas
    if peak_events_list:
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

def _create_v3_conditional_demand_line(fig, df, power_col, target_kw, selected_tariff, holidays, trace_name):
    """
    V3 Enhanced conditional coloring logic with tariff-aware classification.
    Follows exact V1/V2 logic: red = above target + peak, green = above target + off-peak, blue = below target.
    """
    # Create color classification following V1/V2 exact logic
    color_segments = []
    current_segment = {'type': None, 'x': [], 'y': []}
    
    for timestamp, row in df.iterrows():
        power_value = row[power_col]
        
        # Get tariff period classification using V3 logic
        if selected_tariff:
            period_type = get_tariff_period_classification_v3(timestamp, selected_tariff, holidays)
        else:
            period_type = 'Peak'  # Default fallback
        
        # Follow exact V1/V2 color logic
        if power_value > target_kw:
            if period_type == 'Peak':
                color_class = 'red'
            else:
                color_class = 'green'
        else:
            color_class = 'blue'
        
        # Handle segment transitions
        if current_segment['type'] != color_class:
            # Finalize previous segment
            if current_segment['type'] is not None and len(current_segment['x']) > 0:
                color_segments.append(current_segment.copy())
            
            # Start new segment
            current_segment = {
                'type': color_class,
                'x': [timestamp],
                'y': [power_value]
            }
        else:
            # Continue current segment
            current_segment['x'].append(timestamp)
            current_segment['y'].append(power_value)
    
    # Don't forget the last segment
    if current_segment['type'] is not None and len(current_segment['x']) > 0:
        color_segments.append(current_segment)
    
    # Define legend names following V1/V2 style
    legend_names = {
        'red': f'{trace_name} (Above Target - Peak Period)',
        'green': f'{trace_name} (Above Target - Off-Peak Period)', 
        'blue': f'{trace_name} (Below Target)'
    }
    
    # Track legend status
    legend_added = {'red': False, 'green': False, 'blue': False}
    
    # Add colored line segments with continuity (V1/V2 style)
    for i, segment in enumerate(color_segments):
        color_class = segment['type']
        segment_x = list(segment['x'])
        segment_y = list(segment['y'])
        
        # Add bridge points for continuity (V1/V2 approach)
        if i > 0:  # Connect to previous segment
            prev_segment = color_segments[i-1]
            if len(prev_segment['x']) > 0:
                segment_x.insert(0, prev_segment['x'][-1])
                segment_y.insert(0, prev_segment['y'][-1])
        
        if i < len(color_segments) - 1:  # Connect to next segment
            next_segment = color_segments[i+1]
            if len(next_segment['x']) > 0:
                segment_x.append(next_segment['x'][0])
                segment_y.append(next_segment['y'][0])
        
        # Only show legend for first occurrence (V1/V2 approach)
        show_legend = not legend_added[color_class]
        legend_added[color_class] = True
        
        # Add line segment with exact V1/V2 styling
        fig.add_trace(go.Scatter(
            x=segment_x,
            y=segment_y,
            mode='lines',
            line=dict(color=color_class, width=2),  # Use color name directly like V1/V2
            name=legend_names[color_class],
            showlegend=show_legend,
            legendgroup=color_class,
            connectgaps=True
        ))
    
    return fig

def _add_v3_peak_event_highlights(fig, df, power_col, target_kw, peak_events_list):
    """
    Add filled areas to highlight individual peak events.
    """
    # Group events by period type for better visualization
    peak_period_events = [e for e in peak_events_list if e['period'] == 'Peak']
    offpeak_period_events = [e for e in peak_events_list if e['period'] == 'Off-Peak']
    
    # Highlight peak period events with red fill
    for i, event in enumerate(peak_period_events):
        timestamp = event['timestamp']
        power_kw = event['power_kw']
        
        # Create a small area around this event (± 15 minutes for visibility)
        start_time = timestamp - timedelta(minutes=7.5)
        end_time = timestamp + timedelta(minutes=7.5)
        
        # Create filled area
        x_coords = [start_time, end_time, end_time, start_time]
        y_coords = [target_kw, target_kw, power_kw, power_kw]
        
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
        
        # Create a small area around this event
        start_time = timestamp - timedelta(minutes=7.5)
        end_time = timestamp + timedelta(minutes=7.5)
        
        # Create filled area
        x_coords = [start_time, end_time, end_time, start_time]
        y_coords = [target_kw, target_kw, power_kw, power_kw]
        
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
                    
                    # Display target information
                    st.info(f"🎯 **Target Configuration:** {target_description}")
                    
                    # V3 Tariff-Aware Analysis Infrastructure
                    _render_v3_analysis_infrastructure(
                        df_processed, power_col, target_kw, selected_tariff, holidays, 
                        overall_max_demand, avg_demand, load_factor
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

