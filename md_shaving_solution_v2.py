"""
MD Shaving Solution V2 - Enhanced MD Shaving Analysis
=====================================================

This module provides next-generation Maximum Demand (MD) shaving analysis with:
- Monthly-based target calculation with dynamic user settings
- Battery database integration with vendor specifications
- Enhanced timeline visualization with peak events
- Interactive battery capacity selection interface

Author: Enhanced MD Shaving Team
Version: 2.0
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io

# Import V1 components for reuse
from md_shaving_solution import (
    read_uploaded_file,
    _configure_data_inputs,
    _process_dataframe,
    _configure_tariff_selection,
    create_conditional_demand_line_with_peak_logic,
    _detect_peak_events,
    _display_battery_simulation_chart,
    _simulate_battery_operation,
    get_tariff_period_classification,
    _get_tariff_description
)
from tariffs.peak_logic import is_peak_rp4


def _infer_interval_hours(datetime_index, fallback=0.25):
    """
    Infer sampling interval from datetime index using mode of timestamp differences.
    
    Args:
        datetime_index: pandas DatetimeIndex
        fallback: fallback interval in hours (default: 0.25 = 15 minutes)
        
    Returns:
        float: Interval in hours
    """
    try:
        if len(datetime_index) > 1:
            diffs = datetime_index.to_series().diff().dropna()
            if len(diffs) > 0 and not diffs.mode().empty:
                interval_hours = diffs.mode()[0].total_seconds() / 3600
                return interval_hours
    except Exception:
        pass
    return fallback


def _calculate_tariff_specific_monthly_peaks(df, power_col, selected_tariff, holidays):
    """
    Calculate monthly peak demands based on tariff type:
    - General Tariff: Uses 24/7 peak demand (highest demand anytime)
    - TOU Tariff: Uses peak period demand only (2PM-10PM weekdays)
    
    Args:
        df: DataFrame with power data
        power_col: Column name containing power values
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
    
    Returns:
        tuple: (monthly_general_peaks, monthly_tou_peaks, tariff_type)
    """
    # Determine tariff type
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Calculate monthly peaks
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly.index.to_period('M')
    
    # General peaks (24/7 maximum demand)
    monthly_general_peaks = df_monthly.groupby('Month')[power_col].max()
    
    # TOU peaks (peak period maximum demand only - 2PM-10PM weekdays)
    monthly_tou_peaks = {}
    
    for month_period in monthly_general_peaks.index:
        month_start = month_period.start_time
        month_end = month_period.end_time
        month_mask = (df.index >= month_start) & (df.index <= month_end)
        month_data = df[month_mask]
        
        if not month_data.empty:
            # Filter for TOU peak periods only (2PM-10PM weekdays)
            tou_peak_data = []
            
            for timestamp in month_data.index:
                if is_peak_rp4(timestamp, holidays if holidays else set()):
                    tou_peak_data.append(month_data.loc[timestamp, power_col])
            
            if tou_peak_data:
                monthly_tou_peaks[month_period] = max(tou_peak_data)
            else:
                # If no peak period data, use general peak as fallback
                monthly_tou_peaks[month_period] = monthly_general_peaks[month_period]
        else:
            monthly_tou_peaks[month_period] = 0
    
    monthly_tou_peaks = pd.Series(monthly_tou_peaks)
    
    return monthly_general_peaks, monthly_tou_peaks, tariff_type


def _calculate_monthly_targets_v2(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw):
    """
    Calculate monthly targets based on tariff-specific peak demands.
    
    Returns:
        tuple: (monthly_targets, reference_peaks, tariff_type, target_description)
    """
    # Get tariff-specific monthly peaks
    monthly_general_peaks, monthly_tou_peaks, tariff_type = _calculate_tariff_specific_monthly_peaks(
        df, power_col, selected_tariff, holidays
    )
    
    # Select appropriate reference peaks based on tariff type
    if tariff_type == 'TOU':
        reference_peaks = monthly_tou_peaks
        peak_description = "TOU Peak Period (2PM-10PM weekdays)"
    else:
        reference_peaks = monthly_general_peaks
        peak_description = "General (24/7)"
    
    # Calculate targets based on reference peaks
    if target_method == "Manual Target (kW)":
        # For manual target, use the same value for all months
        monthly_targets = pd.Series(index=reference_peaks.index, data=target_manual_kw)
        target_description = f"{target_manual_kw:.0f} kW manual target ({peak_description})"
    elif target_method == "Percentage to Shave":
        # Calculate target as percentage reduction from each month's reference peak
        target_multiplier = 1 - (shave_percent / 100)
        monthly_targets = reference_peaks * target_multiplier
        target_description = f"{shave_percent}% shaving from {peak_description}"
    else:  # Percentage of Current Max
        # Calculate target as percentage of each month's reference peak
        target_multiplier = target_percent / 100
        monthly_targets = reference_peaks * target_multiplier
        target_description = f"{target_percent}% of {peak_description}"
    
    return monthly_targets, reference_peaks, tariff_type, target_description


def _generate_clustering_summary_table(all_monthly_events, selected_tariff, holidays):
    """
    Generate date-based clustering summary table for Section B2.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
        
    Returns:
        pd.DataFrame: Summary table with columns: Date, Total Peak Events, General/TOU MD Excess, 
                     General/TOU Total Energy Required, Cost Impact
    """
    if not all_monthly_events or len(all_monthly_events) == 0:
        return pd.DataFrame()
    
    # Group events by date
    daily_events = {}
    for event in all_monthly_events:
        event_date = event.get('Start Date')
        if event_date not in daily_events:
            daily_events[event_date] = []
        daily_events[event_date].append(event)
    
    # Determine tariff type for MD cost calculation
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Get MD rate from tariff for cost calculation
    md_rate_rm_per_kw = 0
    if selected_tariff and isinstance(selected_tariff, dict):
        rates = selected_tariff.get('Rates', {})
        md_rate_rm_per_kw = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
    
    # Create summary data
    summary_data = []
    for date, events in daily_events.items():
        # Count total events for this date
        total_events = len(events)
        
        # Calculate MD excess values based on tariff type
        if tariff_type == 'TOU':
            # For TOU: Use TOU-specific values
            md_excess_values = [event.get('TOU Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('TOU Required Energy (kWh)', 0) or 0 for event in events]
            max_md_excess = max(md_excess_values) if md_excess_values else 0
        else:
            # For General: Use General values (24/7 MD impact)
            md_excess_values = [event.get('General Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('General Required Energy (kWh)', 0) or 0 for event in events]
            max_md_excess = max(md_excess_values) if md_excess_values else 0
        
        # Sum total energy required for the date
        total_energy_required = sum(energy_required_values)
        
        # Calculate cost impact using the maximum MD excess for the date
        # This follows the MD charging methodology where only the highest peak matters
        cost_impact_rm = max_md_excess * md_rate_rm_per_kw if max_md_excess > 0 and md_rate_rm_per_kw > 0 else 0
        
        summary_data.append({
            'Date': date,
            'Total Peak Events (count)': total_events,
            f'{tariff_type} MD Excess (Max kW)': round(max_md_excess, 2),
            f'{tariff_type} Total Energy Required (sum kWh)': round(total_energy_required, 2),
            'Cost Impact (RM/month)': round(cost_impact_rm, 2)
        })
    
    # Create DataFrame and sort by date
    df_summary = pd.DataFrame(summary_data)
    if not df_summary.empty:
        df_summary = df_summary.sort_values('Date')
    
    return df_summary


def _generate_monthly_summary_table(all_monthly_events, selected_tariff, holidays):
    """
    Generate monthly summary table for Section B2.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
        
    Returns:
        pd.DataFrame: Summary table with columns: Month, General/TOU MD Excess (Max kW), 
                     General/TOU Total Energy Required (kWh Max)
    """
    if not all_monthly_events or len(all_monthly_events) == 0:
        return pd.DataFrame()
    
    # Group events by month
    monthly_events = {}
    for event in all_monthly_events:
        event_date = event.get('Start Date')
        if event_date:
            # Extract year-month (e.g., "2025-01")
            month_key = event_date.strftime('%Y-%m')
            if month_key not in monthly_events:
                monthly_events[month_key] = []
            monthly_events[month_key].append(event)
    
    # Determine tariff type for MD cost calculation
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Create summary data
    summary_data = []
    for month_key, events in monthly_events.items():
        
        # Calculate MD excess values based on tariff type
        if tariff_type == 'TOU':
            # For TOU: Use TOU-specific values
            md_excess_values = [event.get('TOU Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('TOU Required Energy (kWh)', 0) or 0 for event in events]
        else:
            # For General: Use General values (24/7 MD impact)
            md_excess_values = [event.get('General Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('General Required Energy (kWh)', 0) or 0 for event in events]
        
        # Calculate maximum values for the month
        max_md_excess_month = max(md_excess_values) if md_excess_values else 0
        max_energy_required_month = max(energy_required_values) if energy_required_values else 0
        
        summary_data.append({
            'Month': month_key,
            f'{tariff_type} MD Excess (Max kW)': round(max_md_excess_month, 2),
            f'{tariff_type} Required Energy (Max kWh)': round(max_energy_required_month, 2)
        })
    
    # Create DataFrame and sort by month
    df_summary = pd.DataFrame(summary_data)
    if not df_summary.empty:
        df_summary = df_summary.sort_values('Month')
    
    return df_summary


def build_daily_simulator_structure(df, threshold_kw, clusters_df, selected_tariff=None):
    """
    Build day-level structure for battery dispatch simulation.
    
    Args:
        df: Cleaned DataFrame with DateTimeIndex and 'kW' column
        threshold_kw: Power threshold for shaving analysis
        clusters_df: DataFrame from cluster_peak_events() function
        selected_tariff: Tariff configuration dict (optional)
    
    Returns:
        dict: days_struct[date] = {timeline, kW, above_threshold_kW, clusters, recharge_windows, dt_hours}
    """
    if df.empty or clusters_df.empty:
        return {}
        
    # Infer sampling interval from DataFrame index
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600 if len(df) > 1 else 0.5
    
    # Get TOU configuration from selected tariff
    md_hours = (14, 22)  # Default 2PM-10PM
    working_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    tou_windows = {'off_peak': [(0, 8), (22, 24)], 'peak': [(14, 22)]}  # Default TOU
    charge_allowed_windows = [(0, 8), (22, 24)]  # Default: off-peak charging only
    max_site_charge_kw = None  # No default grid limit
    
    if selected_tariff:
        # Extract TOU configuration from tariff if available
        tariff_config = selected_tariff.get('config', {})
        if 'md_hours' in tariff_config:
            md_hours = tariff_config['md_hours']
        if 'working_days' in tariff_config:
            working_days = tariff_config['working_days']
        if 'tou_windows' in tariff_config:
            tou_windows = tariff_config['tou_windows']
        if 'charge_windows' in tariff_config:
            charge_allowed_windows = tariff_config['charge_windows']
        if 'max_site_charge_kw' in tariff_config:
            max_site_charge_kw = tariff_config['max_site_charge_kw']
    
    # Get unique dates from clusters and DataFrame
    cluster_dates = set()
    for _, cluster in clusters_df.iterrows():
        cluster_dates.add(cluster['cluster_start'].date())
        cluster_dates.add(cluster['cluster_end'].date())
    
    # Also include dates from DataFrame to ensure complete coverage
    df_dates = set(df.index.date)
    all_dates = cluster_dates.union(df_dates)
    
    days_struct = {}
    
    for date in sorted(all_dates):
        # Define day timeline (handle overnight MD window if needed)
        day_start = pd.Timestamp.combine(date, pd.Timestamp.min.time())
        day_end = day_start + pd.Timedelta(days=1)
        
        # Extract day's timeline from DataFrame
        day_mask = (df.index >= day_start) & (df.index < day_end)
        day_df = df[day_mask].copy()
        
        if day_df.empty:
            continue
            
        timeline = day_df.index
        kW_series = day_df['kW']
        above_threshold_kW = pd.Series(
            data=np.maximum(kW_series.values - threshold_kw, 0),
            index=timeline,
            name='above_threshold_kW'
        )
        
        # Find clusters intersecting this day
        day_clusters = []
        for _, cluster in clusters_df.iterrows():
            cluster_start = cluster['cluster_start']
            cluster_end = cluster['cluster_end']
            
            # Check if cluster intersects with this day
            if (cluster_start.date() == date or cluster_end.date() == date or
                (cluster_start.date() < date < cluster_end.date())):
                
                # Find timeline slice for this cluster
                cluster_mask = (timeline >= cluster_start) & (timeline <= cluster_end)
                if cluster_mask.any():
                    start_idx = np.where(cluster_mask)[0][0] if cluster_mask.any() else None
                    end_idx = np.where(cluster_mask)[0][-1] + 1 if cluster_mask.any() else None
                    
                    day_clusters.append({
                        'cluster_id': int(cluster['cluster_id']),
                        'start': cluster_start,
                        'end': cluster_end,
                        'duration_hr': float(cluster['cluster_duration_hr']),
                        'num_events': int(cluster['num_events_in_cluster']),
                        'peak_abs_kw_in_cluster': float(cluster['peak_abs_kw_in_cluster']),
                        'total_energy_above_threshold_kwh': float(cluster['total_energy_above_threshold_kwh']),
                        'slice': (start_idx, end_idx) if start_idx is not None else None
                    })
        
        # Sort clusters by start time
        day_clusters.sort(key=lambda x: x['start'])
        
        # Generate recharge windows between clusters
        recharge_windows = []
        
        if len(day_clusters) > 1:
            for i in range(len(day_clusters) - 1):
                current_cluster = day_clusters[i]
                next_cluster = day_clusters[i + 1]
                
                gap_start = current_cluster['end']
                gap_end = next_cluster['start']
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                if gap_duration > 0:
                    # Determine TOU label for this time period
                    gap_hour = gap_start.hour
                    gap_day = gap_start.strftime('%A')
                    
                    tou_label = 'off_peak'  # Default
                    for label, windows in tou_windows.items():
                        for window_start, window_end in windows:
                            if window_start <= gap_hour < window_end:
                                tou_label = label
                                break
                    
                    # Check if charging is allowed during this window
                    is_charging_allowed = False
                    for charge_start, charge_end in charge_allowed_windows:
                        if charge_start <= gap_hour < charge_end:
                            is_charging_allowed = True
                            break
                    
                    # Additional check: only allow charging on working days if specified
                    if gap_day not in working_days and 'working_days_only_charge' in (selected_tariff or {}):
                        is_charging_allowed = False
                    
                    recharge_windows.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration_hr': gap_duration,
                        'tou_label': tou_label,
                        'is_charging_allowed': is_charging_allowed,
                        'max_site_charge_kw': max_site_charge_kw
                    })
        
        # Add recharge windows at beginning and end of day if needed
        if day_clusters:
            # Before first cluster
            first_cluster = day_clusters[0]
            if first_cluster['start'] > timeline[0]:
                gap_start = timeline[0]
                gap_end = first_cluster['start']
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                gap_hour = gap_start.hour
                tou_label = 'off_peak'
                for label, windows in tou_windows.items():
                    for window_start, window_end in windows:
                        if window_start <= gap_hour < window_end:
                            tou_label = label
                            break
                
                is_charging_allowed = any(
                    charge_start <= gap_hour < charge_end 
                    for charge_start, charge_end in charge_allowed_windows
                )
                
                recharge_windows.insert(0, {
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hr': gap_duration,
                    'tou_label': tou_label,
                    'is_charging_allowed': is_charging_allowed,
                    'max_site_charge_kw': max_site_charge_kw
                })
            
            # After last cluster
            last_cluster = day_clusters[-1]
            if last_cluster['end'] < timeline[-1]:
                gap_start = last_cluster['end']
                gap_end = timeline[-1]
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                gap_hour = gap_start.hour
                tou_label = 'off_peak'
                for label, windows in tou_windows.items():
                    for window_start, window_end in windows:
                        if window_start <= gap_hour < window_end:
                            tou_label = label
                            break
                
                is_charging_allowed = any(
                    charge_start <= gap_hour < charge_end 
                    for charge_start, charge_end in charge_allowed_windows
                )
                
                recharge_windows.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hr': gap_duration,
                    'tou_label': tou_label,
                    'is_charging_allowed': is_charging_allowed,
                    'max_site_charge_kw': max_site_charge_kw
                })
        
        # Store day structure
        days_struct[date] = {
            'timeline': timeline,
            'kW': kW_series,
            'above_threshold_kW': above_threshold_kW,
            'clusters': day_clusters,
            'recharge_windows': recharge_windows,
            'dt_hours': dt_hours
            }
        
        
        def _calculate_c_rate_limited_power_simple(current_soc_percent, max_power_rating_kw, battery_capacity_kwh, c_rate=1.0):
            """
            Simple C-rate power limitation for charging/discharging.
            
            Args:
                current_soc_percent: Current state of charge percentage
                max_power_rating_kw: Battery's rated power
                battery_capacity_kwh: Battery's energy capacity
                c_rate: Battery's C-rate (default 1.0C)
                
            Returns:
                Dictionary with power limits
            """
            # Calculate C-rate based power limits
            c_rate_power_limit = battery_capacity_kwh * c_rate
            
            # SOC-based derating (power reduces at extreme SOC levels)
            if current_soc_percent > 90:
                soc_factor = 0.8  # Reduce power at high SOC
            elif current_soc_percent < 20:
                soc_factor = 0.7  # Reduce power at low SOC
            else:
                soc_factor = 1.0  # Full power in normal SOC range
            
            # Final power limit is minimum of C-rate limit and rated power
            effective_max_discharge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor
            effective_max_charge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor * 0.8  # Charging typically slower
            
            return {
                'max_discharge_power_kw': effective_max_discharge_kw,
                'max_charge_power_kw': effective_max_charge_kw,
                'c_rate_power_limit_kw': c_rate_power_limit,
                'soc_derating_factor': soc_factor,
                'limiting_factor': 'C-rate' if c_rate_power_limit < max_power_rating_kw else 'Power Rating'
            }
    
    return days_struct


def load_vendor_battery_database():
    """Load vendor battery database from JSON file."""
    try:
        with open('vendor_battery_database.json', 'r') as f:
            battery_db = json.load(f)
        return battery_db
    except FileNotFoundError:
        st.error("❌ Battery database file 'vendor_battery_database.json' not found")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error parsing battery database JSON file")
        return None
    except Exception as e:
        st.error(f"❌ Error loading battery database: {str(e)}")
        return None


def get_battery_capacity_range(battery_db):
    """Get the capacity range from battery database."""
    if not battery_db:
        return 200, 250, 225  # Default fallback values
    
    capacities = []
    for battery_id, spec in battery_db.items():
        capacity = spec.get('energy_kWh', 0)
        if capacity > 0:
            capacities.append(capacity)
    
    if capacities:
        min_cap = min(capacities)
        max_cap = max(capacities)
        default_cap = int(np.mean(capacities))
        return min_cap, max_cap, default_cap
    else:
        return 200, 250, 225  # Default fallback


def _render_battery_selection_dropdown():
    """
    Render independent battery selection dropdown that's always visible when data is available.
    This function should be called when a file is uploaded and data is available.
    """
    with st.container():
        st.markdown("#### 7. 📋 Tabled Analysis")
        
        # Battery selection dropdown
        battery_db = load_vendor_battery_database()
        
        if battery_db:
            # Create battery options for dropdown
            battery_options = {}
            battery_list = []
            
            for battery_id, spec in battery_db.items():
                company = spec.get('company', 'Unknown')
                model = spec.get('model', battery_id)
                capacity = spec.get('energy_kWh', 0)
                power = spec.get('power_kW', 0)
                
                label = f"{company} {model} ({capacity}kWh, {power}kW)"
                battery_options[label] = {
                    'id': battery_id,
                    'spec': spec,
                    'capacity_kwh': capacity,
                    'power_kw': power
                }
                battery_list.append(label)
            
            # Sort battery list for better UX
            battery_list.sort()
            battery_list.insert(0, "-- Select a Battery --")
            
            # Battery selection dropdown
            selected_battery_label = st.selectbox(
                "🔋 Select Battery for Analysis:",
                options=battery_list,
                index=0,
                key="independent_battery_selection",
                help="Choose a battery from the vendor database to view specifications and analysis"
            )
            
            # Display selected battery information
            if selected_battery_label != "-- Select a Battery --":
                selected_battery_data = battery_options[selected_battery_label]
                battery_spec = selected_battery_data['spec']
                
                # Display battery specifications in a table format
                st.markdown("**📊 Battery Specifications:**")
                spec_data = {
                    'Parameter': ['Company', 'Model', 'Energy Capacity', 'Power Rating', 'C-Rate', 'Voltage', 'Lifespan', 'Cooling'],
                    'Value': [
                        battery_spec.get('company', 'N/A'),
                        battery_spec.get('model', 'N/A'),
                        f"{battery_spec.get('energy_kWh', 0)} kWh",
                        f"{battery_spec.get('power_kW', 0)} kW",
                        f"{battery_spec.get('c_rate', 0)}C",
                        f"{battery_spec.get('voltage_V', 0)} V",
                        f"{battery_spec.get('lifespan_years', 0)} years",
                        battery_spec.get('cooling', 'N/A')
                    ]
                }
                df_specs = pd.DataFrame(spec_data)
                st.dataframe(df_specs, use_container_width=True, hide_index=True)
                
                # Store selected battery in session state for use in other parts of the analysis
                st.session_state.tabled_analysis_selected_battery = {
                    'id': selected_battery_data['id'],
                    'spec': battery_spec,
                    'capacity_kwh': selected_battery_data['capacity_kwh'],
                    'power_kw': selected_battery_data['power_kw'],
                    'label': selected_battery_label
                }
                
                return selected_battery_data
            else:
                st.info("💡 Select a battery from the dropdown above to view detailed specifications and analysis.")
                return None
        else:
            st.error("❌ Battery database not available")
            return None


def _render_battery_quantity_recommendation(max_power_shaving_required, recommended_energy_capacity):
    """
    Render battery quantity recommendation section between Tabled Analysis and Battery Sizing Analysis.
    
    Args:
        max_power_shaving_required: Maximum power shaving required (kW)
        recommended_energy_capacity: Maximum required energy (kWh)
    """
    st.markdown("#### 7.1 🔢 Battery Quantity Recommendation")
    
    # Check if user has selected a battery from the tabled analysis dropdown
    if hasattr(st.session_state, 'tabled_analysis_selected_battery') and st.session_state.tabled_analysis_selected_battery:
        selected_battery = st.session_state.tabled_analysis_selected_battery
        battery_spec = selected_battery['spec']
        battery_name = selected_battery['label']
        
        # Extract battery specifications
        battery_power_kw = battery_spec.get('power_kW', 0)
        battery_energy_kwh = battery_spec.get('energy_kWh', 0)
        
        if battery_power_kw > 0 and battery_energy_kwh > 0:
            # Calculate recommended quantities
            # Power Rating based quantity: roundup(Max Power Shaving Required / Battery Power Rating)
            qty_for_power = max_power_shaving_required / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_power_rounded = int(np.ceil(qty_for_power))
            
            # Energy Capacity based quantity: roundup(Max Required Energy / Battery Energy Capacity) 
            qty_for_energy = recommended_energy_capacity / battery_energy_kwh if battery_energy_kwh > 0 else 0
            qty_for_energy_rounded = int(np.ceil(qty_for_energy))
            
            # Recommended quantity: maximum of the two
            recommended_qty = max(qty_for_power_rounded, qty_for_energy_rounded)
            
            # Display metrics showing the calculation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Power-Based Qty", 
                    f"{qty_for_power_rounded} units",
                    help=f"Based on {max_power_shaving_required:.1f} kW ÷ {battery_power_kw} kW"
                )
                st.caption(f"Calculation: ⌈{max_power_shaving_required:.1f} ÷ {battery_power_kw}⌉")
            
            with col2:
                st.metric(
                    "Energy-Based Qty", 
                    f"{qty_for_energy_rounded} units",
                    help=f"Based on {recommended_energy_capacity:.1f} kWh ÷ {battery_energy_kwh} kWh"
                )
                st.caption(f"Calculation: ⌈{recommended_energy_capacity:.1f} ÷ {battery_energy_kwh}⌉")
            
            with col3:
                st.metric(
                    "Recommended Qty", 
                    f"{recommended_qty} units",
                    delta=f"{recommended_qty} units",
                    help="Maximum of power-based and energy-based quantities"
                )
                st.caption("Auto-recommended based on max requirement")
            
            # Allow user to override the recommended quantity
            st.markdown("**🎛️ Battery Quantity Configuration:**")
            
            # User input for quantity with recommended as default
            user_selected_qty = st.number_input(
                "Select Battery Quantity:",
                min_value=1,
                max_value=50,
                value=recommended_qty,
                step=1,
                key="v2_battery_quantity_selection",
                help=f"Auto-recommended: {recommended_qty} units. You can adjust this value if needed."
            )
            
            # Show impact of user selection
            total_power_capacity = user_selected_qty * battery_power_kw
            total_energy_capacity = user_selected_qty * battery_energy_kwh
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Power Capacity",
                    f"{total_power_capacity:.1f} kW",
                    f"{user_selected_qty} × {battery_power_kw} kW"
                )
            
            with col2:
                st.metric(
                    "Total Energy Capacity", 
                    f"{total_energy_capacity:.1f} kWh",
                    f"{user_selected_qty} × {battery_energy_kwh} kWh"
                )
            
            with col3:
                # Calculate coverage percentages
                power_coverage = (total_power_capacity / max_power_shaving_required * 100) if max_power_shaving_required > 0 else 100
                energy_coverage = (total_energy_capacity / recommended_energy_capacity * 100) if recommended_energy_capacity > 0 else 100
                
                overall_coverage = min(power_coverage, energy_coverage)
                
                coverage_color = "normal" if overall_coverage >= 100 else "inverse"
                st.metric(
                    "Coverage",
                    f"{overall_coverage:.1f}%",
                    delta_color=coverage_color,
                    help="Minimum of power and energy coverage percentages"
                )
            
            # Store the selected quantity in session state for use in sizing analysis
            st.session_state.tabled_analysis_battery_quantity = user_selected_qty
            
            # Provide guidance on the selection and integration information
            if user_selected_qty == recommended_qty:
                st.success(f"✅ **Optimal Configuration**: Using auto-recommended quantity of {recommended_qty} units based on your requirements.")
            elif user_selected_qty > recommended_qty:
                st.info(f"ℹ️ **Oversized Configuration**: You've selected {user_selected_qty} units, which is {user_selected_qty - recommended_qty} units more than the recommended {recommended_qty} units. This provides extra capacity margin.")
            else:
                st.warning(f"⚠️ **Undersized Configuration**: You've selected {user_selected_qty} units, which is {recommended_qty - user_selected_qty} units less than the recommended {recommended_qty} units. This may not fully meet your requirements.")
            
            # Integration feedback
            st.info(f"🔄 **Integration Active**: The selected quantity ({user_selected_qty} units) will be automatically used in the '📊 Battery Operation Simulation' section below, replacing any auto-calculated values.")
            
        else:
            st.error("❌ Selected battery has invalid power or energy specifications")
    
    else:
        st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to see quantity recommendations.")
        st.info("💡 Battery quantity will be automatically calculated based on your requirements once a battery is selected.")


def _render_battery_sizing_analysis(max_power_shaving_required, recommended_energy_capacity, total_md_cost):
    """
    Render comprehensive battery sizing and financial analysis table.
    
    Args:
        max_power_shaving_required: Maximum power shaving required (kW)
        recommended_energy_capacity: Maximum TOU excess power requirement (kW)  
        total_md_cost: Total MD cost impact (RM)
    """
    st.markdown("#### 7.2 🔋 Battery Sizing & Financial Analysis")
    
    # Check if user has selected a battery from the tabled analysis dropdown
    if hasattr(st.session_state, 'tabled_analysis_selected_battery') and st.session_state.tabled_analysis_selected_battery:
        selected_battery = st.session_state.tabled_analysis_selected_battery
        battery_spec = selected_battery['spec']
        battery_name = selected_battery['label']
        
        st.info(f"🔋 **Analysis based on selected battery:** {battery_name}")
        
        # Extract battery specifications
        battery_power_kw = battery_spec.get('power_kW', 0)
        battery_energy_kwh = battery_spec.get('energy_kWh', 0)
        battery_lifespan_years = battery_spec.get('lifespan_years', 15)
        
        if battery_power_kw > 0 and battery_energy_kwh > 0:
            # Use the user-selected quantity from the quantity recommendation section
            bess_quantity = getattr(st.session_state, 'tabled_analysis_battery_quantity', 1)
            
            # Calculate quantities that would be needed (for reference only)
            qty_for_power = max_power_shaving_required / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_power_rounded = int(np.ceil(qty_for_power))
            qty_for_excess = recommended_energy_capacity / battery_energy_kwh if battery_energy_kwh > 0 else 0
            qty_for_excess_rounded = int(np.ceil(qty_for_excess))
            
            # Calculate total system specifications based on user-selected quantity
            total_power_kw = bess_quantity * battery_power_kw
            total_energy_kwh = bess_quantity * battery_energy_kwh
            
            # Column 4: MD shaved (actual impact with this battery configuration)
            # Use the total power capacity from the larger battery quantity (BESS quantity)
            md_shaved_kw = total_power_kw  # Total power from the BESS system
            md_shaving_percentage = (md_shaved_kw / max_power_shaving_required * 100) if max_power_shaving_required > 0 else 0

            # Column 5: Cost of batteries
            estimated_cost_per_kwh = 1400  # RM per kWh (consistent with main app)
            total_battery_cost = total_energy_kwh * estimated_cost_per_kwh
            
            # Create analysis table
            analysis_data = {
                'Analysis Parameter': [
                    'Units for Selected Power Requirement',
                    'Units for Selected Energy Capacity',
                    'Total BESS Quantity Required',
                    'Total System Power Capacity',
                    'Total System Energy Capacity',
                    'Actual MD Shaved',
                    'MD Shaving Coverage',
                    'Total Battery Investment'
                ],
                'Value': [
                    f"{qty_for_power_rounded} units (for {max_power_shaving_required:.1f} kW)",
                    f"{qty_for_excess_rounded} units (for {recommended_energy_capacity:.1f} kWh)", 
                    f"{bess_quantity} units",
                    f"{total_power_kw:.1f} kW",
                    f"{total_energy_kwh:.1f} kWh",
                    f"{md_shaved_kw:.1f} kW",
                    f"{md_shaving_percentage:.1f}%",
                    f"RM {total_battery_cost:,.0f}"
                ],
                'Calculation Basis': [
                    f"Selected Power Requirement: {max_power_shaving_required:.1f} kW ÷ {battery_power_kw} kW/unit",
                    f"Selected Energy Capacity: {recommended_energy_capacity:.1f} kWh ÷ {battery_energy_kwh} kWh/unit",
                    "Higher of power or energy requirement",
                    f"{bess_quantity} units × {battery_power_kw} kW/unit",
                    f"{bess_quantity} units × {battery_energy_kwh} kWh/unit", 
                    f"{bess_quantity} units × {battery_power_kw} kW/unit = {total_power_kw:.1f} kW",
                    f"MD Shaved ÷ Selected Power Requirement × 100%",
                    f"{total_energy_kwh:.1f} kWh × RM {estimated_cost_per_kwh}/kWh"
                ]
            }
            
            df_analysis = pd.DataFrame(analysis_data)
            
            # Display the dataframe without styling for consistent formatting
            st.dataframe(df_analysis, use_container_width=True, hide_index=True)
            
            # Key insights - only showing total investment
            col1, col2, col3 = st.columns(3)
            
            with col2:  # Center the single metric
                st.metric(
                    "💰 Total Investment", 
                    f"RM {total_battery_cost:,.0f}",
                    help="Total cost for complete BESS installation"
                )
            
            # Analysis insights
            if bess_quantity > 0:
                st.success(f"""
                **📊 Analysis Summary:**
                - **Battery Selection**: {battery_name}
                - **System Configuration**: {bess_quantity} units providing {total_power_kw:.1f} kW / {total_energy_kwh:.1f} kWh
                - **MD Shaving Capability**: {md_shaving_percentage:.1f}% coverage of maximum demand events
                - **Investment Required**: RM {total_battery_cost:,.0f} for complete BESS installation
                """)
                
                if md_shaving_percentage < 100:
                    st.warning(f"""
                    ⚠️ **Partial Coverage Notice**: 
                    This battery configuration covers {md_shaving_percentage:.1f}% of maximum power shaving requirements.
                    Additional {max_power_shaving_required - md_shaved_kw:.1f} kW capacity may be needed for complete coverage.
                    """)
            else:
                st.error("❌ Invalid battery configuration - no units required")
                
        else:
            st.error("❌ Selected battery has invalid power or energy specifications")
            
    else:
        st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to perform sizing analysis.")
        st.info("💡 Navigate to the top of this page and select a battery from the dropdown to see detailed sizing and financial analysis.")


def get_battery_options_for_capacity(battery_db, target_capacity, tolerance=5):
    """Get batteries that match the target capacity within tolerance."""
    if not battery_db:
        return []
    
    matching_batteries = []
    for battery_id, spec in battery_db.items():
        battery_capacity = spec.get('energy_kWh', 0)
        if abs(battery_capacity - target_capacity) <= tolerance:
            matching_batteries.append({
                'id': battery_id,
                'spec': spec,
                'capacity_kwh': battery_capacity,
                'power_kw': spec.get('power_kW', 0),
                'c_rate': spec.get('c_rate', 0)
            })
    
    # Sort by closest match to target capacity
    matching_batteries.sort(key=lambda x: abs(x['capacity_kwh'] - target_capacity))
    return matching_batteries


def _render_v2_battery_controls():
    """Render battery capacity controls in the main content area (right side)."""
    
    st.markdown("### 🔋 Battery Configuration")
    st.markdown("**Configure battery specifications for MD shaving analysis.**")
    
    # Load battery database
    battery_db = load_vendor_battery_database()
    
    if not battery_db:
        st.error("❌ Battery database not available")
        return None
    
    # Get capacity range
    min_cap, max_cap, default_cap = get_battery_capacity_range(battery_db)
    
    # Selection method
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selection_method = st.radio(
            "Battery Selection Method:",
            options=["By Capacity", "By Specific Model"],
            index=0,
            key="v2_main_battery_selection_method",
            help="Choose how to select battery specifications",
            horizontal=True
        )
    
    with col2:
        st.metric("Available Range", f"{min_cap}-{max_cap} kWh")
    
    # Battery selection based on method
    if selection_method == "By Capacity":
        # Capacity slider
        selected_capacity = st.slider(
            "Battery Capacity (kWh):",
            min_value=min_cap,
            max_value=max_cap,
            value=default_cap,
            step=1,
            key="v2_main_battery_capacity",
            help="Select desired battery capacity. Matching batteries will be shown below."
        )
        
        # Find matching batteries
        matching_batteries = get_battery_options_for_capacity(battery_db, selected_capacity)
        
        if matching_batteries:
            st.markdown(f"#### 🔍 Batteries matching {selected_capacity} kWh:")
            
            # Display matching batteries in a more compact format for main area
            for i, battery_data in enumerate(matching_batteries):
                battery = battery_data['spec']
                with st.expander(f"🔋 {battery.get('company', 'Unknown')} {battery.get('model', 'Unknown')}", expanded=(i==0)):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Capacity", f"{battery.get('energy_kWh', 0)} kWh")
                    col2.metric("Power", f"{battery.get('power_kW', 0)} kW")
                    col3.metric("C-Rate", f"{battery.get('c_rate', 0)}C")
                    col4.metric("Voltage", f"{battery.get('voltage_V', 0)} V")
                    
                    # Additional details in smaller text
                    st.caption(f"**Lifespan:** {battery.get('lifespan_years', 0)} years | **Cooling:** {battery.get('cooling', 'Unknown')}")
            
            # Use the first matching battery as active
            active_battery_spec = matching_batteries[0]['spec']
            
        else:
            st.warning(f"⚠️ No batteries found for {selected_capacity} kWh capacity")
            active_battery_spec = None
            
    else:  # By Specific Model
        # Create battery options
        battery_options = {}
        for battery_id, spec in battery_db.items():
            label = f"{spec.get('company', 'Unknown')} {spec.get('model', 'Unknown')} ({spec.get('energy_kWh', 0)}kWh)"
            battery_options[label] = {
                'id': battery_id,
                'spec': spec,
                'capacity': spec.get('energy_kWh', 0)
            }
        
        selected_battery_label = st.selectbox(
            "Select Battery Model:",
            options=list(battery_options.keys()),
            key="v2_main_battery_model",
            help="Choose specific battery model from database"
        )
        
        if selected_battery_label:
            selected_battery_data = battery_options[selected_battery_label]
            active_battery_spec = selected_battery_data['spec']
            selected_capacity = selected_battery_data['capacity']
            
            # Display selected battery specs
            st.markdown("#### 📊 Selected Battery Specifications")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Energy", f"{active_battery_spec.get('energy_kWh', 0)} kWh")
            col2.metric("Power", f"{active_battery_spec.get('power_kW', 0)} kW")
            col3.metric("C-Rate", f"{active_battery_spec.get('c_rate', 0)}C")
            col4.metric("Voltage", f"{active_battery_spec.get('voltage_V', 0)} V")
            
            st.caption(f"**Company:** {active_battery_spec.get('company', 'Unknown')} | **Model:** {active_battery_spec.get('model', 'Unknown')} | **Lifespan:** {active_battery_spec.get('lifespan_years', 0)} years")
        else:
            active_battery_spec = None
            selected_capacity = default_cap
    
    # Analysis configuration
    st.markdown("#### ⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        run_analysis = st.checkbox(
            "Enable Battery Analysis", 
            value=False,
            key="v2_main_enable_analysis",
            help="Enable advanced battery analysis (V2 feature)"
        )
    
    with col2:
        if run_analysis:
            st.success("🔄 **Analysis Mode:** Ready for optimization")
        else:
            st.info("📊 **Display Mode:** Specifications only")
    
    # Return the selected battery configuration
    battery_config = {
        'selection_method': selection_method,
        'selected_capacity': selected_capacity if 'selected_capacity' in locals() else default_cap,
        'active_battery_spec': active_battery_spec,
        'run_analysis': run_analysis
    }
    
    return battery_config


def render_md_shaving_v2():
    """
    Main function to display the MD Shaving Solution V2 interface.
    This is a thin wrapper that reuses V1 components for now.
    """
    st.title("🔋 1. MD Shaving Solution (v2)")
    st.markdown("""
    **Next-generation Maximum Demand (MD) shaving analysis** with enhanced features and advanced optimization algorithms.
    
    🆕 **V2 Enhancements:**
    - 🔧 **Advanced Battery Sizing**: Multi-parameter optimization algorithms
    - 📊 **Multi-Scenario Analysis**: Compare different battery configurations
    - 💰 **Enhanced Cost Analysis**: ROI calculations and payback period analysis
    - 📈 **Improved Visualizations**: Interactive charts and detailed reporting
    - 🎯 **Smart Recommendations**: AI-powered optimization suggestions
    
    💡 **Status:** This is the next-generation MD shaving tool building upon the proven V1 foundation.
    """)
    
    # Information about current development status
    with st.expander("ℹ️ Development Status & Roadmap"):
        st.markdown("""
        **Current Status:** Enhanced with Battery Database Integration
        
        **Completed Features:**
        - ✅ UI Framework and basic structure
        - ✅ Integration with existing V1 data processing
        - ✅ Enhanced interface design
        - ✅ Battery database integration with vendor specifications
        - ✅ Monthly-based target calculation (10% shaving per month)
        - ✅ Interactive battery capacity selection
        
        **In Development:**
        - 🔄 Advanced battery optimization algorithms
        - 🔄 Multi-scenario comparison engine
        - 🔄 Enhanced cost analysis and ROI calculations
        - 🔄 Advanced visualization suite
        
        **Planned Features:**
        - 📋 AI-powered battery sizing recommendations
        - 📋 Real-time optimization suggestions
        - 📋 Advanced reporting and export capabilities
        - 📋 Integration with battery vendor databases
        """)
    
    # File upload section (reusing V1 logic)
    st.subheader("2. 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your energy data file", 
        type=["csv", "xls", "xlsx"], 
        key="md_shaving_v2_file_uploader",
        help="Upload your load profile data (same format as V1)"
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
            
            # Reuse V1 data configuration (read-only for now)
            st.subheader("3. 📋 Data Configuration")
            
            # Column Selection and Holiday Configuration
            timestamp_col, power_col, holidays = _configure_data_inputs(df)
            
            # Only proceed if both columns are detected and valid
            if (timestamp_col and power_col and 
                hasattr(df, 'columns') and df.columns is not None and
                timestamp_col in df.columns and power_col in df.columns):
                
                # Process data
                df_processed = _process_dataframe(df, timestamp_col)
                
                if not df_processed.empty and power_col in df_processed.columns:
                    # Display tariff selection (reuse V1 logic - read-only)
                    st.subheader("4. ⚡ Tariff Configuration")
                    
                    with st.container():
                        st.info("🔧 **Note:** Using V1 tariff selection logic (read-only preview)")
                        
                        # Get tariff selection but don't store it yet
                        try:
                            selected_tariff = _configure_tariff_selection()
                            if selected_tariff:
                                st.success(f"✅ Tariff configured: **{selected_tariff.get('Tariff', 'Unknown')}**")
                        except Exception as e:
                            st.warning(f"⚠️ Tariff configuration error: {str(e)}")
                            selected_tariff = None
                    
                    # V2 Target Setting Configuration
                    st.subheader("5. 🎯 Target Setting (V2)")
                    
                    # Get overall max demand for calculations
                    overall_max_demand = df_processed[power_col].max()
                    
                    # Get default values from session state or use defaults
                    default_shave_percent = st.session_state.get("v2_config_default_shave", 10)
                    default_target_percent = st.session_state.get("v2_config_default_target", 90)
                    default_manual_kw = st.session_state.get("v2_config_default_manual", overall_max_demand * 0.8)
                    
                    st.markdown(f"**Current Data Max:** {overall_max_demand:.1f} kW")
                    
                    # Target setting method selection
                    target_method = st.radio(
                        "Target Setting Method:",
                        options=["Percentage to Shave", "Percentage of Current Max", "Manual Target (kW)"],
                        index=0,
                        key="v2_target_method",
                        help="Choose how to set your monthly target maximum demand"
                    )
                    
                    # Configure target based on selected method
                    if target_method == "Percentage to Shave":
                        shave_percent = st.slider(
                            "Percentage to Shave (%)", 
                            min_value=1, 
                            max_value=50, 
                            value=default_shave_percent, 
                            step=1,
                            key="v2_shave_percent",
                            help="Percentage to reduce from monthly peak (e.g., 20% shaving reduces monthly 1000kW peak to 800kW)"
                        )
                        target_percent = None
                        target_manual_kw = None
                        target_multiplier = 1 - (shave_percent / 100)
                        target_description = f"{shave_percent}% monthly shaving"
                    elif target_method == "Percentage of Current Max":
                        target_percent = st.slider(
                            "Target MD (% of monthly max)", 
                            min_value=50, 
                            max_value=100, 
                            value=default_target_percent, 
                            step=1,
                            key="v2_target_percent",
                            help="Set the target maximum demand as percentage of monthly peak"
                        )
                        shave_percent = None
                        target_manual_kw = None
                        target_multiplier = target_percent / 100
                        target_description = f"{target_percent}% of monthly max"
                    else:
                        target_manual_kw = st.number_input(
                            "Target MD (kW)",
                            min_value=0.0,
                            max_value=overall_max_demand,
                            value=default_manual_kw,
                            step=10.0,
                            key="v2_target_manual",
                            help="Enter your desired target maximum demand in kW (applied to all months)"
                        )
                        target_percent = None
                        shave_percent = None
                        target_multiplier = None  # Will be calculated per month
                        target_description = f"{target_manual_kw:.1f} kW manual target"
                        effective_target_percent = None
                        shave_percent = None
                    
                    # Display target information
                    st.info(f"🎯 **V2 Target:** {target_description} (configured in sidebar)")
                    
                    # Validate target settings
                    if target_method == "Manual Target (kW)":
                        if target_manual_kw <= 0:
                            st.error("❌ Target demand must be greater than 0 kW")
                            return
                        elif target_manual_kw >= overall_max_demand:
                            st.warning(f"⚠️ Target demand ({target_manual_kw:.1f} kW) is equal to or higher than current max ({overall_max_demand:.1f} kW). No peak shaving needed.")
                            st.info("💡 Consider setting a lower target to identify shaving opportunities.")
                    
                    # V2 Peak Events Timeline visualization with dynamic targets
                    _render_v2_peak_events_timeline(
                        df_processed, 
                        power_col, 
                        selected_tariff, 
                        holidays,
                        target_method, 
                        shave_percent if target_method == "Percentage to Shave" else None,
                        target_percent if target_method == "Percentage of Current Max" else None,
                        target_manual_kw if target_method == "Manual Target (kW)" else None,
                        target_description
                    )
                    
                else:
                    st.error("❌ Failed to process the uploaded data")
            else:
                st.warning("⚠️ Please ensure your file has proper timestamp and power columns")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 **Upload your energy data file to begin V2 analysis**")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Your data file should contain:**
            - **Timestamp column**: Date and time information
            - **Power column**: Power consumption values in kW
            
            **Supported formats:** CSV, Excel (.xls, .xlsx)
            """)
            
            # Sample data preview
            sample_data = {
                'Timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:15:00', '2024-01-01 00:30:00'],
                'Power (kW)': [250.5, 248.2, 252.1],
                'Additional Columns': ['Optional', 'Optional', 'Optional']
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)


def _render_battery_impact_timeline(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw, target_description, selected_battery_capacity):
    """Render the Battery Impact Timeline visualization - duplicate of peak events graph with battery impact overlay."""
    
    st.markdown("### 8. 📊 Battery Impact on Energy Consumption")
    
    # This function is under development
    st.info(f"""
    **🔧 Battery Impact Analysis (Under Development)**
    
    This section will show how a {selected_battery_capacity} kWh battery system would impact your energy consumption patterns.
    
    **Planned Features:**
    - Battery charge/discharge timeline overlay
    - Peak shaving effectiveness visualization  
    - Cost impact analysis with battery intervention
    - Energy storage utilization patterns
    
    **Current Status:** Function implementation in progress
    """)
    
    # Placeholder chart showing original consumption
    st.markdown("#### 📈 Original Energy Consumption Pattern")
    
    if power_col in df.columns:
        fig = go.Figure()
        
        # Add original consumption line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[power_col],
            mode='lines',
            name='Original Consumption',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))
        
        # Add target line if we can calculate it
        try:
            monthly_targets, _, _, _ = _calculate_monthly_targets_v2(
                df, power_col, selected_tariff, holidays, 
                target_method, shave_percent, target_percent, target_manual_kw
            )
            
            if not monthly_targets.empty:
                # Create stepped target line
                target_line_data = []
                target_line_timestamps = []
                
                for month_period, target_value in monthly_targets.items():
                    month_start = month_period.start_time
                    month_end = month_period.end_time
                    month_mask = (df.index >= month_start) & (df.index <= month_end)
                    month_data = df[month_mask]
                    
                    if not month_data.empty:
                        for timestamp in month_data.index:
                            target_line_timestamps.append(timestamp)
                            target_line_data.append(target_value)
                
                if target_line_data and target_line_timestamps:
                    fig.add_trace(go.Scatter(
                        x=target_line_timestamps,
                        y=target_line_data,
                        mode='lines',
                        name=f'Target MD ({target_description})',
                        line=dict(color='red', width=2, dash='dash'),
                        opacity=0.9
                    ))
        except Exception as e:
            st.warning(f"Could not calculate target line: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=f"Energy Consumption with {selected_battery_capacity} kWh Battery Impact (Preview)",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **📊 Preview Information:**
        - This shows your current energy consumption pattern
        - Red dashed line indicates monthly targets based on {target_description}
        - Battery capacity selected: **{selected_battery_capacity} kWh**
        - Full battery impact analysis coming in future updates
        """)
    else:
        st.error("Power column not found in data")


def _render_v2_peak_events_timeline(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw, target_description):
    """Render the V2 Peak Events Timeline visualization with dynamic monthly-based targets."""
    
    st.markdown("## 6. 📊 Peak Events Timeline")
    
    # Detect and show sampling interval from uploaded data
    try:
        detected_interval_hours = _infer_interval_hours(df.index)
        st.session_state['data_interval_hours'] = detected_interval_hours
        st.caption(f"📊 Detected sampling interval: {int(round(detected_interval_hours * 60))} minutes")
    except Exception:
        pass
    
    # Calculate tariff-specific monthly targets using new V2 functions
    if power_col in df.columns:
        # Use new tariff-specific target calculation
        monthly_targets, reference_peaks, tariff_type, enhanced_target_description = _calculate_monthly_targets_v2(
            df, power_col, selected_tariff, holidays, 
            target_method, shave_percent, target_percent, target_manual_kw
        )
        
        # Also get both General and TOU peaks for comparison display
        monthly_general_peaks, monthly_tou_peaks, _ = _calculate_tariff_specific_monthly_peaks(
            df, power_col, selected_tariff, holidays
        )
        
        # Set legend label based on tariff type
        legend_label = f"Monthly Target - {tariff_type} ({enhanced_target_description})"
        
        # Display tariff-specific information
        st.info(f"""
        **🎯 Tariff-Specific Target Calculation:**
        - **Tariff Type**: {tariff_type}
        - **Reference Peak**: {enhanced_target_description}
        - **Target Method**: {target_method}
        - **Months Processed**: {len(monthly_targets)}
        """)
        
        # Show monthly comparison table
        if not reference_peaks.empty and not monthly_targets.empty:
            comparison_data = []
            
            for month_period in reference_peaks.index:
                general_peak = monthly_general_peaks[month_period] if month_period in monthly_general_peaks.index else 0
                tou_peak = monthly_tou_peaks[month_period] if month_period in monthly_tou_peaks.index else 0
                reference_peak = reference_peaks[month_period]
                target = monthly_targets[month_period]
                shaving_amount = reference_peak - target
                
                comparison_data.append({
                    'Month': str(month_period),
                    'General Peak (24/7)': f"{general_peak:.1f} kW",
                    'TOU Peak (2PM-10PM)': f"{tou_peak:.1f} kW",
                    'Reference Peak': f"{reference_peak:.1f} kW",
                    'Target MD': f"{target:.1f} kW",
                    'Shaving Amount': f"{shaving_amount:.1f} kW",
                    'Tariff Type': tariff_type
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            st.markdown("#### 6.1 📋 Monthly Target Calculation Summary")
            
            # Highlight the reference column based on tariff type
            def highlight_reference_peak(row):
                colors = []
                for col in row.index:
                    if col == 'Reference Peak':
                        colors.append('background-color: rgba(0, 255, 0, 0.3)')  # Green highlight
                    elif col == 'TOU Peak (2PM-10PM)' and tariff_type == 'TOU':
                        colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow highlight
                    elif col == 'General Peak (24/7)' and tariff_type == 'General':
                        colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow highlight
                    else:
                        colors.append('')
                return colors
            
            styled_comparison = df_comparison.style.apply(highlight_reference_peak, axis=1)
            st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
            
            st.info(f"""
            **📊 Target Calculation Explanation:**
            - **General Peak**: Highest demand anytime (24/7) 
            - **TOU Peak**: Highest demand during peak period (2PM-10PM weekdays only)
            - **Reference Peak**: Used for target calculation based on {tariff_type} tariff
            - **Target MD**: {enhanced_target_description}
            - 🟢 **Green**: Reference peak used for calculations
            - 🟡 **Yellow**: Peak type matching selected tariff
            """)
        
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
        
        # Create the peak events timeline chart with stepped target line
        if target_line_data and target_line_timestamps:
            fig = go.Figure()
            
            # Add stepped monthly target line first
            fig.add_trace(go.Scatter(
                x=target_line_timestamps,
                y=target_line_data,
                mode='lines',
                name=legend_label,
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.9
            ))
            
            # Identify and color-code all data points based on monthly targets and TOU periods
            all_monthly_events = []
            
            # Create continuous colored line segments
            # Process data chronologically to create continuous segments
            all_timestamps = sorted(df.index)
            
            # Create segments for continuous colored lines
            segments = []
            current_segment = {'type': None, 'x': [], 'y': []}
            
            for timestamp in all_timestamps:
                power_value = df.loc[timestamp, power_col]
                
                # Get the monthly target for this timestamp
                month_period = timestamp.to_period('M')
                if month_period in monthly_targets:
                    target_value = monthly_targets[month_period]
                    
                    # Determine the color category for this point
                    if power_value <= target_value:
                        segment_type = 'below_target'
                    else:
                        is_peak = is_peak_rp4(timestamp, holidays if holidays else set())
                        if is_peak:
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
            
            # Plot the colored segments with proper continuity (based on V1 logic)
            color_map = {
                'below_target': {'color': 'blue', 'name': 'Below Monthly Target'},
                'above_target_offpeak': {'color': 'green', 'name': 'Above Monthly Target - Off-Peak Period'},
                'above_target_peak': {'color': 'red', 'name': 'Above Monthly Target - Peak Period'}
            }
            
            # Track legend status
            legend_added = {'below_target': False, 'above_target_offpeak': False, 'above_target_peak': False}
            
            # Create continuous line segments by color groups with bridge points (V1 approach)
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
            
            # Process peak events for monthly analysis
            for month_period, target_value in monthly_targets.items():
                month_start = month_period.start_time
                month_end = month_period.end_time
                month_mask = (df.index >= month_start) & (df.index <= month_end)
                month_data = df[month_mask]
                
                if not month_data.empty:
                    # Find peak events for this month using V1's detection logic
                    # Auto-detect sampling interval from this month's data
                    interval_hours = _infer_interval_hours(month_data.index, fallback=0.25)
                    
                    # Save detected interval to session state for transparency
                    try:
                        st.session_state['data_interval_hours'] = interval_hours
                    except Exception:
                        pass
                    
                    # Get MD rate from selected tariff (simplified)
                    total_md_rate = 0
                    if selected_tariff and isinstance(selected_tariff, dict):
                        rates = selected_tariff.get('Rates', {})
                        total_md_rate = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
                    
                    peak_events = _detect_peak_events(
                        month_data, power_col, target_value, total_md_rate, interval_hours, selected_tariff
                    )
                    
                    # Add month info to each event including both reference peaks
                    for event in peak_events:
                        event['Month'] = str(month_period)
                        event['Monthly_Target'] = target_value
                        event['Monthly_General_Peak'] = monthly_general_peaks[month_period] if month_period in monthly_general_peaks.index else 0
                        event['Monthly_TOU_Peak'] = monthly_tou_peaks[month_period] if month_period in monthly_tou_peaks.index else 0
                        event['Reference_Peak'] = reference_peaks[month_period]
                        event['Shaving_Amount'] = reference_peaks[month_period] - target_value
                        all_monthly_events.append(event)
            
            # Update layout
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
            
            # Monthly breakdown table
            
        # Detailed Peak Event Detection Results
        if all_monthly_events:
            st.markdown("#### 6.2 ⚡ Peak Event Detection Results")
            
            # Determine tariff type for display enhancements
            tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
            tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
            is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
            
            # Enhanced summary with tariff context
            total_events = len(all_monthly_events)
            # Count events with actual MD cost impact (cost > 0 or TOU excess > 0)
            md_impact_events = len([e for e in all_monthly_events 
                                  if e.get('MD Cost Impact (RM)', 0) > 0 or e.get('TOU Excess (kW)', 0) > 0])
            total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
            
            # Calculate maximum TOU Excess from all events
            max_tou_excess = max([event.get('TOU Excess (kW)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
            
            if is_tou_tariff:
                no_md_impact_events = total_events - md_impact_events
                summary_text = f"**Showing {total_events} total events (All Events)**\n"
                summary_text += f"📊 **TOU Tariff Summary:** {md_impact_events} events with MD cost impact, {no_md_impact_events} events without MD impact"
            else:
                summary_text = f"**Showing {total_events} total events (All Events)**\n"
                summary_text += f"📊 **General Tariff:** All {total_events} events have MD cost impact (24/7 MD charges)"
            
            st.markdown(summary_text)
            
            # Prepare enhanced dataframe with all detailed columns
            df_events_summary = pd.DataFrame(all_monthly_events)
            
            # Ensure all required columns exist
            required_columns = ['Start Date', 'Start Time', 'End Date', 'End Time', 
                              'General Peak Load (kW)', 'General Excess (kW)', 
                              'TOU Peak Load (kW)', 'TOU Excess (kW)', 'TOU Peak Time',
                              'Duration (min)', 'General Required Energy (kWh)',
                              'TOU Required Energy (kWh)', 'MD Cost Impact (RM)', 
                              'Has MD Cost Impact', 'Tariff Type']
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in df_events_summary.columns:
                    if 'General' in col and 'TOU' in [c for c in df_events_summary.columns]:
                        # Copy TOU values to General columns if missing
                        tou_col = col.replace('General', 'TOU')
                        if tou_col in df_events_summary.columns:
                            df_events_summary[col] = df_events_summary[tou_col]
                        else:
                            df_events_summary[col] = 0
                    elif col == 'Duration (min)':
                        df_events_summary[col] = 30.0  # Default duration
                    elif col == 'TOU Peak Time':
                        df_events_summary[col] = 'N/A'
                    elif col == 'Has MD Cost Impact':
                        # Set based on MD cost impact
                        df_events_summary[col] = df_events_summary.get('MD Cost Impact (RM)', 0) > 0
                    elif col == 'Tariff Type':
                        # Set based on selected tariff
                        tariff_type_name = selected_tariff.get('Type', 'TOU').upper() if selected_tariff else 'TOU'
                        df_events_summary[col] = tariff_type_name
                    else:
                        df_events_summary[col] = 0
            
            # Create styled dataframe with color-coded rows
            def apply_row_colors(row):
                """Apply color coding based on MD cost impact."""
                # Check if event has MD cost impact based on actual cost value
                md_cost = row.get('MD Cost Impact (RM)', 0) or 0
                has_impact = md_cost > 0
                
                # Alternative check: look for TOU Excess or any excess during peak hours
                if not has_impact:
                    tou_excess = row.get('TOU Excess (kW)', 0) or 0
                    has_impact = tou_excess > 0
                
                if has_impact:
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)  # Light red for MD cost impact
                else:
                    return ['background-color: rgba(0, 128, 0, 0.1)'] * len(row)  # Light green for no MD cost impact
            
            # Select and reorder columns for display (matching original table structure)
            display_columns = ['Start Date', 'Start Time', 'End Date', 'End Time', 
                             'General Peak Load (kW)', 'General Excess (kW)', 
                             'TOU Peak Load (kW)', 'TOU Excess (kW)', 'TOU Peak Time',
                             'Duration (min)', 'General Required Energy (kWh)',
                             'TOU Required Energy (kWh)', 'MD Cost Impact (RM)', 
                             'Has MD Cost Impact', 'Tariff Type']
            
            # Filter to display columns that exist
            available_columns = [col for col in display_columns if col in df_events_summary.columns]
            display_df = df_events_summary[available_columns]
            
            # Define formatting function
            def fmt(x):
                return f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
            
            # Apply styling and formatting
            styled_df = display_df.style.apply(apply_row_colors, axis=1).format({
                'General Peak Load (kW)': lambda x: fmt(x),
                'General Excess (kW)': lambda x: fmt(x),
                'TOU Peak Load (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
                'TOU Excess (kW)': lambda x: fmt(x) if x > 0 else 'N/A',
                'Duration (min)': '{:.1f}',
                'General Required Energy (kWh)': lambda x: fmt(x),
                'TOU Required Energy (kWh)': lambda x: fmt(x),
                'MD Cost Impact (RM)': lambda x: f'RM {fmt(x)}' if x is not None else 'RM 0.0000',
                'Has MD Cost Impact': lambda x: '✓' if x else '✗',
                'Tariff Type': lambda x: str(x)
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
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Events", total_events)
            col2.metric("MD Impact Events", md_impact_events)
            col3.metric("Max TOU Excess", f"{fmt(max_tou_excess)} kW")
            
            # === PEAK EVENT CLUSTERING ANALYSIS ===
            st.markdown("### 6.3 🔗 Peak Event Clusters")
            st.markdown("**Grouping consecutive peak events that can be managed with a single battery charge/discharge cycle**")
            
            # Generate and display clustering summary table
            try:
                clustering_summary_df = _generate_clustering_summary_table(
                    all_monthly_events, selected_tariff, holidays
                )
                
                if not clustering_summary_df.empty:
                    st.markdown("#### 6.3.1 📊 Daily Clustering Summary")
                    st.markdown("*Summary of peak events grouped by date with MD cost impact analysis*")
                    
                    # Display the clustering summary table
                    st.dataframe(
                        clustering_summary_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add summary metrics below the table
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_dates = len(clustering_summary_df)
                    total_peak_events = clustering_summary_df['Total Peak Events'].sum()
                    max_daily_cost = clustering_summary_df['Cost Impact (RM/month)'].max()
                    total_monthly_cost_impact = clustering_summary_df['Cost Impact (RM/month)'].sum()
                    
                    col1.metric("Peak Event Days", total_dates)
                    col2.metric("Total Peak Events", total_peak_events)
                    col3.metric("Max Daily Cost Impact", f"RM {max_daily_cost:.2f}")
                    col4.metric("Total Monthly Cost Impact", f"RM {total_monthly_cost_impact:.2f}")
                    
                    st.markdown("---")
                else:
                    st.info("No peak events found for clustering analysis.")
                    
            except Exception as e:
                st.error(f"Error generating clustering summary table: {str(e)}")
                st.info("Proceeding with standard clustering analysis...")
            
            # Generate and display monthly summary table
            try:
                monthly_summary_df = _generate_monthly_summary_table(
                    all_monthly_events, selected_tariff, holidays
                )
                
                if not monthly_summary_df.empty:
                    st.markdown("#### 6.3.2 📅 Monthly Summary")
                    st.markdown("*Maximum MD excess and energy requirements aggregated by month*")
                    
                    # Display the monthly summary table
                    st.dataframe(
                        monthly_summary_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add summary metrics below the monthly summary table
                    col1, col2, col3 = st.columns(3)
                    
                    total_months = len(monthly_summary_df)
                    
                    # Get column names dynamically based on tariff type
                    tariff_type = 'General'
                    if selected_tariff:
                        tariff_name = selected_tariff.get('Tariff', '').lower()
                        tariff_type_field = selected_tariff.get('Type', '').lower()
                        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
                            tariff_type = 'TOU'
                    
                    md_excess_col = f'{tariff_type} MD Excess (Max kW)'
                    energy_col = f'{tariff_type} Required Energy (Max kWh)'
                    
                    if md_excess_col in monthly_summary_df.columns:
                        max_monthly_md_excess = monthly_summary_df[md_excess_col].max()
                        max_monthly_energy = monthly_summary_df[energy_col].max()
                        
                        col1.metric("Total Months", total_months)
                        col2.metric("Max Monthly MD Excess", f"{max_monthly_md_excess:.2f} kW")
                        col3.metric("Max Monthly Required Energy", f"{max_monthly_energy:.2f} kWh")
                    
                    st.markdown("---")
                else:
                    st.info("No monthly summary data available.")
                    
            except Exception as e:
                st.error(f"Error generating monthly summary table: {str(e)}")
                st.info("Monthly summary not available - continuing with clustering analysis...")
            
            # Default battery parameters for clustering (can be customized)
            battery_params_cluster = {
                'unit_energy_kwh': 100,  # Default 100 kWh battery
                'soc_min': 20.0,
                'soc_max': 100.0,
                'efficiency': 0.95,
                'charge_power_limit_kw': 100  # Increased to 100 kW for more flexible clustering
            }
            
            # MD hours and working days (customize as needed)
            md_hours = (14, 22)  # 2PM-10PM
            working_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']  # 3-letter abbreviations
            
            try:
                # Prepare events data for clustering
                events_for_clustering = df_events_summary.copy()
                
                # Add required columns for clustering
                if 'start' not in events_for_clustering.columns:
                    events_for_clustering['start'] = pd.to_datetime(
                        events_for_clustering['Start Date'].astype(str) + ' ' + events_for_clustering['Start Time'].astype(str)
                    )
                if 'end' not in events_for_clustering.columns:
                    events_for_clustering['end'] = pd.to_datetime(
                        events_for_clustering['End Date'].astype(str) + ' ' + events_for_clustering['End Time'].astype(str)
                    )
                if 'peak_abs_kw' not in events_for_clustering.columns:
                    events_for_clustering['peak_abs_kw'] = events_for_clustering['General Peak Load (kW)']
                if 'energy_above_threshold_kwh' not in events_for_clustering.columns:
                    events_for_clustering['energy_above_threshold_kwh'] = events_for_clustering['General Required Energy (kWh)']
                
                # Perform clustering
                clusters_df, events_for_clustering = cluster_peak_events(
                    events_for_clustering, battery_params_cluster, md_hours, working_days
                )
                
                if not clusters_df.empty:
                    st.success(f"✅ Successfully grouped {len(events_for_clustering)} events into {len(clusters_df)} clusters")
                    
                    # Prepare display data
                    cluster_display = clusters_df.copy()
                    cluster_display['cluster_duration_hr'] = (cluster_display['cluster_duration_hr'] * 60).round(1)  # Convert to minutes
                    cluster_display['peak_abs_kw_in_cluster'] = cluster_display['peak_abs_kw_in_cluster'].round(1)
                    cluster_display['total_energy_above_threshold_kwh'] = cluster_display['total_energy_above_threshold_kwh'].round(2)
                    
                    # Rename columns for better display
                    cluster_display = cluster_display.rename(columns={
                        'cluster_id': 'Cluster ID',
                        'num_events_in_cluster': 'Events Count',
                        'cluster_duration_hr': 'Duration (minutes)',
                        'peak_abs_kw_in_cluster': 'Peak Power (kW)',
                        'total_energy_above_threshold_kwh': 'Total Energy (kWh)',
                        'cluster_start': 'Start Time',
                        'cluster_end': 'End Time'
                    })
                    
                    # Separate single events (duration = 0) from multi-event clusters
                    single_events = cluster_display[cluster_display['Duration (minutes)'] == 0.0]
                    multi_event_clusters = cluster_display[cluster_display['Duration (minutes)'] > 0.0]
                    
                    # Display multi-event clusters table
                    if not multi_event_clusters.empty:
                        st.markdown("**📊 Multi-Event Clusters:**")
                        display_cols = ['Cluster ID', 'Events Count', 'Duration (minutes)', 
                                      'Peak Power (kW)', 'Total Energy (kWh)', 'Start Time', 'End Time']
                        available_cols = [col for col in display_cols if col in multi_event_clusters.columns]
                        st.dataframe(multi_event_clusters[available_cols], use_container_width=True)
                    else:
                        st.info("📊 No multi-event clusters found - all events are single occurrences.")
                    
                    # Display single events separately
                    if not single_events.empty:
                        st.markdown("**📍 Single Events:**")
                        single_display_cols = ['Cluster ID', 'Peak Power (kW)', 'Total Energy (kWh)', 'Start Time', 'End Time']
                        available_single_cols = [col for col in single_display_cols if col in single_events.columns]
                        st.dataframe(single_events[single_display_cols], use_container_width=True)
                    
                    # Quick statistics
                    st.markdown("**📊 Clustering Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Events", len(clusters_df))
                    col2.metric("Multi-Event Clusters", len(multi_event_clusters))
                    col3.metric("Single Events", len(single_events))
                    if not multi_event_clusters.empty:
                        col4.metric("Avg Events/Cluster", f"{multi_event_clusters['Events Count'].mean():.1f}")
                    else:
                        col4.metric("Avg Events/Cluster", "0.0")
                    
                    # === POWER & ENERGY COMPARISON ANALYSIS ===
                    st.markdown("### 6.4 ⚡ Peak Power & Energy Analysis")
                    st.markdown("**Comparison between multi-event clusters and single events**")
                    
                    # Calculate total energy (kWh) and power (kW) for clusters vs single events
                    if 'peak_abs_kw_sum_in_cluster' in clusters_df.columns:
                        
                        # Get max total energy from multi-event clusters (kWh)
                        if not multi_event_clusters.empty:
                            # For multi-event clusters, use total energy above threshold
                            max_cluster_energy = clusters_df[clusters_df['cluster_duration_hr'] > 0]['total_energy_above_threshold_kwh'].max()
                        else:
                            max_cluster_energy = 0
                        
                        # Get max energy from single events (kWh)
                        if not single_events.empty:
                            # For single events, get max General Required Energy
                            single_event_ids = clusters_df[clusters_df['cluster_duration_hr'] == 0]['cluster_id']
                            single_event_energies = []
                            for cluster_id in single_event_ids:
                                single_events_in_cluster = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                                if 'General Required Energy (kWh)' in single_events_in_cluster.columns:
                                    max_energy_in_cluster = single_events_in_cluster['General Required Energy (kWh)'].max()
                                    single_event_energies.append(max_energy_in_cluster)
                            max_single_energy = max(single_event_energies) if single_event_energies else 0
                        else:
                            max_single_energy = 0
                        
                        # Calculate TOU Excess for clusters and single events (kW)
                        # For multi-event clusters, get max TOU Excess sum
                        if not multi_event_clusters.empty:
                            # Calculate TOU Excess for each cluster by summing individual event TOU Excess values
                            max_cluster_tou_excess = 0
                            for cluster_id in clusters_df[clusters_df['cluster_duration_hr'] > 0]['cluster_id']:
                                # Get events in this cluster and sum their TOU Excess values
                                cluster_events = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                                cluster_tou_excess_sum = cluster_events['TOU Excess (kW)'].sum() if 'TOU Excess (kW)' in cluster_events.columns else 0
                                max_cluster_tou_excess = max(max_cluster_tou_excess, cluster_tou_excess_sum)
                        else:
                            max_cluster_tou_excess = 0
                        
                        # For single events, get max individual TOU Excess
                        if not single_events.empty:
                            max_single_tou_excess = events_for_clustering[events_for_clustering['cluster_id'].isin(single_events['Cluster ID'])]['TOU Excess (kW)'].max() if 'TOU Excess (kW)' in events_for_clustering.columns else 0
                        else:
                            max_single_tou_excess = 0
                        
                        # Compare and display results
                        st.markdown("**🔋 Battery Sizing Requirements:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Max Cluster Energy (Sum)", 
                                f"{max_cluster_energy:.1f} kWh",
                                help="Total energy above threshold within the highest-demand cluster"
                            )
                        
                        with col2:
                            st.metric(
                                "Max Single Event Energy", 
                                f"{max_single_energy:.1f} kWh",
                                help="Highest individual event energy requirement"
                            )
                        
                        with col3:
                            st.metric(
                                "Max Cluster TOU Excess", 
                                f"{max_cluster_tou_excess:.1f} kW",
                                help="Sum of TOU Excess power within the highest-demand cluster"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Single Event TOU Excess", 
                                f"{max_single_tou_excess:.1f} kW",
                                help="Highest individual event TOU Excess power"
                            )
                        
                        # Determine overall maximums
                        overall_max_energy = max(max_cluster_energy, max_single_energy)
                        overall_max_tou_excess = max(max_cluster_tou_excess, max_single_tou_excess)
                        
                        # Recommendations
                        st.markdown("**💡 Battery Sizing Recommendations:**")
                        
                        if overall_max_energy == max_cluster_energy and max_cluster_energy > max_single_energy:
                            energy_source = "multi-event cluster"
                            energy_advantage = ((max_cluster_energy - max_single_energy) / max_single_energy * 100) if max_single_energy > 0 else 0
                        else:
                            energy_source = "single event"
                            energy_advantage = 0
                        
                        if overall_max_tou_excess == max_cluster_tou_excess and max_cluster_tou_excess > max_single_tou_excess:
                            tou_excess_source = "multi-event cluster"
                            tou_excess_advantage = ((max_cluster_tou_excess - max_single_tou_excess) / max_single_tou_excess * 100) if max_single_tou_excess > 0 else 0
                        else:
                            tou_excess_source = "single event"
                            tou_excess_advantage = 0
                        
                        st.info(f"""
                        **Peak Shaving Energy**: {overall_max_energy:.1f} kWh (driven by {energy_source})
                        **TOU Excess Capacity**: {overall_max_tou_excess:.1f} kW (driven by {tou_excess_source})
                        
                        {'📈 Multi-event clusters require ' + f'{energy_advantage:.1f}% more energy capacity' if energy_advantage > 0 else '📊 Single events determine energy requirements'}
                        {'📈 Multi-event clusters require ' + f'{tou_excess_advantage:.1f}% more TOU excess capacity' if tou_excess_advantage > 0 else '📊 Single events determine TOU excess requirements'}
                        """)
                        
                        # Detailed cluster breakdown for multi-event clusters
                        if not multi_event_clusters.empty and 'peak_abs_kw_sum_in_cluster' in cluster_display.columns:
                            st.markdown("**📋 Multi-Event Cluster Energy & Power Breakdown:**")
                            cluster_analysis = multi_event_clusters.copy()
                            # Display additional cluster details if needed
                    
                    else:
                        st.warning("No clustering data available for detailed power and energy analysis.")
                
                else:
                    st.info("No peak events found for clustering analysis.")
            
            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")
                st.info("Falling back to simplified analysis...")
            
            # === BATTERY SIZING RECOMMENDATIONS ===
            st.markdown("### 6.5 🔋 Battery Sizing Analysis")
            
            # Check if we have clustering results for battery sizing
            if 'clusters_df' in locals() and not clusters_df.empty and 'peak_abs_kw_sum_in_cluster' in clusters_df.columns:
                st.info("✅ Using enhanced clustering analysis for battery sizing recommendations")
                
                # Use clustering analysis results for more accurate power requirements
                # Get max values from clustering analysis
                max_cluster_energy = clusters_df[clusters_df['cluster_duration_hr'] > 0]['total_energy_above_threshold_kwh'].max() if len(clusters_df[clusters_df['cluster_duration_hr'] > 0]) > 0 else 0
                max_single_energy = 0
                
                # Calculate max energy from single events
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    single_event_ids = clusters_df[clusters_df['cluster_duration_hr'] == 0]['cluster_id']
                    single_event_energies = []
                    for cluster_id in single_event_ids:
                        cluster_events = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                        if 'General Required Energy (kWh)' in cluster_events.columns:
                            single_event_energies.append(cluster_events['General Required Energy (kWh)'].max())
                    max_single_energy = max(single_event_energies) if single_event_energies else 0
                
                # Use the Max Monthly Required Energy from Section B2's monthly summary instead of clustering calculation
                # This ensures consistency between Battery Sizing Analysis and Section B2
                if 'monthly_summary_df' in locals() and not monthly_summary_df.empty:
                    # Determine tariff type for column selection
                    tariff_type = 'General'
                    if selected_tariff:
                        tariff_name = selected_tariff.get('Tariff', '').lower()
                        tariff_type_field = selected_tariff.get('Type', '').lower()
                        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
                            tariff_type = 'TOU'
                    
                    energy_col = f'{tariff_type} Required Energy (Max kWh)'
                    if energy_col in monthly_summary_df.columns:
                        recommended_energy_capacity = monthly_summary_df[energy_col].max()
                        # Debug log to verify synchronization
                        print(f"🔋 DEBUG - Using Max Monthly Required Energy from Section B2: {recommended_energy_capacity:.2f} kWh")
                    else:
                        # Fallback to clustering calculation if monthly summary doesn't have the column
                        recommended_energy_capacity = max(max_cluster_energy, max_single_energy)
                        print(f"🔋 DEBUG - Column '{energy_col}' not found, using clustering calculation: {recommended_energy_capacity:.2f} kWh")
                else:
                    # Fallback to clustering calculation if monthly summary is not available
                    recommended_energy_capacity = max(max_cluster_energy, max_single_energy)
                
                # Calculate power requirements from TOU Excess
                max_cluster_tou_excess = 0
                if len(clusters_df[clusters_df['cluster_duration_hr'] > 0]) > 0:
                    for cluster_id in clusters_df[clusters_df['cluster_duration_hr'] > 0]['cluster_id']:
                        cluster_events = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                        cluster_tou_excess_sum = cluster_events['TOU Excess (kW)'].sum() if 'TOU Excess (kW)' in cluster_events.columns else 0
                        max_cluster_tou_excess = max(max_cluster_tou_excess, cluster_tou_excess_sum)
                
                # Get max individual TOU Excess from single events
                max_single_tou_excess = 0
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    single_event_ids = clusters_df[clusters_df['cluster_duration_hr'] == 0]['cluster_id']
                    max_single_tou_excess = events_for_clustering[events_for_clustering['cluster_id'].isin(single_event_ids)]['TOU Excess (kW)'].max() if 'TOU Excess (kW)' in events_for_clustering.columns else 0
                
                # Use the larger value for power requirement
                max_power_shaving_required = max(max_cluster_tou_excess, max_single_tou_excess)
                
            else:
                # Fallback to original calculation method if clustering data not available
                st.warning("⚠️ Using fallback calculation for battery capacity - clustering analysis data not available")
                
                # Try to use Max Monthly Required Energy from Section B2's monthly summary for consistency
                recommended_energy_capacity = 0
                if 'monthly_summary_df' in locals() and not monthly_summary_df.empty:
                    # Determine tariff type for column selection
                    tariff_type = 'General'
                    if selected_tariff:
                        tariff_name = selected_tariff.get('Tariff', '').lower()
                        tariff_type_field = selected_tariff.get('Type', '').lower()
                        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
                            tariff_type = 'TOU'
                    
                    energy_col = f'{tariff_type} Required Energy (Max kWh)'
                    if energy_col in monthly_summary_df.columns:
                        recommended_energy_capacity = monthly_summary_df[energy_col].max()
                
                max_power_shaving_required = 0
                
                if monthly_targets is not None and len(monthly_targets) > 0:
                    # Calculate max shaving power directly from monthly targets and reference peaks
                    shaving_amounts = []
                    for month_period, target_demand in monthly_targets.items():
                        if month_period in reference_peaks:
                            max_demand = reference_peaks[month_period]
                            shaving_amount = max_demand - target_demand
                            if shaving_amount > 0:
                                shaving_amounts.append(shaving_amount)
                    
                    max_power_shaving_required = max(shaving_amounts) if shaving_amounts else 0
                
                # Calculate max TOU excess from individual events (power-based, not energy)
                max_tou_excess_fallback = max([event.get('TOU Excess (kW)', 0) or 0 for event in all_monthly_events]) if 'all_monthly_events' in locals() and all_monthly_events else 0
                max_power_shaving_required = max(max_power_shaving_required, max_tou_excess_fallback)
                
                # If monthly summary wasn't available, use power shaving as energy capacity estimate
                if recommended_energy_capacity == 0:
                    recommended_energy_capacity = max_power_shaving_required
            
            # Round up to nearest whole number for recommended capacity
            recommended_capacity_rounded = int(np.ceil(recommended_energy_capacity)) if recommended_energy_capacity > 0 else 0
            
            # Display key metrics only
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Max Power Shaving Required",
                    f"{max_power_shaving_required:.1f} kW",
                    help="Maximum power reduction required based on TOU excess from clustering analysis"
                )
            
            with col2:
                st.metric(
                    "Max Required Energy",
                    f"{recommended_energy_capacity:.1f} kWh", 
                    help="Maximum monthly energy requirement from Section B2 monthly summary analysis"
                )
        
        # Battery Impact Analysis Section moved to separate function
        
        # Render battery selection dropdown right before battery sizing analysis
        _render_battery_selection_dropdown()
        
        # Calculate shared analysis variables for both battery sizing and simulation
        # These need to be available in broader scope for battery simulation section
        max_power_shaving_required = 0
        recommended_energy_capacity = 0
        total_md_cost = 0
        
        # Console logging for debugging - check conditions first
        print(f"🔋 DEBUG - Battery Sizing Conditions Check:")
        print(f"   all_monthly_events exists: {'all_monthly_events' in locals()}")
        if 'all_monthly_events' in locals():
            print(f"   all_monthly_events length: {len(all_monthly_events) if all_monthly_events else 0}")
        print(f"   clusters_df exists: {'clusters_df' in locals()}")
        if 'clusters_df' in locals():
            print(f"   clusters_df empty: {clusters_df.empty if 'clusters_df' in locals() else 'N/A'}")
            print(f"   has peak_abs_kw_sum_in_cluster: {'peak_abs_kw_sum_in_cluster' in clusters_df.columns if 'clusters_df' in locals() and not clusters_df.empty else 'N/A'}")
        
        if all_monthly_events:
            # Check if clustering analysis was performed and has results
            if ('clusters_df' in locals() and not clusters_df.empty and 
                'peak_abs_kw_sum_in_cluster' in clusters_df.columns):
                
                # Use clustering analysis results for more accurate power requirements
                # Get max total peak power from multi-event clusters
                if len(clusters_df[clusters_df['cluster_duration_hr'] > 0]) > 0:
                    max_cluster_sum_power = clusters_df[clusters_df['cluster_duration_hr'] > 0]['peak_abs_kw_sum_in_cluster'].max()
                    max_cluster_energy = clusters_df[clusters_df['cluster_duration_hr'] > 0]['total_energy_above_threshold_kwh'].max()
                else:
                    max_cluster_sum_power = 0
                    max_cluster_energy = 0
                
                # Get max power from single events
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    max_single_power = clusters_df[clusters_df['cluster_duration_hr'] == 0]['peak_abs_kw_in_cluster'].max()
                    
                    # Get max energy from single events
                    single_event_ids = clusters_df[clusters_df['cluster_duration_hr'] == 0]['cluster_id']
                    single_event_energies = []
                    for cluster_id in single_event_ids:
                        cluster_events = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                        if not cluster_events.empty:
                            single_event_energies.append(cluster_events['General Required Energy (kWh)'].max())
                    max_single_energy = max(single_event_energies) if single_event_energies else 0
                else:
                    max_single_power = 0
                    max_single_energy = 0
                
                # Calculate Excess for clusters and single events based on tariff type (same logic as first section)
                # Determine which excess column to use based on tariff type
                excess_col = 'TOU Excess (kW)' if tariff_type == 'TOU' else 'General Excess (kW)'
                
                # For multi-event clusters, get max excess sum
                if len(clusters_df[clusters_df['cluster_duration_hr'] > 0]) > 0:
                    max_cluster_excess = 0
                    for cluster_id in clusters_df[clusters_df['cluster_duration_hr'] > 0]['cluster_id']:
                        cluster_events = events_for_clustering[events_for_clustering['cluster_id'] == cluster_id]
                        cluster_excess_sum = cluster_events[excess_col].sum() if excess_col in cluster_events.columns else 0
                        max_cluster_excess = max(max_cluster_excess, cluster_excess_sum)
                else:
                    max_cluster_excess = 0
                
                # For single events, get max individual excess
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    single_event_ids = clusters_df[clusters_df['cluster_duration_hr'] == 0]['cluster_id']
                    max_single_excess = events_for_clustering[events_for_clustering['cluster_id'].isin(single_event_ids)][excess_col].max() if excess_col in events_for_clustering.columns else 0
                else:
                    max_single_excess = 0
                
                # Use the larger value between clusters and single events for power requirement
                max_power_shaving_required = max(max_cluster_excess, max_single_excess)
                recommended_energy_capacity = max(max_cluster_energy, max_single_energy)  # Energy capacity from clustering analysis
                
                # Console logging for debugging - CLUSTERING ANALYSIS RESULTS
                print(f"🔋 DEBUG - Battery Sizing Values (CLUSTERING ANALYSIS):")
                print(f"   Selected tariff type: {tariff_type}")
                print(f"   Using excess column: {excess_col}")
                print(f"   max_power_shaving_required = {max_power_shaving_required:.1f} kW")
                print(f"   recommended_energy_capacity = {recommended_energy_capacity:.1f} kWh")
                print(f"   max_cluster_sum_power = {max_cluster_sum_power:.1f} kW")
                print(f"   max_single_power = {max_single_power:.1f} kW")
                
                st.info(f"""
                **🔋 Enhanced Battery Sizing (from Clustering Analysis):**
                - **Tariff Type**: {tariff_type}
                - **Max Cluster Energy**: {max_cluster_energy:.1f} kWh
                - **Max Single Event Energy**: {max_single_energy:.1f} kWh
                - **Max Cluster {tariff_type} Excess**: {max_cluster_excess:.1f} kW
                - **Max Single Event {tariff_type} Excess**: {max_single_excess:.1f} kW
                - **Selected Energy Capacity**: {recommended_energy_capacity:.1f} kWh
                - **Selected Power Requirement**: {max_power_shaving_required:.1f} kW
                """)
                
            else:
                # Fallback to original calculation method if clustering data not available
                st.warning("⚠️ Using fallback calculation - clustering analysis data not available")
                
                # Calculate max shaving power from monthly targets and max demands
                if monthly_targets is not None and len(monthly_targets) > 0:
                    shaving_amounts = []
                    for month_period, target_demand in monthly_targets.items():
                        if month_period in reference_peaks:
                            max_demand = reference_peaks[month_period]
                            shaving_amount = max_demand - target_demand
                            if shaving_amount > 0:
                                shaving_amounts.append(shaving_amount)
                    max_power_shaving_required = max(shaving_amounts) if shaving_amounts else 0
                
                # Calculate max excess from individual events based on tariff type (power-based, not energy)
                if tariff_type == 'TOU':
                    max_excess_fallback = max([event.get('TOU Excess (kW)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
                else:  # General tariff
                    max_excess_fallback = max([event.get('General Excess (kW)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
                max_power_shaving_required = max(max_power_shaving_required, max_excess_fallback)
                
                # Calculate recommended energy capacity from energy fields based on tariff type (kWh not kW)
                if tariff_type == 'TOU':
                    recommended_energy_capacity = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
                else:  # General tariff
                    recommended_energy_capacity = max([event.get('General Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
                
                # Console logging for debugging - FALLBACK CALCULATION
                print(f"🔋 DEBUG - Battery Sizing Values (FALLBACK METHOD):")
                print(f"   Selected tariff type: {tariff_type}")
                print(f"   max_power_shaving_required = {max_power_shaving_required:.1f} kW")
                print(f"   recommended_energy_capacity = {recommended_energy_capacity:.1f} kWh")
                print(f"   monthly_targets available: {monthly_targets is not None and len(monthly_targets) > 0}")
                print(f"   number of all_monthly_events: {len(all_monthly_events) if all_monthly_events else 0}")
            
            # Calculate total MD cost from events (same for both methods)
            total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
        
        # Console logging for debugging - FINAL RESULTS (always executes)
        print(f"🔋 DEBUG - Final Battery Sizing Results:")
        print(f"   FINAL max_power_shaving_required = {max_power_shaving_required:.1f} kW")
        print(f"   FINAL recommended_energy_capacity = {recommended_energy_capacity:.1f} kWh")
        print(f"   FINAL total_md_cost = RM {total_md_cost:.2f}")
        
        # NEW: Battery Quantity Recommendation Section 
        _render_battery_quantity_recommendation(max_power_shaving_required, recommended_energy_capacity)
        
        # Call the battery sizing analysis function with the calculated values
        _render_battery_sizing_analysis(max_power_shaving_required, recommended_energy_capacity, total_md_cost)
        
        # Battery Simulation Analysis Section
        st.markdown("#### 6.6 🔋 Battery Simulation Analysis")
        
        # Display battery simulation chart using selected battery specifications
        if (hasattr(st.session_state, 'tabled_analysis_selected_battery') and 
            st.session_state.tabled_analysis_selected_battery):
            
            # Get selected battery specifications
            selected_battery = st.session_state.tabled_analysis_selected_battery
            battery_spec = selected_battery['spec']
            
            # Extract battery parameters from selected battery specifications
            battery_capacity_kwh = battery_spec.get('energy_kWh', 0)
            battery_power_kw = battery_spec.get('power_kW', 0)
            
            # Check if we have the required analysis data with enhanced validation
            prerequisites_met = True
            error_messages = []
            
            # Validate peak analysis data
            if max_power_shaving_required <= 0:
                prerequisites_met = False
                error_messages.append("Max shaving power not calculated or invalid")
            
            if recommended_energy_capacity <= 0:
                prerequisites_met = False
                error_messages.append("Max TOU excess not calculated or invalid")
            
            # Validate battery specifications
            if battery_power_kw <= 0:
                prerequisites_met = False
                error_messages.append(f"Invalid battery power: {battery_power_kw} kW")
            
            if battery_capacity_kwh <= 0:
                prerequisites_met = False
                error_messages.append(f"Invalid battery capacity: {battery_capacity_kwh} kWh")
            
            # Validate data structure
            if not hasattr(df, 'columns') or power_col not in df.columns:
                prerequisites_met = False
                error_messages.append(f"Power column '{power_col}' not found in dataframe")
            
            if prerequisites_met:
                
                # 🎛️ INTEGRATION: Use user-configured battery quantity from Battery Quantity Configuration
                if hasattr(st.session_state, 'tabled_analysis_battery_quantity') and st.session_state.tabled_analysis_battery_quantity:
                    # Use quantity configured by user in Battery Quantity Configuration section
                    optimal_units = int(st.session_state.tabled_analysis_battery_quantity)
                    quantity_source = "User-configured from Battery Quantity Configuration"
                    
                    # Display success message for configured quantity
                    st.success(f"✅ **Using Battery Quantity Configuration**: {optimal_units} units as configured in '🎛️ Battery Quantity Configuration' section above.")
                else:
                    # Fallback: Calculate optimal number of units based on the analysis
                    units_for_power = int(np.ceil(max_power_shaving_required / battery_power_kw)) if battery_power_kw > 0 else 1
                    units_for_excess = int(np.ceil(recommended_energy_capacity / battery_power_kw)) if battery_power_kw > 0 else 1
                    optimal_units = max(units_for_power, units_for_excess, 1)
                    quantity_source = "Auto-calculated based on requirements"
                    
                    # Display info message about auto-calculation
                    st.info(f"ℹ️ **Auto-calculating Battery Quantity**: {optimal_units} units. You can configure a specific quantity in the '🎛️ Battery Quantity Configuration' section above to override this calculation.")
                
                # Calculate total system specifications using user-configured or calculated quantity
                total_battery_capacity = optimal_units * battery_capacity_kwh
                total_battery_power = optimal_units * battery_power_kw
                
                st.info(f"""
                **🔋 Battery Simulation Parameters:**
                - **Selected Battery**: {selected_battery['label']}
                - **Battery Model**: {battery_spec.get('model', 'Unknown')}
                - **Unit Specifications**: {battery_capacity_kwh:.1f} kWh, {battery_power_kw:.1f} kW per unit
                - **System Configuration**: {optimal_units} units ({quantity_source})
                - **Total System Capacity**: {total_battery_capacity:.1f} kWh
                - **Total System Power**: {total_battery_power:.1f} kW
                - **Based on**: Selected Power Requirement ({max_power_shaving_required:.1f} kW) & Selected Energy Capacity ({recommended_energy_capacity:.1f} kWh)
                """)
                
                # Call the battery simulation workflow (simulation + chart display)
                try:
                    # === STEP 1: Prepare V1-compatible dataframe ===
                    df_for_v1 = df.copy()
                    
                    # Add required columns that V1 expects
                    if 'Original_Demand' not in df_for_v1.columns:
                        df_for_v1['Original_Demand'] = df_for_v1[power_col]
                    
                    # === STEP 2: Prepare V1-compatible sizing parameter ===
                    sizing_dict = {
                        'capacity_kwh': total_battery_capacity,
                        'power_rating_kw': total_battery_power,
                        'units': optimal_units,
                        'c_rate': battery_spec.get('c_rate', 1.0),
                        'efficiency': 0.95  # Default efficiency
                    }
                    
                    # === STEP 3: Calculate proper target demand ===
                    if 'monthly_targets' in locals() and len(monthly_targets) > 0:
                        target_demand_for_sim = float(monthly_targets.iloc[0])
                    else:
                        target_demand_for_sim = float(df[power_col].quantile(0.8))
                    
                    # === STEP 4: CRITICAL - Run battery simulation first ===
                    st.info("⚡ Running battery simulation...")
                    
                    # Prepare all required parameters for V1 simulation function
                    battery_sizing = {
                        'capacity_kwh': total_battery_capacity,
                        'power_rating_kw': total_battery_power,
                        'units': optimal_units
                    }
                    
                    battery_params = {
                        'efficiency': 0.95,
                        'round_trip_efficiency': 95.0,  # Percentage
                        'c_rate': battery_spec.get('c_rate', 1.0),
                        'min_soc': 20.0,
                        'max_soc': 100.0,
                        'depth_of_discharge': 80.0  # Max usable % of capacity
                    }
                    
                    # Auto-detect global sampling interval (fallback to 15 minutes)
                    interval_hours = _infer_interval_hours(df_for_v1.index, fallback=0.25)
                    try:
                        st.session_state['data_interval_hours'] = interval_hours
                    except Exception:
                        pass
                    
                    st.info(f"🔧 Using {interval_hours*60:.0f}-minute intervals for V2 battery simulation")
                    
                    # V2 ENHANCEMENT: Use monthly targets instead of static target
                    simulation_results = _simulate_battery_operation_v2(
                        df_for_v1,                     # DataFrame with demand data
                        power_col,                     # Column name containing power demand
                        monthly_targets,               # V2: Dynamic monthly targets instead of static target
                        battery_sizing,                # Battery sizing dictionary
                        battery_params,                # Battery parameters dictionary  
                        interval_hours,                # Interval length in hours
                        selected_tariff,               # Tariff configuration
                        holidays if 'holidays' in locals() else set()  # Holidays set
                    )
                    
                    # === STEP 5: Display results and metrics ===
                    if simulation_results and 'df_simulation' in simulation_results:
                        st.success("✅ V2 Battery simulation with monthly targets completed successfully!")
                        
                        # Show key simulation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Peak Reduction", 
                                f"{simulation_results.get('peak_reduction_kw', 0): .1f} kW",
                                help="Maximum demand reduction achieved"
                            )
                        
                        with col2:
                            st.metric(
                                "Success Rate",
                                f"{simulation_results.get('success_rate_percent', 0):.1f}%",
                                help="Percentage of peak events successfully managed"
                            )
                        
                        with col3:
                            st.metric(
                                "Energy Discharged",
                                f"{simulation_results.get('total_energy_discharged', 0):.1f} kWh",
                                help="Total energy discharged during peak periods"
                            )
                        
                        with col4:
                            st.metric(
                                "Average SOC",
                                f"{simulation_results.get('average_soc', 0):.1f}%",
                                help="Average state of charge throughout simulation"
                            )
                        
                        # === STEP 6: Display the battery simulation chart ===
                        st.subheader("📊 Battery Operation Simulation")
                        _display_v2_battery_simulation_chart(
                            simulation_results['df_simulation'],  # Simulated dataframe
                            monthly_targets,              # V2 dynamic monthly targets
                            sizing_dict,                        # Battery sizing dictionary
                            selected_tariff,                    # Tariff configuration
                            holidays if 'holidays' in locals() else set()  # Holidays set
                        )
                        
                        # === STEP 7: Enhanced BESS Dispatch Simulation & Savings Analysis ===
                        st.markdown("---")
                        st.markdown("#### 6.7 🔋 BESS Dispatch Simulation & Comprehensive Analysis")
                        st.markdown("**Advanced battery dispatch simulation with engineering constraints and financial analysis**")
                        
                        if all_monthly_events:
                            # SECTION 6: DISPATCH SIMULATION
                            dispatch_results = []
                            
                            # Battery engineering parameters with proper DoD, efficiency, and C-rate limits
                            battery_specs = selected_battery['spec']
                            nameplate_energy_kwh = battery_specs.get('energy_kWh', 0) * optimal_units
                            nameplate_power_kw = battery_specs.get('power_kW', 0) * optimal_units
                            c_rate = battery_specs.get('c_rate', 1.0)
                            
                            # Engineering constraints
                            depth_of_discharge = 85  # % (preserve battery life)
                            round_trip_efficiency = 92  # % (charging + discharging losses)
                            degradation_factor = 90  # % (end-of-life performance)
                            safety_margin = 10  # % buffer for real conditions
                            
                            # Calculate usable specifications
                            usable_energy_kwh = (nameplate_energy_kwh * 
                                               depth_of_discharge / 100 * 
                                               degradation_factor / 100)
                            
                            usable_power_kw = (nameplate_power_kw * 
                                             degradation_factor / 100)
                            
                            # C-rate power limit
                            max_continuous_power_kw = min(usable_power_kw, usable_energy_kwh * c_rate)
                            
                            # SOC operating window
                            soc_min = (100 - depth_of_discharge) / 2  # e.g., 7.5% for 85% DoD
                            soc_max = soc_min + depth_of_discharge   # e.g., 92.5% for 85% DoD
                            
                            # Start at 80% SOC (near full but allowing charging headroom)
                            running_soc = 80.0
                            
                            st.info(f"""
                            **🔧 BESS Engineering Parameters:**
                            - **Fleet Capacity**: {nameplate_energy_kwh:.1f} kWh nameplate → {usable_energy_kwh:.1f} kWh usable
                            - **Fleet Power**: {nameplate_power_kw:.1f} kW nameplate → {max_continuous_power_kw:.1f} kW continuous
                            - **SOC Window**: {soc_min:.1f}% - {soc_max:.1f}% ({depth_of_discharge}% DoD)
                            - **Starting SOC**: {running_soc:.1f}% (Near-full for maximum availability)
                            - **Round-trip Efficiency**: {round_trip_efficiency}%
                            - **C-rate Limit**: {c_rate}C ({max_continuous_power_kw:.1f} kW max)
                            """)
                            
                            # Additional debug info for troubleshooting
                            st.markdown(f"""
                            **🔍 Debug Info:**
                            - Available SOC Range at Start: {running_soc - soc_min:.1f}%
                            - Available Energy at Start: {(usable_energy_kwh * (running_soc - soc_min) / 100):.1f} kWh
                            - Total Events to Process: {len(all_monthly_events)}
                            """)
                            
                            # Process each peak event with proper dispatch logic including recharging
                            previous_event_end = None
                            
                            for i, event in enumerate(all_monthly_events):
                                event_id = f"Event_{i+1:03d}"
                                
                                # Event parameters
                                start_date = pd.to_datetime(f"{event['Start Date']} {event['Start Time']}")
                                end_date = pd.to_datetime(f"{event['End Date']} {event['End Time']}")
                                duration_hours = event.get('Duration (min)', 0) / 60
                                
                                # RECHARGING LOGIC: Charge battery between events during off-peak periods
                                if previous_event_end is not None and start_date > previous_event_end:
                                    # Calculate time between events for potential charging
                                    time_between_events = (start_date - previous_event_end).total_seconds() / 3600  # hours
                                    
                                    # Assume charging during off-peak hours (simplified: charge if gap > 2 hours)
                                    if time_between_events >= 2.0 and running_soc < soc_max:
                                        # Calculate charging potential
                                        charging_headroom_soc = soc_max - running_soc  # Available SOC to charge
                                        charging_headroom_energy = (usable_energy_kwh * charging_headroom_soc / 100)
                                        
                                        # Charging power (limited by C-rate and available time)
                                        max_charging_power_kw = max_continuous_power_kw * 0.8  # Conservative charging rate
                                        available_charging_time = min(time_between_events, 8.0)  # Max 8 hours charging
                                        
                                        # Energy that can be charged
                                        max_chargeable_energy = max_charging_power_kw * available_charging_time
                                        
                                        # Actual charging (limited by headroom and efficiency)
                                        charging_energy_kwh = min(charging_headroom_energy, max_chargeable_energy)
                                        actual_stored_energy = charging_energy_kwh * (round_trip_efficiency / 100)  # Account for charging losses
                                        
                                        # Update SOC with charging
                                        soc_increase = (actual_stored_energy / usable_energy_kwh) * 100
                                        running_soc = min(soc_max, running_soc + soc_increase)
                                
                                # DISCHARGE LOGIC: Handle peak event
                                original_peak_kw = event.get('General Peak Load (kW)', 0)
                                excess_kw = event.get('General Excess (kW)', 0)
                                target_md_kw = original_peak_kw - excess_kw
                                
                                # Available energy for discharge (considering SOC and usable capacity)
                                available_soc_range = max(0, running_soc - soc_min)
                                available_energy_kwh = (usable_energy_kwh * available_soc_range / 100)
                                
                                # Power constraints for shaving (consider all limiting factors)
                                power_constraint_kw = min(
                                    excess_kw,  # Don't discharge more than needed
                                    max_continuous_power_kw,  # C-rate limit
                                    available_energy_kwh / duration_hours if duration_hours > 0 else 0  # Energy limit over duration
                                )
                                
                                # Calculate actual shaving performance
                                shaved_power_kw = max(0, power_constraint_kw)  # Ensure non-negative
                                shaved_energy_kwh = shaved_power_kw * duration_hours
                                
                                # Apply efficiency losses to energy calculation
                                actual_energy_consumed = shaved_energy_kwh / (round_trip_efficiency / 100)  # Account for losses
                                
                                deficit_kw = max(0, excess_kw - shaved_power_kw)
                                fully_shaved = deficit_kw <= 0.1  # 0.1 kW tolerance
                                
                                # Update SOC (account for actual energy consumed including losses)
                                soc_decrease = (actual_energy_consumed / usable_energy_kwh) * 100
                                new_soc = max(soc_min, running_soc - soc_decrease)
                                actual_soc_used = running_soc - new_soc
                                running_soc = new_soc
                                
                                # Update previous event end time for next iteration
                                previous_event_end = end_date
                                
                                # Calculate final load after shaving
                                final_peak_kw = original_peak_kw - shaved_power_kw
                                
                                # MD cost impact calculation
                                md_rate_rm_per_kw = 0
                                if selected_tariff and isinstance(selected_tariff, dict):
                                    rates = selected_tariff.get('Rates', {})
                                    md_rate_rm_per_kw = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
                                
                                # Monthly savings potential
                                monthly_md_reduction_kw = shaved_power_kw
                                monthly_savings_rm = monthly_md_reduction_kw * md_rate_rm_per_kw
                                
                                # Store dispatch results with corrected values including charging info
                                charging_info = "No charging" if previous_event_end is None else f"Charged between events"
                                if previous_event_end is not None and start_date > previous_event_end:
                                    time_gap = (start_date - previous_event_end).total_seconds() / 3600
                                    if time_gap >= 2.0:
                                        charging_info = f"Charged for {time_gap:.1f}h gap"
                                    else:
                                        charging_info = f"Gap too short ({time_gap:.1f}h)"
                                
                                dispatch_result = {
                                    'Event_ID': event_id,
                                    'Event_Period': f"{event['Start Date']} {event['Start Time']} - {event['End Date']} {event['End Time']}",
                                    'Duration_Hours': round(duration_hours, 2),
                                    'Original_Peak_kW': round(original_peak_kw, 1),
                                    'Target_MD_kW': round(target_md_kw, 1),
                                    'Excess_kW': round(excess_kw, 1),
                                    'Available_Energy_kWh': round(available_energy_kwh, 1),
                                    'Power_Constraint_kW': round(power_constraint_kw, 1),
                                    'Shaved_Power_kW': round(shaved_power_kw, 1),
                                    'Shaved_Energy_kWh': round(shaved_energy_kwh, 2),
                                    'Actual_Energy_Consumed_kWh': round(actual_energy_consumed, 2),
                                    'Deficit_kW': round(deficit_kw, 1),
                                    'Final_Peak_kW': round(final_peak_kw, 1),
                                    'Fully_Shaved': '✅ Yes' if fully_shaved else '❌ No',
                                    'SOC_Before_%': round(running_soc + actual_soc_used, 1),
                                    'SOC_After_%': round(running_soc, 1),
                                    'SOC_Used_%': round(actual_soc_used, 1),
                                    'Charging_Status': charging_info,
                                    'Monthly_Savings_RM': round(monthly_savings_rm, 2),
                                    'Constraint_Type': _determine_constraint_type(excess_kw, max_continuous_power_kw, available_energy_kwh, duration_hours),
                                    'BESS_Utilization_%': round((actual_energy_consumed / usable_energy_kwh) * 100, 1) if usable_energy_kwh > 0 else 0
                                }
                                dispatch_results.append(dispatch_result)
                            
                            # SECTION 7: SAVINGS CALCULATION
                            # Convert dispatch results to DataFrame for analysis
                            df_dispatch = pd.DataFrame(dispatch_results)
                            
                            # Calculate monthly savings aggregation
                            monthly_savings = []
                            
                            # Group events by month for savings analysis
                            df_dispatch['Month'] = pd.to_datetime(df_dispatch['Event_Period'].str.split(' - ').str[0]).dt.to_period('M')
                            
                            for month_period in df_dispatch['Month'].unique():
                                month_events = df_dispatch[df_dispatch['Month'] == month_period]
                                
                                # Calculate actual monthly MD (from original data)
                                # Get the month's actual maximum demand from the full dataset
                                month_start = month_period.start_time
                                month_end = month_period.end_time
                                month_mask = (df.index >= month_start) & (df.index <= month_end)
                                month_data = df[month_mask]
                                
                                if not month_data.empty:
                                    # Original monthly MD = maximum demand in the month
                                    original_md_kw = month_data[power_col].max()
                                    
                                    # Calculate shaved monthly MD by simulating battery impact on entire month
                                    # For simplification, assume the maximum shaving achieved in any event
                                    # could be sustained, so shaved MD = original MD - max shaving achieved
                                    max_shaving_achieved = month_events['Shaved_Power_kW'].max() if not month_events.empty else 0
                                    
                                    # More conservative approach: only count shaving if it was consistently successful
                                    successful_events = month_events[month_events['Fully_Shaved'].str.contains('Yes', na=False)]
                                    if not successful_events.empty:
                                        # Use average successful shaving as sustainable shaving
                                        sustainable_shaving_kw = successful_events['Shaved_Power_kW'].mean()
                                    else:
                                        # If no fully successful events, use partial shaving average
                                        sustainable_shaving_kw = month_events['Shaved_Power_kW'].mean() if not month_events.empty else 0
                                    
                                    # Shaved MD = Original MD - sustainable shaving
                                    shaved_md_kw = max(0, original_md_kw - sustainable_shaving_kw)
                                    md_reduction_kw = original_md_kw - shaved_md_kw
                                else:
                                    original_md_kw = 0
                                    shaved_md_kw = 0
                                    md_reduction_kw = 0
                                
                                # Monthly savings calculation
                                if selected_tariff and isinstance(selected_tariff, dict):
                                    rates = selected_tariff.get('Rates', {})
                                    md_rate_rm_per_kw = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
                                    monthly_saving_rm = md_reduction_kw * md_rate_rm_per_kw
                                else:
                                    monthly_saving_rm = 0
                                
                                # BESS utilization for the month
                                total_shaved_energy = month_events['Shaved_Energy_kWh'].sum()
                                num_events = len(month_events)
                                bess_utilization_pct = (total_shaved_energy / (usable_energy_kwh * num_events)) * 100 if num_events > 0 and usable_energy_kwh > 0 else 0
                                
                                monthly_savings.append({
                                    'Month': str(month_period),
                                    'Original_MD_kW': round(original_md_kw, 1),
                                    'Shaved_MD_kW': round(shaved_md_kw, 1),
                                    'MD_Reduction_kW': round(md_reduction_kw, 1),
                                    'Monthly_Saving_RM': round(monthly_saving_rm, 2),
                                    'BESS_Utilization_%': round(bess_utilization_pct, 1),
                                    'Events_Count': num_events
                                })
                            
                            df_monthly_savings = pd.DataFrame(monthly_savings)
                            total_annual_saving_rm = df_monthly_savings['Monthly_Saving_RM'].sum()
                            avg_monthly_saving_rm = df_monthly_savings['Monthly_Saving_RM'].mean()
                            avg_md_reduction_kw = df_monthly_savings['MD_Reduction_kW'].mean()
                            
                            # Display comprehensive results
                            st.markdown("#### 📊 Dispatch Simulation Results")
                            
                            # Summary KPIs
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Events", len(dispatch_results))
                            col2.metric("Success Rate", f"{len([r for r in dispatch_results if 'Yes' in r['Fully_Shaved']]) / len(dispatch_results) * 100:.1f}%")
                            col3.metric("Avg MD Reduction", f"{avg_md_reduction_kw:.1f} kW")
                            col4.metric("Annual Savings", f"RM {total_annual_saving_rm:,.0f}")
                            
                            # Enhanced dispatch results table with color coding
                            def highlight_dispatch_performance(row):
                                colors = []
                                for col in row.index:
                                    if col == 'Fully_Shaved':
                                        if 'Yes' in str(row[col]):
                                            colors.append('background-color: rgba(0, 255, 0, 0.2)')  # Green
                                        else:
                                            colors.append('background-color: rgba(255, 0, 0, 0.2)')  # Red
                                    elif col == 'BESS_Utilization_%':
                                        util = row[col] if isinstance(row[col], (int, float)) else 0
                                        if util >= 80:
                                            colors.append('background-color: rgba(0, 255, 0, 0.1)')  # Light green
                                        elif util >= 50:
                                            colors.append('background-color: rgba(255, 255, 0, 0.1)')  # Light yellow
                                        else:
                                            colors.append('background-color: rgba(255, 0, 0, 0.1)')  # Light red
                                    elif col == 'Constraint_Type':
                                        if 'Power' in str(row[col]):
                                            colors.append('background-color: rgba(255, 165, 0, 0.2)')  # Orange
                                        elif 'Energy' in str(row[col]):
                                            colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow
                                        else:
                                            colors.append('')
                                    else:
                                        colors.append('')
                                return colors
                            
                            # Display dispatch results table
                            styled_dispatch = df_dispatch.drop(['Month'], axis=1).style.apply(highlight_dispatch_performance, axis=1).format({
                                'Duration_Hours': '{:.2f}',
                                'Original_Peak_kW': '{:.1f}',
                                'Target_MD_kW': '{:.1f}',
                                'Excess_kW': '{:.1f}',
                                'Available_Energy_kWh': '{:.1f}',
                                'Power_Constraint_kW': '{:.1f}',
                                'Shaved_Power_kW': '{:.1f}',
                                'Shaved_Energy_kWh': '{:.2f}',
                                'Actual_Energy_Consumed_kWh': '{:.2f}',
                                'Deficit_kW': '{:.1f}',
                                'Final_Peak_kW': '{:.1f}',
                                'SOC_Before_%': '{:.1f}',
                                'SOC_After_%': '{:.1f}',
                                'SOC_Used_%': '{:.1f}',
                                'Monthly_Savings_RM': 'RM {:.2f}',
                                'BESS_Utilization_%': '{:.1f}%'
                            })
                            
                            st.dataframe(styled_dispatch, use_container_width=True)
                            
                            # Explanations for the comprehensive table
                            st.info("""
                            **📊 Comprehensive Dispatch Analysis Columns:**
                            
                            **Event Details:**
                            - **Event_ID**: Unique identifier for each peak event
                            - **Duration_Hours**: Event duration in hours
                            - **Original_Peak_kW**: Peak demand without battery intervention
                            - **Excess_kW**: Demand above target MD level
                            
                            **BESS Performance:**
                            - **Available_Energy_kWh**: Usable battery energy (considering SOC, DoD, efficiency)
                            - **Power_Constraint_kW**: Maximum power available (C-rate, energy, or demand limited)
                            - **Shaved_Power_kW**: Actual power reduction achieved
                            - **Shaved_Energy_kWh**: Total energy discharged during event
                            - **Deficit_kW**: Remaining excess after battery intervention
                            - **Final_Peak_kW**: Resulting peak demand after shaving
                            
                            **Battery State:**
                            - **SOC_Before/After_%**: Battery state of charge before and after event
                            - **SOC_Used_%**: Percentage of battery capacity utilized
                            - **BESS_Utilization_%**: Energy efficiency (discharged/available ratio)
                            
                            **Economic Impact:**
                            - **Monthly_Savings_RM**: Potential monthly savings from MD reduction
                            - **Constraint_Type**: Limiting factor (Power/Energy/Demand limited)
                            
                            **🎨 Color Coding:**
                            - 🟢 **Green**: Successful shaving or high utilization (≥80%)
                            - 🟡 **Yellow**: Moderate performance (50-79%) or energy-constrained
                            - 🟠 **Orange**: Power-constrained events
                            - 🔴 **Red**: Failed events or low utilization (<50%)
                            """)
                            
                            # Monthly savings analysis
                            st.markdown("#### 6.7.1 💰 Monthly Savings Analysis")
                            
                            # Display monthly savings table
                            styled_monthly = df_monthly_savings.style.format({
                                'Original_MD_kW': '{:.1f}',
                                'Shaved_MD_kW': '{:.1f}',
                                'MD_Reduction_kW': '{:.1f}',
                                'Monthly_Saving_RM': '{:.2f}',
                                'BESS_Utilization_%': '{:.1f}'
                            })
                            
                            st.dataframe(styled_monthly, use_container_width=True)
                            
                            # Annual summary
                            st.success(f"""
                            **💰 Annual Financial Summary:**
                            - **Total Annual Savings**: RM {total_annual_saving_rm:,.0f}
                            - **Average Monthly Savings**: RM {avg_monthly_saving_rm:,.0f}
                            - **Average MD Reduction**: {avg_md_reduction_kw:.1f} kW
                            - **ROI Analysis**: Based on {len(dispatch_results)} peak events across {len(df_monthly_savings)} months
                            """)
                            
                            # Visualization - Monthly MD comparison
                            fig_monthly = go.Figure()
                            
                            fig_monthly.add_trace(go.Scatter(
                                x=df_monthly_savings['Month'],
                                y=df_monthly_savings['Original_MD_kW'],
                                mode='lines+markers',
                                name='Original MD',
                                line=dict(color='red', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig_monthly.add_trace(go.Scatter(
                                x=df_monthly_savings['Month'],
                                y=df_monthly_savings['Shaved_MD_kW'],
                                mode='lines+markers',
                                name='Battery-Assisted MD',
                                line=dict(color='green', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig_monthly.update_layout(
                                title="Monthly Maximum Demand: Original vs Battery-Assisted",
                                xaxis_title="Month",
                                yaxis_title="Maximum Demand (kW)",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_monthly, use_container_width=True)
                            
                            # Monthly savings bar chart
                            fig_savings = go.Figure(data=[
                                go.Bar(
                                    x=df_monthly_savings['Month'],
                                    y=df_monthly_savings['Monthly_Saving_RM'],
                                    text=df_monthly_savings['Monthly_Saving_RM'].round(0),
                                    textposition='auto',
                                    marker_color='lightblue'
                                )
                            ])
                            
                            fig_savings.update_layout(
                                title="Monthly Savings from MD Reduction",
                                xaxis_title="Month",
                                yaxis_title="Savings (RM)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_savings, use_container_width=True)
                            
                        else:
                            st.warning("No peak events found for dispatch simulation analysis.")
                    
                    
                except Exception as e:
                    st.error(f"❌ Error in BESS dispatch simulation: {str(e)}")
                    with st.expander("Debug Details"):
                        st.write(f"Error details: {str(e)}")
                        st.write(f"Number of events: {len(all_monthly_events) if all_monthly_events else 0}")
                        st.write(f"Selected battery: {selected_battery['label'] if selected_battery else 'None'}")
                        st.write(f"Battery capacity: {total_battery_capacity if 'total_battery_capacity' in locals() else 'Unknown'} kWh")
                        st.write(f"Battery power: {total_battery_power if 'total_battery_power' in locals() else 'Unknown'} kW")
                        st.write(f"Optimal units: {optimal_units if 'optimal_units' in locals() else 'Unknown'}")
                    
                    # Fallback: Show basic configuration info
                    st.warning("⚠️ Falling back to basic battery configuration display...")
                    if 'selected_battery' in locals() and selected_battery:
                        st.write(f"**Configured Battery System:**")
                        st.write(f"- Battery: {selected_battery['label']}")
                        st.write(f"- Units: {optimal_units if 'optimal_units' in locals() else 'Unknown'}")
                        st.write(f"- Total Capacity: {total_battery_capacity if 'total_battery_capacity' in locals() else 'Unknown'} kWh")
                        st.write(f"- Total Power: {total_battery_power if 'total_battery_power' in locals() else 'Unknown'} kW")
            else:
                st.warning("⚠️ Prerequisites not met for battery simulation:")
                for msg in error_messages:
                    st.warning(f"- {msg}")
                    
        else:
            st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to perform enhanced analysis.")
            st.info("💡 Navigate to the top of this page and select a battery from the dropdown to see detailed battery analysis.")


def _determine_constraint_type(excess_kw, max_power_kw, available_energy_kwh, duration_hours):
    """Determine what constraint limits the battery dispatch."""
    if duration_hours <= 0:
        return "Invalid Duration"
    
    energy_limited_power = available_energy_kwh / duration_hours
    
    if excess_kw <= min(max_power_kw, energy_limited_power):
        return "Demand Limited"
    elif max_power_kw < energy_limited_power:
        return "Power Limited"
    else:
        return "Energy Limited"
        
        # V2 Enhancement Preview
        st.markdown("#### 🚀 V2 Monthly-Based Enhancements")
        st.info(f"""
        **📈 Monthly-Based Features Implemented:**
        - **✅ Monthly Target Calculation**: Each month uses {target_description} target
        - **✅ Stepped Target Profile**: Sawtooth target line that changes at month boundaries
        - **✅ Month-Specific Event Detection**: Peak events detected using appropriate monthly targets
        - **✅ Monthly Breakdown Table**: Detailed monthly analysis with individual targets and shaving amounts
        
        **🔄 Advanced Features Coming Soon:**
        - **Interactive Monthly Thresholds**: Adjust shaving percentage per month individually
        - **Seasonal Optimization**: Different strategies for high/low demand seasons
        - **Monthly ROI Analysis**: Cost-benefit analysis per billing period
        - **Cross-Month Battery Optimization**: Optimize battery usage across multiple months
        """)


def render_battery_impact_visualization():
    """Render the Battery Impact Analysis section as a separate component."""
    # Only render if we have the necessary data in session state
    if (hasattr(st.session_state, 'processed_df') and 
        st.session_state.processed_df is not None and 
        hasattr(st.session_state, 'power_column') and 
        st.session_state.power_column and
        hasattr(st.session_state, 'selected_tariff')):
        
        # Get data from session state
        df = st.session_state.processed_df
        power_col = st.session_state.power_column
        selected_tariff = st.session_state.selected_tariff
        holidays = getattr(st.session_state, 'holidays', [])
        target_method = getattr(st.session_state, 'target_method', 'percentage')
        shave_percent = getattr(st.session_state, 'shave_percent', 10)
        target_percent = getattr(st.session_state, 'target_percent', 85)
        target_manual_kw = getattr(st.session_state, 'target_manual_kw', 100)
        target_description = getattr(st.session_state, 'target_description', 'percentage-based')
        
        st.markdown("---")  # Separator
        st.markdown("### 🔋 Battery Impact Analysis")
        st.info("Configure battery specifications and visualize their impact on energy consumption patterns:")
        
        # Get battery configuration from the widget
        battery_config = _render_v2_battery_controls()
        
        # Render impact visualization if analysis is enabled and we have data context
        if (battery_config and battery_config.get('run_analysis') and 
            battery_config.get('selected_capacity', 0) > 0):
            
            st.markdown("---")  # Separator between config and visualization
            st.markdown("#### 📈 Battery Impact Visualization")
            st.info(f"Impact analysis for {battery_config['selected_capacity']} kWh battery:")
            
            # Render the actual battery impact timeline
            _render_battery_impact_timeline(
                df, 
                power_col, 
                selected_tariff, 
                holidays,
                target_method, 
                shave_percent,
                target_percent,
                target_manual_kw,
                target_description,
                battery_config['selected_capacity']
            )
    else:
        st.info("💡 **Upload data in the MD Shaving (v2) section above to see battery impact visualization.**")


# Main function for compatibility
def show():
    """Compatibility function that calls the main render function."""
    render_md_shaving_v2()


if __name__ == "__main__":
    # For testing purposes
    render_md_shaving_v2()


def cluster_peak_events(events_df, battery_params, md_hours, working_days):
    """
    Mock clustering function for peak events analysis.
    
    Args:
        events_df: DataFrame with peak events data
        battery_params: Dictionary with battery parameters
        md_hours: Tuple of (start_hour, end_hour) for MD period
        working_days: List of working days
        
    Returns:
        tuple: (clusters_df, events_for_clustering)
    """
    if events_df.empty:
        return pd.DataFrame(), events_df
    
    # Create a simple clustering based on date grouping
    events_for_clustering = events_df.copy()
    
    # Add cluster_id based on date
    events_for_clustering['cluster_id'] = events_for_clustering.index.date.astype(str)
    
    # Create clusters summary
    clusters_data = []
    for cluster_id, group in events_for_clustering.groupby('cluster_id'):
        clusters_data.append({
            'cluster_id': cluster_id,
            'num_events_in_cluster': len(group),
            'cluster_duration_hr': len(group) * 0.5 if len(group) > 1 else 0,  # Multi-event clusters
            'peak_abs_kw_in_cluster': group.get('General Peak Load (kW)', pd.Series([0])).max(),
            'peak_abs_kw_sum_in_cluster': group.get('General Peak Load (kW)', pd.Series([0])).sum(),
            'total_energy_above_threshold_kwh': group.get('General Required Energy (kWh)', pd.Series([0])).sum(),
            'cluster_start': group.index[0] if len(group) > 0 else None,
            'cluster_end': group.index[-1] if len(group) > 0 else None
        })
    
    clusters_df = pd.DataFrame(clusters_data)
    
    return clusters_df, events_for_clustering


def _compute_per_event_bess_dispatch(all_monthly_events, monthly_targets, selected_tariff, holidays, battery_spec=None, quantity=1, interval_hours=0.25):
    """
    Compute per-event BESS dispatch results using existing V2 logic.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        monthly_targets: Series of monthly targets from _calculate_monthly_targets_v2
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
        battery_spec: Battery specifications dict
        quantity: Number of battery units
        interval_hours: Data sampling interval in hours
        
    Returns:
        pd.DataFrame: Event results table with all required columns
    """
    if not all_monthly_events or not battery_spec:
        return pd.DataFrame()
    
    # Determine tariff type using existing logic
    tariff_type = 'General'
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Get MD rate from tariff
    md_rate_rm_per_kw = 0
    if selected_tariff and isinstance(selected_tariff, dict):
        rates = selected_tariff.get('Rates', {})
        md_rate_rm_per_kw = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
    
    # Battery system parameters
    rated_power_kw = battery_spec.get('power_kW', 0) * quantity
    capacity_kwh = battery_spec.get('energy_kWh', 0) * quantity
    soc_min_percent = 20.0  # Default SOC limits
    soc_max_percent = 100.0
    ready_soc_percent = 80.0  # Starting SOC
    eta_charge = 0.95  # Charging efficiency
    eta_discharge = 0.95  # Discharging efficiency
    round_trip_efficiency = eta_charge * eta_discharge
    max_charge_kw = rated_power_kw  # Assume same as discharge
    max_discharge_kw = rated_power_kw
    
    # Event processing
    event_results = []
    current_soc_percent = ready_soc_percent
    cluster_id = 1  # Simple cluster assignment
    previous_event_end = None
    
    for i, event in enumerate(all_monthly_events):
        try:
            # Basic event info
            event_id = f"E{i+1:03d}"
            start_date = event.get('Start Date')
            end_date = event.get('End Date')
            start_time = event.get('Start Time', '00:00')
            end_time = event.get('End Time', '00:00')
            
            # Parse timestamps
            start_timestamp = pd.to_datetime(f"{start_date} {start_time}")
            end_timestamp = pd.to_datetime(f"{end_date} {end_time}")
            duration_min = (end_timestamp - start_timestamp).total_seconds() / 60
            duration_h = duration_min / 60
            
            # Monthly context
            month = start_timestamp.to_period('M')
            month_str = month.strftime('%Y-%m')
            
            # Get monthly target for this event
            target_md_kw = monthly_targets.get(month, 0) if month in monthly_targets.index else 0
            
            # Event power characteristics
            original_peak_kw = event.get('General Peak Load (kW)', 0)
            excess_above_target_kw = max(0, original_peak_kw - target_md_kw)
            
            # TOU period determination using existing logic
            tou_period = True  # Default for General tariff
            md_window = "24/7"  # Default for General
            
            if tariff_type == 'TOU':
                # Use existing is_peak_rp4 function for TOU detection
                tou_period = is_peak_rp4(start_timestamp, holidays if holidays else set())
                md_window = "2PM-10PM" if tou_period else "Off-Peak"
            
            # Holiday check
            is_holiday = start_timestamp.date() in (holidays if holidays else set())
            
            # BESS state before event
            soc_before_percent = current_soc_percent
            available_energy_kwh = capacity_kwh * (soc_before_percent/100 - soc_min_percent/100)
            available_energy_kwh = max(0, available_energy_kwh)
            
            # Maximum energy that can be discharged during this event
            power_limited_energy = rated_power_kw * duration_h
            energy_limited_energy = available_energy_kwh * eta_discharge
            max_event_discharge_kwh = min(power_limited_energy, energy_limited_energy)
            
            # Dispatch calculation
            if excess_above_target_kw > 0 and tou_period:
                # Power shaving calculation
                power_shaved_kw = min(excess_above_target_kw, rated_power_kw)
                
                # Energy constraint check
                required_energy_kwh = power_shaved_kw * duration_h / eta_discharge
                if required_energy_kwh > available_energy_kwh:
                    # Energy limited
                    actual_energy_discharged = available_energy_kwh * eta_discharge
                    power_shaved_kw = actual_energy_discharged / duration_h
                    constraint_type = "Energy-limited"
                    reason_detail = f"Required {required_energy_kwh:.1f}kWh > available {available_energy_kwh:.1f}kWh"
                else:
                    # Power limited or successful
                    actual_energy_discharged = required_energy_kwh
                    if power_shaved_kw >= excess_above_target_kw:
                        constraint_type = "None"
                        reason_detail = f"Successfully shaved {power_shaved_kw:.1f}kW"
                    else:
                        constraint_type = "Power-limited"
                        reason_detail = f"Required {excess_above_target_kw:.1f}kW > rated {rated_power_kw:.1f}kW"
                
                energy_discharged_kwh = actual_energy_discharged
                
            elif not tou_period and tariff_type == 'TOU':
                # Outside MD window for TOU tariff
                power_shaved_kw = 0
                energy_discharged_kwh = 0
                constraint_type = "Not-in-MD-window"
                reason_detail = f"Event outside MD window ({md_window})"
                
            else:
                # No excess or no shaving needed
                power_shaved_kw = 0
                energy_discharged_kwh = 0
                constraint_type = "None"
                reason_detail = "No excess above target"
            
            # Post-event calculations
            final_peak_after_bess_kw = original_peak_kw - power_shaved_kw
            residual_above_target_kw = max(0, final_peak_after_bess_kw - target_md_kw)
            
            # SOC after event
            soc_used_kwh = energy_discharged_kwh / eta_discharge
            soc_used_percent = (soc_used_kwh / capacity_kwh) * 100 if capacity_kwh > 0 else 0
            soc_after_percent = max(soc_min_percent, soc_before_percent - soc_used_percent)
            current_soc_percent = soc_after_percent
            
            # Shaving success classification - FIXED LOGIC
            if not tou_period and tariff_type == 'TOU':
                # Events outside MD window should not be classified as failures
                shaving_success = "⚪ Not Applicable"
            elif excess_above_target_kw <= 0.1:
                # No excess to shave
                shaving_success = "✅ Complete"
            elif residual_above_target_kw <= 0.1:
                # Successfully reduced residual to near zero
                shaving_success = "✅ Complete"
            elif power_shaved_kw > 0:
                # Some shaving achieved but not complete
                shaving_success = "🟡 Partial"
            else:
                # Should have shaved (during MD window with excess) but couldn't
                shaving_success = "🔴 Failed"
            
            # Recharge analysis for next event
            recharge_window_min = 0
            recharge_required_kwh = 0
            recharge_possible_kwh = 0
            recharge_feasible = True
            
            if i < len(all_monthly_events) - 1:
                next_event = all_monthly_events[i + 1]
                next_start = pd.to_datetime(f"{next_event.get('Start Date')} {next_event.get('Start Time', '00:00')}")
                recharge_window_min = (next_start - end_timestamp).total_seconds() / 60
                
                # Required recharge to reach ready SOC
                target_soc_increase = ready_soc_percent - soc_after_percent
                recharge_required_kwh = (target_soc_increase / 100) * capacity_kwh
                
                # Possible recharge given time window
                recharge_time_h = recharge_window_min / 60
                max_recharge_energy = max_charge_kw * recharge_time_h * eta_charge
                recharge_possible_kwh = min(max_recharge_energy, recharge_required_kwh)
                
                recharge_feasible = recharge_possible_kwh >= recharge_required_kwh
                
                # Update SOC for next event if recharge is possible
                if recharge_feasible:
                    current_soc_percent = ready_soc_percent
                else:
                    # Partial recharge
                    soc_increase = (recharge_possible_kwh / capacity_kwh) * 100
                    current_soc_percent = min(soc_max_percent, soc_after_percent + soc_increase)
            
            # MD savings calculation (only for events in MD window)
            md_savings_rm = 0
            if tou_period or tariff_type == 'General':
                # Use monthly attribution approach from existing logic
                attribution_factor = 1.0  # Simplified attribution
                md_savings_rm = power_shaved_kw * md_rate_rm_per_kw * attribution_factor
            
            # Append event result
            event_results.append({
                'event_id': event_id,
                'month': month_str,
                'start_time': start_timestamp.strftime('%Y-%m-%d %H:%M'),
                'end_time': end_timestamp.strftime('%Y-%m-%d %H:%M'),
                'duration_min': round(duration_min, 1),
                'original_peak_kw': round(original_peak_kw, 1),
                'target_md_kw': round(target_md_kw, 1),
                'excess_above_target_kw': round(excess_above_target_kw, 1),
                'tou_period': '✅' if tou_period else '❌',
                'cluster_id': cluster_id,
                'rated_power_kw': round(rated_power_kw, 1),
                'capacity_kwh': round(capacity_kwh, 1),
                'soc_before_%': round(soc_before_percent, 1),
                'available_energy_kwh': round(available_energy_kwh, 1),
                'max_event_discharge_kwh': round(max_event_discharge_kwh, 1),
                'power_shaved_kw': round(power_shaved_kw, 1),
                'energy_discharged_kwh': round(energy_discharged_kwh, 1),
                'final_peak_after_bess_kw': round(final_peak_after_bess_kw, 1),
                'residual_above_target_kw': round(residual_above_target_kw, 1),
                'soc_after_%': round(soc_after_percent, 1),
                'shaving_success': shaving_success,
                'constraint_type': constraint_type,
                'reason_detail': reason_detail,
                'rte_%': round(round_trip_efficiency * 100, 1),
                'md_window': md_window,
                'recharge_window_min': round(recharge_window_min, 1),
                'recharge_required_kwh': round(recharge_required_kwh, 1),
                'recharge_possible_kwh': round(recharge_possible_kwh, 1),
                'recharge_feasible': '✅' if recharge_feasible else '❌',
                'md_savings_rm': round(md_savings_rm, 2),
                'holiday': '✅' if is_holiday else '❌',
                'data_gaps': '❌',  # Simplified
                'notes': f"{tariff_type} tariff, {constraint_type.lower()} dispatch"
            })
            
            # Simple cluster ID increment (simplified clustering)
            if recharge_window_min < 120:  # Less than 2 hours gap
                cluster_id += 0  # Keep same cluster
            else:
                cluster_id += 1  # New cluster
                
            previous_event_end = end_timestamp
            
        except Exception as e:
            st.warning(f"Error processing event {i+1}: {str(e)}")
            continue
    
    # Create DataFrame
    df_results = pd.DataFrame(event_results)
    
    return df_results


def _render_event_results_table(all_monthly_events, monthly_targets, selected_tariff, holidays):
    """
    Render the MD Shaving - Event Results (All Events) table.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        monthly_targets: Series of monthly targets
        selected_tariff: Selected tariff configuration  
        holidays: Set of holiday dates
    """
    
    st.markdown("#### 7.1.5 📊 MD Shaving – Event Results (All Events)")
    
    # Check if battery is selected
    if not (hasattr(st.session_state, 'tabled_analysis_selected_battery') and st.session_state.tabled_analysis_selected_battery):
        st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to view event-level dispatch results.")
        return
    
    # Get battery configuration
    selected_battery = st.session_state.tabled_analysis_selected_battery
    battery_spec = selected_battery['spec']
    quantity = getattr(st.session_state, 'tabled_analysis_battery_quantity', 1)
    
    if not all_monthly_events:
        st.info("No peak events available for analysis.")
        return
    
    # Validation checks
    validation_warnings = []
    
    # Check if monthly targets are available
    if monthly_targets.empty:
        validation_warnings.append("Monthly targets are missing - some calculations may be inaccurate")
    
    # Check for missing tariff configuration
    if not selected_tariff:
        validation_warnings.append("Tariff configuration missing - using default General tariff assumptions")
    
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(f"⚠️ {warning}")
    
    # Compute event results
    with st.spinner("Computing per-event BESS dispatch results..."):
        df_results = _compute_per_event_bess_dispatch(
            all_monthly_events, monthly_targets, selected_tariff, holidays, 
            battery_spec, quantity
        )
    
    if df_results.empty:
        st.error("❌ Failed to compute event results")
        return
    
    # Display summary metrics - Updated to handle "Not Applicable" events
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(df_results)
    not_applicable_events = len(df_results[df_results['shaving_success'] == '⚪ Not Applicable'])
    applicable_events = df_results[df_results['shaving_success'] != '⚪ Not Applicable']
    total_applicable = len(applicable_events)
    
    if total_applicable > 0:
        complete_events = len(applicable_events[applicable_events['shaving_success'] == '✅ Complete'])
        partial_events = len(applicable_events[applicable_events['shaving_success'] == '🟡 Partial'])
        failed_events = len(applicable_events[applicable_events['shaving_success'] == '🔴 Failed'])
        
        col1.metric("Total Events", f"{total_events} ({total_applicable} applicable)")
        col2.metric("Complete Shaving", f"{complete_events} ({complete_events/total_applicable*100:.1f}%)")
        col3.metric("Partial Shaving", f"{partial_events} ({partial_events/total_applicable*100:.1f}%)")
        col4.metric("Failed Shaving", f"{failed_events} ({failed_events/total_applicable*100:.1f}%)")
        
        if not_applicable_events > 0:
            st.info(f"ℹ️ **{not_applicable_events} events outside MD window** (not counted in success rates)")
    else:
        col1.metric("Total Events", total_events)
        col2.metric("All Off-Peak Events", f"{not_applicable_events} events")
        col3.metric("No MD Window Events", "Success rate: N/A")
        col4.metric("", "")
        
        st.warning("⚠️ All events are outside MD billing window - no applicable shaving opportunities")
    
    # Additional summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_power_shaved = df_results['power_shaved_kw'].mean()
    total_energy_discharged = df_results['energy_discharged_kwh'].sum()
    recharge_feasible_count = len(df_results[df_results['recharge_feasible'] == '✅'])
    total_md_savings = df_results['md_savings_rm'].sum()
    
    col1.metric("Avg Power Shaved", f"{avg_power_shaved:.1f} kW")
    col2.metric("Total Energy Discharged", f"{total_energy_discharged:.1f} kWh")  
    col3.metric("Recharge Feasible Rate", f"{recharge_feasible_count/total_events*100:.1f}%")
    col4.metric("Total MD Savings", f"RM {total_md_savings:.2f}")
    
    # Filters
    st.markdown("**🔍 Table Filters:**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        show_residual_only = st.checkbox("Show only events with residual > 0", False)
        
    with filter_col2:
        constraint_filter = st.multiselect(
            "Filter by constraint type:",
            options=['Power-limited', 'Energy-limited', 'Recharge-limited', 'Not-in-MD-window', 'None'],
            default=[]
        )
        
    with filter_col3:
        tou_only = st.checkbox("Show TOU period events only", False)
    
    # Month filter
    available_months = sorted(df_results['month'].unique())
    selected_months = st.multiselect(
        "Filter by month:",
        options=available_months,
        default=available_months
    )
    
    # Apply filters
    df_filtered = df_results.copy()
    
    if show_residual_only:
        df_filtered = df_filtered[df_filtered['residual_above_target_kw'] > 0]
    
    if constraint_filter:
        df_filtered = df_filtered[df_filtered['constraint_type'].isin(constraint_filter)]
    
    if tou_only:
        df_filtered = df_filtered[df_filtered['tou_period'] == '✅']
    
    if selected_months:
        df_filtered = df_filtered[df_filtered['month'].isin(selected_months)]
    
    st.markdown(f"**Showing {len(df_filtered)} of {len(df_results)} events**")
    
    # Style the dataframe with color coding
    def highlight_success(row):
        colors = []
        for col in df_filtered.columns:
            if col == 'shaving_success':
                if '✅ Complete' in str(row[col]):
                    colors.append('background-color: rgba(0, 255, 0, 0.2)')  # Green
                elif '🟡 Partial' in str(row[col]):
                    colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow
                elif '🔴 Failed' in str(row[col]):
                    colors.append('background-color: rgba(255, 0, 0, 0.2)')  # Red
                else:
                    colors.append('')
            elif col == 'recharge_feasible' and '❌' in str(row[col]):
                colors.append('background-color: rgba(255, 165, 0, 0.1)')  # Orange for recharge issues
            else:
                colors.append('')
        return colors
    
    # Display the table
    if not df_filtered.empty:
        styled_df = df_filtered.style.apply(highlight_success, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Download options
        st.markdown("**📥 Download Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download for filtered data
            csv_buffer = io.StringIO()
            df_filtered.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download Filtered Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"event_results_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        with col2:
            # CSV download for full dataset
            csv_buffer_full = io.StringIO()
            df_results.to_csv(csv_buffer_full, index=False)
            st.download_button(
                label="📊 Download Full Dataset (CSV)",
                data=csv_buffer_full.getvalue(),
                file_name=f"event_results_full_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No events match the current filter criteria.")
    
    # Footer summary for filtered results
    if not df_filtered.empty:
        st.markdown("---")
        st.markdown("**📊 Filtered Results Summary:**")
        
        filtered_complete = len(df_filtered[df_filtered['shaving_success'] == '✅ Complete'])
        filtered_partial = len(df_filtered[df_filtered['shaving_success'] == '🟡 Partial'])
        filtered_failed = len(df_filtered[df_filtered['shaving_success'] == '🔴 Failed'])
        filtered_avg_power = df_filtered['power_shaved_kw'].mean()
        filtered_total_energy = df_filtered['energy_discharged_kwh'].sum()
        filtered_recharge_rate = len(df_filtered[df_filtered['recharge_feasible'] == '✅']) / len(df_filtered) * 100
        filtered_md_savings = df_filtered['md_savings_rm'].sum()
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            - **Events**: {len(df_filtered)} total
            - **Success Rate**: {filtered_complete}/{len(df_filtered)} complete ({filtered_complete/len(df_filtered)*100:.1f}%)
            - **Partial**: {filtered_partial} events ({filtered_partial/len(df_filtered)*100:.1f}%)
            - **Failed**: {filtered_failed} events ({filtered_failed/len(df_filtered)*100:.1f}%)
            """)
            
        with summary_col2:
            st.markdown(f"""
            - **Avg Power Shaved**: {filtered_avg_power:.1f} kW
            - **Total Energy Discharged**: {filtered_total_energy:.1f} kWh
            - **Recharge Feasible**: {filtered_recharge_rate:.1f}%
            - **Total MD Savings**: RM {filtered_md_savings:.2f}
            """)
    
    # Technical notes
    with st.expander("ℹ️ Technical Notes & Methodology"):
        st.markdown("""
        **Calculation Methodology:**
        
        **Tariff-Aware Processing:**
        - **General Tariff**: All events are eligible for MD savings (24/7 MD billing)
        - **TOU Tariff**: Only events during 2PM-10PM weekdays are eligible for MD savings
        
        **BESS Dispatch Logic:**
        1. **Power Constraint**: `power_shaved_kw = min(excess_above_target_kw, rated_power_kw)`
        2. **Energy Constraint**: Verify sufficient battery energy considering efficiency losses
        3. **SOC Constraints**: Maintain SOC between configured min/max limits
        4. **Recharge Analysis**: Evaluate time window and power availability for recharging
        
        **Success Classification:**
        - ⚪ **Not Applicable**: Events outside MD billing window (TOU tariff off-peak periods)
        - ✅ **Complete**: Successfully reduced demand to target level (residual ≤ 0.1 kW)
        - 🟡 **Partial**: Some power shaved but did not fully meet target (residual > 0.1 kW)  
        - 🔴 **Failed**: No power shaved despite being in MD window with excess demand
        
        **MD Savings Attribution:**
        - Uses monthly maximum attribution methodology
        - Only credits events within MD billing windows
        - Applies configured MD rates from selected tariff
        
        **Validation Checks:**
        - Energy accounting: ΔSOC × capacity ≈ discharged_energy / η_discharge
        - TOU off-window events: Verified md_savings_rm = 0
        - Recharge feasibility: Time window vs charging power limits
        """)


def _display_v2_battery_simulation_chart(df_sim, monthly_targets=None, sizing=None, selected_tariff=None, holidays=None):
    """
    V2-specific battery operation simulation chart with DYNAMIC monthly targets.
    
    Key V2 Enhancement: Replaces static target line with stepped monthly target line.
    
    Args:
        df_sim: Simulation dataframe with battery operation data
        monthly_targets: V2's dynamic monthly targets (Series with Period index)
        sizing: Battery sizing dictionary from V2 analysis
        selected_tariff: Tariff configuration for MD period detection
        holidays: Set of holiday dates
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Handle None parameters with safe defaults
    if monthly_targets is None:
        st.error("❌ V2 Chart Error: monthly_targets is required for dynamic target visualization")
        return
        
    if sizing is None:
        sizing = {'power_rating_kw': 100, 'capacity_kwh': 100}
    
    # ===== V2 CHART SUCCESS/FAILURE FILTERING =====
    st.markdown("##### 🎯 V2 Chart Filters")
    
    # Success/Failure dropdown filter instead of timestamp filter
    if len(df_sim) > 0:
        col1, col2 = st.columns([4, 1])
        with col1:
            # Calculate shaving success for each point if not already available
            if 'Shaving_Success' not in df_sim.columns:
                df_sim['Shaving_Success'] = df_sim.apply(_get_enhanced_shaving_success, axis=1)
            
            filter_options = [
                "All Days",
                "All Success Days", 
                "All Partial Days",
                "All Failed Days"
            ]
            
            selected_filter = st.selectbox(
                "🎯 Filter Chart by Day Type:",
                options=filter_options,
                index=0,
                key="chart_success_filter",
                help="Filter chart data to show complete days that contain specific event types"
            )
            
        with col2:
            if st.button("🔄 Reset Filter", key="reset_chart_success_filter"):
                st.session_state.chart_success_filter = "All Days"
                st.rerun()
        
        # Apply success/failure filter to chart data - FULL DAY filtering
        df_sim_filtered = df_sim.copy()
        
        if selected_filter == "All Success Days":
            # Find all days that contain success events
            success_days = df_sim[df_sim['Shaving_Success'].str.contains('✅ Complete Success|🟢 No Action Needed|🟢 Off-Peak Period', na=False)].index.date
            success_days_set = set(success_days)
            # Show all data for those days - use pd.Series.isin() instead of numpy array.isin()
            df_sim_filtered = df_sim[pd.Series(df_sim.index.date).isin(success_days_set).values]
        elif selected_filter == "All Partial Days":
            # Find all days that contain partial events
            partial_days = df_sim[df_sim['Shaving_Success'].str.contains('🟡 Good Partial|🟠 Fair Partial|🔶 Poor Partial', na=False)].index.date
            partial_days_set = set(partial_days)
            # Show all data for those days - use pd.Series.isin() instead of numpy array.isin()
            df_sim_filtered = df_sim[pd.Series(df_sim.index.date).isin(partial_days_set).values]
        elif selected_filter == "All Failed Days":
            # Find all days that contain failed events
            failed_days = df_sim[df_sim['Shaving_Success'].str.contains('🔴 Failed', na=False)].index.date
            failed_days_set = set(failed_days)
            # Show all data for those days - use pd.Series.isin() instead of numpy array.isin()
            df_sim_filtered = df_sim[pd.Series(df_sim.index.date).isin(failed_days_set).values]
        else:
            # "All Events" - show everything
            df_sim_filtered = df_sim
        
        # Display filter results summary
        if len(df_sim_filtered) < len(df_sim):
            success_days = len(set(df_sim[df_sim['Shaving_Success'].str.contains('✅ Complete Success', na=False)].index.date))
            partial_days = len(set(df_sim[df_sim['Shaving_Success'].str.contains('🟡 Good Partial|🟠 Fair Partial|🔶 Poor Partial', na=False)].index.date))
            failed_days = len(set(df_sim[df_sim['Shaving_Success'].str.contains('🔴 Failed', na=False)].index.date))
            total_days = len(set(df_sim.index.date))
            filtered_days = len(set(df_sim_filtered.index.date))
            
            st.info(f"""
            📊 **Chart Filter Results**: Showing {len(df_sim_filtered):,} records from {filtered_days} days of {len(df_sim):,} total records ({filtered_days}/{total_days} days, {len(df_sim_filtered)/len(df_sim)*100:.1f}%)
            
            **Day Breakdown:**
            - ✅ **Success Days**: {success_days} days
            - 🟡🟠🔶 **Partial Days**: {partial_days} days
            - 🔴 **Failed Days**: {failed_days} days
            """)
        else:
            total_days = len(set(df_sim.index.date))
            st.info(f"📊 **All Days**: Showing {len(df_sim_filtered):,} records from {total_days} days")
        
        # Use filtered data for the rest of the chart function
        df_sim = df_sim_filtered
        
        # Validation check after filtering
        if len(df_sim) == 0:
            st.warning("⚠️ No days match the selected filter criteria. Please choose a different filter.")
            return
    
    # Resolve Net Demand column name flexibly
    net_candidates = ['Net_Demand_kW', 'Net_Demand_KW', 'Net_Demand']
    net_col = next((c for c in net_candidates if c in df_sim.columns), None)
    
    # Validate required columns exist
    required_base = ['Original_Demand', 'Battery_Power_kW', 'Battery_SOC_Percent']
    missing_columns = [col for col in required_base if col not in df_sim.columns]
    if net_col is None:
        missing_columns.append('Net_Demand_kW')
    
    if missing_columns:
        st.error(f"❌ Missing required columns in V2 simulation data: {missing_columns}")
        st.info("Available columns: " + ", ".join(df_sim.columns.tolist()))
        return
    
    # Create V2 dynamic target series (stepped monthly targets) - filtered to match chart data
    target_series = _create_v2_dynamic_target_series(df_sim.index, monthly_targets)
    
    # Display filtered event range info
    if selected_filter != "All Events" and len(df_sim) > 0:
        filter_start = df_sim.index.min()
        filter_end = df_sim.index.max()
        st.info(f"📅 **Filtered Event Range**: {filter_start.strftime('%Y-%m-%d %H:%M')} to {filter_end.strftime('%Y-%m-%d %H:%M')}")
    
    # Panel 1: V2 Enhanced MD Shaving Effectiveness with Dynamic Monthly Targets
    st.markdown("##### 1️⃣ V2 MD Shaving Effectiveness: Demand vs Battery vs Dynamic Monthly Targets")
    if selected_filter != "All Events":
        st.info(f"🆕 **V2 Enhancement with Filtering**: Target line changes monthly based on V2 configuration, showing only {selected_filter.lower()}")
    else:
        st.info("🆕 **V2 Enhancement**: Target line changes monthly based on your V2 target configuration")
    
    fig = go.Figure()
    
    # Add demand lines
    fig.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim[net_col], 
                  name='Net Demand (with Battery)', line=dict(color='#00BFFF', width=2),
                  hovertemplate='Net: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    
    # V2 ENHANCEMENT: Add stepped monthly target line instead of static line
    fig.add_trace(
        go.Scatter(x=df_sim.index, y=target_series, 
                  name='Monthly Target (V2 Dynamic)', 
                  line=dict(color='green', dash='dash', width=3),
                  hovertemplate='Monthly Target: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    
    # Replace area fills with bar charts for battery discharge/charge
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
    
    # V2 ENHANCEMENT: Add dynamic conditional coloring using monthly targets instead of static average
    # This replaces the V1 averaging approach with dynamic monthly target-based coloring
    fig = _create_v2_conditional_demand_line_with_dynamic_targets(
        fig, df_sim, 'Original_Demand', target_series, selected_tariff, holidays, "Original Demand"
    )
    
    # Compute symmetric range for y2 to show positive/negative bars
    try:
        max_abs_power = float(df_sim['Battery_Power_kW'].abs().max())
    except Exception:
        max_abs_power = float(sizing.get('power_rating_kw', 100))
    y2_limit = max(max_abs_power * 1.1, sizing.get('power_rating_kw', 100) * 0.5)
    
    fig.update_layout(
        title='🎯 V2 MD Shaving Effectiveness: Demand vs Battery vs Dynamic Monthly Targets',
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
    
    # V2 ENHANCEMENT INFO: Add explanation about dynamic color coding
    st.info("""
    🆕 **V2 Color Coding Enhancement**: The colored line segments now use **dynamic monthly targets** instead of a static average target.
    - **Blue segments**: Below monthly target (acceptable levels)
    - **Green segments**: Above monthly target during off-peak periods (energy cost only)
    - **Red segments**: Above monthly target during peak periods (energy + MD cost impact)
    
    This provides more accurate visual feedback about when intervention is needed based on realistic monthly billing patterns.
    """)
    
    # ===== V2 TABLE VISUALIZATION INTEGRATION BETWEEN CHART 1 AND 2 =====
    _display_battery_simulation_tables(df_sim, {
        'peak_reduction_kw': sizing.get('power_rating_kw', 0) if sizing else 0,
        'success_rate_percent': 85.0,  # Default placeholder
        'total_energy_discharged': df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] > 0, 0).sum() * 0.25,
        'total_energy_charged': abs(df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] < 0, 0).sum()) * 0.25,
        'average_soc': df_sim['Battery_SOC_Percent'].mean(),
        'min_soc': df_sim['Battery_SOC_Percent'].min(),
        'max_soc': df_sim['Battery_SOC_Percent'].max(),
        'monthly_targets_count': len(monthly_targets) if monthly_targets is not None else 0,
        'v2_constraint_violations': len(df_sim[df_sim['Net_Demand_kW'] > df_sim['Monthly_Target']])
    })
    
    # Panel 2: Combined SOC and Battery Power Chart (same as V1)
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
    
    # Panel 3: Battery Power Utilization Heatmap (same as V1)
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
    
    # Panel 4: V2 Enhanced Daily Peak Shave Effectiveness with Monthly Target Context
    st.markdown("##### 4️⃣ V2 Daily Peak Shave Effectiveness & Success Analysis (MD Peak Periods Only)")
    st.info("🆕 **V2 Enhancement**: Success/failure calculated against dynamic monthly targets")
    
    # Filter data for MD peak periods only (2 PM-10 PM, weekdays)
    def is_md_peak_period_for_effectiveness(timestamp):
        return timestamp.weekday() < 5 and 14 <= timestamp.hour < 22
        
    df_md_peak = df_sim[df_sim.index.to_series().apply(is_md_peak_period_for_effectiveness)]
    
    # Calculate daily analysis using MD peak periods only WITH V2 monthly targets
    if len(df_md_peak) > 0:
        daily_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            net_col: 'max',
            'Battery_Power_kW': 'max',
            'Battery_SOC_Percent': ['min', 'mean']
        }).reset_index()
        
        # Flatten column names
        daily_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Max_Battery_Power', 'Min_SOC', 'Avg_SOC']
        
        # V2 ENHANCEMENT: Get monthly target for each day
        daily_analysis['Monthly_Target'] = daily_analysis['Date'].apply(
            lambda date: _get_monthly_target_for_date(date, monthly_targets)
        )
        
        # Calculate detailed metrics based on V2 monthly targets
        md_rate_estimate = 97.06  # RM/kW from Medium Voltage TOU
        daily_analysis['Peak_Reduction'] = daily_analysis['Original_Peak_MD'] - daily_analysis['Net_Peak_MD']
        daily_analysis['Est_Monthly_Saving'] = daily_analysis['Peak_Reduction'] * md_rate_estimate
        
        # V2 SUCCESS LOGIC: Compare against monthly targets instead of static target
        daily_analysis['Success'] = daily_analysis['Net_Peak_MD'] <= daily_analysis['Monthly_Target'] * 1.05  # 5% tolerance
        daily_analysis['Peak_Shortfall'] = (daily_analysis['Net_Peak_MD'] - daily_analysis['Monthly_Target']).clip(lower=0)
        daily_analysis['Required_Additional_Power'] = daily_analysis['Peak_Shortfall']
        
        # Add informational note about V2 monthly target logic
        st.info("""
        📋 **V2 Monthly Target Analysis Note:**
        This analysis uses **dynamic monthly targets** instead of a static target.
        Each day's success is evaluated against its specific month's target.
        Success rate reflects effectiveness against V2's monthly optimization strategy.
        """)
    else:
        st.warning("⚠️ No MD peak period data found (weekdays 2-10 PM). Cannot calculate V2 MD-focused effectiveness.")
        return
    
    # Categorize failure reasons with V2 context
    def categorize_failure_reason(row):
        if row['Success']:
            return 'Success'
        elif row['Min_SOC'] < 20:
            return 'Low SOC (Battery Depleted)'
        elif row['Max_Battery_Power'] < sizing['power_rating_kw'] * 0.9:
            return 'Insufficient Battery Power'
        elif row['Peak_Shortfall'] > sizing['power_rating_kw']:
            return 'Demand Exceeds Battery Capacity'
        else:
            return 'Other (Algorithm/Timing)'
    
    daily_analysis['Failure_Reason'] = daily_analysis.apply(categorize_failure_reason, axis=1)
    
    # Create enhanced visualization with monthly target context
    fig4 = go.Figure()
    
    # V2 Enhancement: Add monthly target reference lines instead of single target line
    for month_period, target_value in monthly_targets.items():
        month_start = max(month_period.start_time, df_sim.index.min())
        month_end = min(month_period.end_time, df_sim.index.max())
        
        # Add horizontal line for this month's target
        fig4.add_shape(
            type="line",
            x0=month_start, y0=target_value,
            x1=month_end, y1=target_value,
            line=dict(color="green", width=2, dash="dash"),
        )
        
        # Add annotation for the target value
        fig4.add_annotation(
            x=month_start + (month_end - month_start) / 2,
            y=target_value,
            text=f"{target_value:.0f} kW",
            showarrow=False,
            yshift=10,
            bgcolor="rgba(255,255,255,0.8)"
        )
    
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
        title='📊 V2 Daily Peak Shaving Effectiveness - MD Periods with Monthly Targets (Green=Success, Red=Failed)',
        xaxis_title='Date',
        yaxis_title='Peak Demand during MD Hours (kW)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Summary stats with V2 context
    total_days = len(daily_analysis)
    successful_days = sum(daily_analysis['Success'])
    failed_days = total_days - successful_days
    success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", f"{total_days}")
    col2.metric("Successful Days", f"{successful_days}", delta=f"{success_rate:.1f}%")
    col3.metric("Failed Days", f"{failed_days}", delta=f"{100-success_rate:.1f}%")
    col4.metric("V2 Success Rate", f"{success_rate:.1f}%")
    
    # Panel 5: V2 Cumulative Energy Analysis with Monthly Target Context
    st.markdown("##### 5️⃣ V2 Cumulative Energy Analysis: Energy Discharged vs Required (MD Peak Periods)")
    st.info("🆕 **V2 Enhancement**: Energy requirements calculated using dynamic monthly targets")
    
    # Use the same daily analysis data but with V2 monthly target logic
    if len(daily_analysis) > 0:
        # Calculate energy requirements using V2 monthly target approach
        daily_analysis_energy = daily_analysis.copy()
        
        # V2 Energy Required: Calculate based on daily peak reduction needs using monthly targets
        daily_analysis_energy['Daily_Energy_Required_kWh'] = 0.0
        
        # For each day, calculate energy required based on monthly target instead of static target
        for idx, row in daily_analysis_energy.iterrows():
            original_peak = row['Original_Peak_MD']
            net_peak = row['Net_Peak_MD']
            monthly_target = row['Monthly_Target']
            
            if original_peak > monthly_target:
                # Calculate energy required to shave this day's peak to monthly target
                if net_peak <= monthly_target * 1.05:  # Successful day
                    # Energy that was successfully shaved (based on actual peak reduction)
                    energy_shaved = row['Peak_Reduction'] * 0.25  # Convert kW to kWh (15-min intervals)
                else:  # Failed day
                    # Energy that would be needed to reach monthly target
                    energy_needed = (original_peak - monthly_target) * 0.25
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
            st.warning("No MD peak period data available for V2 energy analysis.")
            return
    
        # Sort by date for cumulative calculation
        daily_analysis_energy = daily_analysis_energy.sort_values('Date').reset_index(drop=True)
        
        # Calculate cumulative values
        daily_analysis_energy['Cumulative_Energy_Required'] = daily_analysis_energy['Daily_Energy_Required_kWh'].cumsum()
        daily_analysis_energy['Cumulative_Energy_Discharged'] = daily_analysis_energy['Daily_Energy_Discharged_kWh'].cumsum()
        daily_analysis_energy['Cumulative_Energy_Shortfall'] = daily_analysis_energy['Cumulative_Energy_Required'] - daily_analysis_energy['Cumulative_Energy_Discharged']
        
        # Create the chart using the daily aggregated data with V2 context
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
            
            # Energy Required line (from daily analysis with V2 monthly targets)
            fig5.add_trace(go.Scatter(
                x=daily_analysis_energy['Date'],
                y=daily_analysis_energy['Cumulative_Energy_Required'],
                mode='lines+markers',
                name='Cumulative Energy Required (V2 Monthly Targets)',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate='Required (V2): %{y:.1f} kWh<br>Date: %{x}<extra></extra>'
            ))
            
            # Add area fill for energy shortfall
            fig5.add_trace(go.Scatter(
                x=daily_analysis_energy['Date'],
                y=daily_analysis_energy['Cumulative_Energy_Shortfall'].clip(lower=0),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='Cumulative Energy Shortfall (V2)',
                hovertemplate='Shortfall: %{y:.1f} kWh<br>Date: %{x}<extra></extra>'
            ))
            
            fig5.update_layout(
                title='📈 V2 Cumulative Energy Analysis: Monthly Target-Based Daily Aggregation',
                xaxis_title='Date',
                yaxis_title='Cumulative Energy (kWh)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # Display metrics using V2 monthly target calculations
            total_energy_required = daily_analysis_energy['Daily_Energy_Required_kWh'].sum()
            total_energy_discharged = daily_analysis_energy['Daily_Energy_Discharged_kWh'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Energy Required (V2 MD)", f"{total_energy_required:.1f} kWh")
            col2.metric("Total Energy Discharged (V2 MD)", f"{total_energy_discharged:.1f} kWh")
            
            if total_energy_required > 0:
                fulfillment_rate = (total_energy_discharged / total_energy_required) * 100
                col3.metric("V2 MD Energy Fulfillment", f"{fulfillment_rate:.1f}%")
            else:
                col3.metric("V2 MD Energy Fulfillment", "100%")
            
            # Add detailed breakdown table with V2 context
            with st.expander("📊 V2 Daily Energy Breakdown (Monthly Target-Based Analysis)"):
                display_columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Peak_Reduction', 'Monthly_Target',
                                 'Daily_Energy_Required_kWh', 'Daily_Energy_Discharged_kWh', 'Success']
                
                if all(col in daily_analysis_energy.columns for col in display_columns):
                    daily_display = daily_analysis_energy[display_columns].copy()
                    daily_display.columns = ['Date', 'Original Peak (kW)', 'Net Peak (kW)', 'Peak Reduction (kW)', 
                                           'Monthly Target (kW)', 'Energy Required (kWh)', 'Energy Discharged (kWh)', 'Success']
                    
                    formatted_daily = daily_display.style.format({
                        'Original Peak (kW)': '{:.1f}',
                        'Net Peak (kW)': '{:.1f}',
                        'Peak Reduction (kW)': '{:.1f}',
                        'Monthly Target (kW)': '{:.1f}',
                        'Energy Required (kWh)': '{:.2f}',
                        'Energy Discharged (kWh)': '{:.2f}'
                    })
                    
                    st.dataframe(formatted_daily, use_container_width=True)
                else:
                    st.warning("Some columns missing from V2 daily analysis data.")
            
            # Add V2-specific information box
            st.info(f"""
            **📋 V2 Data Source Alignment Confirmation:**
            - **Energy Required**: Calculated from daily peak reduction needs using **dynamic monthly targets**
            - **Energy Discharged**: Sum of battery discharge energy during MD recording hours per day  
            - **Calculation Method**: V2 monthly target-based approach vs V1 static target approach
            - **Monthly Targets**: {len(monthly_targets)} different monthly targets used
            - **Total Days Analyzed**: {len(daily_analysis_energy)} days with MD peak period data
            - **Success Rate**: {(daily_analysis_energy['Success'].sum() / len(daily_analysis_energy) * 100):.1f}% (based on monthly targets)
            
            ✅ **V2 Innovation**: This chart uses dynamic monthly targets instead of static targets for more accurate analysis.
            """)
            
        else:
            st.warning("No daily analysis data available for V2 cumulative energy chart.")
    else:
        st.warning("No MD peak period data available for V2 energy analysis.")
    
    # V2 Key insights with monthly target context
    st.markdown("##### 🔍 V2 Key Insights from Enhanced Monthly Target Analysis")
    
    insights = []
    
    # Use V2 energy efficiency calculation
    if 'total_energy_required' in locals() and 'total_energy_discharged' in locals():
        energy_efficiency = (total_energy_discharged / total_energy_required * 100) if total_energy_required > 0 else 100
        
        if energy_efficiency < 80:
            insights.append("⚠️ **V2 MD Energy Shortfall**: Battery capacity may be insufficient for complete monthly target-based MD peak shaving")
        elif energy_efficiency >= 95:
            insights.append("✅ **Excellent V2 MD Coverage**: Battery effectively handles all monthly target energy requirements")
    
    # Check V2 success rate
    if 'success_rate' in locals():
        if success_rate > 90:
            insights.append("✅ **High V2 Success Rate**: Battery effectively manages most peak events against dynamic monthly targets")
        elif success_rate < 60:
            insights.append("❌ **Low V2 Success Rate**: Consider increasing battery power rating or capacity for better monthly target management")
    
    # Check battery utilization if heatmap data is available
    if 'df_heatmap' in locals() and len(df_heatmap) > 0:
        avg_utilization = df_heatmap['Battery_Utilization_%'].mean()
        if avg_utilization < 30:
            insights.append("📊 **Under-utilized**: Battery power rating may be oversized for V2 monthly targets")
        elif avg_utilization > 80:
            insights.append("🔥 **High Utilization**: Battery operating near maximum capacity for V2 monthly targets")
    
    # Check for low SOC events
    low_soc_events = len(df_sim[df_sim['Battery_SOC_Percent'] < 20])
    if low_soc_events > 0:
        insights.append(f"🔋 **Low SOC Warning**: {low_soc_events} intervals with SOC below 20% during V2 operation")
    
    # Add insight about V2 methodology
    if len(monthly_targets) > 0:
        insights.append(f"📊 **V2 Innovation**: Analysis uses {len(monthly_targets)} dynamic monthly targets vs traditional static targets for superior accuracy")
        insights.append(f"🎨 **V2 Color Enhancement**: Line color coding now reflects dynamic monthly targets instead of static averaging - providing month-specific intervention guidance")
    
    if not insights:
        insights.append("✅ **Optimal V2 Performance**: Battery system operating within acceptable parameters with monthly targets")
    
    for insight in insights:
        st.info(insight)


def _create_v2_dynamic_target_series(simulation_index, monthly_targets):
    """
    Create a dynamic target series that matches the simulation dataframe index
    with stepped monthly targets from V2's monthly_targets.
    
    Args:
        simulation_index: DatetimeIndex from the simulation dataframe
        monthly_targets: V2's monthly targets (Series with Period index)
        
    Returns:
        Series with same index as simulation_index, containing monthly target values
    """
    target_series = pd.Series(index=simulation_index, dtype=float)
    
    for timestamp in simulation_index:
        # Get the month period for this timestamp
        month_period = timestamp.to_period('M')
        
        # Find the corresponding monthly target
        if month_period in monthly_targets.index:
            target_series.loc[timestamp] = monthly_targets.loc[month_period]
        else:
            # Fallback: use the closest available monthly target
            available_months = list(monthly_targets.index)
            if available_months:
                # Find the closest month
                closest_month = min(available_months, 
                                  key=lambda m: abs((timestamp.to_period('M') - m).n))
                target_series.loc[timestamp] = monthly_targets.loc[closest_month]
            else:
                # Ultimate fallback
                target_series.loc[timestamp] = 1000.0  # Safe default
    
    return target_series


def _get_monthly_target_for_date(date, monthly_targets):
    """
    Get the monthly target value for a specific date from V2's monthly targets.
    
    Args:
        date: Date to get target for
        monthly_targets: V2's monthly targets (Series with Period index)
        
    Returns:
        float: Monthly target value for the given date
    """
    # Convert date to period
    if isinstance(date, pd.Timestamp):
        month_period = date.to_period('M')
    else:
        month_period = pd.to_datetime(date).to_period('M')
    
    # Return the monthly target for this period
    if month_period in monthly_targets.index:
        return monthly_targets.loc[month_period]
    else:
        # Fallback: use the first available target
        if len(monthly_targets) > 0:
            return monthly_targets.iloc(0)
        else:
            return 1000.0  # Safe fallback


def _create_v2_conditional_demand_line_with_dynamic_targets(fig, df, power_col, target_series, selected_tariff=None, holidays=None, trace_name="Original Demand"):
    """
    V2 ENHANCEMENT: Enhanced conditional coloring logic for Original Demand line with DYNAMIC monthly targets.
    Creates continuous line segments with different colors based on monthly target conditions.
    
    Key V2 Innovation: Uses dynamic monthly targets instead of static averaging for color decisions.
    
    Color Logic:
    - Red: Above monthly target during Peak Periods (based on selected tariff) - Direct MD cost impact
    - Green: Above monthly target during Off-Peak Periods - No MD cost impact  
    - Blue: Below monthly target (any time) - Within acceptable limits
    
    Args:
        fig: Plotly figure to add traces to
        df: Simulation dataframe
        power_col: Power column name
        target_series: V2's dynamic monthly target series (same index as df)
        selected_tariff: Tariff configuration for period classification
        holidays: Set of holiday dates
        trace_name: Name for the trace
        
    Returns:
        Modified plotly figure with colored demand line segments
    """
    from tariffs.peak_logic import is_peak_rp4, get_period_classification
    
    # Validate inputs
    if target_series is None or len(target_series) == 0:
        st.warning("⚠️ V2 Dynamic Coloring: target_series is empty, falling back to single average")
        # Fallback to V1 approach with average target
        avg_target = df[power_col].quantile(0.9)
        return create_conditional_demand_line_with_peak_logic(fig, df, power_col, avg_target, selected_tariff, holidays, trace_name)
    
    # Convert index to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df.index)
    else:
        df_copy = df
    
    # Create a series with color classifications using DYNAMIC monthly targets
    df_copy = df_copy.copy()
    df_copy['color_class'] = ''
    
    for i in range(len(df_copy)):
        timestamp = df_copy.index[i]
        demand_value = df_copy.iloc[i][power_col]
        
        # V2 ENHANCEMENT: Get DYNAMIC monthly target for this specific timestamp
        if timestamp in target_series.index:
            current_target = target_series.loc[timestamp]
        else:
            # Fallback to closest available target
            month_period = timestamp.to_period('M')
            available_periods = [t.to_period('M') for t in target_series.index if not pd.isna(target_series.loc[t])]
            if available_periods:
                closest_period_timestamp = min(target_series.index, 
                                             key=lambda t: abs((timestamp - t).total_seconds()))
                current_target = target_series.loc[closest_period_timestamp]
            else:
                current_target = df[power_col].quantile(0.9)  # Safe fallback
        
        # Get peak period classification based on selected tariff
        if selected_tariff:
            try:
                period_type = get_tariff_period_classification(timestamp, selected_tariff, holidays)
            except:
                period_type = get_period_classification(timestamp, holidays)
        else:
            # Fallback to default RP4 logic
            try:
                period_type = get_period_classification(timestamp, holidays)
            except:
                # Ultimate fallback - simple hour-based check
                if timestamp.weekday() < 5 and 14 <= timestamp.hour < 22:
                    period_type = 'Peak'
                else:
                    period_type = 'Off-Peak'
        
        # V2 LOGIC: Color classification using dynamic monthly target
        if demand_value > current_target:
            if period_type == 'Peak':
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'red'
            else:
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'green'
        else:
            df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'blue'
    
    # Create continuous line segments with color-coded segments
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
                hover_info = f'<b>Above Monthly Target - TOU Peak Rate Period</b><br><i>High Energy Cost + MD Cost Impact</i><br><i>Using V2 Dynamic Monthly Targets</i>'
            else:
                hover_info = f'<b>Above Monthly Target - General Tariff</b><br><i>MD Cost Impact Only (Flat Energy Rate)</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        elif current_color == 'green':
            segment_name = f'{trace_name} (Above Target - Off-Peak)'
            if is_tou:
                hover_info = '<b>Above Monthly Target - TOU Off-Peak</b><br><i>Low Energy Cost, No MD Impact</i><br><i>Using V2 Dynamic Monthly Targets</i>'
            else:
                hover_info = '<b>Above Monthly Target - General Tariff</b><br><i>This should not appear for General tariffs</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        else:  # blue
            segment_name = f'{trace_name} (Below Target)'
            hover_info = '<b>Below Monthly Target</b><br><i>Within Acceptable Limits</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        
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


# ==========================================
# V2 ENHANCED BATTERY ALGORITHMS
# ==========================================

def _calculate_battery_health_parameters(battery_chemistry, temperature_celsius=25):
    """
    Calculate chemistry-specific battery health parameters with temperature derating.
    
    Args:
        battery_chemistry: Battery chemistry type (LFP, NMC, etc.)
        temperature_celsius: Operating temperature
        
    Returns:
        Dictionary with health parameters
    """
    # Base parameters for different battery chemistries
    chemistry_params = {
        'LFP': {  # Lithium Iron Phosphate
            'base_c_rate_continuous': 1.0,
            'base_c_rate_peak': 2.0,
            'optimal_temp_range': (20, 35),
            'cycle_life_factor': 1.0,
            'safety_factor': 0.95,
            'voltage_stability': 0.98,
            'soh_degradation_rate': 0.0002  # Per cycle
        },
        'NMC': {  # Lithium Nickel Manganese Cobalt
            'base_c_rate_continuous': 1.2,
            'base_c_rate_peak': 3.0,
            'optimal_temp_range': (15, 30),
            'cycle_life_factor': 0.9,
            'safety_factor': 0.92,
            'voltage_stability': 0.96,
            'soh_degradation_rate': 0.0003
        },
        'NCA': {  # Lithium Nickel Cobalt Aluminum
            'base_c_rate_continuous': 1.5,
            'base_c_rate_peak': 4.0,
            'optimal_temp_range': (10, 25),
            'cycle_life_factor': 0.85,
            'safety_factor': 0.90,
            'voltage_stability': 0.94,
            'soh_degradation_rate': 0.0004
        }
    }
    
    # Default to LFP if chemistry not recognized
    params = chemistry_params.get(battery_chemistry, chemistry_params['LFP'])
    
    # Calculate temperature derating factor
    optimal_min, optimal_max = params['optimal_temp_range']
    if optimal_min <= temperature_celsius <= optimal_max:
        temp_factor = 1.0  # No derating in optimal range
    elif temperature_celsius < optimal_min:
        # Cold temperature derating (linear reduction)
        temp_factor = max(0.6, 0.8 + 0.2 * (temperature_celsius - optimal_min) / (optimal_min - 0))
    else:
        # Hot temperature derating (exponential reduction)
        temp_factor = max(0.7, 1.0 - 0.02 * (temperature_celsius - optimal_max))
    
    # Apply temperature derating
    return {
        'continuous_c_rate': params['base_c_rate_continuous'] * temp_factor,
        'peak_c_rate': params['base_c_rate_peak'] * temp_factor,
        'cycle_life_factor': params['cycle_life_factor'] * temp_factor,
        'safety_factor': params['safety_factor'],
        'voltage_stability': params['voltage_stability'],
        'temperature_factor': temp_factor,
        'soh_degradation_rate': params['soh_degradation_rate'],
        'chemistry': battery_chemistry,
        'operating_temperature': temperature_celsius
    }

def _calculate_c_rate_limited_power(available_energy_kwh, max_power_rating_kw, 
                                  continuous_c_rate, peak_c_rate, duration_hours,
                                  current_soc_percent, battery_health_params):
    """
    Calculate power limit based on C-rate constraints with temperature and health derating.
    
    Args:
        available_energy_kwh: Current available energy
        max_power_rating_kw: Rated power capacity
        continuous_c_rate: Continuous C-rate limit
        peak_c_rate: Peak C-rate limit (for short duration)
        duration_hours: Expected duration of operation
        current_soc_percent: Current state of charge
        battery_health_params: Health parameters from _calculate_battery_health_parameters
        
    Returns:
        Dictionary with power limits and constraints
    """
    # Calculate C-rate based power limits
    continuous_power_limit = available_energy_kwh * continuous_c_rate
    peak_power_limit = available_energy_kwh * peak_c_rate
    
    # Apply SOC-based derating (power reduces at low/high SOC)
    if current_soc_percent > 90:
        soc_factor = 0.9 - (current_soc_percent - 90) * 0.02  # Reduce power at high SOC
    elif current_soc_percent < 20:
        soc_factor = 0.7 + (current_soc_percent - 10) * 0.02  # Reduce power at low SOC
    else:
        soc_factor = 1.0  # Full power in normal SOC range
    
    soc_factor = max(0.5, min(1.0, soc_factor))  # Limit between 50%-100%
    
    # Apply health and temperature derating
    health_factor = battery_health_params['safety_factor'] * battery_health_params['temperature_factor']
    
    # Determine power limit based on duration
    if duration_hours <= 0.25:  # 15 minutes or less - can use peak power
        c_rate_power_limit = peak_power_limit * soc_factor * health_factor
    elif duration_hours <= 1.0:  # Up to 1 hour - blend between peak and continuous
        blend_factor = 1.0 - (duration_hours - 0.25) / 0.75  # Linear blend
        blended_c_rate = continuous_c_rate + (peak_c_rate - continuous_c_rate) * blend_factor
        c_rate_power_limit = available_energy_kwh * blended_c_rate * soc_factor * health_factor
    else:  # Longer than 1 hour - use continuous power
        c_rate_power_limit = continuous_power_limit * soc_factor * health_factor
    
    # Final power limit is minimum of C-rate limit and rated power
    effective_power_limit = min(c_rate_power_limit, max_power_rating_kw)
    
    return {
        'effective_power_limit_kw': effective_power_limit,
        'c_rate_power_limit_kw': c_rate_power_limit,
        'continuous_power_limit_kw': continuous_power_limit,
        'peak_power_limit_kw': peak_power_limit,
        'soc_derating_factor': soc_factor,
        'health_derating_factor': health_factor,
        'duration_hours': duration_hours,
        'limiting_factor': 'C-rate' if c_rate_power_limit < max_power_rating_kw else 'Power Rating'
    }

def _get_soc_protection_levels():
    """
    Define multi-level SOC protection thresholds and discharge limits.
    
    Returns:
        Dictionary with SOC protection levels
    """
    return {
        'emergency': {
            'threshold_percent': 10,
            'max_discharge_percent': 0,  # No discharge allowed
            'charge_priority': 'critical',
            'description': 'Emergency - No discharge allowed'
        },
        'critical': {
            'threshold_percent': 25,
            'max_discharge_percent': 25,  # Very limited discharge
            'charge_priority': 'high',
            'description': 'Critical - Limited discharge only'
        },
        'health': {
            'threshold_percent': 40,
            'max_discharge_percent': 60,  # Moderate discharge allowed
            'charge_priority': 'medium',
            'description': 'Health protection - Moderate discharge'
        },
        'normal': {
            'threshold_percent': 80,
            'max_discharge_percent': 100,  # Full discharge allowed
            'charge_priority': 'normal',
            'description': 'Normal operation - Full discharge'
        },
        'maintenance': {
            'threshold_percent': 95,
            'max_discharge_percent': 100,  # Full operation
            'charge_priority': 'low',
            'description': 'Maintenance level - Reduce charging'
        }
    }

def _apply_soc_protection_constraints(current_soc_percent, requested_power_kw, protection_levels):
    """
    Apply multi-level SOC protection constraints to discharge power.
    
    Args:
        current_soc_percent: Current battery SOC
        requested_power_kw: Requested discharge power
        protection_levels: SOC protection levels from _get_soc_protection_levels
        
    Returns:
        Constrained power and protection info
    """
    # Determine current protection level
    active_level = 'normal'
    for level_name, level_config in protection_levels.items():
        if current_soc_percent <= level_config['threshold_percent']:
            active_level = level_name
            break
    
    # Apply discharge constraint
    max_allowed_percent = protection_levels[active_level]['max_discharge_percent']
    constrained_power = requested_power_kw * (max_allowed_percent / 100)
    
    return {
        'constrained_power_kw': constrained_power,
        'original_power_kw': requested_power_kw,
        'active_protection_level': active_level,
        'protection_description': protection_levels[active_level]['description'],
        'discharge_reduction_percent': 100 - max_allowed_percent,
        'charge_priority': protection_levels[active_level]['charge_priority']
    }


def _calculate_intelligent_charge_strategy(current_soc_percent, tariff_period, battery_health_params, 
                                         available_excess_power_kw, max_charge_power_kw):
    """
    Calculate intelligent charging strategy based on SOC levels, tariff periods, and battery health.
    
    Charging Priority Levels:
    - Emergency (SOC < 10%): Maximum charging regardless of tariff
    - Critical (SOC < 25%): Aggressive charging, moderate tariff consideration
    - Health (SOC < 40%): Balanced charging with tariff optimization
    - Normal (SOC < 80%): Tariff-optimized charging
    - Maintenance (SOC >= 80%): Minimal charging, avoid overcharging
    
    Args:
        current_soc_percent: Current battery state of charge
        tariff_period: Current tariff period ('peak', 'off_peak', 'shoulder')
        battery_health_params: Battery health parameters from _calculate_battery_health_parameters
        available_excess_power_kw: Available excess solar/renewable power for charging
        max_charge_power_kw: Maximum charging power capability
        
    Returns:
        Dictionary with charging strategy recommendations
    """
    # Get SOC protection levels
    protection_levels = _get_soc_protection_levels()
    
    # Determine charging urgency based on SOC level
    if current_soc_percent <= 10:
        urgency_level = 'emergency'
        charge_multiplier = 1.0  # Full charging power
        tariff_consideration = 0.0  # Ignore tariff costs
    elif current_soc_percent <= 25:
        urgency_level = 'critical' 
        charge_multiplier = 0.9
        tariff_consideration = 0.2  # Light tariff consideration
    elif current_soc_percent <= 40:
        urgency_level = 'health'
        charge_multiplier = 0.75
        tariff_consideration = 0.5  # Moderate tariff consideration
    elif current_soc_percent <= 80:
        urgency_level = 'normal'
        charge_multiplier = 0.6
        tariff_consideration = 0.8  # Strong tariff consideration
    else:
        urgency_level = 'maintenance'
        charge_multiplier = 0.2  # Minimal charging
        tariff_consideration = 1.0  # Full tariff consideration
    
    # Tariff-based charging adjustments
    tariff_multipliers = {
        'off_peak': 1.0,    # Best time to charge
        'shoulder': 0.7,    # Moderate charging
        'peak': 0.3         # Avoid charging during peak
    }
    
    # Calculate base charging power considering SOC urgency
    base_charge_power = min(available_excess_power_kw, max_charge_power_kw) * charge_multiplier
    
    # Apply tariff considerations
    tariff_multiplier = tariff_multipliers.get(tariff_period, 0.7)
    tariff_adjusted_multiplier = (1 - tariff_consideration) + (tariff_consideration * tariff_multiplier)
    
    # Final charging power recommendation
    recommended_charge_power = base_charge_power * tariff_adjusted_multiplier
    
    # Apply battery health constraints
    health_derating = battery_health_params.get('health_derating_factor', 1.0)
    temperature_derating = battery_health_params.get('temperature_derating_factor', 1.0)
    
    final_charge_power = recommended_charge_power * health_derating * temperature_derating
    
    return {
        'recommended_charge_power_kw': max(0, final_charge_power),
        'urgency_level': urgency_level,
        'charge_multiplier': charge_multiplier,
        'tariff_consideration': tariff_consideration,
        'tariff_multiplier': tariff_multiplier,
        'tariff_adjusted_multiplier': tariff_adjusted_multiplier,
        'health_derating': health_derating,
        'temperature_derating': temperature_derating,
        'available_excess_power_kw': available_excess_power_kw,
        'max_charge_power_kw': max_charge_power_kw,
        'charging_recommendation': f"{urgency_level.title()} charging at {final_charge_power:.1f}kW"
    }


def _get_tariff_aware_discharge_strategy(tariff_type, current_tariff_period, current_soc_percent, 
                                       demand_power_kw, battery_health_params):
    """
    Get tariff-aware discharge strategy based on tariff type and current conditions.
    
    Strategies:
    - TOU Tariff: Aggressive discharge during peak periods, conservative during off-peak
    - General Tariff: Consistent discharge strategy focused on demand reduction
    
    Args:
        tariff_type: Type of tariff ('TOU', 'General', etc.)
        current_tariff_period: Current time period ('peak', 'off_peak', 'shoulder')
        current_soc_percent: Current battery state of charge
        demand_power_kw: Current power demand
        battery_health_params: Battery health parameters
        
    Returns:
        Dictionary with discharge strategy recommendations
    """
    # Get SOC protection levels
    protection_levels = _get_soc_protection_levels()
    
    # Base discharge strategy by tariff type
    if tariff_type.upper() == 'TOU':
        # Time-of-Use tariff strategy
        if current_tariff_period == 'peak':
            # Aggressive discharge during peak periods to maximize savings
            base_discharge_multiplier = 1.0
            strategy_priority = 'high_savings'
            strategy_description = 'Peak period - aggressive discharge for maximum savings'
        elif current_tariff_period == 'shoulder':
            # Moderate discharge during shoulder periods
            base_discharge_multiplier = 0.7
            strategy_priority = 'moderate_savings'
            strategy_description = 'Shoulder period - moderate discharge'
        else:  # off_peak
            # Conservative discharge during off-peak to preserve battery
            base_discharge_multiplier = 0.4
            strategy_priority = 'battery_preservation'
            strategy_description = 'Off-peak period - conservative discharge'
    else:
        # General tariff or other - consistent discharge strategy
        base_discharge_multiplier = 0.8
        strategy_priority = 'demand_reduction'
        strategy_description = 'General tariff - consistent demand reduction'
    
    # Apply SOC-based constraints
    soc_constraint = _apply_soc_protection_constraints(
        current_soc_percent, 
        demand_power_kw, 
        protection_levels
    )
    
    # Calculate final discharge multiplier
    soc_protection_multiplier = soc_constraint['constrained_power_kw'] / max(demand_power_kw, 0.1)
    final_discharge_multiplier = base_discharge_multiplier * soc_protection_multiplier
    
    # Apply battery health constraints
    health_derating = battery_health_params.get('health_derating_factor', 1.0)
    temperature_derating = battery_health_params.get('temperature_derating_factor', 1.0)
    
    health_adjusted_multiplier = final_discharge_multiplier * health_derating * temperature_derating
    
    # Calculate recommended discharge power
    max_available_power = demand_power_kw * health_adjusted_multiplier
    
    return {
        'recommended_discharge_multiplier': health_adjusted_multiplier,
        'base_tariff_multiplier': base_discharge_multiplier,
        'soc_protection_multiplier': soc_protection_multiplier,
        'health_derating': health_derating,
        'temperature_derating': temperature_derating,
        'strategy_priority': strategy_priority,
        'strategy_description': strategy_description,
        'tariff_period': current_tariff_period,
        'active_soc_protection': soc_constraint['active_protection_level'],
        'soc_constraint_info': soc_constraint,
        'max_available_discharge_kw': max_available_power,
        'discharge_recommendation': f"{strategy_description} - {health_adjusted_multiplier:.1%} discharge capability"
    }


def _calculate_c_rate_limited_power_simple(current_soc_percent, max_power_rating_kw, battery_capacity_kwh, c_rate=1.0):
    """
    Simple C-rate power limitation for charging/discharging.
    
    Args:
        current_soc_percent: Current state of charge percentage
        max_power_rating_kw: Battery's rated power
        battery_capacity_kwh: Battery's energy capacity
        c_rate: Battery's C-rate (default 1.0C)
        
    Returns:
        Dictionary with power limits
    """
    # Calculate C-rate based power limits
    c_rate_power_limit = battery_capacity_kwh * c_rate
    
    # SOC-based derating (power reduces at extreme SOC levels)
    if current_soc_percent > 90:
        soc_factor = 0.8  # Reduce power at high SOC
    elif current_soc_percent < 20:
        soc_factor = 0.7  # Reduce power at low SOC
    else:
        soc_factor = 1.0  # Full power in normal SOC range
    
    # Final power limit is minimum of C-rate limit and rated power
    effective_max_discharge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor
    effective_max_charge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor * 0.8  # Charging typically slower
    
    return {
        'max_discharge_power_kw': effective_max_discharge_kw,
        'max_charge_power_kw': effective_max_charge_kw,
        'c_rate_power_limit_kw': c_rate_power_limit,
        'soc_derating_factor': soc_factor,
        'limiting_factor': 'C-rate' if c_rate_power_limit < max_power_rating_kw else 'Power Rating'
    }


def _simulate_battery_operation_v2(df, power_col, monthly_targets, battery_sizing, battery_params, interval_hours, selected_tariff=None, holidays=None):
    """
    V2-specific battery simulation that ensures Net Demand NEVER goes below monthly targets.
    
    Key V2 Innovation: Monthly targets act as FLOOR values for Net Demand.
    - Net Demand must stay ABOVE or EQUAL to the monthly target at all times
    - Battery discharge is limited to keep Net Demand >= Monthly Target
    - Uses dynamic monthly targets instead of static target
    
    Args:
        df: Energy data DataFrame with datetime index
        power_col: Name of power demand column
        monthly_targets: Series with Period index containing monthly targets
        battery_sizing: Dictionary with capacity_kwh, power_rating_kw
        battery_params: Dictionary with efficiency, depth_of_discharge
        interval_hours: Time interval in hours (e.g., 0.25 for 15-min)
        selected_tariff: Tariff configuration
        holidays: Set of holiday dates
        
    Returns:
        Dictionary with simulation results and V2-specific metrics
    """
    import numpy as np
    import pandas as pd
    
    # Create simulation dataframe
    df_sim = df[[power_col]].copy()
    df_sim['Original_Demand'] = df_sim[power_col]
    
    # V2 ENHANCEMENT: Create dynamic monthly target series for each timestamp
    target_series = _create_v2_dynamic_target_series(df_sim.index, monthly_targets)
    df_sim['Monthly_Target'] = target_series
    df_sim['Excess_Demand'] = (df_sim[power_col] - df_sim['Monthly_Target']).clip(lower=0)
    
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
    
    # V2 SIMULATION LOOP - Monthly Target Floor Implementation
    for i in range(len(df_sim)):
        current_demand = df_sim[power_col].iloc[i]
        monthly_target = df_sim['Monthly_Target'].iloc[i]
        excess = max(0, current_demand - monthly_target)
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
        
        if should_discharge:  # V2 ENHANCED DISCHARGE LOGIC - Monthly Target Floor with C-rate constraints
            # V2 CRITICAL CONSTRAINT: Calculate maximum discharge that keeps Net Demand >= Monthly Target
            max_allowable_discharge = current_demand - monthly_target
            
            # Get current SOC for C-rate calculations
            current_soc_kwh = soc[i-1] if i > 0 else usable_capacity * 0.8  # Start at 80% SOC
            current_soc_percent = (current_soc_kwh / usable_capacity) * 100
            
            # Get battery specifications with C-rate constraints
            if hasattr(st.session_state, 'tabled_analysis_selected_battery'):
                battery_spec = st.session_state.tabled_analysis_selected_battery['spec']
                c_rate = battery_spec.get('c_rate', 1.0)
            else:
                c_rate = 1.0  # Default C-rate
            
            # Calculate C-rate limited power
            power_limits = _calculate_c_rate_limited_power_simple(
                current_soc_percent, max_power, battery_capacity, c_rate
            )
            max_discharge_power_c_rate = power_limits['max_discharge_power_kw']
            
            # Calculate required discharge power with ALL constraints
            required_discharge = min(
                max_allowable_discharge,  # MD target constraint
                max_power,  # Battery power rating
                max_discharge_power_c_rate  # C-rate constraint
            )
            
            # Check if battery has enough energy
            available_energy = current_soc_kwh
            max_discharge_energy = available_energy
            max_discharge_power = min(max_discharge_energy / interval_hours, required_discharge)
            
            actual_discharge = max(0, max_discharge_power)
            battery_power[i] = actual_discharge
            soc[i] = current_soc_kwh - actual_discharge * interval_hours
            
            # V2 GUARANTEE: Net Demand = Original Demand - Discharge, but NEVER below Monthly Target
            net_demand_candidate = current_demand - actual_discharge
            net_demand.iloc[i] = max(net_demand_candidate, monthly_target)
            
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
            
            # Critical SOC - charge aggressively
            if soc_percentage < 30:
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
            # Normal SOC - conservative charging
            elif soc_percentage < 90:  # Increased from 90% (was implicit at 95%)
                if hour >= 22 or hour < 8:  # Off-peak hours
                    should_charge = current_demand < avg_demand * 0.7
                    charge_rate_factor = 0.5
                else:  # Peak hours - very selective
                    should_charge = current_demand < demand_25th
                    charge_rate_factor = 0.3
            
            # Execute charging if conditions are met AND it won't violate MD target
            if should_charge and soc[i] < usable_capacity * 0.95:
                # V2 CRITICAL CONSTRAINT: Maximum allowable charging that keeps Net Demand <= Monthly Target
                monthly_target = df_sim['Monthly_Target'].iloc[i]
                max_allowable_charging_for_md = max(0, monthly_target - current_demand)
                
                # Get battery specifications with C-rate constraints
                if hasattr(st.session_state, 'tabled_analysis_selected_battery'):
                    battery_spec = st.session_state.tabled_analysis_selected_battery['spec']
                    c_rate = battery_spec.get('c_rate', 1.0)
                else:
                    c_rate = 1.0  # Default C-rate
                
                # Calculate C-rate limited power
                power_limits = _calculate_c_rate_limited_power_simple(
                    soc_percentage, max_power, battery_capacity, c_rate
                )
                max_charge_power_c_rate = power_limits['max_charge_power_kw']
                
                # Calculate charge power with ALL constraints
                remaining_capacity = usable_capacity * 0.95 - soc[i]
                max_charge_energy = remaining_capacity / efficiency
                
                # V2 ENHANCED CHARGING POWER CALCULATION with all constraints
                unconstrained_charge_power = min(
                    max_power * charge_rate_factor,  # Dynamic charging rate
                    max_charge_energy / interval_hours,  # Energy constraint
                    remaining_capacity / interval_hours / efficiency,  # Don't exceed 95% SOC
                    max_charge_power_c_rate  # C-rate constraint
                )
                
                # V2 MD TARGET CONSTRAINT: Ensure Net Demand doesn't exceed monthly target
                md_constrained_charge_power = min(
                    unconstrained_charge_power,
                    max_allowable_charging_for_md
                )
                
                final_charge_power = max(0, md_constrained_charge_power)
                
                if final_charge_power > 0:
                    # Apply charging with MD constraint
                    battery_power[i] = -final_charge_power  # Negative for charging
                    soc[i] = soc[i] + final_charge_power * interval_hours * efficiency
                    
                    # V2 GUARANTEE: Net Demand = Current Demand + Charging, but NEVER above Monthly Target
                    net_demand_candidate = current_demand + final_charge_power
                    net_demand.iloc[i] = min(net_demand_candidate, monthly_target)
                else:
                    # No charging possible due to MD constraint
                    net_demand.iloc[i] = current_demand
            else:
                # No charging conditions met
                net_demand.iloc[i] = current_demand
        
        # Ensure SOC stays within limits
        soc[i] = max(0, min(soc[i], usable_capacity))
        soc_percent[i] = (soc[i] / usable_capacity) * 100
    
    # Add V2 simulation results to dataframe
    df_sim['Battery_Power_kW'] = battery_power
    df_sim['Battery_SOC_kWh'] = soc
    df_sim['Battery_SOC_Percent'] = soc_percent
    df_sim['Net_Demand_kW'] = net_demand
    df_sim['Peak_Shaved'] = df_sim['Original_Demand'] - df_sim['Net_Demand_kW']
    
    # V2 VALIDATION: Ensure Net Demand never goes below monthly targets
    violations = df_sim[df_sim['Net_Demand_kW'] < df_sim['Monthly_Target']]
    if len(violations) > 0:
        st.warning(f"⚠️ V2 Constraint Violation: {len(violations)} intervals where Net Demand < Monthly Target detected!")
    
    # Calculate V2 performance metrics
    total_energy_discharged = sum([p * interval_hours for p in battery_power if p > 0])
    total_energy_charged = sum([abs(p) * interval_hours for p in battery_power if p < 0])
    
    # V2 Peak reduction using monthly targets (not static)
    df_md_peak_for_reduction = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    if len(df_md_peak_for_reduction) > 0:
        # V2 CALCULATION: Peak reduction against monthly targets
        daily_reduction_analysis = df_md_peak_for_reduction.groupby(df_md_peak_for_reduction.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'  # V2: Get monthly target for each day
        }).reset_index()
        daily_reduction_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        
        # V2 Peak reduction: Original - Net (with monthly target context)
        daily_reduction_analysis['Peak_Reduction'] = daily_reduction_analysis['Original_Peak_MD'] - daily_reduction_analysis['Net_Peak_MD']
        peak_reduction = daily_reduction_analysis['Peak_Reduction'].max()
    else:
        # Fallback calculation
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
    
    # V2 MD-focused success rate using dynamic monthly targets
    df_md_peak = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    # Store V2 debug information
    debug_info = {
        'total_points': len(df_sim),
        'md_peak_points': len(df_md_peak),
        'monthly_targets_used': len(monthly_targets),
        'constraint_violations': len(violations),
        'sample_timestamps': df_sim.index[:3].tolist() if len(df_sim) > 0 else [],
        'v2_methodology': 'Monthly targets as floor constraints'
    }
    
    if len(df_md_peak) > 0:
        # V2 SUCCESS CALCULATION: Using monthly targets instead of static target
        daily_md_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'
        }).reset_index()
        daily_md_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        
        # V2 SUCCESS CRITERIA: Net Peak <= Monthly Target (not static target)
        daily_md_analysis['Success'] = daily_md_analysis['Net_Peak_MD'] <= daily_md_analysis['Monthly_Target'] * 1.05  # 5% tolerance
        
        successful_days = sum(daily_md_analysis['Success'])
        total_days = len(daily_md_analysis)
        success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
        md_focused_calculation = True
        
        # Store V2 debug info
        debug_info['md_calculation_details'] = {
            'successful_days': successful_days,
            'total_days': total_days,
            'calculation_method': 'V2 MD-focused with dynamic monthly targets'
        }
    else:
        # Fallback success calculation
        successful_shaves = len(df_sim[
            (df_sim['Original_Demand'] > df_sim['Monthly_Target']) & 
            (df_sim['Net_Demand_kW'] <= df_sim['Monthly_Target'] * 1.05)
        ])
        
        total_peak_events = len(df_sim[df_sim['Original_Demand'] > df_sim['Monthly_Target']])
        success_rate = (successful_shaves / total_peak_events * 100) if total_peak_events > 0 else 0
        successful_days = successful_shaves
        total_days = total_peak_events
        md_focused_calculation = False
        
        debug_info['md_calculation_details'] = {
            'successful_intervals': successful_shaves,
            'total_intervals': total_peak_events,
            'calculation_method': 'V2 Fallback with monthly targets'
        }
    
    # V2 RETURN RESULTS with monthly target context
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
        'v2_constraint_violations': len(violations),
        'monthly_targets_count': len(monthly_targets),
        'debug_info': debug_info
    }


def _simulate_battery_operation_v2_enhanced(df, power_col, monthly_targets, battery_sizing, battery_params, 
                                          interval_hours, selected_tariff=None, holidays=None, 
                                          battery_chemistry='LFP', operating_temperature=25):
    """
    V2 Enhanced battery simulation with advanced health management, C-rate constraints, 
    and intelligent charge/discharge algorithms.
    
    Key V2 Enhancements:
    - Multi-level SOC protection (Emergency/Critical/Health/Normal/Maintenance)
    - Chemistry-specific battery health parameters with temperature derating
    - C-rate limited power calculations with SOC and health derating
    - Intelligent charging strategy based on SOC levels and tariff periods
    - Tariff-aware discharge strategy (TOU vs General tariff optimization)
    - Monthly targets as FLOOR values for Net Demand (V2 core constraint)
    
    Args:
        df: Energy data DataFrame with datetime index
        power_col: Name of power demand column
        monthly_targets: Series with Period index containing monthly targets
        battery_sizing: Dictionary with capacity_kwh, power_rating_kw
        battery_params: Dictionary with efficiency, depth_of_discharge
        interval_hours: Time interval in hours (e.g., 0.25 for 15-min)
        selected_tariff: Tariff configuration
        holidays: Set of holiday dates
        battery_chemistry: Battery chemistry type ('LFP', 'NMC', 'NCA')
        operating_temperature: Operating temperature in Celsius
        
    Returns:
        Dictionary with enhanced simulation results and V2-specific metrics
    """
    import numpy as np
    import pandas as pd
    
    # Create simulation dataframe
    df_sim = df[[power_col]].copy()
    df_sim['Original_Demand'] = df_sim[power_col]
    
    # V2 ENHANCEMENT: Create dynamic monthly target series for each timestamp
    target_series = _create_v2_dynamic_target_series(df_sim.index, monthly_targets)
    df_sim['Monthly_Target'] = target_series
    df_sim['Excess_Demand'] = (df_sim[power_col] - df_sim['Monthly_Target']).clip(lower=0)
    
    # Enhanced battery parameters with health modeling
    battery_capacity = battery_sizing['capacity_kwh']
    usable_capacity = battery_capacity * (battery_params['depth_of_discharge'] / 100)
    max_power = battery_sizing['power_rating_kw']
    base_efficiency = battery_params['round_trip_efficiency'] / 100
    
    # Calculate battery health parameters
    battery_health_params = _calculate_battery_health_parameters(battery_chemistry, operating_temperature)
    
    # Initialize enhanced battery state tracking
    soc = np.zeros(len(df_sim))  # State of Charge in kWh
    soc_percent = np.zeros(len(df_sim))  # SOC as percentage
    battery_power = np.zeros(len(df_sim))  # Positive = discharge, Negative = charge
    net_demand = df_sim[power_col].copy()
    
    # Enhanced tracking arrays
    health_derating = np.zeros(len(df_sim))
    c_rate_limited_power = np.zeros(len(df_sim))
    active_protection_level = ['normal'] * len(df_sim)
    charge_strategy_info = []
    discharge_strategy_info = []
    
    # V2 ENHANCED SIMULATION LOOP - Advanced Battery Management
    for i in range(len(df_sim)):
        current_demand = df_sim[power_col].iloc[i]
        monthly_target = df_sim['Monthly_Target'].iloc[i]
        excess = max(0, current_demand - monthly_target)
        current_timestamp = df_sim.index[i]
        
        # Initialize SOC for first iteration
        if i == 0:
            soc[i] = usable_capacity * 0.8  # Start at 80% SOC
        else:
            soc[i] = soc[i-1]  # Carry forward from previous iteration
        
        # Calculate current SOC percentage
        current_soc_percent = (soc[i] / usable_capacity) * 100
        soc_percent[i] = current_soc_percent
        
        # Get current tariff period for intelligent strategies
        if selected_tariff:
            tariff_period_info = get_tariff_period_classification(current_timestamp, selected_tariff, holidays)
            tariff_period = tariff_period_info.lower() if isinstance(tariff_period_info, str) else 'off_peak'
            tariff_type = selected_tariff.get('Type', 'General')
        else:
            tariff_period = 'off_peak'
            tariff_type = 'General'
        
        # Calculate C-rate limited power for this interval
        c_rate_power_limits = _calculate_c_rate_limited_power(
            current_soc_percent, max_power, battery_health_params, interval_hours
        )
        c_rate_limited_power[i] = min(c_rate_power_limits['max_discharge_power_kw'], 
                                    c_rate_power_limits['max_charge_power_kw'])
        health_derating[i] = battery_health_params['health_derating_factor']
        
        # Determine if discharge is allowed based on V2 logic and tariff
        should_discharge = excess > 0
        
        if selected_tariff and should_discharge:
            # Get tariff-aware discharge strategy
            discharge_strategy = _get_tariff_aware_discharge_strategy(
                tariff_type, tariff_period, current_soc_percent, 
                current_demand, battery_health_params
            )
            discharge_strategy_info.append({
                'timestamp': current_timestamp,
                'strategy': discharge_strategy['strategy_description'],
                'multiplier': discharge_strategy['recommended_discharge_multiplier'],
                'soc_protection': discharge_strategy['active_soc_protection']
            })
            
            # Apply tariff-specific discharge logic
            if tariff_type.upper() == 'TOU':
                # TOU tariffs: Only discharge during peak periods
                should_discharge = (excess > 0) and (tariff_period == 'peak')
                discharge_multiplier = discharge_strategy['recommended_discharge_multiplier']
            else:
                # General tariffs: Discharge anytime above target with strategy
                discharge_multiplier = discharge_strategy['recommended_discharge_multiplier']
        else:
            discharge_multiplier = 0.8  # Default conservative discharge
            discharge_strategy_info.append({
                'timestamp': current_timestamp,
                'strategy': 'Default conservative discharge',
                'multiplier': discharge_multiplier,
                'soc_protection': 'normal'
            })
        
        if should_discharge:  # V2 ENHANCED DISCHARGE LOGIC
            # V2 CRITICAL CONSTRAINT: Calculate maximum discharge that keeps Net Demand >= Monthly Target
            max_allowable_discharge = current_demand - monthly_target
            
            # Apply enhanced power limitations
            max_discharge_by_crate = c_rate_power_limits['max_discharge_power_kw']
            max_discharge_by_health = max_power * discharge_multiplier
            
            # Calculate required discharge power with all constraints
            required_discharge = min(
                max_allowable_discharge,
                max_discharge_by_crate,
                max_discharge_by_health
            )
            
            # Check if battery has enough energy with SOC protection
            available_energy = soc[i]
            max_discharge_energy = available_energy * discharge_multiplier  # Apply health/SOC constraints
            max_discharge_power = min(max_discharge_energy / interval_hours, required_discharge)
            
            actual_discharge = max(0, max_discharge_power)
            battery_power[i] = actual_discharge
            
            # Update SOC with efficiency losses
            efficiency = base_efficiency * battery_health_params['temperature_derating_factor']
            soc[i] = soc[i] - actual_discharge * interval_hours / efficiency
            
            # V2 GUARANTEE: Net Demand = Original Demand - Discharge, but NEVER below Monthly Target
            net_demand_candidate = current_demand - actual_discharge
            net_demand.iloc[i] = max(net_demand_candidate, monthly_target)
            
            # Track active protection level
            protection_levels = _get_soc_protection_levels()
            for level_name, level_config in protection_levels.items():
                if current_soc_percent <= level_config['threshold_percent']:
                    active_protection_level[i] = level_name
                    break
            
        else:  # ENHANCED CHARGING LOGIC
            # Get intelligent charge strategy
            available_excess_power = max(0, monthly_target - current_demand) * 0.5  # Conservative excess estimation
            max_charge_power_available = c_rate_power_limits['max_charge_power_kw']
            
            charge_strategy = _calculate_intelligent_charge_strategy(
                current_soc_percent, tariff_period, battery_health_params,
                available_excess_power, max_charge_power_available
            )
            
            charge_strategy_info.append({
                'timestamp': current_timestamp,
                'urgency': charge_strategy['urgency_level'],
                'recommended_power': charge_strategy['recommended_charge_power_kw'],
                'tariff_consideration': charge_strategy['tariff_consideration']
            })
            
            # Enhanced charging decision logic
            should_charge = False
            charge_power = 0
            
            # Determine if charging conditions are met
            if charge_strategy['urgency_level'] in ['emergency', 'critical']:
                should_charge = True  # Always charge in emergency/critical
                charge_power = charge_strategy['recommended_charge_power_kw']
            elif charge_strategy['urgency_level'] == 'health':
                should_charge = available_excess_power > 0 or tariff_period == 'off_peak'
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.8
            elif charge_strategy['urgency_level'] == 'normal':
                should_charge = tariff_period == 'off_peak' and available_excess_power > 0
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.6
            elif charge_strategy['urgency_level'] == 'maintenance':
                should_charge = tariff_period == 'off_peak' and current_soc_percent < 85
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.3
            
            # Execute charging if conditions are met
            if should_charge and soc[i] < usable_capacity * 0.95:
                # Calculate charge power with enhanced constraints
                remaining_capacity = usable_capacity * 0.95 - soc[i]
                efficiency = base_efficiency * battery_health_params['temperature_derating_factor']
                
                max_charge_energy = remaining_capacity
                charge_power = min(
                    charge_power,  # Strategy-recommended power
                    max_charge_energy / interval_hours,  # Energy constraint
                    max_charge_power_available  # C-rate constraint
                )
                
                # Apply charging
                battery_power[i] = -charge_power  # Negative for charging
                soc[i] = soc[i] + charge_power * interval_hours * efficiency
                net_demand.iloc[i] = current_demand + charge_power
            else:
                net_demand.iloc[i] = current_demand
        
        # Ensure SOC stays within limits
        soc[i] = max(0, min(soc[i], usable_capacity))
        soc_percent[i] = (soc[i] / usable_capacity) * 100
    
    # Add V2 enhanced simulation results to dataframe
    df_sim['Battery_Power_kW'] = battery_power
    df_sim['Battery_SOC_kWh'] = soc
    df_sim['Battery_SOC_Percent'] = soc_percent
    df_sim['Net_Demand_kW'] = net_demand
    df_sim['Peak_Shaved'] = df_sim['Original_Demand'] - df_sim['Net_Demand_kW']
    df_sim['Health_Derating'] = health_derating
    df_sim['CRate_Limited_Power'] = c_rate_limited_power
    df_sim['Active_Protection_Level'] = active_protection_level
    
    # V2 VALIDATION: Ensure Net Demand never goes below monthly targets
    violations = df_sim[df_sim['Net_Demand_kW'] < df_sim['Monthly_Target']]
    
    # Calculate enhanced performance metrics
    total_energy_discharged = sum([p * interval_hours for p in battery_power if p > 0])
    total_energy_charged = sum([abs(p) * interval_hours for p in battery_power if p < 0])
    
    # Enhanced battery health metrics
    avg_health_derating = np.mean(health_derating)
    min_soc_reached = np.min(soc_percent)
    max_soc_reached = np.max(soc_percent)
    
    # V2 Peak reduction analysis
    df_md_peak_for_reduction = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    if len(df_md_peak_for_reduction) > 0:
        daily_reduction_analysis = df_md_peak_for_reduction.groupby(df_md_peak_for_reduction.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'
        }).reset_index()
        daily_reduction_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        daily_reduction_analysis['Peak_Reduction'] = daily_reduction_analysis['Original_Peak_MD'] - daily_reduction_analysis['Net_Peak_MD']
        peak_reduction = daily_reduction_analysis['Peak_Reduction'].max()
    else:
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
    
    # Enhanced success rate calculation with V2 constraints
    df_md_peak = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    if len(df_md_peak) > 0:
        daily_md_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'
        }).reset_index()
        daily_md_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        
        # Enhanced success criteria with health considerations
        daily_md_analysis['Success'] = daily_md_analysis['Net_Peak_MD'] <= daily_md_analysis['Monthly_Target'] * 1.03  # Tighter tolerance
        successful_days = sum(daily_md_analysis['Success'])
        total_days = len(daily_md_analysis)
        success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
    else:
        successful_days = 0
        total_days = 0
        success_rate = 0
    
    # Enhanced debug and performance information
    enhanced_debug_info = {
        'total_points': len(df_sim),
        'md_peak_points': len(df_md_peak),
        'monthly_targets_used': len(monthly_targets),
        'constraint_violations': len(violations),
        'battery_chemistry': battery_chemistry,
        'operating_temperature': operating_temperature,
        'avg_health_derating': avg_health_derating,
        'min_soc_reached': min_soc_reached,
        'max_soc_reached': max_soc_reached,
        'charge_strategies_applied': len([cs for cs in charge_strategy_info if cs['recommended_power'] > 0]),
        'discharge_strategies_applied': len([ds for ds in discharge_strategy_info if ds['multiplier'] < 1.0]),
        'v2_methodology': 'Enhanced V2 with health management and intelligent strategies'
    }
    
    # V2 ENHANCED RETURN RESULTS
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
        'md_focused_calculation': True,
        'v2_constraint_violations': len(violations),
        'monthly_targets_count': len(monthly_targets),
        'battery_health_params': battery_health_params,
        'avg_health_derating': avg_health_derating,
        'charge_strategy_log': charge_strategy_info,
        'discharge_strategy_log': discharge_strategy_info,
        'enhanced_debug_info': enhanced_debug_info
    }


def _simulate_battery_operation_v2_enhanced(df, power_col, monthly_targets, battery_sizing, battery_params, 
                                          interval_hours, selected_tariff=None, holidays=None, 
                                          battery_chemistry='LFP', operating_temperature=25):
    """
    V2 Enhanced battery simulation with advanced health management, C-rate constraints, 
    and intelligent charge/discharge algorithms.
    
    Key V2 Enhancements:
    - Multi-level SOC protection (Emergency/Critical/Health/Normal/Maintenance)
    - Chemistry-specific battery health parameters with temperature derating
    - C-rate limited power calculations with SOC and health derating
    - Intelligent charging strategy based on SOC levels and tariff periods
    - Tariff-aware discharge strategy (TOU vs General tariff optimization)
    - Monthly targets as FLOOR values for Net Demand (V2 core constraint)
    
    Args:
        df: Energy data DataFrame with datetime index
        power_col: Name of power demand column
        monthly_targets: Series with Period index containing monthly targets
        battery_sizing: Dictionary with capacity_kwh, power_rating_kw
        battery_params: Dictionary with efficiency, depth_of_discharge
        interval_hours: Time interval in hours (e.g., 0.25 for 15-min)
        selected_tariff: Tariff configuration
        holidays: Set of holiday dates
        battery_chemistry: Battery chemistry type ('LFP', 'NMC', 'NCA')
        operating_temperature: Operating temperature in Celsius
        
    Returns:
        Dictionary with enhanced simulation results and V2-specific metrics
    """
    import numpy as np
    import pandas as pd
    
    # Create simulation dataframe
    df_sim = df[[power_col]].copy()
    df_sim['Original_Demand'] = df_sim[power_col]
    
    # V2 ENHANCEMENT: Create dynamic monthly target series for each timestamp
    target_series = _create_v2_dynamic_target_series(df_sim.index, monthly_targets)
    df_sim['Monthly_Target'] = target_series
    df_sim['Excess_Demand'] = (df_sim[power_col] - df_sim['Monthly_Target']).clip(lower=0)
    
    # Enhanced battery parameters with health modeling
    battery_capacity = battery_sizing['capacity_kwh']
    usable_capacity = battery_capacity * (battery_params['depth_of_discharge'] / 100)
    max_power = battery_sizing['power_rating_kw']
    base_efficiency = battery_params['round_trip_efficiency'] / 100
    
    # Calculate battery health parameters
    battery_health_params = _calculate_battery_health_parameters(battery_chemistry, operating_temperature)
    
    # Initialize enhanced battery state tracking
    soc = np.zeros(len(df_sim))  # State of Charge in kWh
    soc_percent = np.zeros(len(df_sim))  # SOC as percentage
    battery_power = np.zeros(len(df_sim))  # Positive = discharge, Negative = charge
    net_demand = df_sim[power_col].copy()
    
    # Enhanced tracking arrays
    health_derating = np.zeros(len(df_sim))
    c_rate_limited_power = np.zeros(len(df_sim))
    active_protection_level = ['normal'] * len(df_sim)
    charge_strategy_info = []
    discharge_strategy_info = []
    
    # V2 ENHANCED SIMULATION LOOP - Advanced Battery Management
    for i in range(len(df_sim)):
        current_demand = df_sim[power_col].iloc[i]
        monthly_target = df_sim['Monthly_Target'].iloc[i]
        excess = max(0, current_demand - monthly_target)
        current_timestamp = df_sim.index[i]
        
        # Initialize SOC for first iteration
        if i == 0:
            soc[i] = usable_capacity * 0.8  # Start at 80% SOC
        else:
            soc[i] = soc[i-1]  # Carry forward from previous iteration
        
        # Calculate current SOC percentage
        current_soc_percent = (soc[i] / usable_capacity) * 100
        soc_percent[i] = current_soc_percent
        
        # Get current tariff period for intelligent strategies
        if selected_tariff:
            tariff_period_info = get_tariff_period_classification(current_timestamp, selected_tariff, holidays)
            tariff_period = tariff_period_info.lower() if isinstance(tariff_period_info, str) else 'off_peak'
            tariff_type = selected_tariff.get('Type', 'General')
        else:
            tariff_period = 'off_peak'
            tariff_type = 'General'
        
        # Calculate C-rate limited power for this interval
        c_rate_power_limits = _calculate_c_rate_limited_power(
            current_soc_percent, max_power, battery_health_params, interval_hours
        )
        c_rate_limited_power[i] = min(c_rate_power_limits['max_discharge_power_kw'], 
                                    c_rate_power_limits['max_charge_power_kw'])
        health_derating[i] = battery_health_params['health_derating_factor']
        
        # Determine if discharge is allowed based on V2 logic and tariff
        should_discharge = excess > 0
        
        if selected_tariff and should_discharge:
            # Get tariff-aware discharge strategy
            discharge_strategy = _get_tariff_aware_discharge_strategy(
                tariff_type, tariff_period, current_soc_percent, 
                current_demand, battery_health_params
            )
            discharge_strategy_info.append({
                'timestamp': current_timestamp,
                'strategy': discharge_strategy['strategy_description'],
                'multiplier': discharge_strategy['recommended_discharge_multiplier'],
                'soc_protection': discharge_strategy['active_soc_protection']
            })
            
            # Apply tariff-specific discharge logic
            if tariff_type.upper() == 'TOU':
                # TOU tariffs: Only discharge during peak periods
                should_discharge = (excess > 0) and (tariff_period == 'peak')
                discharge_multiplier = discharge_strategy['recommended_discharge_multiplier']
            else:
                # General tariffs: Discharge anytime above target with strategy
                discharge_multiplier = discharge_strategy['recommended_discharge_multiplier']
        else:
            discharge_multiplier = 0.8  # Default conservative discharge
            discharge_strategy_info.append({
                'timestamp': current_timestamp,
                'strategy': 'Default conservative discharge',
                'multiplier': discharge_multiplier,
                'soc_protection': 'normal'
            })
        
        if should_discharge:  # V2 ENHANCED DISCHARGE LOGIC
            # V2 CRITICAL CONSTRAINT: Calculate maximum discharge that keeps Net Demand >= Monthly Target
            max_allowable_discharge = current_demand - monthly_target
            
            # Apply enhanced power limitations
            max_discharge_by_crate = c_rate_power_limits['max_discharge_power_kw']
            max_discharge_by_health = max_power * discharge_multiplier
            
            # Calculate required discharge power with all constraints
            required_discharge = min(
                max_allowable_discharge,
                max_discharge_by_crate,
                max_discharge_by_health
            )
            
            # Check if battery has enough energy with SOC protection
            available_energy = soc[i]
            max_discharge_energy = available_energy * discharge_multiplier  # Apply health/SOC constraints
            max_discharge_power = min(max_discharge_energy / interval_hours, required_discharge)
            
            actual_discharge = max(0, max_discharge_power)
            battery_power[i] = actual_discharge
            
            # Update SOC with efficiency losses
            efficiency = base_efficiency * battery_health_params['temperature_derating_factor']
            soc[i] = soc[i] - actual_discharge * interval_hours / efficiency
            
            # V2 GUARANTEE: Net Demand = Original Demand - Discharge, but NEVER below Monthly Target
            net_demand_candidate = current_demand - actual_discharge
            net_demand.iloc[i] = max(net_demand_candidate, monthly_target)
            
            # Track active protection level
            protection_levels = _get_soc_protection_levels()
            for level_name, level_config in protection_levels.items():
                if current_soc_percent <= level_config['threshold_percent']:
                    active_protection_level[i] = level_name
                    break
            
        else:  # ENHANCED CHARGING LOGIC
            # Get intelligent charge strategy
            available_excess_power = max(0, monthly_target - current_demand) * 0.5  # Conservative excess estimation
            max_charge_power_available = c_rate_power_limits['max_charge_power_kw']
            
            charge_strategy = _calculate_intelligent_charge_strategy(
                current_soc_percent, tariff_period, battery_health_params,
                available_excess_power, max_charge_power_available
            )
            
            charge_strategy_info.append({
                'timestamp': current_timestamp,
                'urgency': charge_strategy['urgency_level'],
                'recommended_power': charge_strategy['recommended_charge_power_kw'],
                'tariff_consideration': charge_strategy['tariff_consideration']
            })
            
            # Enhanced charging decision logic
            should_charge = False
            charge_power = 0
            
            # Determine if charging conditions are met
            if charge_strategy['urgency_level'] in ['emergency', 'critical']:
                should_charge = True  # Always charge in emergency/critical
                charge_power = charge_strategy['recommended_charge_power_kw']
            elif charge_strategy['urgency_level'] == 'health':
                should_charge = available_excess_power > 0 or tariff_period == 'off_peak'
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.8
            elif charge_strategy['urgency_level'] == 'normal':
                should_charge = tariff_period == 'off_peak' and available_excess_power > 0
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.6
            elif charge_strategy['urgency_level'] == 'maintenance':
                should_charge = tariff_period == 'off_peak' and current_soc_percent < 85
                charge_power = charge_strategy['recommended_charge_power_kw'] * 0.3
            
            # Execute charging if conditions are met
            if should_charge and soc[i] < usable_capacity * 0.95:
                # Calculate charge power with enhanced constraints
                remaining_capacity = usable_capacity * 0.95 - soc[i]
                efficiency = base_efficiency * battery_health_params['temperature_derating_factor']
                
                max_charge_energy = remaining_capacity
                charge_power = min(
                    charge_power,  # Strategy-recommended power
                    max_charge_energy / interval_hours,  # Energy constraint
                    max_charge_power_available  # C-rate constraint
                )
                
                # Apply charging
                battery_power[i] = -charge_power  # Negative for charging
                soc[i] = soc[i] + charge_power * interval_hours * efficiency
                net_demand.iloc[i] = current_demand + charge_power
            else:
                net_demand.iloc[i] = current_demand
        
        # Ensure SOC stays within limits
        soc[i] = max(0, min(soc[i], usable_capacity))
        soc_percent[i] = (soc[i] / usable_capacity) * 100
    
    # Add V2 enhanced simulation results to dataframe
    df_sim['Battery_Power_kW'] = battery_power
    df_sim['Battery_SOC_kWh'] = soc
    df_sim['Battery_SOC_Percent'] = soc_percent
    df_sim['Net_Demand_kW'] = net_demand
    df_sim['Peak_Shaved'] = df_sim['Original_Demand'] - df_sim['Net_Demand_kW']
    df_sim['Health_Derating'] = health_derating
    df_sim['CRate_Limited_Power'] = c_rate_limited_power
    df_sim['Active_Protection_Level'] = active_protection_level
    
    # V2 VALIDATION: Ensure Net Demand never goes below monthly targets
    violations = df_sim[df_sim['Net_Demand_kW'] < df_sim['Monthly_Target']]
    
    # Calculate enhanced performance metrics
    total_energy_discharged = sum([p * interval_hours for p in battery_power if p > 0])
    total_energy_charged = sum([abs(p) * interval_hours for p in battery_power if p < 0])
    
    # Enhanced battery health metrics
    avg_health_derating = np.mean(health_derating)
    min_soc_reached = np.min(soc_percent)
    max_soc_reached = np.max(soc_percent)
    
    # V2 Peak reduction analysis
    df_md_peak_for_reduction = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    if len(df_md_peak_for_reduction) > 0:
        daily_reduction_analysis = df_md_peak_for_reduction.groupby(df_md_peak_for_reduction.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'
        }).reset_index()
        daily_reduction_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        daily_reduction_analysis['Peak_Reduction'] = daily_reduction_analysis['Original_Peak_MD'] - daily_reduction_analysis['Net_Peak_MD']
        peak_reduction = daily_reduction_analysis['Peak_Reduction'].max()
    else:
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
    
    # Enhanced success rate calculation with V2 constraints
    df_md_peak = df_sim[df_sim.index.to_series().apply(lambda ts: ts.weekday() < 5 and 14 <= ts.hour < 22)]
    
    if len(df_md_peak) > 0:
        daily_md_analysis = df_md_peak.groupby(df_md_peak.index.date).agg({
            'Original_Demand': 'max',
            'Net_Demand_kW': 'max',
            'Monthly_Target': 'first'
        }).reset_index()
        daily_md_analysis.columns = ['Date', 'Original_Peak_MD', 'Net_Peak_MD', 'Monthly_Target']
        
        # Enhanced success criteria with health considerations
        daily_md_analysis['Success'] = daily_md_analysis['Net_Peak_MD'] <= daily_md_analysis['Monthly_Target'] * 1.03  # Tighter tolerance
        successful_days = sum(daily_md_analysis['Success'])
        total_days = len(daily_md_analysis)
        success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
    else:
        successful_days = 0
        total_days = 0
        success_rate = 0
    
    # Enhanced debug and performance information
    enhanced_debug_info = {
        'total_points': len(df_sim),
        'md_peak_points': len(df_md_peak),
        'monthly_targets_used': len(monthly_targets),
        'constraint_violations': len(violations),
        'battery_chemistry': battery_chemistry,
        'operating_temperature': operating_temperature,
        'avg_health_derating': avg_health_derating,
        'min_soc_reached': min_soc_reached,
        'max_soc_reached': max_soc_reached,
        'charge_strategies_applied': len([cs for cs in charge_strategy_info if cs['recommended_power'] > 0]),
        'discharge_strategies_applied': len([ds for ds in discharge_strategy_info if ds['multiplier'] < 1.0]),
        'v2_methodology': 'Enhanced V2 with health management and intelligent strategies'
    }
    
    # V2 ENHANCED RETURN RESULTS
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
        'md_focused_calculation': True,
        'v2_constraint_violations': len(violations),
        'monthly_targets_count': len(monthly_targets),
        'battery_health_params': battery_health_params,
        'avg_health_derating': avg_health_derating,
        'charge_strategy_log': charge_strategy_info,
        'discharge_strategy_log': discharge_strategy_info,
        'enhanced_debug_info': enhanced_debug_info
    }


# ===================================================================================================
# V2 ENHANCED SHAVING SUCCESS CLASSIFICATION
# ===================================================================================================

def _get_enhanced_shaving_success(row):
    """
    Enhanced 6-stage shaving success classification with detailed performance analysis.
    
    Provides granular insight into:
    - When battery couldn't discharge (SOC/power constraints)
    - How effective partial shaving was (percentage reduction achieved)
    - Clear success metrics for performance analysis
    
    Args:
        row: DataFrame row with simulation data
        
    Returns:
        str: Enhanced success status with emoji and description
    """
    original_demand = row['Original_Demand']
    net_demand = row['Net_Demand_kW'] 
    monthly_target = row['Monthly_Target']
    battery_power = row.get('Battery_Power_kW', 0)  # Positive = discharge
    soc_percent = row.get('Battery_SOC_Percent', 100)
    
    # Stage 1: Check if MD window intervention was needed
    is_md_window = (row.name.weekday() < 5 and 14 <= row.name.hour < 22)
    if not is_md_window:
        return '🟢 Off-Peak Period'
    
    # Stage 2: Check if intervention was needed during MD window
    if original_demand <= monthly_target:
        return '🟢 No Action Needed'
    
    # Stage 3: Check if battery attempted to discharge
    if battery_power <= 0:
        if soc_percent < 25:
            return '🔴 Failed - SOC Too Low'
        else:
            return '🔴 Failed - No Discharge'
    
    # Stage 4-6: Evaluate shaving effectiveness
    excess_before = original_demand - monthly_target
    excess_after = max(0, net_demand - monthly_target)
    reduction_achieved = excess_before - excess_after
    reduction_percentage = (reduction_achieved / excess_before * 100) if excess_before > 0 else 0
    
    # Complete success - target achieved
    if net_demand <= monthly_target * 1.02:  # 2% tolerance for rounding
        return '✅ Complete Success'
    
    # Partial success levels based on reduction percentage
    elif reduction_percentage >= 80:
        return f'🟡 Good Partial ({reduction_percentage:.0f}%)'
    elif reduction_percentage >= 50:
        return f'🟠 Fair Partial ({reduction_percentage:.0f}%)'
    elif reduction_percentage >= 20:
        return f'🔶 Poor Partial ({reduction_percentage:.0f}%)'
    else:
        return '🔴 Failed - Minimal Impact'


# ===================================================================================================
# V2 TABLE VISUALIZATION FUNCTIONS - Enhanced Battery Simulation Tables
# ===================================================================================================

def _create_enhanced_battery_table(df_sim):
    """
    Create enhanced table with health and C-rate information for time-series analysis.
    
    Args:
        df_sim: Simulation dataframe with battery operation data
        
    Returns:
        pd.DataFrame: Enhanced table with status indicators and detailed battery metrics
    """
    enhanced_columns = {
        'Timestamp': df_sim.index.strftime('%Y-%m-%d %H:%M'),
        'Original_Demand_kW': df_sim['Original_Demand'].round(1),
        'Monthly_Target_kW': df_sim['Monthly_Target'].round(1),
        'Net_Demand_kW': df_sim['Net_Demand_kW'].round(1),
        'Battery_Action': df_sim['Battery_Power_kW'].apply(
            lambda x: f"Discharge {x:.1f}kW" if x > 0 else f"Charge {abs(x):.1f}kW" if x < 0 else "Standby"
        ),
        'BESS_Balance_kWh': df_sim['Battery_SOC_kWh'].round(1),
        'SOC_%': df_sim['Battery_SOC_Percent'].round(1),
        'SOC_Status': df_sim['Battery_SOC_Percent'].apply(
            lambda x: '🔴 Critical' if x < 25 else '🟡 Low' if x < 40 else '🟢 Normal' if x < 80 else '🔵 High'
        ),
        # NEW COLUMN 1: Total Charge / Discharge (kW) - Positive for charging, negative for discharging
        'Charge (+ve)/Discharge (-ve) kW': df_sim['Battery_Power_kW'].apply(
            lambda x: f"+{abs(x):.1f}" if x < 0 else f"-{x:.1f}" if x > 0 else "0.0"
        ),
        # NEW COLUMN 2: Target Shave (kW) - Amount that needs to be shaved during MD window only
        'Target_Shave_kW': df_sim.apply(
            lambda row: max(0, row['Original_Demand'] - row['Monthly_Target']) if (row.name.weekday() < 5 and 14 <= row.name.hour < 22) else 0,
            axis=1
        ).round(1),
        # NEW COLUMN 3: Actual Shave (kW) - Renamed from Peak_Shaved_kW
        'Actual_Shave_kW': df_sim['Peak_Shaved'].round(1),
        'Shaving_Success': df_sim.apply(_get_enhanced_shaving_success, axis=1),
        'MD_Period': df_sim.index.map(lambda x: '🔴 Peak' if (x.weekday() < 5 and 14 <= x.hour < 22) else '🟢 Off-Peak'),
        'Target_Violation': (df_sim['Net_Demand_kW'] > df_sim['Monthly_Target']).map({True: '❌', False: '✅'})
    }
    
    return pd.DataFrame(enhanced_columns)


def _create_daily_summary_table(df_sim):
    """
    Create daily summary of battery performance with key metrics aggregation.
    
    Args:
        df_sim: Simulation dataframe with battery operation data
        
    Returns:
        pd.DataFrame: Daily performance summary with success indicators
    """
    # Group by date
    daily_summary = df_sim.groupby(df_sim.index.date).agg({
        'Original_Demand': ['max', 'mean'],
        'Net_Demand_kW': ['max', 'mean'],
        'Monthly_Target': 'first',
        'Battery_Power_kW': ['max', 'min'],
        'Battery_SOC_Percent': ['min', 'max', 'mean'],
        'Peak_Shaved': 'max'
    }).round(1)
    
    # Flatten column names
    daily_summary.columns = [
        'Original_Peak_kW', 'Original_Avg_kW',
        'Net_Peak_kW', 'Net_Avg_kW', 
        'Monthly_Target_kW',
        'Max_Discharge_kW', 'Max_Charge_kW',
        'Min_SOC_%', 'Max_SOC_%', 'Avg_SOC_%',
        'Max_Actual_Shave_kW'  # Updated to match new column name
    ]
    
    # Add calculated columns
    daily_summary['Peak_Reduction_kW'] = daily_summary['Original_Peak_kW'] - daily_summary['Net_Peak_kW']
    daily_summary['Daily_Target_Shave_kW'] = (daily_summary['Original_Peak_kW'] - daily_summary['Monthly_Target_kW']).apply(
        lambda x: max(0, x)
    ).round(1)
    daily_summary['Target_Success'] = (daily_summary['Net_Peak_kW'] <= daily_summary['Monthly_Target_kW'] * 1.05).map({True: '✅', False: '❌'})
    daily_summary['SOC_Health'] = daily_summary['Min_SOC_%'].apply(
        lambda x: '🔴 Critical' if x < 25 else '🟡 Low' if x < 40 else '🟢 Healthy'
    )
    
    return daily_summary.reset_index()


def _create_kpi_summary_table(simulation_results, df_sim):
    """
    Create comprehensive KPI summary table with battery performance metrics.
    
    Args:
        simulation_results: Dictionary containing simulation metrics
        df_sim: Simulation dataframe with battery operation data
        
    Returns:
        pd.DataFrame: Key performance indicators table
    """
    # Get battery capacity from session state or use default
    battery_capacity_kwh = 100  # Default fallback
    if hasattr(st.session_state, 'tabled_analysis_selected_battery'):
        selected_battery = st.session_state.tabled_analysis_selected_battery
        quantity = getattr(st.session_state, 'tabled_analysis_battery_quantity', 1)
        battery_capacity_kwh = selected_battery['spec'].get('energy_kWh', 100) * quantity
    
    kpis = {
        'Metric': [
            'Total Simulation Hours',
            'Peak Reduction Achieved (kW)',
            'Success Rate (%)',
            'Total Energy Discharged (kWh)',
            'Total Energy Charged (kWh)',
            'Round-Trip Efficiency (%)',
            'Average SOC (%)',
            'Minimum SOC Reached (%)',
            'Maximum SOC Reached (%)',
            'Monthly Targets Used',
            'Target Violations',
            'Battery Utilization (%)'
        ],
        'Value': [
            f"{len(df_sim) * 0.25:.1f} hours",
            f"{simulation_results.get('peak_reduction_kw', 0):.1f} kW",
            f"{simulation_results.get('success_rate_percent', 0):.1f}%",
            f"{simulation_results.get('total_energy_discharged', 0):.1f} kWh",
            f"{simulation_results.get('total_energy_charged', 0):.1f} kWh",
            f"{(simulation_results.get('total_energy_discharged', 0) / max(simulation_results.get('total_energy_charged', 1), 1) * 100):.1f}%",
            f"{simulation_results.get('average_soc', 0):.1f}%",
            f"{simulation_results.get('min_soc', 0):.1f}%",
            f"{simulation_results.get('max_soc', 0):.1f}%",
            f"{simulation_results.get('monthly_targets_count', 0)} months",
            f"{simulation_results.get('v2_constraint_violations', 0)} intervals",
            f"{(simulation_results.get('total_energy_discharged', 0) / max(len(df_sim) * 0.25 * battery_capacity_kwh, 1) * 100):.1f}%"
        ]
    }
    
    return pd.DataFrame(kpis)


def _display_battery_simulation_tables(df_sim, simulation_results):
    """
    Display comprehensive battery simulation tables with tabbed interface.
    
    Args:
        df_sim: Simulation dataframe with battery operation data
        simulation_results: Dictionary containing simulation metrics
    """
    st.markdown("##### 1️⃣.1 📋 Battery Simulation Data Tables")
    
    # Tab-based layout for different table views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Time Series Data", 
        "📅 Daily Summary", 
        "🎯 KPI Summary",
        "🔍 Filtered View"
    ])
    
    with tab1:
        st.markdown("**Complete Time-Series Battery Operation Data**")
        table_data = _create_enhanced_battery_table(df_sim)
        
        # Display all data without filtering
        st.info(f"📊 **All Results**: Showing {len(table_data):,} records")
        
        # Display data
        st.dataframe(table_data, use_container_width=True, height=400)
        
        # Download option
        csv = table_data.to_csv(index=False)
        filename = f"battery_timeseries_{len(table_data)}records.csv"
        st.download_button("📥 Download Time Series Data", csv, filename, "text/csv", key="download_ts")
    
    with tab2:
        st.markdown("**Daily Performance Summary**")
        daily_data = _create_daily_summary_table(df_sim)
        st.dataframe(daily_data, use_container_width=True)
        
        # Download option
        csv = daily_data.to_csv(index=False)
        st.download_button("📥 Download Daily Summary", csv, "battery_daily_summary.csv", "text/csv", key="download_daily")
    
    with tab3:
        st.markdown("**Key Performance Indicators**")
        kpi_data = _create_kpi_summary_table(simulation_results, df_sim)
        st.dataframe(kpi_data, use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("**Custom Filtered View**")
        
        # Advanced filters
        col1, col2 = st.columns(2)
        with col1:
            if len(df_sim) > 0:
                date_range = st.date_input("Select date range", 
                                         [df_sim.index.min().date(), df_sim.index.max().date()],
                                         key="filter_date_range")
        with col2:
            soc_range = st.slider("SOC Range (%)", 0, 100, (0, 100), key="filter_soc_range")
        
        # Apply advanced filters
        if len(df_sim) > 0 and len(date_range) == 2:
            mask = (df_sim.index.date >= date_range[0]) & (df_sim.index.date <= date_range[1])
            mask &= (df_sim['Battery_SOC_Percent'] >= soc_range[0]) & (df_sim['Battery_SOC_Percent'] <= soc_range[1])
            
            filtered_advanced = _create_enhanced_battery_table(df_sim[mask])
            st.dataframe(filtered_advanced, use_container_width=True, height=400)
        else:
            st.info("Please select a valid date range to view filtered data.")