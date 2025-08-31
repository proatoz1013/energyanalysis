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

# Import V1 components for reuse
from md_shaving_solution import (
    read_uploaded_file,
    _configure_data_inputs,
    _process_dataframe,
    _configure_tariff_selection,
    create_conditional_demand_line_with_peak_logic,
    _detect_peak_events,
    _display_battery_simulation_chart,
    _simulate_battery_operation
)
from tariffs.peak_logic import is_peak_rp4


def cluster_peak_events(events_df, battery_params, md_hours, working_days, tou_charge_windows=None, grid_charge_limit=None):
    """
    Group peak events into operational clusters for dispatch simulation.

    Args:
        events_df: DataFrame of peak events (from "Peak Event Detection Results").
        battery_params: dict with keys 'unit_energy_kwh', 'soc_min', 'soc_max', 'efficiency', 'charge_power_limit_kw'.
        md_hours: tuple (start_hour, end_hour) for MD cost impact window.
        working_days: set/list of valid working days (e.g., ['Mon', 'Tue', ...]).
        tou_charge_windows: optional list of (start_hour, end_hour) tuples for allowed charging periods.
        grid_charge_limit: optional float, site/grid charge power limit (kW).

    Returns:
        tuple: (clusters_df, events_with_clusters_df) where:
        - clusters_df: DataFrame with one row per cluster, columns:
            ['cluster_id', 'cluster_start', 'cluster_end', 'cluster_duration_hr',
             'num_events_in_cluster', 'peak_abs_kw_in_cluster', 'peak_abs_kw_sum_in_cluster', 'total_energy_above_threshold_kwh',
             'min_inter_event_gap_hr', 'max_inter_event_gap_hr', 'md_window_label']
        - events_with_clusters_df: Original events DataFrame with added 'cluster_id' column
    """
    # Compute recharge time (hours)
    E_max = battery_params['unit_energy_kwh']
    SOC_min = battery_params['soc_min']
    SOC_max = battery_params['soc_max']
    eta_charge = battery_params.get('efficiency', 1.0)
    P_charge_limit = battery_params['charge_power_limit_kw']
    if grid_charge_limit is not None:
        P_charge_limit = min(P_charge_limit, grid_charge_limit)
    recharge_time_hours = (E_max * (SOC_max - SOC_min) / 100) / (eta_charge * P_charge_limit)

    # Filter events to MD cost impact hours and working days
    def in_md_window(ts):
        day = ts.strftime('%a')  # 3-letter day abbreviation (Mon, Tue, etc.)
        hour = ts.hour
        # Make filtering more flexible - include events that intersect with MD hours
        return (day in working_days) and (md_hours[0] <= hour <= md_hours[1])

    # Apply more flexible filtering - include all events, not just those strictly within MD window
    # This allows clustering of events that might be adjacent to MD periods
    if len(events_df) > 0:
        # For initial clustering, include all events and filter later if needed
        filtered_events = events_df.copy()
    else:
        filtered_events = events_df[
            events_df['start'].apply(in_md_window) | events_df['end'].apply(in_md_window)
        ].copy()
    
    filtered_events = filtered_events.sort_values('start').reset_index(drop=True)

    clusters = []
    # Add cluster_id column to track event assignments
    events_with_clusters = filtered_events.copy()
    events_with_clusters['cluster_id'] = 0  # Initialize with 0
    
    cluster_id = 1
    i = 0
    n = len(filtered_events)
    while i < n:
        # Start new cluster
        cluster_events = [filtered_events.iloc[i]]
        cluster_event_indices = [i]  # Track indices for cluster assignment
        cluster_start = filtered_events.iloc[i]['start']
        cluster_end = filtered_events.iloc[i]['end']
        inter_event_gaps = []
        md_window_label = f"{cluster_start.strftime('%a')} {md_hours[0]:02d}-{md_hours[1]:02d}"

        # Group consecutive events within recharge window and same day
        while i + 1 < n:
            next_event = filtered_events.iloc[i + 1]
            gap_hr = (next_event['start'] - cluster_end).total_seconds() / 3600
            # Optionally restrict recharge time if TOU charge windows are defined
            effective_recharge_time = recharge_time_hours
            if tou_charge_windows:
                # If gap falls outside charge windows, reduce effective recharge time
                in_charge_window = any(
                    window[0] <= cluster_end.hour < window[1] for window in tou_charge_windows
                )
                if not in_charge_window:
                    effective_recharge_time = 0  # Can't recharge outside allowed windows
            # Check day boundary - allow cross-day clustering within reasonable limits
            time_diff = abs((next_event['start'] - cluster_start).total_seconds() / 3600)
            same_day_or_close = time_diff <= 24  # Allow events within 24 hours
            if gap_hr <= effective_recharge_time and same_day_or_close:
                cluster_events.append(next_event)
                cluster_event_indices.append(i + 1)
                inter_event_gaps.append(gap_hr)
                cluster_end = next_event['end']
                i += 1
            else:
                break
        
        # Assign cluster_id to all events in this cluster
        for idx in cluster_event_indices:
            events_with_clusters.loc[idx, 'cluster_id'] = cluster_id
            
        # Aggregate cluster metrics
        peak_abs_kw_max = max(e['peak_abs_kw'] for e in cluster_events)  # Maximum peak within cluster
        peak_abs_kw_sum = sum(e['peak_abs_kw'] for e in cluster_events)  # Sum of all event peaks
        total_energy = sum(e['energy_above_threshold_kwh'] for e in cluster_events)
        duration_hr = (cluster_end - cluster_start).total_seconds() / 3600
        min_gap = min(inter_event_gaps) if inter_event_gaps else None
        max_gap = max(inter_event_gaps) if inter_event_gaps else None
        clusters.append({
            'cluster_id': cluster_id,
            'cluster_start': cluster_start,
            'cluster_end': cluster_end,
            'cluster_duration_hr': duration_hr,
            'num_events_in_cluster': len(cluster_events),
            'peak_abs_kw_in_cluster': peak_abs_kw_max,  # Keep max for backward compatibility
            'peak_abs_kw_sum_in_cluster': peak_abs_kw_sum,  # New: sum of event peaks
            'total_energy_above_threshold_kwh': total_energy,
            'min_inter_event_gap_hr': min_gap,
            'max_inter_event_gap_hr': max_gap,
            'md_window_label': md_window_label
        })
        cluster_id += 1
        i += 1

    clusters_df = pd.DataFrame(clusters)
    return clusters_df, events_with_clusters


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

    return events_df.reset_index(drop=True)
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
        st.markdown("#### 📋 Tabled Analysis")
        
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
                col1, col2 = st.columns(2)
                
                with col1:
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
                
                with col2:
                        st.markdown("**💰 Financial Analysis:**")
                        st.markdown("**Edit Unit Cost Only:**")
                        unit_cost = st.number_input("Unit Cost (RM/kWh)", min_value=0, value=1400, step=10, key="fa_unit_cost")
                        # Calculate estimated cost based on unit cost and number of batteries required
                        num_batteries = battery_spec.get('quantity', 1) if 'quantity' in battery_spec else 1
                        battery_cost = int(unit_cost * num_batteries)
                        cost_per_kw = int(battery_cost / max(battery_spec.get('power_kW', 1), 1))
                        cost_per_kwh = unit_cost
                        financial_data = {
                            'Metric': ['Unit Cost', 'Estimated Total Cost', 'Cost per kW'],
                            'Value': [
                                f"RM {unit_cost}/kWh",
                                f"RM {battery_cost:,}",
                                f"RM {cost_per_kw}/kW",
                            ]
                        }
                        df_financial = pd.DataFrame(financial_data)
                        st.dataframe(df_financial, use_container_width=True, hide_index=True)
                
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


def _render_battery_sizing_analysis(max_shaving_power, max_tou_excess, total_md_cost):
    """
    Render comprehensive battery sizing and financial analysis table.
    
    Args:
        max_shaving_power: Maximum power shaving required (kW)
        max_tou_excess: Maximum TOU excess power requirement (kW)  
        total_md_cost: Total MD cost impact (RM)
    """
    st.markdown("#### 🔋 Battery Sizing & Financial Analysis")
    
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
            # Calculate battery quantities required
            
            # Column 1: Battery quantity for max power shaving
            qty_for_power = max_shaving_power / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_power_rounded = int(np.ceil(qty_for_power))
            
            # Column 2: Battery quantity for max TOU excess power requirement  
            qty_for_excess = max_tou_excess / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_excess_rounded = int(np.ceil(qty_for_excess))
            
            # Column 3: BESS quantity (higher of the two)
            bess_quantity = max(qty_for_power_rounded, qty_for_excess_rounded)
            
            # Calculate total system specifications
            total_power_kw = bess_quantity * battery_power_kw
            total_energy_kwh = bess_quantity * battery_energy_kwh
            
            # Column 4: MD shaved (actual impact with this battery configuration)
            # Use the total power capacity from the larger battery quantity (BESS quantity)
            md_shaved_kw = total_power_kw  # Total power from the BESS system
            md_shaving_percentage = (md_shaved_kw / max_shaving_power * 100) if max_shaving_power > 0 else 0
            
            # Column 5: Cost of batteries
            estimated_cost_per_kwh = 1400  # RM per kWh (consistent with main app)
            total_battery_cost = total_energy_kwh * estimated_cost_per_kwh
            
            # Create analysis table
            analysis_data = {
                'Analysis Parameter': [
                    'Units for Max Power Shaving',
                    'Units for Max TOU Excess Power',
                    'Total BESS Quantity Required',
                    'Total System Power Capacity',
                    'Total System Energy Capacity',
                    'Actual MD Shaved',
                    'MD Shaving Coverage',
                    'Total Battery Investment'
                ],
                'Value': [
                    f"{qty_for_power_rounded} units ({qty_for_power:.2f} calculated)",
                    f"{qty_for_excess_rounded} units ({qty_for_excess:.2f} calculated)", 
                    f"{bess_quantity} units",
                    f"{total_power_kw:.1f} kW",
                    f"{total_energy_kwh:.1f} kWh",
                    f"{md_shaved_kw:.1f} kW",
                    f"{md_shaving_percentage:.1f}%",
                    f"RM {total_battery_cost:,.0f}"
                ],
                'Calculation Basis': [
                    f"Max Power Required: {max_shaving_power:.1f} kW ÷ {battery_power_kw} kW/unit",
                    f"Max TOU Excess: {max_tou_excess:.1f} kW ÷ {battery_power_kw} kW/unit",
                    "Higher of power or TOU excess requirement",
                    f"{bess_quantity} units × {battery_power_kw} kW/unit",
                    f"{bess_quantity} units × {battery_energy_kwh} kWh/unit", 
                    f"{bess_quantity} units × {battery_power_kw} kW/unit = {total_power_kw:.1f} kW",
                    f"MD Shaved ÷ Max Power Required × 100%",
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
                    Additional {max_shaving_power - md_shaved_kw:.1f} kW capacity may be needed for complete coverage.
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
    st.title("🔋 MD Shaving Solution (v2)")
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
    st.subheader("📁 Data Upload")
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
                    # Display tariff selection (reuse V1 logic - read-only)
                    st.subheader("⚡ Tariff Configuration")
                    
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
                    st.subheader("🎯 Target Setting (V2)")
                    
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
    
    st.markdown("### 📊 Battery Impact on Energy Consumption")
    
    # Calculate monthly-based target demands using dynamic user settings (same as original)
    if power_col in df.columns:
        # Calculate monthly maximum demands
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        monthly_max_demands = df_monthly.groupby('Month')[power_col].max()
        
        # Calculate monthly targets using CORRECTED dynamic user settings
        if target_method == "Manual Target (kW)":
            # For manual target, use the same value for all months
            monthly_targets = pd.Series(index=monthly_max_demands.index, data=target_manual_kw)
            legend_label = f"Monthly Target ({target_manual_kw:.0f} kW)"
        elif target_method == "Percentage to Shave":
            # Calculate target as percentage reduction from each month's max
            target_multiplier = 1 - (shave_percent / 100)
            monthly_targets = monthly_max_demands * target_multiplier
            legend_label = f"Monthly Target ({shave_percent}% shaving)"
        else:  # Percentage of Current Max
            # Calculate target as percentage of each month's max
            target_multiplier = target_percent / 100
            monthly_targets = monthly_max_demands * target_multiplier
            legend_label = f"Monthly Target ({target_percent}% of max)"
        
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
        
        # Create the battery impact timeline chart with stepped target line
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
            
            # Identify and color-code all data points based on monthly targets and TOU periods (same as original)
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
            
            # Plot the colored segments with proper continuity (same as original)
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
            
            # Update layout
            fig.update_layout(
                title=f"Battery Impact Visualization - {selected_battery_capacity} kWh Capacity",
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
            
            # Add information about what this visualization shows
            st.info(f"""
            **📊 Graph Information:**
            - This graph shows your original energy consumption pattern
            - Battery capacity selected: **{selected_battery_capacity} kWh**
            - The colored segments indicate where battery intervention would be beneficial
            - 🔴 Red areas: Peak period events where battery discharge would reduce MD costs
            - 🟢 Green areas: Off-peak period events where battery can charge at lower rates
            - 🔵 Blue areas: Consumption already below target levels
            
            💡 **Next steps:** Further analysis will show specific shaving amounts and cost impacts.
            """)


def _render_v2_peak_events_timeline(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw, target_description):
    """Render the V2 Peak Events Timeline visualization with dynamic monthly-based targets."""
    
    st.markdown("### 📊 Peak Events Timeline")
    
    # Calculate monthly-based target demands using dynamic user settings
    if power_col in df.columns:
        # Calculate monthly maximum demands
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        monthly_max_demands = df_monthly.groupby('Month')[power_col].max()
        
        # Calculate monthly targets using CORRECTED dynamic user settings
        if target_method == "Manual Target (kW)":
            # For manual target, use the same value for all months
            monthly_targets = pd.Series(index=monthly_max_demands.index, data=target_manual_kw)
            legend_label = f"Monthly Target ({target_manual_kw:.0f} kW)"
        elif target_method == "Percentage to Shave":
            # Calculate target as percentage reduction from each month's max
            target_multiplier = 1 - (shave_percent / 100)
            monthly_targets = monthly_max_demands * target_multiplier
            legend_label = f"Monthly Target ({shave_percent}% shaving)"
        else:  # Percentage of Current Max
            # Calculate target as percentage of each month's max
            target_multiplier = target_percent / 100
            monthly_targets = monthly_max_demands * target_multiplier
            legend_label = f"Monthly Target ({target_percent}% of max)"
        
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
                    interval_hours = 0.25  # 15 minutes = 0.25 hours
                    
                    # Get MD rate from selected tariff (simplified)
                    total_md_rate = 0
                    if selected_tariff and isinstance(selected_tariff, dict):
                        rates = selected_tariff.get('Rates', {})
                        total_md_rate = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
                    
                    peak_events = _detect_peak_events(
                        month_data, power_col, target_value, total_md_rate, interval_hours, selected_tariff
                    )
                    
                    # Add month info to each event
                    for event in peak_events:
                        event['Month'] = str(month_period)
                        event['Monthly_Target'] = target_value
                        event['Monthly_Max'] = monthly_max_demands[month_period]
                        event['Shaving_Amount'] = monthly_max_demands[month_period] - target_value
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
            st.markdown("#### ⚡ Peak Event Detection Results")
            
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
            st.markdown("### 🔗 Peak Event Clusters")
            st.markdown("**Grouping consecutive peak events that can be managed with a single battery charge/discharge cycle**")
            
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
                        st.dataframe(single_events[available_single_cols], use_container_width=True)
                    
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
                    st.markdown("### ⚡ Peak Power & Energy Analysis")
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
                            if 'peak_abs_kw_sum_in_cluster' in clusters_df.columns:
                                # Add the sum power column to display
                                cluster_analysis['Total Peak Power (Sum)'] = clusters_df[clusters_df['cluster_duration_hr'] > 0]['peak_abs_kw_sum_in_cluster'].round(1)
                                
                                analysis_cols = ['Cluster ID', 'Events Count', 'Peak Power (kW)', 'Total Peak Power (Sum)', 'Total Energy (kWh)', 'Duration (minutes)']
                                available_analysis_cols = [col for col in analysis_cols if col in cluster_analysis.columns]
                                st.dataframe(cluster_analysis[available_analysis_cols], use_container_width=True)
                    else:
                        st.warning("⚠️ Peak power sum data not available - please re-run clustering analysis")
                    
                else:
                    st.warning("⚠️ No clusters generated - events may be too far apart for battery management")
                    
                    # Debug information to help understand why clustering failed
                    st.markdown("**🔍 Debug Information:**")
                    recharge_time = ((battery_params_cluster['unit_energy_kwh'] * (battery_params_cluster['soc_max'] - battery_params_cluster['soc_min']) / 100) / 
                                   (battery_params_cluster['efficiency'] * battery_params_cluster['charge_power_limit_kw']))
                    debug_info = f"""
                    - **Total events to cluster**: {len(events_for_clustering)}
                    - **MD hours filter**: {md_hours[0]:02d}:00 - {md_hours[1]:02d}:00
                    - **Working days**: {', '.join(working_days)}
                    - **Battery recharge power**: {battery_params_cluster['charge_power_limit_kw']} kW
                    - **Estimated recharge time**: {recharge_time:.1f} hours
                    """
                    st.info(debug_info)
                    
                    # Show first few events for debugging
                    if len(events_for_clustering) > 0:
                        st.markdown("**📋 Sample Events (first 5):**")
                        sample_events = events_for_clustering.head()
                        debug_cols = ['Start Date', 'Start Time', 'End Date', 'End Time']
                        available_debug_cols = [col for col in debug_cols if col in sample_events.columns]
                        if available_debug_cols:
                            st.dataframe(sample_events[available_debug_cols], use_container_width=True)
                        else:
                            st.write("Available columns:", list(sample_events.columns))
                    
                    st.markdown("**💡 Suggestions:**")
                    st.write("- Try increasing the battery recharge power limit")
                    st.write("- Check if events fall within the specified MD hours and working days")
                    st.write("- Consider if events are too far apart in time for practical battery management")
                    
            except Exception as e:
                st.error(f"❌ Clustering analysis failed: {str(e)}")
                st.write("**Debug info:**", {
                    'events_df_shape': df_events_summary.shape if 'df_events_summary' in locals() else 'N/A',
                    'events_df_columns': list(df_events_summary.columns) if 'df_events_summary' in locals() else 'N/A'
                })
            
        else:
            st.success("🎉 No peak events detected above monthly targets!")
            st.info("Current demand profile is within monthly target limits for all analyzed months")
        
        # Calculate optimal battery capacity based on shaving requirements
        if monthly_targets is not None and len(monthly_targets) > 0:
            st.markdown("#### 🔋 Recommended Battery Capacity")
            
            # Calculate maximum power shaving required using clustering analysis if available
            max_shaving_power = 0
            max_tou_excess = 0
            
            # Check if clustering analysis was performed and has results
            if ('clusters_df' in locals() and not clusters_df.empty and 
                'peak_abs_kw_sum_in_cluster' in clusters_df.columns):
                
                # Use clustering analysis results for more accurate requirements
                # Get max total peak power from multi-event clusters
                if len(clusters_df[clusters_df['cluster_duration_hr'] > 0]) > 0:
                    max_cluster_sum_power = clusters_df[clusters_df['cluster_duration_hr'] > 0]['peak_abs_kw_sum_in_cluster'].max()
                else:
                    max_cluster_sum_power = 0
                
                # Get max power from single events
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    max_single_power = clusters_df[clusters_df['cluster_duration_hr'] == 0]['peak_abs_kw_in_cluster'].max()
                else:
                    max_single_power = 0
                
                # Use the larger value between clusters and single events for power requirement
                max_shaving_power = max(max_cluster_sum_power, max_single_power)
                max_tou_excess = max_shaving_power  # TOU Excess is the power requirement
                
                st.info(f"""
                **🔋 Battery Capacity Calculation (Enhanced with Clustering Analysis):**
                - **Max Cluster Power (Sum)**: {max_cluster_sum_power:.1f} kW
                - **Max Single Event Power**: {max_single_power:.1f} kW
                - **Selected Max Power**: {max_shaving_power:.1f} kW
                - **TOU Excess Requirement**: {max_tou_excess:.1f} kW
                """)
                
            else:
                # Fallback to original calculation method if clustering data not available
                st.warning("⚠️ Using fallback calculation for battery capacity - clustering analysis data not available")
                
                if monthly_targets is not None and len(monthly_targets) > 0:
                    # Calculate max shaving power directly from monthly targets and max demands
                    shaving_amounts = []
                    for month_period, target_demand in monthly_targets.items():
                        if month_period in monthly_max_demands:
                            max_demand = monthly_max_demands[month_period]
                            shaving_amount = max_demand - target_demand
                            if shaving_amount > 0:
                                shaving_amounts.append(shaving_amount)
                    
                    max_shaving_power = max(shaving_amounts) if shaving_amounts else 0
                
                # Calculate max TOU excess from individual events (power-based, not energy)
                max_tou_excess = max([event.get('TOU Excess (kW)', 0) or 0 for event in all_monthly_events]) if 'all_monthly_events' in locals() and all_monthly_events else 0
            
            # Recommended battery capacity uses the TOU excess (power requirement)
            recommended_capacity = max_tou_excess if max_tou_excess is not None and max_tou_excess > 0 else max_shaving_power
            
            # Ensure recommended_capacity is not None
            if recommended_capacity is None:
                recommended_capacity = 0
            
            # Round up to nearest whole number
            recommended_capacity_rounded = int(np.ceil(recommended_capacity)) if recommended_capacity > 0 else 0
            
            # Display key metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Max Power Shaving Required", 
                    f"{max_shaving_power:.1f} kW",
                    help="Maximum power reduction required - enhanced with clustering analysis"
                )
            
            with col2:
                st.metric(
                    "Recommended Battery Capacity", 
                    f"{recommended_capacity_rounded} kWh",
                    help="Battery capacity based on enhanced clustering analysis of peak events"
                )
            
            # Main recommendation
            st.markdown("##### 💡 Battery Capacity Recommendation")
            
            if recommended_capacity_rounded > 0:
                # Check which method was used for the recommendation
                if ('clusters_df' in locals() and not clusters_df.empty and 
                    'peak_abs_kw_sum_in_cluster' in clusters_df.columns):
                    analysis_method = "enhanced clustering analysis"
                    rationale = "Battery capacity (kWh) is set to match the maximum TOU excess power requirement from either single events or clustered multi-event scenarios to ensure complete peak shaving capability."
                else:
                    analysis_method = "standard peak events analysis"
                    rationale = "Battery capacity (kWh) is set to match the maximum TOU excess power requirement during any single TOU peak event to ensure complete peak shaving capability."
                
                st.success(f"""
                **Recommended Battery Capacity: {recommended_capacity_rounded} kWh**
                
                This recommendation is based on the maximum TOU Excess of {max_tou_excess:.1f} kW from the {analysis_method}.
                
                **Rationale**: {rationale}
                """)
                
                # Load battery database to show matching options
                battery_db = load_vendor_battery_database()
                if battery_db:
                    matching_batteries = get_battery_options_for_capacity(battery_db, recommended_capacity_rounded, tolerance=20)
                    
                    if matching_batteries:
                        st.markdown("##### 🏭 Available Battery Options")
                        st.info(f"Found {len(matching_batteries)} battery options within ±20 kWh of recommended capacity:")
                        
                        for i, battery in enumerate(matching_batteries[:5]):  # Show top 5 matches
                            spec = battery['spec']
                            st.markdown(f"""
                            **{spec.get('manufacturer', 'Unknown')} - {spec.get('model', battery['id'])}**
                            - Capacity: {battery['capacity_kwh']} kWh
                            - Power: {battery['power_kw']} kW  
                            - C-Rate: {battery['c_rate']}C
                            """)
                    else:
                        st.warning("No matching batteries found in database for the recommended capacity.")
            else:
                st.info("No peak events detected - battery may not be required with current target settings.")
            
            # Create comprehensive battery analysis table
            if recommended_capacity_rounded > 0:
                pass
        
        # Battery Impact Analysis Section moved to separate function
        
        # Render battery selection dropdown right before battery sizing analysis
        _render_battery_selection_dropdown()
        
        # Calculate shared analysis variables for both battery sizing and simulation
        # These need to be available in broader scope for battery simulation section
        max_shaving_power = 0
        max_tou_excess = 0
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
                else:
                    max_cluster_sum_power = 0
                
                # Get max power from single events
                if len(clusters_df[clusters_df['cluster_duration_hr'] == 0]) > 0:
                    max_single_power = clusters_df[clusters_df['cluster_duration_hr'] == 0]['peak_abs_kw_in_cluster'].max()
                else:
                    max_single_power = 0
                
                # Use the larger value between clusters and single events for power requirement
                max_shaving_power = max(max_cluster_sum_power, max_single_power)
                max_tou_excess = max_shaving_power  # TOU Excess is the power requirement
                
                # Console logging for debugging - CLUSTERING ANALYSIS RESULTS
                print(f"🔋 DEBUG - Battery Sizing Values (CLUSTERING ANALYSIS):")
                print(f"   max_shaving_power = {max_shaving_power:.1f} kW")
                print(f"   max_tou_excess = {max_tou_excess:.1f} kW")
                print(f"   max_cluster_sum_power = {max_cluster_sum_power:.1f} kW")
                print(f"   max_single_power = {max_single_power:.1f} kW")
                
                st.info(f"""
                **🔋 Enhanced Battery Sizing (from Clustering Analysis):**
                - **Max Cluster Power (Sum)**: {max_cluster_sum_power:.1f} kW
                - **Max Single Event Power**: {max_single_power:.1f} kW
                - **Selected Max Power**: {max_shaving_power:.1f} kW
                - **Selected TOU Excess (Power Requirement)**: {max_tou_excess:.1f} kW
                """)
                
            else:
                # Fallback to original calculation method if clustering data not available
                st.warning("⚠️ Using fallback calculation - clustering analysis data not available")
                
                # Calculate max shaving power from monthly targets and max demands
                if monthly_targets is not None and len(monthly_targets) > 0:
                    shaving_amounts = []
                    for month_period, target_demand in monthly_targets.items():
                        if month_period in monthly_max_demands:
                            max_demand = monthly_max_demands[month_period]
                            shaving_amount = max_demand - target_demand
                            if shaving_amount > 0:
                                shaving_amounts.append(shaving_amount)
                    max_shaving_power = max(shaving_amounts) if shaving_amounts else 0
                
                # Calculate max TOU excess from individual events (power-based, not energy)
                max_tou_excess = max([event.get('TOU Excess (kW)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
                
                # Console logging for debugging - FALLBACK CALCULATION
                print(f"🔋 DEBUG - Battery Sizing Values (FALLBACK METHOD):")
                print(f"   max_shaving_power = {max_shaving_power:.1f} kW")
                print(f"   max_tou_excess = {max_tou_excess:.1f} kW")
                print(f"   monthly_targets available: {monthly_targets is not None and len(monthly_targets) > 0}")
                print(f"   number of all_monthly_events: {len(all_monthly_events) if all_monthly_events else 0}")
            
            # Calculate total MD cost from events (same for both methods)
            total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
        
        # Console logging for debugging - FINAL RESULTS (always executes)
        print(f"🔋 DEBUG - Final Battery Sizing Results:")
        print(f"   FINAL max_shaving_power = {max_shaving_power:.1f} kW")
        print(f"   FINAL max_tou_excess = {max_tou_excess:.1f} kW") 
        print(f"   FINAL total_md_cost = RM {total_md_cost:.2f}")
        
        # Call the battery sizing analysis function with the calculated values
        _render_battery_sizing_analysis(max_shaving_power, max_tou_excess, total_md_cost)
        
        # Battery Simulation Analysis Section
        st.markdown("#### 🔋 Battery Simulation Analysis")
        
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
            if max_shaving_power <= 0:
                prerequisites_met = False
                error_messages.append("Max shaving power not calculated or invalid")
            
            if max_tou_excess <= 0:
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
                
                # Calculate optimal number of units based on the analysis
                units_for_power = int(np.ceil(max_shaving_power / battery_power_kw)) if battery_power_kw > 0 else 1
                units_for_excess = int(np.ceil(max_tou_excess / battery_power_kw)) if battery_power_kw > 0 else 1
                optimal_units = max(units_for_power, units_for_excess, 1)
                
                # Calculate total system specifications
                total_battery_capacity = optimal_units * battery_capacity_kwh
                total_battery_power = optimal_units * battery_power_kw
                
                st.info(f"""
                **🔋 Battery Simulation Parameters:**
                - **Selected Battery**: {selected_battery['label']}
                - **Battery Model**: {battery_spec.get('model', 'Unknown')}
                - **Unit Specifications**: {battery_capacity_kwh:.1f} kWh, {battery_power_kw:.1f} kW per unit
                - **System Configuration**: {optimal_units} units
                - **Total System Capacity**: {total_battery_capacity:.1f} kWh
                - **Total System Power**: {total_battery_power:.1f} kW
                - **Based on**: Max Power Shaving ({max_shaving_power:.1f} kW) & Max TOU Excess ({max_tou_excess:.1f} kW)
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
                    
                    interval_hours = 0.25  # 15-minute intervals
                    
                    simulation_results = _simulate_battery_operation(
                        df_for_v1,                     # DataFrame with demand data
                        power_col,                     # Column name containing power demand
                        target_demand_for_sim,         # Target demand value
                        battery_sizing,                # Battery sizing dictionary
                        battery_params,                # Battery parameters dictionary  
                        interval_hours,                # Interval length in hours
                        selected_tariff,               # Tariff configuration
                        holidays if 'holidays' in locals() else set()  # Holidays set
                    )
                    
                    # === STEP 5: Display results and metrics ===
                    if simulation_results and 'df_simulation' in simulation_results:
                        st.success("✅ Battery simulation completed successfully!")
                        
                        # Show key simulation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Peak Reduction", 
                                f"{simulation_results.get('peak_reduction_kw', 0):.1f} kW",
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
                        _display_battery_simulation_chart(
                            simulation_results['df_simulation'],  # Simulated dataframe
                            target_demand_for_sim,              # Target demand (scalar)
                            sizing_dict,                        # Battery sizing dictionary
                            selected_tariff,                    # Tariff configuration
                            holidays if 'holidays' in locals() else set()  # Holidays set
                        )
                        
                        # === STEP 7: Enhanced BESS Dispatch Simulation & Savings Analysis ===
                        st.markdown("---")
                        st.markdown("#### 🔋 BESS Dispatch Simulation & Comprehensive Analysis")
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
                            st.markdown("#### 💰 Monthly Savings Analysis")
                            
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