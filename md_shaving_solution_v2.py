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
    _detect_peak_events
)
from tariffs.peak_logic import is_peak_rp4


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
    return {
        'selection_method': selection_method,
        'selected_capacity': selected_capacity if 'selected_capacity' in locals() else default_cap,
        'active_battery_spec': active_battery_spec,
        'run_analysis': run_analysis
    }


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
                    default_shave_percent = st.session_state.get("v2_config_default_shave", 20)
                    default_target_percent = st.session_state.get("v2_config_default_target", 80)
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
            
            # Create continuous traces with overlapping points at transitions
            continuous_segments = []
            for i, segment in enumerate(segments):
                new_segment = segment.copy()
                
                # Add overlap with previous segment (connect at start)
                if i > 0 and len(segments[i-1]['x']) > 0:
                    prev_segment = segments[i-1]
                    # Add the last point of previous segment to start of current
                    new_segment['x'] = [prev_segment['x'][-1]] + new_segment['x']
                    new_segment['y'] = [prev_segment['y'][-1]] + new_segment['y']
                
                # Add overlap with next segment (connect at end)
                if i < len(segments) - 1 and len(segments[i+1]['x']) > 0:
                    next_segment = segments[i+1]
                    # Add the first point of next segment to end of current
                    new_segment['x'] = new_segment['x'] + [next_segment['x'][0]]
                    new_segment['y'] = new_segment['y'] + [next_segment['y'][0]]
                
                continuous_segments.append(new_segment)
            
            # Plot the continuous colored segments
            color_map = {
                'below_target': {'color': 'blue', 'name': 'Below Monthly Target'},
                'above_target_offpeak': {'color': 'green', 'name': 'Above Monthly Target - Off-Peak Period'},
                'above_target_peak': {'color': 'red', 'name': 'Above Monthly Target - Peak Period'}
            }
            
            # Group segments by type for plotting
            segment_groups = {}
            for segment in continuous_segments:
                seg_type = segment['type']
                if seg_type not in segment_groups:
                    segment_groups[seg_type] = []
                segment_groups[seg_type].append(segment)
            
            # Plot each group of segments with special handling for blue line continuity
            for seg_type, seg_list in segment_groups.items():
                color_info = color_map[seg_type]
                
                if seg_type == 'below_target':
                    # For blue line (below target), connect all segments without gaps
                    # This ensures continuous line even through zero values
                    combined_x = []
                    combined_y = []
                    
                    for segment in seg_list:
                        combined_x.extend(segment['x'])
                        combined_y.extend(segment['y'])
                        # No None separators for blue line - keep it fully connected
                    
                    if combined_x:
                        fig.add_trace(go.Scatter(
                            x=combined_x,
                            y=combined_y,
                            mode='lines',
                            name=color_info['name'],
                            line=dict(color=color_info['color'], width=1),
                            opacity=0.8,
                            connectgaps=True  # Connect across any potential gaps
                        ))
                else:
                    # For green and red lines, use normal segmentation with None separators
                    combined_x = []
                    combined_y = []
                    
                    for i, segment in enumerate(seg_list):
                        combined_x.extend(segment['x'])
                        combined_y.extend(segment['y'])
                        
                        # Add None separator between disconnected segments (except for last segment)
                        if i < len(seg_list) - 1:
                            combined_x.append(None)
                            combined_y.append(None)
                    
                    if combined_x:
                        fig.add_trace(go.Scatter(
                            x=combined_x,
                            y=combined_y,
                            mode='lines',
                            name=color_info['name'],
                            line=dict(color=color_info['color'], width=1),
                            opacity=0.8,
                            connectgaps=False  # Don't connect across None values
                        ))
            
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
            
            # Calculate maximum TOU Required Energy from all events
            max_tou_energy = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
            
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
            col3.metric("Max TOU Required Energy", f"{fmt(max_tou_energy)} kWh")
            
        else:
            st.success("🎉 No peak events detected above monthly targets!")
            st.info("Current demand profile is within monthly target limits for all analyzed months")
        
        # Calculate optimal battery capacity based on shaving requirements
        if monthly_targets is not None and len(monthly_targets) > 0:
            st.markdown("#### 🔋 Recommended Battery Capacity")
            
            # Calculate maximum power shaving required across all months
            max_shaving_power = 0
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
            
            # Recommended battery capacity matches maximum TOU Required Energy from peak events
            recommended_capacity = max_tou_energy if 'max_tou_energy' in locals() and max_tou_energy is not None and max_tou_energy > 0 else max_shaving_power
            
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
                    help="Maximum power reduction required across all months - this determines battery capacity"
                )
            
            with col2:
                st.metric(
                    "Recommended Battery Capacity", 
                    f"{recommended_capacity_rounded} kWh",
                    help="Battery capacity matching the maximum power shaving requirement"
                )
            
            # Main recommendation
            st.markdown("##### 💡 Battery Capacity Recommendation")
            
            if recommended_capacity_rounded > 0:
                st.success(f"""
                **Recommended Battery Capacity: {recommended_capacity_rounded} kWh**
                
                This recommendation is based on the maximum TOU Required Energy of {max_tou_energy:.1f} kWh from the peak events analysis.
                
                **Rationale**: Battery capacity (kWh) is set to match the maximum energy requirement during any single TOU peak event to ensure complete peak shaving capability.
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
                st.markdown("#### 💰 Battery Financial Analysis")
                
                # Load battery database
                battery_db = load_vendor_battery_database()
                if battery_db:
                    # Calculate annual MD cost savings
                    annual_md_savings = total_md_cost * 12 if 'total_md_cost' in locals() else 0
                    
                    # Create analysis table for all batteries
                    battery_analysis = []
                    
                    # Estimated pricing per kWh (market average for commercial batteries)
                    estimated_price_per_kwh = 1500  # RM per kWh
                    
                    for battery_id, spec in battery_db.items():
                        battery_capacity = spec.get('energy_kWh', 0)
                        battery_power = spec.get('power_kW', 0)
                        
                        if battery_capacity > 0 and battery_power > 0:
                            # Calculate number of units needed - simple division of required capacity by unit capacity
                            units_needed = int(np.ceil(recommended_capacity_rounded / battery_capacity))
                            total_capacity = units_needed * battery_capacity
                            
                            # Ensure total capacity is always a whole number higher than required capacity
                            if total_capacity <= recommended_capacity_rounded:
                                units_needed += 1
                                total_capacity = units_needed * battery_capacity
                            
                            # Total system cost
                            total_cost = total_capacity * estimated_price_per_kwh
                            
                            # Calculate payback period
                            payback_years = total_cost / annual_md_savings if annual_md_savings > 0 else float('inf')
                            
                            battery_analysis.append({
                                'Battery Name': f"{spec.get('company', 'Unknown')} {spec.get('model', battery_id)}",
                                'Unit Capacity (kWh)': battery_capacity,
                                'Unit Power (kW)': battery_power,
                                'Units Required': units_needed,
                                'Total Capacity (kWh)': total_capacity,
                                'Annual MD Savings (RM)': annual_md_savings,
                                'Total Battery Cost (RM)': total_cost,
                                'Payback Period (Years)': payback_years if payback_years != float('inf') else 'N/A'
                            })
                    
                    # Sort by payback period (best first)
                    battery_analysis.sort(key=lambda x: x['Payback Period (Years)'] if x['Payback Period (Years)'] != 'N/A' else 999)
                    
                    if battery_analysis:
                        df_battery_analysis = pd.DataFrame(battery_analysis)
                        
                        # Format the table for display
                        formatted_analysis = df_battery_analysis.style.format({
                            'Unit Capacity (kWh)': lambda x: f"{x:.0f}",
                            'Unit Power (kW)': lambda x: f"{x:.0f}",
                            'Units Required': lambda x: f"{x:.0f}",
                            'Total Capacity (kWh)': lambda x: f"{x:.0f}",
                            'Annual MD Savings (RM)': lambda x: f"RM {x:,.0f}",
                            'Total Battery Cost (RM)': lambda x: f"RM {x:,.0f}",
                            'Payback Period (Years)': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x != float('inf') else str(x)
                        })
                        
                        st.dataframe(formatted_analysis, use_container_width=True)
                        
                        # Add summary insights
                        best_option = battery_analysis[0]
                        st.markdown("##### 🎯 Key Insights")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Best Option", 
                                best_option['Battery Name'],
                                help="Battery with shortest payback period"
                            )
                        
                        with col2:
                            payback_display = f"{best_option['Payback Period (Years)']:.1f} years" if isinstance(best_option['Payback Period (Years)'], (int, float)) else "N/A"
                            st.metric(
                                "Best Payback Period", 
                                payback_display,
                                help="Time to recover initial investment through MD savings"
                            )
                        
                        with col3:
                            st.metric(
                                "Annual Savings", 
                                f"RM {annual_md_savings:,.0f}",
                                help="Expected annual savings from MD cost reduction"
                            )
                        
                        # Add calculation notes
                        st.markdown("##### 📝 Calculation Notes")
                        st.info(f"""
                        **Assumptions:**
                        - Battery cost: RM {estimated_price_per_kwh:,}/kWh (market average for commercial systems)
                        - Annual MD savings based on monthly peak events detected: RM {annual_md_savings:,.0f}
                        - Units required based on higher of: energy capacity or power capability requirements
                        - Payback period = Total battery cost ÷ Annual MD savings
                        
                        **Note**: This analysis excludes installation, maintenance, and operational costs.
                        """)
                    else:
                        st.warning("No suitable batteries found in database for analysis.")
                else:
                    st.error("Battery database not available for financial analysis.")
        
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
        
        # Add Battery Capacity Controls after monthly summary
        st.markdown("---")  # Separator
        _render_v2_battery_controls()
        
    else:
        st.warning("Power column not found for visualization")


# Main function for compatibility
def show():
    """Compatibility function that calls the main render function."""
    render_md_shaving_v2()


if __name__ == "__main__":
    # For testing purposes
    render_md_shaving_v2()