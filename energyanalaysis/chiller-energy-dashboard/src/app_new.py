"""
Chiller Plant Energy Dashboard - Main Application

This Streamlit application provides comprehensive analysis of chiller plant energy efficiency,
including data upload, column mapping, efficiency calculations, and visualizations.

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom components
from components.data_upload import render_data_upload
from components.data_preview import render_data_preview
from components.column_mapper import render_column_mapper, calculate_derived_metrics
from components.metrics_calculator import render_metrics_display, calculate_efficiency_metrics
from components.visualizations import render_visualizations

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Chiller Plant Energy Dashboard",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .step-indicator {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">❄️ Chiller Plant Energy Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        steps = [
            "📁 Data Upload",
            "👀 Data Preview", 
            "🔗 Column Mapping",
            "📊 Analysis & Results",
            "🛠️ Equipment Performance"
        ]
        
        # Determine current step based on session state
        current_step = 0
        if st.session_state.uploaded_data is not None:
            current_step = max(current_step, 1)
        if st.session_state.column_mapping:
            current_step = max(current_step, 2)
        if st.session_state.processed_data is not None:
            current_step = max(current_step, 3)
            
        selected_step = st.radio("Steps:", steps, index=current_step)
        
        # Progress indicator
        st.markdown("---")
        st.markdown("### 📊 Progress")
        progress_value = (current_step) / (len(steps) - 1)
        st.progress(progress_value)
        st.caption(f"Step {current_step + 1} of {len(steps)}")
        
        # Add some helpful information
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        **Supported File Formats:**
        - Excel (.xlsx, .xls)
        - CSV (.csv)
        
        **Required Data:**
        - Timestamp column
        - Power consumption data
        - Cooling load information
        
        **Calculations:**
        - Total Power = Sum of all power components
        - kW/TR = Total Power / Cooling Load
        - COP = (Cooling Load × 3.51685) / Total Power
        """)
    
    # Main content area based on selected step
    if selected_step == "📁 Data Upload":
        st.session_state.uploaded_data = render_data_upload()
        
    elif selected_step == "👀 Data Preview":
        if st.session_state.uploaded_data is not None:
            render_data_preview(st.session_state.uploaded_data)
        else:
            st.warning("Please upload data first.")
            
    elif selected_step == "🔗 Column Mapping":
        if st.session_state.uploaded_data is not None:
            mapping = render_column_mapper(st.session_state.uploaded_data)
            if mapping:
                st.session_state.column_mapping = mapping
                
                # Calculate derived metrics
                st.session_state.processed_data = calculate_derived_metrics(
                    st.session_state.uploaded_data, 
                    mapping
                )
                
                # Show metrics
                render_metrics_display(st.session_state.uploaded_data, mapping)
        else:
            st.warning("Please upload data first.")
            
    elif selected_step == "📊 Analysis & Results":
        if st.session_state.column_mapping and st.session_state.processed_data is not None:
            # Show final results and visualizations
            render_visualizations(st.session_state.processed_data, st.session_state.column_mapping)
            
            # Export options
            st.markdown("---")
            st.subheader("📤 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Processed Data"):
                    csv = st.session_state.processed_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="chiller_analysis_results.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📋 Download Analysis Report"):
                    # Generate a simple text report
                    metrics = calculate_efficiency_metrics(
                        st.session_state.uploaded_data, 
                        st.session_state.column_mapping
                    )
                    
                    report = f"""
Chiller Plant Energy Analysis Report
===================================

Data Overview:
- Total Records: {len(st.session_state.uploaded_data)}
- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Average kW/TR: {metrics['avg_kw_tr']:.3f}
- Average COP: {metrics['avg_cop']:.2f}
- Total kW/TR: {metrics['total_kw_tr']:.3f}
- Total COP: {metrics['total_cop']:.2f}

Column Mapping Used:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in st.session_state.column_mapping.items() if v])}
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="chiller_analysis_report.txt",
                        mime="text/plain"
                    )
        else:
            st.warning("Please complete the column mapping step first.")
    
    elif selected_step == "🛠️ Equipment Performance":
        if st.session_state.column_mapping and st.session_state.processed_data is not None:
            st.markdown("### 🛠️ Equipment Performance Breakdown")
            st.markdown("Detailed performance analysis for individual equipment components")
            
            df = st.session_state.processed_data
            original_df = st.session_state.uploaded_data
            mapping = st.session_state.column_mapping
            
            # Create tabs for different equipment types
            eq_tabs = st.tabs(["❄️ Chiller", "💧 Pump System", "🌀 Cooling Tower", "📊 Comparative Analysis"])
            
            with eq_tabs[0]:
                render_chiller_performance(original_df, df, mapping)
            
            with eq_tabs[1]:
                render_pump_performance(original_df, df, mapping)
            
            with eq_tabs[2]:
                render_cooling_tower_performance(original_df, df, mapping)
            
            with eq_tabs[3]:
                render_comparative_analysis(original_df, df, mapping)
        
        else:
            st.warning("Please complete the column mapping step first to see equipment performance.")

def render_chiller_performance(original_df, processed_df, mapping):
    """Render chiller-specific performance analysis."""
    st.subheader("❄️ Chiller Performance Analysis")
    
    # Check if chiller data is available
    chiller_power_col = mapping.get('chiller_power')
    cooling_load_col = mapping.get('cooling_load')
    
    if not chiller_power_col and not cooling_load_col:
        st.warning("No chiller performance data available. Please ensure chiller power and cooling load are mapped.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Chiller Power Analysis
    if chiller_power_col and chiller_power_col in original_df.columns:
        with col1:
            avg_chiller_power = original_df[chiller_power_col].mean()
            max_chiller_power = original_df[chiller_power_col].max()
            min_chiller_power = original_df[chiller_power_col].min()
            
            st.metric("Average Chiller Power", f"{avg_chiller_power:.2f} kW")
            st.metric("Max Chiller Power", f"{max_chiller_power:.2f} kW")
            st.metric("Min Chiller Power", f"{min_chiller_power:.2f} kW")
    
    # Cooling Load Analysis
    if cooling_load_col and cooling_load_col in original_df.columns:
        with col2:
            avg_cooling_load = original_df[cooling_load_col].mean()
            max_cooling_load = original_df[cooling_load_col].max()
            min_cooling_load = original_df[cooling_load_col].min()
            
            st.metric("Average Cooling Load", f"{avg_cooling_load:.2f} TR")
            st.metric("Max Cooling Load", f"{max_cooling_load:.2f} TR")
            st.metric("Min Cooling Load", f"{min_cooling_load:.2f} TR")
    
    # Efficiency Metrics
    if 'kW_per_TR' in processed_df.columns:
        with col3:
            avg_kw_tr = processed_df['kW_per_TR'].mean()
            best_kw_tr = processed_df['kW_per_TR'].min()
            worst_kw_tr = processed_df['kW_per_TR'].max()
            
            st.metric("Average kW/TR", f"{avg_kw_tr:.3f}")
            st.metric("Best kW/TR", f"{best_kw_tr:.3f}")
            st.metric("Worst kW/TR", f"{worst_kw_tr:.3f}")
    
    # Performance Charts
    st.markdown("#### 📈 Chiller Performance Trends")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if chiller_power_col and chiller_power_col in original_df.columns:
            st.markdown("**Chiller Power Consumption Over Time**")
            st.line_chart(original_df[chiller_power_col])
    
    with chart_col2:
        if 'kW_per_TR' in processed_df.columns:
            st.markdown("**Chiller Efficiency (kW/TR) Over Time**")
            st.line_chart(processed_df['kW_per_TR'])

def render_pump_performance(original_df, processed_df, mapping):
    """Render pump system performance analysis."""
    st.subheader("💧 Pump System Performance Analysis")
    
    pump_power_col = mapping.get('pump_power')
    
    # Look for flow and head columns in the original data
    flow_cols = [col for col in original_df.columns if 'flow' in col.lower() and 'gpm' in col.lower()]
    head_cols = [col for col in original_df.columns if 'head' in col.lower()]
    
    if not pump_power_col and not flow_cols:
        st.warning("No pump performance data available. Please ensure pump power is mapped.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Pump Power Analysis
    if pump_power_col and pump_power_col in original_df.columns:
        with col1:
            avg_pump_power = original_df[pump_power_col].mean()
            max_pump_power = original_df[pump_power_col].max()
            min_pump_power = original_df[pump_power_col].min()
            
            st.metric("Average Pump Power", f"{avg_pump_power:.2f} kW")
            st.metric("Max Pump Power", f"{max_pump_power:.2f} kW")
            st.metric("Min Pump Power", f"{min_pump_power:.2f} kW")
    
    # Flow Analysis
    if flow_cols:
        flow_col = flow_cols[0]  # Use first flow column found
        with col2:
            avg_flow = original_df[flow_col].mean()
            max_flow = original_df[flow_col].max()
            min_flow = original_df[flow_col].min()
            
            st.metric("Average Flow", f"{avg_flow:.2f} GPM")
            st.metric("Max Flow", f"{max_flow:.2f} GPM")
            st.metric("Min Flow", f"{min_flow:.2f} GPM")
    
    # Head Analysis
    if head_cols:
        head_col = head_cols[0]  # Use first head column found
        with col3:
            avg_head = original_df[head_col].mean()
            max_head = original_df[head_col].max()
            min_head = original_df[head_col].min()
            
            st.metric("Average Head", f"{avg_head:.2f} ft")
            st.metric("Max Head", f"{max_head:.2f} ft")
            st.metric("Min Head", f"{min_head:.2f} ft")
    
    # Performance Charts
    st.markdown("#### 📈 Pump Performance Trends")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if pump_power_col and pump_power_col in original_df.columns:
            st.markdown("**Pump Power Consumption Over Time**")
            st.line_chart(original_df[pump_power_col])
    
    with chart_col2:
        if flow_cols:
            st.markdown("**Pump Flow Over Time**")
            st.line_chart(original_df[flow_cols[0]])

def render_cooling_tower_performance(original_df, processed_df, mapping):
    """Render cooling tower performance analysis."""
    st.subheader("🌀 Cooling Tower Performance Analysis")
    
    cooling_tower_power_col = mapping.get('cooling_tower_power')
    
    # Look for temperature columns
    temp_cols = [col for col in original_df.columns if 'temp' in col.lower()]
    
    if not cooling_tower_power_col and not temp_cols:
        st.warning("No cooling tower performance data available. Please ensure cooling tower power is mapped.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Cooling Tower Power Analysis
    if cooling_tower_power_col and cooling_tower_power_col in original_df.columns:
        with col1:
            avg_ct_power = original_df[cooling_tower_power_col].mean()
            max_ct_power = original_df[cooling_tower_power_col].max()
            min_ct_power = original_df[cooling_tower_power_col].min()
            
            st.metric("Average CT Power", f"{avg_ct_power:.2f} kW")
            st.metric("Max CT Power", f"{max_ct_power:.2f} kW")
            st.metric("Min CT Power", f"{min_ct_power:.2f} kW")
    
    # Temperature Analysis
    if temp_cols:
        temp_col = temp_cols[0]  # Use first temperature column found
        with col2:
            avg_temp = original_df[temp_col].mean()
            max_temp = original_df[temp_col].max()
            min_temp = original_df[temp_col].min()
            
            st.metric("Average Temperature", f"{avg_temp:.1f} °C")
            st.metric("Max Temperature", f"{max_temp:.1f} °C")
            st.metric("Min Temperature", f"{min_temp:.1f} °C")
    
    # Performance Charts
    st.markdown("#### 📈 Cooling Tower Performance Trends")
    
    if cooling_tower_power_col and cooling_tower_power_col in original_df.columns:
        st.markdown("**Cooling Tower Power Consumption Over Time**")
        st.line_chart(original_df[cooling_tower_power_col])

def render_comparative_analysis(original_df, processed_df, mapping):
    """Render comparative analysis across all equipment."""
    st.subheader("📊 Comparative Equipment Analysis")
    
    # Power Distribution Analysis
    st.markdown("#### ⚡ Power Distribution")
    
    power_data = {}
    
    if mapping.get('chiller_power') and mapping['chiller_power'] in original_df.columns:
        power_data['Chiller'] = original_df[mapping['chiller_power']].mean()
    
    if mapping.get('pump_power') and mapping['pump_power'] in original_df.columns:
        power_data['Pump'] = original_df[mapping['pump_power']].mean()
    
    if mapping.get('cooling_tower_power') and mapping['cooling_tower_power'] in original_df.columns:
        power_data['Cooling Tower'] = original_df[mapping['cooling_tower_power']].mean()
    
    if mapping.get('aux_power') and mapping['aux_power'] in original_df.columns:
        power_data['Auxiliary'] = original_df[mapping['aux_power']].mean()
    
    if power_data:
        # Create power distribution chart
        import plotly.express as px
        power_df = pd.DataFrame(list(power_data.items()), columns=['Equipment', 'Average Power (kW)'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(power_df, x='Equipment', y='Average Power (kW)', 
                           title='Average Power Consumption by Equipment')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(power_df, values='Average Power (kW)', names='Equipment',
                           title='Power Distribution by Equipment')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Power consumption table
        st.markdown("#### 📋 Power Consumption Summary")
        
        total_power = sum(power_data.values())
        power_summary = []
        
        for equipment, power in power_data.items():
            percentage = (power / total_power) * 100 if total_power > 0 else 0
            power_summary.append({
                'Equipment': equipment,
                'Average Power (kW)': f"{power:.2f}",
                'Percentage of Total': f"{percentage:.1f}%"
            })
        
        summary_df = pd.DataFrame(power_summary)
        st.dataframe(summary_df, use_container_width=True)
    
    else:
        st.warning("No power consumption data available for comparative analysis.")

if __name__ == "__main__":
    main()
