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
            "📊 Analysis & Results"
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

if __name__ == "__main__":
    main()
