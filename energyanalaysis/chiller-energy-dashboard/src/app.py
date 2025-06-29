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
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Chiller+Plant", use_column_width=True)
        st.markdown("### Navigation")
        
        steps = [
            "📁 Data Upload",
            "👀 Data Preview", 
            "🔗 Column Mapping",
            "📊 Analysis & Results"
        ]
        
        # Determine current step based on session state
        current_step = 0
        if st.session_state.uploaded_data is not None:
            current_step = 1
        if st.session_state.column_mapping:
            current_step = 2
        if st.session_state.processed_data is not None:
            current_step = 3
            
        selected_step = st.radio("Steps:", steps, index=current_step)
        
        # Add some helpful information
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        **Supported File Formats:**
        - Excel (.xlsx, .xls)
        - CSV (.csv)
        
        **Required Columns:**
        - Timestamp
        - Chiller Power (kW)
        - Pump Power (kW)
        - Cooling Load (TR)
        """)
    
    # Main content based on selected step
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

def display_upload_section():
    """Display the data upload section."""
    st.markdown('<h2 class="section-header">📁 Data Upload</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your chiller plant data file. The system supports Excel and CSV formats.
    Make sure your data includes timestamp, power consumption, and cooling load information.
    """)
    
    # File upload component
    uploaded_file = handle_file_upload()
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = process_uploaded_data(uploaded_file)
            st.session_state.uploaded_data = df
            
            # Display upload status
            display_upload_status(df, uploaded_file.name)
            
            # Show next step button
            if st.button("📊 Preview Data", type="primary"):
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check your file format and try again.")

def display_preview_section():
    """Display the data preview section."""
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a data file first.")
        return
        
    st.markdown('<h2 class="section-header">👀 Data Preview</h2>', unsafe_allow_html=True)
    
    df = st.session_state.uploaded_data
    display_data_preview(df)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.uploaded_data = None
            st.rerun()
    with col2:
        if st.button("🔗 Map Columns", type="primary"):
            st.rerun()

def display_mapping_section():
    """Display the column mapping section."""
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a data file first.")
        return
        
    st.markdown('<h2 class="section-header">🔗 Column Mapping</h2>', unsafe_allow_html=True)
    
    df = st.session_state.uploaded_data
    
    st.markdown("""
    Map your data columns to the required fields. This helps the system understand
    your data structure and perform accurate calculations.
    """)
    
    # Display column mapper
    column_mapping = display_column_mapper(df)
    
    if column_mapping:
        st.session_state.column_mapping = column_mapping
        
        # Validate mapping
        validation_result = validate_column_mapping(column_mapping, df)
        
        if validation_result['valid']:
            st.success("✅ Column mapping is valid!")
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Back to Preview"):
                    st.rerun()
            with col2:
                if st.button("📊 Analyze Data", type="primary"):
                    st.rerun()
        else:
            st.error("❌ Column mapping validation failed:")
            for error in validation_result['errors']:
                st.error(f"• {error}")

def display_analysis_section():
    """Display the analysis and results section."""
    if st.session_state.uploaded_data is None or not st.session_state.column_mapping:
        st.warning("Please complete the upload and column mapping steps first.")
        return
        
    st.markdown('<h2 class="section-header">📊 Analysis & Results</h2>', unsafe_allow_html=True)
    
    df = st.session_state.uploaded_data
    column_mapping = st.session_state.column_mapping
    
    # Calculate metrics if not already done
    if st.session_state.processed_data is None:
        with st.spinner("Calculating efficiency metrics..."):
            processed_data, metrics = calculate_chiller_metrics(df, column_mapping)
            st.session_state.processed_data = processed_data
            st.session_state.metrics = metrics
    
    processed_data = st.session_state.processed_data
    metrics = st.session_state.metrics
    
    # Display key metrics
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average kW/TR",
            value=f"{metrics['avg_kw_per_tr']:.3f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Average COP",
            value=f"{metrics['avg_cop']:.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Plant Efficiency",
            value=f"{metrics['avg_efficiency']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Total Power",
            value=f"{metrics['avg_total_power']:.1f} kW",
            delta=None
        )
    
    # Display visualizations
    display_efficiency_charts(processed_data, metrics)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Mapping"):
            st.rerun()
    with col2:
        if st.button("📥 Download Results"):
            # TODO: Implement download functionality
            st.info("Download functionality coming soon!")

if __name__ == "__main__":
    main()