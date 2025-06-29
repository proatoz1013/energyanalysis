"""
Data Preview Component for Chiller Plant Energy Dashboard

Handles data preview and basic analysis functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_data_preview(df):
    """
    Render the data preview interface.
    
    Args:
        df (pandas.DataFrame): The dataframe to preview
    """
    if df is None:
        st.warning("No data to preview. Please upload a file first.")
        return
    
    display_data_preview(df)

def display_data_preview(df):
    """
    Display comprehensive data preview with statistics and visualizations.
    
    Args:
        df (pandas.DataFrame): The dataframe to preview
    """
    st.markdown("### 👀 Data Preview")
    
    # Basic information tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Sample", "📈 Statistics", "🔍 Data Quality", "📋 Column Info"])
    
    with tab1:
        display_data_sample(df)
    
    with tab2:
        display_data_statistics(df)
    
    with tab3:
        display_data_quality(df)
    
    with tab4:
        display_column_information(df)

def display_data_sample(df):
    """Display sample of the data."""
    st.markdown("#### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("#### Last 10 Rows")
    st.dataframe(df.tail(10), use_container_width=True)
    
    # Random sample
    if len(df) > 20:
        st.markdown("#### Random Sample (10 rows)")
        random_sample = df.sample(n=min(10, len(df)), random_state=42)
        st.dataframe(random_sample, use_container_width=True)

def display_data_statistics(df):
    """Display statistical summary of the data."""
    st.markdown("#### Numerical Columns Summary")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Create distribution plots for numeric columns
        st.markdown("#### Distribution of Numeric Columns")
        
        # Select columns for visualization
        if len(numeric_cols) > 0:
            selected_cols = st.multiselect(
                "Select columns to visualize:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:min(4, len(numeric_cols))]
            )
            
            if selected_cols:
                create_distribution_plots(df[selected_cols])
    else:
        st.warning("No numeric columns found in the dataset.")
    
    # Text columns summary
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        st.markdown("#### Text Columns Summary")
        text_summary = []
        for col in text_cols:
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            text_summary.append({
                "Column": col,
                "Unique Values": unique_count,
                "Most Common": str(most_common)
            })
        
        text_df = pd.DataFrame(text_summary)
        st.dataframe(text_df, use_container_width=True)

def display_data_quality(df):
    """Display data quality analysis."""
    st.markdown("#### Data Quality Assessment")
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    quality_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_data.values,
        'Missing %': missing_percent.values,
        'Data Type': [str(dtype) for dtype in df.dtypes]
    })
    
    # Color code based on missing percentage
    def color_missing(val):
        if val > 20:
            return 'background-color: #ffcccc'  # Red for >20%
        elif val > 5:
            return 'background-color: #ffffcc'  # Yellow for 5-20%
        else:
            return 'background-color: #ccffcc'  # Green for <5%
    
    styled_df = quality_df.style.applymap(color_missing, subset=['Missing %'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Missing data visualization
    if missing_data.sum() > 0:
        st.markdown("#### Missing Data Visualization")
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    st.metric("🔄 Duplicate Rows", duplicate_count)
    
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows in the dataset.")

def display_column_information(df):
    """Display detailed column information."""
    st.markdown("#### Column Information")
    
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Try to identify potential column types
        col_type_guess = guess_column_type(df[col], col)
        
        col_info.append({
            'Column Name': col,
            'Data Type': dtype,
            'Unique Values': unique_count,
            'Null Count': null_count,
            'Likely Type': col_type_guess
        })
    
    info_df = pd.DataFrame(col_info)
    st.dataframe(info_df, use_container_width=True)

def create_distribution_plots(df_numeric):
    """Create distribution plots for numeric columns."""
    cols = df_numeric.columns
    n_cols = len(cols)
    
    if n_cols == 0:
        return
    
    # Create subplots
    n_rows = (n_cols + 1) // 2
    
    for i, col in enumerate(cols):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
        
        with col1 if i % 2 == 0 else col2:
            fig = px.histogram(
                df_numeric, 
                x=col, 
                title=f'Distribution of {col}',
                nbins=30
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def guess_column_type(series, col_name):
    """
    Guess the type of column based on its content and name.
    
    Args:
        series (pandas.Series): The column data
        col_name (str): The column name
        
    Returns:
        str: Guessed column type
    """
    col_name_lower = col_name.lower()
    
    # Check for time-related columns
    if any(keyword in col_name_lower for keyword in ['time', 'date', 'timestamp', 'hour', 'day']):
        return "🕐 Timestamp"
    
    # Check for power-related columns
    elif any(keyword in col_name_lower for keyword in ['kw', 'power', 'watt', 'kwh', 'energy']):
        return "⚡ Power/Energy"
    
    # Check for temperature columns
    elif any(keyword in col_name_lower for keyword in ['temp', 'temperature', '°c', 'celsius']):
        return "🌡️ Temperature"
    
    # Check for flow columns
    elif any(keyword in col_name_lower for keyword in ['flow', 'gpm', 'lpm', 'm3/h']):
        return "💧 Flow Rate"
    
    # Check for cooling load columns
    elif any(keyword in col_name_lower for keyword in ['tr', 'ton', 'tonnage', 'cool', 'load']):
        return "❄️ Cooling Load"
    
    # Check for efficiency columns
    elif any(keyword in col_name_lower for keyword in ['eff', 'cop', '%', 'percent']):
        return "📊 Efficiency/Ratio"
    
    # Check for chiller-specific columns
    elif any(keyword in col_name_lower for keyword in ['ch1', 'ch2', 'ch3', 'chiller']):
        return "🏭 Chiller"
    
    # Check for pump columns
    elif any(keyword in col_name_lower for keyword in ['pump', 'chwp', 'cdwp']):
        return "🔄 Pump"
    
    # Check for cooling tower columns
    elif any(keyword in col_name_lower for keyword in ['ct', 'tower', 'cooling tower']):
        return "🏢 Cooling Tower"
    
    # Based on data type
    elif pd.api.types.is_numeric_dtype(series):
        return "🔢 Numeric"
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "📅 DateTime"
    
    else:
        return "📝 Text"

def detect_datetime_columns(df):
    """
    Detect potential datetime columns in the dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        
    Returns:
        list: List of potential datetime column names
    """
    datetime_candidates = []
    
    for col in df.columns:
        col_name_lower = col.lower()
        
        # Check by column name
        if any(keyword in col_name_lower for keyword in ['time', 'date', 'timestamp']):
            datetime_candidates.append(col)
            continue
        
        # Check by data type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_candidates.append(col)
            continue
        
        # Check by trying to parse as datetime
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().iloc[:100])  # Test first 100 non-null values
                datetime_candidates.append(col)
            except:
                pass
    
    return datetime_candidates