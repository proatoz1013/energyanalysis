"""
Chiller Plant Efficiency Analysis Module

This module provides comprehensive chiller plant efficiency analysis functionality
following the existing app patterns and helpers:
- File upload using existing read_uploaded_file() helper
- Data processing with pandas
- KPI cards in st.columns()
- Plotly charts consistent with app style
- Optional expanders for details

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Import existing helpers for consistency
try:
    from tnb_tariff_comparison import read_uploaded_file
except ImportError:
    # Fallback to simple file reader
    def read_uploaded_file(uploaded_file):
        """Simple file reader fallback"""
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None

def preprocess_timestamp_column(column):
    """Preprocess timestamp column to handle various formats - consistent with existing pattern"""
    # Remove leading/trailing spaces
    column = column.str.strip()
    # Replace text-based months with numeric equivalents
    month_replacements = {
        r'\bJan\b': '01', r'\bFeb\b': '02', r'\bMar\b': '03', r'\bApr\b': '04',
        r'\bMay\b': '05', r'\bJun\b': '06', r'\bJul\b': '07', r'\bAug\b': '08',
        r'\bSep\b': '09', r'\bOct\b': '10', r'\bNov\b': '11', r'\bDec\b': '12'
    }
    for pattern, replacement in month_replacements.items():
        column = column.str.replace(pattern, replacement, regex=True)
    return column

def calculate_chiller_derived_columns(df):
    """
    Calculate derived chiller efficiency columns using the specified formulas.
    
    Formulas:
    - Cooling_kW = 1.163 * FLOWRATE_m3hr * DeltaT
    - TR = Cooling_kW / 3.517
    - kW_per_RT = Operating kW / TR
    - Load_pct = TR / Total Chiller Tonnage * 100
    - Chiller_fraction = TR / Total Chiller Tonnage (0–1)
    
    Args:
        df: DataFrame with chiller data
        
    Returns:
        df: DataFrame with calculated columns added/updated
    """
    df = df.copy()
    
    # Only calculate if we have the required base columns
    required_cols = ['FLOWRATE m3hr', 'Delta T', 'Operating kW', 'Total Chiller Tonnage']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Cannot calculate derived columns. Missing: {', '.join(missing_cols)}")
        return df
    
    # Filter out invalid rows (FLOWRATE <= 0 or Delta T <= 0)
    valid_mask = (df['FLOWRATE m3hr'] > 0) & (df['Delta T'] > 0)
    
    if not valid_mask.any():
        st.warning("⚠️ No valid data rows found (FLOWRATE > 0 and Delta T > 0)")
        return df
    
    # Calculate derived columns for valid rows
    df.loc[valid_mask, 'Cooling_kW'] = 1.163 * df.loc[valid_mask, 'FLOWRATE m3hr'] * df.loc[valid_mask, 'Delta T']
    df.loc[valid_mask, 'TR_Calculated'] = df.loc[valid_mask, 'Cooling_kW'] / 3.517
    
    # Use calculated TR or existing Actual Chiller RT
    if 'Actual Chiller RT' in df.columns and not df['Actual Chiller RT'].isna().all():
        # Use existing values where available, calculated where missing
        df['TR'] = df['Actual Chiller RT'].fillna(df['TR_Calculated'])
    else:
        df['TR'] = df['TR_Calculated']
    
    # Calculate efficiency metrics for valid rows with TR > 0
    efficiency_mask = valid_mask & (df['TR'] > 0)
    
    if efficiency_mask.any():
        # kW/RT calculation - use existing if available, otherwise calculate
        if 'Efficiency kW/RT' not in df.columns or df['Efficiency kW/RT'].isna().all():
            df.loc[efficiency_mask, 'kW_per_RT'] = df.loc[efficiency_mask, 'Operating kW'] / df.loc[efficiency_mask, 'TR']
        else:
            df['kW_per_RT'] = df['Efficiency kW/RT'].fillna(
                df.loc[efficiency_mask, 'Operating kW'] / df.loc[efficiency_mask, 'TR']
            )
        
        # Load percentage calculations
        df.loc[efficiency_mask, 'Load_pct'] = (df.loc[efficiency_mask, 'TR'] / df.loc[efficiency_mask, 'Total Chiller Tonnage']) * 100
        df.loc[efficiency_mask, 'Chiller_fraction'] = df.loc[efficiency_mask, 'TR'] / df.loc[efficiency_mask, 'Total Chiller Tonnage']
        
        # Use existing Chiller % if available, otherwise use calculated
        if 'Chiller %' not in df.columns or df['Chiller %'].isna().all():
            df['Chiller_pct'] = df['Chiller_fraction']
        else:
            df['Chiller_pct'] = df['Chiller %'].fillna(df['Chiller_fraction'])
    
    # Clean up temporary columns
    if 'TR_Calculated' in df.columns:
        df = df.drop('TR_Calculated', axis=1)
    
    return df

def create_daily_summary_table(df, timestamp_col):
    """Create daily summary table with key metrics"""
    if df.empty or timestamp_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure timestamp column is datetime
    df['Date'] = pd.to_datetime(df[timestamp_col]).dt.date
    
    # Calculate interval for energy integration
    df_sorted = df.sort_values(timestamp_col)
    time_diffs = pd.to_datetime(df_sorted[timestamp_col]).diff().dt.total_seconds() / 3600  # hours
    interval_hours = time_diffs.mode().iloc[0] if not time_diffs.empty else 0.25  # fallback to 15min
    
    # Group by date and calculate daily metrics
    daily_summary = df.groupby('Date').agg({
        'kW_per_RT': 'mean',
        'Load_pct': 'mean', 
        'Operating kW': lambda x: (x * interval_hours).sum(),  # Total kWh
        'TR': lambda x: (x * interval_hours).sum()  # Total TRh
    }).round(2)
    
    daily_summary.columns = ['Avg kW/RT', 'Avg Load %', 'Total kWh', 'Total TRh']
    daily_summary = daily_summary.reset_index()
    
    return daily_summary

def render_chiller_plant_efficiency_tab():
    """
    Main function to render the Chiller Plant Efficiency tab.
    Follows existing app patterns and conventions.
    """
    st.title("❄️ Chiller Plant Efficiency")
    st.markdown("""
    **Comprehensive chiller plant efficiency analysis** with automated calculations and interactive visualizations.
    
    📋 **Key Features:**
    - 📁 **Excel Template Support**: Upload your fixed template data
    - 🔧 **Automatic Calculations**: Cooling load, efficiency, and performance metrics
    - 📊 **Interactive Charts**: Time-series trends and scatter plots
    - 📈 **Daily Summaries**: Aggregated performance metrics
    - 💾 **Data Export**: Download processed data with all calculations
    """)
    
    # 1. File Upload & Data Preparation
    st.subheader("1. 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your chiller plant data file",
        type=["csv", "xls", "xlsx"],
        key="chiller_efficiency_uploader",
        help="Upload Excel template with chiller plant data"
    )
    
    if not uploaded_file:
        st.info("👆 **Upload your chiller plant data file to begin analysis**")
        
        # Show expected data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Your Excel template should contain these columns:**
            - **Date**: Date information
            - **Time**: Time information  
            - **Timestamp**: Combined date/time (optional if Date+Time provided)
            - **Total Chiller Rated kW**: Rated power capacity
            - **Total Chiller Tonnage**: Total cooling capacity (TR)
            - **Operating kW**: Actual power consumption
            - **CHWST °C**: Chilled water supply temperature
            - **CHWRT °C**: Chilled water return temperature
            - **Delta T**: Temperature difference (optional - will be calculated)
            - **FLOWRATE m3hr**: Water flow rate
            - **Efficiency kW/RT**: Efficiency metric (optional - will be calculated)
            - **Chiller %**: Load fraction (optional - will be calculated)
            - **Actual Chiller RT**: Actual cooling load (optional - will be calculated)
            
            **📝 Notes:**
            - Missing calculated columns will be computed automatically
            - Rows with FLOWRATE ≤ 0 or Delta T ≤ 0 will be ignored
            - Column names should match exactly (case-sensitive)
            """)
        return
    
    # Read and process uploaded file
    try:
        df = read_uploaded_file(uploaded_file)
        
        if df is None or df.empty:
            st.error("❌ Failed to read the uploaded file or file is empty.")
            return
            
        st.success(f"✅ File uploaded successfully! Found {len(df)} rows of data.")
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.markdown(f"**Available columns:** {', '.join(df.columns)}")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))  
            with col3:
                numeric_cols = df.select_dtypes(include=['number']).columns
                st.metric("Numeric Columns", len(numeric_cols))
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return
    
    # Process timestamp column
    timestamp_col = None
    if 'Timestamp' in df.columns:
        timestamp_col = 'Timestamp'
    elif 'Date' in df.columns and 'Time' in df.columns:
        # Combine Date and Time columns
        df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        timestamp_col = 'Timestamp'
    else:
        st.error("❌ No timestamp column found. Please ensure your data has either 'Timestamp' or 'Date'+'Time' columns.")
        return
    
    # Process timestamp using existing pattern
    if isinstance(df[timestamp_col].dtype, object):
        df[timestamp_col] = preprocess_timestamp_column(df[timestamp_col])
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=[timestamp_col])
    
    if df.empty:
        st.error("❌ No valid timestamp data found after processing.")
        return
    
    # Calculate derived columns
    df = calculate_chiller_derived_columns(df)
    
    # Store processed data in session state following existing pattern
    st.session_state["chiller_efficiency_df"] = df
    
    # 2. Date Range Filters
    st.subheader("2. 📅 Date Range Filter")
    
    min_date = df[timestamp_col].min().date()
    max_date = df[timestamp_col].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="chiller_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="chiller_end_date"
        )
    
    # Filter data by date range
    mask = (df[timestamp_col].dt.date >= start_date) & (df[timestamp_col].dt.date <= end_date)
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected date range.")
        return
    
    st.info(f"📊 Filtered data: {len(df_filtered):,} rows from {start_date} to {end_date}")
    
    # 3. KPI Metrics Row
    st.subheader("3. 📊 Key Performance Indicators")
    
    # Calculate KPIs using filtered data
    if 'kW_per_RT' in df_filtered.columns and not df_filtered['kW_per_RT'].isna().all():
        avg_efficiency = df_filtered['kW_per_RT'].mean()
        avg_load_pct = df_filtered['Load_pct'].mean()
        min_load_pct = df_filtered['Load_pct'].min() 
        max_load_pct = df_filtered['Load_pct'].max()
        avg_tr = df_filtered['TR'].mean()
        
        # Display KPIs in columns following existing pattern
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Avg kW/RT",
                f"{avg_efficiency:.2f}",
                help="Average energy efficiency (lower is better)"
            )
        
        with col2:
            st.metric(
                "Avg Load %", 
                f"{avg_load_pct:.1f}%",
                help="Average chiller loading percentage"
            )
        
        with col3:
            st.metric(
                "Min Load %",
                f"{min_load_pct:.1f}%", 
                help="Minimum chiller loading"
            )
        
        with col4:
            st.metric(
                "Max Load %",
                f"{max_load_pct:.1f}%",
                help="Maximum chiller loading"
            )
        
        with col5:
            st.metric(
                "Avg TR",
                f"{avg_tr:.1f}",
                help="Average cooling load in tons of refrigeration"
            )
    else:
        st.warning("⚠️ Cannot calculate KPIs - missing required calculated columns")
    
    # 4. Time-Series Charts
    st.subheader("4. 📈 Performance Trends")
    
    # Chart 1: Chiller kW Trend
    st.markdown("#### 4.1 Chiller Power Consumption Trend")
    if 'Operating kW' in df_filtered.columns:
        fig1 = px.line(
            df_filtered, 
            x=timestamp_col, 
            y='Operating kW',
            title='Chiller Power Consumption Over Time',
            labels={'Operating kW': 'Power (kW)', timestamp_col: 'Time'}
        )
        fig1.update_layout(height=400, showlegend=False)
        fig1.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠️ Operating kW column not found")
    
    # Chart 2: Flow Rate Trend  
    st.markdown("#### 4.2 Chiller Flow Rate Trend")
    if 'FLOWRATE m3hr' in df_filtered.columns:
        fig2 = px.line(
            df_filtered,
            x=timestamp_col,
            y='FLOWRATE m3hr', 
            title='Chiller Water Flow Rate Over Time',
            labels={'FLOWRATE m3hr': 'Flow Rate (m³/hr)', timestamp_col: 'Time'}
        )
        fig2.update_layout(height=400, showlegend=False)
        fig2.update_traces(line_color='#2ca02c')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ FLOWRATE m3hr column not found")
    
    # Chart 3: Supply & Return Temperature Trends
    st.markdown("#### 4.3 Chilled Water Supply & Return Temperature Trends")
    if 'CHWST ᵒC' in df_filtered.columns and 'CHWRT ᵒC' in df_filtered.columns:
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=df_filtered[timestamp_col],
            y=df_filtered['CHWST ᵒC'],
            mode='lines',
            name='Supply Temp (CHWST)',
            line=dict(color='#1f77b4'),
            hovertemplate='Supply: %{y:.1f}°C<br>%{x}<extra></extra>'
        ))
        
        fig3.add_trace(go.Scatter(
            x=df_filtered[timestamp_col], 
            y=df_filtered['CHWRT ᵒC'],
            mode='lines',
            name='Return Temp (CHWRT)',
            line=dict(color='#ff7f0e'),
            hovertemplate='Return: %{y:.1f}°C<br>%{x}<extra></extra>'
        ))
        
        fig3.update_layout(
            title='Chilled Water Supply & Return Temperature',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ Temperature columns (CHWST ᵒC, CHWRT ᵒC) not found")
    
    # Chart 4: Efficiency Trend
    st.markdown("#### 4.4 Chiller Efficiency (kW/RT) Trend")
    if 'kW_per_RT' in df_filtered.columns:
        fig4 = px.line(
            df_filtered,
            x=timestamp_col,
            y='kW_per_RT',
            title='Chiller Efficiency Over Time (Lower is Better)',
            labels={'kW_per_RT': 'Efficiency (kW/RT)', timestamp_col: 'Time'}
        )
        fig4.update_layout(height=400, showlegend=False)
        fig4.update_traces(line_color='#d62728')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("⚠️ kW/RT efficiency column not available")
    
    # Chart 5: Load Trend (Dual Series)
    st.markdown("#### 4.5 Chiller Load Trend (TR & Percentage)")
    if 'TR' in df_filtered.columns and 'Load_pct' in df_filtered.columns:
        # Create subplot with secondary y-axis
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig5.add_trace(
            go.Scatter(
                x=df_filtered[timestamp_col],
                y=df_filtered['TR'],
                mode='lines',
                name='Cooling Load (TR)',
                line=dict(color='#9467bd'),
                hovertemplate='Load: %{y:.1f} TR<br>%{x}<extra></extra>'
            ),
            secondary_y=False,
        )
        
        fig5.add_trace(
            go.Scatter(
                x=df_filtered[timestamp_col],
                y=df_filtered['Load_pct'], 
                mode='lines',
                name='Load Percentage (%)',
                line=dict(color='#8c564b'),
                hovertemplate='Load: %{y:.1f}%<br>%{x}<extra></extra>'
            ),
            secondary_y=True,
        )
        
        fig5.update_xaxes(title_text="Time")
        fig5.update_yaxes(title_text="Cooling Load (TR)", secondary_y=False)
        fig5.update_yaxes(title_text="Load Percentage (%)", secondary_y=True)
        fig5.update_layout(
            title='Chiller Load: Absolute (TR) and Percentage (%)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("⚠️ Load data columns (TR, Load_pct) not available")
    
    # 5. Additional Visualizations
    st.subheader("5. 🎯 Performance Analysis")
    
    # Chart 6: Efficiency vs Load Scatter Plot
    st.markdown("#### 5.1 Efficiency vs Load Performance Map")
    if 'kW_per_RT' in df_filtered.columns and 'Load_pct' in df_filtered.columns:
        fig6 = px.scatter(
            df_filtered,
            x='Load_pct',
            y='kW_per_RT',
            title='Chiller Efficiency vs Load Percentage',
            labels={'Load_pct': 'Load Percentage (%)', 'kW_per_RT': 'Efficiency (kW/RT)'},
            opacity=0.6,
            hover_data=[timestamp_col]
        )
        fig6.update_traces(marker_color='#17becf')
        fig6.update_layout(height=500)
        
        # Add trend line (optional - requires statsmodels)
        if len(df_filtered) > 1:
            try:
                # Try to add OLS trendline
                fig6.add_traces(px.scatter(
                    df_filtered,
                    x='Load_pct', 
                    y='kW_per_RT',
                    trendline='ols'
                ).data[1])
            except ImportError:
                # statsmodels not available - skip trendline
                st.info("ℹ️ Trendline not available (requires statsmodels package)")
            except Exception as e:
                # Other errors - just skip trendline
                pass
        
        st.plotly_chart(fig6, use_container_width=True)
        
        st.info("""
        💡 **Performance Insights:**
        - **Lower kW/RT values** indicate better efficiency
        - **Optimal loading** typically occurs at 70-90% capacity
        - **Part-load efficiency** can reveal equipment sizing issues
        """)
    else:
        st.warning("⚠️ Cannot create efficiency scatter plot - missing data")
    
    # Chart 6.5: Efficiency Indicator Chart (Color-coded Performance)
    st.markdown("#### 5.2 Efficiency Performance Indicator")
    if 'kW_per_RT' in df_filtered.columns:
        # Create efficiency zones based on industry standards
        # Typical zones: Good (< 0.8), Fair (0.8-1.0), Need Improvement (> 1.0)
        st.markdown("""
        **Industry Standard Efficiency Zones:**
        - 🟢 **Good**: kW/RT < 0.8 (Excellent efficiency)
        - 🟡 **Fair**: 0.8 ≤ kW/RT < 1.0 (Acceptable efficiency)
        - 🔴 **Need Improvement**: kW/RT ≥ 1.0 (Poor efficiency - action required)
        """)
        
        # Create figure with color-coded zones
        fig_indicator = go.Figure()
        
        # Define efficiency zones
        def get_efficiency_zone(kw_rt):
            if kw_rt < 0.8:
                return 'Good', '#2ecc71'  # Green
            elif kw_rt < 1.0:
                return 'Fair', '#f39c12'  # Yellow/Orange
            else:
                return 'Need Improvement', '#e74c3c'  # Red
        
        # Add color-coded zones as shapes
        fig_indicator.add_hrect(
            y0=0, y1=0.8,
            fillcolor='rgba(46, 204, 113, 0.2)',
            layer='below',
            line_width=0,
            annotation_text='Good',
            annotation_position='right'
        )
        fig_indicator.add_hrect(
            y0=0.8, y1=1.0,
            fillcolor='rgba(243, 156, 18, 0.2)',
            layer='below',
            line_width=0,
            annotation_text='Fair',
            annotation_position='right'
        )
        fig_indicator.add_hrect(
            y0=1.0, y1=df_filtered['kW_per_RT'].max() * 1.1,
            fillcolor='rgba(231, 76, 60, 0.2)',
            layer='below',
            line_width=0,
            annotation_text='Need Improvement',
            annotation_position='right'
        )
        
        # Add the actual kW/RT data as line
        fig_indicator.add_trace(go.Scatter(
            x=df_filtered[timestamp_col],
            y=df_filtered['kW_per_RT'],
            mode='lines+markers',
            name='Actual kW/RT',
            line=dict(color='#2c3e50', width=2),
            marker=dict(size=4, color='#2c3e50'),
            hovertemplate='<b>%{x}</b><br>kW/RT: %{y:.3f}<extra></extra>'
        ))
        
        # Add horizontal reference lines
        fig_indicator.add_hline(
            y=0.8, 
            line_dash='dash', 
            line_color='green',
            annotation_text='Target: 0.8 kW/RT',
            annotation_position='bottom right'
        )
        fig_indicator.add_hline(
            y=1.0, 
            line_dash='dash', 
            line_color='orange',
            annotation_text='Warning: 1.0 kW/RT',
            annotation_position='bottom right'
        )
        
        # Calculate statistics for each zone
        good_count = len(df_filtered[df_filtered['kW_per_RT'] < 0.8])
        fair_count = len(df_filtered[(df_filtered['kW_per_RT'] >= 0.8) & (df_filtered['kW_per_RT'] < 1.0)])
        poor_count = len(df_filtered[df_filtered['kW_per_RT'] >= 1.0])
        total_count = len(df_filtered)
        
        fig_indicator.update_layout(
            title='Chiller Plant Efficiency Performance Over Time<br><sub>Color zones indicate performance levels based on industry standards</sub>',
            xaxis_title='Date/Time',
            yaxis_title='Efficiency (kW/RT)',
            height=600,
            hovermode='x unified',
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        st.plotly_chart(fig_indicator, use_container_width=True)
        
        # Display performance statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "🟢 Good Performance",
                f"{good_count:,} readings",
                f"{(good_count/total_count*100):.1f}%",
                delta_color="normal"
            )
        with col2:
            st.metric(
                "🟡 Fair Performance",
                f"{fair_count:,} readings",
                f"{(fair_count/total_count*100):.1f}%",
                delta_color="off"
            )
        with col3:
            st.metric(
                "🔴 Poor Performance",
                f"{poor_count:,} readings",
                f"{(poor_count/total_count*100):.1f}%",
                delta_color="inverse"
            )
        with col4:
            avg_efficiency = df_filtered['kW_per_RT'].mean()
            zone, color = get_efficiency_zone(avg_efficiency)
            st.metric(
                "Average Efficiency",
                f"{avg_efficiency:.3f} kW/RT",
                zone
            )
        
        # Add recommendations based on performance
        st.markdown("---")
        st.markdown("### 📊 Performance Recommendations")
        
        poor_percentage = (poor_count / total_count * 100)
        fair_percentage = (fair_count / total_count * 100)
        
        if poor_percentage > 20:
            st.error(f"""
            ⚠️ **Critical**: {poor_percentage:.1f}% of readings show poor efficiency (kW/RT ≥ 1.0)
            
            **Immediate Actions Required:**
            - Investigate chiller sequencing and loading patterns
            - Check condenser water temperature and flow
            - Verify refrigerant charge levels
            - Review maintenance records for fouling issues
            - Consider chiller plant optimization controls
            """)
        elif fair_percentage > 30:
            st.warning(f"""
            ⚠️ **Attention**: {fair_percentage:.1f}% of readings show fair efficiency (0.8-1.0 kW/RT)
            
            **Recommended Improvements:**
            - Optimize chiller staging sequences
            - Review setpoint temperatures (CHWST/CHWRT)
            - Implement variable speed drives if not present
            - Schedule preventive maintenance
            - Monitor condenser approach temperatures
            """)
        else:
            st.success(f"""
            ✅ **Excellent**: {good_count/total_count*100:.1f}% of readings show good efficiency
            
            **Maintain Performance:**
            - Continue current operational practices
            - Monitor for any degradation trends
            - Keep up with scheduled maintenance
            - Document best practices for training
            """)
    else:
        st.warning("⚠️ Cannot create efficiency indicator chart - missing kW/RT data")
    
    # Chart 7: Daily Summary Table
    st.markdown("#### 5.3 Daily Performance Summary")
    daily_summary = create_daily_summary_table(df_filtered, timestamp_col)
    
    if not daily_summary.empty:
        # Display as formatted table
        st.dataframe(
            daily_summary.style.format({
                'Avg kW/RT': '{:.2f}',
                'Avg Load %': '{:.1f}%', 
                'Total kWh': '{:,.0f}',
                'Total TRh': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Daily Efficiency", f"{daily_summary['Avg kW/RT'].min():.2f} kW/RT")
        with col2:
            st.metric("Highest Daily Load", f"{daily_summary['Avg Load %'].max():.1f}%")
        with col3:
            st.metric("Total Energy Period", f"{daily_summary['Total kWh'].sum():,.0f} kWh")
    else:
        st.warning("⚠️ Cannot generate daily summary - insufficient data")
    
    # 6. Data Download
    st.subheader("6. 💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare processed data for download
        export_df = df_filtered.copy()
        
        # Add metadata columns
        export_df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        export_df['Data_Source'] = uploaded_file.name
        export_df['Filter_Start_Date'] = start_date
        export_df['Filter_End_Date'] = end_date
        
        # Create download buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Processed_Data', index=False)
            daily_summary.to_excel(writer, sheet_name='Daily_Summary', index=False)
        
        st.download_button(
            label="📥 Download Processed Data (Excel)",
            data=buffer.getvalue(),
            file_name=f"chiller_efficiency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download processed data with all calculated columns"
        )
    
    with col2:
        # CSV download option
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"chiller_efficiency_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv",
            help="Download as CSV format"
        )
    
    # Export information
    with st.expander("ℹ️ Export Data Information"):
        st.markdown("""
        **Exported Data Contains:**
        
        **Original Columns:** All your uploaded template data
        
        **Calculated Columns:**
        - `Cooling_kW`: Calculated cooling load (1.163 × Flow × ΔT)
        - `TR`: Cooling load in tons of refrigeration (Cooling_kW ÷ 3.517)
        - `kW_per_RT`: Energy efficiency ratio (Operating kW ÷ TR)
        - `Load_pct`: Load percentage (TR ÷ Total Tonnage × 100)
        - `Chiller_fraction`: Load fraction (0-1 scale)
        
        **Metadata:**
        - `Analysis_Date`: When this analysis was performed
        - `Data_Source`: Original filename
        - `Filter_Start_Date` / `Filter_End_Date`: Applied date filters
        
        **Daily Summary Sheet:**
        - Daily aggregated metrics for trend analysis
        - Total energy consumption and cooling load per day
        """)
    
    st.markdown("---")
    st.success("✅ Chiller Plant Efficiency analysis completed successfully!")
    
    # Performance insights
    with st.expander("💡 Performance Optimization Tips"):
        st.markdown("""
        **Efficiency Optimization Strategies:**
        
        🎯 **Optimal Loading:**
        - Target 70-90% chiller loading for best efficiency
        - Avoid prolonged operation below 30% capacity
        
        🌡️ **Temperature Management:**
        - Maintain consistent ΔT (typically 5-7°C)
        - Higher supply temperature improves efficiency when possible
        
        💧 **Flow Rate Optimization:**
        - Maintain design flow rates for optimal heat transfer
        - Variable flow systems can improve part-load efficiency
        
        📊 **Monitoring KPIs:**
        - Track kW/RT trends to identify performance degradation
        - Monitor load patterns for potential right-sizing opportunities
        - Daily summaries help identify operational patterns
        """)
