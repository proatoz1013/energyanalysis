import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta
import io
import xlsxwriter
from typing import Dict, List, Tuple, Optional, Any

# Import existing modules
from tariffs.rp4_tariffs import TARIFF_DATA
from tariffs.peak_logic import MALAYSIA_HOLIDAYS_2025
from utils.cost_calculator import calculate_cost

def _format_rm_value(value: float) -> str:
    """Format RM values with appropriate decimal places and thousand separators."""
    if pd.isna(value) or value is None:
        return "RM0.00"
    if abs(value) >= 1:
        return f"RM{value:,.2f}"
    else:
        return f"RM{value:.4f}"

def _format_number_value(value: float) -> str:
    """Format number values with appropriate decimal places and thousand separators."""
    if pd.isna(value) or value is None:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"

def read_uploaded_file(uploaded_file):
    """Read uploaded CSV or Excel file."""
    if uploaded_file is None:
        return None
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def extract_all_tariffs() -> Dict[str, Dict]:
    """Extract all available tariffs from the existing TARIFF_DATA."""
    all_tariffs = {}
    
    # Process the hierarchical TARIFF_DATA structure
    for sector, sector_data in TARIFF_DATA.items():
        if isinstance(sector_data, dict) and "Tariff Groups" in sector_data:
            tariff_groups = sector_data["Tariff Groups"]
            for group_name, group_data in tariff_groups.items():
                if "Tariffs" in group_data:
                    for tariff in group_data["Tariffs"]:
                        tariff_name = tariff.get("Tariff", "Unknown")
                        tariff_key = f"{sector}_{group_name}_{tariff_name}".replace(" ", "_")
                        
                        all_tariffs[tariff_key] = {
                            'name': f"{sector} - {group_name} - {tariff_name}",
                            'type': sector,
                            'category': group_name,
                            'data': tariff
                        }
    
    return all_tariffs

def show():
    """Main function to display the TNB New Tariff Comparison tool with 5-step process."""
    
    st.title("🏢 TNB New Tariff Comparison")
    st.markdown("---")
    
    st.markdown("""
    **Compare TNB tariffs with comprehensive peak/off-peak analysis** to find the most cost-effective option.
    
    **Follow these 4 steps:**
    1. 📁 Upload your energy consumption data
    2. 🔍 Map columns and process data  
    3. 📈 Compare peak and off-peak consumption by months
    4. 📋 Select tariffs for comparison and view results
    """)
    
    # Initialize session state for data persistence
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = pd.DataFrame()
    if 'selected_tariffs' not in st.session_state:
        st.session_state.selected_tariffs = []
    
    # Progress tracking
    st.markdown("### 📊 Progress Overview")
    
    progress_steps = [
        ("Upload Data", not st.session_state.uploaded_data.empty),
        ("Map Columns", not st.session_state.uploaded_data.empty),
        ("Peak Analysis", not st.session_state.uploaded_data.empty),
        ("Tariff Comparison", len(st.session_state.selected_tariffs) >= 2 and not st.session_state.uploaded_data.empty)
    ]
    
    progress_cols = st.columns(4)
    for i, (step_name, completed) in enumerate(progress_steps):
        with progress_cols[i]:
            if completed:
                st.markdown(f"✅ **{step_name}**")
            else:
                st.markdown(f"⏳ {step_name}")
    
    # Overall progress bar
    completed_steps = sum(1 for _, completed in progress_steps if completed)
    st.progress(completed_steps / len(progress_steps))
    
    st.markdown("---")
    
    # STEP 1: Upload Data
    st.header("📁 Step 1: Upload Data")
    
    st.markdown("""
    **Upload your energy consumption data files** (CSV or Excel format).
    
    📋 **What you need:**
    - ✅ Any file with **date/time** information 
    - ✅ Any file with **energy consumption** data 
    - 🔄 **Demand/Power** data is optional
    
    💡 **Column names don't matter** - you'll map them in the next step!
    
    📁 **Supported formats:** CSV, Excel (.xlsx, .xls)
    """)
    
    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="You can upload multiple files. They will be combined for analysis.",
        key="tariff_comparison_upload"
    )
    
    # Process uploaded files
    raw_files_data = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = read_uploaded_file(uploaded_file)
            if df is not None:
                st.success(f"✅ Successfully loaded '{uploaded_file.name}' ({len(df):,} records)")
                
                # Show data preview first
                with st.expander(f"📋 Raw Data Preview - '{uploaded_file.name}'", expanded=True):
                    st.markdown(f"**Available columns:** {', '.join(df.columns)}")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Data statistics
                    st.markdown("**📊 Data Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with stats_col2:
                        st.metric("Total Columns", len(df.columns))
                    with stats_col3:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        st.metric("Numeric Columns", len(numeric_cols))
                
                # Store raw file data for Step 2
                raw_files_data.append({
                    'name': uploaded_file.name,
                    'data': df
                })
        
        # Store raw files data in session state for Step 2
        if raw_files_data:
            st.session_state.raw_files_data = raw_files_data
            st.success(f"✅ Successfully uploaded {len(raw_files_data)} file(s). Proceed to Step 2 for column mapping.")
        
    # Use raw files data from session state if available
    if 'raw_files_data' not in st.session_state:
        st.session_state.raw_files_data = []
    
    if not st.session_state.raw_files_data:
        st.info("📂 Please upload your energy consumption data files to proceed.")
    
    # Clear any previous processed data when new files are uploaded
    if uploaded_files and raw_files_data:
        st.session_state.uploaded_data = pd.DataFrame()
    
    st.markdown("---")
    
    # STEP 2: Column Mapping and Data Processing
    st.header("🔍 Step 2: Map Columns and Process Data")
    
    if st.session_state.raw_files_data:
        st.markdown("**Map your data columns to required fields for analysis:**")
        
        st.info("""
        📋 **Required mappings:**
        - **DateTime Column**: Column containing date and time information
        - **kW Import (MD) Column**: Column containing power/demand data (kW) - used for Maximum Demand analysis
        - **Energy Consumption Column** (optional): Column containing energy consumption data (kWh)
        """)
        
        # Column mapping for each file
        processed_files_data = []
        
        for file_info in st.session_state.raw_files_data:
            file_name = file_info['name']
            df = file_info['data']
            
            st.subheader(f"📄 Map Columns for '{file_name}'")
            
            # Get available columns
            available_columns = ['-- Select Column --'] + list(df.columns)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                datetime_col = st.selectbox(
                    "📅 DateTime Column",
                    available_columns,
                    key=f"datetime_{file_name}",
                    help="Select the column containing date/time information"
                )
            
            with col2:
                power_col = st.selectbox(
                    "🔋 kW Import (MD) Column",
                    available_columns,
                    key=f"power_{file_name}",
                    help="Select the column containing kW import/demand data (MD)"
                )
            
            with col3:
                energy_col = st.selectbox(
                    "⚡ Energy Consumption Column (Optional)",
                    available_columns,
                    key=f"energy_{file_name}",
                    help="Select the column containing energy consumption data (kWh) - optional"
                )
            
            # Show preview of selected columns
            if datetime_col != '-- Select Column --' and power_col != '-- Select Column --':
                st.markdown("**🔍 Selected Columns Preview:**")
                preview_cols = [datetime_col, power_col]
                if energy_col != '-- Select Column --':
                    preview_cols.append(energy_col)
                
                preview_df = df[preview_cols].head(5)
                st.dataframe(preview_df, use_container_width=True)
                
                # Store mapping info
                processed_files_data.append({
                    'name': file_name,
                    'data': df,
                    'datetime_col': datetime_col,
                    'power_col': power_col,
                    'energy_col': energy_col if energy_col != '-- Select Column --' else None
                })
        
        # Process and combine data automatically
        if processed_files_data:
            all_mapped = all(
                info['datetime_col'] != '-- Select Column --' and 
                info['power_col'] != '-- Select Column --' 
                for info in processed_files_data
            )
            
            if all_mapped:
                # Auto-process when all columns are mapped
                combined_data = []
                
                with st.spinner("🔄 Processing and combining data..."):
                    for file_info in processed_files_data:
                        df = file_info['data'].copy()
                        datetime_col = file_info['datetime_col']
                        energy_col = file_info['energy_col']
                        power_col = file_info['power_col']
                        
                        try:
                            # Create standardized columns
                            processed_df = pd.DataFrame()
                            
                            # Process datetime column
                            processed_df['datetime'] = pd.to_datetime(df[datetime_col])
                            
                            # Process power/demand (required)
                            processed_df['power_demand'] = pd.to_numeric(df[power_col], errors='coerce')
                            
                            # Calculate energy consumption from power demand (kW to kWh)
                            # Assume interval between readings (default to 1 hour if not specified)
                            processed_df = processed_df.sort_values('datetime')
                            time_diff = processed_df['datetime'].diff().dt.total_seconds() / 3600  # Convert to hours
                            time_diff = time_diff.fillna(1.0)  # Default to 1 hour for first row
                            processed_df['energy_consumption'] = processed_df['power_demand'] * time_diff
                            
                            # If energy column is provided, use it instead of calculated
                            if energy_col:
                                energy_data = pd.to_numeric(df[energy_col], errors='coerce')
                                processed_df['energy_consumption'] = energy_data
                            
                            # Add file source
                            processed_df['source_file'] = file_info['name']
                            
                            # Remove rows with invalid data
                            processed_df = processed_df.dropna(subset=['datetime', 'power_demand'])
                            
                            combined_data.append(processed_df)
                            
                            st.success(f"✅ Processed '{file_info['name']}': {len(processed_df):,} valid records")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing '{file_info['name']}': {str(e)}")
                
                if combined_data:
                    # Combine all processed data
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
                    
                    # Store processed data
                    st.session_state.uploaded_data = combined_df
                    
                    st.success(f"🎉 Successfully processed and combined data: {len(combined_df):,} total records")
                    
                    # Show combined data summary
                    st.markdown("**📊 Combined Data Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", f"{len(combined_df):,}")
                    
                    with col2:
                        date_range = f"{combined_df['datetime'].dt.date.min()} to {combined_df['datetime'].dt.date.max()}"
                        st.metric("Date Range", date_range)
                    
                    with col3:
                        if 'energy_consumption' in combined_df.columns:
                            total_energy = f"{combined_df['energy_consumption'].sum():,.0f}"
                            st.metric("Total Energy (kWh)", total_energy)
                        else:
                            st.metric("Max Demand (kW)", f"{combined_df['power_demand'].max():.1f}")
                    
                    with col4:
                        files_count = combined_df['source_file'].nunique()
                        st.metric("Source Files", files_count)
                    
                    # Show processed data preview
                    with st.expander("📋 Processed Data Preview", expanded=True):
                        st.dataframe(combined_df.head(10), use_container_width=True)
            else:
                st.warning("⚠️ Please map the required columns (DateTime and kW Import/MD) for all files.")
    
    else:
        st.info("📂 Please complete Step 1 to upload data first.")
    st.markdown("---")
    
    # STEP 3: Compare Peak and Off-Peak for Months
    st.header("📈 Step 3: Compare Peak and Off-Peak for Months")
    
    if not st.session_state.uploaded_data.empty:
        filtered_df = st.session_state.uploaded_data
        
        st.markdown("**Monthly Peak vs Off-Peak Analysis:**")
        
        # Prepare data for analysis
        analysis_df = filtered_df.copy()
        analysis_df['hour'] = analysis_df['datetime'].dt.hour
        analysis_df['month'] = analysis_df['datetime'].dt.strftime('%Y-%m')
        analysis_df['is_peak'] = (analysis_df['hour'] >= 8) & (analysis_df['hour'] < 22)
        
        # Calculate monthly peak and off-peak consumption
        monthly_analysis = analysis_df.groupby(['month', 'is_peak'])['energy_consumption'].agg([
            ('total_consumption', 'sum'),
            ('avg_consumption', 'mean'),
            ('max_consumption', 'max'),
            ('record_count', 'count')
        ]).reset_index()
        
        monthly_analysis['period'] = monthly_analysis['is_peak'].map({True: 'Peak', False: 'Off-Peak'})
        
        # Create summary table
        monthly_summary = []
        for month in sorted(analysis_df['month'].unique()):
            month_data = monthly_analysis[monthly_analysis['month'] == month]
            peak_data = month_data[month_data['is_peak'] == True]
            offpeak_data = month_data[month_data['is_peak'] == False]
            
            peak_total = peak_data['total_consumption'].iloc[0] if len(peak_data) > 0 else 0
            offpeak_total = offpeak_data['total_consumption'].iloc[0] if len(offpeak_data) > 0 else 0
            peak_avg = peak_data['avg_consumption'].iloc[0] if len(peak_data) > 0 else 0
            offpeak_avg = offpeak_data['avg_consumption'].iloc[0] if len(offpeak_data) > 0 else 0
            peak_records = peak_data['record_count'].iloc[0] if len(peak_data) > 0 else 0
            offpeak_records = offpeak_data['record_count'].iloc[0] if len(offpeak_data) > 0 else 0
            
            total_consumption = peak_total + offpeak_total
            peak_ratio = (peak_total / total_consumption * 100) if total_consumption > 0 else 0
            
            monthly_summary.append({
                'Month': month,
                'Peak Total (kWh)': peak_total,
                'Off-Peak Total (kWh)': offpeak_total,
                'Peak Average (kWh)': peak_avg,
                'Off-Peak Average (kWh)': offpeak_avg,
                'Peak Records': int(peak_records),
                'Off-Peak Records': int(offpeak_records),
                'Peak Ratio (%)': peak_ratio
            })
        
        summary_df = pd.DataFrame(monthly_summary)
        
        # Display summary table
        st.subheader("📊 Monthly Peak vs Off-Peak Summary")
        
        # Format display table
        display_df = summary_df.copy()
        display_df['Peak Total (kWh)'] = display_df['Peak Total (kWh)'].apply(lambda x: f"{x:,.0f}")
        display_df['Off-Peak Total (kWh)'] = display_df['Off-Peak Total (kWh)'].apply(lambda x: f"{x:,.0f}")
        display_df['Peak Average (kWh)'] = display_df['Peak Average (kWh)'].apply(lambda x: f"{x:.2f}")
        display_df['Off-Peak Average (kWh)'] = display_df['Off-Peak Average (kWh)'].apply(lambda x: f"{x:.2f}")
        display_df['Peak Ratio (%)'] = display_df['Peak Ratio (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Create visualizations
        st.subheader("📈 Peak vs Off-Peak Visualization")
        
        # Bar chart comparing peak vs off-peak by month
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            name='Peak Hours (8 AM - 10 PM)',
            x=summary_df['Month'],
            y=summary_df['Peak Total (kWh)'],
            marker_color='#ff7f0e',
            text=summary_df['Peak Total (kWh)'].apply(lambda x: f"{x:,.0f}"),
            textposition='auto'
        ))
        
        fig1.add_trace(go.Bar(
            name='Off-Peak Hours (10 PM - 8 AM)',
            x=summary_df['Month'],
            y=summary_df['Off-Peak Total (kWh)'],
            marker_color='#1f77b4',
            text=summary_df['Off-Peak Total (kWh)'].apply(lambda x: f"{x:,.0f}"),
            textposition='auto'
        ))
        
        fig1.update_layout(
            title='Monthly Peak vs Off-Peak Energy Consumption',
            xaxis_title='Month',
            yaxis_title='Energy Consumption (kWh)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Peak ratio trend line
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=summary_df['Month'],
            y=summary_df['Peak Ratio (%)'],
            mode='lines+markers+text',
            name='Peak Consumption Ratio',
            line=dict(width=3, color='#2ca02c'),
            marker=dict(size=10),
            text=summary_df['Peak Ratio (%)'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center'
        ))
        
        fig2.update_layout(
            title='Peak Hour Consumption Ratio by Month',
            xaxis_title='Month',
            yaxis_title='Peak Ratio (%)',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("📊 Please complete Steps 1 and 2 to view peak/off-peak analysis.")
    
    st.markdown("---")
    
    # STEP 4: Select Comparison Tariffs
    st.header("📋 Step 4: Select Comparison Tariffs")
    
    # Get all available tariffs
    all_tariffs = extract_all_tariffs()
    
    st.markdown("**Select multiple tariffs to compare:**")
    st.info("💡 You can select multiple tariffs for comprehensive comparison analysis.")
    
    # Group tariffs by type for better organization
    tariff_groups = {}
    for tariff_key, tariff_info in all_tariffs.items():
        tariff_type = tariff_info['type']
        if tariff_type not in tariff_groups:
            tariff_groups[tariff_type] = []
        tariff_groups[tariff_type].append((tariff_key, tariff_info['name']))
    
    selected_tariffs = []
    
    # Create selection interface organized by tariff type
    for tariff_type, tariff_list in tariff_groups.items():
        st.subheader(f"🏭 {tariff_type.replace('_', ' ').title()}")
        
        # Create columns for tariffs in this type
        cols = st.columns(min(3, len(tariff_list)))
        for i, (tariff_key, tariff_name) in enumerate(tariff_list):
            with cols[i % len(cols)]:
                if st.checkbox(
                    tariff_name.replace(f"{tariff_type.replace('_', ' ').title()} - ", ""), 
                    key=f"tariff_{tariff_key}"
                ):
                    selected_tariffs.append(tariff_key)
    
    # Store selected tariffs in session state
    st.session_state.selected_tariffs = selected_tariffs
    selected_tariff_names = [all_tariffs[key]['name'] for key in selected_tariffs]
    
    if len(selected_tariffs) < 2:
        st.warning("⚠️ Please select at least 2 tariffs to compare.")
    else:
        st.success(f"✅ Selected {len(selected_tariffs)} tariffs for comparison:")
        for name in selected_tariff_names:
            st.markdown(f"   • {name}")
    
    # Configuration section
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Analysis Period**")
        
        if not st.session_state.uploaded_data.empty:
            available_months = sorted(st.session_state.uploaded_data['datetime'].dt.to_period('M').unique().astype(str))
            default_months = available_months
        else:
            available_months = []
            default_months = []
        
        selected_months = st.multiselect(
            "Select months to analyze",
            available_months,
            default=default_months,
            help="Select specific months for analysis, or leave empty to analyze all available data"
        )
        
        st.markdown("**🏖️ Holiday Configuration**")
        use_default_holidays = st.checkbox(
            "Use Malaysia 2025 holidays",
            value=True,
            help=f"Includes {len(MALAYSIA_HOLIDAYS_2025)} public holidays for Malaysia in 2025"
        )
        
        holidays = MALAYSIA_HOLIDAYS_2025 if use_default_holidays else []
    
    with col2:
        st.markdown("**⚡ AFA (Actual Fuel Adjustment) Rate**")
        
        afa_rate = st.number_input(
            "Actual AFA Rate (sen/kWh)",
            min_value=-10.0,
            max_value=10.0,
            value=-1.10,
            step=0.01,
            help="Current AFA rate applicable to all voltage levels (can be negative)"
        )
        
        st.markdown(
            """
            💡 **Check current rates:** [TNB Official AFA Rates →](https://www.mytnb.com.my/tariff/index.html?v=1.1.43#afa)
            """,
            help="Click to view the latest AFA rates from TNB official website"
        )

    # Additional configuration section
    st.markdown("---")
    st.markdown("**📊 MD Excess Analysis Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        contract_demand_method = st.radio(
            "Contract Demand Reference",
            options=["Auto (% of Peak)", "Manual Entry"],
            index=0,
            help="Choose how to determine contract demand for MD excess calculation"
        )
        
        if contract_demand_method == "Auto (% of Peak)":
            mv_contract_percentage = st.slider(
                "MV Contract Demand (% of Monthly Peak)",
                min_value=70,
                max_value=95,
                value=90,
                step=1,
                help="Typical Medium Voltage contract demand as percentage of monthly peak"
            )
            lv_contract_percentage = st.slider(
                "LV Contract Demand (% of Monthly Peak)",
                min_value=70,
                max_value=95,
                value=85,
                step=1,
                help="Typical Low Voltage contract demand as percentage of monthly peak"
            )
        else:
            st.info("Manual contract demand entry will use fixed values for all months.")
    
    with col2:
        if contract_demand_method == "Manual Entry":
            manual_mv_contract = st.number_input(
                "MV Contract Demand (kW)",
                min_value=0.0,
                value=500.0,
                step=10.0,
                help="Your actual Medium Voltage contract demand limit"
            )
            manual_lv_contract = st.number_input(
                "LV Contract Demand (kW)",
                min_value=0.0,
                value=200.0,
                step=5.0,
                help="Your actual Low Voltage contract demand limit"
            )
    
    # Store configuration in session state
    st.session_state.analysis_config = {
        'months': selected_months if selected_months else None,
        'holidays': holidays,
        'afa_rate': afa_rate / 100  # Convert sen to RM
    }
    
    st.markdown("---")
    
    # STEP 4: Cost Comparison Results
    st.header("💰 Step 4: Cost Comparison Results")
    
    # Check if we can proceed with analysis
    can_analyze = (
        len(st.session_state.selected_tariffs) >= 2 and
        not st.session_state.uploaded_data.empty
    )
    
    if not can_analyze:
        if len(st.session_state.selected_tariffs) < 2:
            st.warning("⚠️ Please select at least 2 tariffs in Step 4.")
        if st.session_state.uploaded_data.empty:
            st.warning("⚠️ Please complete Steps 1 and 2 to prepare your data.")
        
        st.info("Complete the above steps to view cost comparison results.")
    
    else:
        # Auto-calculate when conditions are met
        st.markdown("🔄 **Automatically calculating costs for selected tariffs...**")
        
        # Perform cost analysis
        with st.spinner("🔄 Calculating costs for selected tariffs..."):
            results = {}
            
            for tariff_key in st.session_state.selected_tariffs:
                tariff_info = all_tariffs[tariff_key]
                try:
                    # Get the actual tariff data object
                    tariff_data = tariff_info['data']
                    
                    # Create a proper DataFrame with 'Parsed Timestamp' column for the cost calculator
                    calc_df = st.session_state.uploaded_data.copy()
                    calc_df['Parsed Timestamp'] = calc_df['datetime']
                    
                    # Use existing cost calculator with correct parameters
                    result = calculate_cost(
                        df=calc_df,
                        tariff=tariff_data,
                        power_col='power_demand',
                        holidays=set(st.session_state.analysis_config['holidays']),
                        afa_kwh=0,  # Can be customized later if needed
                        afa_rate=st.session_state.analysis_config.get('afa_rate', -0.011)  # Use configured AFA rate
                    )
                    results[tariff_info['name']] = result
                except Exception as e:
                    st.error(f"Error calculating cost for {tariff_info['name']}: {str(e)}")
                    results[tariff_info['name']] = {}
            
            if any(results.values()):
                st.success("✅ Cost analysis complete! Here are your results:")
                
                # Create detailed monthly breakdown table first
                st.subheader("📅 Monthly Breakdown Analysis")
                
                # Add explanation about MD Excess
                with st.expander("ℹ️ Understanding Maximum Demand (MD)", expanded=False):
                    st.markdown("""
                    **What is Maximum Demand (MD)?**
                    
                    **Maximum Demand** is the highest power demand recorded during a billing period. Different tariff types record MD differently:
                    
                    **In the table below:**
                    - **General MD (kW)**: Maximum demand recorded over the entire month (24/7 recording)
                    - **TOU MD (kW)**: Maximum demand recorded only during TOU peak hours (2PM-10PM weekdays)
                    
                    **Key Differences:**
                    - **General Tariffs**: Record MD continuously (24 hours/day, 7 days/week)
                    - **TOU Tariffs**: Record MD only during peak periods (weekdays 2PM-10PM)
                    - This is why TOU MD is often lower than General MD for the same facility
                    
                    **Why This Matters:**
                    - MD charges are calculated based on the recorded maximum demand
                    - TOU tariffs may offer savings if your peak demand occurs outside TOU recording hours
                    - Understanding your demand patterns helps choose the right tariff
                    """)
                
                # Prepare monthly breakdown data
                monthly_breakdown_data = []
                calc_df = st.session_state.uploaded_data.copy()
                calc_df['month'] = calc_df['datetime'].dt.strftime('%Y-%m')
                calc_df['hour'] = calc_df['datetime'].dt.hour
                calc_df['is_peak'] = (calc_df['hour'] >= 8) & (calc_df['hour'] < 22)
                
                # Group by month and calculate metrics
                for month in sorted(calc_df['month'].unique()):
                    month_data = calc_df[calc_df['month'] == month].copy()
                    
                    # Peak and Off-Peak energy consumption
                    peak_energy = month_data[month_data['is_peak'] == True]['energy_consumption'].sum()
                    offpeak_energy = month_data[month_data['is_peak'] == False]['energy_consumption'].sum()
                    
                    # Calculate Maximum Demand for different tariff types
                    # General MD: Maximum demand over entire month (24/7)
                    general_md = month_data['power_demand'].max()
                    
                    # TOU MD: Maximum demand during peak hours only (2PM-10PM weekdays)
                    # Create TOU peak period mask (weekdays 2PM-10PM)
                    month_data['weekday'] = month_data['datetime'].dt.weekday
                    month_data['hour'] = month_data['datetime'].dt.hour
                    tou_peak_mask = (month_data['weekday'] < 5) & (month_data['hour'] >= 14) & (month_data['hour'] < 22)
                    
                    tou_peak_data = month_data[tou_peak_mask]
                    tou_md = tou_peak_data['power_demand'].max() if len(tou_peak_data) > 0 else 0
                    
                    monthly_breakdown_data.append({
                        'Month': month,
                        'Peak (kWh)': peak_energy,
                        'Off-Peak (kWh)': offpeak_energy,
                        'General MD (kW)': general_md,
                        'TOU MD (kW)': tou_md
                    })
                
                if monthly_breakdown_data:
                    breakdown_df = pd.DataFrame(monthly_breakdown_data)
                    
                    # Format for display
                    display_breakdown_df = breakdown_df.copy()
                    display_breakdown_df['Peak (kWh)'] = display_breakdown_df['Peak (kWh)'].apply(lambda x: f"{x:,.0f}")
                    display_breakdown_df['Off-Peak (kWh)'] = display_breakdown_df['Off-Peak (kWh)'].apply(lambda x: f"{x:,.0f}")
                    display_breakdown_df['General MD (kW)'] = display_breakdown_df['General MD (kW)'].apply(lambda x: f"{x:.1f}")
                    display_breakdown_df['TOU MD (kW)'] = display_breakdown_df['TOU MD (kW)'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(display_breakdown_df, use_container_width=True, hide_index=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_peak = breakdown_df['Peak (kWh)'].sum()
                        st.metric("Total Peak Energy", f"{total_peak:,.0f} kWh")
                    with col2:
                        total_offpeak = breakdown_df['Off-Peak (kWh)'].sum()
                        st.metric("Total Off-Peak Energy", f"{total_offpeak:,.0f} kWh")
                    with col3:
                        overall_max_demand = breakdown_df['General MD (kW)'].max()
                        st.metric("Overall Max Demand", f"{overall_max_demand:.1f} kW")
                    with col4:
                        peak_ratio = (total_peak / (total_peak + total_offpeak) * 100) if (total_peak + total_offpeak) > 0 else 0
                        st.metric("Peak Energy Ratio", f"{peak_ratio:.1f}%")
                
                st.markdown("---")
                
                # Create detailed monthly cost comparison table
                st.subheader("📊 Cost Comparison Summary")
                
                summary_data = []
                
                # Calculate monthly breakdown for each tariff
                for tariff_name, result in results.items():
                    if result:
                        # Calculate monthly costs using the uploaded data
                        calc_df = st.session_state.uploaded_data.copy()
                        calc_df['month'] = calc_df['datetime'].dt.strftime('%Y-%m')
                        
                        # Group by month and calculate costs for each month
                        for month in sorted(calc_df['month'].unique()):
                            month_data = calc_df[calc_df['month'] == month].copy()
                            
                            if len(month_data) > 0:
                                # Prepare month data for cost calculation
                                month_data['Parsed Timestamp'] = month_data['datetime']
                                
                                try:
                                    # Get tariff data for this result
                                    tariff_key = None
                                    for key, info in all_tariffs.items():
                                        if info['name'] == tariff_name:
                                            tariff_key = key
                                            break
                                    
                                    if tariff_key:
                                        tariff_data = all_tariffs[tariff_key]['data']
                                        
                                        # Calculate monthly cost
                                        monthly_result = calculate_cost(
                                            df=month_data,
                                            tariff=tariff_data,
                                            power_col='power_demand',
                                            holidays=set(st.session_state.analysis_config['holidays']),
                                            afa_kwh=0,
                                            afa_rate=st.session_state.analysis_config.get('afa_rate', -0.011)
                                        )
                                        
                                        if monthly_result and 'error' not in monthly_result:
                                            total_cost = monthly_result.get('Total Cost', 0)
                                            
                                            # Calculate total energy cost (handle both General and TOU tariffs)
                                            energy_cost = monthly_result.get('Energy Cost (RM)', 0)  # General tariff
                                            if energy_cost == 0:  # TOU tariff
                                                energy_cost = monthly_result.get('Peak Energy Cost', 0) + monthly_result.get('Off-Peak Energy Cost', 0)
                                            
                                            # Calculate AFA cost
                                            afa_cost = monthly_result.get('AFA Adjustment', 0)
                                            
                                            # Calculate demand cost (Capacity + Network) - handle both key formats
                                            # General tariffs use "Capacity Cost (RM)", TOU tariffs use "Capacity Cost"
                                            capacity_cost = monthly_result.get('Capacity Cost (RM)', 0) or monthly_result.get('Capacity Cost', 0)
                                            network_cost = monthly_result.get('Network Cost (RM)', 0) or monthly_result.get('Network Cost', 0)
                                            demand_cost = capacity_cost + network_cost
                                            
                                            # Service charge (retail cost)
                                            service_charge = monthly_result.get('Retail Cost', 0)
                                            
                                            # 1.6% (KTWBB - Kumpulan Wang Industri Elektrik)
                                            kwie_cost = monthly_result.get('KTWBB Cost', 0)
                                            if kwie_cost == 0:  # Calculate manually if not in result
                                                kwie_cost = energy_cost * 0.016
                                            
                                            # Total energy
                                            total_energy = monthly_result.get('Total kWh', 0)
                                            
                                            # Cost per kWh
                                            avg_cost_kwh = total_cost / total_energy if total_energy > 0 else 0
                                            
                                            summary_row = {
                                                'Tariff': tariff_name,
                                                'Month': month,
                                                'Total Cost (RM)': total_cost,
                                                'Energy Cost (RM)': energy_cost,
                                                'AFA Cost (RM)': afa_cost,
                                                'Capacity + Network (RM)': demand_cost,
                                                'Service Charge (RM)': service_charge,
                                                '1.6% (RM)': kwie_cost,
                                                'Average Cost Per kWh (RM)': avg_cost_kwh
                                            }
                                            summary_data.append(summary_row)
                                        
                                except Exception as e:
                                    st.error(f"Error calculating monthly cost for {tariff_name} - {month}: {str(e)}")
                                    # Add a row with zero values to maintain structure
                                    summary_row = {
                                        'Tariff': tariff_name,
                                        'Month': month,
                                        'Total Cost (RM)': 0,
                                        'Energy Cost (RM)': 0,
                                        'AFA Cost (RM)': 0,
                                        'Capacity + Network (RM)': 0,
                                        'Service Charge (RM)': 0,
                                        '1.6% (RM)': 0,
                                        'Average Cost Per kWh (RM)': 0
                                    }
                                    summary_data.append(summary_row)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Sort by tariff name and then by month
                    summary_df = summary_df.sort_values(['Tariff', 'Month'])
                    
                    # Format for display
                    display_df = summary_df.copy()
                    currency_cols = ['Total Cost (RM)', 'Energy Cost (RM)', 'AFA Cost (RM)', 
                                   'Capacity + Network (RM)', 'Service Charge (RM)', '1.6% (RM)', 
                                   'Average Cost Per kWh (RM)']
                    
                    for col in currency_cols:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(_format_rm_value)
                    
                    # Display the table
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Show best options by month
                    st.markdown("**🏆 Best Options by Month:**")
                    for month in sorted(summary_df['Month'].unique()):
                        month_data = summary_df[summary_df['Month'] == month]
                        if len(month_data) > 0:
                            best_month = month_data.sort_values('Total Cost (RM)').iloc[0]
                            st.markdown(f"- **{month}**: {best_month['Tariff']} - {_format_rm_value(best_month['Total Cost (RM)'])}")
                
                # Summary totals section
                st.subheader("📈 Summary Totals by Tariff")
                
                # Calculate totals for each tariff
                total_summary_data = []
                
                if summary_data:
                    # Group by tariff and sum the monthly costs
                    summary_df_totals = pd.DataFrame(summary_data)
                    tariff_totals = summary_df_totals.groupby('Tariff').agg({
                        'Total Cost (RM)': 'sum',
                        'Energy Cost (RM)': 'sum',
                        'AFA Cost (RM)': 'sum',
                        'Capacity + Network (RM)': 'sum',
                        'Service Charge (RM)': 'sum',
                        '1.6% (RM)': 'sum'
                    }).reset_index()
                    
                    # Calculate additional metrics from original results
                    for tariff_name in tariff_totals['Tariff'].unique():
                        original_result = results.get(tariff_name, {})
                        if original_result:
                            # Get demand values
                            max_demand = original_result.get('Max Demand (kW)', 0)
                            if max_demand == 0:
                                max_demand = original_result.get('Peak Demand (kW, Peak Period Only)', 0)
                            
                            # Total energy
                            total_energy = original_result.get('Total kWh', 0)
                            
                            # Get totals from grouped data
                            tariff_row = tariff_totals[tariff_totals['Tariff'] == tariff_name].iloc[0]
                            total_cost = tariff_row['Total Cost (RM)']
                            
                            # Cost per kWh
                            avg_cost_kwh = total_cost / total_energy if total_energy > 0 else 0
                            
                            total_summary_row = {
                                'Tariff': tariff_name,
                                'Total Cost (RM)': total_cost,
                                'Energy Cost (RM)': tariff_row['Energy Cost (RM)'],
                                'AFA Cost (RM)': tariff_row['AFA Cost (RM)'],
                                'Capacity + Network (RM)': tariff_row['Capacity + Network (RM)'],
                                'Service Charge (RM)': tariff_row['Service Charge (RM)'],
                                '1.6% (RM)': tariff_row['1.6% (RM)'],
                                'Average Cost Per kWh (RM)': avg_cost_kwh,
                                'Peak Demand (kW)': max_demand,
                                'Total Energy (kWh)': total_energy
                            }
                            total_summary_data.append(total_summary_row)
                
                if total_summary_data:
                    total_summary_df = pd.DataFrame(total_summary_data)
                    total_summary_df = total_summary_df.sort_values('Total Cost (RM)')
                    total_summary_df['Rank'] = range(1, len(total_summary_df) + 1)
                    
                    # Reorder columns
                    cols = ['Rank'] + [col for col in total_summary_df.columns if col != 'Rank']
                    total_summary_df = total_summary_df[cols]
                    
                    # Format for display
                    display_totals_df = total_summary_df.copy()
                    currency_cols = ['Total Cost (RM)', 'Energy Cost (RM)', 'AFA Cost (RM)', 
                                   'Capacity + Network (RM)', 'Service Charge (RM)', '1.6% (RM)', 
                                   'Average Cost Per kWh (RM)']
                    
                    for col in currency_cols:
                        if col in display_totals_df.columns:
                            display_totals_df[col] = display_totals_df[col].apply(_format_rm_value)
                    
                    display_totals_df['Peak Demand (kW)'] = display_totals_df['Peak Demand (kW)'].apply(_format_number_value)
                    display_totals_df['Total Energy (kWh)'] = display_totals_df['Total Energy (kWh)'].apply(_format_number_value)
                    
                    st.dataframe(display_totals_df, use_container_width=True, hide_index=True)
                
                # Monthly cost comparison chart
                st.subheader("📈 Monthly Cost Comparison")
                
                # Prepare monthly data
                monthly_costs = {}
                months = set()
                
                for tariff_name, result in results.items():
                    if result and 'monthly_breakdown' in result and result['monthly_breakdown']:
                        monthly_breakdown = result['monthly_breakdown']
                        monthly_costs[tariff_name] = monthly_breakdown
                        months.update(monthly_breakdown.keys())
                
                if monthly_costs and months:
                    months = sorted(list(months))
                    
                    fig = go.Figure()
                    
                    for tariff_name, monthly_data in monthly_costs.items():
                        costs = [monthly_data.get(month, {}).get('total_cost', 0) for month in months]
                        
                        fig.add_trace(go.Scatter(
                            x=months,
                            y=costs,
                            mode='lines+markers+text',
                            name=tariff_name,
                            line=dict(width=3),
                            marker=dict(size=8),
                            text=[_format_rm_value(cost) for cost in costs],
                            textposition='top center'
                        ))
                    
                    fig.update_layout(
                        title='Monthly Cost Comparison Across Tariffs',
                        xaxis_title='Month',
                        yaxis_title='Monthly Cost (RM)',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown chart
                st.subheader("💰 Cost Breakdown Analysis")
                
                tariff_names = []
                energy_costs = []
                demand_costs = []
                service_charges = []
                
                for tariff_name, result in results.items():
                    if result:
                        tariff_names.append(tariff_name)
                        
                        # Extract energy cost (handle both General and TOU tariffs)
                        energy_cost = result.get('Energy Cost (RM)', 0)  # General tariff
                        if energy_cost == 0:  # TOU tariff
                            energy_cost = result.get('Peak Energy Cost', 0) + result.get('Off-Peak Energy Cost', 0)
                        energy_costs.append(energy_cost)
                        
                        # Extract demand cost (Capacity + Network)
                        demand_cost = result.get('Capacity Cost (RM)', 0) + result.get('Network Cost (RM)', 0)
                        demand_costs.append(demand_cost)
                        
                        # Extract service charge (Retail)
                        service_charge = result.get('Retail Cost', 0)
                        service_charges.append(service_charge)
                
                if tariff_names:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Energy Cost',
                        x=tariff_names,
                        y=energy_costs,
                        marker_color='#1f77b4'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Demand Cost',
                        x=tariff_names,
                        y=demand_costs,
                        marker_color='#ff7f0e'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Service Charge',
                        x=tariff_names,
                        y=service_charges,
                        marker_color='#2ca02c'
                    ))
                    
                    fig.update_layout(
                        title='Cost Breakdown by Tariff',
                        xaxis_title='Tariff',
                        yaxis_title='Cost (RM)',
                        barmode='stack',
                        height=500
                    )
                    
                    fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.subheader("📥 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    if summary_data:
                        csv = pd.DataFrame(summary_data).to_csv(index=False)
                        st.download_button(
                            label="📋 Download Summary CSV",
                            data=csv,
                            file_name=f"tnb_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    # Excel export (simplified)
                    if summary_data:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        st.download_button(
                            label="📊 Download Excel Report",
                            data=output.getvalue(),
                            file_name=f"tnb_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            else:
                st.error("❌ Unable to calculate costs. Please check your data and tariff selections.")


if __name__ == "__main__":
    show()
