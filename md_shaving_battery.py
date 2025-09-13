import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def md_shaving_battery_page():
    """
    MD Shaving Solutions - Battery Page
    Focused analysis for maximum demand shaving using battery storage solutions
    """
    
    st.title("🔋 MD Shaving Solutions - Battery")
    st.markdown("""
    **Maximum Demand (MD) Shaving with Battery Storage**
    
    This tool helps you determine the optimal battery storage solution for reducing your maximum demand charges.
    Upload your energy consumption data to analyze peak demand patterns and calculate battery requirements for MD shaving.
    
    Key Features:
    - Peak demand pattern analysis
    - Battery sizing recommendations for MD shaving
    - Financial impact assessment
    - Optimal discharge strategies
    - ROI calculations for battery investments
    """)
    
    # Check if data is available from session state (from main app)
    if 'processed_df' in st.session_state and 'power_column' in st.session_state:
        # Use data from main app
        df = st.session_state['processed_df'].copy()
        power_col = st.session_state['power_column']
        
        # Validate that the power column exists in the dataframe
        if power_col not in df.columns:
            st.error(f"❌ Power column '{power_col}' not found in data. Available columns: {list(df.columns)}")
            st.info("💡 Please go back to the 'Load Profile Analysis' tab and select the correct power column.")
            return
        
        st.success("✅ Using data from Load Profile Analysis tab")
        
        # Show data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(df.index.max() - df.index.min()).days} days")
        with col3:
            st.metric("Peak Demand", f"{df[power_col].max():.2f} kW")
            
        # Battery Analysis Section
        st.markdown("---")
        st.subheader("🔋 Battery Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_max_demand = df[power_col].max()
            target_max_demand = st.number_input(
                "Target Maximum Demand (kW)",
                min_value=0.0,
                max_value=current_max_demand,
                value=current_max_demand * 0.85,  # Default to 15% reduction
                step=1.0,
                help="Set your desired maximum demand limit. The battery will discharge to keep demand below this level."
            )
            
        with col2:
            battery_efficiency = st.slider(
                "Battery Round-trip Efficiency (%)",
                min_value=70,
                max_value=95,
                value=85,
                step=1,
                help="Typical battery efficiency including inverter losses"
            ) / 100
            
        with col3:
            md_rate = st.number_input(
                "MD Rate (RM/kW/month)",
                min_value=0.0,
                value=30.0,
                step=1.0,
                help="Maximum demand charge rate from your tariff"
            )
        
        # Calculate peak events that exceed target
        peak_events = df[df[power_col] > target_max_demand].copy()
        
        if len(peak_events) > 0:
            # Calculate excess energy
            peak_events['excess_power'] = peak_events[power_col] - target_max_demand
            
            # Simple battery sizing calculation
            total_excess_energy = peak_events['excess_power'].sum() * 0.25  # Assuming 15-min intervals
            max_excess_power = peak_events['excess_power'].max()
            
            st.markdown("---")
            st.subheader("📊 Peak Demand Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Events", len(peak_events))
            with col2:
                st.metric("Total Excess Energy", f"{total_excess_energy:.1f} kWh")
            with col3:
                st.metric("Max Excess Power", f"{max_excess_power:.1f} kW")
            with col4:
                required_battery = total_excess_energy / battery_efficiency
                st.metric("Required Battery Size", f"{required_battery:.1f} kWh")
            
            # Battery sizing recommendations
            st.markdown("---")
            st.subheader("🔋 Battery Sizing Recommendations")
            
            # Calculate different sizing strategies
            excess_energies = []
            current_excess = 0
            
            for i, row in peak_events.iterrows():
                current_excess += row['excess_power'] * 0.25  # 15-min intervals
                if i == peak_events.index[-1] or peak_events.index[peak_events.index.get_loc(i) + 1] != i + pd.Timedelta(minutes=15):
                    if current_excess > 0:
                        excess_energies.append(current_excess)
                    current_excess = 0
            
            if excess_energies:
                strategies = {
                    "Conservative (50th percentile)": np.percentile(excess_energies, 50),
                    "Moderate (75th percentile)": np.percentile(excess_energies, 75),
                    "Aggressive (90th percentile)": np.percentile(excess_energies, 90),
                    "Maximum Coverage (100%)": max(excess_energies)
                }
                
                for strategy, size in strategies.items():
                    battery_size = size / battery_efficiency
                    coverage = sum(1 for e in excess_energies if e <= size) / len(excess_energies) * 100
                    monthly_savings = min(max_excess_power, size / 0.25) * md_rate
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{strategy}", f"{battery_size:.1f} kWh")
                    with col2:
                        st.metric("Event Coverage", f"{coverage:.0f}%")
                    with col3:
                        st.metric("Est. Monthly Savings", f"RM {monthly_savings:.0f}")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Demand Profile Visualization")
            
            # Create demand profile chart
            fig = go.Figure()
            
            # Add demand profile
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[power_col],
                mode='lines',
                name='Actual Demand',
                line=dict(color='blue', width=1)
            ))
            
            # Add target demand line
            fig.add_hline(
                y=target_max_demand,
                line_dash="dash",
                line_color="red",
                annotation_text="Target Max Demand"
            )
            
            # Highlight peak events
            if len(peak_events) > 0:
                fig.add_trace(go.Scatter(
                    x=peak_events.index,
                    y=peak_events[power_col],
                    mode='markers',
                    name='Peak Events',
                    marker=dict(color='red', size=3)
                ))
            
            fig.update_layout(
                title="Demand Profile with Peak Events",
                xaxis_title="Time",
                yaxis_title="Power (kW)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No peak events found above the target maximum demand.")
            
    else:
        # File upload for this specific page
        st.info("💡 **Tip**: Upload data in the 'Load Profile Analysis' tab first, or upload here for MD shaving analysis only.")
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["csv", "xls", "xlsx"],
            help="Upload your energy consumption data with timestamp and power columns",
            key="md_shaving_battery_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload CSV, XLS, or XLSX files.")
                    return
                
                # Display the first few rows
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                st.subheader("⚙️ Column Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    timestamp_col = st.selectbox(
                        "Select Timestamp Column",
                        df.columns,
                        help="Choose the column containing timestamp data"
                    )
                
                with col2:
                    power_col = st.selectbox(
                        "Select Power Column",
                        df.columns,
                        help="Choose the column containing power consumption data (kW)"
                    )
                
                if st.button("Process Data for Battery Analysis", type="primary"):
                    # Process the data
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    df = df.set_index(timestamp_col)
                    df = df.sort_index()
                    
                    # Store in session state
                    st.session_state['processed_df'] = df
                    st.session_state['power_column'] = power_col
                    
                    st.success("✅ Data processed successfully! The analysis will appear above.")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.markdown("""
            ### How to use this tool:
            
            1. **Upload your data**: CSV file with timestamp and power consumption columns
            2. **Configure columns**: Select the appropriate timestamp and power columns
            3. **Set parameters**: Define your target maximum demand and battery efficiency
            4. **Analyze results**: Review battery sizing recommendations and financial projections
            
            ### Data Requirements:
            - CSV format with headers
            - Timestamp column (various formats supported)
            - Power consumption column in kW
            - Preferably 15-minute or hourly intervals
            """)

if __name__ == "__main__":
    md_shaving_battery_page()
