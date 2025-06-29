import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple

def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns in the dataframe."""
    return df.select_dtypes(include=['number']).columns.tolist()

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect datetime or potentially datetime columns."""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Also check for string columns that might contain datetime data
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(10)
        if len(sample_values) > 0:
            # Try to parse a few values to see if they're datetime-like
            try:
                pd.to_datetime(sample_values.iloc[0])
                datetime_cols.append(col)
            except:
                pass
    
    return datetime_cols

def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest column mappings based on column names."""
    suggestions = {}
    cols_lower = [col.lower() for col in df.columns]
    
    # Time column suggestions
    time_keywords = ['time', 'date', 'timestamp', 'datetime']
    for i, col_lower in enumerate(cols_lower):
        if any(keyword in col_lower for keyword in time_keywords):
            suggestions['timestamp'] = df.columns[i]
            break
    
    # Power column suggestions
    chiller_keywords = ['chiller', 'ch_', 'ch1', 'ch2', 'ch3']
    pump_keywords = ['pump', 'chwp', 'cdwp', 'chw_pump', 'cdw_pump']
    ct_keywords = ['cooling_tower', 'ct_', 'tower', 'fan']
    
    for i, col_lower in enumerate(cols_lower):
        col_name = df.columns[i]
        
        # Look for power/kw indicators
        if 'kw' in col_lower or 'power' in col_lower:
            if any(keyword in col_lower for keyword in chiller_keywords):
                if 'chillers' not in suggestions:
                    suggestions['chillers'] = {}
                chiller_num = 1  # Default to chiller 1
                if 'chiller_1' not in suggestions['chillers']:
                    suggestions['chillers']['chiller_1'] = {'power': col_name}
            elif any(keyword in col_lower for keyword in pump_keywords):
                if 'pumps' not in suggestions:
                    suggestions['pumps'] = {}
                pump_type = 'chwp' if 'chwp' in col_lower or 'chw' in col_lower else 'cdwp'
                if f'{pump_type}_1' not in suggestions['pumps']:
                    suggestions['pumps'][f'{pump_type}_1'] = {'power': col_name}
            elif any(keyword in col_lower for keyword in ct_keywords):
                suggestions['cooling_tower_power'] = col_name
        
        # Look for flow rate indicators
        if ('flow' in col_lower or 'gpm' in col_lower) and 'chw' in col_lower:
            suggestions['chw_flow'] = col_name
        
        # Look for cooling load indicators
        if 'tr' in col_lower or 'ton' in col_lower or 'load' in col_lower:
            suggestions['cooling_load'] = col_name
        
        # Look for temperature indicators
        if 'temp' in col_lower:
            if 'chw' in col_lower and 'supply' in col_lower:
                suggestions['chws_temp'] = col_name
            elif 'chw' in col_lower and 'return' in col_lower:
                suggestions['chwr_temp'] = col_name
            elif 'cdw' in col_lower and 'supply' in col_lower:
                suggestions['cdws_temp'] = col_name
            elif 'cdw' in col_lower and 'return' in col_lower:
                suggestions['cdwr_temp'] = col_name
    
    return suggestions

def render_column_mapper(df: pd.DataFrame) -> Optional[Dict]:
    """Render enhanced column mapping interface supporting multiple chillers and pumps."""
    if df is None:
        st.warning("No data available for column mapping. Please upload a file first.")
        return None
    
    st.subheader("🔗 Advanced Column Mapping")
    st.markdown("Map your data columns to support multiple chillers, pumps, and system parameters:")
    
    # Get available columns
    all_columns = ["None"] + df.columns.tolist()
    numeric_columns = ["None"] + detect_numeric_columns(df)
    datetime_columns = ["None"] + detect_datetime_columns(df)
    
    # Get suggestions
    suggestions = suggest_column_mapping(df)
    
    # Display data info
    with st.expander("📊 Data Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns", len(df.columns))
            st.write("**Sample Columns:**")
            for col in df.columns[:5]:
                st.write(f"• {col}")
        with col2:
            st.metric("Numeric Columns", len(numeric_columns)-1)
            st.write("**Numeric Columns:**")
            for col in numeric_columns[1:6]:  # Skip "None" and show first 5
                st.write(f"• {col}")
        with col3:
            st.metric("Data Rows", len(df))
            st.write("**Time Columns:**")
            for col in datetime_columns[1:4]:  # Skip "None" and show first 3
                st.write(f"• {col}")
    
    # Initialize mapping dictionary
    mapping = {}
    
    # Timestamp mapping
    st.markdown("### ⏰ Time Data")
    timestamp_col = st.selectbox(
        "Select Timestamp Column",
        options=datetime_columns + all_columns,
        index=0,
        help="Choose the column containing timestamp information"
    )
    mapping['timestamp'] = timestamp_col if timestamp_col != "None" else None
    
    # Chillers Section
    st.markdown("### ❄️ Chiller Configuration")
    
    # Number of chillers
    num_chillers = st.number_input("Number of Chillers", min_value=1, max_value=10, value=1, step=1)
    
    chillers = {}
    chiller_tabs = st.tabs([f"Chiller {i+1}" for i in range(num_chillers)])
    
    for i, tab in enumerate(chiller_tabs):
        with tab:
            chiller_id = f"chiller_{i+1}"
            chillers[chiller_id] = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Chiller {i+1} - Power & Load**")
                
                # Chiller power
                power_col = st.selectbox(
                    f"Power (kW)",
                    options=numeric_columns,
                    key=f"ch{i+1}_power",
                    help=f"Power consumption of Chiller {i+1}"
                )
                chillers[chiller_id]['power'] = power_col if power_col != "None" else None
                
                # Cooling load
                load_col = st.selectbox(
                    f"Cooling Load (TR)",
                    options=numeric_columns,
                    key=f"ch{i+1}_load",
                    help=f"Cooling load output of Chiller {i+1}"
                )
                chillers[chiller_id]['cooling_load'] = load_col if load_col != "None" else None
                
                # Efficiency
                eff_col = st.selectbox(
                    f"Efficiency (kW/TR) - Optional",
                    options=numeric_columns,
                    key=f"ch{i+1}_eff",
                    help=f"Pre-calculated efficiency of Chiller {i+1}"
                )
                chillers[chiller_id]['efficiency'] = eff_col if eff_col != "None" else None
            
            with col2:
                st.markdown(f"**Chiller {i+1} - Temperatures**")
                
                # CHW Supply Temperature
                chws_temp = st.selectbox(
                    f"CHW Supply Temp (°C)",
                    options=numeric_columns,
                    key=f"ch{i+1}_chws_temp",
                    help=f"Chilled water supply temperature from Chiller {i+1}"
                )
                chillers[chiller_id]['chws_temp'] = chws_temp if chws_temp != "None" else None
                
                # CHW Return Temperature
                chwr_temp = st.selectbox(
                    f"CHW Return Temp (°C)",
                    options=numeric_columns,
                    key=f"ch{i+1}_chwr_temp",
                    help=f"Chilled water return temperature to Chiller {i+1}"
                )
                chillers[chiller_id]['chwr_temp'] = chwr_temp if chwr_temp != "None" else None
                
                # CDW Supply Temperature
                cdws_temp = st.selectbox(
                    f"CDW Supply Temp (°C)",
                    options=numeric_columns,
                    key=f"ch{i+1}_cdws_temp",
                    help=f"Condenser water supply temperature from Chiller {i+1}"
                )
                chillers[chiller_id]['cdws_temp'] = cdws_temp if cdws_temp != "None" else None
                
                # CDW Return Temperature
                cdwr_temp = st.selectbox(
                    f"CDW Return Temp (°C)",
                    options=numeric_columns,
                    key=f"ch{i+1}_cdwr_temp",
                    help=f"Condenser water return temperature to Chiller {i+1}"
                )
                chillers[chiller_id]['cdwr_temp'] = cdwr_temp if cdwr_temp != "None" else None
    
    mapping['chillers'] = chillers
    
    # Pumps Section
    st.markdown("### 💧 Pump Configuration")
    
    pump_types = ['CHWP (Chilled Water Pump)', 'CDWP (Condenser Water Pump)']
    selected_pump_types = st.multiselect("Select Pump Types", pump_types, default=pump_types)
    
    pumps = {}
    
    for pump_type in selected_pump_types:
        pump_key = 'chwp' if 'CHWP' in pump_type else 'cdwp'
        st.markdown(f"#### {pump_type}")
        
        # Number of pumps of this type
        num_pumps = st.number_input(
            f"Number of {pump_key.upper()}s", 
            min_value=1, max_value=5, value=1, step=1,
            key=f"num_{pump_key}"
        )
        
        pump_tabs = st.tabs([f"{pump_key.upper()} {i+1}" for i in range(num_pumps)])
        
        for i, tab in enumerate(pump_tabs):
            with tab:
                pump_id = f"{pump_key}_{i+1}"
                pumps[pump_id] = {}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{pump_key.upper()} {i+1} - Power & Performance**")
                    
                    # Pump power
                    power_col = st.selectbox(
                        f"Power (kW)",
                        options=numeric_columns,
                        key=f"{pump_id}_power",
                        help=f"Power consumption of {pump_key.upper()} {i+1}"
                    )
                    pumps[pump_id]['power'] = power_col if power_col != "None" else None
                    
                    # Flow rate
                    flow_col = st.selectbox(
                        f"Flow Rate (GPM/LPM)",
                        options=numeric_columns,
                        key=f"{pump_id}_flow",
                        help=f"Flow rate of {pump_key.upper()} {i+1}"
                    )
                    pumps[pump_id]['flow_rate'] = flow_col if flow_col != "None" else None
                
                with col2:
                    st.markdown(f"**{pump_key.upper()} {i+1} - System Parameters**")
                    
                    # Head
                    head_col = st.selectbox(
                        f"Head (ft/m)",
                        options=numeric_columns,
                        key=f"{pump_id}_head",
                        help=f"Pump head of {pump_key.upper()} {i+1}"
                    )
                    pumps[pump_id]['head'] = head_col if head_col != "None" else None
                    
                    # Speed/Frequency
                    speed_col = st.selectbox(
                        f"Speed/Frequency (Hz/RPM)",
                        options=numeric_columns,
                        key=f"{pump_id}_speed",
                        help=f"Operating speed/frequency of {pump_key.upper()} {i+1}"
                    )
                    pumps[pump_id]['speed'] = speed_col if speed_col != "None" else None
    
    mapping['pumps'] = pumps
    
    # System-wide parameters
    st.markdown("### 🏭 System Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cooling Tower & Auxiliary**")
        
        # Cooling tower power
        ct_power = st.selectbox(
            "Cooling Tower Power (kW)",
            options=numeric_columns,
            help="Total cooling tower fan power consumption"
        )
        mapping['cooling_tower_power'] = ct_power if ct_power != "None" else None
        
        # Auxiliary power
        aux_power = st.selectbox(
            "Auxiliary Power (kW)",
            options=numeric_columns,
            help="Other auxiliary equipment power consumption"
        )
        mapping['auxiliary_power'] = aux_power if aux_power != "None" else None
    
    with col2:
        st.markdown("**Environmental Conditions**")
        
        # Ambient temperature
        ambient_temp = st.selectbox(
            "Ambient Temperature (°C)",
            options=numeric_columns,
            help="Outdoor ambient temperature"
        )
        mapping['ambient_temp'] = ambient_temp if ambient_temp != "None" else None
        
        # Humidity
        humidity = st.selectbox(
            "Humidity (%RH)",
            options=numeric_columns,
            help="Relative humidity"
        )
        mapping['humidity'] = humidity if humidity != "None" else None
    
    # Overall cooling load if not specified per chiller
    st.markdown("### 📊 Overall System Load")
    overall_load = st.selectbox(
        "Total Cooling Load (TR) - Optional",
        options=numeric_columns,
        help="Total system cooling load (use if individual chiller loads not available)"
    )
    mapping['total_cooling_load'] = overall_load if overall_load != "None" else None
    
    return mapping

def render_column_mapper(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Render the column mapping interface and return the mapping if valid."""
    st.subheader("📋 Step 3: Column Mapping")
    st.write("Map your data columns to the required chiller plant fields:")
    
    # Get available columns
    all_columns = df.columns.tolist()
    numeric_columns = detect_numeric_columns(df)
    datetime_columns = detect_datetime_columns(df)
    
    # Display column information
    with st.expander("📊 Column Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Columns:**")
            for col in numeric_columns:
                st.write(f"• {col}")
        
        with col2:
            st.write("**Potential Time Columns:**")
            for col in datetime_columns:
                st.write(f"• {col}")
    
    # Column mapping interface
    st.markdown("### Required Field Mappings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Time & Power Columns:**")
        
        # Time column
        time_options = ["None"] + datetime_columns + all_columns
        time_column = st.selectbox(
            "⏰ Time Column",
            options=time_options,
            help="Select the column containing timestamp data"
        )
        if time_column == "None":
            time_column = None
        
        # Chiller power
        chiller_power = st.selectbox(
            "🏭 Chiller Power (kW)",
            options=["None"] + numeric_columns,
            help="Total chiller power consumption"
        )
        if chiller_power == "None":
            chiller_power = None
            
        # CHWP power
        chwp_power = st.selectbox(
            "💧 CHWP Power (kW)",
            options=["None"] + numeric_columns,
            help="Chilled Water Pump power consumption"
        )
        if chwp_power == "None":
            chwp_power = None
            
        # CDWP power
        cdwp_power = st.selectbox(
            "🌊 CDWP Power (kW)",
            options=["None"] + numeric_columns,
            help="Condenser Water Pump power consumption"
        )
        if cdwp_power == "None":
            cdwp_power = None
    
    with col2:
        st.markdown("**System Parameters:**")
        
        # Cooling tower power
        cooling_tower_power = st.selectbox(
            "🌪️ Cooling Tower Power (kW)",
            options=["None"] + numeric_columns,
            help="Cooling tower fan power consumption"
        )
        if cooling_tower_power == "None":
            cooling_tower_power = None
            
        # CHW flow
        chw_flow = st.selectbox(
            "🔄 CHW Flow Rate",
            options=["None"] + numeric_columns,
            help="Chilled water flow rate"
        )
        if chw_flow == "None":
            chw_flow = None
            
        # Cooling load
        cooling_load = st.selectbox(
            "❄️ Cooling Load (TR)",
            options=["None"] + numeric_columns,
            help="Cooling load in tons of refrigeration"
        )
        if cooling_load == "None":
            cooling_load = None
            
        # Plant efficiency (optional)
        plant_efficiency = st.selectbox(
            "⚡ Plant Efficiency (Optional)",
            options=["None"] + numeric_columns,
            help="Pre-calculated plant efficiency values"
        )
        if plant_efficiency == "None":
            plant_efficiency = None
    
    # Create mapping dictionary
    mapping = {
        'time_column': time_column,
        'chiller_power': chiller_power,
        'chwp_power': chwp_power,
        'cdwp_power': cdwp_power,
        'cooling_tower_power': cooling_tower_power,
        'chw_flow': chw_flow,
        'cooling_load': cooling_load,
        'plant_efficiency': plant_efficiency
    }
    
    # Validation
    st.markdown("### Validation")
    
    # Check required fields
    required_fields = ['chiller_power', 'chwp_power', 'cdwp_power', 'cooling_tower_power', 'cooling_load']
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    
    if missing_fields:
        st.error(f"❌ Missing required fields: {', '.join(missing_fields)}")
        return None
    
    # Validate mapping
    is_valid, errors = validate_column_mapping(df, mapping)
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return None
    
    # Show mapping summary
    st.success("✅ Column mapping is valid!")
    
    with st.expander("📝 Mapping Summary", expanded=True):
        for field, column in mapping.items():
            if column:
                field_display = field.replace('_', ' ').title()
                st.write(f"• **{field_display}**: {column}")
    
    return mapping

def calculate_derived_metrics(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Calculate derived metrics based on the column mapping."""
    df_calc = df.copy()
    
    # Calculate total power (kW)
    power_columns = []
    for field in ['chiller_power', 'chwp_power', 'cdwp_power', 'cooling_tower_power']:
        if mapping.get(field):
            power_columns.append(mapping[field])
    
    if power_columns:
        df_calc['Total_Power_kW'] = df_calc[power_columns].sum(axis=1)
    
    # Calculate kW/TR
    if mapping.get('cooling_load') and 'Total_Power_kW' in df_calc.columns:
        df_calc['kW_per_TR'] = df_calc['Total_Power_kW'] / df_calc[mapping['cooling_load']]
        df_calc['kW_per_TR'] = df_calc['kW_per_TR'].replace([float('inf'), -float('inf')], 0)
    
    # Calculate COP
    if mapping.get('cooling_load') and 'Total_Power_kW' in df_calc.columns:
        # COP = (Tons × 3.51685) / Total Power
        df_calc['COP'] = (df_calc[mapping['cooling_load']] * 3.51685) / df_calc['Total_Power_kW']
        df_calc['COP'] = df_calc['COP'].replace([float('inf'), -float('inf')], 0)
    
    return df_calc

def render_column_mapper(df):
    """
    Render the column mapping interface.
    
    Args:
        df (pandas.DataFrame): The dataframe to map columns for
        
    Returns:
        dict: Column mapping dictionary
    """
    if df is None:
        st.warning("No data available for column mapping. Please upload a file first.")
        return {}
    
    st.markdown("### 🔗 Column Mapping")
    st.markdown("Map your data columns to the required fields for analysis:")
    
    # Get available columns
    available_cols = [''] + list(df.columns)
    numeric_cols = [''] + detect_numeric_columns(df)
    datetime_cols = [''] + detect_datetime_columns(df)
    
    # Create mapping interface
    col1, col2 = st.columns(2)
    
    mapping = {}
    
    with col1:
        st.markdown("#### 📅 Time Data")
        mapping['timestamp'] = st.selectbox(
            "Timestamp Column",
            datetime_cols,
            help="Select the column containing timestamp data"
        )
        
        st.markdown("#### ❄️ Cooling Load")
        mapping['cooling_load'] = st.selectbox(
            "Cooling Load (TR)",
            numeric_cols,
            help="Select the column containing cooling load in TR (Tons of Refrigeration)"
        )
    
    with col2:
        st.markdown("#### ⚡ Power Consumption")
        mapping['chiller_power'] = st.selectbox(
            "Chiller Power (kW)",
            numeric_cols,
            help="Select the column containing chiller power consumption"
        )
        
        mapping['pump_power'] = st.selectbox(
            "Pump Power (kW)",
            numeric_cols,
            help="Select the column containing pump power consumption"
        )
        
        mapping['cooling_tower_power'] = st.selectbox(
            "Cooling Tower Power (kW)",
            numeric_cols,
            help="Select the column containing cooling tower power consumption"
        )
        
        mapping['aux_power'] = st.selectbox(
            "Auxiliary Power (kW)",
            numeric_cols,
            help="Select the column containing auxiliary equipment power consumption"
        )
    
    # Validate mapping
    validation, errors = validate_column_mapping(df, mapping)
    
    if not validation and errors:
        st.error("❌ Column mapping validation failed:")
        for error in errors:
            st.error(f"• {error}")
        return {}
    
    # Show mapping summary
    if any(mapping.values()):
        st.markdown("#### 📋 Mapping Summary")
        mapping_df = pd.DataFrame([
            {"Field": k.replace('_', ' ').title(), "Mapped Column": v or "Not mapped"}
            for k, v in mapping.items()
        ])
        st.dataframe(mapping_df, use_container_width=True)
        
        return {k: v for k, v in mapping.items() if v}
    
    return {}

def calculate_derived_metrics(df, mapping):
    """
    Calculate derived metrics based on column mapping.
    
    Args:
        df (pandas.DataFrame): Original dataframe
        mapping (dict): Column mapping dictionary
        
    Returns:
        pandas.DataFrame: Dataframe with derived metrics
    """
    if not mapping:
        return df.copy()
    
    processed_df = df.copy()
    
    # Calculate total power
    power_columns = [
        mapping.get('chiller_power'),
        mapping.get('pump_power'), 
        mapping.get('cooling_tower_power'),
        mapping.get('aux_power')
    ]
    
    # Filter out None values and empty strings
    power_columns = [col for col in power_columns if col and col in df.columns]
    
    if power_columns:
        processed_df['Total_Power_kW'] = processed_df[power_columns].sum(axis=1)
    
    # Calculate efficiency metrics
    cooling_load_col = mapping.get('cooling_load')
    if cooling_load_col and 'Total_Power_kW' in processed_df.columns:
        # kW/TR calculation
        processed_df['kW_per_TR'] = processed_df['Total_Power_kW'] / processed_df[cooling_load_col]
        
        # COP calculation (COP = Cooling Load in kW / Total Power in kW)
        # Convert TR to kW: 1 TR = 3.51685 kW
        processed_df['Cooling_Load_kW'] = processed_df[cooling_load_col] * 3.51685
        processed_df['COP'] = processed_df['Cooling_Load_kW'] / processed_df['Total_Power_kW']
    
    return processed_df