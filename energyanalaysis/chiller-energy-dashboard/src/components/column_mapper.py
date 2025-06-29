import streamlit as st
import pandas as pd
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

def validate_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Validate the column mapping and return validation status and errors."""
    errors = []
    numeric_cols = detect_numeric_columns(df)
    
    # Check if time column exists
    if mapping.get('time_column') and mapping['time_column'] not in df.columns:
        errors.append(f"Time column '{mapping['time_column']}' not found in data")
    
    # Validate numeric columns
    required_numeric = ['chiller_power', 'chwp_power', 'cdwp_power', 'cooling_tower_power', 
                       'chw_flow', 'cooling_load']
    
    for field in required_numeric:
        col_name = mapping.get(field)
        if col_name and col_name not in numeric_cols:
            errors.append(f"Column '{col_name}' for {field} must be numeric")
    
    return len(errors) == 0, errors

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