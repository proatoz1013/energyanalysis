import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def render_metrics_display(df, mapping):
    """
    Render the metrics display interface.
    
    Args:
        df (pandas.DataFrame): The original dataframe
        mapping (dict): Column mapping dictionary
    """
    if not mapping:
        st.warning("Please complete column mapping first.")
        return
    
    st.markdown("### 📊 Performance Metrics")
    
    # Calculate metrics
    metrics = calculate_efficiency_metrics(df, mapping)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average kW/TR",
            value=f"{metrics.get('avg_kw_tr', 0):.3f}",
            help="Average power consumption per ton of refrigeration"
        )
    
    with col2:
        st.metric(
            label="Average COP",
            value=f"{metrics.get('avg_cop', 0):.2f}",
            help="Average Coefficient of Performance"
        )
    
    with col3:
        st.metric(
            label="Total kW/TR",
            value=f"{metrics.get('total_kw_tr', 0):.3f}",
            help="Total power consumption per ton of refrigeration"
        )
    
    with col4:
        st.metric(
            label="Total COP",
            value=f"{metrics.get('total_cop', 0):.2f}",
            help="Total Coefficient of Performance"
        )
    
    # Show additional details
    with st.expander("📋 Detailed Metrics", expanded=False):
        st.write("**Power Components:**")
        power_cols = ['chiller_power', 'pump_power', 'cooling_tower_power', 'aux_power']
        for col in power_cols:
            if mapping.get(col):
                avg_power = df[mapping[col]].mean()
                st.write(f"- {col.replace('_', ' ').title()}: {avg_power:.2f} kW")

def calculate_total_power_components(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.Series:
    """Calculate total power from individual components."""
    power_components = []
    
    # Check for power components in mapping
    power_fields = ['chiller_power', 'pump_power', 'cooling_tower_power', 'aux_power']
    
    for component in power_fields:
        if mapping.get(component) and mapping[component] in df.columns:
            power_components.append(df[mapping[component]])
    
    if power_components:
        return pd.concat(power_components, axis=1).sum(axis=1)
    else:
        return pd.Series(0, index=df.index)

def calculate_efficiency_metrics(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, float]:
    """Calculate various efficiency metrics."""
    metrics = {}
    
    # Get total power
    total_power = calculate_total_power_components(df, mapping)
    cooling_load_col = mapping.get('cooling_load')
    
    if cooling_load_col and cooling_load_col in df.columns:
        cooling_load = df[cooling_load_col]
        
        # Calculate average kW/TR
        kw_tr_values = total_power / cooling_load
        kw_tr_values = kw_tr_values.replace([np.inf, -np.inf], np.nan)
        metrics['avg_kw_tr'] = kw_tr_values.mean()
        
        # Calculate total kW/TR
        total_power_sum = total_power.sum()
        total_cooling_load = cooling_load.sum()
        if total_cooling_load > 0:
            metrics['total_kw_tr'] = total_power_sum / total_cooling_load
        else:
            metrics['total_kw_tr'] = 0
        
        # Calculate COP (Coefficient of Performance)
        # COP = Cooling Load in kW / Total Power in kW
        # Convert TR to kW: 1 TR = 3.51685 kW
        cooling_load_kw = cooling_load * 3.51685
        cop_values = cooling_load_kw / total_power
        cop_values = cop_values.replace([np.inf, -np.inf], np.nan)
        metrics['avg_cop'] = cop_values.mean()
        
        # Calculate total COP
        total_cooling_load_kw = cooling_load_kw.sum()
        if total_power_sum > 0:
            metrics['total_cop'] = total_cooling_load_kw / total_power_sum
        else:
            metrics['total_cop'] = 0
    else:
        metrics['avg_kw_tr'] = 0
        metrics['total_kw_tr'] = 0
        metrics['avg_cop'] = 0
        metrics['total_cop'] = 0
    
    # Handle NaN values
    for key in metrics:
        if pd.isna(metrics[key]):
            metrics[key] = 0
    
    return metrics