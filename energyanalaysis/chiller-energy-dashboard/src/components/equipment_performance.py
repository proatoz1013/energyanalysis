import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional

def render_equipment_performance(original_df, processed_df, mapping):
    """
    Render equipment performance analysis with detailed breakdowns.
    
    Args:
        original_df (pandas.DataFrame): Original uploaded data
        processed_df (pandas.DataFrame): Processed data with derived metrics
        mapping (dict): Column mapping dictionary
    """
    st.markdown("### 🛠️ Equipment Performance Breakdown")
    st.markdown("Detailed performance analysis for individual equipment components")
    
    # Create tabs for different equipment types
    eq_tabs = st.tabs(["❄️ Chiller", "💧 Pump System", "🌀 Cooling Tower", "📊 Comparative Analysis"])
    
    with eq_tabs[0]:
        render_chiller_performance(original_df, processed_df, mapping)
    
    with eq_tabs[1]:
        render_pump_performance(original_df, processed_df, mapping)
    
    with eq_tabs[2]:
        render_cooling_tower_performance(original_df, processed_df, mapping)
    
    with eq_tabs[3]:
        render_comparative_analysis(original_df, processed_df, mapping)

def render_chiller_performance(original_df, processed_df, mapping):
    """Render chiller-specific performance analysis."""
    st.subheader("❄️ Chiller Performance Analysis")
    
    chiller_power_col = mapping.get('chiller_power')
    cooling_load_col = mapping.get('cooling_load')
    
    if not chiller_power_col and not cooling_load_col:
        st.warning("No chiller performance data available. Please ensure chiller power and cooling load are mapped.")
        return
    
    # Key Performance Metrics
    col1, col2, col3 = st.columns(3)
    
    if chiller_power_col and chiller_power_col in original_df.columns:
        with col1:
            st.markdown("**🔌 Chiller Power**")
            avg_power = original_df[chiller_power_col].mean()
            max_power = original_df[chiller_power_col].max()
            min_power = original_df[chiller_power_col].min()
            
            st.metric("Average", f"{avg_power:.2f} kW")
            st.metric("Maximum", f"{max_power:.2f} kW")
            st.metric("Minimum", f"{min_power:.2f} kW")
    
    if cooling_load_col and cooling_load_col in original_df.columns:
        with col2:
            st.markdown("**❄️ Cooling Load**")
            avg_load = original_df[cooling_load_col].mean()
            max_load = original_df[cooling_load_col].max()
            min_load = original_df[cooling_load_col].min()
            
            st.metric("Average", f"{avg_load:.2f} TR")
            st.metric("Maximum", f"{max_load:.2f} TR")
            st.metric("Minimum", f"{min_load:.2f} TR")
    
    if 'kW_per_TR' in processed_df.columns:
        with col3:
            st.markdown("**⚡ Efficiency**")
            avg_eff = processed_df['kW_per_TR'].mean()
            best_eff = processed_df['kW_per_TR'].min()
            worst_eff = processed_df['kW_per_TR'].max()
            
            st.metric("Avg kW/TR", f"{avg_eff:.3f}")
            st.metric("Best kW/TR", f"{best_eff:.3f}")
            st.metric("Worst kW/TR", f"{worst_eff:.3f}")
    
    # Performance Charts
    st.markdown("#### 📈 Chiller Performance Trends")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if chiller_power_col and chiller_power_col in original_df.columns:
            fig = px.line(original_df.reset_index(), y=chiller_power_col, 
                         title="Chiller Power Consumption Over Time",
                         labels={chiller_power_col: "Power (kW)", "index": "Time"})
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        if 'kW_per_TR' in processed_df.columns:
            fig = px.line(processed_df.reset_index(), y='kW_per_TR',
                         title="Chiller Efficiency (kW/TR) Over Time",
                         labels={'kW_per_TR': 'kW/TR', 'index': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Load vs Power Scatter Plot
    if chiller_power_col and cooling_load_col and both_cols_exist(original_df, [chiller_power_col, cooling_load_col]):
        st.markdown("#### 🎯 Load vs Power Relationship")
        fig_scatter = px.scatter(original_df, x=cooling_load_col, y=chiller_power_col,
                               title="Chiller Power vs Cooling Load",
                               labels={cooling_load_col: "Cooling Load (TR)", 
                                     chiller_power_col: "Chiller Power (kW)"},
                               trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

def render_pump_performance(original_df, processed_df, mapping):
    """Render pump system performance analysis."""
    st.subheader("💧 Pump System Performance Analysis")
    
    pump_power_col = mapping.get('pump_power')
    flow_cols = [col for col in original_df.columns if 'flow' in col.lower() and 'gpm' in col.lower()]
    head_cols = [col for col in original_df.columns if 'head' in col.lower()]
    
    if not pump_power_col and not flow_cols:
        st.warning("No pump performance data available. Please ensure pump power is mapped.")
        return
    
    # Key Performance Metrics
    col1, col2, col3 = st.columns(3)
    
    if pump_power_col and pump_power_col in original_df.columns:
        with col1:
            st.markdown("**🔌 Pump Power**")
            avg_power = original_df[pump_power_col].mean()
            max_power = original_df[pump_power_col].max()
            min_power = original_df[pump_power_col].min()
            
            st.metric("Average", f"{avg_power:.2f} kW")
            st.metric("Maximum", f"{max_power:.2f} kW")
            st.metric("Minimum", f"{min_power:.2f} kW")
    
    if flow_cols:
        flow_col = flow_cols[0]
        with col2:
            st.markdown("**💧 Flow Rate**")
            avg_flow = original_df[flow_col].mean()
            max_flow = original_df[flow_col].max()
            min_flow = original_df[flow_col].min()
            
            st.metric("Average", f"{avg_flow:.1f} GPM")
            st.metric("Maximum", f"{max_flow:.1f} GPM")
            st.metric("Minimum", f"{min_flow:.1f} GPM")
    
    if head_cols:
        head_col = head_cols[0]
        with col3:
            st.markdown("**📏 Head Pressure**")
            avg_head = original_df[head_col].mean()
            max_head = original_df[head_col].max()
            min_head = original_df[head_col].min()
            
            st.metric("Average", f"{avg_head:.1f} ft")
            st.metric("Maximum", f"{max_head:.1f} ft")
            st.metric("Minimum", f"{min_head:.1f} ft")
    
    # Performance Charts
    st.markdown("#### 📈 Pump Performance Trends")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if pump_power_col and pump_power_col in original_df.columns:
            fig = px.line(original_df.reset_index(), y=pump_power_col,
                         title="Pump Power Consumption Over Time",
                         labels={pump_power_col: "Power (kW)", "index": "Time"})
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        if flow_cols:
            fig = px.line(original_df.reset_index(), y=flow_cols[0],
                         title="Pump Flow Rate Over Time",
                         labels={flow_cols[0]: "Flow (GPM)", "index": "Time"})
            st.plotly_chart(fig, use_container_width=True)

def render_cooling_tower_performance(original_df, processed_df, mapping):
    """Render cooling tower performance analysis."""
    st.subheader("🌀 Cooling Tower Performance Analysis")
    
    cooling_tower_power_col = mapping.get('cooling_tower_power')
    temp_cols = [col for col in original_df.columns if 'temp' in col.lower()]
    humidity_cols = [col for col in original_df.columns if 'humidity' in col.lower()]
    
    if not cooling_tower_power_col and not temp_cols:
        st.warning("No cooling tower performance data available. Please ensure cooling tower power is mapped.")
        return
    
    # Key Performance Metrics
    col1, col2, col3 = st.columns(3)
    
    if cooling_tower_power_col and cooling_tower_power_col in original_df.columns:
        with col1:
            st.markdown("**🔌 CT Power**")
            avg_power = original_df[cooling_tower_power_col].mean()
            max_power = original_df[cooling_tower_power_col].max()
            min_power = original_df[cooling_tower_power_col].min()
            
            st.metric("Average", f"{avg_power:.2f} kW")
            st.metric("Maximum", f"{max_power:.2f} kW")
            st.metric("Minimum", f"{min_power:.2f} kW")
    
    if temp_cols:
        temp_col = temp_cols[0]
        with col2:
            st.markdown("**🌡️ Temperature**")
            avg_temp = original_df[temp_col].mean()
            max_temp = original_df[temp_col].max()
            min_temp = original_df[temp_col].min()
            
            st.metric("Average", f"{avg_temp:.1f} °C")
            st.metric("Maximum", f"{max_temp:.1f} °C")
            st.metric("Minimum", f"{min_temp:.1f} °C")
    
    if humidity_cols:
        humidity_col = humidity_cols[0]
        with col3:
            st.markdown("**💨 Humidity**")
            avg_humidity = original_df[humidity_col].mean()
            max_humidity = original_df[humidity_col].max()
            min_humidity = original_df[humidity_col].min()
            
            st.metric("Average", f"{avg_humidity:.1f} %")
            st.metric("Maximum", f"{max_humidity:.1f} %")
            st.metric("Minimum", f"{min_humidity:.1f} %")
    
    # Performance Charts
    st.markdown("#### 📈 Cooling Tower Performance Trends")
    
    if cooling_tower_power_col and cooling_tower_power_col in original_df.columns:
        fig = px.line(original_df.reset_index(), y=cooling_tower_power_col,
                     title="Cooling Tower Power Consumption Over Time",
                     labels={cooling_tower_power_col: "Power (kW)", "index": "Time"})
        st.plotly_chart(fig, use_container_width=True)

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
        power_df = pd.DataFrame(list(power_data.items()), columns=['Equipment', 'Average Power (kW)'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(power_df, x='Equipment', y='Average Power (kW)', 
                           title='Average Power Consumption by Equipment',
                           color='Equipment')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(power_df, values='Average Power (kW)', names='Equipment',
                           title='Power Distribution by Equipment')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Power consumption summary table
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
        
        # Energy Cost Analysis (if applicable)
        st.markdown("#### 💰 Energy Cost Analysis")
        st.info("💡 **Tip**: Consider focusing optimization efforts on equipment with the highest power consumption percentages.")
        
    else:
        st.warning("No power consumption data available for comparative analysis.")
    
    # Equipment Performance Rating
    st.markdown("#### 🏆 Equipment Performance Rating")
    
    if 'kW_per_TR' in processed_df.columns:
        avg_kw_tr = processed_df['kW_per_TR'].mean()
        
        if avg_kw_tr < 0.6:
            rating = "🌟 Excellent"
            color = "green"
            message = "Your chiller plant is operating at excellent efficiency!"
        elif avg_kw_tr < 0.8:
            rating = "✅ Good"
            color = "blue"
            message = "Your chiller plant is operating efficiently with room for minor improvements."
        elif avg_kw_tr < 1.0:
            rating = "⚠️ Average"
            color = "orange"
            message = "Your chiller plant efficiency is average. Consider optimization opportunities."
        else:
            rating = "🔴 Poor"
            color = "red"
            message = "Your chiller plant efficiency needs improvement. Review equipment operation."
        
        st.markdown(f"**Overall Plant Efficiency Rating:** :{color}[{rating}]")
        st.info(message)

def both_cols_exist(df, columns):
    """Check if both columns exist in dataframe."""
    return all(col in df.columns for col in columns)
