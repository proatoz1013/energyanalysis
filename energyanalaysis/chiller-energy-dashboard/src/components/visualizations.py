import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Optional

def create_power_trend_chart(df: pd.DataFrame, mapping: Dict[str, str]) -> Optional[go.Figure]:
    """Create power consumption trend chart."""
    fig = go.Figure()
    
    # Add individual power components
    component_mapping = {
        'chiller_power': {'name': 'Chiller Power', 'color': '#FF6B6B'},
        'chwp_power': {'name': 'CHWP Power', 'color': '#4ECDC4'},
        'cdwp_power': {'name': 'CDWP Power', 'color': '#45B7D1'},
        'cooling_tower_power': {'name': 'Cooling Tower', 'color': '#96CEB4'}
    }
    
    time_col = mapping.get('time_column')
    if time_col and time_col in df.columns:
        x_axis = df[time_col]
    else:
        x_axis = df.index
    
    for key, info in component_mapping.items():
        if mapping.get(key) and mapping[key] in df.columns:
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=df[mapping[key]],
                mode='lines',
                name=info['name'],
                line=dict(color=info['color'])
            ))
    
    # Add total power line
    total_power = sum([df[mapping[key]] for key in component_mapping.keys() 
                      if mapping.get(key) and mapping[key] in df.columns])
    
    if len(total_power) > 0:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=total_power,
            mode='lines',
            name='Total Power',
            line=dict(color='#2C3E50', width=3, dash='dash')
        ))
    
    fig.update_layout(
        title='Power Consumption Trend',
        xaxis_title='Time',
        yaxis_title='Power (kW)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_efficiency_trend_chart(df: pd.DataFrame, mapping: Dict[str, str]) -> Optional[go.Figure]:
    """Create efficiency trend chart (kW/TR and COP)."""
    if not mapping.get('cooling_load') or mapping['cooling_load'] not in df.columns:
        return None
        
    # Calculate metrics
    power_components = []
    for component in ['chiller_power', 'chwp_power', 'cdwp_power', 'cooling_tower_power']:
        if mapping.get(component) and mapping[component] in df.columns:
            power_components.append(df[mapping[component]])
    
    if not power_components:
        return None
        
    total_power = pd.concat(power_components, axis=1).sum(axis=1)
    cooling_load = df[mapping['cooling_load']]
    
    # Calculate kW/TR and COP
    kw_tr = total_power / cooling_load
    kw_tr = kw_tr.replace([np.inf, -np.inf], np.nan)
    
    cop = (cooling_load * 3.51685) / total_power
    cop = cop.replace([np.inf, -np.inf], np.nan)
    
    # Get time axis
    time_col = mapping.get('time_column')
    if time_col and time_col in df.columns:
        x_axis = df[time_col]
    else:
        x_axis = df.index
    
    # Create subplot
    fig = go.Figure()
    
    # Add kW/TR trace
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=kw_tr,
        mode='lines',
        name='kW/TR',
        line=dict(color='#E74C3C'),
        yaxis='y'
    ))
    
    # Add COP trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=cop,
        mode='lines',
        name='COP',
        line=dict(color='#2ECC71'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Chiller Plant Efficiency Trends',
        xaxis_title='Time',
        yaxis=dict(
            title='kW/TR',
            side='left',
            color='#E74C3C'
        ),
        yaxis2=dict(
            title='COP',
            side='right',
            overlaying='y',
            color='#2ECC71'
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_cooling_load_chart(df: pd.DataFrame, mapping: Dict[str, str]) -> Optional[go.Figure]:
    """Create cooling load trend chart."""
    if not mapping.get('cooling_load') or mapping['cooling_load'] not in df.columns:
        return None
        
    time_col = mapping.get('time_column')
    if time_col and time_col in df.columns:
        x_axis = df[time_col]
    else:
        x_axis = df.index
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df[mapping['cooling_load']],
        mode='lines',
        name='Cooling Load',
        line=dict(color='#3498DB'),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title='Cooling Load Profile',
        xaxis_title='Time',
        yaxis_title='Cooling Load (TR)',
        height=400
    )
    
    return fig

def create_power_distribution_chart(df: pd.DataFrame, mapping: Dict[str, str]) -> Optional[go.Figure]:
    """Create power distribution pie chart."""
    component_mapping = {
        'chiller_power': 'Chiller Power',
        'chwp_power': 'CHWP Power',
        'cdwp_power': 'CDWP Power',
        'cooling_tower_power': 'Cooling Tower'
    }
    
    values = []
    labels = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (key, label) in enumerate(component_mapping.items()):
        if mapping.get(key) and mapping[key] in df.columns:
            total_power = df[mapping[key]].sum()
            if total_power > 0:
                values.append(total_power)
                labels.append(label)
    
    if not values:
        return None
        
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(values)]),
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Power Distribution by Component',
        height=400
    )
    
    return fig

def create_scatter_plot(df: pd.DataFrame, mapping: Dict[str, str], x_metric: str, y_metric: str) -> Optional[go.Figure]:
    """Create scatter plot for correlation analysis."""
    x_col = mapping.get(x_metric)
    y_col = mapping.get(y_metric)
    
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return None
        
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=f'{y_col} vs {x_col}',
        labels={x_col: x_col, y_col: y_col}
    )
    
    # Add trendline
    fig.add_scatter(
        x=df[x_col],
        y=np.poly1d(np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1))(df[x_col]),
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    )
    
    return fig

def render_visualizations(processed_df, mapping):
    """
    Render all visualizations for the processed data.
    
    Args:
        processed_df (pandas.DataFrame): The processed dataframe with derived metrics
        mapping (dict): Column mapping dictionary
    """
    if processed_df is None or processed_df.empty:
        st.warning("No data available for visualizations.")
        return
    
    st.markdown("### 📊 Data Visualizations")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs(["🔌 Power Trends", "❄️ Efficiency Metrics", "📈 Performance Analysis", "🔍 Correlation Analysis"])
    
    with tab1:
        render_power_trends(processed_df, mapping)
    
    with tab2:
        render_efficiency_metrics(processed_df, mapping)
    
    with tab3:
        render_performance_analysis(processed_df, mapping)
    
    with tab4:
        render_correlation_analysis(processed_df, mapping)

def render_power_trends(df, mapping):
    """Render power consumption trend visualizations."""
    st.markdown("#### 🔌 Power Consumption Trends")
    
    # Power trend chart
    fig = create_power_trend_chart(df, mapping)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Total power over time
    if 'Total_Power_kW' in df.columns:
        fig_total = px.line(
            df, 
            y='Total_Power_kW',
            title="Total Power Consumption Over Time",
            labels={'Total_Power_kW': 'Total Power (kW)', 'index': 'Time'}
        )
        fig_total.update_layout(showlegend=False)
        st.plotly_chart(fig_total, use_container_width=True)

def render_efficiency_metrics(df, mapping):
    """Render efficiency metrics visualizations."""
    st.markdown("#### ❄️ Efficiency Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'kW_per_TR' in df.columns:
            fig_kw_tr = px.line(
                df,
                y='kW_per_TR',
                title="kW/TR Over Time",
                labels={'kW_per_TR': 'kW/TR', 'index': 'Time'}
            )
            fig_kw_tr.update_layout(showlegend=False)
            st.plotly_chart(fig_kw_tr, use_container_width=True)
    
    with col2:
        if 'COP' in df.columns:
            fig_cop = px.line(
                df,
                y='COP',
                title="COP Over Time",
                labels={'COP': 'Coefficient of Performance', 'index': 'Time'}
            )
            fig_cop.update_layout(showlegend=False)
            st.plotly_chart(fig_cop, use_container_width=True)

def render_performance_analysis(df, mapping):
    """Render performance analysis visualizations."""
    st.markdown("#### 📈 Performance Analysis")
    
    cooling_load_col = mapping.get('cooling_load')
    
    if cooling_load_col and 'Total_Power_kW' in df.columns:
        # Scatter plot of cooling load vs power consumption
        fig_scatter = px.scatter(
            df,
            x=cooling_load_col,
            y='Total_Power_kW',
            title="Cooling Load vs Total Power Consumption",
            labels={
                cooling_load_col: 'Cooling Load (TR)',
                'Total_Power_kW': 'Total Power (kW)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Box plot of efficiency metrics
        if 'kW_per_TR' in df.columns:
            fig_box = px.box(
                df,
                y='kW_per_TR',
                title="kW/TR Distribution",
                labels={'kW_per_TR': 'kW/TR'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

def render_correlation_analysis(df, mapping):
    """Render correlation analysis visualizations."""
    st.markdown("#### 🔍 Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_heatmap.update_layout(
            width=800,
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Show correlation values
        with st.expander("📋 Correlation Values", expanded=False):
            st.dataframe(corr_matrix)