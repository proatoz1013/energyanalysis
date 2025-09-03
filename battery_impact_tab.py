"""
Battery Impact Analysis Tab
==========================

This module provides a comprehensive battery impact analysis interface for energy storage systems.
It includes battery selection, financial analysis, and performance visualization tools.

Author: Energy Analysis Team
Version: 1.0
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta


def load_battery_database():
    """Load vendor battery database from JSON file."""
    try:
        with open('vendor_battery_database.json', 'r') as f:
            battery_db = json.load(f)
        return battery_db
    except FileNotFoundError:
        st.error("❌ Battery database file 'vendor_battery_database.json' not found")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error parsing battery database JSON file")
        return None
    except Exception as e:
        st.error(f"❌ Error loading battery database: {str(e)}")
        return None


def calculate_battery_metrics(battery_spec, quantity, target_power_kw, target_energy_kwh):
    """Calculate key battery performance metrics."""
    total_power = battery_spec.get('power_kW', 0) * quantity
    total_energy = battery_spec.get('energy_kWh', 0) * quantity
    
    power_adequacy = total_power >= target_power_kw
    energy_adequacy = total_energy >= target_energy_kwh
    
    utilization_power = (target_power_kw / total_power * 100) if total_power > 0 else 0
    utilization_energy = (target_energy_kwh / total_energy * 100) if total_energy > 0 else 0
    
    return {
        'total_power_kw': total_power,
        'total_energy_kwh': total_energy,
        'power_adequacy': power_adequacy,
        'energy_adequacy': energy_adequacy,
        'utilization_power_pct': utilization_power,
        'utilization_energy_pct': utilization_energy,
        'oversizing_power_pct': max(0, 100 - utilization_power),
        'oversizing_energy_pct': max(0, 100 - utilization_energy)
    }


def calculate_financial_metrics(initial_investment, annual_savings, analysis_years=20):
    """Calculate comprehensive financial metrics."""
    if annual_savings <= 0:
        return {
            'payback_years': float('inf'),
            'npv': -initial_investment,
            'irr': -100,
            'total_savings': 0,
            'roi_pct': -100
        }
    
    payback_years = initial_investment / annual_savings
    
    # Calculate NPV (assuming 5% discount rate)
    discount_rate = 0.05
    npv = sum(annual_savings / ((1 + discount_rate) ** year) for year in range(1, analysis_years + 1)) - initial_investment
    
    # Calculate IRR (simplified approximation)
    irr = (annual_savings / initial_investment * 100) - 5  # Rough approximation
    
    total_savings = annual_savings * analysis_years
    roi_pct = (total_savings / initial_investment - 1) * 100
    
    return {
        'payback_years': payback_years,
        'npv': npv,
        'irr': max(-100, irr),
        'total_savings': total_savings,
        'roi_pct': roi_pct
    }


def render_battery_selection_interface():
    """Render the battery selection interface."""
    st.header("🔋 Battery System Configuration")
    
    # Load battery database
    battery_db = load_battery_database()
    if not battery_db:
        return None, None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Battery selection dropdown
        battery_options = {}
        for battery_id, spec in battery_db.items():
            company = spec.get('company', 'Unknown')
            model = spec.get('model', battery_id)
            capacity = spec.get('energy_kWh', 0)
            power = spec.get('power_kW', 0)
            label = f"{company} {model} ({capacity} kWh / {power} kW)"
            battery_options[label] = {'id': battery_id, 'spec': spec}
        
        selected_battery_label = st.selectbox(
            "Select Battery Model:",
            options=list(battery_options.keys()),
            key="battery_impact_selection"
        )
        
        selected_battery = battery_options[selected_battery_label]
        
        # Quantity selection
        quantity = st.number_input(
            "Number of Units:",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Number of battery units to install"
        )
    
    with col2:
        # Display selected battery specs
        spec = selected_battery['spec']
        st.markdown("**Selected Battery:**")
        st.write(f"**Company:** {spec.get('company', 'N/A')}")
        st.write(f"**Model:** {spec.get('model', 'N/A')}")
        st.write(f"**Capacity:** {spec.get('energy_kWh', 0)} kWh")
        st.write(f"**Power:** {spec.get('power_kW', 0)} kW")
        st.write(f"**Voltage:** {spec.get('voltage_V', 'N/A')} V")
        st.write(f"**Lifespan:** {spec.get('lifespan_years', 'N/A')} years")
    
    return selected_battery, quantity


def render_target_configuration():
    """Render target configuration interface."""
    st.header("🎯 Target Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Power Requirements")
        target_power_kw = st.number_input(
            "Target Power Shaving (kW):",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=5.0,
            help="Maximum power demand reduction target"
        )
        
        power_duration_hours = st.slider(
            "Power Duration (hours):",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            help="How long the power shaving is needed"
        )
    
    with col2:
        st.subheader("Energy Requirements")
        target_energy_kwh = st.number_input(
            "Target Energy Storage (kWh):",
            min_value=0.0,
            max_value=5000.0,
            value=target_power_kw * power_duration_hours,
            step=10.0,
            help="Total energy storage requirement"
        )
        
        cycles_per_day = st.slider(
            "Cycles per Day:",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            help="Expected number of charge/discharge cycles per day"
        )
    
    return {
        'target_power_kw': target_power_kw,
        'target_energy_kwh': target_energy_kwh,
        'power_duration_hours': power_duration_hours,
        'cycles_per_day': cycles_per_day
    }


def render_financial_configuration():
    """Render financial analysis configuration."""
    st.header("💰 Financial Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost Parameters")
        battery_cost_per_kwh = st.number_input(
            "Battery Cost (RM/kWh):",
            min_value=500.0,
            max_value=3000.0,
            value=1400.0,
            step=50.0,
            help="Cost per kWh of battery capacity"
        )
        
        pcs_cost_per_kw = st.number_input(
            "Power Conversion System Cost (RM/kW):",
            min_value=200.0,
            max_value=800.0,
            value=400.0,
            step=25.0,
            help="Cost per kW of power conversion system"
        )
        
        installation_multiplier = st.slider(
            "Installation Cost Multiplier:",
            min_value=1.1,
            max_value=2.0,
            value=1.4,
            step=0.1,
            help="Multiplier for installation, commissioning, and other costs"
        )
    
    with col2:
        st.subheader("Savings Parameters")
        md_rate_rm_per_kw = st.number_input(
            "MD Rate (RM/kW/month):",
            min_value=10.0,
            max_value=100.0,
            value=35.0,
            step=1.0,
            help="Maximum demand charge rate"
        )
        
        energy_savings_rm_per_kwh = st.number_input(
            "Energy Savings (RM/kWh):",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Energy cost savings from peak shaving"
        )
        
        analysis_years = st.slider(
            "Analysis Period (years):",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            help="Financial analysis period"
        )
    
    return {
        'battery_cost_per_kwh': battery_cost_per_kwh,
        'pcs_cost_per_kw': pcs_cost_per_kw,
        'installation_multiplier': installation_multiplier,
        'md_rate_rm_per_kw': md_rate_rm_per_kw,
        'energy_savings_rm_per_kwh': energy_savings_rm_per_kwh,
        'analysis_years': analysis_years
    }


def render_performance_analysis(selected_battery, quantity, targets):
    """Render battery performance analysis."""
    st.header("📊 Performance Analysis")
    
    spec = selected_battery['spec']
    
    # Calculate metrics
    metrics = calculate_battery_metrics(
        spec, quantity, 
        targets['target_power_kw'], 
        targets['target_energy_kwh']
    )
    
    # Display system overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total System Power",
            f"{metrics['total_power_kw']:.0f} kW",
            f"{quantity} × {spec.get('power_kW', 0)} kW"
        )
    
    with col2:
        st.metric(
            "Total System Energy",
            f"{metrics['total_energy_kwh']:.0f} kWh",
            f"{quantity} × {spec.get('energy_kWh', 0)} kWh"
        )
    
    with col3:
        power_status = "✅" if metrics['power_adequacy'] else "❌"
        st.metric(
            "Power Adequacy",
            power_status,
            f"{metrics['utilization_power_pct']:.1f}% utilization"
        )
    
    with col4:
        energy_status = "✅" if metrics['energy_adequacy'] else "❌"
        st.metric(
            "Energy Adequacy",
            energy_status,
            f"{metrics['utilization_energy_pct']:.1f}% utilization"
        )
    
    # Performance charts
    st.subheader("📈 Performance Visualization")
    
    # Create adequacy chart
    adequacy_data = {
        'Requirement': ['Power (kW)', 'Energy (kWh)'],
        'Required': [targets['target_power_kw'], targets['target_energy_kwh']],
        'Available': [metrics['total_power_kw'], metrics['total_energy_kwh']],
        'Utilization (%)': [metrics['utilization_power_pct'], metrics['utilization_energy_pct']]
    }
    
    adequacy_df = pd.DataFrame(adequacy_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for requirements vs available
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name='Required', x=adequacy_df['Requirement'], y=adequacy_df['Required'], marker_color='red'))
        fig1.add_trace(go.Bar(name='Available', x=adequacy_df['Requirement'], y=adequacy_df['Available'], marker_color='green'))
        fig1.update_layout(
            title='Required vs Available Capacity',
            yaxis_title='Capacity',
            barmode='group'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Utilization pie chart
        fig2 = go.Figure(data=[
            go.Pie(
                labels=['Power Utilization', 'Energy Utilization'],
                values=[metrics['utilization_power_pct'], metrics['utilization_energy_pct']],
                hole=.3
            )
        ])
        fig2.update_layout(title='System Utilization (%)')
        st.plotly_chart(fig2, use_container_width=True)
    
    return metrics


def render_financial_analysis(selected_battery, quantity, targets, financial_config, metrics):
    """Render comprehensive financial analysis."""
    st.header("💰 Financial Analysis")
    
    spec = selected_battery['spec']
    
    # Calculate costs
    battery_cost = spec.get('energy_kWh', 0) * quantity * financial_config['battery_cost_per_kwh']
    pcs_cost = spec.get('power_kW', 0) * quantity * financial_config['pcs_cost_per_kw']
    equipment_cost = battery_cost + pcs_cost
    total_investment = equipment_cost * financial_config['installation_multiplier']
    
    # Calculate savings
    monthly_md_savings = targets['target_power_kw'] * financial_config['md_rate_rm_per_kw']
    annual_energy_kwh = targets['target_energy_kwh'] * targets['cycles_per_day'] * 365
    annual_energy_savings = annual_energy_kwh * financial_config['energy_savings_rm_per_kwh']
    annual_md_savings = monthly_md_savings * 12
    total_annual_savings = annual_md_savings + annual_energy_savings
    
    # Calculate financial metrics
    fin_metrics = calculate_financial_metrics(
        total_investment, 
        total_annual_savings, 
        financial_config['analysis_years']
    )
    
    # Display cost breakdown
    st.subheader("💵 Cost Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Battery Cost",
            f"RM {battery_cost:,.0f}",
            f"RM {financial_config['battery_cost_per_kwh']}/kWh"
        )
    
    with col2:
        st.metric(
            "PCS Cost",
            f"RM {pcs_cost:,.0f}",
            f"RM {financial_config['pcs_cost_per_kw']}/kW"
        )
    
    with col3:
        st.metric(
            "Installation & Other",
            f"RM {total_investment - equipment_cost:,.0f}",
            f"{(financial_config['installation_multiplier'] - 1) * 100:.0f}% markup"
        )
    
    with col4:
        st.metric(
            "Total Investment",
            f"RM {total_investment:,.0f}",
            ""
        )
    
    # Display savings breakdown
    st.subheader("💰 Savings Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Monthly MD Savings",
            f"RM {monthly_md_savings:,.0f}",
            f"{targets['target_power_kw']:.0f} kW × RM {financial_config['md_rate_rm_per_kw']}/kW"
        )
    
    with col2:
        st.metric(
            "Annual Energy Savings",
            f"RM {annual_energy_savings:,.0f}",
            f"{annual_energy_kwh:,.0f} kWh/year"
        )
    
    with col3:
        st.metric(
            "Annual MD Savings",
            f"RM {annual_md_savings:,.0f}",
            "12 months"
        )
    
    with col4:
        st.metric(
            "Total Annual Savings",
            f"RM {total_annual_savings:,.0f}",
            ""
        )
    
    # Display financial metrics
    st.subheader("📊 Financial Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        payback_color = "normal" if fin_metrics['payback_years'] <= 10 else "inverse"
        st.metric(
            "Payback Period",
            f"{fin_metrics['payback_years']:.1f} years",
            delta_color=payback_color
        )
    
    with col2:
        npv_color = "normal" if fin_metrics['npv'] > 0 else "inverse"
        st.metric(
            "Net Present Value",
            f"RM {fin_metrics['npv']:,.0f}",
            delta_color=npv_color
        )
    
    with col3:
        roi_color = "normal" if fin_metrics['roi_pct'] > 50 else "inverse"
        st.metric(
            f"{financial_config['analysis_years']}-Year ROI",
            f"{fin_metrics['roi_pct']:.1f}%",
            delta_color=roi_color
        )
    
    with col4:
        st.metric(
            "Total Savings",
            f"RM {fin_metrics['total_savings']:,.0f}",
            f"{financial_config['analysis_years']} years"
        )
    
    # Financial visualization
    st.subheader("📈 Financial Projections")
    
    # Calculate year-by-year cash flow
    years = list(range(0, financial_config['analysis_years'] + 1))
    cumulative_savings = [0]  # Year 0
    
    for year in range(1, financial_config['analysis_years'] + 1):
        cumulative_savings.append(total_annual_savings * year)
    
    # Create cash flow chart
    fig = go.Figure()
    
    # Investment line
    fig.add_trace(go.Scatter(
        x=years,
        y=[total_investment] * len(years),
        mode='lines',
        name='Initial Investment',
        line=dict(color='red', dash='dash')
    ))
    
    # Cumulative savings line
    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_savings,
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='green')
    ))
    
    # Break-even point
    if fin_metrics['payback_years'] <= financial_config['analysis_years']:
        fig.add_vline(
            x=fin_metrics['payback_years'],
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Break-even: {fin_metrics['payback_years']:.1f} years"
        )
    
    fig.update_layout(
        title='Investment Recovery Timeline',
        xaxis_title='Years',
        yaxis_title='Amount (RM)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Investment recommendation
    st.subheader("🎯 Investment Recommendation")
    
    if fin_metrics['payback_years'] <= 7:
        st.success(f"""
        ✅ **HIGHLY RECOMMENDED INVESTMENT**
        
        **Strong Financial Case:**
        - Excellent payback period: {fin_metrics['payback_years']:.1f} years
        - Positive NPV: RM {fin_metrics['npv']:,.0f}
        - Strong ROI: {fin_metrics['roi_pct']:.1f}%
        
        **Risk Level:** Low - Strong financial returns expected
        """)
    elif fin_metrics['payback_years'] <= 12:
        st.info(f"""
        💡 **RECOMMENDED INVESTMENT**
        
        **Good Financial Case:**
        - Acceptable payback period: {fin_metrics['payback_years']:.1f} years
        - NPV: RM {fin_metrics['npv']:,.0f}
        - ROI: {fin_metrics['roi_pct']:.1f}%
        
        **Risk Level:** Moderate - Consider operational benefits
        """)
    else:
        st.warning(f"""
        ⚠️ **MARGINAL INVESTMENT**
        
        **Weak Financial Case:**
        - Long payback period: {fin_metrics['payback_years']:.1f} years
        - NPV: RM {fin_metrics['npv']:,.0f}
        - ROI: {fin_metrics['roi_pct']:.1f}%
        
        **Risk Level:** High - Consider alternative solutions
        """)
    
    return {
        'total_investment': total_investment,
        'total_annual_savings': total_annual_savings,
        'financial_metrics': fin_metrics
    }


def show_battery_impact_analysis():
    """Main function to display the Battery Impact Analysis tab."""
    st.title("🔋 Battery Impact Analysis")
    st.markdown("""
    **Comprehensive battery energy storage system analysis and impact assessment.**
    
    This tool provides detailed analysis of:
    - 🔋 **Battery Selection & Sizing** - Choose optimal battery configuration
    - 🎯 **Performance Analysis** - Evaluate system adequacy and utilization
    - 💰 **Financial Assessment** - Calculate ROI, payback, and savings
    - 📊 **Impact Visualization** - Visualize battery system performance
    
    Configure your requirements and analyze the impact of different battery solutions.
    """)
    
    # Check if battery database is available
    battery_db = load_battery_database()
    if not battery_db:
        st.error("Cannot proceed without battery database. Please ensure 'vendor_battery_database.json' exists.")
        return
    
    # Step 1: Battery Selection
    selected_battery, quantity = render_battery_selection_interface()
    if not selected_battery:
        return
    
    st.markdown("---")
    
    # Step 2: Target Configuration
    targets = render_target_configuration()
    
    st.markdown("---")
    
    # Step 3: Financial Configuration
    financial_config = render_financial_configuration()
    
    st.markdown("---")
    
    # Step 4: Performance Analysis
    metrics = render_performance_analysis(selected_battery, quantity, targets)
    
    st.markdown("---")
    
    # Step 5: Financial Analysis
    financial_results = render_financial_analysis(
        selected_battery, quantity, targets, financial_config, metrics
    )
    
    st.markdown("---")
    
    # Step 6: Summary Report
    st.header("📋 Executive Summary")
    
    spec = selected_battery['spec']
    
    summary_data = {
        'Configuration': [
            f"{quantity} × {spec.get('company', '')} {spec.get('model', '')}",
            f"{metrics['total_power_kw']:.0f} kW / {metrics['total_energy_kwh']:.0f} kWh",
            f"Power Adequacy: {'✅' if metrics['power_adequacy'] else '❌'}",
            f"Energy Adequacy: {'✅' if metrics['energy_adequacy'] else '❌'}"
        ],
        'Financial': [
            f"Total Investment: RM {financial_results['total_investment']:,.0f}",
            f"Annual Savings: RM {financial_results['total_annual_savings']:,.0f}",
            f"Payback Period: {financial_results['financial_metrics']['payback_years']:.1f} years",
            f"15-Year ROI: {financial_results['financial_metrics']['roi_pct']:.1f}%"
        ],
        'Performance': [
            f"Power Utilization: {metrics['utilization_power_pct']:.1f}%",
            f"Energy Utilization: {metrics['utilization_energy_pct']:.1f}%",
            f"Target Power: {targets['target_power_kw']:.0f} kW",
            f"Target Energy: {targets['target_energy_kwh']:.0f} kWh"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Detailed Report"):
            # Create detailed report data
            report_data = {
                'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Battery Model': [f"{spec.get('company', '')} {spec.get('model', '')}"],
                'Quantity': [quantity],
                'Total Power (kW)': [metrics['total_power_kw']],
                'Total Energy (kWh)': [metrics['total_energy_kwh']],
                'Target Power (kW)': [targets['target_power_kw']],
                'Target Energy (kWh)': [targets['target_energy_kwh']],
                'Power Adequacy': [metrics['power_adequacy']],
                'Energy Adequacy': [metrics['energy_adequacy']],
                'Total Investment (RM)': [financial_results['total_investment']],
                'Annual Savings (RM)': [financial_results['total_annual_savings']],
                'Payback Years': [financial_results['financial_metrics']['payback_years']],
                'NPV (RM)': [financial_results['financial_metrics']['npv']],
                'ROI (%)': [financial_results['financial_metrics']['roi_pct']]
            }
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="Download Detailed Report (CSV)",
                data=csv,
                file_name=f"battery_impact_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Generate Summary Report"):
            summary_text = f"""
Battery Impact Analysis Summary Report
====================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BATTERY CONFIGURATION
-------------------
Selected Battery: {spec.get('company', '')} {spec.get('model', '')}
Quantity: {quantity} units
Total System: {metrics['total_power_kw']:.0f} kW / {metrics['total_energy_kwh']:.0f} kWh

PERFORMANCE ANALYSIS
------------------
Target Power: {targets['target_power_kw']:.0f} kW
Target Energy: {targets['target_energy_kwh']:.0f} kWh
Power Adequacy: {'✅ Adequate' if metrics['power_adequacy'] else '❌ Inadequate'}
Energy Adequacy: {'✅ Adequate' if metrics['energy_adequacy'] else '❌ Inadequate'}
Power Utilization: {metrics['utilization_power_pct']:.1f}%
Energy Utilization: {metrics['utilization_energy_pct']:.1f}%

FINANCIAL ANALYSIS
----------------
Total Investment: RM {financial_results['total_investment']:,.0f}
Annual Savings: RM {financial_results['total_annual_savings']:,.0f}
Payback Period: {financial_results['financial_metrics']['payback_years']:.1f} years
NPV: RM {financial_results['financial_metrics']['npv']:,.0f}
15-Year ROI: {financial_results['financial_metrics']['roi_pct']:.1f}%

RECOMMENDATION
-------------
{'✅ RECOMMENDED' if financial_results['financial_metrics']['payback_years'] <= 10 else '⚠️ MARGINAL' if financial_results['financial_metrics']['payback_years'] <= 15 else '❌ NOT RECOMMENDED'}

Generated by Battery Impact Analysis Tool
"""
            
            st.download_button(
                label="Download Summary Report (TXT)",
                data=summary_text,
                file_name=f"battery_impact_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


# Main function for compatibility
def show():
    """Compatibility function that calls the main render function."""
    show_battery_impact_analysis()


if __name__ == "__main__":
    # For testing purposes
    show_battery_impact_analysis()
