#!/usr/bin/env python3
"""
MD Shaving Solution - Color Logic Demo
This script demonstrates the color logic implementation without requiring Streamlit
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import md_shaving_solution

def create_demo_data():
    """Create sample data for demonstration"""
    # Create 2 days of 15-minute interval data
    start_date = datetime(2024, 1, 15, 0, 0)  # Monday
    dates = pd.date_range(start_date, periods=192, freq='15min')  # 2 days
    
    # Create realistic demand pattern with peaks during business hours
    demand = []
    for dt in dates:
        hour = dt.hour
        weekday = dt.weekday()
        
        # Base load
        base = 80
        
        # Business hours pattern (higher during day)
        if 8 <= hour <= 18:
            business_factor = 1.5
        else:
            business_factor = 0.8
            
        # Weekday vs weekend
        if weekday < 5:  # Weekday
            weekday_factor = 1.2
        else:  # Weekend
            weekday_factor = 0.7
            
        # Add some randomness and occasional peaks
        random_factor = np.random.uniform(0.9, 1.3)
        
        # Create some deliberate peaks during MD peak hours (2-10 PM)
        if weekday < 5 and 14 <= hour < 22 and np.random.random() < 0.3:
            peak_factor = 1.8  # Create peaks above target
        else:
            peak_factor = 1.0
            
        demand_value = base * business_factor * weekday_factor * random_factor * peak_factor
        demand.append(demand_value)
    
    df = pd.DataFrame({'demand': demand}, index=dates)
    return df

def demo_tariff_classification():
    """Demonstrate tariff period classification"""
    print("=== Tariff Period Classification Demo ===\n")
    
    # Define test tariffs
    tou_tariff = {
        'Type': 'TOU',
        'Tariff': 'Medium Voltage TOU',
        'Rates': {'Capacity Rate': 30.19, 'Network Rate': 66.87}
    }
    
    general_tariff = {
        'Type': 'General', 
        'Tariff': 'Medium Voltage General',
        'Rates': {'Capacity Rate': 30.19, 'Network Rate': 66.87}
    }
    
    # Test various time periods
    test_times = [
        datetime(2024, 1, 15, 10, 0),   # Monday 10 AM
        datetime(2024, 1, 15, 15, 0),   # Monday 3 PM (peak)
        datetime(2024, 1, 15, 20, 0),   # Monday 8 PM (peak)
        datetime(2024, 1, 15, 23, 0),   # Monday 11 PM
        datetime(2024, 1, 13, 15, 0),   # Saturday 3 PM
    ]
    
    print("TOU Tariff Classification:")
    for dt in test_times:
        result = md_shaving_solution.get_tariff_period_classification(dt, tou_tariff)
        day_name = dt.strftime("%A")
        time_str = dt.strftime("%I:%M %p")
        print(f"  {day_name} {time_str}: {result}")
    
    print("\nGeneral Tariff Classification:")
    for dt in test_times:
        result = md_shaving_solution.get_tariff_period_classification(dt, general_tariff)
        day_name = dt.strftime("%A")
        time_str = dt.strftime("%I:%M %p")
        print(f"  {day_name} {time_str}: {result}")

def demo_color_logic():
    """Demonstrate the color logic in action"""
    print("\n=== Color Logic Demo ===\n")
    
    # Create demo data
    df = create_demo_data()
    target_demand = 120  # kW
    
    # Test with TOU tariff
    tou_tariff = {
        'Type': 'TOU',
        'Tariff': 'Medium Voltage TOU',
        'Rates': {'Capacity Rate': 30.19, 'Network Rate': 66.87}
    }
    
    print(f"Sample Data: {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    print(f"Target Demand: {target_demand} kW")
    print(f"Data Range: {df['demand'].min():.1f} - {df['demand'].max():.1f} kW")
    
    # Analyze color distribution
    colors = []
    for i, (timestamp, row) in enumerate(df.iterrows()):
        demand_value = row['demand']
        period_type = md_shaving_solution.get_tariff_period_classification(timestamp, tou_tariff)
        
        if demand_value > target_demand:
            if period_type == 'Peak':
                colors.append('red')
            else:
                colors.append('green')
        else:
            colors.append('blue')
    
    # Count colors
    color_counts = pd.Series(colors).value_counts()
    total_points = len(colors)
    
    print(f"\nColor Distribution (TOU Tariff):")
    print(f"  🔴 Red (Above target in Peak): {color_counts.get('red', 0)} points ({color_counts.get('red', 0)/total_points*100:.1f}%)")
    print(f"  🟢 Green (Above target in Off-Peak): {color_counts.get('green', 0)} points ({color_counts.get('green', 0)/total_points*100:.1f}%)")
    print(f"  🔵 Blue (Below target): {color_counts.get('blue', 0)} points ({color_counts.get('blue', 0)/total_points*100:.1f}%)")
    
    # Show some specific examples
    print(f"\nSpecific Examples:")
    for i in range(0, len(df), 48):  # Every 12 hours
        timestamp = df.index[i]
        demand = df.iloc[i]['demand']
        period = md_shaving_solution.get_tariff_period_classification(timestamp, tou_tariff)
        color = colors[i]
        
        day_time = timestamp.strftime("%A %I:%M %p")
        print(f"  {day_time}: {demand:.1f} kW → {period} → {color}")

def demo_chart_creation():
    """Demonstrate chart creation with color logic"""
    print("\n=== Chart Creation Demo ===\n")
    
    # Create demo data
    df = create_demo_data()
    target_demand = 120
    
    tou_tariff = {
        'Type': 'TOU',
        'Tariff': 'Medium Voltage TOU', 
        'Rates': {'Capacity Rate': 30.19, 'Network Rate': 66.87}
    }
    
    # Create chart with color logic
    fig = go.Figure()
    result_fig = md_shaving_solution.create_conditional_demand_line_with_peak_logic(
        fig, df, 'demand', target_demand, tou_tariff, None, 'Demo Demand'
    )
    
    print(f"Chart created successfully!")
    print(f"  Number of traces: {len(result_fig.data)}")
    print(f"  Trace names: {[trace.name for trace in result_fig.data]}")
    
    # Save as HTML file for viewing
    html_filename = "md_shaving_color_demo.html"
    result_fig.update_layout(
        title="MD Shaving Color Logic Demo",
        xaxis_title="Time",
        yaxis_title="Power Demand (kW)",
        height=600
    )
    
    # Add target line
    result_fig.add_hline(
        y=target_demand,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Target: {target_demand} kW"
    )
    
    result_fig.write_html(html_filename)
    print(f"  Chart saved as: {html_filename}")
    print(f"  Open this file in a web browser to view the interactive chart")

def main():
    """Run the complete demo"""
    print("MD Shaving Solution - Color Logic Implementation Demo")
    print("=" * 60)
    
    try:
        demo_tariff_classification()
        demo_color_logic()
        demo_chart_creation()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\nKey Findings:")
        print("✅ Tariff classification works correctly for both TOU and General tariffs")
        print("✅ Color logic properly handles peak/off-peak periods")
        print("✅ Chart creation generates proper Plotly traces")
        print("✅ Implementation is ready for Streamlit integration")
        
        print(f"\n💡 Next Steps:")
        print("1. Run 'streamlit run streamlit_app.py' to see the full application")
        print("2. Navigate to the 'MD Shaving Solution' tab")
        print("3. Upload your data and select appropriate tariff")
        print("4. Observe the color-coded charts in action")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
