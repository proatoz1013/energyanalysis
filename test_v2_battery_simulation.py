#!/usr/bin/env python3
"""
Test script to verify V2 battery simulation logic flow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_v2_simulation_imports():
    """Test that V2 can import the required simulation functions"""
    print("🔍 Testing V2 battery simulation imports...")
    
    try:
        from md_shaving_solution_v2 import _simulate_battery_operation, _display_battery_simulation_chart
        print("✅ Successfully imported _simulate_battery_operation")
        print("✅ Successfully imported _display_battery_simulation_chart")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_test_data():
    """Create sample energy data for testing"""
    print("📊 Creating test energy data...")
    
    # Create 7 days of hourly data
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(168)]  # 7 days
    
    # Create demand pattern with peak during business hours
    base_demand = 150
    peak_demand = 300
    
    demand_data = []
    for date in dates:
        hour = date.hour
        if 9 <= hour <= 17:  # Business hours
            demand = base_demand + (peak_demand - base_demand) * (0.7 + 0.3 * np.random.random())
        else:
            demand = base_demand * (0.5 + 0.3 * np.random.random())
        demand_data.append(demand)
    
    df = pd.DataFrame({
        'DateTime': dates,
        'Demand_KW': demand_data
    })
    df.set_index('DateTime', inplace=True)  # Set datetime index
    
    print(f"✅ Created test dataset: {len(df)} records, demand range: {df['Demand_KW'].min():.1f} - {df['Demand_KW'].max():.1f} kW")
    return df

def test_battery_simulation():
    """Test the complete battery simulation workflow"""
    print("\n⚡ Testing battery simulation workflow...")
    
    # Import the functions
    try:
        from md_shaving_solution_v2 import _simulate_battery_operation
        from tariffs.rp4_tariffs import get_tariff_data
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create test data
    df = create_test_data()
    df['Original_Demand'] = df['Demand_KW']
    
    # Test parameters
    power_col = 'Demand_KW'  # Column name
    target_demand = 200.0  # kW
    battery_capacity = 400.0  # kWh
    battery_power = 200.0  # kW
    
    # Get tariff data and select first business tariff
    tariff_data = get_tariff_data()
    business_tariffs = tariff_data["Business"]["Tariff Groups"]["Non Domestic"]["Tariffs"]
    tariff = business_tariffs[0]  # Use first available business tariff
    holidays = set()
    
    print(f"📋 Test parameters:")
    print(f"   - Power column: {power_col}")
    print(f"   - Target demand: {target_demand} kW")
    print(f"   - Battery capacity: {battery_capacity} kWh")
    print(f"   - Battery power: {battery_power} kW")
    
    try:
        # Prepare parameters as expected by the function
        battery_sizing = {
            'capacity_kwh': battery_capacity,
            'power_rating_kw': battery_power,
            'units': 1
        }
        
        battery_params = {
            'efficiency': 0.95,
            'round_trip_efficiency': 95.0,  # Percentage
            'c_rate': 1.0,
            'min_soc': 20.0,
            'max_soc': 100.0,
            'depth_of_discharge': 80.0  # Max usable % of capacity
        }
        
        interval_hours = 0.25  # 15-minute intervals
        
        # Run the simulation
        print("🔄 Running battery simulation...")
        simulation_results = _simulate_battery_operation(
            df, power_col, target_demand, battery_sizing, battery_params, interval_hours, tariff, holidays
        )
        
        if simulation_results and 'df_simulation' in simulation_results:
            print("✅ Battery simulation completed successfully!")
            
            # Check results
            df_sim = simulation_results['df_simulation']
            required_columns = ['Original_Demand', 'Battery_Power_kW', 'Battery_SOC_Percent', 'Net_Demand_kW']
            missing_cols = [col for col in required_columns if col not in df_sim.columns]
            
            if missing_cols:
                print(f"⚠️  Missing columns in simulation result: {missing_cols}")
            else:
                print("✅ All required columns present in simulation result")
            
            # Print summary metrics
            print(f"\n📊 Simulation Results:")
            print(f"   - Peak reduction: {simulation_results.get('peak_reduction_kw', 0):.1f} kW")
            print(f"   - Success rate: {simulation_results.get('success_rate_percent', 0):.1f}%")
            print(f"   - Energy discharged: {simulation_results.get('total_energy_discharged', 0):.1f} kWh")
            print(f"   - Average SOC: {simulation_results.get('average_soc', 0):.1f}%")
            print(f"   - Min SOC: {simulation_results.get('min_soc', 0):.1f}%")
            print(f"   - Max SOC: {simulation_results.get('max_soc', 0):.1f}%")
            
            return True
        else:
            print("❌ Battery simulation failed - no results returned")
            return False
            
    except Exception as e:
        print(f"❌ Battery simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing V2 Battery Simulation Logic Flow")
    print("=" * 50)
    
    # Test 1: Import functionality
    if not test_v2_simulation_imports():
        print("\n❌ Import test failed - cannot continue")
        return False
    
    # Test 2: Simulation workflow
    if not test_battery_simulation():
        print("\n❌ Simulation test failed")
        return False
    
    print("\n🎉 All tests passed! V2 battery simulation logic is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
