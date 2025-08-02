#!/usr/bin/env python3
"""
Test Enhanced Battery Algorithm - RP4 Peak-Period-Only Discharge
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_battery_algorithm():
    """Test the enhanced battery algorithm for RP4 compliance."""
    
    print("🧪 Testing Enhanced Battery Algorithm")
    print("=" * 60)
    
    try:
        # Import the enhanced battery algorithms
        from battery_algorithms import BatteryAlgorithms
        
        # Create test data
        start_date = datetime(2025, 1, 6)  # Monday
        date_range = pd.date_range(start=start_date, periods=7*24*4, freq='15min')  # 1 week
        
        # Create realistic demand pattern
        np.random.seed(42)
        demand_data = []
        target_demand = 150
        
        for dt in date_range:
            hour = dt.hour
            weekday = dt.weekday()
            
            # Higher demand during RP4 peak periods (2PM-10PM weekdays)
            if weekday < 5 and 14 <= hour < 22:
                demand = 100 + np.random.normal(80, 25)  # Often exceeds target
            else:
                demand = 100 + np.random.normal(30, 15)  # Lower demand
            
            demand_data.append(max(50, demand))
        
        # Create DataFrame
        df = pd.DataFrame({
            'demand_kw': demand_data
        }, index=date_range)
        
        print(f"📊 Test Data:")
        print(f"   • Time period: {df.index[0]} to {df.index[-1]}")
        print(f"   • Data points: {len(df)}")
        print(f"   • Target demand: {target_demand} kW")
        print(f"   • Demand range: {df['demand_kw'].min():.1f} - {df['demand_kw'].max():.1f} kW")
        
        # Battery parameters
        battery_sizing = {
            'capacity_kwh': 100,
            'power_rating_kw': 50
        }
        
        battery_params = {
            'depth_of_discharge': 90,
            'round_trip_efficiency': 92,
            'c_rate': 0.5
        }
        
        interval_hours = 0.25  # 15 minutes
        
        # Test the enhanced algorithm
        print(f"\n🔋 Testing Enhanced Battery Algorithm...")
        
        battery_alg = BatteryAlgorithms()
        
        # Run simulation
        results = battery_alg.simulate_battery_operation(
            df, 'demand_kw', target_demand, battery_sizing, battery_params, interval_hours
        )
        
        print(f"\n📋 Algorithm Test Results:")
        
        if 'validation_results' in results:
            validation = results['validation_results']
            
            print(f"   • Total discharge events: {validation['total_discharges']}")
            print(f"   • Compliant discharges (RP4 peak): {validation['compliant_discharges']}")
            print(f"   • Violation discharges (off-peak): {validation['violation_discharges']}")
            print(f"   • Compliance rate: {validation['compliance_rate']:.1f}%")
            print(f"   • Potential violations logged: {validation['potential_violations_logged']}")
            print(f"   • Holidays considered: {validation['holidays_count']}")
            
            # Check compliance
            if validation['is_fully_compliant']:
                print(f"\n✅ SUCCESS: Algorithm prevents all off-peak discharges!")
                print(f"   🎉 The enhanced algorithm is working correctly")
            else:
                print(f"\n❌ FAILURE: {validation['violation_discharges']} off-peak discharges detected!")
                print(f"   🔧 Algorithm needs further debugging")
                
        else:
            print(f"   ❌ No validation results found in algorithm output")
            
        # Additional metrics
        if 'rp4_peak_periods' in results and 'off_peak_periods' in results:
            print(f"\n📈 Period Analysis:")
            print(f"   • RP4 peak periods: {results['rp4_peak_periods']}")
            print(f"   • Off-peak periods: {results['off_peak_periods']}")
            print(f"   • Total periods: {results['rp4_peak_periods'] + results['off_peak_periods']}")
        
        return validation['is_fully_compliant'] if 'validation_results' in results else False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_battery_algorithm()
    if success:
        print(f"\n🎯 OVERALL RESULT: ✅ PASS - Enhanced algorithm working correctly!")
    else:
        print(f"\n🎯 OVERALL RESULT: ❌ FAIL - Algorithm needs debugging!")
    
    sys.exit(0 if success else 1)
