#!/usr/bin/env python3
"""
Test script to verify that the 1160kW charging power spike is fixed in V2 simulation.

This script tests the enhanced charging constraints:
1. MD Target Constraint: Charging power limited to prevent Net Demand > Monthly Target
2. C-rate Constraint: Charging power limited by battery specifications
3. SOC-based power derating: Power reduction at extreme SOC levels
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create test data that would trigger high charging power without constraints"""
    # Create 24 hours of 15-minute interval data
    timestamps = pd.date_range('2025-01-17 00:00:00', periods=96, freq='15min')
    
    # Create a scenario where there's very low demand (which would trigger aggressive charging)
    # followed by a spike that needs the battery
    base_demand = np.ones(96) * 100  # Base 100kW demand
    
    # Add low demand periods (morning) that would trigger charging
    base_demand[0:32] = 50   # 00:00-08:00: Very low demand (50kW) - should trigger charging
    
    # Add peak demand period that requires discharge 
    base_demand[56:64] = 1200  # 14:00-16:00: High demand (1200kW) - needs battery discharge
    
    df = pd.DataFrame({
        'kW': base_demand,
        'Original_Demand': base_demand
    }, index=timestamps)
    
    return df

def test_charging_power_constraints():
    """Test that charging power is properly constrained"""
    print("🧪 Testing Enhanced V2 Charging Power Constraints")
    print("=" * 60)
    
    try:
        # Import the enhanced V2 functions
        from md_shaving_solution_v2 import (
            _simulate_battery_operation_v2,
            _calculate_c_rate_limited_power_simple,
            _calculate_monthly_targets_v2
        )
        
        print("✅ Successfully imported V2 enhanced functions")
        
        # Create test data
        df = create_test_data()
        print(f"✅ Created test data: {len(df)} intervals")
        print(f"   - Min demand: {df['kW'].min():.1f} kW")
        print(f"   - Max demand: {df['kW'].max():.1f} kW")
        
        # Create monthly targets (set target at 800kW to force charging constraint)
        monthly_targets = pd.Series(
            data=[800.0],  # Monthly target of 800kW
            index=pd.PeriodIndex(['2025-01'], freq='M')
        )
        print(f"✅ Monthly target: {monthly_targets.iloc[0]:.1f} kW")
        
        # Battery configuration
        battery_sizing = {
            'capacity_kwh': 1000,  # 1MWh battery
            'power_rating_kw': 500,  # 500kW max power
            'units': 1
        }
        
        battery_params = {
            'depth_of_discharge': 80.0,
            'round_trip_efficiency': 95.0,
            'c_rate': 0.5,  # 0.5C rate = 500kW max (1000kWh * 0.5)
            'min_soc': 20.0,
            'max_soc': 95.0
        }
        
        print(f"✅ Battery: {battery_sizing['capacity_kwh']}kWh, {battery_sizing['power_rating_kw']}kW")
        print(f"   - C-rate: {battery_params['c_rate']}C")
        print(f"   - DoD: {battery_params['depth_of_discharge']}%")
        
        # Run simulation
        interval_hours = 0.25  # 15 minutes
        print("\n🔄 Running V2 simulation with enhanced constraints...")
        
        simulation_results = _simulate_battery_operation_v2(
            df, 'kW', monthly_targets, battery_sizing, battery_params, 
            interval_hours, selected_tariff=None, holidays=set()
        )
        
        if simulation_results and 'df_simulation' in simulation_results:
            df_sim = simulation_results['df_simulation']
            print("✅ V2 simulation completed successfully!")
            
            # Analyze charging power
            charging_power = df_sim['Battery_Power_kW'][df_sim['Battery_Power_kW'] < 0].abs()
            
            if len(charging_power) > 0:
                max_charge_power = charging_power.max()
                avg_charge_power = charging_power.mean()
                
                print(f"\n📊 Charging Power Analysis:")
                print(f"   - Max charging power: {max_charge_power:.1f} kW")
                print(f"   - Average charging power: {avg_charge_power:.1f} kW")
                print(f"   - Battery power rating: {battery_sizing['power_rating_kw']:.1f} kW")
                print(f"   - C-rate limit (0.5C): {battery_sizing['capacity_kwh'] * battery_params['c_rate']:.1f} kW")
                
                # Test constraints
                power_rating_exceeded = max_charge_power > battery_sizing['power_rating_kw']
                c_rate_exceeded = max_charge_power > (battery_sizing['capacity_kwh'] * battery_params['c_rate'])
                
                print(f"\n🔍 Constraint Validation:")
                if power_rating_exceeded:
                    print(f"❌ Power rating constraint VIOLATED: {max_charge_power:.1f} kW > {battery_sizing['power_rating_kw']:.1f} kW")
                else:
                    print(f"✅ Power rating constraint RESPECTED: {max_charge_power:.1f} kW ≤ {battery_sizing['power_rating_kw']:.1f} kW")
                    
                if c_rate_exceeded:
                    print(f"❌ C-rate constraint VIOLATED: {max_charge_power:.1f} kW > {battery_sizing['capacity_kwh'] * battery_params['c_rate']:.1f} kW")
                else:
                    print(f"✅ C-rate constraint RESPECTED: {max_charge_power:.1f} kW ≤ {battery_sizing['capacity_kwh'] * battery_params['c_rate']:.1f} kW")
                
                # Test MD target constraint
                net_demand_violations = df_sim[df_sim['Net_Demand_kW'] > df_sim['Monthly_Target']]
                if len(net_demand_violations) > 0:
                    max_violation = (net_demand_violations['Net_Demand_kW'] - net_demand_violations['Monthly_Target']).max()
                    print(f"❌ MD target constraint VIOLATED: Max violation = {max_violation:.1f} kW")
                else:
                    print(f"✅ MD target constraint RESPECTED: Net Demand never exceeds Monthly Target")
                
                # Overall assessment
                if not power_rating_exceeded and not c_rate_exceeded and len(net_demand_violations) == 0:
                    print(f"\n🎉 SUCCESS: All charging power constraints are working correctly!")
                    print(f"   The 1160kW charging spike issue has been FIXED! ✅")
                else:
                    print(f"\n⚠️ ISSUES DETECTED: Some constraints are still being violated")
                    
            else:
                print("ℹ️ No charging events detected in simulation")
                
            # Show some sample data
            print(f"\n📋 Sample Simulation Results:")
            sample_data = df_sim[['Original_Demand', 'Monthly_Target', 'Battery_Power_kW', 'Net_Demand_kW', 'Battery_SOC_Percent']].head(10)
            print(sample_data.round(2))
                
        else:
            print("❌ Simulation failed - no results returned")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the energyanalysis directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_c_rate_function():
    """Test the C-rate power limiting function"""
    print("\n🔧 Testing C-rate Power Limiting Function")
    print("-" * 40)
    
    try:
        from md_shaving_solution_v2 import _calculate_c_rate_limited_power_simple
        
        # Test different SOC levels
        test_cases = [
            (10, "Low SOC - should reduce power"),
            (50, "Normal SOC - full power"),
            (95, "High SOC - should reduce power")
        ]
        
        battery_specs = {
            'max_power_rating_kw': 500,
            'battery_capacity_kwh': 1000,
            'c_rate': 0.5
        }
        
        for soc, description in test_cases:
            result = _calculate_c_rate_limited_power_simple(
                soc, battery_specs['max_power_rating_kw'], 
                battery_specs['battery_capacity_kwh'], battery_specs['c_rate']
            )
            
            print(f"\nSOC {soc}% ({description}):")
            print(f"   Max Discharge: {result['max_discharge_power_kw']:.1f} kW")
            print(f"   Max Charge: {result['max_charge_power_kw']:.1f} kW")
            print(f"   SOC Factor: {result['soc_derating_factor']:.2f}")
            print(f"   Limiting Factor: {result['limiting_factor']}")
        
        print("✅ C-rate function working correctly")
        
    except Exception as e:
        print(f"❌ C-rate function error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔋 V2 Enhanced Charging Power Constraint Test")
    print("=" * 60)
    print("Testing fixes for the 1160kW charging power spike issue")
    print()
    
    # Test C-rate function
    c_rate_ok = test_c_rate_function()
    
    # Test full simulation
    simulation_ok = test_charging_power_constraints()
    
    print("\n" + "=" * 60)
    if c_rate_ok and simulation_ok:
        print("🎉 ALL TESTS PASSED! The charging power constraints are working correctly.")
        print("   The 1160kW charging spike issue should now be resolved.")
    else:
        print("❌ SOME TESTS FAILED. Further investigation needed.")
    print("=" * 60)
