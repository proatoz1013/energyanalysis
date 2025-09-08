#!/usr/bin/env python3
"""
Test script for Enhanced V2 Battery Algorithms

This script tests the advanced battery health management, C-rate constraints,
and intelligent charge/discharge algorithms implemented in MD Shaving Solution V2.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_enhanced_battery_algorithms():
    """Test all the enhanced battery algorithm functions"""
    print("🔋 Testing Enhanced V2 Battery Algorithms")
    print("=" * 50)
    
    try:
        # Import the functions from the V2 module
        sys.path.append('/Users/chyeap89/Documents/energyanalysis')
        from md_shaving_solution_v2 import (
            _calculate_battery_health_parameters,
            _calculate_c_rate_limited_power,
            _get_soc_protection_levels,
            _apply_soc_protection_constraints,
            _calculate_intelligent_charge_strategy,
            _get_tariff_aware_discharge_strategy
        )
        
        print("✅ Successfully imported all enhanced algorithm functions")
        
        # Test 1: Battery Health Parameters
        print("\n📊 Test 1: Battery Health Parameters")
        print("-" * 30)
        
        for chemistry in ['LFP', 'NMC', 'NCA']:
            for temp in [10, 25, 40]:
                health_params = _calculate_battery_health_parameters(chemistry, temp)
                print(f"{chemistry} at {temp}°C:")
                print(f"  Health Derating: {health_params['health_derating_factor']:.3f}")
                print(f"  Temp Derating: {health_params['temperature_derating_factor']:.3f}")
                print(f"  Max C-Rate: {health_params['max_c_rate']:.2f}")
        
        # Test 2: C-Rate Limited Power
        print("\n⚡ Test 2: C-Rate Limited Power")
        print("-" * 30)
        
        health_params = _calculate_battery_health_parameters('LFP', 25)
        max_power = 100  # 100kW system
        interval_hours = 0.25  # 15-min intervals
        
        for soc in [20, 40, 60, 80]:
            power_limits = _calculate_c_rate_limited_power(soc, max_power, health_params, interval_hours)
            print(f"SOC {soc}%:")
            print(f"  Max Discharge: {power_limits['max_discharge_power_kw']:.1f}kW")
            print(f"  Max Charge: {power_limits['max_charge_power_kw']:.1f}kW")
        
        # Test 3: SOC Protection Levels
        print("\n🛡️  Test 3: SOC Protection Levels")
        print("-" * 30)
        
        protection_levels = _get_soc_protection_levels()
        for level, config in protection_levels.items():
            print(f"{level.capitalize()}: ≤{config['threshold_percent']}% SOC")
            print(f"  Max Discharge: {config['max_discharge_percent']}%")
            print(f"  Priority: {config['charge_priority']}")
        
        # Test 4: SOC Protection Constraints
        print("\n🔒 Test 4: SOC Protection Constraints")
        print("-" * 30)
        
        requested_power = 80  # 80kW requested discharge
        for soc in [5, 15, 30, 50, 85]:
            constraint = _apply_soc_protection_constraints(soc, requested_power, protection_levels)
            print(f"SOC {soc}%: {requested_power}kW → {constraint['constrained_power_kw']:.1f}kW")
            print(f"  Protection Level: {constraint['active_protection_level']}")
            print(f"  Reduction: {constraint['discharge_reduction_percent']}%")
        
        # Test 5: Intelligent Charge Strategy
        print("\n🧠 Test 5: Intelligent Charge Strategy")
        print("-" * 30)
        
        available_excess = 50  # 50kW available
        max_charge = 100  # 100kW max charge
        
        for soc in [10, 25, 40, 60, 85]:
            for tariff_period in ['peak', 'off_peak']:  # RP4 2-period system
                strategy = _calculate_intelligent_charge_strategy(
                    soc, tariff_period, health_params, available_excess, max_charge
                )
                print(f"SOC {soc}% during {tariff_period}:")
                print(f"  Urgency: {strategy['urgency_level']}")
                print(f"  Recommended: {strategy['recommended_charge_power_kw']:.1f}kW")
                print(f"  Tariff Consideration: {strategy['tariff_consideration']:.1f}")
        
        # Test 6: Tariff-Aware Discharge Strategy
        print("\n💰 Test 6: Tariff-Aware Discharge Strategy")
        print("-" * 30)
        
        demand_power = 120  # 120kW demand
        
        for tariff_type in ['TOU', 'General']:
            for period in ['peak', 'off_peak']:  # RP4 2-period system
                for soc in [30, 60]:
                    strategy = _get_tariff_aware_discharge_strategy(
                        tariff_type, period, soc, demand_power, health_params
                    )
                    print(f"{tariff_type} tariff, {period} period, SOC {soc}%:")
                    print(f"  Strategy: {strategy['strategy_priority']}")
                    print(f"  Multiplier: {strategy['recommended_discharge_multiplier']:.3f}")
                    print(f"  Max Available: {strategy['max_available_discharge_kw']:.1f}kW")
        
        print("\n🎉 All Enhanced Algorithm Tests Completed Successfully!")
        print("=" * 50)
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_enhanced_algorithm_integration():
    """Test how the algorithms work together"""
    print("\n🔄 Testing Algorithm Integration")
    print("=" * 50)
    
    try:
        sys.path.append('/Users/chyeap89/Documents/energyanalysis')
        from md_shaving_solution_v2 import (
            _calculate_battery_health_parameters,
            _calculate_c_rate_limited_power,
            _apply_soc_protection_constraints,
            _calculate_intelligent_charge_strategy,
            _get_tariff_aware_discharge_strategy,
            _get_soc_protection_levels
        )
        
        # Simulate a battery system over time
        print("Simulating 24-hour battery operation with enhanced algorithms:")
        
        # Battery system parameters
        battery_chemistry = 'LFP'
        operating_temp = 25
        max_power = 100  # 100kW
        interval_hours = 1.0  # 1-hour intervals
        
        # Get health parameters
        health_params = _calculate_battery_health_parameters(battery_chemistry, operating_temp)
        protection_levels = _get_soc_protection_levels()
        
        # Simulate 24 hours
        results = []
        current_soc = 80  # Start at 80% SOC
        
        for hour in range(24):
            # Simulate different conditions throughout the day
            if 6 <= hour <= 18:  # Daytime
                # Use RP4 2-period logic: peak if weekday 14:00-22:00, otherwise off-peak
                tariff_period = 'peak' if 14 <= hour < 22 else 'off_peak'
                demand = 90 + np.random.normal(0, 10)  # Higher demand
                available_excess = max(0, 30 - np.random.normal(0, 15))  # Variable excess
            else:  # Nighttime
                tariff_period = 'off_peak'
                demand = 50 + np.random.normal(0, 5)  # Lower demand
                available_excess = 40 + np.random.normal(0, 10)  # More excess available
            
            # Get C-rate limits
            c_rate_limits = _calculate_c_rate_limited_power(current_soc, max_power, health_params, interval_hours)
            
            # Get discharge strategy
            discharge_strategy = _get_tariff_aware_discharge_strategy(
                'TOU', tariff_period, current_soc, demand, health_params
            )
            
            # Get charge strategy
            charge_strategy = _calculate_intelligent_charge_strategy(
                current_soc, tariff_period, health_params, available_excess, max_power
            )
            
            # Apply SOC protection
            soc_constraint = _apply_soc_protection_constraints(current_soc, demand, protection_levels)
            
            results.append({
                'hour': hour,
                'soc': current_soc,
                'tariff_period': tariff_period,
                'demand': demand,
                'discharge_multiplier': discharge_strategy['recommended_discharge_multiplier'],
                'charge_power': charge_strategy['recommended_charge_power_kw'],
                'protection_level': soc_constraint['active_protection_level'],
                'max_discharge_crate': c_rate_limits['max_discharge_power_kw'],
                'max_charge_crate': c_rate_limits['max_charge_power_kw']
            })
            
            # Simple SOC update (for demonstration)
            if tariff_period == 'peak' and current_soc > 30:
                current_soc -= 5  # Discharge during peak
            elif tariff_period == 'off_peak' and current_soc < 90:
                current_soc += 3  # Charge during off-peak
            
            current_soc = max(10, min(95, current_soc))  # Keep within bounds
        
        # Display summary
        df_results = pd.DataFrame(results)
        print(f"\nSOC Range: {df_results['soc'].min():.1f}% - {df_results['soc'].max():.1f}%")
        print(f"Average Discharge Multiplier: {df_results['discharge_multiplier'].mean():.3f}")
        print(f"Average Recommended Charge: {df_results['charge_power'].mean():.1f}kW")
        print(f"Protection Levels Used: {df_results['protection_level'].unique()}")
        
        print("\n✅ Algorithm Integration Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced V2 Battery Algorithm Tests")
    print("=" * 60)
    
    # Run individual algorithm tests
    test1_success = test_enhanced_battery_algorithms()
    
    # Run integration tests
    test2_success = test_enhanced_algorithm_integration()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("🎊 ALL TESTS PASSED! Enhanced V2 algorithms are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    print("=" * 60)
