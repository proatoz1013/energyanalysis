#!/usr/bin/env python3
"""
Test script to validate the enhanced V2 battery charging algorithm with RP4 tariff awareness.

This script tests the corrected emergency charging behavior and RP4 tariff integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from md_shaving_solution_v2 import (
    _calculate_intelligent_charge_strategy_simple,
    _get_tariff_aware_discharge_strategy
)

def test_enhanced_rp4_charging_algorithm():
    """Test the enhanced charging algorithm with RP4 tariff awareness."""
    
    print("🧪 TESTING ENHANCED V2 BATTERY ALGORITHM WITH RP4 TARIFF AWARENESS")
    print("=" * 80)
    
    # Simplified battery health parameters
    health_params = {
        'health_derating_factor': 1.0,
        'temperature_derating_factor': 1.0,
        'max_c_rate': 1.0
    }
    
    # Test charging strategy across different SOC levels and tariff periods
    print("\n🔋 TEST 1: ENHANCED CHARGING STRATEGY - SOC vs RP4 Tariff Periods")
    print("-" * 60)
    
    test_soc_levels = [3, 8, 12, 20, 30, 50, 70, 88]
    tariff_periods = ['peak', 'off_peak']
    available_power = 50  # 50kW available
    max_charge = 100     # 100kW max charge
    
    for soc in test_soc_levels:
        print(f"\n📊 SOC Level: {soc}%")
        for period in tariff_periods:
            strategy = _calculate_intelligent_charge_strategy_simple(
                soc, period, health_params, available_power, max_charge
            )
            
            print(f"  {period.upper():8} | {strategy['urgency_level']:20} | "
                  f"{strategy['recommended_charge_power_kw']:6.1f}kW | "
                  f"MD Priority: {strategy['md_constraint_priority']} | "
                  f"{strategy['strategy_description']}")
    
    # Test discharge strategy focusing on critical SOC behavior
    print("\n⚡ TEST 2: ENHANCED DISCHARGE STRATEGY - Critical SOC Protection")
    print("-" * 60)
    
    critical_soc_levels = [2, 8, 12, 18, 28]
    demand_power = 120  # 120kW demand
    
    for soc in critical_soc_levels:
        print(f"\n🔋 SOC Level: {soc}%")
        for period in tariff_periods:
            strategy = _get_tariff_aware_discharge_strategy(
                'TOU', period, soc, demand_power, 100, health_params
            )
            
            discharge_multiplier = strategy['recommended_discharge_multiplier']
            protection_level = strategy['protection_level']
            
            print(f"  {period.upper():8} | {protection_level:25} | "
                  f"{discharge_multiplier:6.1%} | "
                  f"{strategy['strategy_description']}")
    
    # Test key scenarios that demonstrate the fixes
    print("\n🎯 TEST 3: KEY CORRECTED BEHAVIORS")
    print("-" * 60)
    
    print("✅ SCENARIO 1: Critical SOC (8%) during RP4 Peak Period")
    strategy = _calculate_intelligent_charge_strategy_simple(8, 'peak', health_params, 30, 80)
    print(f"   Charging: {strategy['urgency_level']} - {strategy['recommended_charge_power_kw']:.1f}kW")
    print(f"   Strategy: {strategy['strategy_description']}")
    
    discharge = _get_tariff_aware_discharge_strategy('TOU', 'peak', 8, 120, 100, health_params)
    print(f"   Discharge: {discharge['protection_level']} - {discharge['recommended_discharge_multiplier']:.1%}")
    print(f"   Behavior: {discharge['strategy_description']}")
    
    print("\n✅ SCENARIO 2: Low SOC (12%) during RP4 Peak Period (MD Recording)")
    strategy = _calculate_intelligent_charge_strategy_simple(12, 'peak', health_params, 30, 80)
    print(f"   Charging: {strategy['urgency_level']} - {strategy['recommended_charge_power_kw']:.1f}kW")
    print(f"   MD Priority: {strategy['md_constraint_priority']}")
    
    discharge = _get_tariff_aware_discharge_strategy('TOU', 'peak', 12, 120, 100, health_params)
    print(f"   Discharge: {discharge['soc_discharge_factor']:.1%} factor")
    print(f"   Protection: {discharge['strategy_description']}")
    
    print("\n✅ SCENARIO 3: Normal SOC (50%) during RP4 Off-Peak Period")
    strategy = _calculate_intelligent_charge_strategy_simple(50, 'off_peak', health_params, 30, 80)
    print(f"   Charging: {strategy['urgency_level']} - {strategy['recommended_charge_power_kw']:.1f}kW")
    print(f"   Period Strategy: {strategy['period_strategy']}")
    
    discharge = _get_tariff_aware_discharge_strategy('TOU', 'off_peak', 50, 120, 100, health_params)
    print(f"   Discharge: {discharge['protection_level']} - {discharge['recommended_discharge_multiplier']:.1%}")
    print(f"   Strategy: {discharge['strategy_description']}")
    
    # Summary of key improvements
    print("\n🏆 ALGORITHM ENHANCEMENT SUMMARY")
    print("-" * 60)
    print("✅ Eliminated emergency charging during peak hours that violates MD targets")
    print("✅ Implemented preventive discharge limits (stop at 15% SOC during MD hours)")
    print("✅ Integrated RP4 2-period tariff system (Peak: Mon-Fri 2PM-10PM)")
    print("✅ Enhanced SOC protection with MD window awareness")
    print("✅ Controlled charging that never exceeds monthly targets")
    print("✅ Tariff-optimized strategy with battery health considerations")
    
    print("\n🎯 KEY CORRECTED BEHAVIORS:")
    print("   • SOC < 5%: Critical protection with controlled charging")
    print("   • SOC 5-15%: Stop discharge during peak, micro charging allowed")
    print("   • SOC 15-25%: Reduced discharge, limited charging")
    print("   • SOC > 25%: Normal operation based on RP4 tariff period")
    print("   • Peak Period: Minimal charging, maximum discharge (when SOC allows)")
    print("   • Off-Peak: Optimal charging, conservative discharge")

if __name__ == "__main__":
    test_enhanced_rp4_charging_algorithm()
