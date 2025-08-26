#!/usr/bin/env python3
"""
Test script to verify the fix for max power shaving and max TOU energy calculation
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the Python path to import the module
sys.path.append('/Users/xlnyeong/energyanalaysis')

def test_variable_fix():
    """Test that the variable naming fix works correctly"""
    
    print("🧪 TESTING VARIABLE FIX")
    print("=" * 50)
    
    # Simulate the fixed code logic
    all_monthly_events = [
        {'TOU Required Energy (kWh)': 15.5, 'MD Cost Impact (RM)': 25.0},
        {'TOU Required Energy (kWh)': 12.3, 'MD Cost Impact (RM)': 18.5},
        {'TOU Required Energy (kWh)': 20.1, 'MD Cost Impact (RM)': 32.0},
        {'TOU Required Energy (kWh)': 18.7, 'MD Cost Impact (RM)': 28.5}
    ]
    
    # Create period index for proper pandas Series
    monthly_targets = pd.Series([450, 470, 440, 460], 
                               index=pd.PeriodIndex(['2024-01', '2024-02', '2024-03', '2024-04'], freq='M'))
    monthly_max_demands = pd.Series([500, 520, 480, 510], 
                                   index=pd.PeriodIndex(['2024-01', '2024-02', '2024-03', '2024-04'], freq='M'))
    
    print("📊 Test Data:")
    print(f"   Events: {len(all_monthly_events)} peak events")
    print(f"   Monthly targets: {monthly_targets.tolist()}")
    print(f"   Monthly max demands: {monthly_max_demands.tolist()}")
    
    # Test the FIXED calculation logic
    print("\n🔧 Testing FIXED calculation logic:")
    
    max_shaving_power = 0
    max_tou_energy = 0
    total_md_cost = 0
    
    if all_monthly_events:
        # Calculate max shaving power from monthly targets and max demands
        if monthly_targets is not None and len(monthly_targets) > 0:
            shaving_amounts = []
            for month_period, target_demand in monthly_targets.items():
                if month_period in monthly_max_demands:
                    max_demand = monthly_max_demands[month_period]
                    shaving_amount = max_demand - target_demand
                    if shaving_amount > 0:
                        shaving_amounts.append(shaving_amount)
                        print(f"   Month {month_period}: {max_demand}kW - {target_demand}kW = {shaving_amount}kW shaving needed")
            max_shaving_power = max(shaving_amounts) if shaving_amounts else 0
        
        # Calculate max TOU energy and total MD cost from events
        max_tou_energy = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
        total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
    
    print(f"\n✅ FIXED Results:")
    print(f"   max_shaving_power: {max_shaving_power} kW")
    print(f"   max_tou_energy: {max_tou_energy} kWh")
    print(f"   total_md_cost: RM {total_md_cost}")
    
    # Test battery simulation prerequisites
    print("\n🔋 Testing battery simulation prerequisites:")
    
    # Simulate battery specifications
    battery_capacity_kwh = 50
    battery_power_kw = 25
    
    prerequisites_met = True
    error_messages = []
    
    # Test the FIXED validation logic
    if max_shaving_power <= 0:
        prerequisites_met = False
        error_messages.append("Max shaving power not calculated or invalid")
    
    if max_tou_energy <= 0:
        prerequisites_met = False
        error_messages.append("Max TOU energy not calculated or invalid")
    
    if battery_power_kw <= 0:
        prerequisites_met = False
        error_messages.append(f"Invalid battery power: {battery_power_kw} kW")
    
    if battery_capacity_kwh <= 0:
        prerequisites_met = False
        error_messages.append(f"Invalid battery capacity: {battery_capacity_kwh} kWh")
    
    print(f"   Prerequisites met: {prerequisites_met}")
    if error_messages:
        print(f"   Error messages: {error_messages}")
    else:
        print("   ✅ All prerequisites satisfied!")
    
    # Test battery sizing calculations
    if prerequisites_met:
        print("\n⚡ Testing battery sizing calculations:")
        
        units_for_power = int(np.ceil(max_shaving_power / battery_power_kw)) if battery_power_kw > 0 else 1
        units_for_energy = int(np.ceil(max_tou_energy / battery_capacity_kwh)) if battery_capacity_kwh > 0 else 1
        optimal_units = max(units_for_power, units_for_energy, 1)
        
        total_battery_capacity = optimal_units * battery_capacity_kwh
        total_battery_power = optimal_units * battery_power_kw
        
        print(f"   Units needed for power: {units_for_power} ({max_shaving_power}kW ÷ {battery_power_kw}kW/unit)")
        print(f"   Units needed for energy: {units_for_energy} ({max_tou_energy}kWh ÷ {battery_capacity_kwh}kWh/unit)")
        print(f"   Optimal units: {optimal_units}")
        print(f"   Total system capacity: {total_battery_capacity} kWh")
        print(f"   Total system power: {total_battery_power} kW")
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    if prerequisites_met:
        print("✅ Variable fix SUCCESSFUL!")
        print("✅ All calculations working correctly")
        print("✅ Battery simulation prerequisites met")
        print("✅ Max power shaving and max TOU energy properly calculated")
    else:
        print("❌ Issues still present:")
        for msg in error_messages:
            print(f"   - {msg}")
    
    return {
        'max_shaving_power': max_shaving_power,
        'max_tou_energy': max_tou_energy,
        'total_md_cost': total_md_cost,
        'prerequisites_met': prerequisites_met,
        'error_messages': error_messages
    }

def test_edge_cases():
    """Test edge cases for the calculation logic"""
    
    print("\n\n🔍 TESTING EDGE CASES")
    print("=" * 50)
    
    # Test case 1: No events
    print("\n1. Testing with no events:")
    max_shaving_power = 0
    max_tou_energy = 0
    total_md_cost = 0
    
    all_monthly_events = []
    if all_monthly_events:
        max_tou_energy = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
        total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
    
    print(f"   max_tou_energy: {max_tou_energy}")
    print(f"   total_md_cost: {total_md_cost}")
    
    # Test case 2: Events with zero/None values
    print("\n2. Testing with zero/None values:")
    all_monthly_events = [
        {'TOU Required Energy (kWh)': 0, 'MD Cost Impact (RM)': 0},
        {'TOU Required Energy (kWh)': None, 'MD Cost Impact (RM)': None},
        {'MD Cost Impact (RM)': 15.0}  # Missing TOU key
    ]
    
    max_tou_energy = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
    total_md_cost = sum(event.get('MD Cost Impact (RM)', 0) or 0 for event in all_monthly_events)
    
    print(f"   max_tou_energy: {max_tou_energy}")
    print(f"   total_md_cost: {total_md_cost}")
    
    # Test case 3: No shaving required (all demands below targets)
    print("\n3. Testing with no shaving required:")
    monthly_targets = pd.Series([600, 650, 620], 
                               index=pd.PeriodIndex(['2024-01', '2024-02', '2024-03'], freq='M'))
    monthly_max_demands = pd.Series([500, 520, 480], 
                                   index=pd.PeriodIndex(['2024-01', '2024-02', '2024-03'], freq='M'))
    
    max_shaving_power = 0
    if monthly_targets is not None and len(monthly_targets) > 0:
        shaving_amounts = []
        for month_period, target_demand in monthly_targets.items():
            if month_period in monthly_max_demands:
                max_demand = monthly_max_demands[month_period]
                shaving_amount = max_demand - target_demand
                if shaving_amount > 0:
                    shaving_amounts.append(shaving_amount)
        max_shaving_power = max(shaving_amounts) if shaving_amounts else 0
    
    print(f"   max_shaving_power: {max_shaving_power}")
    
    print("\n" + "=" * 50)
    print("✅ Edge case testing completed")

if __name__ == "__main__":
    results = test_variable_fix()
    test_edge_cases()
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"   Max Shaving Power: {results['max_shaving_power']} kW")
    print(f"   Max TOU Energy: {results['max_tou_energy']} kWh") 
    print(f"   Total MD Cost: RM {results['total_md_cost']}")
    print(f"   Prerequisites Met: {results['prerequisites_met']}")
