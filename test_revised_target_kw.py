#!/usr/bin/env python3
"""
Test script to verify that the new Revised Target kW column calculation works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required functions
from tariffs.peak_logic import get_malaysia_holidays
from md_shaving_solution_v2 import _calculate_revised_target_kw, _calculate_target_shave_kw_holiday_aware

def test_revised_target_kw():
    """Test that Revised Target kW calculation works correctly under different scenarios."""
    
    print("🧪 Testing Revised Target kW Calculation")
    print("=" * 60)
    
    # Get 2025 holidays for testing
    holidays_2025 = get_malaysia_holidays(2025)
    print(f"✅ Loaded {len(holidays_2025)} holidays for 2025")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Normal Operation - High SOC',
            'timestamp': datetime(2025, 5, 13, 16, 0),  # Regular weekday
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 80.0,
                'Conserve_Activated': False,
                'Battery Conserved kW': 0.0
            },
            'expected_base': 30.0,  # 150 - 120
            'expected_revised': 30.0  # No reduction at high SOC
        },
        {
            'name': 'Low SOC Scenario',
            'timestamp': datetime(2025, 5, 13, 16, 0),
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 25.0,  # Low SOC
                'Conserve_Activated': False,
                'Battery Conserved kW': 0.0
            },
            'expected_base': 30.0,
            'expected_revised': 24.0  # 80% of base target
        },
        {
            'name': 'Medium SOC Scenario',
            'timestamp': datetime(2025, 5, 13, 16, 0),
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 40.0,  # Medium SOC
                'Conserve_Activated': False,
                'Battery Conserved kW': 0.0
            },
            'expected_base': 30.0,
            'expected_revised': 27.0  # 90% of base target
        },
        {
            'name': 'Conservation Mode Active',
            'timestamp': datetime(2025, 5, 13, 16, 0),
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 45.0,
                'Conserve_Activated': True,
                'Battery Conserved kW': 10.0  # 10 kW conserved
            },
            'expected_base': 30.0,
            'expected_revised': 20.0  # 30 - 10 conserved
        },
        {
            'name': 'Holiday - Should be Zero',
            'timestamp': datetime(2025, 5, 12, 16, 0),  # Wesak Day
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 80.0,
                'Conserve_Activated': False,
                'Battery Conserved kW': 0.0
            },
            'expected_base': 0.0,  # Holiday
            'expected_revised': 0.0  # Holiday
        },
        {
            'name': 'Off-Peak Period - Should be Zero',
            'timestamp': datetime(2025, 5, 13, 12, 0),  # Noon - outside MD window
            'data': {
                'Original_Demand': 150.0,
                'Monthly_Target': 120.0,
                'Battery_SOC_Percent': 80.0,
                'Conserve_Activated': False,
                'Battery Conserved kW': 0.0
            },
            'expected_base': 0.0,  # Off-peak
            'expected_revised': 0.0  # Off-peak
        }
    ]
    
    # Run tests
    all_passed = True
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['name']}")
        print(f"   🕐 Time: {scenario['timestamp'].strftime('%Y-%m-%d %H:%M')} ({scenario['timestamp'].strftime('%A')})")
        
        # Create test row
        test_row = pd.Series(scenario['data'], name=scenario['timestamp'])
        
        # Calculate base target
        base_target = _calculate_target_shave_kw_holiday_aware(test_row, holidays_2025)
        
        # Calculate revised target
        revised_target = _calculate_revised_target_kw(test_row, holidays_2025)
        
        print(f"   📊 Original Demand: {scenario['data']['Original_Demand']} kW")
        print(f"   🎯 Monthly Target: {scenario['data']['Monthly_Target']} kW")
        print(f"   🔋 Battery SOC: {scenario['data']['Battery_SOC_Percent']}%")
        print(f"   🛡️ Conservation: {'Active' if scenario['data']['Conserve_Activated'] else 'Inactive'}")
        if scenario['data']['Conserve_Activated']:
            print(f"   🔒 Conserved kW: {scenario['data']['Battery Conserved kW']} kW")
        
        print(f"   ✂️ Base Target Shave: {base_target} kW (Expected: {scenario['expected_base']} kW)")
        print(f"   🔧 Revised Target: {revised_target} kW (Expected: {scenario['expected_revised']} kW)")
        
        # Check results
        base_passed = abs(base_target - scenario['expected_base']) < 0.1
        revised_passed = abs(revised_target - scenario['expected_revised']) < 0.1
        
        if base_passed and revised_passed:
            print(f"   ✅ PASS - Both calculations correct")
        else:
            print(f"   ❌ FAIL - Issues detected:")
            if not base_passed:
                print(f"      Base target: got {base_target}, expected {scenario['expected_base']}")
            if not revised_passed:
                print(f"      Revised target: got {revised_target}, expected {scenario['expected_revised']}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests PASSED! Revised Target kW calculation is working correctly.")
    else:
        print("❌ Some tests FAILED. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    test_revised_target_kw()
