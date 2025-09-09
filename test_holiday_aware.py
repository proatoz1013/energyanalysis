#!/usr/bin/env python3
"""
Test script to verify that the holiday-aware Target_Shave_kW calculation
correctly returns 0 on Malaysian public holidays like May 12, 2025 (Wesak Day).
"""

import pandas as pd
from datetime import datetime, date
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required functions
from tariffs.peak_logic import get_malaysia_holidays, is_public_holiday
from md_shaving_solution_v2 import _calculate_target_shave_kw_holiday_aware

def test_holiday_awareness():
    """Test that Target_Shave_kW returns 0 on holidays during what would normally be peak hours."""
    
    print("🧪 Testing Holiday-Aware Target_Shave_kW Calculation")
    print("=" * 60)
    
    # Get 2025 holidays
    holidays_2025 = get_malaysia_holidays(2025)
    print(f"✅ Loaded {len(holidays_2025)} holidays for 2025")
    
    # Confirm May 12, 2025 is in holidays
    wesak_day = date(2025, 5, 12)
    print(f"📅 May 12, 2025 (Wesak Day) in holidays: {wesak_day in holidays_2025}")
    
    # Create test data for May 12, 2025 during peak hours (2PM-10PM)
    test_times = [
        datetime(2025, 5, 12, 14, 0),  # 2:00 PM - start of MD period
        datetime(2025, 5, 12, 16, 30), # 4:30 PM - middle of MD period
        datetime(2025, 5, 12, 21, 45), # 9:45 PM - near end of MD period
    ]
    
    # Create test scenarios with high demand that would normally trigger shaving
    for test_time in test_times:
        # Create a mock row with high demand during what would be MD peak hours
        test_row = pd.Series({
            'Original_Demand': 150.0,  # High demand that exceeds typical targets
            'Monthly_Target': 120.0,   # Lower target that would normally trigger shaving
        }, name=test_time)
        
        # Test the holiday-aware function
        target_shave = _calculate_target_shave_kw_holiday_aware(test_row, holidays_2025)
        
        print(f"🕐 {test_time.strftime('%Y-%m-%d %H:%M')} (Wesak Day, {test_time.strftime('%A')}):")
        print(f"   📈 Original Demand: {test_row['Original_Demand']} kW")
        print(f"   🎯 Monthly Target: {test_row['Monthly_Target']} kW")
        print(f"   ✂️ Target Shave: {target_shave} kW")
        print(f"   ✅ Expected: 0.0 kW (holiday - no MD charges)")
        
        if target_shave == 0.0:
            print(f"   🎉 PASS - Correctly returns 0 on holiday")
        else:
            print(f"   ❌ FAIL - Should return 0 on holiday, got {target_shave}")
        print()
    
    # Test non-holiday for comparison
    print("🔍 Comparison Test: Non-Holiday Weekday")
    non_holiday = datetime(2025, 5, 13, 16, 0)  # Tuesday after Wesak Day
    
    test_row_normal = pd.Series({
        'Original_Demand': 150.0,
        'Monthly_Target': 120.0,
    }, name=non_holiday)
    
    target_shave_normal = _calculate_target_shave_kw_holiday_aware(test_row_normal, holidays_2025)
    
    print(f"🕐 {non_holiday.strftime('%Y-%m-%d %H:%M')} (Regular {non_holiday.strftime('%A')}):")
    print(f"   📈 Original Demand: {test_row_normal['Original_Demand']} kW")
    print(f"   🎯 Monthly Target: {test_row_normal['Monthly_Target']} kW")
    print(f"   ✂️ Target Shave: {target_shave_normal} kW")
    print(f"   ✅ Expected: 30.0 kW (150 - 120 = 30 kW shaving needed)")
    
    if target_shave_normal == 30.0:
        print(f"   🎉 PASS - Correctly calculates shaving on regular weekday")
    else:
        print(f"   ❌ May need adjustment - Expected 30.0, got {target_shave_normal}")

if __name__ == "__main__":
    test_holiday_awareness()
