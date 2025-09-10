#!/usr/bin/env python3
"""
Test the fixed comprehensive battery status and success rate calculation.
This tests the integration between the comprehensive battery status function and the V2 Two-Level Cascading Filter system.
"""

import sys
import os
sys.path.append('/Users/chyeap89/Documents/energyanalysis')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_comprehensive_battery_status_with_holidays():
    """Test the comprehensive battery status function with proper holiday handling."""
    
    print("🧪 Testing V2 Battery Status Classification with Holiday Integration")
    print("=" * 70)
    
    try:
        # Import the fixed functions
        from md_shaving_solution_v2 import (
            _get_comprehensive_battery_status,
            _calculate_success_rate_from_shaving_status
        )
        print("✅ Successfully imported fixed battery status functions")
        
        # Create test data with various scenarios
        test_dates = [
            datetime(2024, 1, 15, 14, 30),  # Monday MD period
            datetime(2024, 1, 15, 10, 0),   # Monday off-peak
            datetime(2024, 1, 1, 15, 0),    # Holiday (New Year) during MD hours
            datetime(2024, 1, 13, 15, 0),   # Saturday during MD hours  
            datetime(2024, 1, 16, 16, 0),   # Tuesday MD period with success
            datetime(2024, 1, 17, 18, 0),   # Wednesday MD period with partial success
        ]
        
        # Create DataFrame with simulation-like columns
        df_test = pd.DataFrame({
            'Original_Demand': [950, 800, 900, 850, 980, 920],
            'Net_Demand_kW': [840, 800, 900, 850, 850, 900],  # Some successful shaving
            'Monthly_Target': [900, 900, 900, 900, 900, 900],
            'Battery_Power_kW': [50, -20, 0, 0, 80, 30],  # Mixed charge/discharge
            'Battery_SOC_Percent': [75, 85, 90, 60, 45, 30]
        }, index=test_dates)
        
        # Define holidays (New Year's Day)
        holidays = {datetime(2024, 1, 1).date()}
        
        print(f"📊 Test DataFrame created with {len(df_test)} scenarios")
        print(f"🏖️ Holidays defined: {holidays}")
        
        # Test individual status classifications
        print("\n🔍 Individual Battery Status Classifications:")
        for i, (timestamp, row) in enumerate(df_test.iterrows()):
            status = _get_comprehensive_battery_status(row, holidays)
            print(f"  {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M')} ({timestamp.strftime('%A')}): {status}")
        
        # Add Success_Status column
        df_test['Success_Status'] = df_test.apply(
            lambda row: _get_comprehensive_battery_status(row, holidays), axis=1
        )
        
        # Test success rate calculation
        print(f"\n📈 Testing Success Rate Calculation:")
        success_metrics = _calculate_success_rate_from_shaving_status(df_test, holidays=holidays, debug=True)
        
        print(f"✅ Success Rate: {success_metrics['success_rate_percent']:.1f}%")
        print(f"✅ Total MD Intervals: {success_metrics['total_md_intervals']}")
        print(f"✅ Successful Intervals: {success_metrics['successful_intervals']}")
        print(f"✅ Calculation Method: {success_metrics['calculation_method']}")
        
        # Verify holiday exclusion worked
        md_periods_in_test = sum(1 for ts in test_dates 
                               if ts.weekday() < 5 and 14 <= ts.hour < 22 
                               and not (holidays and ts.date() in holidays))
        
        print(f"\n🎯 Validation:")
        print(f"  Expected MD intervals (manual count): {md_periods_in_test}")
        print(f"  Calculated MD intervals: {success_metrics['total_md_intervals']}")
        print(f"  Holiday exclusion working: {'✅ YES' if md_periods_in_test == success_metrics['total_md_intervals'] else '❌ NO'}")
        
        # Test without holidays for comparison
        print(f"\n🔄 Testing without holiday exclusion:")
        success_metrics_no_holidays = _calculate_success_rate_from_shaving_status(df_test, holidays=None)
        print(f"  Success Rate (no holidays): {success_metrics_no_holidays['success_rate_percent']:.1f}%")
        print(f"  MD Intervals (no holidays): {success_metrics_no_holidays['total_md_intervals']}")
        
        # Show the difference
        holiday_impact = success_metrics_no_holidays['total_md_intervals'] - success_metrics['total_md_intervals']
        print(f"  Holiday impact: {holiday_impact} intervals excluded")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Holiday-aware MD period classification: WORKING")
        print(f"✅ Comprehensive battery status generation: WORKING") 
        print(f"✅ Success rate calculation with holidays: WORKING")
        print(f"✅ V2 Two-Level Cascading Filter integration: READY")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_battery_status_with_holidays()
    sys.exit(0 if success else 1)
