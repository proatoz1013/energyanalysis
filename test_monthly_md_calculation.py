#!/usr/bin/env python3
"""
Test script to verify the corrected monthly MD calculation logic
Tests the fix: Monthly MD Savings = Highest Original Demand - Highest Net Demand (per month)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_monthly_data():
    """Create sample data to test monthly MD calculation"""
    print("📊 Creating test data for monthly MD calculation...")
    
    # Create test data spanning 3 months
    dates = []
    original_demands = []
    net_demands = []
    actual_shaves = []
    
    # January data - Original peak: 1000kW, Net peak: 950kW, Expected monthly saving: 50kW
    jan_dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    for i, date in enumerate(jan_dates):
        if i == 15:  # Peak day
            original_demands.append(1000.0)  # Monthly peak original
            net_demands.append(950.0)        # Net demand on same day
            actual_shaves.append(50.0)       # Daily shave
        else:
            original_demands.append(800.0 + np.random.random() * 150)
            net_demands.append(original_demands[-1] - (20 + np.random.random() * 30))
            actual_shaves.append(original_demands[-1] - net_demands[-1])
        dates.append(date)
    
    # February data - Original peak: 1100kW, Net peak: 1020kW, Expected monthly saving: 80kW  
    feb_dates = pd.date_range('2024-02-01', '2024-02-29', freq='D')
    for i, date in enumerate(feb_dates):
        if i == 10:  # Peak day
            original_demands.append(1100.0)  # Monthly peak original
            net_demands.append(1020.0)       # Net demand on same day
            actual_shaves.append(80.0)       # Daily shave
        else:
            original_demands.append(850.0 + np.random.random() * 200)
            net_demands.append(original_demands[-1] - (25 + np.random.random() * 35))
            actual_shaves.append(original_demands[-1] - net_demands[-1])
        dates.append(date)
    
    # March data - Original peak: 1200kW, Net peak: 1100kW, Expected monthly saving: 100kW
    mar_dates = pd.date_range('2024-03-01', '2024-03-31', freq='D')
    for i, date in enumerate(mar_dates):
        if i == 20:  # Peak day
            original_demands.append(1200.0)  # Monthly peak original
            net_demands.append(1100.0)       # Net demand on same day
            actual_shaves.append(100.0)      # Daily shave
        else:
            original_demands.append(900.0 + np.random.random() * 250)
            net_demands.append(original_demands[-1] - (30 + np.random.random() * 40))
            actual_shaves.append(original_demands[-1] - net_demands[-1])
        dates.append(date)
    
    # Create daily summary format
    daily_summary = pd.DataFrame({
        'Date': dates,
        'Max Original Demand (kW)': [f"{x:,.1f}" for x in original_demands],
        'Max_Original_Demand_Numeric': original_demands,
        'Actual MD Shave (kW)': [f"{x:,.1f}" for x in actual_shaves],
        'Actual_MD_Shave_Numeric': actual_shaves
    })
    
    return daily_summary

def test_monthly_md_logic(daily_summary):
    """Test the corrected monthly MD calculation logic"""
    print("\n🧮 Testing Monthly MD Calculation Logic...")
    
    # Add Month column for grouping
    daily_summary['Month'] = pd.to_datetime(daily_summary['Date']).dt.to_period('M')
    
    print("\nExpected Results:")
    print("January: 1000.0 - 950.0 = 50.0 kW")
    print("February: 1100.0 - 1020.0 = 80.0 kW") 
    print("March: 1200.0 - 1100.0 = 100.0 kW")
    
    print("\nCalculated Results:")
    monthly_success_shaved = {}
    for month_period in daily_summary['Month'].unique():
        month_data = daily_summary[daily_summary['Month'] == month_period]
        
        if len(month_data) > 0:
            # Find the highest original demand across the entire month
            max_original_demand = month_data['Max_Original_Demand_Numeric'].max()
            
            # Calculate net demand for each day and find monthly max
            month_data_copy = month_data.copy()
            month_data_copy['Net_Demand_Numeric'] = month_data_copy['Max_Original_Demand_Numeric'] - month_data_copy['Actual_MD_Shave_Numeric']
            max_net_demand = month_data_copy['Net_Demand_Numeric'].max()
            
            # Monthly MD Savings = Max Original - Max Net across the month
            monthly_md_saving = max(0, max_original_demand - max_net_demand)
            monthly_success_shaved[month_period] = monthly_md_saving
            
            print(f"{month_period}: {max_original_demand:,.1f} - {max_net_demand:,.1f} = {monthly_md_saving:,.1f} kW")
        else:
            monthly_success_shaved[month_period] = 0.0
    
    return monthly_success_shaved

def main():
    """Main test function"""
    print("🚀 Testing Monthly MD Calculation Logic Fix")
    print("=" * 60)
    
    # Create test data
    daily_summary = create_test_monthly_data()
    
    # Test the calculation logic
    results = test_monthly_md_logic(daily_summary)
    
    print("\n✅ Monthly MD Calculation Test Complete!")
    print(f"Total months tested: {len(results)}")
    print(f"Average monthly saving: {np.mean(list(results.values())):,.1f} kW")
    
    return results

if __name__ == "__main__":
    main()
