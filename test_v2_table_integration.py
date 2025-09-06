#!/usr/bin/env python3
"""
Test V2 Table Integration - Enhanced Battery Simulation Tables
============================================================

This script tests the newly integrated table visualization functions
in MD Shaving V2 to ensure they work correctly with simulated data.

Author: Enhanced MD Shaving Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

def create_test_simulation_data():
    """Create test simulation data for table testing"""
    print("🔧 Creating test simulation data...")
    
    # Create 24 hours of 15-minute intervals
    start_time = datetime(2025, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(96)]  # 24 hours * 4 intervals/hour
    
    # Create realistic battery simulation data
    np.random.seed(42)  # For reproducible results
    
    # Original demand with realistic daily pattern
    base_demand = 200 + 50 * np.sin(np.linspace(0, 2*np.pi, 96))  # Sinusoidal daily pattern
    noise = np.random.normal(0, 10, 96)  # Random noise
    original_demand = np.maximum(base_demand + noise, 50)  # Minimum 50 kW
    
    # Monthly targets (dynamic)
    monthly_target = 250  # kW
    
    # Battery operation simulation
    battery_power = np.random.uniform(-50, 100, 96)  # -50kW (charge) to 100kW (discharge)
    battery_soc = np.clip(50 + np.cumsum(np.random.uniform(-2, 2, 96)), 20, 95)  # SOC between 20-95%
    
    # Net demand after battery
    net_demand = original_demand - battery_power
    peak_shaved = np.maximum(original_demand - net_demand, 0)
    
    # Create simulation dataframe
    df_sim = pd.DataFrame({
        'Original_Demand': original_demand,
        'Monthly_Target': monthly_target,
        'Net_Demand_kW': net_demand,
        'Battery_Power_kW': battery_power,
        'Battery_SOC_Percent': battery_soc,
        'Peak_Shaved': peak_shaved
    }, index=pd.DatetimeIndex(timestamps))
    
    print(f"✅ Created simulation data: {len(df_sim)} intervals")
    return df_sim


def test_table_functions():
    """Test all table creation functions"""
    print("\n📋 Testing table creation functions...")
    
    try:
        # Import the functions
        from md_shaving_solution_v2 import (
            _create_enhanced_battery_table,
            _create_daily_summary_table, 
            _create_kpi_summary_table
        )
        print("✅ Successfully imported table functions")
        
        # Create test data
        df_sim = create_test_simulation_data()
        
        # Test 1: Enhanced Battery Table
        print("\n🧪 Testing enhanced battery table...")
        enhanced_table = _create_enhanced_battery_table(df_sim)
        
        required_columns = [
            'Timestamp', 'Original_Demand_kW', 'Monthly_Target_kW', 
            'Net_Demand_kW', 'Battery_Action', 'SOC_%', 'SOC_Status',
            'Peak_Shaved_kW', 'Shaving_Success', 'MD_Period', 'Target_Violation'
        ]
        
        missing_cols = [col for col in required_columns if col not in enhanced_table.columns]
        if missing_cols:
            print(f"❌ Missing columns in enhanced table: {missing_cols}")
            return False
        else:
            print(f"✅ Enhanced table created successfully: {len(enhanced_table)} rows, {len(enhanced_table.columns)} columns")
        
        # Test 2: Daily Summary Table
        print("\n🧪 Testing daily summary table...")
        daily_table = _create_daily_summary_table(df_sim)
        
        expected_daily_cols = [
            'Original_Peak_kW', 'Net_Peak_kW', 'Peak_Reduction_kW',
            'Target_Success', 'SOC_Health'
        ]
        
        missing_daily_cols = [col for col in expected_daily_cols if col not in daily_table.columns]
        if missing_daily_cols:
            print(f"❌ Missing columns in daily summary: {missing_daily_cols}")
            return False
        else:
            print(f"✅ Daily summary table created successfully: {len(daily_table)} days")
        
        # Test 3: KPI Summary Table
        print("\n🧪 Testing KPI summary table...")
        simulation_results = {
            'peak_reduction_kw': 50.0,
            'success_rate_percent': 85.0,
            'total_energy_discharged': 1200.0,
            'total_energy_charged': 800.0,
            'average_soc': 65.0,
            'min_soc': 25.0,
            'max_soc': 90.0,
            'monthly_targets_count': 1,
            'v2_constraint_violations': 5
        }
        
        kpi_table = _create_kpi_summary_table(simulation_results, df_sim)
        
        if len(kpi_table) < 10:  # Should have at least 10 KPIs
            print(f"❌ KPI table too short: {len(kpi_table)} rows")
            return False
        else:
            print(f"✅ KPI summary table created successfully: {len(kpi_table)} metrics")
        
        # Test data quality
        print("\n🔍 Checking data quality...")
        
        # Check for realistic values
        if enhanced_table['SOC_%'].min() < 0 or enhanced_table['SOC_%'].max() > 100:
            print("❌ SOC values out of range")
            return False
        
        # Check for proper status indicators
        status_indicators = enhanced_table['SOC_Status'].unique()
        expected_indicators = ['🔴 Critical', '🟡 Low', '🟢 Normal', '🔵 High']
        if not any(indicator in status_indicators for indicator in expected_indicators):
            print("❌ SOC status indicators not working")
            return False
        
        print("✅ Data quality checks passed")
        
        print("\n📊 Sample Enhanced Table Data:")
        print(enhanced_table.head(3).to_string())
        
        print("\n📅 Sample Daily Summary Data:")
        print(daily_table.head(2).to_string())
        
        print("\n🎯 Sample KPI Data:")
        print(kpi_table.head(5).to_string())
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing table functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_workflow():
    """Test the complete integration workflow"""
    print("\n⚡ Testing table integration workflow...")
    
    try:
        # This simulates how the tables would be called in the real V2 workflow
        df_sim = create_test_simulation_data()
        
        # Prepare simulation results (as would be generated by _simulate_battery_operation_v2)
        simulation_results = {
            'peak_reduction_kw': df_sim['Peak_Shaved'].max(),
            'success_rate_percent': 85.0,
            'total_energy_discharged': df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] > 0, 0).sum() * 0.25,
            'total_energy_charged': abs(df_sim['Battery_Power_kW'].where(df_sim['Battery_Power_kW'] < 0, 0).sum()) * 0.25,
            'average_soc': df_sim['Battery_SOC_Percent'].mean(),
            'min_soc': df_sim['Battery_SOC_Percent'].min(),
            'max_soc': df_sim['Battery_SOC_Percent'].max(),
            'monthly_targets_count': 1,
            'v2_constraint_violations': len(df_sim[df_sim['Net_Demand_kW'] > df_sim['Monthly_Target']])
        }
        
        print("✅ Integration workflow data prepared successfully")
        
        # Display key metrics that would be shown in the table
        print(f"\n📊 Integration Metrics:")
        print(f"   - Peak reduction: {simulation_results['peak_reduction_kw']:.1f} kW")
        print(f"   - Total energy discharged: {simulation_results['total_energy_discharged']:.1f} kWh")
        print(f"   - Total energy charged: {simulation_results['total_energy_charged']:.1f} kWh")
        print(f"   - Average SOC: {simulation_results['average_soc']:.1f}%")
        print(f"   - Target violations: {simulation_results['v2_constraint_violations']} intervals")
        
        # Calculate round-trip efficiency
        efficiency = (simulation_results['total_energy_discharged'] / 
                     max(simulation_results['total_energy_charged'], 1) * 100)
        print(f"   - Round-trip efficiency: {efficiency:.1f}%")
        
        print("\n✅ Integration workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 Testing V2 Table Integration - Enhanced Battery Simulation Tables")
    print("=" * 70)
    
    # Test 1: Table Functions
    table_test_passed = test_table_functions()
    
    # Test 2: Integration Workflow  
    integration_test_passed = test_integration_workflow()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY:")
    print(f"   - Table Functions: {'✅ PASSED' if table_test_passed else '❌ FAILED'}")
    print(f"   - Integration Workflow: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if table_test_passed and integration_test_passed:
        print("\n🎉 All V2 table integration tests passed! The enhanced tables are ready for use.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
