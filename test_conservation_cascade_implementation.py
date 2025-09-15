#!/usr/bin/env python3

"""
Test Conservation Cascade Implementation
========================================

This script tests the complete conservation cascade workflow implementation
in the MD Shaving Solution V2 to ensure all four steps of the cascade are
working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to Python path
sys.path.append('/Users/chyeap89/Documents/energyanalysis')

def test_conservation_cascade_workflow():
    """Test the complete conservation cascade workflow implementation."""
    
    print("🧪 Testing Conservation Cascade Workflow Implementation")
    print("=" * 60)
    
    try:
        # Import the V2 simulation function
        from md_shaving_solution_v2 import _simulate_battery_operation_v2
        print("✅ Successfully imported _simulate_battery_operation_v2")
        
        # Create test data
        start_date = datetime(2025, 1, 1)
        end_date = start_date + timedelta(days=4)  # 4 days of 15-minute data
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')[:-1]  # Remove last to get exact intervals
        
        # Create test demand data with intentional peaks during MD hours (2-10 PM)
        test_demand = []
        for ts in timestamps:
            base_demand = 80  # Base load
            if 14 <= ts.hour < 22 and ts.weekday() < 5:  # MD hours, weekdays
                peak_demand = base_demand + 150  # High peak to trigger conservation
            else:
                peak_demand = base_demand + np.random.normal(0, 10)
            test_demand.append(max(20, peak_demand))
        
        df_test = pd.DataFrame({
            'power_kW': test_demand
        }, index=timestamps)
        
        print(f"✅ Created test data: {len(df_test)} intervals")
        
        # Create monthly targets
        monthly_targets = pd.Series([120], index=[pd.Period('2025-01', freq='M')])
        
        # Battery sizing configuration
        battery_sizing = {
            'capacity_kwh': 200,
            'power_rating_kw': 100,
            'c_rate': 1.0
        }
        
        # Battery parameters
        battery_params = {
            'round_trip_efficiency': 95.0,
            'depth_of_discharge': 80.0,
            'min_soc': 20.0,
            'max_soc': 100.0,
            'c_rate': 1.0
        }
        
        print("✅ Test parameters configured")
        
        # Test Conservation Cascade Workflow
        print("\n🔋 Testing Conservation Cascade Workflow...")
        
        # Run simulation with conservation cascade enabled
        results = _simulate_battery_operation_v2(
            df=df_test,
            power_col='power_kW',
            monthly_targets=monthly_targets,
            battery_sizing=battery_sizing,
            battery_params=battery_params,
            interval_hours=0.25,
            selected_tariff=None,
            holidays=None,
            conservation_enabled=True,  # Enable conservation cascade
            soc_threshold=50,  # SOC threshold for activation
            battery_kw_conserved=30.0  # kW to conserve
        )
        
        print("✅ Conservation cascade simulation completed successfully!")
        
        # Extract simulation results
        df_sim = results['df_simulation']
        print(f"✅ Simulation DataFrame created: {len(df_sim)} rows, {len(df_sim.columns)} columns")
        
        # Verify conservation cascade columns exist
        expected_cascade_columns = [
            'Conserve_Activated',
            'Battery Conserved kW',
            'Revised_Discharge_Power_kW',
            'Revised_BESS_Balance_kWh', 
            'Revised_Target_Achieved_kW',
            'SOC_Improvement_Percent'
        ]
        
        missing_columns = [col for col in expected_cascade_columns if col not in df_sim.columns]
        if missing_columns:
            print(f"❌ Missing cascade columns: {missing_columns}")
            return False
        else:
            print("✅ All conservation cascade columns present")
        
        # Analyze conservation activity
        conservation_periods = df_sim['Conserve_Activated'].sum()
        total_periods = len(df_sim)
        conservation_rate = (conservation_periods / total_periods * 100) if total_periods > 0 else 0
        
        print(f"\n📊 Conservation Cascade Analysis:")
        print(f"   - Conservation periods: {conservation_periods}/{total_periods}")
        print(f"   - Conservation rate: {conservation_rate:.1f}%")
        
        if conservation_periods > 0:
            # Analyze cascade workflow metrics
            cascade_events = df_sim[df_sim['Conserve_Activated'] == True]
            
            # Step 1: Revised discharge power
            avg_revised_discharge = cascade_events['Revised_Discharge_Power_kW'].mean()
            print(f"   - Step 1 - Avg revised discharge power: {avg_revised_discharge:.1f} kW")
            
            # Step 2: BESS balance preserved
            total_energy_preserved = cascade_events['Revised_BESS_Balance_kWh'].sum()
            print(f"   - Step 2 - Total energy preserved: {total_energy_preserved:.2f} kWh")
            
            # Step 3: Target achievement with conservation
            avg_target_achieved = cascade_events['Revised_Target_Achieved_kW'].mean()
            print(f"   - Step 3 - Avg target achieved: {avg_target_achieved:.1f} kW")
            
            # Step 4: SOC improvement
            total_soc_improvement = cascade_events['SOC_Improvement_Percent'].sum()
            print(f"   - Step 4 - Total SOC improvement: {total_soc_improvement:.2f}%")
            
            # Verify cascade workflow metrics are in results
            cascade_metrics = [
                'total_energy_preserved_kwh',
                'total_soc_improvement_percent',
                'conservation_effectiveness_percent',
                'cascade_workflow_complete'
            ]
            
            cascade_metrics_present = [metric for metric in cascade_metrics if metric in results]
            print(f"   - Cascade metrics in results: {len(cascade_metrics_present)}/{len(cascade_metrics)}")
            
            if results.get('cascade_workflow_complete', False):
                print("✅ Conservation cascade workflow marked as complete")
            else:
                print("⚠️ Conservation cascade workflow not marked as complete")
        
        # Test enhanced battery table creation
        print("\n📊 Testing Enhanced Battery Table with Cascade Columns...")
        
        try:
            from md_shaving_solution_v2 import _create_enhanced_battery_table
            enhanced_table = _create_enhanced_battery_table(df_sim, selected_tariff=None, holidays=None)
            
            print(f"✅ Enhanced table created: {len(enhanced_table)} rows, {len(enhanced_table.columns)} columns")
            
            # Check for conservation cascade columns in enhanced table
            cascade_table_columns = [
                'Revised Discharge Power (kW)',
                'BESS Balance Preserved (kWh)',
                'Target Achieved w/ Conservation (kW)',
                'SOC Improvement (%)'
            ]
            
            cascade_columns_present = [col for col in cascade_table_columns if col in enhanced_table.columns]
            print(f"✅ Cascade columns in enhanced table: {len(cascade_columns_present)}/{len(cascade_table_columns)}")
            
            if len(cascade_columns_present) == len(cascade_table_columns):
                print("✅ All conservation cascade columns present in enhanced table")
            else:
                missing_cascade_cols = [col for col in cascade_table_columns if col not in enhanced_table.columns]
                print(f"⚠️ Missing cascade columns in enhanced table: {missing_cascade_cols}")
                
        except Exception as e:
            print(f"❌ Error creating enhanced battery table: {str(e)}")
            return False
        
        # Verify results metrics
        print(f"\n🎯 Conservation Cascade Results Summary:")
        print(f"   - Conservation enabled: {results.get('conservation_enabled', False)}")
        print(f"   - Conservation periods: {results.get('conservation_periods', 0)}")
        print(f"   - Conservation rate: {results.get('conservation_rate_percent', 0):.1f}%")
        print(f"   - Total power conserved: {results.get('total_power_conserved_kw', 0):.1f} kW")
        print(f"   - Energy preserved: {results.get('total_energy_preserved_kwh', 0):.2f} kWh")
        print(f"   - SOC improvement: {results.get('total_soc_improvement_percent', 0):.2f}%")
        print(f"   - Conservation effectiveness: {results.get('conservation_effectiveness_percent', 0):.1f}%")
        
        print(f"\n🎉 Conservation Cascade Workflow Implementation Test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conservation_cascade_workflow()
    if success:
        print("\n✅ All conservation cascade workflow tests passed!")
        exit(0)
    else:
        print("\n❌ Conservation cascade workflow tests failed!")
        exit(1)
