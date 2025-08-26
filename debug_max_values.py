#!/usr/bin/env python3
"""
Debug script to test the max power shaving and max TOU energy calculation logic
"""

import pandas as pd
import numpy as np

def debug_variable_scope():
    """Debug the variable scope issue in md_shaving_solution_v2.py"""
    
    print("🔍 DEBUGGING VARIABLE SCOPE ISSUE")
    print("=" * 50)
    
    # Simulate the problematic code logic
    print("\n1. Testing the original problematic condition:")
    
    # These variables are never defined in the original code
    max_shaving_power_defined = 'max_shaving_power' in locals()
    max_tou_energy_defined = 'max_tou_energy' in locals()
    total_md_cost_defined = 'total_md_cost' in locals()
    
    print(f"   max_shaving_power in locals(): {max_shaving_power_defined}")
    print(f"   max_tou_energy in locals(): {max_tou_energy_defined}")
    print(f"   total_md_cost in locals(): {total_md_cost_defined}")
    
    # Simulate the calculation block
    print("\n2. Simulating calculation within conditional block:")
    all_monthly_events = [
        {'TOU Required Energy (kWh)': 15.5, 'MD Cost Impact (RM)': 25.0},
        {'TOU Required Energy (kWh)': 12.3, 'MD Cost Impact (RM)': 18.5},
        {'TOU Required Energy (kWh)': 20.1, 'MD Cost Impact (RM)': 32.0}
    ]
    
    monthly_targets = pd.Series([450, 470, 440], index=['2024-01', '2024-02', '2024-03'])
    monthly_max_demands = pd.Series([500, 520, 480], index=['2024-01', '2024-02', '2024-03'])
    
    # Simulate the calculation logic
    max_shaving_power_calc = 0
    max_tou_energy_calc = 0
    total_md_cost_calc = 0
    
    if all_monthly_events:
        if monthly_targets is not None and len(monthly_targets) > 0:
            shaving_amounts = []
            for month_period, target_demand in monthly_targets.items():
                if month_period in monthly_max_demands:
                    max_demand = monthly_max_demands[month_period]
                    shaving_amount = max_demand - target_demand
                    if shaving_amount > 0:
                        shaving_amounts.append(shaving_amount)
            max_shaving_power_calc = max(shaving_amounts) if shaving_amounts else 0
        
        max_tou_energy_calc = max([event.get('TOU Required Energy (kWh)', 0) or 0 for event in all_monthly_events]) if all_monthly_events else 0
        total_md_cost_calc = sum(event.get('MD Cost Impact (RM)', 0) for event in all_monthly_events)
    
    print(f"   max_shaving_power_calc: {max_shaving_power_calc}")
    print(f"   max_tou_energy_calc: {max_tou_energy_calc}")
    print(f"   total_md_cost_calc: {total_md_cost_calc}")
    
    # Test the conditional logic
    print("\n3. Testing the conditional logic:")
    condition1 = 'max_shaving_power' in locals() and 'max_tou_energy' in locals() and 'total_md_cost' in locals()
    print(f"   First condition (original variables): {condition1}")
    print(f"   → Would use: max_shaving_power, max_tou_energy, total_md_cost (UNDEFINED!)")
    
    print(f"   Second condition (calc variables): Always used")
    print(f"   → Would use: {max_shaving_power_calc}, {max_tou_energy_calc}, {total_md_cost_calc}")
    
    # Test the later validation logic
    print("\n4. Testing later validation logic (simulation section):")
    max_shaving_power_calc_exists = 'max_shaving_power_calc' in locals()
    print(f"   max_shaving_power_calc in locals(): {max_shaving_power_calc_exists}")
    print(f"   max_shaving_power_calc > 0: {max_shaving_power_calc > 0}")
    
    # Demonstrate the scope issue
    print("\n5. Demonstrating scope issue:")
    
    def test_scope():
        # Variables defined inside function have local scope
        inner_var = 42
        return 'inner_var' in locals()
    
    result = test_scope()
    inner_var_exists_outside = 'inner_var' in locals()
    
    print(f"   Variable defined inside function visible inside: {result}")
    print(f"   Variable defined inside function visible outside: {inner_var_exists_outside}")
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("The issue is that max_shaving_power, max_tou_energy, and total_md_cost")
    print("are never defined, so the code always uses the _calc versions.")
    print("But later validation checks for _calc variables in wrong scope.")
    print("This creates inconsistent variable naming and validation failures.")
    
    return {
        'max_shaving_power_calc': max_shaving_power_calc,
        'max_tou_energy_calc': max_tou_energy_calc,
        'total_md_cost_calc': total_md_cost_calc
    }

if __name__ == "__main__":
    debug_variable_scope()
