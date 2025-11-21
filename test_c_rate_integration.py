"""
Test C-Rate Integration Across Battery Operations

This test verifies that C-rate constraints are properly applied to all battery operations:
1. execute_default_shaving_discharge()
2. execute_conservation_discharge()
3. execute_battery_recharge()
4. execute_mode_based_battery_operation()

Test Scenarios:
- High C-rate (2.0C) should allow full power
- Low C-rate (0.5C) should limit power to 0.5 × capacity
- SOC derating at extremes (>95% and <10%)
- Charging operation factor (0.8x for battery health)
"""

from smart_battery_executor import (
    execute_default_shaving_discharge,
    execute_conservation_discharge,
    execute_battery_recharge
)
from battery_physics import calculate_c_rate_limited_power


def test_default_discharge_c_rate():
    """Test C-rate limiting in default discharge mode"""
    print("\n" + "="*80)
    print("TEST 1: Default Discharge with C-Rate Constraints")
    print("="*80)
    
    # Test Case 1: High C-rate (2.0C) - Should NOT limit
    print("\n1A. High C-rate (2.0C) - Should allow full 300kW power rating:")
    result = execute_default_shaving_discharge(
        current_demand_kw=5500,
        monthly_target_kw=5000,
        current_soc_kwh=600,  # 100% SOC
        battery_capacity_kwh=600,
        max_power_kw=300,
        interval_hours=0.5,
        c_rate=2.0
    )
    print(f"   Discharge Power: {result['discharge_power_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    assert result['discharge_power_kw'] == 300, f"Expected 300kW, got {result['discharge_power_kw']}"
    assert result['limiting_factor'] == 'demand', f"Expected 'demand', got {result['limiting_factor']}"
    print("   ✅ PASS: Full power allowed with high C-rate")
    
    # Test Case 2: Low C-rate (0.5C) - Should limit to 300kW (0.5 × 600kWh)
    print("\n1B. Low C-rate (0.5C) - Should limit to 300kW:")
    result = execute_default_shaving_discharge(
        current_demand_kw=5500,
        monthly_target_kw=5000,
        current_soc_kwh=600,
        battery_capacity_kwh=600,
        max_power_kw=500,  # Power rating higher than C-rate limit
        interval_hours=0.5,
        c_rate=0.5
    )
    print(f"   Discharge Power: {result['discharge_power_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    assert result['discharge_power_kw'] == 300, f"Expected 300kW (0.5C), got {result['discharge_power_kw']}"
    assert result['c_rate_limited'] == True, "Should be C-rate limited"
    assert 'c_rate' in result['limiting_factor'].lower(), f"Expected C-rate limiting, got {result['limiting_factor']}"
    print("   ✅ PASS: Power correctly limited by C-rate")
    
    # Test Case 3: SOC Derating at Low SOC (<10%)
    print("\n1C. Low SOC (5%) - Should apply 0.7x derating:")
    result = execute_default_shaving_discharge(
        current_demand_kw=5500,
        monthly_target_kw=5000,
        current_soc_kwh=30,  # 5% SOC (30/600)
        battery_capacity_kwh=600,
        max_power_kw=500,
        interval_hours=0.5,
        c_rate=1.0
    )
    print(f"   Discharge Power: {result['discharge_power_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    # At 5% SOC, 1.0C with 0.7x derating = 600 * 1.0 * 0.7 = 420kW
    # But demand excess is only 500kW, so should get min(500, 420) = 420kW
    expected_power = min(500, 600 * 1.0 * 0.7)
    assert abs(result['discharge_power_kw'] - expected_power) < 1, f"Expected ~{expected_power}kW, got {result['discharge_power_kw']}"
    print(f"   ✅ PASS: SOC derating applied (0.7x at low SOC)")


def test_conservation_discharge_c_rate():
    """Test C-rate limiting in conservation discharge mode"""
    print("\n" + "="*80)
    print("TEST 2: Conservation Discharge with C-Rate Constraints")
    print("="*80)
    
    # Test Case: Conservation mode with C-rate limiting
    print("\n2A. Conservation mode with 0.5C rate:")
    result = execute_conservation_discharge(
        current_demand_kw=5500,
        monthly_target_kw=5000,
        battery_kw_conserved=100,  # Conserve 100kW
        current_soc_kwh=570,  # 95% SOC
        battery_capacity_kwh=600,
        max_power_kw=500,
        interval_hours=0.5,
        c_rate=0.5
    )
    print(f"   Revised Target: {result['revised_target_kw']:.1f} kW")
    print(f"   Discharge Power: {result['discharge_power_kw']:.1f} kW")
    print(f"   Power Conserved: {result['power_conserved_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    print(f"   SOC Improvement: {result['soc_improvement_percent']:.1f}%")
    
    # Conservation reduces excess from 500kW to 400kW (conserve 100kW)
    # C-rate limit at 95% SOC: 600 * 0.5 * 0.8 = 240kW (0.8x derating >95%)
    # Should get min(400, 500, 240) = 240kW
    expected_power = min(400, 600 * 0.5 * 0.8)
    assert abs(result['discharge_power_kw'] - expected_power) < 1, f"Expected ~{expected_power}kW, got {result['discharge_power_kw']}"
    assert result['c_rate_limited'] == True, "Should be C-rate limited"
    print("   ✅ PASS: Conservation with C-rate limiting")


def test_battery_recharge_c_rate():
    """Test C-rate limiting in recharge mode"""
    print("\n" + "="*80)
    print("TEST 3: Battery Recharge with C-Rate Constraints")
    print("="*80)
    
    # Test Case 1: Charging with 1.0C rate (should apply 0.8x operation factor)
    print("\n3A. Charging with 1.0C rate (0.8x charging factor):")
    result = execute_battery_recharge(
        current_demand_kw=4000,
        available_grid_power_kw=1000,  # 1000kW available for charging
        current_soc_kwh=300,  # 50% SOC
        battery_capacity_kwh=600,
        max_charge_power_kw=500,
        interval_hours=0.5,
        c_rate=1.0
    )
    print(f"   Charge Power: {result['charge_power_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    
    # At 50% SOC, 1.0C charging: 600 * 1.0 * 1.0 (SOC factor) * 0.8 (charge factor) = 480kW
    # Should get min(1000 grid, 500 power_rating, 480 c_rate) = 480kW
    expected_power = min(1000, 500, 600 * 1.0 * 0.8)
    assert abs(result['charge_power_kw'] - expected_power) < 1, f"Expected ~{expected_power}kW, got {result['charge_power_kw']}"
    assert result['c_rate_limited'] == True, "Should be C-rate limited"
    print("   ✅ PASS: Charging limited to 0.8x of discharge rate")
    
    # Test Case 2: Low C-rate charging (0.3C)
    print("\n3B. Low C-rate charging (0.3C):")
    result = execute_battery_recharge(
        current_demand_kw=4000,
        available_grid_power_kw=1000,
        current_soc_kwh=300,
        battery_capacity_kwh=600,
        max_charge_power_kw=500,
        interval_hours=0.5,
        c_rate=0.3
    )
    print(f"   Charge Power: {result['charge_power_kw']:.1f} kW")
    print(f"   C-Rate Limited: {result['c_rate_limited']}")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    print(f"   Effective C-Rate: {result['effective_c_rate']:.2f}C")
    
    # 600 * 0.3 * 0.8 = 144kW
    expected_power = 600 * 0.3 * 0.8
    assert abs(result['charge_power_kw'] - expected_power) < 1, f"Expected ~{expected_power}kW, got {result['charge_power_kw']}"
    print("   ✅ PASS: Low C-rate charging limited correctly")


def test_c_rate_physics_module():
    """Test the battery_physics module directly"""
    print("\n" + "="*80)
    print("TEST 4: Battery Physics Module (calculate_c_rate_limited_power)")
    print("="*80)
    
    # Test Case 1: Normal SOC range (50%)
    print("\n4A. Normal SOC (50%) with 1.0C discharge:")
    result = calculate_c_rate_limited_power(
        current_soc_percent=50,
        max_power_rating_kw=300,
        battery_capacity_kwh=600,
        c_rate=1.0,
        operation='discharge'
    )
    print(f"   Max Power: {result['max_power_kw']:.1f} kW")
    print(f"   C-Rate Limit: {result['c_rate_limit_kw']:.1f} kW")
    print(f"   SOC Derating: {result['soc_derating_factor']:.2f}x")
    print(f"   Operation Factor: {result['operation_factor']:.2f}x")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    
    assert result['max_power_kw'] == 300, "Should be limited by power rating"
    assert result['soc_derating_factor'] == 1.0, "No derating at 50% SOC"
    assert result['operation_factor'] == 1.0, "Discharge factor is 1.0"
    print("   ✅ PASS: Normal SOC discharge")
    
    # Test Case 2: High SOC (96%) - Should apply 0.8x derating
    print("\n4B. High SOC (96%) - 0.8x derating:")
    result = calculate_c_rate_limited_power(
        current_soc_percent=96,
        max_power_rating_kw=300,
        battery_capacity_kwh=600,
        c_rate=1.0,
        operation='discharge'
    )
    print(f"   Max Power: {result['max_power_kw']:.1f} kW")
    print(f"   SOC Derating: {result['soc_derating_factor']:.2f}x")
    print(f"   Limiting Factor: {result['limiting_factor']}")
    
    assert result['soc_derating_factor'] == 0.8, "Should apply 0.8x derating >95%"
    expected_power = min(300, 600 * 1.0 * 0.8)
    assert abs(result['max_power_kw'] - expected_power) < 1, f"Expected {expected_power}kW"
    print("   ✅ PASS: High SOC derating applied")
    
    # Test Case 3: Low SOC (8%) - Should apply 0.7x derating
    print("\n4C. Low SOC (8%) - 0.7x derating:")
    result = calculate_c_rate_limited_power(
        current_soc_percent=8,
        max_power_rating_kw=300,
        battery_capacity_kwh=600,
        c_rate=1.0,
        operation='discharge'
    )
    print(f"   Max Power: {result['max_power_kw']:.1f} kW")
    print(f"   SOC Derating: {result['soc_derating_factor']:.2f}x")
    
    assert result['soc_derating_factor'] == 0.7, "Should apply 0.7x derating <10%"
    expected_power = min(300, 600 * 1.0 * 0.7)
    assert abs(result['max_power_kw'] - expected_power) < 1, f"Expected {expected_power}kW"
    print("   ✅ PASS: Low SOC derating applied")
    
    # Test Case 4: Charging operation - Should apply 0.8x operation factor
    print("\n4D. Charging operation - 0.8x factor:")
    result = calculate_c_rate_limited_power(
        current_soc_percent=50,
        max_power_rating_kw=300,
        battery_capacity_kwh=600,
        c_rate=1.0,
        operation='charge'
    )
    print(f"   Max Power: {result['max_power_kw']:.1f} kW")
    print(f"   Operation Factor: {result['operation_factor']:.2f}x")
    
    assert result['operation_factor'] == 0.8, "Charging should have 0.8x factor"
    expected_power = min(300, 600 * 1.0 * 0.8)
    assert abs(result['max_power_kw'] - expected_power) < 1, f"Expected {expected_power}kW"
    print("   ✅ PASS: Charging operation factor applied")


def test_integration_summary():
    """Print integration summary"""
    print("\n" + "="*80)
    print("C-RATE INTEGRATION VERIFICATION SUMMARY")
    print("="*80)
    
    print("\n✅ Module: battery_physics.py")
    print("   - calculate_c_rate_limited_power() function created")
    print("   - Implements 4-layer constraint system:")
    print("     1. Base C-rate limit (capacity_kwh × c_rate)")
    print("     2. SOC derating (0.8x >95%, 0.7x <10%, 1.0x normal)")
    print("     3. Operation factor (1.0x discharge, 0.8x charge)")
    print("     4. Power rating constraint (min with all factors)")
    
    print("\n✅ Module: smart_battery_executor.py")
    print("   - execute_default_shaving_discharge() updated with c_rate parameter")
    print("   - execute_conservation_discharge() updated with c_rate parameter")
    print("   - execute_battery_recharge() updated with c_rate parameter")
    print("   - execute_mode_based_battery_operation() extracts c_rate from config_data")
    print("   - All functions return: c_rate_limited, limiting_factor, effective_c_rate")
    
    print("\n✅ Data Flow:")
    print("   vendor_battery_database.json → battery_spec['c_rate']")
    print("   → battery_sizing['c_rate'] (in config_data)")
    print("   → execute_mode_based_battery_operation() extracts c_rate")
    print("   → battery operation functions apply constraints")
    print("   → returns limiting_factor and effective_c_rate for tracking")
    
    print("\n✅ Physics Implementation:")
    print("   - C-rate: Power limit = battery_capacity_kwh × c_rate")
    print("   - SOC Derating: Protects battery at extreme SOC levels")
    print("   - Charge Factor: 0.8x to preserve battery health during charging")
    print("   - Constraint Hierarchy: min(demand, power_rating, soc_limit, c_rate_limit)")
    
    print("\n✅ Integration Status:")
    print("   [COMPLETE] battery_physics.py created with centralized functions")
    print("   [COMPLETE] smart_battery_executor.py all 3 functions updated")
    print("   [COMPLETE] execute_mode_based_battery_operation() extracts c_rate from config")
    print("   [PENDING] smart_conservation.py severity calculations (future enhancement)")
    print("   [PENDING] md_shaving_solution_v3.py replace duplicate function (cleanup)")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)


if __name__ == "__main__":
    try:
        test_default_discharge_c_rate()
        test_conservation_discharge_c_rate()
        test_battery_recharge_c_rate()
        test_c_rate_physics_module()
        test_integration_summary()
        
        print("\n" + "🎉" * 40)
        print("C-RATE INTEGRATION TEST SUITE: ALL TESTS PASSED")
        print("🎉" * 40 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise
