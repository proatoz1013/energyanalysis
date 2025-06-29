import pytest
from src.utils.energy_calculations import calculate_total_power, calculate_cop, calculate_efficiency

def test_calculate_total_power():
    # Test case for total power calculation
    power_data = [100, 200, 300]
    expected_total_power = sum(power_data)
    assert calculate_total_power(power_data) == expected_total_power

def test_calculate_cop():
    # Test case for COP calculation
    cooling_load = 500
    power_usage = 100
    expected_cop = cooling_load / power_usage
    assert calculate_cop(cooling_load, power_usage) == expected_cop

def test_calculate_efficiency():
    # Test case for efficiency calculation
    actual_output = 400
    theoretical_input = 500
    expected_efficiency = (actual_output / theoretical_input) * 100
    assert calculate_efficiency(actual_output, theoretical_input) == expected_efficiency

def test_calculate_efficiency_zero_input():
    # Test case for efficiency calculation with zero theoretical input
    actual_output = 400
    theoretical_input = 0
    with pytest.raises(ZeroDivisionError):
        calculate_efficiency(actual_output, theoretical_input)