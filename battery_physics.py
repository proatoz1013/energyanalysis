"""
Battery Physics Module

Centralized battery physics calculations for C-rate limiting, SOC derating,
and power constraints. This module provides a single source of truth for
battery operation physics across all analysis modules.

Author: Energy Analysis System
Created: November 2025
Version: 1.0.0
"""


def calculate_c_rate_limited_power(current_soc_percent, max_power_rating_kw, 
                                   battery_capacity_kwh, c_rate=1.0, operation='discharge'):
    """
    Calculate power limits with C-rate and SOC derating constraints.
    
    This function serves as the single source of truth for C-rate calculations across:
    - md_shaving_solution_v3.py (V3 simulations)
    - smart_battery_executor.py (battery operations)
    - smart_conservation.py (severity calculations)
    
    The function applies three layers of constraints:
    1. C-rate theoretical limit (capacity × C-rate)
    2. SOC-based derating (reduces power at extreme SOC levels)
    3. Operation-specific factors (charging typically slower than discharging)
    
    Args:
        current_soc_percent (float): Current state of charge (0-100%)
        max_power_rating_kw (float): Battery's rated power capacity (kW)
        battery_capacity_kwh (float): Battery's energy capacity (kWh)
        c_rate (float): Battery's C-rate specification (default: 1.0C)
        operation (str): 'discharge' or 'charge' (default: 'discharge')
    
    Returns:
        dict: {
            'max_power_kw': float - Final constrained power limit
            'c_rate_limit_kw': float - C-rate theoretical limit
            'soc_derating_factor': float - SOC-based derating factor (0.0-1.0)
            'operation_factor': float - Operation-specific factor
            'limiting_factor': str - What's limiting power ('C-rate', 'SOC Derating', or 'Power Rating')
            'effective_c_rate': float - Actual C-rate after all constraints
        }
    
    Example:
        >>> limits = calculate_c_rate_limited_power(
        ...     current_soc_percent=85.0,
        ...     max_power_rating_kw=1734.4,
        ...     battery_capacity_kwh=600.0,
        ...     c_rate=1.0,
        ...     operation='discharge'
        ... )
        >>> print(f"Max discharge power: {limits['max_power_kw']:.1f} kW")
        >>> print(f"Limiting factor: {limits['limiting_factor']}")
    """
    # 1. Calculate C-rate theoretical limit
    c_rate_limit_kw = battery_capacity_kwh * c_rate
    
    # 2. Apply SOC-based derating (safety margins at extreme SOC levels)
    # Updated to match 5%-95% SOC range used in the system
    if current_soc_percent > 95:
        soc_factor = 0.8  # Reduce power at high SOC (approaching 95% max)
    elif current_soc_percent < 10:  # 5% safety minimum + 5% buffer
        soc_factor = 0.7  # Reduce power at low SOC (approaching 5% min)
    else:
        soc_factor = 1.0  # Full power in normal SOC range (10%-95%)
    
    # 3. Apply operation-specific factor
    # Charging is typically slower than discharging for battery health
    if operation == 'charge':
        operation_factor = 0.8  # 80% of discharge rate for charging
    else:
        operation_factor = 1.0  # Full rate for discharge
    
    # 4. Calculate effective C-rate limit with all constraints
    effective_c_rate_limit = c_rate_limit_kw * soc_factor * operation_factor
    
    # 5. Final power limit is minimum of C-rate limit and rated power
    final_power_limit = min(effective_c_rate_limit, max_power_rating_kw)
    
    # 6. Determine which constraint is limiting
    if final_power_limit == effective_c_rate_limit and effective_c_rate_limit < max_power_rating_kw:
        if soc_factor < 1.0:
            limiting_factor = 'SOC Derating'
        else:
            limiting_factor = 'C-rate'
    else:
        limiting_factor = 'Power Rating'
    
    # 7. Calculate actual C-rate being used
    effective_c_rate = final_power_limit / battery_capacity_kwh if battery_capacity_kwh > 0 else 0.0
    
    return {
        'max_power_kw': final_power_limit,
        'c_rate_limit_kw': c_rate_limit_kw,
        'soc_derating_factor': soc_factor,
        'operation_factor': operation_factor,
        'limiting_factor': limiting_factor,
        'effective_c_rate': effective_c_rate
    }


def get_c_rate_info_string(limits_dict):
    """
    Generate human-readable string describing C-rate limits.
    
    Args:
        limits_dict (dict): Output from calculate_c_rate_limited_power()
    
    Returns:
        str: Formatted description of C-rate constraints
    
    Example:
        >>> limits = calculate_c_rate_limited_power(85, 1734, 600, 1.0, 'discharge')
        >>> print(get_c_rate_info_string(limits))
        "Max Power: 600.0 kW (1.00C) | Limited by: Power Rating"
    """
    max_power = limits_dict['max_power_kw']
    c_rate = limits_dict['effective_c_rate']
    limiting = limits_dict['limiting_factor']
    
    return f"Max Power: {max_power:.1f} kW ({c_rate:.2f}C) | Limited by: {limiting}"


def validate_c_rate_parameters(current_soc_percent, max_power_rating_kw, 
                               battery_capacity_kwh, c_rate):
    """
    Validate C-rate calculation parameters.
    
    Args:
        current_soc_percent (float): SOC percentage
        max_power_rating_kw (float): Rated power
        battery_capacity_kwh (float): Battery capacity
        c_rate (float): C-rate specification
    
    Returns:
        tuple: (is_valid, error_message)
    
    Example:
        >>> valid, msg = validate_c_rate_parameters(85, 1734, 600, 1.0)
        >>> if not valid:
        ...     print(f"Error: {msg}")
    """
    if not (0 <= current_soc_percent <= 100):
        return False, f"SOC must be 0-100%, got {current_soc_percent}%"
    
    if max_power_rating_kw <= 0:
        return False, f"Power rating must be positive, got {max_power_rating_kw} kW"
    
    if battery_capacity_kwh <= 0:
        return False, f"Battery capacity must be positive, got {battery_capacity_kwh} kWh"
    
    if c_rate <= 0:
        return False, f"C-rate must be positive, got {c_rate}C"
    
    if c_rate > 3.0:
        return False, f"C-rate {c_rate}C seems unrealistic (max 3C recommended)"
    
    return True, "Valid"
