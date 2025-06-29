def calculate_total_power(chiller_power, pump_power):
    """Calculate total power usage of the chiller plant."""
    return chiller_power + pump_power

def calculate_kw_tr(total_power, cooling_load):
    """Calculate kW/TR (kilowatts per ton of refrigeration)."""
    if cooling_load == 0:
        return 0
    return total_power / cooling_load

def calculate_cop(total_power, cooling_load):
    """Calculate Coefficient of Performance (COP)."""
    if total_power == 0:
        return 0
    return cooling_load / total_power

def calculate_average_efficiency(efficiency_list):
    """Calculate average efficiency from a list of efficiency values."""
    if not efficiency_list:
        return 0
    return sum(efficiency_list) / len(efficiency_list)