import re
from old_rate import charging_rates

def calculate_old_cost(
    tariff_name,
    total_kwh,
    max_demand_kw=0,
    peak_kwh=0,
    offpeak_kwh=0,
    icpt=0.16
):
    """
    Calculate estimated cost for old TNB tariffs.
    Args:
        tariff_name (str): Name of the old tariff.
        total_kwh (float): Total energy consumed (kWh).
        max_demand_kw (float): Maximum demand (kW).
        peak_kwh (float): Peak period energy (kWh), if applicable.
        offpeak_kwh (float): Off-peak period energy (kWh), if applicable.
        icpt (float): Imbalance Cost Pass-Through (RM/kWh), default 0.16.
    Returns:
        dict: Cost breakdown.
    """
    rate_info = charging_rates.get(tariff_name)
    if not rate_info:
        return {"error": "Tariff not found."}

    result = {"Tariff": tariff_name}

    try:
        # Medium Voltage General, Medium Voltage Commercial
        if "Base" in rate_info and "MD" in rate_info:
            base_rate = float(re.search(r"Base: RM ([\d.]+)", rate_info).group(1))
            md_rate = float(re.search(r"MD: RM ([\d.]+)", rate_info).group(1))
            # For non-TOU tariffs, put all energy in off-peak (common practice)
            result["Peak Energy (kWh)"] = 0
            result["Off-Peak Energy (kWh)"] = total_kwh
            result["Peak Energy Cost"] = 0
            result["Off-Peak Energy Cost"] = total_kwh * base_rate
            result["Energy Cost"] = total_kwh * base_rate
            result["MD Cost"] = max_demand_kw * md_rate
            result["ICPT Cost"] = total_kwh * icpt
            result["Total Cost"] = result["Energy Cost"] + result["MD Cost"] + result["ICPT Cost"]

        # Peak/Off-Peak (TOU)
        elif "Peak:" in rate_info and "Off-Peak:" in rate_info:
            peak_rate = float(re.search(r"Peak: RM ([\d.]+)", rate_info).group(1))
            offpeak_rate = float(re.search(r"Off-Peak: RM ([\d.]+)", rate_info).group(1))
            md_rate = float(re.search(r"MD: RM ([\d.]+)", rate_info).group(1))
            result["Peak Energy (kWh)"] = peak_kwh
            result["Off-Peak Energy (kWh)"] = offpeak_kwh
            result["Peak Energy Cost"] = peak_kwh * peak_rate
            result["Off-Peak Energy Cost"] = offpeak_kwh * offpeak_rate
            result["MD Cost"] = max_demand_kw * md_rate
            result["ICPT Cost"] = total_kwh * icpt
            result["Total Cost"] = (
                result["Peak Energy Cost"] + result["Off-Peak Energy Cost"] + result["MD Cost"] + result["ICPT Cost"]
            )

        # Tiered (Low Voltage Industrial, Domestic)
        elif "Tiered" in rate_info:
            matches = re.findall(r"RM ([\d.]+)", rate_info)
            if matches:
                # Simple logic: use first rate for <=200, last for >200
                if total_kwh <= 200:
                    rate = float(matches[0])
                else:
                    rate = float(matches[-1])
                # For non-TOU tariffs, put all energy in off-peak
                result["Peak Energy (kWh)"] = 0
                result["Off-Peak Energy (kWh)"] = total_kwh
                result["Peak Energy Cost"] = 0
                result["Off-Peak Energy Cost"] = total_kwh * rate
                result["Energy Cost"] = total_kwh * rate
                result["ICPT Cost"] = total_kwh * icpt
                result["Total Cost"] = result["Energy Cost"] + result["ICPT Cost"]
            else:
                result["error"] = "Tiered rate not found."

        # Flat (Low Voltage Commercial)
        elif "Flat" in rate_info:
            flat_rate = float(re.search(r"Flat: RM ([\d.]+)", rate_info).group(1))
            # For non-TOU tariffs, put all energy in off-peak
            result["Peak Energy (kWh)"] = 0
            result["Off-Peak Energy (kWh)"] = total_kwh
            result["Peak Energy Cost"] = 0
            result["Off-Peak Energy Cost"] = total_kwh * flat_rate
            result["Energy Cost"] = total_kwh * flat_rate
            result["ICPT Cost"] = total_kwh * icpt
            result["Total Cost"] = result["Energy Cost"] + result["ICPT Cost"]

        else:
            result["error"] = "Tariff format not recognized."

    except Exception as e:
        result["error"] = f"Calculation error: {e}"

    return result