import numpy as np

# Cost breakdown logic (4-part charges) will be defined here.

# Example stub:
def calculate_cost(df, tariff, power_col, holidays=None):
    """
    Calculate and return the cost breakdown for the given data and tariff object.
    df: DataFrame with a 'Parsed Timestamp' column and power_col (kW)
    tariff: dict, selected tariff object from rp4_tariffs
    power_col: str, name of the power column in df
    holidays: set of datetime.date, optional
    Returns: dict with cost breakdown
    """
    if df.empty or power_col not in df.columns:
        return {"error": "No data or invalid power column."}
    # Sort and calculate time deltas
    df = df.sort_values("Parsed Timestamp")
    time_deltas = df["Parsed Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    interval_kwh = df[power_col] * time_deltas
    total_kwh = interval_kwh.sum()
    # Capacity (kW): use max demand
    max_demand_kw = df[power_col].max()
    # Default breakdown
    breakdown = {
        "Total Energy (kWh)": total_kwh,
        "Max Demand (kW)": max_demand_kw,
        "Energy Cost (RM)": 0,
        "Capacity Cost (RM)": 0,
        "Network Cost (RM)": 0,
        "Retail Cost (RM)": 0,
        "Total Cost (RM)": 0
    }
    # General Tariff (not split)
    if not tariff.get("Split", False):
        energy_cost = total_kwh * tariff.get("Energy Rate", 0)
        capacity_cost = max_demand_kw * tariff.get("Capacity Rate", 0)
        network_cost = max_demand_kw * tariff.get("Network Rate", 0)
        retail_cost = tariff.get("Retail Rate", 0)
        breakdown.update({
            "Energy Cost (RM)": energy_cost,
            "Capacity Cost (RM)": capacity_cost,
            "Network Cost (RM)": network_cost,
            "Retail Cost (RM)": retail_cost,
            "Total Cost (RM)": energy_cost + capacity_cost + network_cost + retail_cost
        })
        return breakdown
    # TOU Tariff (split peak/off-peak)
    from tariffs.peak_logic import is_peak_rp4
    is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays))
    peak_kwh = interval_kwh[is_peak].sum()
    offpeak_kwh = interval_kwh[~is_peak].sum()
    peak_cost = peak_kwh * tariff.get("Peak Rate", 0)
    offpeak_cost = offpeak_kwh * tariff.get("OffPeak Rate", 0)
    capacity_cost = max_demand_kw * tariff.get("Capacity Rate", 0)
    network_cost = max_demand_kw * tariff.get("Network Rate", 0)
    retail_cost = tariff.get("Retail Rate", 0)
    breakdown.update({
        "Peak Energy (kWh)": peak_kwh,
        "OffPeak Energy (kWh)": offpeak_kwh,
        "Peak Cost (RM)": peak_cost,
        "OffPeak Cost (RM)": offpeak_cost,
        "Energy Cost (RM)": peak_cost + offpeak_cost,
        "Capacity Cost (RM)": capacity_cost,
        "Network Cost (RM)": network_cost,
        "Retail Cost (RM)": retail_cost,
        "Total Cost (RM)": peak_cost + offpeak_cost + capacity_cost + network_cost + retail_cost
    })
    return breakdown
