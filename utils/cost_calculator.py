import numpy as np
import pandas as pd
from tariffs.peak_logic import is_peak_rp4

def calculate_cost(df, tariff, power_col, holidays=None, afa_kwh=0, afa_rate=0):
    """
    Calculate and return the cost breakdown for the given data and tariff object.
    df: DataFrame with a 'Parsed Timestamp' column and power_col (kW)
    tariff: dict, selected tariff object from rp4_tariffs
    power_col: str, name of the power column in df
    holidays: set of datetime.date, optional
    afa_kwh: float, optional, AFA kWh adjustment
    afa_rate: float, optional, AFA rate (RM/kWh)
    Returns: dict with cost breakdown
    """
    if df.empty or power_col not in df.columns:
        return {"error": "No data or invalid power column."}
    # Sort and calculate time deltas
    df = df.sort_values("Parsed Timestamp")
    time_deltas = df["Parsed Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    interval_kwh = df[power_col] * time_deltas
    total_kwh = interval_kwh.sum()
    max_demand_kw = df[power_col].max()
    breakdown = {}
    voltage = tariff.get("Voltage", "Low Voltage")
    # General Tariff (not split)
    if not tariff.get("Split", False):
        energy_cost = total_kwh * tariff.get("Energy Rate", 0)
        if voltage == "Low Voltage":
            capacity_cost = total_kwh * tariff.get("Capacity Rate", 0)
            network_cost = total_kwh * tariff.get("Network Rate", 0)
        else:
            capacity_cost = max_demand_kw * tariff.get("Capacity Rate", 0)
            network_cost = max_demand_kw * tariff.get("Network Rate", 0)
        retail_cost = tariff.get("Retail Rate", 0)
        # Calculate AFA adjustment cost
        # If afa_kwh is not provided, use total_kwh
        if afa_kwh == 0:
            afa_kwh = total_kwh
        afa_cost = afa_kwh * afa_rate if afa_rate else 0
        breakdown.update({
            "Total kWh": total_kwh,
            "Max Demand (kW)": max_demand_kw,
            "Energy Cost (RM)": energy_cost,
            "Capacity Cost (RM)": capacity_cost,
            "Network Cost (RM)": network_cost,
            "Retail Cost (RM)": retail_cost,
            "AFA Adjustment": afa_cost,
            "Total Cost": energy_cost + capacity_cost + network_cost + retail_cost + afa_cost
        })
        return breakdown
    # TOU Tariff (split peak/off-peak)
    is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays or set()))
    peak_kwh = interval_kwh[is_peak].sum()
    offpeak_kwh = interval_kwh[~is_peak].sum()
    peak_rate = tariff.get("Peak Rate", 0)
    # Support both 'OffPeak Rate' and 'Off-Peak Rate' keys
    offpeak_rate = (
        tariff.get("Off-Peak Rate")
        if "Off-Peak Rate" in tariff else tariff.get("OffPeak Rate", 0)
    )
    peak_cost = peak_kwh * peak_rate
    offpeak_cost = offpeak_kwh * offpeak_rate
    if voltage == "Low Voltage":
        capacity_cost = total_kwh * tariff.get("Capacity Rate", 0)
        network_cost = total_kwh * tariff.get("Network Rate", 0)
    else:
        capacity_cost = max_demand_kw * tariff.get("Capacity Rate", 0)
        network_cost = max_demand_kw * tariff.get("Network Rate", 0)
    retail_cost = tariff.get("Retail Rate", 0)
    # Calculate AFA adjustment cost
    # If afa_kwh is not provided, use total_kwh
    if afa_kwh == 0:
        afa_kwh = total_kwh
    afa_cost = afa_kwh * afa_rate if afa_rate else 0
    breakdown.update({
        "Peak kWh": peak_kwh,
        "Off-Peak kWh": offpeak_kwh,
        "Peak Rate": peak_rate,
        "Off-Peak Rate": offpeak_rate,
        "Peak Energy Cost": peak_cost,
        "Off-Peak Energy Cost": offpeak_cost,
        "AFA kWh": afa_kwh,
        "AFA Rate": afa_rate,
        "AFA Adjustment": afa_cost,
        "Max Demand (kW)": max_demand_kw,
        "Capacity Rate": tariff.get("Capacity Rate", 0),
        "Network Rate": tariff.get("Network Rate", 0),
        "Capacity Cost": capacity_cost,
        "Network Cost": network_cost,
        "Retail Cost": retail_cost,
        # Add AFA adjustment to total cost
        "Total Cost": peak_cost + offpeak_cost + capacity_cost + network_cost + retail_cost + afa_cost
    })
    return breakdown
