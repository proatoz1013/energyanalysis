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
    # Sort and calculate time deltas using proper interval detection
    df = df.sort_values("Parsed Timestamp")
    
    # === FIXED INTERVAL DETECTION ===
    # Detect data interval from the entire dataset using mode (most common interval)
    if len(df) > 1:
        time_diffs = df["Parsed Timestamp"].diff().dropna()
        if len(time_diffs) > 0:
            # Get the most common time interval (mode)
            most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
            interval_seconds = most_common_interval.total_seconds()
            # Protect against zero division
            if interval_seconds > 0:
                interval_hours = interval_seconds / 3600
            else:
                # Fallback to 15 minutes if interval is zero
                interval_hours = 0.25
        else:
            # Fallback to 15 minutes if we can't determine interval
            interval_hours = 0.25
    else:
        interval_hours = 0.25
    
    # Calculate energy using consistent interval detection
    # Each power reading represents the average power for that interval
    interval_kwh = df[power_col] * interval_hours
    total_kwh = interval_kwh.sum()
    
    # --- Calculate 30-min rolling average demand for accurate billing ---
    # Protect against zero interval hours
    if interval_hours > 0:
        intervals_per_30min = max(1, int(round(0.5 / interval_hours)))
    else:
        intervals_per_30min = 1  # Fallback to 1 interval
    
    rolling_30min_avg = df[power_col].rolling(window=intervals_per_30min, min_periods=1).mean()
    max_demand_kw = rolling_30min_avg.max() if not rolling_30min_avg.empty and not rolling_30min_avg.isna().all() else 0
    
    # For TOU tariffs, also calculate peak period max demand using 30-min rolling average
    is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays or set()))
    peak_rolling_30min_avg = df[power_col].where(is_peak).rolling(window=intervals_per_30min, min_periods=1).mean()
    peak_demand_kw = (peak_rolling_30min_avg.max() 
                     if is_peak.any() and not peak_rolling_30min_avg.empty and not peak_rolling_30min_avg.isna().all() 
                     else 0)
    
    # Ensure no NaN values
    max_demand_kw = max_demand_kw if pd.notna(max_demand_kw) else 0
    peak_demand_kw = peak_demand_kw if pd.notna(peak_demand_kw) else 0
    
    rules = tariff.get("Rules", {})
    rates = tariff.get("Rates", {})
    
    # Check if this is a TOU tariff or General tariff
    if not rules.get("has_peak_split", False):
        # ===== GENERAL TARIFF (single rate for all periods) =====
        energy_cost = total_kwh * rates.get("Energy Rate", 0)
        
        # Use rules for capacity and network charge basis
        if rules.get("charge_capacity_by", "kWh") == "kWh":
            capacity_basis = total_kwh
            show_capacity_demand = False
        elif rules.get("charge_capacity_by") == "kW (peak only)":
            capacity_basis = peak_demand_kw
            show_capacity_demand = True
        else:
            capacity_basis = max_demand_kw
            show_capacity_demand = True
            
        if rules.get("charge_network_by", "kWh") == "kWh":
            network_basis = total_kwh
            show_network_demand = False
        elif rules.get("charge_network_by") == "kW (peak only)":
            network_basis = peak_demand_kw
            show_network_demand = True
        else:
            network_basis = max_demand_kw
            show_network_demand = True
            
        capacity_cost = capacity_basis * rates.get("Capacity Rate", 0)
        network_cost = network_basis * rates.get("Network Rate", 0)
        retail_cost = rates.get("Retail Rate", 0)
        
        # Calculate AFA adjustment cost only if applicable
        afa_cost = 0
        if rules.get("afa_applicable", False):
            if afa_kwh == 0:
                afa_kwh = total_kwh
            afa_cost = afa_kwh * afa_rate if afa_rate else 0
        else:
            afa_kwh = 0
            afa_rate = 0
            
        # Calculate KTWBB adjustment cost only if applicable
        ktwbb_cost = 0
        if rules.get("ktwbb_applicable", False):
            ktwbb_rate = rates.get("KTWBB Rate", 0)
            # KTWBB = 1.6% of (Energy + Capacity + Network + Retail + AFA)
            ktwbb_base = energy_cost + capacity_cost + network_cost + retail_cost + afa_cost
            ktwbb_cost = ktwbb_base * ktwbb_rate
        else:
            ktwbb_rate = 0
            
        breakdown = {
            "Total kWh": total_kwh,
            "Energy Rate": rates.get("Energy Rate", 0),
            "Capacity Rate": rates.get("Capacity Rate", 0),
            "Network Rate": rates.get("Network Rate", 0),
            "Retail Rate": rates.get("Retail Rate", 0),
            "KTWBB Rate": rates.get("KTWBB Rate", 0) if rules.get("ktwbb_applicable", False) else 0,
            "Energy Cost (RM)": energy_cost,
            "Capacity Cost (RM)": capacity_cost,
            "Network Cost (RM)": network_cost,
            "Retail Cost": retail_cost,
            "KTWBB Cost": ktwbb_cost,
            "AFA Adjustment": afa_cost,
            "Total Cost": energy_cost + capacity_cost + network_cost + retail_cost + afa_cost + ktwbb_cost,
            # For General tariffs, all energy is at a single rate (no peak/off-peak distinction)
            "Total Energy kWh": total_kwh,
            "Total Energy Cost": energy_cost,
            "Peak kWh": 0,  # No peak/off-peak for General tariffs
            "Off-Peak kWh": 0,
            "Peak Energy Cost": 0,
            "Off-Peak Energy Cost": 0,
            "AFA kWh": afa_kwh,
            "AFA Rate": afa_rate
        }
        
        # Only include demand fields if relevant (kW-based)
        if show_capacity_demand:
            breakdown["Max Demand (kW)"] = peak_demand_kw
        if show_network_demand and not show_capacity_demand:
            breakdown["Max Demand (kW)"] = peak_demand_kw
            
    else:
        # ===== TOU TARIFF (split peak/off-peak) =====
        peak_kwh = interval_kwh[is_peak].sum()
        offpeak_kwh = interval_kwh[~is_peak].sum()
        peak_rate = rates.get("Peak Rate", 0)
        offpeak_rate = rates.get("Off-Peak Rate", rates.get("OffPeak Rate", 0))
        peak_cost = peak_kwh * peak_rate
        offpeak_cost = offpeak_kwh * offpeak_rate
        
        # Use rules for capacity and network charge basis
        if rules.get("charge_capacity_by", "kWh") == "kWh":
            capacity_basis = total_kwh
            show_capacity_demand = False
        elif rules.get("charge_capacity_by") == "kW (peak only)":
            capacity_basis = peak_demand_kw
            show_capacity_demand = True
        else:
            capacity_basis = max_demand_kw
            show_capacity_demand = True
            
        if rules.get("charge_network_by", "kWh") == "kWh":
            network_basis = total_kwh
            show_network_demand = False
        elif rules.get("charge_network_by") == "kW (peak only)":
            network_basis = peak_demand_kw
            show_network_demand = True
        else:
            network_basis = max_demand_kw
            show_network_demand = True
            
        capacity_cost = capacity_basis * rates.get("Capacity Rate", 0)
        network_cost = network_basis * rates.get("Network Rate", 0)
        retail_cost = rates.get("Retail Rate", 0)
        
        # Calculate AFA adjustment cost only if applicable
        afa_cost = 0
        if rules.get("afa_applicable", False):
            if afa_kwh == 0:
                afa_kwh = total_kwh
            afa_cost = afa_kwh * afa_rate if afa_rate else 0
        else:
            afa_kwh = 0
            afa_rate = 0
            
        # Calculate KTWBB adjustment cost only if applicable
        ktwbb_cost = 0
        if rules.get("ktwbb_applicable", False):
            ktwbb_rate = rates.get("KTWBB Rate", 0)
            # KTWBB = 1.6% of (Energy + Capacity + Network + Retail + AFA)
            ktwbb_base = peak_cost + offpeak_cost + capacity_cost + network_cost + retail_cost + afa_cost
            ktwbb_cost = ktwbb_base * ktwbb_rate
        else:
            ktwbb_rate = 0
            
        breakdown = {
            "Peak kWh": peak_kwh,
            "Off-Peak kWh": offpeak_kwh,
            "Peak Rate": peak_rate,
            "Off-Peak Rate": offpeak_rate,
            "Capacity Rate": rates.get("Capacity Rate", 0),
            "Network Rate": rates.get("Network Rate", 0),
            "Retail Rate": retail_cost,
            "KTWBB Rate": rates.get("KTWBB Rate", 0) if rules.get("ktwbb_applicable", False) else 0,
            "Peak Energy Cost": peak_cost,
            "Off-Peak Energy Cost": offpeak_cost,
            "AFA kWh": afa_kwh if rules.get("afa_applicable", False) else 0,
            "AFA Rate": afa_rate if rules.get("afa_applicable", False) else 0,
            "AFA Adjustment": afa_cost,
            "KTWBB Cost": ktwbb_cost,
            "Capacity Cost": capacity_cost,
            "Network Cost": network_cost,
            "Retail Cost": retail_cost,
            "Total Cost": peak_cost + offpeak_cost + capacity_cost + network_cost + retail_cost + afa_cost + ktwbb_cost,
            "Total kWh": total_kwh
        }
        
        # Only include demand fields if relevant (kW-based)
        if show_capacity_demand:
            breakdown["Max Demand (kW)"] = max_demand_kw
            breakdown["Peak Demand (kW, Peak Period Only)"] = peak_demand_kw
        if show_network_demand and not show_capacity_demand:
            breakdown["Max Demand (kW)"] = max_demand_kw
    
    # --- Add Cost per kWh for both tariff types ---
    if breakdown.get("Total Cost", 0) and total_kwh:
        try:
            breakdown["Cost per kWh (Total Cost / Total kWh)"] = breakdown["Total Cost"] / total_kwh
        except ZeroDivisionError:
            breakdown["Cost per kWh (Total Cost / Total kWh)"] = None
    else:
        breakdown["Cost per kWh (Total Cost / Total kWh)"] = None
        
    return breakdown

def format_cost_breakdown(breakdown):
    """
    Format the cost breakdown dictionary into a DataFrame matching the requested table format, with improved clarity and formatting.
    """
    def fmt(val):
        if val is None or val == "":
            return ""
        if isinstance(val, (int, float)):
            if abs(val) < 1e-6:
                return ""
            return f"{val:,.2f}"
        return val

    is_tou = "Peak Energy Cost" in breakdown and "Off-Peak Energy Cost" in breakdown
    if is_tou:
        total_energy_cost = (breakdown.get("Peak Energy Cost", 0) or 0) + (breakdown.get("Off-Peak Energy Cost", 0) or 0)
    else:
        total_energy_cost = breakdown.get("Energy Cost (RM)", "")

    # Section A: Energy Consumption
    rows = [
        {"No": "A", "Description": "A. Energy Consumption kWh", "Unit": "kWh", "Value": fmt(breakdown.get("Total kWh", "")), "Unit Rate (RM)": "", "Total Cost (RM)": fmt(total_energy_cost)},
        {"No": "1", "Description": "Peak Period Consumption", "Unit": "kWh", "Value": fmt(breakdown.get("Peak kWh", "")), "Unit Rate (RM)": fmt(breakdown.get("Peak Rate", "")), "Total Cost (RM)": fmt(breakdown.get("Peak Energy Cost", ""))},
        {"No": "2", "Description": "Off-Peak Consumption", "Unit": "kWh", "Value": fmt(breakdown.get("Off-Peak kWh", "")), "Unit Rate (RM)": fmt(breakdown.get("Off-Peak Rate", "")), "Total Cost (RM)": fmt(breakdown.get("Off-Peak Energy Cost", ""))},
        {"No": "3", "Description": "AFA Consumption", "Unit": "kWh", "Value": fmt(breakdown.get("AFA kWh", "")), "Unit Rate (RM)": fmt(breakdown.get("AFA Rate", "")), "Total Cost (RM)": fmt(breakdown.get("AFA Adjustment", ""))},
        {"No": "4", "Description": "KTWBB Consumption", "Unit": "kWh", "Value": fmt(breakdown.get("Total kWh", "")) if breakdown.get("KTWBB Rate", 0) > 0 else "", "Unit Rate (RM)": fmt(breakdown.get("KTWBB Rate", "")), "Total Cost (RM)": fmt(breakdown.get("KTWBB Cost", ""))},
    ]
    # Section B: Maximum Demand
    # Always show correct Capacity Cost (check both keys)
    capacity_cost_val = breakdown.get("Capacity Cost", breakdown.get("Capacity Cost (RM)", ""))
    rows += [
        {"No": "B", "Description": "B. Maximum Demand (Peak Demand)", "Unit": "kW", "Value": fmt(breakdown.get("Max Demand (kW)", "")), "Unit Rate (RM)": "", "Total Cost (RM)": fmt(capacity_cost_val)},
        {"No": "1", "Description": "Network Charge", "Unit": "kW", "Value": fmt(breakdown.get("Max Demand (kW)", "")), "Unit Rate (RM)": fmt(breakdown.get("Network Rate", "")), "Total Cost (RM)": fmt(breakdown.get("Network Cost", ""))},
        {"No": "2", "Description": "Retail Charge", "Unit": "kW", "Value": fmt(breakdown.get("Max Demand (kW)", "")), "Unit Rate (RM)": fmt(breakdown.get("Retail Rate", "")), "Total Cost (RM)": fmt(breakdown.get("Retail Cost", ""))},
    ]
    # Section C: Others Charges
    rows.append({"No": "C", "Description": "Others Charges", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": fmt(breakdown.get("Others Charges", 0))})
    # Total row
    rows.append({"No": "", "Description": "Total Estimated Cost", "Unit": "", "Value": "", "Unit Rate (RM)": "", "Total Cost (RM)": fmt(breakdown.get("Total Cost", ""))})

    df = pd.DataFrame(rows)
    # Remove the first column (index 0) from the DataFrame before returning
    df = df.iloc[:, 1:]
    return df

def html_cost_table(cost_breakdown):
    """
    Generate an HTML table for the cost breakdown.
    cost_breakdown: dict, the cost breakdown dictionary.
    Returns: str, HTML string for the table.
    """
    rows = []
    for key, value in cost_breakdown.items():
        rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    table_html = (
        "<table class='cost-table'>"
        "<thead><tr><th>Description</th><th>Value</th></tr></thead>"
        "<tbody>"
        + "".join(rows) +
        "</tbody></table>"
    )
    return table_html
