"""
Battery Sizing Analysis Module

This module provides comprehensive battery sizing analysis for:
- Peak demand shaving optimization
- Load profile analysis for battery sizing
- Economic analysis of battery systems
- Battery performance simulation
- ROI calculations for different battery configurations

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Import utility modules
from tariffs.rp4_tariffs import get_tariff_data
from utils.cost_calculator import calculate_cost


def calculate_battery_metrics(power_profile, target_peak_kw, battery_capacity_kwh, battery_power_kw, efficiency=0.9, peak_period_only=True):
    """
    Calculate battery performance metrics for peak shaving during peak periods only
    
    Args:
        power_profile: pandas Series with power consumption (kW) and datetime index
        target_peak_kw: float, target peak demand after shaving (kW)
        battery_capacity_kwh: float, battery energy capacity (kWh)
        battery_power_kw: float, battery power rating (kW)
        efficiency: float, round-trip efficiency (default 0.9)
        peak_period_only: bool, if True, only discharge during peak periods (8AM-10PM weekdays)
    
    Returns:
        dict with battery performance metrics
    """
    # Initialize battery state
    battery_soc = 50  # Start at 50% SOC
    battery_energy = battery_capacity_kwh * 0.5
    
    # Track battery operations
    charging_power = []
    discharging_power = []
    soc_profile = []
    grid_power = []
    peak_shaving_events = []
    peak_period_flags = []
    
    # Convert power_profile to have datetime index if it doesn't already
    if not hasattr(power_profile.index, 'hour'):
        # If no datetime index, assume starting from Monday 8AM for demonstration
        import pandas as pd
        start_time = pd.Timestamp('2024-01-01 00:00:00')  # Start on Monday
        freq = pd.infer_freq(power_profile.index) or '15T'  # Default to 15-min intervals
        power_profile.index = pd.date_range(start=start_time, periods=len(power_profile), freq=freq)
    
    for timestamp, power_demand in power_profile.items():
        # Determine if current time is in peak period (Monday-Friday 8AM-10PM)
        is_peak_period = (timestamp.weekday() < 5 and 8 <= timestamp.hour < 22)
        peak_period_flags.append(is_peak_period)
        
        # Battery discharge logic - only during peak periods if peak_period_only is True
        can_discharge = True
        if peak_period_only:
            can_discharge = is_peak_period
        
        # Determine if we need peak shaving
        if power_demand > target_peak_kw and can_discharge:
            # Need to discharge battery during peak period
            required_discharge = min(
                power_demand - target_peak_kw,  # Required shaving
                battery_power_kw,  # Battery power limit
                battery_energy * efficiency  # Available energy
            )
            
            # Update battery energy
            battery_energy -= required_discharge / efficiency
            battery_soc = (battery_energy / battery_capacity_kwh) * 100
            
            discharging_power.append(required_discharge)
            charging_power.append(0)
            grid_power.append(power_demand - required_discharge)
            peak_shaving_events.append(1)
            
        else:
            # Battery charging logic - prioritize off-peak periods for charging
            available_charging_capacity = battery_capacity_kwh - battery_energy
            max_charging_power = min(battery_power_kw, available_charging_capacity * efficiency)
            
            # Charge during off-peak periods when demand is below target and battery needs charging
            should_charge = (
                not is_peak_period and  # Off-peak period
                available_charging_capacity > battery_capacity_kwh * 0.1 and  # Battery not nearly full
                power_demand < target_peak_kw * 0.8 and  # Low demand period
                battery_soc < 80  # Don't overcharge
            )
            
            if should_charge:
                # Calculate optimal charging power to avoid increasing demand too much
                charging_power_actual = min(
                    max_charging_power,
                    target_peak_kw * 0.2,  # Limit charging to avoid demand spikes
                    (target_peak_kw * 0.9 - power_demand)  # Don't exceed 90% of target
                )
                charging_power_actual = max(0, charging_power_actual)  # Ensure non-negative
                
                battery_energy += charging_power_actual * efficiency
                battery_soc = (battery_energy / battery_capacity_kwh) * 100
                
                charging_power.append(charging_power_actual)
                discharging_power.append(0)
                grid_power.append(power_demand + charging_power_actual)
                peak_shaving_events.append(0)
            else:
                # No battery operation
                charging_power.append(0)
                discharging_power.append(0)
                grid_power.append(power_demand)
                peak_shaving_events.append(0)
        
        soc_profile.append(battery_soc)
        
        # Ensure SOC stays within limits
        battery_soc = max(10, min(90, battery_soc))  # 10-90% SOC limits
        battery_energy = (battery_soc / 100) * battery_capacity_kwh
    
    # Calculate peak period statistics
    peak_period_flags = np.array(peak_period_flags)
    peak_periods_total = np.sum(peak_period_flags)
    discharge_in_peak_periods = np.sum(np.array(discharging_power)[peak_period_flags])
    charge_in_off_peak_periods = np.sum(np.array(charging_power)[~peak_period_flags])
    
    return {
        'charging_power': np.array(charging_power),
        'discharging_power': np.array(discharging_power),
        'soc_profile': np.array(soc_profile),
        'grid_power': np.array(grid_power),
        'peak_shaving_events': np.array(peak_shaving_events),
        'peak_period_flags': peak_period_flags,
        'original_peak': power_profile.max(),
        'achieved_peak': max(grid_power),
        'peak_reduction': power_profile.max() - max(grid_power),
        'total_discharge_energy': sum(discharging_power),
        'total_charge_energy': sum(charging_power),
        'discharge_in_peak_periods': discharge_in_peak_periods,
        'charge_in_off_peak_periods': charge_in_off_peak_periods,
        'peak_periods_total': peak_periods_total,
        'peak_period_utilization': discharge_in_peak_periods / peak_periods_total if peak_periods_total > 0 else 0,
        'cycle_count': sum(peak_shaving_events) / len(power_profile) * 365  # Annualized
    }


def calculate_battery_economics(original_cost, shaved_cost, battery_capex, battery_opex_annual, 
                               battery_replacement_years=10, analysis_years=20, discount_rate=0.06):
    """
    Calculate battery system economics and ROI
    
    Args:
        original_cost: float, annual electricity cost without battery (RM)
        shaved_cost: float, annual electricity cost with battery (RM)
        battery_capex: float, initial battery system cost (RM)
        battery_opex_annual: float, annual O&M cost (RM)
        battery_replacement_years: int, battery replacement cycle (years)
        analysis_years: int, total analysis period (years)
        discount_rate: float, discount rate for NPV calculation
    
    Returns:
        dict with economic metrics
    """
    annual_savings = original_cost - shaved_cost
    
    # Calculate cash flows
    cash_flows = [-battery_capex]  # Initial investment
    
    for year in range(1, analysis_years + 1):
        annual_cash_flow = annual_savings - battery_opex_annual
        
        # Add battery replacement cost
        if year % battery_replacement_years == 0 and year < analysis_years:
            annual_cash_flow -= battery_capex * 0.8  # Assume 20% cost reduction over time
        
        cash_flows.append(annual_cash_flow)
    
    # Calculate NPV
    npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
    
    # Calculate IRR (simplified)
    if annual_savings > 0:
        simple_payback = battery_capex / annual_savings
        irr = (annual_savings / battery_capex) * 100  # Simplified IRR
    else:
        simple_payback = float('inf')
        irr = 0
    
    return {
        'annual_savings': annual_savings,
        'npv': npv,
        'irr': irr,
        'simple_payback': simple_payback,
        'total_opex': battery_opex_annual * analysis_years,
        'total_savings': annual_savings * analysis_years,
        'cash_flows': cash_flows
    }


def analyze_load_profile_for_battery_sizing(power_data):
    """
    Analyze load profile to automatically suggest optimal battery sizing with peak period awareness
    
    Args:
        power_data: pandas Series with power consumption data and datetime index
        
    Returns:
        dict with analysis results and auto-suggestions
    """
    # Ensure datetime index
    if not hasattr(power_data.index, 'hour'):
        import pandas as pd
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        freq = pd.infer_freq(power_data.index) or '15T'
        power_data.index = pd.date_range(start=start_time, periods=len(power_data), freq=freq)
    
    # Basic statistics
    max_demand = power_data.max()
    avg_demand = power_data.mean()
    min_demand = power_data.min()
    load_factor = avg_demand / max_demand
    
    # Peak period analysis (Monday-Friday 8AM-10PM)
    peak_period_mask = power_data.index.to_series().apply(
        lambda ts: ts.weekday() < 5 and 8 <= ts.hour < 22
    )
    
    peak_period_data = power_data[peak_period_mask]
    off_peak_data = power_data[~peak_period_mask]
    
    peak_period_stats = {
        'max_demand': peak_period_data.max() if len(peak_period_data) > 0 else 0,
        'avg_demand': peak_period_data.mean() if len(peak_period_data) > 0 else 0,
        'hours_total': len(peak_period_data),
        'percentage_of_total': len(peak_period_data) / len(power_data) * 100
    }
    
    off_peak_stats = {
        'max_demand': off_peak_data.max() if len(off_peak_data) > 0 else 0,
        'avg_demand': off_peak_data.mean() if len(off_peak_data) > 0 else 0,
        'hours_total': len(off_peak_data),
        'percentage_of_total': len(off_peak_data) / len(power_data) * 100
    }
    
    # Percentile analysis for peak shaving targets
    percentiles = {
        'p99': power_data.quantile(0.99),
        'p95': power_data.quantile(0.95),
        'p90': power_data.quantile(0.90),
        'p85': power_data.quantile(0.85),
        'p80': power_data.quantile(0.80)
    }
    
    # Peak shaving opportunities analysis
    peak_analysis = {}
    for name, threshold in percentiles.items():
        above_threshold = power_data[power_data > threshold]
        # Analyze how much occurs during peak periods
        above_threshold_peak_periods = above_threshold[peak_period_mask[above_threshold.index]]
        
        peak_analysis[name] = {
            'threshold': threshold,
            'reduction_potential': max_demand - threshold,
            'reduction_percentage': (max_demand - threshold) / max_demand * 100,
            'hours_above': len(above_threshold),
            'hours_percentage': len(above_threshold) / len(power_data) * 100,
            'energy_above': above_threshold.sum() - threshold * len(above_threshold),
            'peak_period_hours_above': len(above_threshold_peak_periods),
            'peak_period_percentage': len(above_threshold_peak_periods) / len(above_threshold) * 100 if len(above_threshold) > 0 else 0
        }
    
    # Auto-suggest optimal targets based on analysis
    # Choose target that affects 1-5% of time but significant reduction
    optimal_target = None
    for name, analysis in peak_analysis.items():
        if 1 <= analysis['hours_percentage'] <= 5 and analysis['reduction_percentage'] >= 10:
            optimal_target = name
            break
    
    if not optimal_target:
        optimal_target = 'p90'  # Fallback to 90th percentile
    
    # Battery sizing recommendations
    target_threshold = percentiles[optimal_target]
    reduction_needed = max_demand - target_threshold
    
    # Battery power rating: should handle the reduction + 20% margin
    suggested_power_kw = reduction_needed * 1.2
    
    # Battery capacity: based on duration analysis
    # Estimate how long peaks typically last
    above_target = power_data[power_data > target_threshold]
    if len(above_target) > 0:
        # Estimate typical peak duration
        peak_duration_hours = len(above_target) / len(power_data) * 24 * 0.5  # Conservative estimate
        suggested_capacity_kwh = suggested_power_kw * peak_duration_hours
    else:
        suggested_capacity_kwh = suggested_power_kw * 2  # Default 2-hour capacity
    
    return {
        'max_demand': max_demand,
        'avg_demand': avg_demand,
        'min_demand': min_demand,
        'load_factor': load_factor,
        'peak_period_stats': peak_period_stats,
        'off_peak_stats': off_peak_stats,
        'percentiles': percentiles,
        'peak_analysis': peak_analysis,
        'optimal_target': optimal_target,
        'suggested_power_kw': suggested_power_kw,
        'suggested_capacity_kwh': suggested_capacity_kwh
    }


def analyze_peak_events_for_battery_sizing(power_data):
    """
    Advanced peak event detection for precise battery sizing and MD reduction
    
    Args:
        power_data: pandas Series with power consumption data and datetime index
        
    Returns:
        dict with detailed peak events analysis and battery sizing recommendations
    """
    # Ensure datetime index
    if not hasattr(power_data.index, 'hour'):
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        freq = pd.infer_freq(power_data.index) or '15T'
        power_data.index = pd.date_range(start=start_time, periods=len(power_data), freq=freq)
    
    # Basic statistics
    max_demand = power_data.max()
    avg_demand = power_data.mean()
    min_demand = power_data.min()
    load_factor = avg_demand / max_demand
    
    # Define peak detection thresholds based on percentiles
    peak_thresholds = {
        'top_1_percent': power_data.quantile(0.99),
        'top_5_percent': power_data.quantile(0.95), 
        'top_10_percent': power_data.quantile(0.90),
        'top_20_percent': power_data.quantile(0.80)
    }
    
    # Peak period analysis (Monday-Friday 8AM-10PM for RP4 MD recording)
    peak_period_mask = power_data.index.to_series().apply(
        lambda ts: ts.weekday() < 5 and 8 <= ts.hour < 22
    )
    
    # Detailed peak events analysis
    peak_events_analysis = {}
    all_peak_events = []
    
    for threshold_name, threshold_value in peak_thresholds.items():
        # Find all events above threshold
        above_threshold_mask = power_data >= threshold_value
        
        # Extract individual peak events
        peak_events = []
        in_peak = False
        peak_start = None
        current_event = {}
        
        for i, (timestamp, power_value) in enumerate(power_data.items()):
            is_above = above_threshold_mask.iloc[i]
            is_peak_period = peak_period_mask.iloc[i]
            
            if is_above and not in_peak:
                # Start of peak event
                in_peak = True
                peak_start = i
                current_event = {
                    'start_time': timestamp,
                    'start_index': i,
                    'peak_load': power_value,
                    'excess_energy': power_value - threshold_value,
                    'in_peak_period': is_peak_period,
                    'duration_intervals': 1
                }
                
            elif is_above and in_peak:
                # Continue peak event
                current_event['duration_intervals'] += 1
                current_event['excess_energy'] += power_value - threshold_value
                if power_value > current_event['peak_load']:
                    current_event['peak_load'] = power_value
                # Update peak period status (true if any part is in peak period)
                if is_peak_period:
                    current_event['in_peak_period'] = True
                    
            elif not is_above and in_peak:
                # End of peak event
                in_peak = False
                current_event['end_time'] = timestamp
                current_event['end_index'] = i - 1
                current_event['duration_hours'] = current_event['duration_intervals'] * 0.25  # 15min intervals
                current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
                current_event['md_reduction_potential'] = current_event['peak_load'] - threshold_value
                
                peak_events.append(current_event.copy())
                
                # Add to all events for overall analysis
                current_event['threshold_name'] = threshold_name
                current_event['threshold_value'] = threshold_value
                all_peak_events.append(current_event.copy())
        
        # Handle case where data ends during peak
        if in_peak and current_event:
            current_event['end_time'] = power_data.index[-1]
            current_event['end_index'] = len(power_data) - 1
            current_event['duration_hours'] = current_event['duration_intervals'] * 0.25
            current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
            current_event['md_reduction_potential'] = current_event['peak_load'] - threshold_value
            peak_events.append(current_event.copy())
            
            current_event['threshold_name'] = threshold_name
            current_event['threshold_value'] = threshold_value
            all_peak_events.append(current_event.copy())
        
        # Calculate summary statistics for this threshold
        if peak_events:
            peak_period_events = [e for e in peak_events if e['in_peak_period']]
            off_peak_events = [e for e in peak_events if not e['in_peak_period']]
            
            total_excess_energy = sum(e['excess_energy'] for e in peak_events) * 0.25  # Convert to kWh
            peak_period_excess = sum(e['excess_energy'] for e in peak_period_events) * 0.25
            
            peak_events_analysis[threshold_name] = {
                'threshold_kw': threshold_value,
                'threshold_value': threshold_value,  # Keep for backward compatibility
                'total_events': len(peak_events),
                'peak_events_count': len(peak_events),  # Add consistent naming
                'peak_period_events': len(peak_period_events),
                'off_peak_events': len(off_peak_events),
                'avg_duration_hours': np.mean([e['duration_hours'] for e in peak_events]),
                'max_duration_hours': max([e['duration_hours'] for e in peak_events]),
                'total_excess_energy_kwh': total_excess_energy,
                'peak_period_excess_kwh': peak_period_excess,
                'md_reduction_potential': max_demand - threshold_value,
                'md_reduction_potential_kw': max_demand - threshold_value,  # Add consistent naming
                'md_reduction_percentage': ((max_demand - threshold_value) / max_demand) * 100,
                'percentage_of_time': (len(peak_events) / len(power_data)) * 100,
                'avg_event_excess_power': np.mean([e['avg_excess_power'] for e in peak_events]),
                'max_event_excess_power': max([e['md_reduction_potential'] for e in peak_events]),
                'peak_period_percentage': len(peak_period_events) / len(peak_events) * 100 if peak_events else 0,
                'events_detail': peak_events
            }
        else:
            peak_events_analysis[threshold_name] = {
                'threshold_kw': threshold_value,
                'threshold_value': threshold_value,  # Keep for backward compatibility
                'total_events': 0,
                'peak_events_count': 0,  # Add consistent naming
                'peak_period_events': 0,
                'off_peak_events': 0,
                'avg_duration_hours': 0,
                'max_duration_hours': 0,
                'total_excess_energy_kwh': 0,
                'peak_period_excess_kwh': 0,
                'md_reduction_potential': max_demand - threshold_value,
                'md_reduction_potential_kw': max_demand - threshold_value,  # Add consistent naming
                'md_reduction_percentage': ((max_demand - threshold_value) / max_demand) * 100,
                'percentage_of_time': 0,
                'avg_event_excess_power': 0,
                'max_event_excess_power': 0,
                'peak_period_percentage': 0,
                'events_detail': []
            }
    
    # Battery sizing recommendations based on peak events
    optimal_threshold = None
    for threshold_name, analysis in peak_events_analysis.items():
        # Select threshold with meaningful events (>0) but not too frequent (prefer 5-20 events)
        if 5 <= analysis['total_events'] <= 20 and analysis['peak_period_events'] > 0:
            optimal_threshold = threshold_name
            break
    
    if not optimal_threshold:
        # Fallback: choose threshold with at least some peak period events
        for threshold_name, analysis in peak_events_analysis.items():
            if analysis['peak_period_events'] > 0:
                optimal_threshold = threshold_name
                break
    
    if not optimal_threshold:
        optimal_threshold = 'top_10_percent'  # Final fallback
    
    optimal_analysis = peak_events_analysis[optimal_threshold]
    
    # Calculate battery sizing based on optimal threshold
    suggested_power_kw = optimal_analysis['max_event_excess_power'] * 1.2  # 20% margin
    suggested_capacity_kwh = optimal_analysis['avg_duration_hours'] * suggested_power_kw * 1.5  # 50% buffer
    
    # Create demand duration curve data
    sorted_power = power_data.sort_values(ascending=False)
    duration_percentage = np.linspace(0, 100, len(sorted_power))
    demand_duration_curve = {
        'duration_percentage': duration_percentage,
        'power_values': sorted_power.values
    }
    
    return {
        'max_demand': max_demand,
        'avg_demand': avg_demand,
        'min_demand': min_demand,
        'load_factor': load_factor,
        'peak_thresholds': peak_thresholds,
        'peak_events_analysis': peak_events_analysis,
        'all_peak_events': all_peak_events,
        'optimal_threshold': optimal_threshold,
        'optimal_analysis': optimal_analysis,
        'suggested_power_kw': suggested_power_kw,
        'suggested_capacity_kwh': suggested_capacity_kwh,
        'peak_period_mask': peak_period_mask,
        'demand_duration_curve': demand_duration_curve
    }


def calculate_md_savings_and_roi(peak_reduction_kw, demand_rate_per_kw_month, battery_capex, 
                                analysis_years=10, discount_rate=0.06, maintenance_rate=0.02):
    """
    Calculate Maximum Demand (MD) savings and ROI for battery investment
    
    Args:
        peak_reduction_kw: float, achieved peak demand reduction in kW
        demand_rate_per_kw_month: float, demand charge rate in RM/kW/month
        battery_capex: float, total battery system cost in RM
        analysis_years: int, analysis period in years
        discount_rate: float, discount rate for NPV calculation
        maintenance_rate: float, annual maintenance cost as percentage of CAPEX
    
    Returns:
        dict with MD savings and ROI metrics
    """
    # Calculate annual MD savings
    monthly_md_savings = peak_reduction_kw * demand_rate_per_kw_month
    annual_md_savings = monthly_md_savings * 12
    
    # Calculate maintenance costs
    annual_maintenance_cost = battery_capex * maintenance_rate
    net_annual_savings = annual_md_savings - annual_maintenance_cost
    
    # Calculate simple payback
    simple_payback = battery_capex / net_annual_savings if net_annual_savings > 0 else float('inf')
    
    # Calculate NPV and IRR
    cash_flows = [-battery_capex]  # Initial investment (negative)
    for year in range(1, analysis_years + 1):
        cash_flows.append(net_annual_savings)
    
    # NPV calculation
    npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
    
    # Simplified IRR calculation
    if net_annual_savings > 0:
        irr = (net_annual_savings / battery_capex) * 100
    else:
        irr = 0
    
    # Calculate total lifetime savings
    total_lifetime_savings = net_annual_savings * analysis_years
    total_maintenance_costs = annual_maintenance_cost * analysis_years
    
    return {
        'monthly_md_savings': monthly_md_savings,
        'annual_md_savings': annual_md_savings,
        'annual_maintenance_cost': annual_maintenance_cost,
        'net_annual_savings': net_annual_savings,
        'simple_payback_years': simple_payback,
        'npv': npv,
        'irr_percentage': irr,
        'total_lifetime_savings': total_lifetime_savings,
        'total_maintenance_costs': total_maintenance_costs,
        'roi_percentage': (total_lifetime_savings / battery_capex) * 100 if battery_capex > 0 else 0
    }


def create_peak_events_visualization_data(power_data, threshold_value):
    """
    Create visualization data for peak events analysis
    
    Args:
        power_data: pandas Series with power consumption data and datetime index
        threshold_value: float, threshold for defining peak events
        
    Returns:
        dict with visualization data
    """
    # Ensure datetime index
    if not hasattr(power_data.index, 'hour'):
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        freq = pd.infer_freq(power_data.index) or '15T'
        power_data.index = pd.date_range(start=start_time, periods=len(power_data), freq=freq)
    
    # Create peak events mask
    peak_events_mask = power_data >= threshold_value
    
    # Peak period mask (weekdays 8AM-10PM)
    peak_period_mask = power_data.index.to_series().apply(
        lambda ts: ts.weekday() < 5 and 8 <= ts.hour < 22
    )
    
    # Extract peak events with start/end times
    peak_events = []
    in_peak = False
    peak_start = None
    
    for i, (timestamp, power_value) in enumerate(power_data.items()):
        is_above = peak_events_mask.iloc[i]
        is_peak_period = peak_period_mask.iloc[i]
        
        if is_above and not in_peak:
            # Start of peak event
            in_peak = True
            peak_start = timestamp
            current_event = {
                'start_time': timestamp,
                'start_index': i,
                'peak_load': power_value,
                'excess_energy': power_value - threshold_value,
                'in_peak_period': is_peak_period,
                'duration_intervals': 1
            }
            
        elif is_above and in_peak:
            # Continue peak event
            current_event['duration_intervals'] += 1
            current_event['excess_energy'] += power_value - threshold_value
            if power_value > current_event['peak_load']:
                current_event['peak_load'] = power_value
            # Update peak period status
            if is_peak_period:
                current_event['in_peak_period'] = True
                
        elif not is_above and in_peak:
            # End of peak event
            in_peak = False
            current_event['end_time'] = timestamp
            current_event['end_index'] = i - 1
            current_event['duration_hours'] = current_event['duration_intervals'] * 0.25
            current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
            current_event['md_reduction_potential'] = current_event['peak_load'] - threshold_value
            
            peak_events.append(current_event.copy())
    
    # Handle case where data ends during peak
    if in_peak and 'current_event' in locals():
        current_event['end_time'] = power_data.index[-1]
        current_event['end_index'] = len(power_data) - 1
        current_event['duration_hours'] = current_event['duration_intervals'] * 0.25
        current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
        current_event['md_reduction_potential'] = current_event['peak_load'] - threshold_value
        peak_events.append(current_event.copy())
    
    # Create time series data with peak event flags
    viz_data = pd.DataFrame({
        'timestamp': power_data.index,
        'power': power_data.values,
        'is_peak_event': peak_events_mask.values,
        'is_peak_period': peak_period_mask.values,
        'excess_power': np.maximum(0, power_data.values - threshold_value)
    })
    
    # Add hour and day of week for heatmap
    viz_data['hour'] = viz_data['timestamp'].dt.hour
    viz_data['day_of_week'] = viz_data['timestamp'].dt.day_name()
    
    return {
        'peak_events': peak_events,
        'viz_data': viz_data,
        'threshold_value': threshold_value,
        'total_events': len(peak_events),
        'peak_period_events': sum(1 for evt in peak_events if evt['in_peak_period']),
        'total_excess_energy': sum(evt['excess_energy'] for evt in peak_events) * 0.25  # Convert to kWh
    }


def analyze_demand_curve_and_peak_events(power_data):
    """
    Analyze demand curve and identify peak events for battery sizing
    This function provides the same interface as analyze_peak_events_for_battery_sizing
    but with a different name for backward compatibility.
    
    Args:
        power_data: pandas Series with power consumption data and datetime index
        
    Returns:
        dict with detailed peak events analysis and battery sizing recommendations
    """
    return analyze_peak_events_for_battery_sizing(power_data)


def battery_sizing_analysis_page():
    """
    Enhanced Battery Sizing Analysis Page following optimal engineering flow:
    1. Data Upload & Tariff Selection → Determine MD cost structure
    2. Peak Demand Analysis → Identify peak shaving opportunities  
    3. Peak Events Analysis → Calculate energy requirements for MD reduction
    4. Battery Sizing Recommendations → Match battery capacity to peak shaving needs
    5. Cost Estimation → Calculate battery system costs
    6. ROI Analysis → Compare savings vs investment with payback periods
    """
    st.title("🔋 Battery Sizing Analysis")
    st.markdown("""
    **Complete Battery Sizing Workflow for Maximum Demand (MD) Cost Savings**
    
    Follow the systematic approach below to optimize your battery investment:
    
    1. 📊 **Data Analysis** → Upload energy data and select tariff type for MD cost determination
    2. 🎯 **Peak Opportunities** → Identify peak demand events and shaving potential
    3. 🔋 **Battery Sizing** → Calculate required kWh capacity for optimal MD reduction
    4. 💰 **Investment Analysis** → Estimate costs and calculate ROI with payback periods
    """)
    
    # Check if data is available from main upload tab or allow direct upload
    if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
        st.warning("⚠️ **Step 1:** Please upload your energy data in the 'Data Upload' tab first.")
        st.markdown("---")
        st.subheader("🔄 Alternative: Upload Data Directly Here")
        
        # Allow direct upload as backup
        uploaded_file_direct = st.file_uploader(
            "Upload your energy consumption data", 
            type=["xlsx", "csv"], 
            help="Upload Excel or CSV file with timestamps and power consumption data",
            key="battery_direct_upload"
        )
        
        if uploaded_file_direct:
            try:
                # Read the uploaded file
                if uploaded_file_direct.name.endswith('.csv'):
                    df_direct = pd.read_csv(uploaded_file_direct)
                else:
                    df_direct = pd.read_excel(uploaded_file_direct)
                
                st.success(f"✅ File uploaded successfully! Shape: {df_direct.shape}")
                
                # Display data preview
                with st.expander("📊 Data Preview", expanded=False):
                    st.dataframe(df_direct.head(10), use_container_width=True)
                    st.write("**Column Names:**", list(df_direct.columns))
                
                # Column selection
                st.subheader("🎯 Column Selection")
                col1, col2 = st.columns(2)
                
                with col1:
                    timestamp_col = st.selectbox(
                        "Select Timestamp Column",
                        df_direct.columns,
                        help="Column containing timestamp information",
                        key="battery_timestamp_col"
                    )
                
                with col2:
                    power_col = st.selectbox(
                        "Select Power Column (kW)",
                        [col for col in df_direct.columns if col != timestamp_col],
                        help="Column containing power consumption data in kW",
                        key="battery_power_col"
                    )
                
                if timestamp_col and power_col:
                    # Process data
                    df_direct['Parsed Timestamp'] = pd.to_datetime(df_direct[timestamp_col])
                    df_direct = df_direct.set_index('Parsed Timestamp').sort_index()
                    df_direct = df_direct.dropna(subset=[power_col])
                    
                    # Convert power to numeric
                    df_direct[power_col] = pd.to_numeric(df_direct[power_col], errors='coerce')
                    df_direct = df_direct.dropna(subset=[power_col])
                    
                    if len(df_direct) > 0:
                        # Store in session state
                        st.session_state['processed_df'] = df_direct
                        st.session_state['power_column'] = power_col
                        st.session_state['uploaded_file'] = uploaded_file_direct
                        
                        st.success("✅ Data processed successfully! You can now proceed with the analysis below.")
                    else:
                        st.error("❌ No valid data found after processing.")
                        return
                else:
                    st.info("👆 Please select both timestamp and power columns to proceed.")
                    return
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your file has timestamp and power consumption columns.")
                return
        else:
            return
    
    # Get uploaded data from session state
    df = st.session_state.get('processed_df')
    power_col = st.session_state.get('power_column')
    
    if df is None or power_col is None:
        st.error("❌ No processed data found. Please process your data in the 'Data Upload' tab first.")
        return
    
    if power_col not in df.columns:
        st.error(f"❌ Power column '{power_col}' not found in data.")
        return
    
    # ===================================================================
    # STEP 1: DATA ANALYSIS & TARIFF SELECTION FOR MD COST STRUCTURE
    # ===================================================================
    st.header("📊 Step 1: Data Analysis & MD Cost Structure")
    st.markdown("*Analyze your load profile and determine Maximum Demand (MD) costs based on tariff type*")
    
    # Display data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{len(df):,}")
    with col2:
        st.metric("Peak Demand", f"{df[power_col].max():.1f} kW")
    with col3:
        st.metric("Average Demand", f"{df[power_col].mean():.1f} kW")
    with col4:
        data_days = len(df) / 96  # Assuming 15-min intervals
        st.metric("Data Period", f"{data_days:.1f} days")
    
    # RP4 Tariff Selection for MD Cost Determination
    st.subheader("💰 RP4 Tariff Selection for MD Cost Analysis")
    st.info("ℹ️ **RP4 Demand Charges:** Under RP4 tariffs, demand charges consist of Capacity Rate + Network Rate (both charged per kW of maximum demand). Battery peak shaving reduces these charges.")
    
    # Get RP4 tariff data
    rp4_data = get_tariff_data()
    business_tariffs = rp4_data["Business"]["Tariff Groups"]["Non Domestic"]["Tariffs"]
    
    col1, col2 = st.columns(2)
    with col1:
        # Create tariff options from RP4 data
        tariff_options = {}
        for tariff in business_tariffs:
            tariff_name = tariff["Tariff"]
            voltage = tariff.get("Voltage", "")
            tariff_type_desc = tariff.get("Type", "")
            display_name = f"{tariff_name} ({voltage}, {tariff_type_desc})"
            tariff_options[display_name] = tariff
        
        selected_tariff_name = st.selectbox(
            "Select Your RP4 Tariff",
            options=list(tariff_options.keys()),
            index=2,  # Default to Medium Voltage General
            help="Choose your actual RP4 tariff for accurate MD cost calculation"
        )
        
        selected_tariff = tariff_options[selected_tariff_name]
        
        # Calculate total demand rate (Capacity + Network)
        capacity_rate = selected_tariff["Rates"].get("Capacity Rate", 0)
        network_rate = selected_tariff["Rates"].get("Network Rate", 0)
        
        # Check if rates are per kW or per kWh
        rules = selected_tariff.get("Rules", {})
        capacity_by = rules.get("charge_capacity_by", "")
        network_by = rules.get("charge_network_by", "")
        
        if "kW" in capacity_by and "kW" in network_by:
            # Both are demand charges per kW
            total_demand_rate = capacity_rate + network_rate
            st.info(f"**Capacity Rate:** RM {capacity_rate:.2f}/kW/month")
            st.info(f"**Network Rate:** RM {network_rate:.2f}/kW/month")
        elif "kW" in capacity_by:
            # Only capacity is demand charge
            total_demand_rate = capacity_rate
            st.info(f"**Capacity Rate:** RM {capacity_rate:.2f}/kW/month")
            st.info(f"**Network Rate:** RM {network_rate:.4f}/kWh (energy-based)")
        else:
            # Neither are demand charges (Low Voltage case)
            total_demand_rate = 0
            st.warning("⚠️ This tariff has no demand charges - battery for peak shaving may not provide MD savings")
        
        # Allow manual override
        use_custom_rate = st.checkbox("Override MD Rate", help="Use custom demand rate if different from tariff")
        if use_custom_rate:
            demand_rate = st.number_input(
                "Custom MD Rate (RM/kW/month)",
                min_value=0.0,
                max_value=100.0,
                value=total_demand_rate,
                step=1.0,
                help="Your actual Maximum Demand charge rate per kW per month"
            )
        else:
            demand_rate = total_demand_rate
    
    with col2:
        # Current monthly MD cost based on selected RP4 tariff
        current_peak = df[power_col].max()
        current_monthly_md_cost = current_peak * demand_rate
        current_annual_md_cost = current_monthly_md_cost * 12
        
        st.metric("Current Peak Demand", f"{current_peak:.1f} kW")
        
        if demand_rate > 0:
            st.metric("Monthly MD Cost", f"RM {current_monthly_md_cost:,.0f}")
            st.metric("Annual MD Cost", f"RM {current_annual_md_cost:,.0f}")
        else:
            st.metric("Monthly MD Cost", "RM 0 (No demand charges)")
            st.metric("Annual MD Cost", "RM 0 (No demand charges)")
        
        # Show tariff details
        with st.expander("📋 Selected RP4 Tariff Details", expanded=False):
            st.write(f"**Tariff:** {selected_tariff_name}")
            st.write(f"**Type:** {selected_tariff.get('Type', 'N/A')}")
            st.write(f"**Voltage Level:** {selected_tariff.get('Voltage', 'N/A')}")
            
            rates = selected_tariff["Rates"]
            if "Peak Rate" in rates:
                st.write(f"**Peak Energy Rate:** RM {rates['Peak Rate']:.4f}/kWh")
                st.write(f"**Off-Peak Energy Rate:** RM {rates['OffPeak Rate']:.4f}/kWh")
            else:
                st.write(f"**Energy Rate:** RM {rates.get('Energy Rate', 0):.4f}/kWh")
            
            if demand_rate > 0:
                st.write(f"**Total Demand Rate:** RM {demand_rate:.2f}/kW/month")
            
            st.write(f"**Retail Rate:** RM {rates.get('Retail Rate', 0):.2f}/month")
    
    # ===================================================================
    # STEP 2: PEAK DEMAND ANALYSIS & SHAVING OPPORTUNITIES
    # ===================================================================
    st.header("🎯 Step 2: Peak Demand Analysis & Shaving Opportunities")
    st.markdown("*Identify peak events and quantify reduction potential for MD savings*")
    
    # Check if tariff has demand charges
    if demand_rate <= 0:
        st.error("❌ **No Demand Charges Detected** - The selected RP4 tariff has no Maximum Demand charges. Battery peak shaving will not provide MD cost savings on this tariff.")
        st.info("💡 **Alternative Benefits:** Consider battery for other applications like:")
        st.markdown("""
        - Energy arbitrage (if TOU rates available)
        - Power quality improvement
        - Backup power during outages
        - Future-proofing for tariff changes
        """)
        st.stop()
    
    with st.spinner("Analyzing demand curve and peak events..."):
        demand_analysis = analyze_demand_curve_and_peak_events(df[power_col])
    
    # Display peak shaving opportunities
    st.subheader("⚡ Peak Shaving Opportunities")
    
    # Create opportunities table with MD savings
    opportunities_data = []
    for key, analysis in demand_analysis['peak_events_analysis'].items():
        md_reduction_kw = analysis['md_reduction_potential_kw']
        monthly_savings = md_reduction_kw * demand_rate
        annual_savings = monthly_savings * 12
        
        # Special formatting for 10% peak shaving option
        if key == '10_percent_peak_shaving':
            target_name = '🎯 10% Peak Shaving (Recommended)'
            feasibility = '🟢 Optimal'
        else:
            target_name = key.replace('_', ' ').title()
            feasibility = '🟢 Excellent' if analysis['percentage_of_time'] < 1 and analysis['md_reduction_percentage'] > 20 \
                         else '🟡 Good' if analysis['percentage_of_time'] < 5 and analysis['md_reduction_percentage'] > 10 \
                         else '🔴 Challenging'
        
        # Calculate max kWh to be shaved for battery sizing
        # This represents the maximum energy needed to handle the worst single peak event
        max_event_duration_hours = analysis.get('max_duration_hours', 0.25)
        max_event_excess_power_kw = analysis.get('max_event_excess_power', 0)
        max_kwh_to_shave = max_event_duration_hours * max_event_excess_power_kw
        
        # If detailed events are available, calculate more accurately
        events_detail = analysis.get('events_detail', [])
        if events_detail:
            # Find the event that requires maximum energy to shave
            max_event_energy = max([evt['excess_energy'] * 0.25 for evt in events_detail], default=0)  # Convert to kWh
            max_kwh_to_shave = max(max_kwh_to_shave, max_event_energy)
        
        opportunities_data.append({
            'Target': target_name,
            'Threshold (kW)': f"{analysis['threshold_kw']:.1f}",
            'MD Reduction (kW)': f"{md_reduction_kw:.1f}",
            'Max kWh to Shave': f"{max_kwh_to_shave:.1f}",
            'Monthly MD Savings': f"RM {monthly_savings:,.0f}",
            'Annual MD Savings': f"RM {annual_savings:,.0f}",
            'Peak Events': f"{analysis['peak_events_count']}",
            'Avg Duration (hrs)': f"{analysis.get('avg_duration_hours', 0.25):.1f}",
            'Time Above (%)': f"{analysis['percentage_of_time']:.2f}%",
            'Feasibility': feasibility
        })
    
    df_opportunities = pd.DataFrame(opportunities_data)
    st.dataframe(df_opportunities, use_container_width=True)
    
    # Battery Sizing Logic Explanation
    st.success("""
    **🔋 Battery Sizing Strategy - "Size for the Worst Case":**
    
    The **"Max kWh to Shave"** column shows the maximum energy needed during the worst single peak event. This is your **minimum battery capacity requirement** because:
    
    **✅ Size for Maximum Event**: If you size your battery to handle the worst-case event (e.g., 2,400 kWh), it can automatically handle ALL smaller events through strategic charging/discharging.
    
    **✅ One Size Fits All**: A battery sized for the largest peak event will have sufficient capacity to shave all other peak events in your data.
    
    **✅ Safety Factor**: Add 10-20% safety margin to account for battery efficiency, degradation, and unexpected events.
    """)
    
    # Find the maximum kWh requirement across all thresholds for sizing recommendation
    max_kwh_across_all = 0
    recommended_threshold = ""
    
    for data_row in opportunities_data:
        kwh_value = float(data_row['Max kWh to Shave'].replace(',', ''))
        if kwh_value > max_kwh_across_all:
            max_kwh_across_all = kwh_value
            recommended_threshold = data_row['Target']
    
    if max_kwh_across_all > 0:
        safety_factor_10 = max_kwh_across_all * 1.1
        safety_factor_20 = max_kwh_across_all * 1.2
        
        st.info(f"""
        **🎯 Battery Sizing Recommendation Based on Your Data:**
        
        • **Worst-Case Event**: {max_kwh_across_all:.1f} kWh (from "{recommended_threshold}")
        • **Recommended Battery Capacity**: {safety_factor_10:.1f} - {safety_factor_20:.1f} kWh (with 10-20% safety factor)
        • **Logic**: This capacity will handle ALL peak events in your dataset through strategic operation
        
        *Example: If worst event needs 2,400 kWh → Size battery at 2,640-2,880 kWh*
        """)
    
    with st.expander("🔬 Technical Details: Why This Approach Works", expanded=False):
        st.markdown("""
        **Strategic Battery Operation for Peak Shaving:**
        
        1. **Charge During Low Demand**: Battery charges during off-peak hours when demand is low
        2. **Discharge During Peaks**: Battery discharges during peak events to reduce grid demand
        3. **Automatic Scaling**: Smaller peak events require less energy → battery handles them easily
        4. **Cycle Management**: Battery can cycle multiple times per day to handle multiple peak events
        
        **Example Scenario:**
        - Worst peak event: 2,400 kWh needed over 4 hours
        - Battery sized: 2,880 kWh (with 20% safety factor)
        - Smaller peak event: 800 kWh needed over 2 hours
        - Result: Same battery easily handles the smaller event (only uses 28% of capacity)
        
        **Key Insight:** You don't need different battery sizes for different events - one properly sized battery handles them all!
        """)
    
    # ===================================================================
    # MD SHAVING PERCENTAGE SELECTOR
    # ===================================================================
    st.subheader("🎚️ Select Peak Shaving Target")
    st.markdown("*Choose the percentage of peak demand you want to shave during peak periods*")
    
    # Get current maximum demand
    current_max_demand = df[power_col].max()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        shaving_percentage = st.selectbox(
            "Peak Demand Reduction Target (%)",
            options=[5, 10, 15, 20, 25, 30],
            index=1,  # Default to 10%
            help="Select percentage of peak demand to shave during peak periods"
        )
        
        # Calculate target demand based on percentage
        target_demand_kw = current_max_demand * (1 - shaving_percentage/100)
        st.info(f"**Target:** Reduce peak from {current_max_demand:.1f} kW to {target_demand_kw:.1f} kW ({shaving_percentage}% reduction)")
    
    with col2:
        # Show potential MD savings
        md_reduction_kw = current_max_demand - target_demand_kw
        potential_monthly_savings = md_reduction_kw * demand_rate
        
        st.metric(
            "MD Rate (Capacity + Network)",
            f"RM {demand_rate:.2f}/kW",
            help=f"Total demand rate per kW per month"
        )
        st.success(f"**Potential Monthly Savings: RM {potential_monthly_savings:.2f}**")
    
    # Recalculate peak events analysis with the selected target
    with st.spinner(f"Analyzing peak events for {shaving_percentage}% reduction target..."):
        peak_events_analysis = analyze_peak_events_with_custom_target(df[power_col], target_demand_kw)
    
    # Display analysis results for selected target
    st.subheader(f"📊 Peak Events Analysis for {shaving_percentage}% Reduction")
    
    custom_analysis = peak_events_analysis['custom_target_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Peak Events", custom_analysis['total_events'])
    with col2:
        st.metric("Peak Period Events", custom_analysis['peak_period_events'])
    with col3:
        peak_period_pct = custom_analysis['peak_period_percentage']
        st.metric("% in Peak Period", f"{peak_period_pct:.1f}%")
    with col4:
        avg_duration = custom_analysis['avg_duration_hours']
        st.metric("Avg Duration", f"{avg_duration:.1f} hrs")
    
    # Effectiveness assessment
    if peak_period_pct > 80:
        st.success("✅ **Excellent!** Most peak events occur during RP4 peak periods - battery will be highly effective for MD reduction")
    elif peak_period_pct > 60:
        st.info("💡 **Good potential** - Many peak events occur during RP4 peak periods")
    else:
        st.warning("⚠️ **Limited effectiveness** - Many peak events occur outside RP4 peak periods")
    
    # Detailed Peak Events Table with Filtering
    st.subheader("📋 Detailed Peak Events Table")
    
    # Get the detailed events from the analysis
    events_detail = peak_events_analysis['peak_events_above_target']
    
    if events_detail:
        # Add filtering options similar to advanced energy analysis
        st.markdown("#### Peak Event Filtering")
        event_filter = st.radio(
            "Select which events to display:",
            options=["All", "Peak Period Only", "Off-Peak Period Only"],
            index=0,
            horizontal=True,
            key="battery_event_filter_radio",
            help="Filter events based on when they occur relative to RP4 MD peak hours (8 AM-10 PM, weekdays)"
        )
        
        # Filter events based on selection
        if event_filter == "Peak Period Only":
            filtered_events = [e for e in events_detail if e['in_peak_period']]
        elif event_filter == "Off-Peak Period Only":
            filtered_events = [e for e in events_detail if not e['in_peak_period']]
        else:
            filtered_events = events_detail
        
        if filtered_events:
            st.markdown(f"**Showing {len(filtered_events)} of {len(events_detail)} total events ({event_filter})**")
            
            # Create detailed events dataframe for display
            events_display_data = []
            for i, event in enumerate(filtered_events):
                events_display_data.append({
                    'Event #': i + 1,
                    'Start Date': event['start_time'].strftime('%Y-%m-%d'),
                    'Start Time': event['start_time'].strftime('%H:%M'),
                    'Duration (hrs)': f"{event['duration_hours']:.2f}",
                    'Peak Load (kW)': f"{event['peak_load']:.1f}",
                    'Excess Power (kW)': f"{event['md_reduction_potential']:.1f}",
                    'Energy to Shave (kWh)': f"{event['excess_energy'] * 0.25:.2f}",
                    'Period Type': "Peak Period" if event['in_peak_period'] else "Off-Peak",
                    'MD Impact': "High" if event['in_peak_period'] else "Low"
                })
            
            df_events_display = pd.DataFrame(events_display_data)
            
            # Display the table with styling
            st.dataframe(
                df_events_display.style.apply(
                    lambda row: ['background-color: rgba(255, 0, 0, 0.1)' if row['Period Type'] == 'Peak Period' 
                                else 'background-color: rgba(0, 128, 0, 0.1)' for _ in row], 
                    axis=1
                ),
                use_container_width=True,
                height=400
            )
            
            # Summary statistics for filtered events
            total_energy_filtered = sum(e['excess_energy'] * 0.25 for e in filtered_events)
            avg_duration_filtered = sum(e['duration_hours'] for e in filtered_events) / len(filtered_events)
            max_excess_power_filtered = max(e['md_reduction_potential'] for e in filtered_events)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filtered Events", len(filtered_events))
            with col2:
                st.metric("Total Energy (kWh)", f"{total_energy_filtered:.1f}")
            with col3:
                st.metric("Avg Duration (hrs)", f"{avg_duration_filtered:.1f}")
            with col4:
                st.metric("Max Excess Power (kW)", f"{max_excess_power_filtered:.1f}")
            
            # Explanation for the table columns
            st.info("""
            **Table Column Explanations:**
            - **Energy to Shave (kWh)**: Energy above target threshold for each individual event
            - **Period Type**: Whether event occurs during MD peak hours (8 AM-10 PM, weekdays) 
            - **MD Impact**: High for peak period events (affects MD charges), Low for off-peak events
            - **Background Color**: Red for peak period events, Green for off-peak events
            """)
            
        else:
            st.warning(f"No events found for '{event_filter}' filter.")
    else:
        st.info("No peak events detected above the selected threshold.")

    # Enhanced Peak Events Visualizations
    st.subheader("📊 Peak Events Analysis & Visualization")
    
    # Create multiple visualization tabs
    viz_tabs = st.tabs([
        "🔍 Peak Events Timeline", 
        "📅 Peak Events Heatmap", 
        "📈 Event Characteristics",
        "🔋 Battery Operation Simulation"
    ])
    
    with viz_tabs[0]:
        st.markdown("**Time Series with Peak Events Highlighted**")
        st.caption("Visualize when peak events occur and their magnitude over time")
        
        # Time period selector for better performance
        data_days = len(df) / 96  # Assuming 15-min intervals
        if data_days > 30:
            st.info("💡 Large dataset detected. Select a time period for detailed view:")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=df.index.min().date(),
                    min_value=df.index.min().date(),
                    max_value=df.index.max().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=min(df.index.min().date() + pd.Timedelta(days=14), df.index.max().date()),
                    min_value=df.index.min().date(),
                    max_value=df.index.max().date()
                )
            
            # Filter data for selected period
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            df_viz = df.loc[mask]
        else:
            df_viz = df
            
        if len(df_viz) > 0:
            # Use the custom target as the threshold for visualization
            threshold_value = target_demand_kw
            
            # Create timeline visualization
            fig_timeline = go.Figure()
            
            # Base power consumption
            fig_timeline.add_trace(go.Scatter(
                x=df_viz.index,
                y=df_viz[power_col],
                mode='lines',
                name='Power Consumption',
                line=dict(color='#2E86C1', width=1),
                hovertemplate='Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
            ))
            
            # Highlight peak events
            peak_mask = df_viz[power_col] >= threshold_value
            if peak_mask.any():
                fig_timeline.add_trace(go.Scatter(
                    x=df_viz.index[peak_mask],
                    y=df_viz[power_col][peak_mask],
                    mode='markers',
                    name=f'Peak Events (>{threshold_value:.1f} kW)',
                    marker=dict(color='#FF6B6B', size=4),
                    hovertemplate='Peak Event<br>Time: %{x}<br>Power: %{y:.1f} kW<extra></extra>'
                ))
            
            # Add threshold line
            fig_timeline.add_hline(
                y=threshold_value,
                line_dash="dash",
                line_color='#FF8E53',
                line_width=2,
                annotation_text=f"Threshold: {threshold_value:.1f} kW"
            )
            
            # Add peak period shading (weekdays 8AM-10PM)
            for date in pd.date_range(df_viz.index.min().date(), df_viz.index.max().date(), freq='D'):
                if date.weekday() < 5:  # Weekdays only
                    peak_start = pd.Timestamp(f"{date} 08:00:00")
                    peak_end = pd.Timestamp(f"{date} 22:00:00")
                    
                    fig_timeline.add_vrect(
                        x0=peak_start, x1=peak_end,
                        fillcolor="rgba(255, 235, 59, 0.2)",
                        layer="below",
                        line_width=0,
                    )
            
            fig_timeline.update_layout(
                title=f"Power Consumption Timeline with {shaving_percentage}% Peak Shaving Target ({threshold_value:.1f} kW)",
                xaxis_title="Time",
                yaxis_title="Power Demand (kW)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Peak events summary for selected period
            events_in_period = sum(df_viz[power_col] >= threshold_value)
            peak_period_events = sum((df_viz[power_col] >= threshold_value) & 
                                   (df_viz.index.to_series().apply(lambda x: x.weekday() < 5 and 8 <= x.hour < 22)))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Events in Period", events_in_period)
            with col2:
                st.metric("Peak Period Events", peak_period_events)
            with col3:
                pct_in_peak_period = (peak_period_events / events_in_period * 100) if events_in_period > 0 else 0
                st.metric("% in Peak Period", f"{pct_in_peak_period:.1f}%")
    
    with viz_tabs[1]:
        st.markdown("**Peak Events Heatmap by Time of Day and Day of Week**")
        st.caption("Understand peak event patterns to optimize battery charging/discharging schedule")
        
        # Use the custom target threshold for heatmap
        threshold_value_hm = target_demand_kw
        
        # Create heatmap data
        df_heatmap = df.copy()
        df_heatmap['hour'] = df_heatmap.index.hour
        df_heatmap['day_of_week'] = df_heatmap.index.day_name()
        df_heatmap['is_peak_event'] = df_heatmap[power_col] >= threshold_value_hm
        
        # Aggregate by hour and day of week
        heatmap_data = df_heatmap.groupby(['day_of_week', 'hour'])['is_peak_event'].agg(['sum', 'count']).reset_index()
        heatmap_data['peak_event_percentage'] = (heatmap_data['sum'] / heatmap_data['count']) * 100
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='peak_event_percentage').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Reds',
            text=heatmap_pivot.values,
            texttemplate="%{text:.1f}%",
            textfont={"size": 8},
            hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Peak Events: %{z:.1f}%<extra></extra>'
        ))
        
        # Add peak period overlay (weekdays 8AM-10PM)
        for day_idx, day in enumerate(day_order):
            if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                # Add rectangle for peak period
                fig_heatmap.add_shape(
                    type="rect",
                    x0=7.5, y0=day_idx-0.4,
                    x1=21.5, y1=day_idx+0.4,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0,0,255,0)",
                )
        
        fig_heatmap.update_layout(
            title=f"Peak Events Heatmap - {shaving_percentage}% Reduction Target ({threshold_value_hm:.1f} kW)<br><sub>Blue rectangles show RP4 peak periods (Weekdays 8AM-10PM)</sub>",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Peak period insights
        peak_period_mask = (df.index.to_series().apply(lambda x: x.weekday() < 5 and 8 <= x.hour < 22))
        peak_events_peak_period = sum((df[power_col] >= threshold_value_hm) & peak_period_mask)
        peak_events_off_peak = sum((df[power_col] >= threshold_value_hm) & ~peak_period_mask)
        total_peak_events = peak_events_peak_period + peak_events_off_peak
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Events in Peak Period", peak_events_peak_period)
        with col2:
            st.metric("Peak Events in Off-Peak", peak_events_off_peak)
        with col3:
            optimal_pct = (peak_events_peak_period / total_peak_events * 100) if total_peak_events > 0 else 0
            st.metric("% Targeting MD Periods", f"{optimal_pct:.1f}%")
            
        if optimal_pct > 80:
            st.success("✅ **Excellent!** Most peak events occur during RP4 peak periods - battery will be highly effective for MD reduction")
        elif optimal_pct > 60:
            st.info("💡 **Good potential** - Many peak events occur during RP4 peak periods")
        else:
            st.warning("⚠️ **Limited effectiveness** - Many peak events occur outside RP4 peak periods")
    
    with viz_tabs[2]:
        st.markdown("**Peak Event Characteristics Analysis**")
        st.caption("Analyze duration, magnitude, and frequency of peak events for optimal battery sizing")
        
        # Use the custom target analysis
        events_detail = peak_events_analysis['peak_events_above_target']
        
        if events_detail:
            events_df = pd.DataFrame(events_detail)
            
            # Create subplots for event characteristics
            fig_chars = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Event Duration Distribution',
                    'Event Magnitude vs Duration', 
                    'Excess Energy Distribution',
                    'Events by Time of Day'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Duration histogram
            fig_chars.add_trace(
                go.Histogram(
                    x=events_df['duration_hours'],
                    name='Duration',
                    nbinsx=20,
                    marker_color='#2E86C1'
                ),
                row=1, col=1
            )
            
            # Magnitude vs Duration scatter
            fig_chars.add_trace(
                go.Scatter(
                    x=events_df['duration_hours'],
                    y=events_df['md_reduction_potential'],
                    mode='markers',
                    name='Events',
                    marker=dict(
                        size=8,
                        color=events_df['excess_energy'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Excess Energy")
                    ),
                    text=[f"Start: {evt['start_time'].strftime('%Y-%m-%d %H:%M')}" for evt in events_detail],
                    hovertemplate='Duration: %{x:.1f} hrs<br>MD Reduction: %{y:.1f} kW<br>%{text}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Excess energy histogram
            fig_chars.add_trace(
                go.Histogram(
                    x=events_df['excess_energy'],
                    name='Excess Energy',
                    nbinsx=20,
                    marker_color='#FF6B6B'
                ),
                row=2, col=1
            )
            
            # Events by hour
            events_df['hour'] = pd.to_datetime(events_df['start_time']).dt.hour
            hourly_events = events_df.groupby('hour').size()
            
            fig_chars.add_trace(
                go.Bar(
                    x=hourly_events.index,
                    y=hourly_events.values,
                    name='Events Count',
                    marker_color='#FF8E53'
                ),
                row=2, col=2
            )
            
            # Add peak period shading to hourly chart
            fig_chars.add_vrect(
                x0=8, x1=22,
                fillcolor="rgba(255, 235, 59, 0.3)",
                layer="below",
                line_width=0,
                row=2, col=2
            )
            
            fig_chars.update_xaxes(title_text="Duration (hours)", row=1, col=1)
            fig_chars.update_yaxes(title_text="Count", row=1, col=1)
            fig_chars.update_xaxes(title_text="Duration (hours)", row=1, col=2)
            fig_chars.update_yaxes(title_text="MD Reduction (kW)", row=1, col=2)
            fig_chars.update_xaxes(title_text="Excess Energy (kWh)", row=2, col=1)
            fig_chars.update_yaxes(title_text="Count", row=2, col=1)
            fig_chars.update_xaxes(title_text="Hour of Day", row=2, col=2)
            fig_chars.update_yaxes(title_text="Events Count", row=2, col=2)
            
            fig_chars.update_layout(
                height=700,
                title_text=f"Peak Event Characteristics Analysis ({shaving_percentage}% Reduction Target: {target_demand_kw:.1f} kW)",
                showlegend=False
            )
            
            st.plotly_chart(fig_chars, use_container_width=True)
            
            # Key insights
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_duration = events_df['duration_hours'].mean()
                st.metric("Avg Duration", f"{avg_duration:.1f} hours")
            with col2:
                max_reduction = events_df['md_reduction_potential'].max()
                st.metric("Max MD Reduction", f"{max_reduction:.1f} kW")
            with col3:
                total_energy = events_df['excess_energy'].sum() * 0.25  # Convert to kWh
                st.metric("Total Excess Energy", f"{total_energy:.0f} kWh")
            with col4:
                peak_period_events = sum(events_df['in_peak_period'])
                st.metric("Peak Period Events", f"{peak_period_events}/{len(events_df)}")
        else:
            st.info("No peak events found for the selected threshold.")
    
    with viz_tabs[3]:
        st.markdown("**Battery Operation Simulation**")
        st.caption("Simulate how the battery would charge and discharge during peak events")
        
        # Allow user to input battery specifications
        col1, col2 = st.columns(2)
        with col1:
            sim_battery_power = st.number_input(
                "Battery Power Rating (kW)",
                min_value=10.0,
                max_value=1000.0,
                value=float(suggested_power_kw) if 'suggested_power_kw' in locals() else 50.0,
                step=10.0,
                key="sim_power"
            )
        with col2:
            sim_battery_capacity = st.number_input(
                "Battery Capacity (kWh)",
                min_value=20.0,
                max_value=2000.0,
                value=float(suggested_capacity_kwh) if 'suggested_capacity_kwh' in locals() else 100.0,
                step=10.0,
                key="sim_capacity"
            )
        
        # Battery simulation target
        sim_target = st.slider(
            "Peak Shaving Target (kW)",
            min_value=float(df[power_col].mean()),
            max_value=float(df[power_col].max()),
            value=float(demand_analysis['peak_events_analysis']['top_5_percent']['threshold_kw']),
            step=1.0,
            key="sim_target"
        )
        
        # Run battery simulation for a representative period
        data_days = len(df) / 96
        if data_days > 7:
            st.info("💡 Simulating battery operation for the first 7 days of data")
            sim_df = df.head(7 * 96)  # First 7 days
        else:
            sim_df = df
        
        # Calculate battery metrics
        with st.spinner("Simulating battery operation..."):
            battery_metrics = calculate_battery_metrics(
                power_profile=sim_df[power_col],
                target_peak_kw=sim_target,
                battery_capacity_kwh=sim_battery_capacity,
                battery_power_kw=sim_battery_power,
                efficiency=0.9,
                peak_period_only=True
            )
        
        # Create battery operation visualization
        fig_battery = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Power Consumption with Battery Operation',
                'Battery Power (Charge/Discharge)',
                'Battery State of Charge (SOC)'
            ],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Original vs grid power
        fig_battery.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df[power_col],
                mode='lines',
                name='Original Load',
                line=dict(color='#2E86C1', width=2),
                hovertemplate='Original: %{y:.1f} kW<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig_battery.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=battery_metrics['grid_power'],
                mode='lines',
                name='Grid Power (with battery)',
                line=dict(color='#27AE60', width=2),
                hovertemplate='Grid Power: %{y:.1f} kW<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Target line
        fig_battery.add_hline(
            y=sim_target,
            line_dash="dash",
            line_color='#E74C3C',
            line_width=2,
            annotation_text=f"Target: {sim_target:.1f} kW",
            row=1
        )
        
        # Battery charging/discharging
        fig_battery.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=battery_metrics['discharging_power'],
                mode='lines',
                name='Discharging',
                line=dict(color='#E74C3C', width=2),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.3)',
                hovertemplate='Discharge: %{y:.1f} kW<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_battery.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=-battery_metrics['charging_power'],
                mode='lines',
                name='Charging',
                line=dict(color='#3498DB', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.3)',
                hovertemplate='Charge: %{y:.1f} kW<extra></extra>'
            ),
            row=2, col=1
        )
        
        # SOC
        fig_battery.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=battery_metrics['soc_profile'],
                mode='lines',
                name='SOC',
                line=dict(color='#9B59B6', width=2),
                hovertemplate='SOC: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add SOC limits
        fig_battery.add_hline(y=90, line_dash="dash", line_color='gray', line_width=1, row=3)
        fig_battery.add_hline(y=10, line_dash="dash", line_color='gray', line_width=1, row=3)
        
        # Add peak period shading
        for date in pd.date_range(sim_df.index.min().date(), sim_df.index.max().date(), freq='D'):
            if date.weekday() < 5:  # Weekdays only
                peak_start = pd.Timestamp(f"{date} 08:00:00")
                peak_end = pd.Timestamp(f"{date} 22:00:00")
                
                for row in [1, 2, 3]:
                    fig_battery.add_vrect(
                        x0=peak_start, x1=peak_end,
                        fillcolor="rgba(255, 235, 59, 0.1)",
                        layer="below",
                        line_width=0,
                        row=row, col=1
                    )
        
        fig_battery.update_xaxes(title_text="Time", row=3, col=1)
        fig_battery.update_yaxes(title_text="Power (kW)", row=1, col=1)
        fig_battery.update_yaxes(title_text="Battery Power (kW)", row=2, col=1)
        fig_battery.update_yaxes(title_text="SOC (%)", row=3, col=1)
        
        fig_battery.update_layout(
            height=800,
            title_text=f"Battery Operation Simulation ({sim_battery_power} kW / {sim_battery_capacity} kWh)",
            showlegend=True
        )
        
        st.plotly_chart(fig_battery, use_container_width=True)
        
        # Simulation results
        st.subheader("🎯 Simulation Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            original_peak = battery_metrics['original_peak']
            achieved_peak = battery_metrics['achieved_peak']
            st.metric("Original Peak", f"{original_peak:.1f} kW")
            st.metric("Achieved Peak", f"{achieved_peak:.1f} kW")
        with col2:
            peak_reduction = battery_metrics['peak_reduction']
            reduction_pct = (peak_reduction / original_peak) * 100
            st.metric("Peak Reduction", f"{peak_reduction:.1f} kW")
            st.metric("Reduction %", f"{reduction_pct:.1f}%")
        with col3:
            discharge_energy = battery_metrics['total_discharge_energy'] * 0.25  # Convert to kWh
            cycle_count = battery_metrics['cycle_count']
            st.metric("Energy Discharged", f"{discharge_energy:.1f} kWh")
            st.metric("Annual Cycles", f"{cycle_count:.0f}")
        with col4:
            peak_utilization = battery_metrics['peak_period_utilization']
            monthly_savings_sim = peak_reduction * demand_rate
            st.metric("Peak Period Utilization", f"{peak_utilization:.1f}%")
            st.metric("Monthly MD Savings", f"RM {monthly_savings_sim:,.0f}")
        
        # Performance assessment
        if reduction_pct > 90:
            st.success("✅ **Excellent Performance** - Battery achieving target peak reduction")
        elif reduction_pct > 70:
            st.info("💡 **Good Performance** - Consider increasing battery capacity for better results")
        else:
            st.warning("⚠️ **Limited Performance** - Battery undersized for target reduction")
            st.markdown("**Recommendations:**")
            st.markdown("- Increase battery power rating for higher peak shaving capability")
            st.markdown("- Increase battery capacity for longer peak event coverage")
            st.markdown("- Consider adjusting peak shaving target to match battery capabilities")
    
    # ===================================================================
    # STEP 3: INTELLIGENT BATTERY SIZING
    # ===================================================================
    st.header("⚡ Step 3: Intelligent Battery Sizing")
    st.markdown("*Smart sizing based on energy requirements with adjustable C-Rate for optimal performance*")
    
    # Use the percentage-based target that was already selected earlier
    target_threshold = target_demand_kw
    md_reduction = current_max_demand - target_demand_kw
    monthly_md_savings = potential_monthly_savings
    
    # Energy-first intelligent battery sizing
    st.subheader("🎯 Energy-First Intelligent Battery Sizing")
    st.info("""
    **New Smart Sizing Logic:**
    1. **Total Energy Requirement** - Sum of ALL peak events above threshold (not just worst single event)
    2. **C-Rate Selection** - Adjustable C-Rate determines optimal power/capacity balance  
    3. **Safety Factor** - Add your preferred safety margin for real-world conditions
    4. **Final Sizing** - Calculate both power (kW) and capacity (kWh) automatically
    
    **Key Change**: Battery is now sized to handle the **total cumulative energy** from all peak events,
    ensuring it can manage the full energy requirement over time through strategic charging/discharging.
    """)
    
    # Calculate base energy requirement from the custom analysis
    # Use total energy requirement from ALL peak events (sum of all events above threshold)
    total_energy_required_kwh = custom_analysis.get('total_excess_energy_kwh', md_reduction * 2.0)
    worst_case_power = md_reduction
    
    # IMPORTANT CHANGE: Using total sum of all peak events, not just single worst-case event
    # This ensures the battery can handle the cumulative energy requirement from all peak events
    
    # Show base energy requirement
    st.subheader("📊 Total Energy Requirement (All Peak Events)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Energy Required",
            f"{total_energy_required_kwh:.1f} kWh",
            help="Total energy needed for peak shaving based on sum of ALL peak events above threshold"
        )
    
    with col2:
        st.metric(
            "Peak Power Needed", 
            f"{worst_case_power:.1f} kW",
            help="Peak power reduction required during events"
        )
    
    with col3:
        estimated_duration = total_energy_required_kwh / worst_case_power if worst_case_power > 0 else 0
        st.metric(
            "Equivalent Duration",
            f"{estimated_duration:.1f} hours", 
            help="Equivalent hours at full power needed to cover all peak events"
        )
    
    # User adjustable parameters
    with st.expander("🔧 Intelligent Battery Sizing Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚡ C-Rate & Performance Settings**")
            
            # C-Rate selection (this is the key parameter)
            c_rate = st.slider(
                "C-Rate Selection",
                min_value=0.1,
                max_value=2.0,
                value=0.9,
                step=0.1,
                help="C-Rate determines power capability: Power = Capacity × C-Rate"
            )
            
            # C-Rate explanation
            if c_rate <= 0.5:
                c_rate_desc = "🟢 Conservative (Long Duration, Lower Cost)"
                c_rate_app = "Long Duration Storage"
            elif c_rate <= 1.0:
                c_rate_desc = "🟡 Balanced (Standard Applications)"  
                c_rate_app = "Commercial Peak Shaving"
            else:
                c_rate_desc = "🔴 Aggressive (High Power, Premium Cost)"
                c_rate_app = "High Power Applications"
            
            st.caption(f"**{c_rate_desc}**")
            st.caption(f"*Application: {c_rate_app}*")
            
            # Round-trip efficiency
            rte = st.slider(
                "Round-Trip Efficiency (%)",
                min_value=80,
                max_value=95,
                value=90,
                step=1,
                help="Battery system efficiency (charge/discharge losses)"
            ) / 100
            
        with col2:
            st.markdown("**🔋 Safety Factor Settings**")
            
            # Safety factor for battery sizing
            safety_factor = st.slider(
                "Battery Safety Factor (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Safety margin for degradation, unexpected events, and performance buffer"
            ) / 100
            
            # Show what the safety factor covers
            st.caption("""
            **Safety Factor Covers:**
            - Battery degradation over time
            - Unexpected peak events  
            - Temperature effects
            - Depth of discharge limits
            """)
    
    # Real-time intelligent sizing calculations
    st.subheader("🧠 Intelligent Sizing Calculations")
    
    # Step 1: Adjust energy requirement for efficiency losses
    adjusted_energy_kwh = total_energy_required_kwh / rte
    
    # Step 2: Apply safety factor to get base capacity
    base_capacity_kwh = adjusted_energy_kwh * (1 + safety_factor)
    
    # Step 3: Calculate power rating from C-Rate
    # Power = Capacity × C-Rate
    calculated_power_kw = base_capacity_kwh * c_rate
    
    # Step 4: Check if calculated power meets minimum requirement
    min_power_needed = worst_case_power
    
    if calculated_power_kw < min_power_needed:
        # If C-Rate gives insufficient power, increase capacity to meet power requirement
        required_capacity_for_power = min_power_needed / c_rate
        final_capacity_kwh = max(base_capacity_kwh, required_capacity_for_power)
        final_power_kw = final_capacity_kwh * c_rate
        sizing_limited_by = "Power Requirement"
    else:
        # C-Rate calculation provides sufficient power
        final_capacity_kwh = base_capacity_kwh
        final_power_kw = calculated_power_kw
        sizing_limited_by = "Energy Requirement"
    
    # Display results in metrics
    st.markdown("**📊 Final Battery Specifications:**")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            "Battery Capacity",
            f"{final_capacity_kwh:.1f} kWh",
            delta=f"+{((final_capacity_kwh/total_energy_required_kwh)-1)*100:.0f}% vs min",
            help="Final battery capacity with safety factor"
        )
    
    with metrics_col2:
        st.metric(
            "Battery Power",
            f"{final_power_kw:.1f} kW",
            delta=f"C-Rate: {c_rate:.1f}",
            help="Battery power rating based on C-Rate"
        )
    
    with metrics_col3:
        max_discharge_duration = final_capacity_kwh / final_power_kw
        st.metric(
            "Max Duration",
            f"{max_discharge_duration:.1f} hours",
            delta="At full power",
            help="How long battery can discharge at full power"
        )
    
    with metrics_col4:
        energy_utilization = (total_energy_required_kwh / final_capacity_kwh) * 100
        st.metric(
            "Energy Utilization",
            f"{energy_utilization:.1f}%",
            delta="Peak event",
            help="Capacity utilization during worst-case event"
        )
    
    # Detailed breakdown
    with st.expander("🔍 Detailed Sizing Breakdown", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Step-by-Step Calculation:**")
            st.write(f"1. **Base Energy Need**: {total_energy_required_kwh:.1f} kWh (Sum of all peak events)")
            st.write(f"2. **Efficiency Adjusted**: {adjusted_energy_kwh:.1f} kWh (÷{rte:.0%})")
            st.write(f"3. **With Safety Factor**: {base_capacity_kwh:.1f} kWh (+{safety_factor:.0%})")
            st.write(f"4. **Power from C-Rate**: {calculated_power_kw:.1f} kW (×{c_rate:.1f})")
            st.write(f"5. **Final Capacity**: {final_capacity_kwh:.1f} kWh")
            st.write(f"6. **Final Power**: {final_power_kw:.1f} kW")
            
        with col2:
            st.markdown("**Design Verification:**")
            st.write(f"• **Sizing Limited By**: {sizing_limited_by}")
            st.write(f"• **Min Power Needed**: {min_power_needed:.1f} kW")
            st.write(f"• **Power Margin**: {((final_power_kw/min_power_needed)-1)*100:.0f}%")
            st.write(f"• **Energy Margin**: {((final_capacity_kwh/total_energy_required_kwh)-1)*100:.0f}%")
            st.write(f"• **Safety Factor**: {safety_factor:.0%}")
            st.write(f"• **Round-Trip Efficiency**: {rte:.0%}")
    
    # C-Rate impact analysis
    st.subheader("📈 C-Rate Impact Analysis")
    
    # Create comparison table for different C-Rates
    c_rates_comparison = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
    comparison_data = []
    
    for cr in c_rates_comparison:
        # Calculate capacity and power for this C-Rate
        comp_power = base_capacity_kwh * cr
        comp_capacity = base_capacity_kwh
        
        # Check if power meets minimum requirement
        if comp_power < min_power_needed:
            comp_capacity = min_power_needed / cr
            comp_power = comp_capacity * cr
            limited_by = "Power"
        else:
            limited_by = "Energy"
        
        duration = comp_capacity / comp_power
        relative_cost = cr * 100  # Higher C-Rate = higher relative cost
        
        comparison_data.append({
            'C-Rate': f"{cr:.1f}",
            'Capacity (kWh)': f"{comp_capacity:.0f}",
            'Power (kW)': f"{comp_power:.0f}",
            'Duration (h)': f"{duration:.1f}",
            'Limited By': limited_by,
            'Relative Cost': f"{relative_cost:.0f}%",
            'Selected': "✅" if abs(cr - c_rate) < 0.05 else ""
        })
    
    df_c_rate = pd.DataFrame(comparison_data)
    st.dataframe(df_c_rate, use_container_width=True)
    
    # Show final recommendations
    annual_md_savings = monthly_md_savings * 12
    st.success(f"""
    **🏆 Final Intelligent Battery Sizing Recommendation:**
    
    **Battery Specifications:**
    • **Capacity**: {final_capacity_kwh:.1f} kWh  
    • **Power Rating**: {final_power_kw:.1f} kW
    • **C-Rate**: {c_rate:.1f} 
    • **Safety Factor**: {safety_factor:.0%}
    
    **Performance Prediction:**
    • **Peak Reduction**: {md_reduction:.1f} kW
    • **Annual MD Savings**: RM {annual_md_savings:,.0f}
    • **Total Energy Coverage**: {energy_utilization:.1f}% of capacity used for all peak events
    • **Max Discharge Duration**: {max_discharge_duration:.1f} hours
    
    **Design Notes:**
    • Sizing based on: Total sum of ALL peak events (not single worst-case)
    • Sizing limited by: {sizing_limited_by}
    • Includes {safety_factor:.0%} safety margin for real-world conditions
    • {rte:.0%} round-trip efficiency accounted for
    """)
    
    # Economic preview
    if demand_rate > 0:
        # Rough battery cost estimation (can be refined later)
        estimated_battery_cost_per_kwh = 800  # RM per kWh (rough estimate)
        estimated_total_cost = final_capacity_kwh * estimated_battery_cost_per_kwh
        simple_payback = estimated_total_cost / annual_md_savings if annual_md_savings > 0 else float('inf')
        
        st.info(f"""
        **💰 Quick Economic Preview:**
        
        • **Estimated Cost**: RM {estimated_total_cost:,.0f} ({estimated_battery_cost_per_kwh:.0f}/kWh × {final_capacity_kwh:.0f} kWh)
        • **Annual Savings**: RM {annual_md_savings:,.0f} 
        • **Simple Payback**: {simple_payback:.1f} years
        
        *Note: Detailed cost analysis and ROI calculations in Step 4 below.*
        """)
    
    # Update variables for downstream calculations
    suggested_power_kw = final_power_kw
    suggested_capacity_kwh = final_capacity_kwh
    
    # ===================================================================
    # STEP 4: COST ESTIMATION
    # ===================================================================
    st.header("💰 Step 4: Battery System Cost Estimation")
    st.markdown("*Estimate total investment required for your battery system*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Cost Components")
        
        # Cost parameters
        battery_cost_per_kwh = st.number_input(
            "Battery Cost (RM/kWh)",
            min_value=500.0,
            max_value=1500.0,
            value=800.0,
            step=50.0,
            help="Installed battery system cost per kWh"
        )
        
        inverter_cost_per_kw = st.number_input(
            "Inverter Cost (RM/kW)",
            min_value=300.0,
            max_value=800.0,
            value=500.0,
            step=50.0,
            help="Power electronics cost per kW"
        )
        
        installation_percentage = st.slider(
            "Installation & Others (%)",
            min_value=10,
            max_value=30,
            value=15,
            help="Installation, commissioning, and other costs as % of equipment"
        )
    
    with col2:
        st.subheader("💸 Cost Breakdown")
        
        # Calculate costs
        battery_cost = suggested_capacity_kwh * battery_cost_per_kwh
        inverter_cost = suggested_power_kw * inverter_cost_per_kw
        equipment_cost = battery_cost + inverter_cost
        installation_cost = equipment_cost * (installation_percentage / 100)
        total_capex = equipment_cost + installation_cost
        
        # Display costs
        st.metric("Battery System", f"RM {battery_cost:,.0f}")
        st.metric("Inverter System", f"RM {inverter_cost:,.0f}")
        st.metric("Installation & Others", f"RM {installation_cost:,.0f}")
        st.metric("**Total CAPEX**", f"**RM {total_capex:,.0f}**")
        
        # Cost per kW reduced
        cost_per_kw_reduced = total_capex / md_reduction if md_reduction > 0 else 0
        st.metric("Cost per kW Reduced", f"RM {cost_per_kw_reduced:,.0f}")
    
    # ===================================================================
    # STEP 5: ROI ANALYSIS & PAYBACK CALCULATION
    # ===================================================================
    st.header("📈 Step 5: ROI Analysis & Investment Justification")
    st.markdown("*Calculate return on investment and payback period for your battery system*")
    
    # Financial parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_years = st.number_input(
            "Analysis Period (years)",
            min_value=5,
            max_value=25,
            value=15,
            step=1
        )
    
    with col2:
        discount_rate = st.slider(
            "Discount Rate (%)",
            min_value=3.0,
            max_value=10.0,
            value=6.0,
            step=0.5,
            help="Cost of capital for NPV calculation"
        ) / 100
    
    with col3:
        maintenance_rate = st.slider(
            "Annual Maintenance (%)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Annual O&M cost as % of CAPEX"
        ) / 100
    
    # Calculate comprehensive ROI
    roi_analysis = calculate_md_savings_and_roi(
        peak_reduction_kw=md_reduction,
        demand_rate_per_kw_month=demand_rate,
        battery_capex=total_capex,
        analysis_years=analysis_years,
        discount_rate=discount_rate,
        maintenance_rate=maintenance_rate
    )
    
    # Display key financial metrics
    st.subheader("🎯 Key Financial Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly MD Savings", f"RM {roi_analysis['monthly_md_savings']:,.0f}")
    with col2:
        st.metric("Annual Net Savings", f"RM {roi_analysis['net_annual_savings']:,.0f}")
    with col3:
        payback_years = roi_analysis['simple_payback_years']
        if payback_years < 20:
            st.metric("Payback Period", f"{payback_years:.1f} years")
        else:
            st.metric("Payback Period", "> 20 years")
    with col4:
        st.metric("NPV", f"RM {roi_analysis['npv']:,.0f}")
    
    # Investment viability assessment
    if payback_years < 5:
        st.success("🎯 **Excellent Investment** - Very attractive payback period!")
        investment_recommendation = "Proceed with confidence"
    elif payback_years < 8:
        st.info("💡 **Good Investment** - Reasonable payback period")
        investment_recommendation = "Good business case"
    elif payback_years < 12:
        st.warning("⚠️ **Marginal Investment** - Consider optimizing sizing")
        investment_recommendation = "Review optimization opportunities"
    else:
        st.error("❌ **Poor Investment** - Not economically viable")
        investment_recommendation = "Not recommended under current assumptions"
    
    # Detailed financial analysis chart
    st.subheader("💹 Financial Analysis Over Time")
    
    # Create financial projection
    years = list(range(0, analysis_years + 1))
    cumulative_investment = [total_capex] + [total_capex] * analysis_years
    cumulative_savings = [0]
    annual_maintenance = roi_analysis['annual_maintenance_cost']
    
    for year in range(1, analysis_years + 1):
        prev_savings = cumulative_savings[-1]
        new_savings = prev_savings + roi_analysis['annual_md_savings'] - annual_maintenance
        cumulative_savings.append(new_savings)
    
    net_position = [savings - investment for savings, investment in zip(cumulative_savings, cumulative_investment)]
    
    fig_financial = go.Figure()
    
    # Cumulative investment
    fig_financial.add_trace(go.Scatter(
        x=years,
        y=cumulative_investment,
        mode='lines',
        name='Cumulative Investment',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Cumulative savings
    fig_financial.add_trace(go.Scatter(
        x=years,
        y=cumulative_savings,
        mode='lines',
        name='Cumulative Savings',
        line=dict(color='green', width=2)
    ))
    
    # Net position
    fig_financial.add_trace(go.Scatter(
        x=years,
        y=net_position,
        mode='lines+markers',
        name='Net Position',
        line=dict(color='blue', width=3),
        fill='tonexty'
    ))
    
    # Break-even point
    if payback_years < analysis_years:
        fig_financial.add_vline(
            x=payback_years,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"Break-even: {payback_years:.1f} years"
        )
    
    fig_financial.update_layout(
        title=f"Investment Analysis - {md_reduction:.1f} kW Peak Reduction",
        xaxis_title="Years",
        yaxis_title="Amount (RM)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_financial, use_container_width=True)
    
    # ===================================================================
    # STEP 6: SIMULATION & VALIDATION
    # ===================================================================
    st.header("🚀 Step 6: Battery Performance Simulation")
    st.markdown("*Simulate actual battery operation to validate the analysis*")
    
    if st.button("🔋 Run Battery Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating intelligent battery operation..."):
            # Run battery simulation
            battery_metrics = calculate_battery_metrics(
                df[power_col],
                target_threshold,
                suggested_capacity_kwh,
                suggested_power_kw,
                efficiency=0.9,
                peak_period_only=True
            )
            
            # Display simulation results
            st.subheader("🎯 Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Peak Achieved",
                    f"{battery_metrics['achieved_peak']:.1f} kW",
                    f"-{battery_metrics['peak_reduction']:.1f} kW"
                )
            with col2:
                actual_savings = battery_metrics['peak_reduction'] * demand_rate
                st.metric(
                    "Actual Monthly Savings",
                    f"RM {actual_savings:,.0f}",
                    f"vs RM {monthly_md_savings:,.0f} target"
                )
            with col3:
                achievement_rate = (battery_metrics['peak_reduction'] / md_reduction) * 100 if md_reduction > 0 else 0
                st.metric("Target Achievement", f"{achievement_rate:.1f}%")
            with col4:
                daily_cycles = battery_metrics['cycle_count'] / (len(df) / 96)
                st.metric("Daily Cycles", f"{daily_cycles:.2f}")
            
            # Performance validation
            target_met = battery_metrics['achieved_peak'] <= target_threshold * 1.05
            if target_met:
                st.success(f"✅ **Target Achieved!** Peak reduced from {current_peak:.1f} kW to {battery_metrics['achieved_peak']:.1f} kW")
            else:
                st.warning(f"⚠️ **Target Not Fully Met** - Consider increasing battery power rating")
            
            # Show battery operation chart (simplified for space)
            if len(df) >= 96:  # At least 24 hours
                sample_hours = min(96 * 3, len(df))  # 3 days
                sample_df = df.head(sample_hours).copy()
                
                fig_operation = go.Figure()
                
                # Original demand
                fig_operation.add_trace(go.Scatter(
                    x=sample_df.index,
                    y=sample_df[power_col],
                    name='Original Demand',
                    line=dict(color='blue', width=2)
                ))
                
                # Grid power with battery
                grid_power_sample = battery_metrics['grid_power'][:sample_hours]
                fig_operation.add_trace(go.Scatter(
                    x=sample_df.index,
                    y=grid_power_sample,
                    name='Grid Power (with Battery)',
                    line=dict(color='green', width=2),
                    fill='tonexty'
                ))
                
                # Target line
                fig_operation.add_hline(
                    y=target_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Target: {target_threshold:.1f} kW"
                )
                
                fig_operation.update_layout(
                    title="Battery Operation Simulation (First 3 Days)",
                    xaxis_title="Time",
                    yaxis_title="Power (kW)",
                    height=400
                )
                
                st.plotly_chart(fig_operation, use_container_width=True)
    
    # ===================================================================
    # FINAL RECOMMENDATIONS & NEXT STEPS
    # ===================================================================
    st.header("📋 Final Recommendations & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Recommended Configuration")
        st.success(f"**RP4 Tariff:** {selected_tariff_name}")
        st.success(f"**Current MD Rate:** RM {demand_rate:.2f}/kW/month")
        st.success(f"**Battery Power:** {suggested_power_kw:.0f} kW")
        st.success(f"**Battery Capacity:** {suggested_capacity_kwh:.0f} kWh")
        st.success(f"**Target Peak:** {target_threshold:.1f} kW")
        st.success(f"**Total Investment:** RM {total_capex:,.0f}")
        st.success(f"**Monthly MD Savings:** RM {monthly_md_savings:,.0f}")
        st.success(f"**Payback Period:** {payback_years:.1f} years")
        
        st.info(f"**Investment Grade:** {investment_recommendation}")
    
    with col2:
        st.subheader("🚀 Next Steps")
        st.markdown("""
        **Technical:**
        1. Site electrical assessment
        2. Detailed engineering design
        3. Regulatory approvals (TNB)
        4. Installation planning
        
        **Commercial:**
        1. Vendor RFQ with specifications
        2. Financing options evaluation
        3. Performance guarantees
        4. Installation timeline
        """)
    
    # Export analysis report
    if st.button("📊 Export Analysis Report", type="secondary"):
        report_data = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'load_profile': {
                'peak_demand_kw': current_peak,
                'average_demand_kw': df[power_col].mean(),
                'data_points': len(df)
            },
            'tariff_info': {
                'rp4_tariff': selected_tariff_name,
                'tariff_type': selected_tariff.get('Type', 'N/A'),
                'voltage_level': selected_tariff.get('Voltage', 'N/A'),
                'demand_rate_rm_per_kw_month': demand_rate,
                'current_annual_md_cost': current_annual_md_cost
            },
            'battery_sizing': {
                'target_peak_kw': target_threshold,
                'battery_power_kw': suggested_power_kw,
                'battery_capacity_kwh': suggested_capacity_kwh,
                'c_rate': c_rate
            },
            'economics': {
                'total_capex_rm': total_capex,
                'monthly_md_savings_rm': monthly_md_savings,
                'annual_net_savings_rm': roi_analysis['net_annual_savings'],
                'payback_years': payback_years,
                'npv_rm': roi_analysis['npv'],
                'irr_percentage': roi_analysis['irr_percentage']
            },
            'recommendation': investment_recommendation
        }
        
        import json
        report_json = json.dumps(report_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Analysis Report",
            data=report_json,
            file_name=f"battery_sizing_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Analysis report generated!")


def show():
    """
    Wrapper function that calls the main battery sizing analysis page
    """
    battery_sizing_analysis_page()


def show_standalone():
    """
    Standalone version of battery sizing analysis (for independent use)
    """
    st.title("🔋 Intelligent Battery Sizing Analysis")
    st.markdown("""
    **Smart Peak Shaving Optimization** - Upload your load profile and get intelligent battery sizing recommendations 
    based on your actual demand patterns and peak shaving opportunities.
    """)
    
    # File upload section
    st.subheader("📁 Load Profile Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your energy consumption data", 
        type=["xlsx", "csv"], 
        help="Upload Excel or CSV file with timestamps and power consumption data"
    )
    
    if uploaded_file:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.write("**Column Names:**", list(df.columns))
            
            # Column selection
            st.subheader("🎯 Column Selection")
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp_col = st.selectbox(
                    "Select Timestamp Column",
                    df.columns,
                    help="Column containing timestamp information"
                )
            
            with col2:
                power_col = st.selectbox(
                    "Select Power Column (kW)",
                    [col for col in df.columns if col != timestamp_col],
                    help="Column containing power consumption data in kW"
                )
            
            if timestamp_col and power_col:
                # Process data
                df['Parsed Timestamp'] = pd.to_datetime(df[timestamp_col])
                df = df.set_index('Parsed Timestamp').sort_index()
                df = df.dropna(subset=[power_col])
                
                # Convert power to numeric
                df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
                df = df.dropna(subset=[power_col])
                
                # Store in session state and call main analysis
                st.session_state['processed_df'] = df
                st.session_state['power_column'] = power_col
                st.session_state['uploaded_file'] = uploaded_file
                
                # Call the main battery sizing analysis
                battery_sizing_analysis_page()
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file has timestamp and power consumption columns.")
    
    else:
        # Getting started section
        st.subheader("📚 Getting Started")
        st.markdown("""
        **Data Requirements:**
        - Timestamp column (any common format)
        - Power consumption column in kW
        - Minimum 24 hours of data
        - 15-minute intervals recommended
        
        **What You'll Get:**
        - Intelligent peak demand analysis
        - Optimal battery sizing recommendations
        - Comprehensive ROI analysis
        - Detailed financial projections
        """)
        

def analyze_peak_events_with_custom_target(power_data, target_demand_kw):
    """
    Analyze peak events based on a custom target demand threshold
    
    Args:
        power_data: pandas Series with power consumption data and datetime index
        target_demand_kw: float, target demand threshold in kW
        
    Returns:
        dict with peak events analysis for the custom target
    """
    # Ensure datetime index
    if not hasattr(power_data.index, 'hour'):
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        freq = pd.infer_freq(power_data.index) or '15T'
        power_data.index = pd.date_range(start=start_time, periods=len(power_data), freq=freq)
    
    # Basic statistics
    max_demand = power_data.max()
    avg_demand = power_data.mean()
    
    # Peak period analysis (Monday-Friday 8AM-10PM for RP4 MD recording)
    peak_period_mask = power_data.index.to_series().apply(
        lambda ts: ts.weekday() < 5 and 8 <= ts.hour < 22
    )
    
    # Find all events above target threshold
    above_threshold_mask = power_data >= target_demand_kw
    
    # Extract individual peak events
    peak_events = []
    in_peak = False
    current_event = {}
    
    for i, (timestamp, power_value) in enumerate(power_data.items()):
        is_above = above_threshold_mask.iloc[i]
        is_peak_period = peak_period_mask.iloc[i]
        
        if is_above and not in_peak:
            # Start of peak event
            in_peak = True
            current_event = {
                'start_time': timestamp,
                'start_index': i,
                'peak_load': power_value,
                'excess_energy': power_value - target_demand_kw,
                'in_peak_period': is_peak_period,
                'duration_intervals': 1
            }
            
        elif is_above and in_peak:
            # Continue peak event
            current_event['duration_intervals'] += 1
            current_event['excess_energy'] += power_value - target_demand_kw
            if power_value > current_event['peak_load']:
                current_event['peak_load'] = power_value
            # Update peak period status (true if any part is in peak period)
            if is_peak_period:
                current_event['in_peak_period'] = True
                
        elif not is_above and in_peak:
            # End of peak event
            in_peak = False
            current_event['end_time'] = timestamp
            current_event['end_index'] = i - 1
            current_event['duration_hours'] = current_event['duration_intervals'] * 0.25  # 15min intervals
            current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
            current_event['md_reduction_potential'] = current_event['peak_load'] - target_demand_kw
            
            peak_events.append(current_event.copy())
    
    # Handle case where data ends during peak
    if in_peak and current_event:
        current_event['end_time'] = power_data.index[-1]
        current_event['end_index'] = len(power_data) - 1
        current_event['duration_hours'] = current_event['duration_intervals'] * 0.25
        current_event['avg_excess_power'] = current_event['excess_energy'] / current_event['duration_intervals']
        current_event['md_reduction_potential'] = current_event['peak_load'] - target_demand_kw
        peak_events.append(current_event.copy())
    
    # Calculate summary statistics
    if peak_events:
        peak_period_events = [e for e in peak_events if e['in_peak_period']]
        off_peak_events = [e for e in peak_events if not e['in_peak_period']]
        
        total_excess_energy = sum(e['excess_energy'] for e in peak_events) * 0.25  # Convert to kWh
        peak_period_excess = sum(e['excess_energy'] for e in peak_period_events) * 0.25
        
        analysis_result = {
            'threshold_kw': target_demand_kw,
            'total_events': len(peak_events),
            'peak_events_count': len(peak_events),
            'peak_period_events': len(peak_period_events),
            'off_peak_events': len(off_peak_events),
            'avg_duration_hours': np.mean([e['duration_hours'] for e in peak_events]),
            'max_duration_hours': max([e['duration_hours'] for e in peak_events]),
            'total_excess_energy_kwh': total_excess_energy,
            'peak_period_excess_kwh': peak_period_excess,
            'md_reduction_potential_kw': max_demand - target_demand_kw,
            'md_reduction_percentage': ((max_demand - target_demand_kw) / max_demand) * 100,
            'percentage_of_time': (len(peak_events) / len(power_data)) * 100,
            'avg_event_excess_power': np.mean([e['avg_excess_power'] for e in peak_events]),
            'max_event_excess_power': max([e['md_reduction_potential'] for e in peak_events]),
            'peak_period_percentage': len(peak_period_events) / len(peak_events) * 100 if peak_events else 0,
            'events_detail': peak_events
        }
    else:
        analysis_result = {
            'threshold_kw': target_demand_kw,
            'total_events': 0,
            'peak_events_count': 0,
            'peak_period_events': 0,
            'off_peak_events': 0,
            'avg_duration_hours': 0,
            'max_duration_hours': 0,
            'total_excess_energy_kwh': 0,
            'peak_period_excess_kwh': 0,
            'md_reduction_potential_kw': max_demand - target_demand_kw,
            'md_reduction_percentage': ((max_demand - target_demand_kw) / max_demand) * 100,
            'percentage_of_time': 0,
            'avg_event_excess_power': 0,
            'max_event_excess_power': 0,
            'peak_period_percentage': 0,
            'events_detail': []
        }
    
    return {
        'max_demand': max_demand,
        'avg_demand': avg_demand,
        'custom_target_analysis': analysis_result,
        'peak_events_above_target': peak_events,
        'peak_period_mask': peak_period_mask
    }
