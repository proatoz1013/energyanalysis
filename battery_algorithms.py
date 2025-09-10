"""
Battery Algorithms Module

This module provides comprehensive battery energy storage system (BESS) algorithms
for maximum demand (MD) shaving applications. It includes:

- Battery sizing algorithms
- Charge/discharge simulation models
- Financial analysis and ROI calculations
- Battery performance optimization
- Cost analysis and lifecycle modeling

Author: Energy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import RP4 peak logic functions
from tariffs.peak_logic import is_peak_rp4, get_malaysia_holidays


def _get_dynamic_interval_hours(df):
    """
    Detect sampling interval from DataFrame and return interval in hours.
    
    Args:
        df: DataFrame with datetime index or containing timestamp column
        
    Returns:
        float: Interval in hours (e.g., 0.25 for 15-min intervals)
    """
    try:
        # If DataFrame has datetime index
        if hasattr(df.index, 'freq') and df.index.freq is not None:
            return pd.Timedelta(df.index.freq).total_seconds() / 3600
        
        # Try to infer from index
        if len(df) > 1:
            time_diff = df.index[1] - df.index[0]
            if hasattr(time_diff, 'total_seconds'):
                return time_diff.total_seconds() / 3600
        
        # Look for timestamp columns
        for col in ['timestamp', 'Timestamp', 'DateTime', 'datetime']:
            if col in df.columns:
                timestamps = pd.to_datetime(df[col])
                if len(timestamps) > 1:
                    time_diff = timestamps.iloc[1] - timestamps.iloc[0]
                    return time_diff.total_seconds() / 3600
        
        # Default fallback to 15-minute intervals
        return 0.25
        
    except Exception:
        # Fallback to 15-minute intervals
        return 0.25


def detect_holidays_from_data(df, timestamp_col):
    """
    Detect holidays from data by analyzing patterns.
    For now, returns default Malaysia holidays based on year range in data.
    """
    if len(df) == 0:
        return get_malaysia_holidays(2025)
    
    # Get year range from data
    try:
        start_year = df.index.min().year if hasattr(df.index.min(), 'year') else 2025
        end_year = df.index.max().year if hasattr(df.index.max(), 'year') else 2025
    except:
        # Fallback if no valid dates
        return get_malaysia_holidays(2025)
    
    # Combine holidays from all years in the data range
    all_holidays = set()
    for year in range(start_year, end_year + 1):
        year_holidays = get_malaysia_holidays(year)
        if year_holidays:
            all_holidays.update(year_holidays)
    
    return all_holidays if all_holidays else get_malaysia_holidays(2025)


class BatteryAlgorithms:
    """
    Comprehensive battery algorithms for MD shaving applications.
    This class contains all battery-related calculations and simulations.
    """
    
    def __init__(self):
        """Initialize battery algorithms with default parameters."""
        self.default_params = {
            'depth_of_discharge': 85,  # %
            'round_trip_efficiency': 92,  # %
            'discharge_efficiency': 94,  # %
            'c_rate': 0.5,  # C-rate
            'capex_per_kwh': 1200,  # RM/kWh
            'pcs_cost_per_kw': 400,  # RM/kW
            'installation_factor': 1.4,
            'opex_percent': 3.0,  # % of CAPEX
            'battery_life_years': 15,
            'discount_rate': 8.0  # %
        }
    
    def calculate_optimal_sizing(self, event_summaries, target_demand, interval_hours, sizing_params):
        """
        Calculate optimal battery sizing based on peak events and sizing strategy.
        
        Args:
            event_summaries: List of peak event dictionaries
            target_demand: Target maximum demand (kW)
            interval_hours: Data interval in hours
            sizing_params: Dictionary of sizing parameters
            
        Returns:
            Dictionary containing sizing results
        """
        if sizing_params['sizing_approach'] == "Manual Capacity":
            return self._manual_sizing(sizing_params)
        
        # Calculate energy requirements from peak events
        energy_requirements = self._analyze_energy_requirements(event_summaries)
        
        if sizing_params['sizing_approach'] == "Auto-size for Peak Events":
            return self._auto_sizing_for_events(energy_requirements, sizing_params)
        
        elif sizing_params['sizing_approach'] == "Energy Duration-based":
            return self._duration_based_sizing(energy_requirements, sizing_params)
        
        else:
            raise ValueError(f"Unknown sizing approach: {sizing_params['sizing_approach']}")
    
    def _manual_sizing(self, sizing_params):
        """Manual battery sizing with safety factors."""
        return {
            'capacity_kwh': sizing_params['manual_capacity_kwh'],
            'power_rating_kw': sizing_params['manual_power_kw'],
            'sizing_method': f"Manual Configuration (Capacity: +{sizing_params.get('capacity_safety_factor', 0)}% safety, Power: +{sizing_params.get('power_safety_factor', 0)}% safety)",
            'safety_factors_applied': True
        }
    
    def _analyze_energy_requirements(self, event_summaries):
        """Analyze energy requirements from peak events."""
        if not event_summaries:
            return {
                'total_energy_to_shave': 0,
                'worst_event_energy_peak_only': 0,
                'max_md_excess': 0
            }
        
        total_energy_to_shave = 0
        worst_event_energy_peak_only = 0
        max_md_excess = 0
        
        for event in event_summaries:
            # Use TOU Required Energy (kWh) for capacity sizing
            energy_kwh_peak_only = event.get('TOU Required Energy (kWh)', 0)
            # Use TOU Excess (kW) for power sizing
            md_excess_power = event.get('TOU Excess (kW)', 0)
            
            total_energy_to_shave += energy_kwh_peak_only
            worst_event_energy_peak_only = max(worst_event_energy_peak_only, energy_kwh_peak_only)
            max_md_excess = max(max_md_excess, md_excess_power)
        
        return {
            'total_energy_to_shave': total_energy_to_shave,
            'worst_event_energy_peak_only': worst_event_energy_peak_only,
            'max_md_excess': max_md_excess
        }
    
    def _auto_sizing_for_events(self, energy_requirements, sizing_params):
        """Auto-sizing based on worst-case peak events."""
        worst_event_energy = energy_requirements['worst_event_energy_peak_only']
        max_md_excess = energy_requirements['max_md_excess']
        
        if worst_event_energy > 0:
            required_capacity = worst_event_energy / (sizing_params['depth_of_discharge'] / 100)
            required_power = max_md_excess
            
            # Apply auto-sizing safety factors
            capacity_safety = sizing_params.get('auto_capacity_safety', 20) / 100
            power_safety = sizing_params.get('auto_power_safety', 15) / 100
            
            required_capacity *= (1 + capacity_safety)
            required_power *= (1 + power_safety)
            
            sizing_method = f"Auto-sized for worst MD peak event ({worst_event_energy:.1f} kWh + {sizing_params.get('auto_capacity_safety', 20)}% safety)"
        else:
            required_capacity = 100  # Minimum
            required_power = 50
            sizing_method = "Default minimum sizing (no peak events detected)"
        
        # Apply C-rate constraints
        c_rate_capacity = required_power / sizing_params.get('c_rate', 0.5)
        final_capacity = max(required_capacity, c_rate_capacity)
        
        return {
            'capacity_kwh': final_capacity,
            'power_rating_kw': required_power,
            'required_energy_kwh': energy_requirements['total_energy_to_shave'],
            'worst_event_energy_peak_only': worst_event_energy,
            'max_md_excess': max_md_excess,
            'sizing_method': sizing_method,
            'c_rate_limited': final_capacity > required_capacity,
            'safety_factors_applied': True
        }
    
    def _duration_based_sizing(self, energy_requirements, sizing_params):
        """Duration-based battery sizing."""
        max_md_excess = energy_requirements['max_md_excess']
        required_power = max_md_excess
        required_capacity = required_power * sizing_params['duration_hours']
        required_capacity = required_capacity / (sizing_params['depth_of_discharge'] / 100)
        
        # Apply duration safety factor
        duration_safety = sizing_params.get('duration_safety_factor', 25) / 100
        required_capacity *= (1 + duration_safety)
        
        sizing_method = f"Duration-based ({sizing_params['duration_hours']} hours + {sizing_params.get('duration_safety_factor', 25)}% safety)"
        
        # Apply C-rate constraints
        c_rate_capacity = required_power / sizing_params.get('c_rate', 0.5)
        final_capacity = max(required_capacity, c_rate_capacity)
        
        return {
            'capacity_kwh': final_capacity,
            'power_rating_kw': required_power,
            'required_energy_kwh': energy_requirements['total_energy_to_shave'],
            'worst_event_energy_peak_only': energy_requirements['worst_event_energy_peak_only'],
            'max_md_excess': max_md_excess,
            'sizing_method': sizing_method,
            'c_rate_limited': final_capacity > required_capacity,
            'safety_factors_applied': True
        }
    
    def calculate_battery_costs(self, battery_sizing, battery_params):
        """Calculate comprehensive battery system costs."""
        capacity_kwh = battery_sizing['capacity_kwh']
        power_kw = battery_sizing['power_rating_kw']
        
        # CAPEX Components
        battery_cost = capacity_kwh * battery_params['capex_per_kwh']
        pcs_cost = power_kw * battery_params['pcs_cost_per_kw']
        
        # Base system cost
        base_system_cost = battery_cost + pcs_cost
        
        # Total installed cost (including installation, civil works, etc.)
        total_capex = base_system_cost * battery_params['installation_factor']
        
        # Annual OPEX
        annual_opex = total_capex * (battery_params['opex_percent'] / 100)
        
        # Total lifecycle cost
        total_lifecycle_opex = annual_opex * battery_params['battery_life_years']
        total_lifecycle_cost = total_capex + total_lifecycle_opex
        
        return {
            'battery_cost': battery_cost,
            'pcs_cost': pcs_cost,
            'base_system_cost': base_system_cost,
            'total_capex': total_capex,
            'annual_opex': annual_opex,
            'total_lifecycle_opex': total_lifecycle_opex,
            'total_lifecycle_cost': total_lifecycle_cost,
            'cost_per_kwh': total_capex / capacity_kwh,
            'cost_per_kw': total_capex / power_kw
        }
    
    def simulate_battery_operation(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours, selected_tariff=None):
        """
        ★ ENHANCED BATTERY ALGORITHM: Tariff-Aware Discharge Strategy ★
        
        This method implements tariff-aware discharge logic that:
        - General Tariff: Discharges ALL THE TIME when demand exceeds target (24/7 capability)
        - TOU Tariff: Only discharges during RP4 peak periods (Mon-Fri 2PM-10PM, excluding holidays)
        - Validates all discharge events against Malaysia holidays
        - Provides compliance monitoring and violation detection
        - Implements smart charging during appropriate periods
        """
        
        # Import peak logic for RP4 period checking
        from tariffs.peak_logic import get_malaysia_holidays, is_peak_rp4, detect_holidays_from_data
        
        # Import tariff classification logic
        from md_shaving_solution import get_tariff_period_classification
        
        # ★ STEP 1: DETERMINE TARIFF TYPE ★
        tariff_type = selected_tariff.get('Type', '').lower() if selected_tariff else 'general'
        tariff_name = selected_tariff.get('Tariff', '').lower() if selected_tariff else ''
        is_tou_tariff = tariff_type == 'tou' or 'tou' in tariff_name
        
        # Create simulation dataframe
        df_sim = df[[power_col]].copy()
        df_sim['Original_Demand'] = df_sim[power_col]
        df_sim['Target_Demand'] = target_demand
        df_sim['Excess_Demand'] = (df_sim[power_col] - target_demand).clip(lower=0)
        
        # ★ STEP 2: AUTO-DETECT HOLIDAYS FROM DATA ★
        try:
            holidays = detect_holidays_from_data(df, df.index.name or 'timestamp')
        except:
            # Fallback to 2025 holidays if auto-detection fails
            holidays = get_malaysia_holidays(2025)
        
        # ★ STEP 3: ADD TARIFF-AWARE PERIOD CLASSIFICATION ★
        if is_tou_tariff:
            # TOU Tariff: Use RP4 peak period logic (2PM-10PM weekdays)
            df_sim['Is_Peak_Period'] = df_sim.index.to_series().apply(lambda ts: is_peak_rp4(ts, holidays))
            df_sim['Is_RP4_Peak'] = df_sim['Is_Peak_Period']  # Backwards compatibility
        else:
            # General Tariff: All periods allow discharge (24/7 capability)
            df_sim['Is_Peak_Period'] = pd.Series(True, index=df_sim.index)
            df_sim['Is_RP4_Peak'] = df_sim['Is_Peak_Period']  # Backwards compatibility
        
        # ★ STEP 4: COMPLIANCE TRACKING FOR VALIDATION ★
        discharge_violations = []
        potential_violations = []
        
        # Battery state variables
        battery_capacity = battery_sizing['capacity_kwh']
        usable_capacity = battery_capacity * (battery_params['depth_of_discharge'] / 100)
        max_power = battery_sizing['power_rating_kw']
        efficiency = battery_params['round_trip_efficiency'] / 100
        
        # Initialize arrays for performance
        soc = np.zeros(len(df_sim))  # State of Charge in kWh
        soc_percent = np.zeros(len(df_sim))  # SOC as percentage
        battery_power = np.zeros(len(df_sim))  # Positive = discharge, Negative = charge
        net_demand = df_sim[power_col].copy()
        
        # ★ ENHANCED BATTERY OPERATION: TARIFF-AWARE DISCHARGE STRATEGY ★
        initial_soc = usable_capacity * 0.8  # Start at 80% SOC
        
        for i in range(len(df_sim)):
            current_demand = df_sim[power_col].iloc[i]
            current_time = df_sim.index[i]
            excess = max(0, current_demand - target_demand)
            is_peak_period = df_sim['Is_Peak_Period'].iloc[i]
            
            # Determine previous SOC
            previous_soc = initial_soc if i == 0 else soc[i-1]
            
            # ★ CRITICAL DECISION LOGIC: TARIFF-AWARE DISCHARGE ★
            if excess > 0:
                if is_tou_tariff:
                    # TOU Tariff: Only discharge during RP4 peak periods (2PM-10PM weekdays)
                    if is_peak_period:
                        # ✅ COMPLIANT DISCHARGE: TOU Peak Period + Peak Event
                        battery_action = self._calculate_discharge_action(
                            excess, previous_soc, max_power, interval_hours
                        )
                        soc[i] = previous_soc - battery_action * interval_hours
                        battery_power[i] = battery_action
                        net_demand.iloc[i] = current_demand - battery_action
                    else:
                        # ❌ VIOLATION PREVENTION: Peak Event but TOU Off-Peak Period
                        potential_violations.append({
                            'timestamp': current_time,
                            'demand': current_demand,
                            'target': target_demand,
                            'excess': excess,
                            'period_type': 'TOU Off-Peak',
                            'weekday': current_time.strftime('%A'),
                            'hour': current_time.hour,
                            'tariff_type': 'TOU'
                        })
                        # NO DISCHARGE - Let demand exceed target during TOU off-peak
                        soc[i] = previous_soc
                        battery_power[i] = 0
                        net_demand.iloc[i] = current_demand
                else:
                    # General Tariff: Discharge ANYTIME when demand exceeds target (24/7 capability)
                    # ✅ COMPLIANT DISCHARGE: General tariff allows discharge at all times
                    battery_action = self._calculate_discharge_action(
                        excess, previous_soc, max_power, interval_hours
                    )
                    soc[i] = previous_soc - battery_action * interval_hours
                    battery_power[i] = battery_action
                    net_demand.iloc[i] = current_demand - battery_action
            else:
                # ⚡ ENHANCED CHARGING LOGIC: Smart Charging Strategy
                battery_action = self._calculate_enhanced_charge_action(
                    current_demand, previous_soc, max_power, usable_capacity,
                    efficiency, interval_hours, df_sim, i, is_peak_period
                )
                soc[i] = previous_soc + battery_action * interval_hours * efficiency
                battery_power[i] = -battery_action  # Negative for charging
                net_demand.iloc[i] = current_demand + battery_action
            
            # Ensure SOC stays within limits
            soc[i] = max(0, min(soc[i], usable_capacity))
            soc_percent[i] = (soc[i] / usable_capacity) * 100
        
        # Add simulation results to dataframe
        df_sim['Battery_Power_kW'] = battery_power
        df_sim['Battery_SOC_kWh'] = soc
        df_sim['Battery_SOC_Percent'] = soc_percent
        df_sim['Net_Demand_kW'] = net_demand
        df_sim['Peak_Shaved'] = df_sim['Original_Demand'] - df_sim['Net_Demand_kW']
        
        # Add tariff-aware period classification for visualization
        df_sim['Is_Tariff_Peak'] = df_sim['Is_Peak_Period']
        
        # ★ POST-SIMULATION VALIDATION ★
        validation_results = self._validate_tariff_compliance(df_sim, potential_violations, holidays, is_tou_tariff)
        
        # Calculate enhanced performance metrics with compliance data
        simulation_metrics = self._calculate_simulation_metrics(df_sim, target_demand, soc_percent)
        
        # Add validation results to simulation metrics
        simulation_metrics.update({
            'validation_results': validation_results,
            'discharge_violations': potential_violations,
            'holidays_used': len(holidays),
            'tariff_type': 'TOU' if is_tou_tariff else 'General',
            'peak_periods': len(df_sim[df_sim['Is_Peak_Period']]),
            'off_peak_periods': len(df_sim[~df_sim['Is_Peak_Period']]),
            # Backwards compatibility
            'rp4_peak_periods': len(df_sim[df_sim['Is_Peak_Period']]),
            'off_peak_periods': len(df_sim[~df_sim['Is_Peak_Period']])
        })
        
        return simulation_metrics
    
    def _calculate_discharge_action(self, excess, previous_soc, max_power, interval_hours):
        """Calculate optimal discharge action during peak events."""
        # Required discharge power to meet target
        required_discharge = excess
        
        # Check battery constraints
        available_energy = previous_soc
        max_discharge_energy = available_energy
        max_discharge_power = min(
            max_discharge_energy / interval_hours,  # Energy constraint
            max_power,  # Power rating constraint
            required_discharge  # Don't discharge more than needed
        )
        
        return max(0, max_discharge_power)
    
    def _calculate_charge_action(self, current_demand, previous_soc, max_power, 
                                usable_capacity, efficiency, interval_hours, df_sim, i):
        """Calculate optimal charging action during low demand periods."""
        # Improved charging logic with better conditions
        current_time = df_sim.index[i]
        hour = current_time.hour
        
        # Check if battery needs charging (updated to 95% max SOC)
        if previous_soc >= usable_capacity * 0.95:  # Standardized 95% max SOC
            return 0  # Battery is sufficiently charged
        
        # Enhanced charging time windows - allow more charging opportunities
        # Primary: Off-peak hours (22:00-08:00)
        # Secondary: Low demand periods during peak hours
        is_primary_charging_time = hour >= 22 or hour < 8
        
        # Calculate dynamic demand threshold based on recent demand patterns
        # Look at last 24 hours (96 intervals for 15-min data) or available data
        lookback_periods = min(96, len(df_sim))
        start_idx = max(0, i - lookback_periods)
        recent_demand = df_sim['Original_Demand'].iloc[start_idx:i+1]
        
        if len(recent_demand) > 0:
            avg_demand = recent_demand.mean()
            demand_percentile_25 = recent_demand.quantile(0.25)  # 25th percentile
            demand_percentile_50 = recent_demand.quantile(0.50)  # Median
        else:
            avg_demand = df_sim['Original_Demand'].mean()
            demand_percentile_25 = avg_demand * 0.6
            demand_percentile_50 = avg_demand * 0.8
        
        # Determine charging conditions based on demand level and SOC urgency
        soc_percentage = (previous_soc / usable_capacity) * 100
        
        # Very low SOC - charge aggressively even during peak hours (updated to 5% safety limit)
        if soc_percentage < 10:  # Updated from 30% - emergency charging only
            charging_threshold = demand_percentile_50  # Allow charging up to median demand
            charge_rate_factor = 1.0  # Use full power if needed
        # Low SOC - charge during off-peak and low demand
        elif soc_percentage < 60:
            if is_primary_charging_time:
                charging_threshold = avg_demand * 0.85  # More lenient during off-peak
                charge_rate_factor = 0.9
            else:
                charging_threshold = demand_percentile_25  # Only very low demand during peak
                charge_rate_factor = 0.6
        # Normal SOC - conservative charging
        else:
            if is_primary_charging_time:
                charging_threshold = avg_demand * 0.75
                charge_rate_factor = 0.7
            else:
                charging_threshold = demand_percentile_25 * 0.8  # Very conservative during peak
                charge_rate_factor = 0.4
        
        # Check if current demand allows charging
        if current_demand > charging_threshold:
            return 0  # Demand too high for charging
        
        # Calculate optimal charge rate with improved logic
        remaining_capacity = usable_capacity - previous_soc
        max_charge_energy = remaining_capacity / efficiency
        
        # Determine charge power based on urgency and conditions
        base_charge_power = min(
            max_charge_energy / interval_hours,  # Energy constraint
            max_power * charge_rate_factor,  # Dynamic charging rate
            (usable_capacity * 0.95 - previous_soc) / interval_hours / efficiency  # Don't exceed 95% SOC
        )
        
        # Boost charging if SOC is critically low (updated to 5% safety limit)
        if soc_percentage < 10:  # Updated from 20% - emergency boost only
            base_charge_power = min(base_charge_power * 1.2, max_power)  # 20% boost for critical SOC
        
        return max(0, base_charge_power)
    
    def _calculate_enhanced_charge_action(self, current_demand, previous_soc, max_power, 
                                          usable_capacity, efficiency, interval_hours, df_sim, i, is_peak_period):
        """
        ★ ENHANCED CHARGING LOGIC: RP4-Aware Smart Charging ★
        
        This function implements intelligent charging with RP4 period awareness:
        - Prioritizes off-peak periods for aggressive charging
        - Reduces charging during RP4 peak periods to save battery for discharge
        - Implements SOC-based charging urgency
        - Uses dynamic demand thresholds for optimal charging decisions
        """
        
        # Check if battery needs charging (standardized 95% max SOC)
        if previous_soc >= usable_capacity * 0.95:
            return 0  # Battery is sufficiently charged
        
        current_time = df_sim.index[i]
        hour = current_time.hour
        soc_percentage = (previous_soc / usable_capacity) * 100
        
        # Calculate dynamic demand thresholds based on recent patterns
        lookback_periods = min(96, len(df_sim))  # 24 hours of 15-min data or available
        start_idx = max(0, i - lookback_periods)
        recent_demand = df_sim['Original_Demand'].iloc[start_idx:i+1]
        
        if len(recent_demand) > 0:
            avg_demand = recent_demand.mean()
            demand_25th = recent_demand.quantile(0.25)
        else:
            avg_demand = df_sim['Original_Demand'].mean()
            demand_25th = avg_demand * 0.6
        
        # ★ PERIOD-AWARE CHARGING STRATEGY ★
        should_charge = False
        charge_rate_factor = 0.3  # Default conservative rate
        
        # Very low SOC - charge aggressively even during peak hours (updated to 5% safety limit)
        if soc_percentage < 10:  # Updated from 20% - emergency charging only
            should_charge = current_demand < avg_demand * 1.0  # Very lenient threshold
            charge_rate_factor = 0.9  # High charge rate
            
        # Low SOC - prioritize charging during off-peak
        elif soc_percentage < 50:
            if not is_peak_period:  # Off-peak hours - charge aggressively
                should_charge = current_demand < avg_demand * 0.85
                charge_rate_factor = 0.7
            else:  # Peak hours - only charge during very low demand
                should_charge = current_demand < demand_25th * 0.9
                charge_rate_factor = 0.3
                
        # Normal SOC - conservative charging, prioritize off-peak (standardized 95% max SOC)
        elif soc_percentage < 95:  # Updated from 80% to 95%
            if not is_peak_period:  # Off-peak hours
                should_charge = current_demand < avg_demand * 0.75
                charge_rate_factor = 0.6
            else:  # Peak hours - very selective
                should_charge = current_demand < demand_25th * 0.8
                charge_rate_factor = 0.2
        
        # High SOC - minimal charging, off-peak only (removed - now handled by 95% limit check above)
        else:
            return 0  # No charging above 95% SOC
        
        if not should_charge:
            return 0
        
        # Calculate optimal charge power with enhanced logic
        remaining_capacity = usable_capacity * 0.95 - previous_soc
        max_charge_energy = remaining_capacity / efficiency
        
        charge_power = min(
            max_power * charge_rate_factor,  # Dynamic charging rate
            max_charge_energy / interval_hours,  # Energy constraint
            (usable_capacity * 0.95 - previous_soc) / interval_hours / efficiency  # Don't exceed 95% SOC
        )
        
        return max(0, charge_power)
    
    def _calculate_simulation_metrics(self, df_sim, target_demand, soc_percent):
        """Calculate comprehensive simulation performance metrics."""
        # Get dynamic interval for energy calculations
        interval_hours = _get_dynamic_interval_hours(df_sim)
        
        # Energy metrics with more detailed analysis
        charging_intervals = df_sim['Battery_Power_kW'] < 0
        discharging_intervals = df_sim['Battery_Power_kW'] > 0
        
        total_energy_discharged = sum([p * interval_hours for p in df_sim['Battery_Power_kW'] if p > 0])
        total_energy_charged = sum([abs(p) * interval_hours for p in df_sim['Battery_Power_kW'] if p < 0])
        
        # Charging/Discharging cycle analysis
        charging_cycles = len(df_sim[charging_intervals])
        discharging_cycles = len(df_sim[discharging_intervals])
        total_intervals = len(df_sim)
        
        # Peak reduction
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
        
        # Success rate analysis
        successful_shaves = len(df_sim[
            (df_sim['Original_Demand'] > target_demand) & 
            (df_sim['Net_Demand_kW'] <= target_demand * 1.05)  # Allow 5% tolerance
        ])
        
        total_peak_events = len(df_sim[df_sim['Original_Demand'] > target_demand])
        success_rate = (successful_shaves / total_peak_events * 100) if total_peak_events > 0 else 0
        
        # Calculate average charging and discharging power
        avg_charge_power = df_sim[charging_intervals]['Battery_Power_kW'].abs().mean() if charging_cycles > 0 else 0
        avg_discharge_power = df_sim[discharging_intervals]['Battery_Power_kW'].mean() if discharging_cycles > 0 else 0
        
        return {
            'df_simulation': df_sim,
            'total_energy_discharged': total_energy_discharged,
            'total_energy_charged': total_energy_charged,
            'peak_reduction_kw': peak_reduction,
            'success_rate_percent': success_rate,
            'successful_shaves': successful_shaves,
            'total_peak_events': total_peak_events,
            'average_soc': np.mean(soc_percent),
            'min_soc': np.min(soc_percent),
            'max_soc': np.max(soc_percent),
            'energy_efficiency': (total_energy_discharged / total_energy_charged * 100) if total_energy_charged > 0 else 0,
            # Enhanced metrics for charging analysis
            'charging_intervals': charging_cycles,
            'discharging_intervals': discharging_cycles,
            'charging_percentage': (charging_cycles / total_intervals * 100) if total_intervals > 0 else 0,
            'discharging_percentage': (discharging_cycles / total_intervals * 100) if total_intervals > 0 else 0,
            'avg_charge_power': avg_charge_power,
            'avg_discharge_power': avg_discharge_power,
            'charge_discharge_ratio': (total_energy_charged / total_energy_discharged) if total_energy_discharged > 0 else 0
        }
    
    def calculate_financial_metrics(self, battery_costs, event_summaries, total_md_rate, battery_params):
        """Calculate comprehensive financial metrics including ROI, IRR, and NPV."""
        # Calculate annual MD savings
        if event_summaries and total_md_rate > 0:
            max_monthly_md_saving = max(event['MD Cost Impact (RM)'] for event in event_summaries)
            annual_md_savings = max_monthly_md_saving * 12
        else:
            annual_md_savings = 0
        
        # Additional potential savings (energy arbitrage, ancillary services, etc.)
        total_annual_savings = annual_md_savings  # Focus on MD savings for now
        
        # Calculate simple payback
        if total_annual_savings > battery_costs['annual_opex']:
            net_annual_savings = total_annual_savings - battery_costs['annual_opex']
            simple_payback_years = battery_costs['total_capex'] / net_annual_savings
        else:
            simple_payback_years = float('inf')
        
        # Calculate NPV and IRR
        discount_rate = battery_params['discount_rate'] / 100
        project_years = battery_params['battery_life_years']
        
        # Cash flows: Initial investment (negative), then annual net savings
        cash_flows = [-battery_costs['total_capex']]  # Initial investment
        for year in range(1, project_years + 1):
            annual_net_cash_flow = total_annual_savings - battery_costs['annual_opex']
            cash_flows.append(annual_net_cash_flow)
        
        # Calculate NPV
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        
        # Calculate IRR (simplified approximation)
        irr = self._calculate_irr_approximation(cash_flows)
        
        # Calculate profitability metrics
        total_lifecycle_savings = total_annual_savings * project_years
        total_lifecycle_costs = battery_costs['total_capex'] + battery_costs['total_lifecycle_opex']
        benefit_cost_ratio = total_lifecycle_savings / total_lifecycle_costs if total_lifecycle_costs > 0 else 0
        
        # Calculate simple annual ROI based on net annual savings
        # ROI% = (Net Annual Savings / Total CAPEX) × 100
        net_annual_savings = total_annual_savings - battery_costs['annual_opex']
        annual_roi_percent = (net_annual_savings / battery_costs['total_capex'] * 100) if battery_costs['total_capex'] > 0 else 0
        
        return {
            'annual_md_savings': annual_md_savings,
            'total_annual_savings': total_annual_savings,
            'net_annual_savings': net_annual_savings,
            'simple_payback_years': simple_payback_years,
            'npv': npv,
            'irr_percent': irr * 100 if irr is not None else None,
            'benefit_cost_ratio': benefit_cost_ratio,
            'total_lifecycle_savings': total_lifecycle_savings,
            'roi_percent': annual_roi_percent,
            'cash_flows': cash_flows
        }
    
    def _calculate_irr_approximation(self, cash_flows):
        """Calculate IRR using Newton-Raphson approximation method."""
        try:
            def npv_at_rate(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            
            # Binary search for IRR between 0% and 100%
            low, high = 0, 1
            tolerance = 1e-6
            max_iterations = 100
            
            for _ in range(max_iterations):
                mid = (low + high) / 2
                npv_mid = npv_at_rate(mid)
                
                if abs(npv_mid) < tolerance:
                    return mid
                
                if npv_mid > 0:
                    low = mid
                else:
                    high = mid
            
            return mid if abs(npv_at_rate(mid)) < tolerance else None
            
        except:
            return None
    
    def optimize_battery_schedule(self, df, power_col, target_demand, battery_sizing, 
                                 battery_params, interval_hours, optimization_strategy='aggressive'):
        """
        Advanced battery optimization algorithm with different strategies.
        
        Args:
            optimization_strategy: 'conservative', 'balanced', 'aggressive'
        """
        if optimization_strategy == 'conservative':
            return self._conservative_optimization(df, power_col, target_demand, battery_sizing, battery_params, interval_hours)
        elif optimization_strategy == 'balanced':
            return self._balanced_optimization(df, power_col, target_demand, battery_sizing, battery_params, interval_hours)
        elif optimization_strategy == 'aggressive':
            return self._aggressive_optimization(df, power_col, target_demand, battery_sizing, battery_params, interval_hours)
        else:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")
    
    def _conservative_optimization(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """Conservative battery operation - prioritize battery life and safety margins."""
        # Implement conservative charging/discharging with larger safety margins
        # Lower depth of discharge, slower charging rates, higher SOC targets
        modified_params = battery_params.copy()
        modified_params['depth_of_discharge'] = min(80, battery_params['depth_of_discharge'])  # Max 80% DoD
        
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, modified_params, interval_hours, selected_tariff=None)
    
    def _balanced_optimization(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """Balanced battery operation - standard operation parameters."""
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, battery_params, interval_hours, selected_tariff=None)
    
    def _aggressive_optimization(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """Aggressive battery operation - maximize MD savings with higher utilization."""
        # Implement aggressive charging/discharging with tighter margins
        # Higher depth of discharge, faster charging rates, lower SOC targets
        modified_params = battery_params.copy()
        modified_params['depth_of_discharge'] = min(95, battery_params['depth_of_discharge'] + 5)  # Increase DoD by 5%
        
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, modified_params, interval_hours, selected_tariff=None)
    
    def _validate_tariff_compliance(self, df_sim, potential_violations, holidays, is_tou_tariff):
        """
        ★ COMPLIANCE VALIDATION: Tariff-Aware Discharge Verification ★
        
        This function validates that battery discharge occurs according to tariff rules:
        - TOU Tariff: Discharge only during peak periods (Mon-Fri 2PM-10PM, excluding holidays)
        - General Tariff: Discharge allowed anytime when demand exceeds target (24/7)
        
        Args:
            df_sim: Simulation results dataframe
            potential_violations: List of periods where demand exceeded target but battery didn't discharge
            holidays: List of holiday dates
            is_tou_tariff: Boolean indicating if this is a TOU tariff
            
        Returns:
            Dictionary with compliance analysis results
        """
        # Get dynamic interval for energy calculations
        interval_hours = _get_dynamic_interval_hours(df_sim)
        
        # Count total discharge events
        discharge_events = df_sim[df_sim['Battery_Power_kW'] > 0]
        total_discharges = len(discharge_events)
        
        if is_tou_tariff:
            # TOU Tariff: Validate discharge only during peak periods
            compliant_discharges = len(discharge_events[discharge_events['Is_Peak_Period']])
            violation_discharges = total_discharges - compliant_discharges
            
            # Energy analysis
            total_energy_discharged = discharge_events['Battery_Power_kW'].sum() * interval_hours
            compliant_energy = discharge_events[discharge_events['Is_Peak_Period']]['Battery_Power_kW'].sum() * interval_hours
            violation_energy = total_energy_discharged - compliant_energy
            
            compliance_rule = "TOU: Discharge only during peak periods (Mon-Fri 2PM-10PM, excluding holidays)"
        else:
            # General Tariff: All discharges are compliant (24/7 discharge allowed)
            compliant_discharges = total_discharges
            violation_discharges = 0
            
            # Energy analysis
            total_energy_discharged = discharge_events['Battery_Power_kW'].sum() * interval_hours
            compliant_energy = total_energy_discharged
            violation_energy = 0
            
            compliance_rule = "General: Discharge allowed anytime when demand exceeds target (24/7)"
        
        # Calculate compliance rate
        compliance_rate = (compliant_discharges / total_discharges * 100) if total_discharges > 0 else 100
        
        return {
            'total_discharges': total_discharges,
            'compliant_discharges': compliant_discharges,
            'violation_discharges': violation_discharges,
            'compliance_rate': compliance_rate,
            'total_energy_discharged': total_energy_discharged,
            'compliant_energy': compliant_energy,
            'violation_energy': violation_energy,
            'potential_violations_logged': len(potential_violations),  # Demand exceeded target but no discharge
            'is_fully_compliant': violation_discharges == 0,
            'holidays_count': len(holidays),
            'tariff_type': 'TOU' if is_tou_tariff else 'General',
            'compliance_rule': compliance_rule
        }
    
    def _validate_peak_period_compliance(self, df_sim, potential_violations, holidays):
        """
        ★ LEGACY COMPLIANCE VALIDATION: RP4 Peak-Period-Only Discharge Verification ★
        
        DEPRECATED: Use _validate_tariff_compliance instead.
        This function is kept for backwards compatibility.
        """
        # Call the new tariff-aware validation with TOU settings for backwards compatibility
        return self._validate_tariff_compliance(df_sim, potential_violations, holidays, is_tou_tariff=True)
    

# Factory function to create battery algorithm instances
def create_battery_algorithms():
    """Factory function to create a BatteryAlgorithms instance."""
    return BatteryAlgorithms()


# Utility functions for battery parameter configuration
def get_battery_parameters_ui(event_summaries=None):
    """
    Create Streamlit UI for battery parameter configuration.
    This function maintains the existing UI interface while using the new algorithms.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔋 Battery System Parameters")
    
    # Calculate defaults from event summaries
    default_capacity = 500
    default_power = 250
    
    if event_summaries:
        # Get maximum energy to shave (peak period only) and maximum MD excess
        max_energy_peak_only = max(event.get('TOU Required Energy (kWh)', 0) for event in event_summaries)
        max_md_excess = max(event.get('TOU Excess (kW)', 0) for event in event_summaries if event.get('TOU Excess (kW)', 0) > 0)
        
        if max_energy_peak_only > 0:
            default_capacity = max_energy_peak_only
        if max_md_excess > 0:
            default_power = max_md_excess
    
    with st.sidebar.expander("⚙️ BESS Configuration", expanded=False):
        battery_params = {}
        
        # Battery Technology
        battery_params['technology'] = st.selectbox(
            "Battery Technology",
            ["Lithium-ion (Li-ion)", "Lithium Iron Phosphate (LiFePO4)", "Sodium-ion"],
            index=1,  # Default to LiFePO4
            help="Different battery technologies have different costs and characteristics"
        )
        
        # System Sizing Approach
        battery_params['sizing_approach'] = st.selectbox(
            "Sizing Approach",
            ["Auto-size for Peak Events", "Manual Capacity", "Energy Duration-based"],
            help="Choose how to determine the battery capacity"
        )
        
        # Add sizing-specific parameters based on approach
        if battery_params['sizing_approach'] == "Manual Capacity":
            # Manual sizing parameters
            st.markdown("**Manual Battery Sizing with Safety Factors**")
            
            capacity_safety_factor = st.slider("Capacity Safety Factor (%)", 0, 100, 20, 5)
            power_safety_factor = st.slider("Power Rating Safety Factor (%)", 0, 100, 15, 5)
            
            suggested_capacity = default_capacity * (1 + capacity_safety_factor / 100)
            suggested_power = default_power * (1 + power_safety_factor / 100)
            
            battery_params['manual_capacity_kwh'] = st.number_input(
                "Battery Capacity (kWh)", 10, 10000, int(suggested_capacity), 10)
            battery_params['manual_power_kw'] = st.number_input(
                "Battery Power Rating (kW)", 10, 5000, int(suggested_power), 10)
            battery_params['capacity_safety_factor'] = capacity_safety_factor
            battery_params['power_safety_factor'] = power_safety_factor
            
        elif battery_params['sizing_approach'] == "Energy Duration-based":
            # Duration-based parameters
            st.markdown("**Duration-based Sizing with Safety Factor**")
            
            battery_params['duration_hours'] = st.number_input("Discharge Duration (hours)", 0.5, 8.0, 2.0, 0.5)
            battery_params['duration_safety_factor'] = st.slider("Duration Safety Factor (%)", 0, 100, 25, 5)
            
        else:  # Auto-size for Peak Events
            # Auto-sizing parameters
            st.markdown("**Auto-sizing Safety Factors**")
            
            battery_params['auto_capacity_safety'] = st.slider("Auto-sizing Capacity Safety (%)", 10, 50, 20, 5)
            battery_params['auto_power_safety'] = st.slider("Auto-sizing Power Safety (%)", 10, 50, 15, 5)
        
        # System specifications
        st.markdown("**System Specifications**")
        battery_params['depth_of_discharge'] = st.slider("Depth of Discharge (%)", 70, 95, 85, 5)
        battery_params['round_trip_efficiency'] = st.slider("Round-trip Efficiency (%)", 85, 98, 92, 1)
        battery_params['discharge_efficiency'] = st.slider("Discharge Efficiency (%)", 85, 98, 94, 1, 
            help="Energy delivered to load during discharge (used in battery sizing)")
        battery_params['c_rate'] = st.slider("C-Rate (Charge/Discharge)", 0.2, 2.0, 0.5, 0.1)
        
        # Financial parameters
        st.markdown("**Financial Parameters**")
        battery_params['capex_per_kwh'] = st.number_input("Battery Cost (RM/kWh)", 500, 3000, 1200, 50)
        battery_params['pcs_cost_per_kw'] = st.number_input("Power Conversion System (RM/kW)", 200, 1000, 400, 25)
        battery_params['installation_factor'] = st.slider("Installation & Integration Factor", 1.1, 2.0, 1.4, 0.1)
        battery_params['opex_percent'] = st.slider("Annual O&M (% of CAPEX)", 1.0, 8.0, 3.0, 0.5)
        battery_params['battery_life_years'] = st.number_input("Battery Life (years)", 5, 25, 15, 1)
        battery_params['discount_rate'] = st.slider("Discount Rate (%)", 3.0, 15.0, 8.0, 0.5)
    
    return battery_params


def perform_comprehensive_battery_analysis(df, power_col, event_summaries, target_demand, 
                                          interval_hours, battery_params, total_md_rate, selected_tariff=None, holidays=None):
    """
    Perform comprehensive battery analysis using the new algorithms.
    This function coordinates all battery analysis steps.
    """
    # Create battery algorithms instance
    battery_algo = create_battery_algorithms()
    
    # Calculate battery sizing
    battery_sizing = battery_algo.calculate_optimal_sizing(
        event_summaries, target_demand, interval_hours, battery_params
    )
    
    # Calculate costs
    battery_costs = battery_algo.calculate_battery_costs(battery_sizing, battery_params)
    
    # Simulate battery operation
    battery_simulation = battery_algo.simulate_battery_operation(
        df, power_col, target_demand, battery_sizing, battery_params, interval_hours, selected_tariff
    )
    
    # Calculate financial metrics
    financial_analysis = battery_algo.calculate_financial_metrics(
        battery_costs, event_summaries, total_md_rate, battery_params
    )
    
    return {
        'sizing': battery_sizing,
        'costs': battery_costs,
        'simulation': battery_simulation,
        'financial': financial_analysis,
        'params': battery_params,
        'algorithms': battery_algo  # Include reference to algorithms for advanced operations
    }
