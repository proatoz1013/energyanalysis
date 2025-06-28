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
            # Use Energy to Shave (Peak Period Only) for capacity sizing
            energy_kwh_peak_only = event.get('Energy to Shave (Peak Period Only)', 0)
            # Use MD Excess (kW) for power sizing
            md_excess_power = event.get('MD Excess (kW)', 0)
            
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
    
    def simulate_battery_operation(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """
        Advanced battery charge/discharge simulation algorithm.
        
        This method implements a sophisticated battery operation simulation that considers:
        - State of charge management
        - Charge/discharge efficiency
        - Power rating constraints
        - Intelligent charging strategies
        - Peak shaving optimization
        """
        # Create simulation dataframe
        df_sim = df[[power_col]].copy()
        df_sim['Original_Demand'] = df_sim[power_col]
        df_sim['Target_Demand'] = target_demand
        df_sim['Excess_Demand'] = (df_sim[power_col] - target_demand).clip(lower=0)
        
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
        
        # Advanced battery management algorithm
        initial_soc = usable_capacity * 0.8  # Start at 80% SOC
        
        for i in range(len(df_sim)):
            current_demand = df_sim[power_col].iloc[i]
            excess = max(0, current_demand - target_demand)
            
            # Determine previous SOC
            previous_soc = initial_soc if i == 0 else soc[i-1]
            
            if excess > 0:  # Peak shaving mode - discharge battery
                battery_action = self._calculate_discharge_action(
                    excess, previous_soc, max_power, interval_hours
                )
                soc[i] = previous_soc - battery_action * interval_hours
                battery_power[i] = battery_action
                net_demand.iloc[i] = current_demand - battery_action
                
            else:  # Charging opportunity
                battery_action = self._calculate_charge_action(
                    current_demand, previous_soc, max_power, usable_capacity, 
                    efficiency, interval_hours, df_sim, i
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
        
        # Calculate performance metrics
        return self._calculate_simulation_metrics(df_sim, target_demand, soc_percent)
    
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
        # Intelligent charging logic
        current_time = df_sim.index[i]
        hour = current_time.hour
        
        # Only charge during off-peak hours (22:00-08:00)
        is_charging_time = hour >= 22 or hour < 8
        
        if not is_charging_time or previous_soc >= usable_capacity * 0.95:
            return 0  # No charging
        
        # Charge when demand is significantly below average
        avg_demand = df_sim['Original_Demand'].mean()
        charging_threshold = avg_demand * 0.7
        
        if current_demand > charging_threshold:
            return 0  # Demand too high for charging
        
        # Calculate optimal charge rate
        remaining_capacity = usable_capacity - previous_soc
        max_charge_energy = remaining_capacity / efficiency
        max_charge_power = min(
            max_charge_energy / interval_hours,  # Energy constraint
            max_power * 0.8,  # Conservative charging rate (80% of max power)
            (usable_capacity * 0.95 - previous_soc) / interval_hours / efficiency  # Don't exceed 95% SOC
        )
        
        return max(0, max_charge_power)
    
    def _calculate_simulation_metrics(self, df_sim, target_demand, soc_percent):
        """Calculate comprehensive simulation performance metrics."""
        # Energy metrics
        total_energy_discharged = sum([p * 0.25 for p in df_sim['Battery_Power_kW'] if p > 0])  # Assuming 15-min intervals
        total_energy_charged = sum([abs(p) * 0.25 for p in df_sim['Battery_Power_kW'] if p < 0])
        
        # Peak reduction
        peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
        
        # Success rate analysis
        successful_shaves = len(df_sim[
            (df_sim['Original_Demand'] > target_demand) & 
            (df_sim['Net_Demand_kW'] <= target_demand * 1.05)  # Allow 5% tolerance
        ])
        
        total_peak_events = len(df_sim[df_sim['Original_Demand'] > target_demand])
        success_rate = (successful_shaves / total_peak_events * 100) if total_peak_events > 0 else 0
        
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
            'energy_efficiency': (total_energy_discharged / total_energy_charged * 100) if total_energy_charged > 0 else 0
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
        
        return {
            'annual_md_savings': annual_md_savings,
            'total_annual_savings': total_annual_savings,
            'net_annual_savings': total_annual_savings - battery_costs['annual_opex'],
            'simple_payback_years': simple_payback_years,
            'npv': npv,
            'irr_percent': irr * 100 if irr is not None else None,
            'benefit_cost_ratio': benefit_cost_ratio,
            'total_lifecycle_savings': total_lifecycle_savings,
            'roi_percent': (npv / battery_costs['total_capex'] * 100) if battery_costs['total_capex'] > 0 else 0,
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
        
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, modified_params, interval_hours)
    
    def _balanced_optimization(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """Balanced battery operation - standard operation parameters."""
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, battery_params, interval_hours)
    
    def _aggressive_optimization(self, df, power_col, target_demand, battery_sizing, battery_params, interval_hours):
        """Aggressive battery operation - maximize MD savings with higher utilization."""
        # Implement aggressive charging/discharging with tighter margins
        # Higher depth of discharge, faster charging rates, lower SOC targets
        modified_params = battery_params.copy()
        modified_params['depth_of_discharge'] = min(95, battery_params['depth_of_discharge'] + 5)  # Increase DoD by 5%
        
        return self.simulate_battery_operation(df, power_col, target_demand, battery_sizing, modified_params, interval_hours)


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
        max_energy_peak_only = max(event.get('Energy to Shave (Peak Period Only)', 0) for event in event_summaries)
        max_md_excess = max(event.get('MD Excess (kW)', 0) for event in event_summaries if event.get('MD Excess (kW)', 0) > 0)
        
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
                                          interval_hours, battery_params, total_md_rate):
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
        df, power_col, target_demand, battery_sizing, battery_params, interval_hours
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
