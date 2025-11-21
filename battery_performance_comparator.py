# battery_performance_comparator.py

"""
Battery Performance Comparator Module

Compares the performance of three battery discharge strategies:
1. Default Shaving (execute_default_shaving_discharge)
2. Simple Conservation (execute_conservation_discharge)
3. Smart Conservation (MainSmartShaving with severity-based mode switching)

Generates comparison tables showing:
- Timestamp, Current Power, Target Power, Is Event, Excess Demand
- Demand shaved by each method
- Status (Success/Failed/Partial) for each method
- Savings lost for each method
"""

from smart_battery_executor import (
    execute_default_shaving_discharge,
    execute_conservation_discharge,
    execute_battery_recharge
)
from smart_conservation import MdOrchestrator, MdExcess, SmartConstants, TriggerEvents
import pandas as pd

# ============================================================================
# GLOBAL CONSTANTS - Battery Operation Defaults
# ============================================================================
# These defaults match the standard values used across the V3 system
# and provide fallbacks when parameters are not in config_data

DEFAULT_DISCHARGE_EFFICIENCY = 0.95  # 95% discharge efficiency (industry standard)
DEFAULT_CHARGE_EFFICIENCY = 0.95     # 95% charge efficiency
DEFAULT_SOC_MIN_PERCENT = 5.0        # 5% minimum SOC (safety limit)
DEFAULT_SOC_MAX_PERCENT = 95.0       # 95% maximum SOC (battery health)
DEFAULT_C_RATE = 1.0                 # 1C rate (charge/discharge in 1 hour)

class BatteryPerformanceComparator:
    """
    Performance comparison framework for battery discharge strategies.
    
    This class runs three different battery strategies on the same dataset
    and compares their effectiveness in terms of:
    - Demand reduction achieved
    - SOC preservation
    - Success rate (full shaving vs partial)
    - Energy efficiency
    """
    
    def __init__(self, config_data, initial_soc_percent=95.0):
        """
        Initialize comparator with configuration and starting SOC.
        
        Args:
            config_data (dict): Configuration with df_sim, battery params, etc.
            initial_soc_percent (float): Starting SOC for all three methods
        """
        self.config_data = config_data
        self.initial_soc_percent = initial_soc_percent
        
        # Extract common parameters
        self.df_sim = config_data['df_sim'].copy()
        self.power_col = config_data['power_col']
        
        # Results storage
        self.results = {
            'default_shaving': None,
            'simple_conservation': None,
            'smart_conservation': None
        }
    
    def run_all_methods(self):
        """
        Execute all three battery strategies and collect results.
        
        Returns:
            dict: Results from all three methods with performance metrics
        """
        # 1. Run default shaving
        self.results['default_shaving'] = self._run_default_shaving()
        
        # 2. Run simple conservation (fixed conservation mode)
        self.results['simple_conservation'] = self._run_simple_conservation()
        
        # 3. Run smart conservation (severity-based mode switching)
        self.results['smart_conservation'] = self._run_smart_conservation()
        
        return self.results

    def _detect_events_with_tou_logic(self, result_df, excess_demand):
        """
        Detect events following TOU logic from smart_conservation.py.
        
        Events are active when BOTH conditions are met:
        1. excess_demand > 0 (current demand exceeds target)
        2. inside_md_window = True (currently in MD recording period)
        
        For TOU tariffs: MD window is only 2PM-10PM weekdays (excluding holidays)
        For General tariffs: MD window is 24/7
        
        This method reuses the existing SmartConstants.is_md_active() and 
        TriggerEvents.set_event_state() from smart_conservation.py.
        
        Args:
            result_df: DataFrame with timestamps as index
            excess_demand: Series with excess demand values
            
        Returns:
            pd.Series: Boolean series indicating which timestamps are events
        """
        # Initialize SmartConstants for MD window checking
        sc = SmartConstants()
        trigger_events = TriggerEvents()
        
        # Create is_event series (default all False)
        is_event_series = pd.Series(False, index=result_df.index)
        
        # Process each timestamp
        for idx in result_df.index:
            current_excess = excess_demand.loc[idx]
            
            # Check if we're inside MD window (respects TOU tariff logic)
            inside_md_window = sc.is_md_active(idx, self.config_data)
            
            # Use TriggerEvents logic: event active when excess > 0 AND inside MD window
            from smart_conservation import _MdEventState
            temp_event_state = _MdEventState()
            is_active = trigger_events.set_event_state(
                current_excess=current_excess,
                inside_md_window=inside_md_window,
                event_state=temp_event_state
            )
            
            is_event_series.loc[idx] = is_active
        
        return is_event_series

    def create_comparison_table(self, max_rows=100):
        """
        Create comparison table matching your Excel format.
        
        Returns:
            pd.DataFrame: Comparison table with columns:
                - timestamp, current power, target power, is event, excess demand
                - demand_shaved_by_default_mode, Status, savings lost
                - demand_shaved_by_simple_conservation_mode, Status, savings lost
                - demand_shaved_by_smart_conservation_mode, Status, savings lost
        """
        import pandas as pd
        
        # Validate that all three methods have been run
        if not all([
            self.results['default_shaving'] is not None,
            self.results['simple_conservation'] is not None,
            self.results['smart_conservation'] is not None
        ]):
            return pd.DataFrame({
                'error': ['Please run run_all_methods() first before creating comparison table']
            })
        
        
        # Extract results from each method
        default_df = self.results['default_shaving']
        simple_df = self.results['simple_conservation']
        smart_df = self.results['smart_conservation']
        
        # Get base columns from config
        power_col = self.config_data['power_col']
        target_series = self.config_data['target_series']
        
        # Build comparison data
        comparison_data = []
        
        # Limit to max_rows for display
        timestamps = default_df.index[:max_rows] if len(default_df) > max_rows else default_df.index
        
        for idx in timestamps:
            # Base information (same for all methods)
            current_power = default_df.loc[idx, power_col] if power_col in default_df.columns else 0
            target_power = target_series.loc[idx] if idx in target_series.index else 0
            excess_demand = default_df.loc[idx, 'excess_demand_kw']
            is_event = default_df.loc[idx, 'is_event']
            
            # Default shaving results
            default_net_demand = default_df.loc[idx, 'net_demand_kw']
            default_shaved = current_power - default_net_demand
            default_status, default_savings_lost = self._calculate_status_and_savings_lost(
                current_power, target_power, default_net_demand, is_event
            )
            
            # Simple conservation results
            simple_net_demand = simple_df.loc[idx, 'net_demand_kw']
            simple_shaved = current_power - simple_net_demand
            simple_status, simple_savings_lost = self._calculate_status_and_savings_lost(
                current_power, target_power, simple_net_demand, is_event
            )
            
            # Smart conservation results
            smart_net_demand = smart_df.loc[idx, 'net_demand_kw']
            smart_shaved = current_power - smart_net_demand
            smart_status, smart_savings_lost = self._calculate_status_and_savings_lost(
                current_power, target_power, smart_net_demand, is_event
            )
            
            # Build row
            row = {
                'timestamp': idx,
                'current_power_kw': round(current_power, 2),
                'target_power_kw': round(target_power, 2),
                'is_event': 'Yes' if is_event else 'No',
                'excess_demand_kw': round(excess_demand, 2),
                
                # Default mode columns
                'demand_shaved_by_default_mode_kw': round(default_shaved, 2),
                'default_status': default_status,
                'default_savings_lost_kw': round(default_savings_lost, 2),
                
                # Simple conservation mode columns
                'demand_shaved_by_simple_conservation_mode_kw': round(simple_shaved, 2),
                'simple_conservation_status': simple_status,
                'simple_conservation_savings_lost_kw': round(simple_savings_lost, 2),
                
                # Smart conservation mode columns
                'demand_shaved_by_smart_conservation_mode_kw': round(smart_shaved, 2),
                'smart_conservation_status': smart_status,
                'smart_conservation_savings_lost_kw': round(smart_savings_lost, 2)
            }
            
            comparison_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add summary statistics
        summary_stats = self._calculate_comparison_summary(comparison_df)
        
        # Store summary in instance for access
        self.comparison_summary = summary_stats
        
        return comparison_df
    
    def _calculate_status_and_savings_lost(self, current_power, target_power, net_demand, is_event):
        """
        Calculate status (Success/Failed/Partial) and savings lost for a single timestamp.
        
        Args:
            current_power: Original power demand before battery
            target_power: Target power limit
            net_demand: Power demand after battery operation
            is_event: Whether this timestamp is classified as an event
            
        Returns:
            tuple: (status_str, savings_lost_kw)
        """
        if not is_event or current_power <= target_power:
            # Not an event or already within target
            return 'N/A', 0.0
        
        excess_demand = current_power - target_power
        demand_shaved = current_power - net_demand
        
        if net_demand <= target_power:
            # Success: brought demand to or below target
            return 'Success', 0.0
        elif demand_shaved > 0:
            # Partial: reduced some demand but didn't reach target
            savings_lost = net_demand - target_power
            return 'Partial', savings_lost
        else:
            # Failed: no demand reduction
            return 'Failed', excess_demand
    
    def _calculate_comparison_summary(self, comparison_df):
        """
        Calculate summary statistics comparing all three methods.
        
        Args:
            comparison_df: DataFrame with comparison results
            
        Returns:
            dict: Summary statistics for all three methods
        """
        summary = {
            'total_rows': len(comparison_df),
            'event_rows': (comparison_df['is_event'] == 'Yes').sum(),
            
            # Default mode summary
            'default': {
                'total_shaved_kwh': comparison_df['demand_shaved_by_default_mode_kw'].sum() * self.config_data.get('interval_hours', 0.5),
                'success_count': (comparison_df['default_status'] == 'Success').sum(),
                'partial_count': (comparison_df['default_status'] == 'Partial').sum(),
                'failed_count': (comparison_df['default_status'] == 'Failed').sum(),
                'total_savings_lost_kwh': comparison_df['default_savings_lost_kw'].sum() * self.config_data.get('interval_hours', 0.5)
            },
            
            # Simple conservation summary
            'simple_conservation': {
                'total_shaved_kwh': comparison_df['demand_shaved_by_simple_conservation_mode_kw'].sum() * self.config_data.get('interval_hours', 0.5),
                'success_count': (comparison_df['simple_conservation_status'] == 'Success').sum(),
                'partial_count': (comparison_df['simple_conservation_status'] == 'Partial').sum(),
                'failed_count': (comparison_df['simple_conservation_status'] == 'Failed').sum(),
                'total_savings_lost_kwh': comparison_df['simple_conservation_savings_lost_kw'].sum() * self.config_data.get('interval_hours', 0.5)
            },
            
            # Smart conservation summary
            'smart_conservation': {
                'total_shaved_kwh': comparison_df['demand_shaved_by_smart_conservation_mode_kw'].sum() * self.config_data.get('interval_hours', 0.5),
                'success_count': (comparison_df['smart_conservation_status'] == 'Success').sum(),
                'partial_count': (comparison_df['smart_conservation_status'] == 'Partial').sum(),
                'failed_count': (comparison_df['smart_conservation_status'] == 'Failed').sum(),
                'total_savings_lost_kwh': comparison_df['smart_conservation_savings_lost_kw'].sum() * self.config_data.get('interval_hours', 0.5)
            }
        }
        
        # Calculate performance percentages
        for method in ['default', 'simple_conservation', 'smart_conservation']:
            event_count = summary['event_rows']
            if event_count > 0:
                summary[method]['success_rate'] = (summary[method]['success_count'] / event_count) * 100
                summary[method]['partial_rate'] = (summary[method]['partial_count'] / event_count) * 100
                summary[method]['failed_rate'] = (summary[method]['failed_count'] / event_count) * 100
            else:
                summary[method]['success_rate'] = 0
                summary[method]['partial_rate'] = 0
                summary[method]['failed_rate'] = 0
        
        return summary
    
    def display_comparison_with_formatting(self, max_rows=100):
        """
        Create and display comparison table with Streamlit-compatible formatting.
        
        Uses SmartConservationDebugger infrastructure for consistent display.
        
        Returns:
            dict: Formatted display result compatible with Streamlit
        """
        # Create comparison table
        comparison_df = self.create_comparison_table(max_rows=max_rows)
        
        if 'error' in comparison_df.columns:
            return {
                'dataframe': comparison_df,
                'summary': {'error': 'Methods not run yet'},
                'metadata': {'error': 'Run run_all_methods() first'}
            }
        
        # Apply conditional formatting for Status columns
        def style_status(val):
            if val == 'Success':
                return 'background-color: #90EE90'  # Light green
            elif val == 'Partial':
                return 'background-color: #FFD700'  # Gold
            elif val == 'Failed':
                return 'background-color: #FF6B6B'  # Light red
            else:
                return ''
        
        # Create styled dataframe
        styled_df = comparison_df.style.applymap(
            style_status,
            subset=['default_status', 'simple_conservation_status', 'smart_conservation_status']
        )
        
        return {
            'dataframe': styled_df,
            'raw_dataframe': comparison_df,
            'summary': self.comparison_summary,
            'metadata': {
                'analysis_type': 'battery_performance_comparison',
                'methods_compared': ['default_shaving', 'simple_conservation', 'smart_conservation'],
                'total_rows_displayed': len(comparison_df),
                'columns': list(comparison_df.columns)
            }
        }    

    def _run_default_shaving(self):
        """
        Run default shaving strategy using execute_default_shaving_discharge.
        """
        result_df = self.df_sim.copy()
        
        # Get battery parameters - THROW ERRORS if missing (no fallbacks)
        battery_sizing = self.config_data.get('battery_sizing', {})
        battery_params = self.config_data.get('battery_params', {})
        
        # Try multiple possible keys for battery capacity - THROW ERROR if not found
        battery_capacity_kwh = (
            self.config_data.get('battery_capacity_kwh') or
            battery_sizing.get('battery_capacity_kwh') or
            battery_params.get('battery_capacity_kwh') or
            battery_sizing.get('capacity_kwh')
        )
        if not battery_capacity_kwh:
            raise KeyError(
                "Missing 'battery_capacity_kwh'. Expected in config_data, "
                "config_data['battery_sizing'], or config_data['battery_params']"
            )
        
        max_discharge_kw = (
            battery_sizing.get('max_discharge_kw') or
            battery_sizing.get('power_rating_kw') or  # Fallback to power_rating_kw
            battery_params.get('max_discharge_kw') or
            self.config_data.get('max_discharge_kw')
        )
        if not max_discharge_kw:
            raise KeyError(
                "Missing 'max_discharge_kw' or 'power_rating_kw'. Expected in "
                "config_data['battery_sizing'], config_data['battery_params'], or config_data"
            )
        
        c_rate = (
            battery_sizing.get('c_rate') or
            battery_params.get('c_rate') or
            self.config_data.get('c_rate') or
            DEFAULT_C_RATE  # Use global constant as final fallback
        )
        if not c_rate:
            raise KeyError(
                "Missing 'c_rate'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        efficiency = (
            battery_params.get('discharge_efficiency') or
            self.config_data.get('discharge_efficiency') or
            DEFAULT_DISCHARGE_EFFICIENCY  # Use global constant as final fallback
        )
        if not efficiency:
            raise KeyError(
                "Missing 'discharge_efficiency'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        soc_min_percent = battery_params.get('min_soc_percent')
        if soc_min_percent is None:  # Allow 0 as valid value
            soc_min_percent = self.config_data.get('min_soc_percent')
        if soc_min_percent is None:
            soc_min_percent = DEFAULT_SOC_MIN_PERCENT  # Use global constant as final fallback
        
        soc_max_percent = battery_params.get('max_soc_percent')
        if soc_max_percent is None:  # Allow 0 as valid value
            soc_max_percent = self.config_data.get('max_soc_percent')
        if soc_max_percent is None:
            soc_max_percent = DEFAULT_SOC_MAX_PERCENT  # Use global constant as final fallback
        
        interval_hours = self.config_data.get('interval_hours')
        if not interval_hours:
            raise KeyError("Missing 'interval_hours' in config_data")
        
        current_soc_kwh = battery_capacity_kwh * (self.initial_soc_percent / 100)
        
        # Get excess demand for event detection
        md_excess = MdExcess(self.config_data)
        excess_demand = md_excess.calculate_excess_demand()
        result_df['excess_demand_kw'] = excess_demand
        
        # Use TOU-aware event detection (respects tariff type and MD window)
        result_df['is_event'] = self._detect_events_with_tou_logic(result_df, excess_demand)
        
        # Initialize result columns
        result_df['battery_power_kw'] = 0.0
        result_df['net_demand_kw'] = result_df[self.power_col]
        result_df['battery_soc_kwh'] = current_soc_kwh
        result_df['battery_soc_percent'] = self.initial_soc_percent
        result_df['operation_type'] = 'none'
        
        # Process each timestamp
        for idx in result_df.index:
            if result_df.loc[idx, 'is_event']:
                current_demand = result_df.loc[idx, self.power_col]
                monthly_target = self.config_data['target_series'].loc[idx]
                
                battery_result = execute_default_shaving_discharge(
                    current_demand_kw=current_demand,
                    monthly_target_kw=monthly_target,
                    current_soc_kwh=current_soc_kwh,
                    battery_capacity_kwh=battery_capacity_kwh,
                    max_power_kw=max_discharge_kw,
                    interval_hours=interval_hours,
                    efficiency=efficiency,
                    soc_min_percent=soc_min_percent,
                    soc_max_percent=soc_max_percent,
                    c_rate=c_rate
                )
                
                result_df.loc[idx, 'battery_power_kw'] = battery_result['discharge_power_kw']
                result_df.loc[idx, 'net_demand_kw'] = battery_result['net_demand_kw']
                result_df.loc[idx, 'battery_soc_kwh'] = battery_result['updated_soc_kwh']
                result_df.loc[idx, 'battery_soc_percent'] = battery_result['updated_soc_percent']
                result_df.loc[idx, 'operation_type'] = 'discharge'
                current_soc_kwh = battery_result['updated_soc_kwh']
            else:
                result_df.loc[idx, 'battery_soc_kwh'] = current_soc_kwh
                result_df.loc[idx, 'battery_soc_percent'] = (current_soc_kwh / battery_capacity_kwh) * 100
        
        return result_df
    
    def _run_simple_conservation(self):
        """
        Run simple conservation strategy using execute_conservation_discharge.
        """
        result_df = self.df_sim.copy()
        
        # Get battery parameters - THROW ERRORS if missing (no fallbacks)
        battery_sizing = self.config_data.get('battery_sizing', {})
        battery_params = self.config_data.get('battery_params', {})
        
        # Try multiple possible keys - THROW ERROR if not found
        battery_capacity_kwh = (
            self.config_data.get('battery_capacity_kwh') or
            battery_sizing.get('battery_capacity_kwh') or
            battery_params.get('battery_capacity_kwh') or
            battery_sizing.get('capacity_kwh')
        )
        if not battery_capacity_kwh:
            raise KeyError(
                "Missing 'battery_capacity_kwh'. Expected in config_data, "
                "config_data['battery_sizing'], or config_data['battery_params']"
            )
        
        max_discharge_kw = (
            battery_sizing.get('max_discharge_kw') or
            battery_sizing.get('power_rating_kw') or  # Fallback to power_rating_kw
            battery_params.get('max_discharge_kw') or
            self.config_data.get('max_discharge_kw')
        )
        if not max_discharge_kw:
            raise KeyError(
                "Missing 'max_discharge_kw' or 'power_rating_kw'. Expected in "
                "config_data['battery_sizing'], config_data['battery_params'], or config_data"
            )
        
        c_rate = (
            battery_sizing.get('c_rate') or
            battery_params.get('c_rate') or
            self.config_data.get('c_rate') or
            DEFAULT_C_RATE  # Use global constant as final fallback
        )
        if not c_rate:
            raise KeyError(
                "Missing 'c_rate'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        efficiency = (
            battery_params.get('discharge_efficiency') or
            self.config_data.get('discharge_efficiency') or
            DEFAULT_DISCHARGE_EFFICIENCY  # Use global constant as final fallback
        )
        if not efficiency:
            raise KeyError(
                "Missing 'discharge_efficiency'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        soc_min_percent = battery_params.get('min_soc_percent')
        if soc_min_percent is None:
            soc_min_percent = self.config_data.get('min_soc_percent')
        if soc_min_percent is None:
            soc_min_percent = DEFAULT_SOC_MIN_PERCENT  # Use global constant as final fallback
        
        soc_max_percent = battery_params.get('max_soc_percent')
        if soc_max_percent is None:
            soc_max_percent = self.config_data.get('max_soc_percent')
        if soc_max_percent is None:
            soc_max_percent = DEFAULT_SOC_MAX_PERCENT  # Use global constant as final fallback
        
        interval_hours = self.config_data.get('interval_hours')
        if not interval_hours:
            raise KeyError("Missing 'interval_hours' in config_data")
        
        # Get conservation parameters from config (following V3 logic)
        soc_threshold = self.config_data.get('soc_threshold', 50.0)
        battery_kw_conserved = self.config_data.get('battery_kw_conserved', 100.0)
        
        # DEBUG: Print configuration
        print(f"\n🔍 DEBUG Simple Conservation:")
        print(f"   max_discharge_kw: {max_discharge_kw}")
        print(f"   battery_capacity_kwh: {battery_capacity_kwh}")
        print(f"   initial_soc_percent: {self.initial_soc_percent}")
        print(f"   soc_threshold: {soc_threshold}%")
        print(f"   battery_kw_conserved (fixed): {battery_kw_conserved} kW")
        print(f"   Conservation strategy: Fixed kW conservation (V3 logic)")
        
        current_soc_kwh = battery_capacity_kwh * (self.initial_soc_percent / 100)
        
        # Get excess demand for event detection
        md_excess = MdExcess(self.config_data)
        excess_demand = md_excess.calculate_excess_demand()
        result_df['excess_demand_kw'] = excess_demand
        
        # Use TOU-aware event detection (respects tariff type and MD window)
        result_df['is_event'] = self._detect_events_with_tou_logic(result_df, excess_demand)
        
        # Initialize result columns
        result_df['battery_power_kw'] = 0.0
        result_df['net_demand_kw'] = result_df[self.power_col]
        result_df['battery_soc_kwh'] = current_soc_kwh
        result_df['battery_soc_percent'] = self.initial_soc_percent
        result_df['operation_type'] = 'none'
        
        # Process each timestamp
        event_count = 0
        discharge_count = 0
        
        for idx in result_df.index:
            if result_df.loc[idx, 'is_event']:
                event_count += 1
                current_demand = result_df.loc[idx, self.power_col]
                monthly_target = self.config_data['target_series'].loc[idx]
                excess = current_demand - monthly_target
                
                # V3 CONSERVATION LOGIC: Check if conservation should activate
                # Conservation activates when SOC < threshold
                conservation_active = (current_soc_kwh / battery_capacity_kwh * 100) < soc_threshold
                
                if conservation_active:
                    # CONSERVATION MODE: Apply FIXED kW conservation amount
                    original_discharge_required = excess
                    power_to_conserve = min(battery_kw_conserved, original_discharge_required)
                    revised_discharge_power = max(0, original_discharge_required - power_to_conserve)
                    
                    # Use revised discharge as the actual excess for battery operation
                    excess = revised_discharge_power
                    
                    if event_count <= 3:
                        print(f"\n   Event {event_count} at {idx}:")
                        print(f"      🔋 CONSERVATION ACTIVE (SOC < {soc_threshold}%)")
                        print(f"      current_demand: {current_demand:.2f} kW")
                        print(f"      monthly_target: {monthly_target:.2f} kW")
                        print(f"      original_excess: {original_discharge_required:.2f} kW")
                        print(f"      power_to_conserve: {power_to_conserve:.2f} kW (fixed)")
                        print(f"      revised_excess: {excess:.2f} kW")
                    
                    # Use conservation discharge function
                    battery_result = execute_conservation_discharge(
                        current_demand_kw=current_demand,
                        monthly_target_kw=monthly_target,
                        battery_kw_conserved=battery_kw_conserved,
                        current_soc_kwh=current_soc_kwh,
                        battery_capacity_kwh=battery_capacity_kwh,
                        max_power_kw=max_discharge_kw,
                        interval_hours=interval_hours,
                        efficiency=efficiency,
                        soc_min_percent=soc_min_percent,
                        soc_max_percent=soc_max_percent,
                        c_rate=c_rate
                    )
                else:
                    # NORMAL MODE: Use DEFAULT discharge (no conservation)
                    if event_count <= 3:
                        print(f"\n   Event {event_count} at {idx}:")
                        print(f"      ⚡ NORMAL MODE (SOC >= {soc_threshold}%)")
                        print(f"      current_demand: {current_demand:.2f} kW")
                        print(f"      monthly_target: {monthly_target:.2f} kW")
                        print(f"      excess_demand: {excess:.2f} kW")
                    
                    # Use default discharge function (full discharge, no conservation)
                    battery_result = execute_default_shaving_discharge(
                        current_demand_kw=current_demand,
                        monthly_target_kw=monthly_target,
                        current_soc_kwh=current_soc_kwh,
                        battery_capacity_kwh=battery_capacity_kwh,
                        max_power_kw=max_discharge_kw,
                        interval_hours=interval_hours,
                        efficiency=efficiency,
                        soc_min_percent=soc_min_percent,
                        soc_max_percent=soc_max_percent,
                        c_rate=c_rate
                    )
                
                discharge_power = battery_result['discharge_power_kw']
                
                if discharge_power > 0:
                    discharge_count += 1
                
                # DEBUG: Print first 3 events
                if event_count <= 3:
                    print(f"      current_soc: {current_soc_kwh:.2f} kWh ({(current_soc_kwh/battery_capacity_kwh)*100:.1f}%)")
                    print(f"      discharge_power: {discharge_power:.2f} kW")
                    print(f"      net_demand: {battery_result['net_demand_kw']:.2f} kW")
                    print(f"      new_soc: {battery_result['updated_soc_kwh']:.2f} kWh ({battery_result['updated_soc_percent']:.1f}%)")
                
                result_df.loc[idx, 'battery_power_kw'] = discharge_power
                result_df.loc[idx, 'net_demand_kw'] = battery_result['net_demand_kw']
                result_df.loc[idx, 'battery_soc_kwh'] = battery_result['updated_soc_kwh']
                result_df.loc[idx, 'battery_soc_percent'] = battery_result['updated_soc_percent']
                result_df.loc[idx, 'operation_type'] = 'discharge_conserve'
                current_soc_kwh = battery_result['updated_soc_kwh']
            else:
                result_df.loc[idx, 'battery_soc_kwh'] = current_soc_kwh
                result_df.loc[idx, 'battery_soc_percent'] = (current_soc_kwh / battery_capacity_kwh) * 100
        
        # DEBUG: Print summary
        print(f"\n   Total events: {event_count}")
        print(f"   Events with discharge > 0: {discharge_count}")
        
        return result_df
    
    def _run_smart_conservation(self):
        """
        Run smart conservation strategy using MainSmartShaving.
        """
        orchestrator = MdOrchestrator()
        return orchestrator.MainSmartShaving(
            self.config_data, 
            initial_soc_percent=self.initial_soc_percent
        )