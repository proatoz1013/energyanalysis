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
from smart_conservation import MdOrchestrator, MdExcess
import pandas as pd

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
            battery_params.get('max_discharge_kw') or
            self.config_data.get('max_discharge_kw')
        )
        if not max_discharge_kw:
            raise KeyError(
                "Missing 'max_discharge_kw'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        c_rate = (
            battery_sizing.get('c_rate') or
            battery_params.get('c_rate') or
            self.config_data.get('c_rate')
        )
        if not c_rate:
            raise KeyError(
                "Missing 'c_rate'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        efficiency = (
            battery_params.get('discharge_efficiency') or
            self.config_data.get('discharge_efficiency')
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
            raise KeyError(
                "Missing 'min_soc_percent'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        soc_max_percent = battery_params.get('max_soc_percent')
        if soc_max_percent is None:
            soc_max_percent = self.config_data.get('max_soc_percent')
        if soc_max_percent is None:
            raise KeyError(
                "Missing 'max_soc_percent'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        interval_hours = self.config_data.get('interval_hours')
        if not interval_hours:
            raise KeyError("Missing 'interval_hours' in config_data")
        
        current_soc_kwh = battery_capacity_kwh * (self.initial_soc_percent / 100)
        
        # Get excess demand for event detection
        md_excess = MdExcess(self.config_data)
        excess_demand = md_excess.calculate_excess_demand()
        result_df['excess_demand_kw'] = excess_demand
        result_df['is_event'] = excess_demand > 0
        
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
            battery_params.get('max_discharge_kw') or
            self.config_data.get('max_discharge_kw')
        )
        if not max_discharge_kw:
            raise KeyError(
                "Missing 'max_discharge_kw'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        c_rate = (
            battery_sizing.get('c_rate') or
            battery_params.get('c_rate') or
            self.config_data.get('c_rate')
        )
        if not c_rate:
            raise KeyError(
                "Missing 'c_rate'. Expected in config_data['battery_sizing'], "
                "config_data['battery_params'], or config_data"
            )
        
        efficiency = (
            battery_params.get('discharge_efficiency') or
            self.config_data.get('discharge_efficiency')
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
            raise KeyError(
                "Missing 'min_soc_percent'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        soc_max_percent = battery_params.get('max_soc_percent')
        if soc_max_percent is None:
            soc_max_percent = self.config_data.get('max_soc_percent')
        if soc_max_percent is None:
            raise KeyError(
                "Missing 'max_soc_percent'. Expected in config_data['battery_params'] "
                "or config_data"
            )
        
        interval_hours = self.config_data.get('interval_hours')
        if not interval_hours:
            raise KeyError("Missing 'interval_hours' in config_data")
        
        # FIXED CONSERVATION: 50% of max discharge power
        battery_kw_conserved = max_discharge_kw * 0.5
        
        current_soc_kwh = battery_capacity_kwh * (self.initial_soc_percent / 100)
        
        # Get excess demand for event detection
        md_excess = MdExcess(self.config_data)
        excess_demand = md_excess.calculate_excess_demand()
        result_df['excess_demand_kw'] = excess_demand
        result_df['is_event'] = excess_demand > 0
        
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
                
                result_df.loc[idx, 'battery_power_kw'] = battery_result['discharge_power_kw']
                result_df.loc[idx, 'net_demand_kw'] = battery_result['net_demand_kw']
                result_df.loc[idx, 'battery_soc_kwh'] = battery_result['updated_soc_kwh']
                result_df.loc[idx, 'battery_soc_percent'] = battery_result['updated_soc_percent']
                result_df.loc[idx, 'operation_type'] = 'discharge_conserve'
                current_soc_kwh = battery_result['updated_soc_kwh']
            else:
                result_df.loc[idx, 'battery_soc_kwh'] = current_soc_kwh
                result_df.loc[idx, 'battery_soc_percent'] = (current_soc_kwh / battery_capacity_kwh) * 100
        
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