"""
Smart Conservation Module for Energy Analysis System

This module implements AI-powered conservation algorithms for battery management
in peak demand scenarios. It provides intelligent decision-making capabilities
for optimizing battery State of Charge (SOC) thresholds and power conservation
based on predictive analytics and machine learning.

Author: Energy Analysis System
Created: November 2025
Version: 1.0.0
"""

from typing import Optional, Dict
from datetime import datetime
from enum import Enum

class MdShavingMode(Enum):
    """Enumeration of MD Shaving operational modes."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ACTIVE = "active"
    CONSERVATION = "conservation"

class MdShavingConfig:
    """
    AI-Powered Smart Conservation Calculator
    
    This class stores static values and includes a simple validation method. 
    """
    def __init__(self, **config):
        """
        Initialize Smart Conservation Calculator with configuration parameters from smart_conservation_config
        
        Args:
            **config: Configuration dictionary containing all smart conservation parameters
        """
        # Load & target context
        self.df_sim = config.get('df_sim')
        self.power_col = config.get('power_col')
        self.monthly_targets = config.get('monthly_targets')
        self.interval_hours = config.get('interval_hours')
        self.selected_tariff = config.get('selected_tariff')
        
        # Battery parameters
        self.battery_params = config.get('battery_params')
        self.battery_sizing = config.get('battery_sizing')
        self.battery_capacity = config.get('battery_capacity')
        self.soc_threshold = config.get('soc_threshold')
        self.battery_kw_conserved = config.get('battery_kw_conserved')
        
        # Conservation settings
        self.conservation_enabled = config.get('conservation_enabled')
        self.conservation_dates = config.get('conservation_dates')
        self.conservation_day_type = config.get('conservation_day_type')
        self.safety_margin = config.get('safety_margin')
        self.min_exceedance_multiplier = config.get('min_exceedance_multiplier')
        
        # Simulation context
        self.holidays = config.get('holidays')
        self.current_timestamp = config.get('current_timestamp')
        self.tariff_type = config.get('tariff_type')
        
        # Smart conservation UI parameters
        self.prediction_horizon = config.get('prediction_horizon')
        self.conservation_aggressiveness = config.get('conservation_aggressiveness')

    def validate_configuration(self):
        """
        Perform simple validation of configuration parameters.
        
        Checks:
        - Data types are correct
        - Values are within expected ranges
        - Critical parameters are non-null
        
        Returns:
            tuple: (is_valid: bool, error_messages: list)
        """
        errors = []
        
        # Check critical non-null parameters
        required_params = {
            'df_sim': 'Simulation dataframe',
            'power_col': 'Power column name',
            'battery_sizing': 'Battery sizing configuration',
            'interval_hours': 'Data interval in hours'
        }
        
        for param, description in required_params.items():
            if getattr(self, param, None) is None:
                errors.append(f"Missing required parameter: {description} ({param})")
        
        # Validate data types and ranges for numeric parameters
        if self.soc_threshold is not None:
            if not isinstance(self.soc_threshold, (int, float)):
                errors.append("SOC threshold must be a number")
            elif not (10 <= self.soc_threshold <= 90):
                errors.append("SOC threshold must be between 10% and 90%")
        
        if self.battery_kw_conserved is not None:
            if not isinstance(self.battery_kw_conserved, (int, float)):
                errors.append("Battery kW conserved must be a number")
            elif self.battery_kw_conserved < 0:
                errors.append("Battery kW conserved cannot be negative")
        
        if self.battery_capacity is not None:
            if not isinstance(self.battery_capacity, (int, float)):
                errors.append("Battery capacity must be a number")
            elif self.battery_capacity <= 0:
                errors.append("Battery capacity must be positive")
        
        if self.interval_hours is not None:
            if not isinstance(self.interval_hours, (int, float)):
                errors.append("Interval hours must be a number")
            elif not (0.01 <= self.interval_hours <= 24):
                errors.append("Interval hours must be between 0.01 and 24")
        
        if self.prediction_horizon is not None:
            if not isinstance(self.prediction_horizon, (int, float)):
                errors.append("Prediction horizon must be a number")
            elif not (1 <= self.prediction_horizon <= 48):
                errors.append("Prediction horizon must be between 1 and 48 hours")
        
        if self.conservation_aggressiveness is not None:
            if not isinstance(self.conservation_aggressiveness, (int, float)):
                errors.append("Conservation aggressiveness must be a number")
            elif not (0.1 <= self.conservation_aggressiveness <= 1.0):
                errors.append("Conservation aggressiveness must be between 0.1 and 1.0")
        
        # Validate boolean parameters
        if self.conservation_enabled is not None:
            if not isinstance(self.conservation_enabled, bool):
                errors.append("Conservation enabled must be a boolean")
        
        # Validate string parameters
        if self.power_col is not None:
            if not isinstance(self.power_col, str) or len(self.power_col.strip()) == 0:
                errors.append("Power column must be a non-empty string")
        
        if self.tariff_type is not None:
            if not isinstance(self.tariff_type, str):
                errors.append("Tariff type must be a string")
        
        if self.conservation_day_type is not None:
            if not isinstance(self.conservation_day_type, str):
                errors.append("Conservation day type must be a string")
        
        # Validate dictionary parameters
        if self.battery_params is not None:
            if not isinstance(self.battery_params, dict):
                errors.append("Battery params must be a dictionary")
        
        if self.battery_sizing is not None:
            if not isinstance(self.battery_sizing, dict):
                errors.append("Battery sizing must be a dictionary")
        
        if self.selected_tariff is not None:
            if not isinstance(self.selected_tariff, dict):
                errors.append("Selected tariff must be a dictionary")
        
        # Validate list/collection parameters
        if self.conservation_dates is not None:
            if not isinstance(self.conservation_dates, (list, tuple)):
                errors.append("Conservation dates must be a list or tuple")
        
        if self.holidays is not None:
            if not isinstance(self.holidays, (set, list, tuple)):
                errors.append("Holidays must be a set, list, or tuple")
        
        # Check dataframe if present
        if self.df_sim is not None:
            try:
                import pandas as pd
                if not isinstance(self.df_sim, pd.DataFrame):
                    errors.append("df_sim must be a pandas DataFrame")
                elif self.df_sim.empty:
                    errors.append("df_sim cannot be empty")
                elif self.power_col and self.power_col not in self.df_sim.columns:
                    errors.append(f"Power column '{self.power_col}' not found in dataframe")
            except ImportError:
                errors.append("pandas is required but not available")
        
        return len(errors) == 0, errors

    def export_config(self):
        """
        Export the configuration data as a dictionary.
        
        This method creates a comprehensive dictionary containing all 
        configuration parameters that have been initialized in the class.
        Useful for serialization, logging, or passing configuration to
        other components.
        
        Returns:
            dict: Dictionary containing all configuration parameters
        """
        config_dict = {}
        
        # Core simulation data
        config_dict['df_sim'] = self.df_sim
        config_dict['power_col'] = self.power_col
        config_dict['interval_hours'] = self.interval_hours
        
        # Battery configuration
        config_dict['battery_params'] = self.battery_params
        config_dict['battery_sizing'] = self.battery_sizing
        config_dict['battery_capacity'] = self.battery_capacity
        config_dict['battery_kw_conserved'] = self.battery_kw_conserved
        config_dict['soc_threshold'] = self.soc_threshold
        
        # Tariff and cost settings
        config_dict['selected_tariff'] = self.selected_tariff
        config_dict['tariff_type'] = self.tariff_type
        
        # Conservation parameters
        config_dict['conservation_enabled'] = self.conservation_enabled
        config_dict['conservation_dates'] = self.conservation_dates
        config_dict['conservation_day_type'] = self.conservation_day_type
        config_dict['conservation_start_time'] = self.conservation_start_time
        config_dict['conservation_end_time'] = self.conservation_end_time
        
        # Calendar and holiday data
        config_dict['holidays'] = self.holidays
        
        # Smart conservation UI parameters
        config_dict['prediction_horizon'] = self.prediction_horizon
        config_dict['conservation_aggressiveness'] = self.conservation_aggressiveness
        
        return config_dict
    
class _MdEventState:
    """
    Internal state for the current MD event.

    This tracks whether an event is active, when it started, how long
    it has lasted, and basic per-event statistics needed for severity
    and reporting.
    """
    active: bool = False
    event_id: int = 0
    start_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    smoothed_excess_kw: float = 0.0
    above_trigger_count: int = 0
    below_trigger_count: int = 0
    max_excess_kw: float = 0.0
    max_severity: float = 0.0
    total_discharged_kwh: float = 0.0
    entered_conservation: bool = False

class _MdControllerState:
    """
    Internal mutable state of the MD controller across the historical run.

    This groups all cross-timestep variables: current mode, event state,
    last timestamp, last severity, and any debug-friendly components.
    """
    
    def __init__(self):
        """Initialize controller state with default values."""
        self.last_timestamp: Optional[datetime] = None
        self.mode: MdShavingMode = MdShavingMode.IDLE
        self.event: _MdEventState = _MdEventState()
        self.last_soc_percent: Optional[float] = None
        self.soc_reserve_percent: float = 0.0
        self.last_severity: float = 0.0
        self.last_severity_components: Optional[Dict[str, float]] = None

class MdShavingController:
    """
    Controller class for MD (Maximum Demand) Shaving operations.
    
    This class manages the sequential processing of energy data, maintaining
    state across historical rows and providing methods for analyzing MD values,
    checking context, and running computations based on the current state.
    
    The controller handles:
    - Sequential data iteration through df_sim
    - Event state management across time periods
    - Mode state persistence during processing
    - Context analysis for MD shaving decisions
    - Coordination of computation workflows
    
    Attributes:
        df_sim: DataFrame containing simulation data to process
        event_state: Current event status for MD shaving operations
        mode_state: Current operational mode state
    """
    
    def __init__(self, df_sim):
        """
        Initialize the MD Shaving Controller.
        
        Args:
            df_sim: pandas DataFrame containing the simulation data
                   with time-series energy consumption data
        """
        self.df_sim = df_sim
        self.config_data = None
        self.state = self._init_state()
        
        # Initialize processing variables
        self.current_row_index = 0
        self.total_rows = len(df_sim) if df_sim is not None else 0
        self.processing_complete = False
        
        # Initialize performance tracking
        self.processing_start_time = None
        self.last_processed_timestamp = None

    def _init_state(self):
        """
        Create and return a fresh controller state object.
        
        This method initializes all state variables to their default values,
        providing a clean state for each controller initialization or reset.
        Used to refresh states during initialization and for state reset operations.
        
        Returns:
            _MdControllerState: Fresh controller state object with default values
        """
        return _MdControllerState()

    def import_config(self, config_source):
        """
        Import configuration data from MdShavingConfig while maintaining modularity.
        
        This method accepts either a MdShavingConfig instance or a configuration
        dictionary, allowing flexible integration while keeping classes decoupled.
        The controller extracts only the data it needs for processing operations.
        
        Args:
            config_source: Either a MdShavingConfig instance or a dictionary
                          containing configuration parameters
        
        Returns:
            bool: True if configuration was imported successfully, False otherwise
        
        Raises:
            TypeError: If config_source is not a supported type
            ValueError: If required configuration parameters are missing
        """
        try:
            # Handle MdShavingConfig instance
            if hasattr(config_source, 'export_config'):
                self.config_data = config_source.export_config()
            # Handle dictionary input
            elif isinstance(config_source, dict):
                self.config_data = config_source.copy()
            else:
                raise TypeError(f"Unsupported config source type: {type(config_source)}")
            
            # Validate that essential parameters are present
            required_params = ['df_sim', 'power_col', 'interval_hours']
            missing_params = [param for param in required_params 
                            if param not in self.config_data or self.config_data[param] is None]
            
            if missing_params:
                raise ValueError(f"Missing required configuration parameters: {missing_params}")
            
            # Update df_sim if provided in config (allows override)
            if self.config_data.get('df_sim') is not None:
                self.df_sim = self.config_data['df_sim']
            
            return True
            
        except (TypeError, ValueError) as e:
            print(f"Configuration import failed: {e}")
            return False

    def get_config_param(self, param_name, default_value=None):
        """
        Safely retrieve a configuration parameter.
        
        Args:
            param_name (str): Name of the configuration parameter
            default_value: Value to return if parameter is not found
        
        Returns:
            The configuration parameter value or default_value
        """
        if self.config_data is None:
            return default_value
        return self.config_data.get(param_name, default_value)

    def reset_state(self):
        """
        Reset controller state to fresh initial values.
        
        This method creates a new state object and resets processing variables,
        useful for restarting analysis or clearing accumulated state between runs.
        """
        self.state = self._init_state()
        self.current_row_index = 0
        self.processing_complete = False
        self.processing_start_time = None
        self.last_processed_timestamp = None

    def is_initialized(self):
        """
        Check if controller is properly initialized for processing.
        
        Returns:
            bool: True if controller has all required components for processing
        """
        initialization_checks = {
            'df_sim': self.df_sim is not None,
            'config_data': self.config_data is not None,
            'state': self.state is not None,
            'df_not_empty': self.df_sim is not None and not self.df_sim.empty if hasattr(self.df_sim, 'empty') else True,
            'power_col_configured': self.get_config_param('power_col') is not None,
            'interval_configured': self.get_config_param('interval_hours') is not None
        }
        
        return all(initialization_checks.values())

    def get_initialization_status(self):
        """
        Get detailed initialization status for debugging.
        
        Returns:
            dict: Dictionary with initialization check results and missing components
        """
        checks = {
            'df_sim_present': self.df_sim is not None,
            'df_sim_not_empty': self.df_sim is not None and not self.df_sim.empty if hasattr(self.df_sim, 'empty') else False,
            'config_imported': self.config_data is not None,
            'state_initialized': self.state is not None,
            'power_col_configured': self.get_config_param('power_col') is not None,
            'interval_configured': self.get_config_param('interval_hours') is not None,
            'total_rows': self.total_rows,
            'current_mode': self.state.mode.value if self.state else 'No State'
        }
        
        missing_components = [key for key, value in checks.items() if not value and key != 'total_rows' and key != 'current_mode']
        
        return {
            'checks': checks,
            'is_ready': len(missing_components) == 0,
            'missing_components': missing_components
        }

    def run_historical(self, df_sim):
        """
        Process historical data sequentially through the MD shaving algorithm.
        
        Args:
            df_sim: DataFrame containing historical simulation data
            
        Returns:
            list: Results from processing all rows
        """
        return self._process_row(df_sim)

    def _process_row(self, df_sim):
        """
        Process all rows of data through the MD shaving logic.
        
        This method handles:
        - Looping through each row in the dataset
        - Reading timestamp, power, and SOC for each row
        - Processing sequential data analysis
        
        Args:
            df_sim: DataFrame containing simulation data
            
        Returns:
            list: Results from processing each row
        """
        results = []
        
        # For each row in the dataset, read the timestamp, power, SOC
        for row in df_sim.itertuples():
            # 1. Read current timestamp
            current_timestamp = row.Index if hasattr(row, 'Index') else getattr(row, df_sim.index.name, None)
            
            # 2. Read the current power (the value in df_sim associated with the timestamp)
            power_col = self.get_config_param('power_col')
            current_power = getattr(row, power_col, None) if power_col else None
            
            # 3. Read the current SOC
            current_soc = getattr(row, 'soc_percent', None)
            
            # Store row data
            row_result = {
                'timestamp': current_timestamp,
                'power': current_power,
                'soc': current_soc
            }
            results.append(row_result)
        
        return results

    def _check_tariff_window_conditions(self, current_timestamp):
        """
        Check tariff conditions and determine MD window state with SOC reserves.
        
        This method analyzes:
        1. Which tariff is currently being used
        2. If TOU tariff, whether we're inside or outside the MD window
        3. What rules apply for non-TOU tariffs
        4. Early/late window definition and boolean states
        5. SOC reserve levels for early/late periods
        
        Args:
            current_timestamp: Current timestamp being processed
            
        Returns:
            dict: Dictionary containing tariff analysis results
        """
        # 1. Which tariff are we using?
        tariff_type = self.get_config_param('tariff_type', 'unknown')
        selected_tariff = self.get_config_param('selected_tariff', {})
        
        # Initialize result dictionary
        result = {
            'tariff_type': tariff_type,
            'selected_tariff': selected_tariff,
            'is_tou': tariff_type.lower() == 'tou',
            'inside_md_window': False,
            'window_rules': 'standard',
            'is_early_window': False,
            'is_late_window': False,
            'soc_reserve_percent': 50.0  # Default reserve
        }
        
        # 2. If TOU, check if we're inside or outside the MD window
        if result['is_tou']:
            # Check MD window based on tariff configuration
            md_start_hour = selected_tariff.get('md_start_hour', 9)  # Default 9 AM
            md_end_hour = selected_tariff.get('md_end_hour', 17)    # Default 5 PM
            
            if current_timestamp and hasattr(current_timestamp, 'hour'):
                current_hour = current_timestamp.hour
                result['inside_md_window'] = md_start_hour <= current_hour < md_end_hour
                result['window_rules'] = 'md_window' if result['inside_md_window'] else 'off_peak'
        
        # 3. Rules for non-TOU tariffs
        else:
            result['window_rules'] = 'flat_rate'
        
        # 4. Define early/late window (50% split of work day)
        if result['inside_md_window']:
            md_start_hour = selected_tariff.get('md_start_hour', 9)
            md_end_hour = selected_tariff.get('md_end_hour', 17)
            
            # Calculate 50% point of the work day
            work_day_duration = md_end_hour - md_start_hour
            midpoint_hour = md_start_hour + (work_day_duration * 0.5)
            
            if current_timestamp and hasattr(current_timestamp, 'hour'):
                current_hour = current_timestamp.hour
                result['is_early_window'] = current_hour < midpoint_hour
                result['is_late_window'] = current_hour >= midpoint_hour
        
        # 5. Assign SOC reserve levels based on early/late window
        if result['is_early_window']:
            # Early part of window → higher reserve (70%)
            result['soc_reserve_percent'] = 70.0
        elif result['is_late_window']:
            # Late part of window → lower reserve (40%)
            result['soc_reserve_percent'] = 40.0
        else:
            # Outside MD window or non-TOU → standard reserve (50%)
            result['soc_reserve_percent'] = 50.0
        
        return result


class SmartConservationDebugger:
    """
    Debugging class for Smart Conservation analysis.
    
    This class provides debugging utilities to analyze and visualize
    tariff window conditions and smart conservation behavior across
    the entire dataset timeline.
    """
    
    def __init__(self, controller):
        """
        Initialize debugger with a MdShavingController instance.
        
        Args:
            controller: MdShavingController instance with imported configuration
        """
        self.controller = controller
    
    def generate_window_analysis_table(self, df_sim=None):
        """
        Generate a comprehensive analysis table of tariff window conditions
        and smart conservation states for all timestamps in the dataset.
        
        This method analyzes each timestamp to determine:
        - TOU vs non-TOU tariff classification
        - Early/late window status within MD periods
        - SOC reserve recommendations
        - Window rules and tariff conditions
        
        Args:
            df_sim: Optional DataFrame to analyze (uses controller's df_sim if None)
            
        Returns:
            dict: Analysis results in format suitable for md_shaving_solution_v2.py
                 Contains 'data' list and 'summary' dict for display
        """
        # Use controller's dataframe if none provided
        if df_sim is None:
            df_sim = self.controller.df_sim
        
        if df_sim is None or df_sim.empty:
            return {
                'data': [],
                'summary': {'error': 'No simulation data available'},
                'metadata': {
                    'total_timestamps': 0,
                    'analysis_type': 'window_analysis'
                }
            }
        
        analysis_data = []
        summary_stats = {
            'total_timestamps': len(df_sim),
            'tou_count': 0,
            'non_tou_count': 0,
            'early_window_count': 0,
            'late_window_count': 0,
            'inside_md_window_count': 0,
            'outside_md_window_count': 0,
            'avg_soc_reserve': 0.0
        }
        
        soc_reserves = []
        
        # Analyze each timestamp
        for timestamp in df_sim.index:
            # Get tariff window conditions for this timestamp
            window_conditions = self.controller._check_tariff_window_conditions(timestamp)
            
            # Create analysis record
            record = {
                'timestamp': timestamp,
                'date': timestamp.date() if hasattr(timestamp, 'date') else str(timestamp),
                'time': timestamp.strftime('%H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 0,
                'tariff_type': window_conditions['tariff_type'],
                'is_tou': window_conditions['is_tou'],
                'inside_md_window': window_conditions['inside_md_window'],
                'is_early_window': window_conditions['is_early_window'],
                'is_late_window': window_conditions['is_late_window'],
                'window_rules': window_conditions['window_rules'],
                'soc_reserve_percent': window_conditions['soc_reserve_percent'],
                'window_status': self._get_window_status_description(window_conditions)
            }
            
            analysis_data.append(record)
            
            # Update summary statistics
            if window_conditions['is_tou']:
                summary_stats['tou_count'] += 1
            else:
                summary_stats['non_tou_count'] += 1
            
            if window_conditions['is_early_window']:
                summary_stats['early_window_count'] += 1
            elif window_conditions['is_late_window']:
                summary_stats['late_window_count'] += 1
            
            if window_conditions['inside_md_window']:
                summary_stats['inside_md_window_count'] += 1
            else:
                summary_stats['outside_md_window_count'] += 1
            
            soc_reserves.append(window_conditions['soc_reserve_percent'])
        
        # Calculate summary statistics
        if soc_reserves:
            summary_stats['avg_soc_reserve'] = sum(soc_reserves) / len(soc_reserves)
        
        summary_stats.update({
            'tou_percentage': (summary_stats['tou_count'] / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 0,
            'early_window_percentage': (summary_stats['early_window_count'] / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 0,
            'late_window_percentage': (summary_stats['late_window_count'] / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 0,
            'inside_md_percentage': (summary_stats['inside_md_window_count'] / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 0
        })
        
        return {
            'data': analysis_data,
            'summary': summary_stats,
            'metadata': {
                'total_timestamps': len(analysis_data),
                'analysis_type': 'smart_conservation_window_analysis',
                'date_range': {
                    'start': analysis_data[0]['timestamp'] if analysis_data else None,
                    'end': analysis_data[-1]['timestamp'] if analysis_data else None
                },
                'columns': [
                    'timestamp', 'date', 'time', 'hour', 'tariff_type', 'is_tou',
                    'inside_md_window', 'is_early_window', 'is_late_window', 
                    'window_rules', 'soc_reserve_percent', 'window_status'
                ]
            }
        }
    
    def _get_window_status_description(self, window_conditions):
        """
        Generate a human-readable description of the window status.
        
        Args:
            window_conditions: Dictionary from _check_tariff_window_conditions
            
        Returns:
            str: Descriptive status string
        """
        if not window_conditions['is_tou']:
            return "Non-TOU Tariff"
        
        if not window_conditions['inside_md_window']:
            return "TOU Off-Peak"
        
        if window_conditions['is_early_window']:
            return "TOU Early MD Window"
        elif window_conditions['is_late_window']:
            return "TOU Late MD Window"
        else:
            return "TOU MD Window"


#LOGOFF 6 Nov 9:41 AM
#TO-DO： Add debug methods to display on main app （tabled data etc.）
#dictionary to store hardcoded static values
