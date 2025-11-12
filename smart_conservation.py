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
import pandas as pd

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
        self.target_series = config.get('target_series')
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
        validation_constants = SmartConstants.get_validation_constants()
        
        if self.soc_threshold is not None:
            if not isinstance(self.soc_threshold, (int, float)):
                errors.append("SOC threshold must be a number")
            else:
                soc_range = validation_constants['soc_threshold']
                if not (soc_range['min'] <= self.soc_threshold <= soc_range['max']):
                    errors.append(f"SOC threshold must be between {soc_range['min']}% and {soc_range['max']}%")
        
        if self.battery_kw_conserved is not None:
            if not isinstance(self.battery_kw_conserved, (int, float)):
                errors.append("Battery kW conserved must be a number")
            else:
                min_val = validation_constants['battery_kw_conserved']['min']
                if self.battery_kw_conserved < min_val:
                    errors.append(f"Battery kW conserved cannot be less than {min_val}")
        
        if self.battery_capacity is not None:
            if not isinstance(self.battery_capacity, (int, float)):
                errors.append("Battery capacity must be a number")
            else:
                min_val = validation_constants['battery_capacity']['min']
                if self.battery_capacity <= min_val:
                    errors.append(f"Battery capacity must be greater than {min_val}")
        
        if self.interval_hours is not None:
            if not isinstance(self.interval_hours, (int, float)):
                errors.append("Interval hours must be a number")
            else:
                interval_range = validation_constants['interval_hours']
                if not (interval_range['min'] <= self.interval_hours <= interval_range['max']):
                    errors.append(f"Interval hours must be between {interval_range['min']} and {interval_range['max']}")
        
        if self.prediction_horizon is not None:
            if not isinstance(self.prediction_horizon, (int, float)):
                errors.append("Prediction horizon must be a number")
            else:
                horizon_range = validation_constants['prediction_horizon']
                if not (horizon_range['min'] <= self.prediction_horizon <= horizon_range['max']):
                    errors.append(f"Prediction horizon must be between {horizon_range['min']} and {horizon_range['max']} hours")
        
        if self.conservation_aggressiveness is not None:
            if not isinstance(self.conservation_aggressiveness, (int, float)):
                errors.append("Conservation aggressiveness must be a number")
            else:
                aggr_range = validation_constants['conservation_aggressiveness']
                if not (aggr_range['min'] <= self.conservation_aggressiveness <= aggr_range['max']):
                    errors.append(f"Conservation aggressiveness must be between {aggr_range['min']} and {aggr_range['max']}")
        
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
        config_dict['target_series'] = self.target_series
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

class SmartConstants:
    """
    Centralized configuration class for Smart Conservation constants.
    
    This class provides a single location for all hardcoded values used throughout
    the Smart Conservation system, enabling easy maintenance and configuration.
    """
    
    @classmethod
    def get_validation_constants(cls):
        """
        Get validation range constants for configuration parameters.
        
        Returns:
            dict: Validation constants with min/max ranges
        """
        return {
            'soc_threshold': {'min': 10, 'max': 90},
            'battery_kw_conserved': {'min': 0, 'max': None},
            'battery_capacity': {'min': 0, 'max': None},
            'interval_hours': {'min': 0.01, 'max': 24},
            'prediction_horizon': {'min': 1, 'max': 48},
            'conservation_aggressiveness': {'min': 0.1, 'max': 1.0}
        }
    
    @classmethod
    def get_tariff_window_constants(cls):
        """
        Get tariff window and SOC reserve constants.
        
        Note: MD start/end hours are NOT included as defaults since they 
        depend on the specific tariff configuration.
        
        Returns:
            dict: Tariff window configuration constants
        """
        return {
            'default_soc_reserve': 50.0,           # Standard reserve percentage
            'early_window_soc_reserve': 70.0,      # Conservative early period
            'late_window_soc_reserve': 40.0,       # Aggressive late period
            'work_day_midpoint_factor': 0.5        # 50% split for early/late window
        }
    
    @classmethod
    def get_display_constants(cls):
        """
        Get display and UI constants.
        
        Returns:
            dict: Display configuration constants
        """
        return {
            'default_max_rows': 10,         # Default table display rows
            'summary_precision': 1,         # Decimal places for summary stats
            'percentage_precision': 1       # Decimal places for percentages
        }
    
    @classmethod
    def get_tariff_classification_rules(cls):
        """
        Get tariff classification rules and identifiers.
        
        Returns:
            dict: Tariff classification constants and identifiers
        """
        return {
            'tou_identifiers': ['tou', 'time-of-use', 'time_of_use'],
            'general_identifiers': ['general', 'flat', 'standard', 'fixed'],
            'default_classification': 'general'
        }
    
    @classmethod 
    def is_tou_tariff(cls, tariff_type):
        """
        Determine if a tariff type represents a Time-of-Use tariff.
        
        Args:
            tariff_type (str): The tariff type string to classify
            
        Returns:
            bool: True if tariff is TOU, False otherwise
        """
        if not tariff_type or not isinstance(tariff_type, str):
            return False
        
        rules = cls.get_tariff_classification_rules()
        tariff_lower = tariff_type.lower().strip()
        
        return tariff_lower in rules['tou_identifiers']
    
    @classmethod
    def classify_tariff_type(cls, tariff_type):
        """
        Classify tariff type into standard categories.
        
        Args:
            tariff_type (str): The tariff type string to classify
            
        Returns:
            str: Standardized tariff classification ('tou' or 'general')
        """
        if cls.is_tou_tariff(tariff_type):
            return 'tou'
        else:
            return 'general'

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
        
        # Get tariff window constants
        tariff_constants = SmartConstants.get_tariff_window_constants()
        
        # Initialize result dictionary
        result = {
            'tariff_type': tariff_type,
            'selected_tariff': selected_tariff,
            'is_tou': SmartConstants.is_tou_tariff(tariff_type),
            'inside_md_window': False,
            'window_rules': 'standard',
            'is_early_window': False,
            'is_late_window': False,
            'soc_reserve_percent': tariff_constants['default_soc_reserve']
        }
        
        # 2. If TOU, check if we're inside or outside the MD window
        if result['is_tou']:
            # Get MD window from tariff configuration (no defaults)
            md_start_hour = selected_tariff.get('md_start_hour')
            md_end_hour = selected_tariff.get('md_end_hour')
            
            # Only process MD window if both start and end hours are defined in tariff
            if md_start_hour is not None and md_end_hour is not None and current_timestamp and hasattr(current_timestamp, 'hour'):
                current_hour = current_timestamp.hour
                result['inside_md_window'] = md_start_hour <= current_hour < md_end_hour
                result['window_rules'] = 'md_window' if result['inside_md_window'] else 'off_peak'
            else:
                # If MD window not defined in TOU tariff, treat as off-peak
                result['inside_md_window'] = False
                result['window_rules'] = 'off_peak'
        
        # 3. Rules for non-TOU tariffs
        else:
            result['window_rules'] = 'flat_rate'
        
        # 4. Define early/late window (using midpoint factor from constants)
        if result['inside_md_window']:
            md_start_hour = selected_tariff.get('md_start_hour')
            md_end_hour = selected_tariff.get('md_end_hour')
            
            if md_start_hour is not None and md_end_hour is not None:
                # Calculate midpoint using factor from constants
                work_day_duration = md_end_hour - md_start_hour
                midpoint_hour = md_start_hour + (work_day_duration * tariff_constants['work_day_midpoint_factor'])
                
                if current_timestamp and hasattr(current_timestamp, 'hour'):
                    current_hour = current_timestamp.hour
                    result['is_early_window'] = current_hour < midpoint_hour
                    result['is_late_window'] = current_hour >= midpoint_hour
        
        # 5. Assign SOC reserve levels based on early/late window
        if result['is_early_window']:
            # Early part of window → higher reserve (conservative)
            result['soc_reserve_percent'] = tariff_constants['early_window_soc_reserve']
        elif result['is_late_window']:
            # Late part of window → lower reserve (aggressive)
            result['soc_reserve_percent'] = tariff_constants['late_window_soc_reserve']
        else:
            # Outside MD window or non-TOU → standard reserve
            result['soc_reserve_percent'] = tariff_constants['default_soc_reserve']
        
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

    def display_analysis_table(self, analysis_function=None, display_config=None, df_sim=None, **kwargs):
        """
        Main method that coordinates table creation and display using the refactored architecture.
        
        This is the main orchestrator method that:
        1. Calls prepare_analysis_function_for_display() to get standardized data
        2. Calls create_dynamic_analysis_table() to format for display  
        3. Returns complete display-ready result
        
        Args:
            analysis_function: Function to call for data analysis (defaults to generate_window_analysis_table)
            display_config: Optional display configuration dictionary
            df_sim: Optional DataFrame to analyze (uses controller's df_sim if None)
            **kwargs: Additional arguments to pass to the analysis function
            
        Returns:
            dict: Dictionary containing 'dataframe', 'summary', and 'metadata' keys
        """
        # Default configuration
        config = {
            'max_rows': 10,
            'show_summary': True,
            'debug_output': True,
            'clear_state': True
        }
        if display_config:
            config.update(display_config)
        
        # Display configuration debug information if requested
        if config['debug_output']:
            print("🔧 Smart Conservation Debug Analysis - Configuration Check")
            self.get_target_series_and_power_col_display()
            print("─" * 60)
        
        # Clear state from any previous analysis runs if requested
        if config['clear_state']:
            # Clear any existing MdEventTrigger state to ensure fresh results
            if hasattr(kwargs, 'get') and kwargs.get('md_event_trigger'):
                kwargs['md_event_trigger'].reset_timestamp_ids()
        
        # Step 1: Get standardized data using function adapter
        standardized_data = self.prepare_analysis_function_for_display(
            analysis_function, 
            df_sim=df_sim,
            **kwargs
        )
        
        # Check if we have data
        if not standardized_data['data']:
            return {
                'dataframe': pd.DataFrame(),
                'summary': standardized_data['summary'],
                'metadata': standardized_data['metadata'],
                'analysis_function': standardized_data['function_name'],
                'display_config': config
            }
        
        # Step 2: Create dynamic table using standardized data
        table_result = self.create_dynamic_analysis_table(
            data_records=standardized_data['data'],
            summary_stats=standardized_data['summary'], 
            metadata=standardized_data['metadata'],
            max_rows=config['max_rows']
        )
        
        # Step 3: Add display metadata and configuration info
        table_result.update({
            'analysis_function': standardized_data['function_name'],
            'data_key_used': standardized_data.get('data_key_used'),
            'display_config': config,
            'max_rows_setting': config['max_rows']
        })
        
        return table_result

    def display_window_analysis_table(self, data_source_func=None, df_sim=None, show_summary=True, max_rows=None, clear_state=True, **kwargs):
        """
        Legacy method that maintains backward compatibility with existing code.
        
        This method now acts as a wrapper around the new display_analysis_table method,
        ensuring all existing code continues to work while providing the new architecture.
        
        Args:
            data_source_func (callable, optional): Function to call for data (defaults to generate_window_analysis_table)
            df_sim: Optional DataFrame to analyze (uses controller's df_sim if None)
            show_summary: Whether to display summary statistics (default: True)
            max_rows: Maximum number of data rows to display (default: 10)
            clear_state: Whether to clear MdEventTrigger state before processing (default: True)
            **kwargs: Additional arguments to pass to the data source function
            
        Returns:
            dict: Dictionary containing 'dataframe', 'summary', and 'metadata' keys
        """
        # Convert legacy parameters to new display_config format
        display_config = {
            'max_rows': max_rows or 10,
            'show_summary': show_summary,
            'debug_output': True,
            'clear_state': clear_state
        }
        
        # Call the new orchestrator method
        return self.display_analysis_table(
            analysis_function=data_source_func,
            display_config=display_config,
            df_sim=df_sim,
            **kwargs
        )

    def create_dynamic_analysis_table(self, data_records, summary_stats=None, metadata=None, max_rows=None):
        """
        Create a dynamic table from any structured data records.
        
        This is the core table creation method that works with any structured data.
        It handles data formatting, type conversion, and creates display-ready DataFrames.
        
        Args:
            data_records: List of dictionaries with analysis data
            summary_stats: Optional summary statistics dictionary
            metadata: Optional metadata dictionary 
            max_rows: Maximum number of rows to display (default: 10)
            
        Returns:
            dict: Formatted table data ready for display with dataframe, summary, and metadata
        """
        # Get display constants
        display_constants = SmartConstants.get_display_constants()
        
        # Use default max_rows if not specified
        if max_rows is None:
            max_rows = display_constants['default_max_rows']
        
        # Handle empty data
        if not data_records:
            return {
                'dataframe': pd.DataFrame(),
                'summary': summary_stats or {'message': 'No data available'},
                'metadata': metadata or {'error': 'No data records provided'},
                'total_records': 0,
                'displayed_records': 0,
                'columns_detected': []
            }
        
        # Convert data to DataFrame with flexible column detection
        df_data = []
        
        for record in data_records:
            row_data = {}
            
            # Handle different data structures flexibly
            if isinstance(record, dict):
                # For dictionary records, process each key-value pair
                for key, value in record.items():
                    # Format the key for display (capitalize and replace underscores)
                    display_key = key.replace('_', ' ').title()
                    
                    # Format the value based on its type
                    if key == 'timestamp' and hasattr(value, 'isoformat'):
                        row_data[display_key] = value
                    elif key == 'date' and hasattr(value, 'strftime'):
                        row_data[display_key] = value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)
                    elif key == 'time' and isinstance(value, str):
                        row_data[display_key] = value
                    elif key.endswith('_percent') or 'percentage' in key.lower():
                        # Format percentage values
                        if value is not None:
                            try:
                                row_data[display_key] = f"{float(value):.{display_constants['percentage_precision']}f}%"
                            except (ValueError, TypeError):
                                row_data[display_key] = str(value)
                        else:
                            row_data[display_key] = "N/A"
                    elif isinstance(value, (int, float)) and value is not None:
                        # Format numeric values
                        try:
                            if isinstance(value, float):
                                row_data[display_key] = f"{value:.{display_constants['summary_precision']}f}"
                            else:
                                row_data[display_key] = str(value)
                        except (ValueError, TypeError):
                            row_data[display_key] = str(value)
                    elif isinstance(value, bool):
                        # Format boolean values
                        row_data[display_key] = "Yes" if value else "No"
                    else:
                        # Default string conversion
                        row_data[display_key] = str(value) if value is not None else "N/A"
            else:
                # For non-dictionary records, convert to string
                row_data['Value'] = str(record)
            
            df_data.append(row_data)
        
        # Create DataFrame
        df_analysis = pd.DataFrame(df_data)
        
        # Limit rows if specified
        if max_rows and len(df_analysis) > max_rows:
            df_display = df_analysis.head(max_rows).copy()
        else:
            df_display = df_analysis.copy()
        
        # Prepare return data
        return_data = {
            'dataframe': df_display,
            'summary': summary_stats or {},
            'metadata': metadata or {},
            'total_records': len(data_records),
            'displayed_records': len(df_display),
            'columns_detected': list(df_display.columns) if not df_display.empty else []
        }
        
        return return_data

    def prepare_analysis_function_for_display(self, analysis_function, **function_kwargs):
        """
        Take functions from other classes and prepare their output for table display.
        
        This adapter method standardizes the output from various analysis functions
        to ensure consistent data structure for table creation.
        
        Args:
            analysis_function: Function from other classes (like format_excess_demand_analysis)
            **function_kwargs: Arguments to pass to the analysis function
            
        Returns:
            dict: Standardized data structure with 'data', 'summary', 'metadata' keys
        """
        try:
            # Call the analysis function
            if analysis_function is None:
                # Default to window analysis
                analysis_function = self.generate_window_analysis_table
            
            result = analysis_function(**function_kwargs)
            
            # Handle different return formats and standardize
            if isinstance(result, dict):
                # Try to detect the data key automatically
                data_key = None
                if 'data' in result:
                    data_key = 'data'
                elif 'active_events' in result:
                    data_key = 'active_events'
                else:
                    # Fallback: try to find first list/array in results
                    for key, value in result.items():
                        if isinstance(value, (list, tuple)) and len(value) > 0:
                            data_key = key
                            break
                
                # Extract data using detected key
                data_records = result.get(data_key, []) if data_key else []
                
                # Standardize the output format
                standardized = {
                    'data': data_records,
                    'summary': result.get('summary', {}),
                    'metadata': result.get('metadata', {}),
                    'function_name': getattr(analysis_function, '__name__', 'unknown'),
                    'data_key_used': data_key
                }
                
            else:
                # Handle non-dictionary returns (fallback)
                standardized = {
                    'data': [{'result': str(result)}],
                    'summary': {'message': 'Function returned non-dictionary result'},
                    'metadata': {'result_type': str(type(result))},
                    'function_name': getattr(analysis_function, '__name__', 'unknown'),
                    'data_key_used': None
                }
            
            return standardized
            
        except Exception as e:
            return {
                'data': [],
                'summary': {'error': f'Function failed: {str(e)}'},
                'metadata': {'error': 'Function execution failed'},
                'function_name': getattr(analysis_function, '__name__', 'unknown') if analysis_function else 'None',
                'data_key_used': None
            }

        """
        Get size and dimensional information for target_series and power_col for Streamlit display.
        
        This method retrieves the target series and power column configuration
        from the controller's config data and prints length information for debugging.
        
        Returns:
            dict: Dictionary containing size/dimension data for Streamlit display
        """
        # Initialize target_series and df_sim from config
        target_series = self.controller.get_config_param('target_series', None)
        df_sim = self.controller.get_config_param('df_sim', None) or self.controller.df_sim
        

        # Return empty dict since this is debug-only function
        return {
            'debug_info': 'Length information printed to console',
            'target_series_available': target_series is not None,
            'df_sim_available': df_sim is not None
        }

    def format_excess_demand_analysis(self, timestamps=None, current_demand=None, excess_demand=None):
        """
        Format timestamp, current demand, and excess demand data for table display.
        
        This method takes demand and excess data and arranges them in a structured
        format suitable for table display in Streamlit or other UI components.
        
        Args:
            timestamps: Series or array-like of timestamps (optional, uses df_sim index if None)
            current_demand: Series or array-like of current demand values (optional, uses power_col from df_sim if None)
            excess_demand: Series or array-like of excess demand values (optional, calculates from MdExcess if None)
            
        Returns:
            dict: Dictionary containing formatted analysis data with keys:
                - 'data': List of dictionaries with timestamp, demand, and excess data
                - 'summary': Summary statistics for the excess analysis
                - 'metadata': Metadata about the analysis including column info and data types
        """
        # Get default data from controller if not provided
        df_sim = self.controller.df_sim
        power_col = self.controller.get_config_param('power_col')
        
        # Use provided timestamps or default to df_sim index
        if timestamps is None:
            if df_sim is not None and not df_sim.empty:
                timestamps = df_sim.index
            else:
                return {
                    'data': [],
                    'summary': {'error': 'No timestamp data available'},
                    'metadata': {'error': 'Missing timestamp data', 'analysis_type': 'excess_demand_analysis'}
                }
        
        # Use provided current_demand or get from df_sim
        if current_demand is None:
            if df_sim is not None and power_col and power_col in df_sim.columns:
                current_demand = df_sim[power_col]
            else:
                return {
                    'data': [],
                    'summary': {'error': f'Power column {power_col} not available in df_sim'},
                    'metadata': {'error': 'Missing current demand data', 'analysis_type': 'excess_demand_analysis'}
                }
        
        # Use provided excess_demand or calculate from MdExcess
        if excess_demand is None:
            try:
                # Create MdExcess instance to calculate excess demand
                config_data = self.controller.config_data or {}
                md_excess = MdExcess(config_data)
                excess_demand = md_excess.calculate_excess_demand()
            except Exception as e:
                return {
                    'data': [],
                    'summary': {'error': f'Failed to calculate excess demand: {str(e)}'},
                    'metadata': {'error': 'Excess demand calculation failed', 'analysis_type': 'excess_demand_analysis'}
                }
        
        # Align all series to same index
        try:
            # Convert to pandas Series if needed and align indices
            if not isinstance(timestamps, pd.Index):
                timestamps = pd.Index(timestamps)
            if not isinstance(current_demand, pd.Series):
                current_demand = pd.Series(current_demand, index=timestamps)
            if not isinstance(excess_demand, pd.Series):
                excess_demand = pd.Series(excess_demand, index=timestamps)
            
            # Align all data to the same index
            common_index = timestamps.intersection(current_demand.index).intersection(excess_demand.index)
            
            if len(common_index) == 0:
                return {
                    'data': [],
                    'summary': {'error': 'No common timestamps between demand and excess data'},
                    'metadata': {'error': 'Index alignment failed', 'analysis_type': 'excess_demand_analysis'}
                }
            
            # Filter data to common index
            aligned_timestamps = common_index
            aligned_current = current_demand.loc[common_index]
            aligned_excess = excess_demand.loc[common_index]
            
        except Exception as e:
            return {
                'data': [],
                'summary': {'error': f'Data alignment failed: {str(e)}'},
                'metadata': {'error': 'Data processing error', 'analysis_type': 'excess_demand_analysis'}
            }
        
        # Format data for table display
        analysis_data = []
        summary_stats = {
            'total_timestamps': len(aligned_timestamps),
            'excess_events': 0,
            'max_demand_kw': 0.0,
            'max_excess_kw': 0.0,
            'avg_demand_kw': 0.0,
            'avg_excess_when_active': 0.0,
            'total_excess_kwh': 0.0
        }
        
        excess_values = []
        demand_values = []
        
        # Process each timestamp
        for i, timestamp in enumerate(aligned_timestamps):
            current_val = aligned_current.iloc[i]
            excess_val = aligned_excess.iloc[i]
            
            # Create formatted record
            record = {
                'timestamp': timestamp,
                'date': timestamp.date() if hasattr(timestamp, 'date') else str(timestamp),
                'time': timestamp.strftime('%H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 0,
                'current_demand_kw': round(float(current_val), 2) if pd.notna(current_val) else 0.0,
                'excess_demand_kw': round(float(excess_val), 2) if pd.notna(excess_val) else 0.0,
                'has_excess': bool(excess_val > 0) if pd.notna(excess_val) else False,
                'demand_status': 'Over Target' if (pd.notna(excess_val) and excess_val > 0) else 'Within Target'
            }
            
            analysis_data.append(record)
            
            # Update statistics
            if pd.notna(current_val):
                demand_values.append(current_val)
            if pd.notna(excess_val) and excess_val > 0:
                excess_values.append(excess_val)
                summary_stats['excess_events'] += 1
        
        # Calculate summary statistics
        if demand_values:
            summary_stats['max_demand_kw'] = max(demand_values)
            summary_stats['avg_demand_kw'] = sum(demand_values) / len(demand_values)
        
        if excess_values:
            summary_stats['max_excess_kw'] = max(excess_values)
            summary_stats['avg_excess_when_active'] = sum(excess_values) / len(excess_values)
            
            # Calculate total excess energy (assuming interval_hours from config)
            interval_hours = self.controller.get_config_param('interval_hours', 0.25)  # Default 15 minutes
            summary_stats['total_excess_kwh'] = sum(excess_values) * interval_hours
        
        # Add percentage calculations
        summary_stats.update({
            'excess_percentage': (summary_stats['excess_events'] / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 0,
            'within_target_percentage': ((summary_stats['total_timestamps'] - summary_stats['excess_events']) / summary_stats['total_timestamps'] * 100) if summary_stats['total_timestamps'] > 0 else 100
        })
        
        # Create metadata
        metadata = {
            'analysis_type': 'excess_demand_analysis',
            'total_records': len(analysis_data),
            'date_range': {
                'start': analysis_data[0]['timestamp'] if analysis_data else None,
                'end': analysis_data[-1]['timestamp'] if analysis_data else None
            },
            'columns': [
                'timestamp', 'date', 'time', 'hour', 'current_demand_kw', 
                'excess_demand_kw', 'has_excess', 'demand_status'
            ],
            'data_types': {
                'timestamp': 'datetime',
                'date': 'date',
                'time': 'time_string',
                'hour': 'integer',
                'current_demand_kw': 'float',
                'excess_demand_kw': 'float',
                'has_excess': 'boolean',
                'demand_status': 'string'
            },
            'units': {
                'current_demand_kw': 'kW',
                'excess_demand_kw': 'kW',
                'total_excess_kwh': 'kWh'
            }
        }
        
        return {
            'data': analysis_data,
            'summary': summary_stats,
            'metadata': metadata
        }

class MdExcess:
    def __init__(self, config_source):
        """
        Initialize MdExcess with configuration data.
        
        Args:
            config_source: Either MdShavingConfig instance or config dictionary
        """
        if hasattr(config_source, 'target_series'):
            # Access from MdShavingConfig instance
            self.target_series = config_source.target_series
            self.power_col = config_source.power_col
            self.df_sim = config_source.df_sim
            self.interval_hours = config_source.interval_hours
            self.selected_tariff = config_source.selected_tariff
            self.tariff_type = config_source.tariff_type
        elif isinstance(config_source, dict):
            # Access from config dictionary
            self.target_series = config_source.get('target_series')
            self.power_col = config_source.get('power_col')
            self.df_sim = config_source.get('df_sim')
            self.interval_hours = config_source.get('interval_hours')
            self.selected_tariff = config_source.get('selected_tariff')
            self.tariff_type = config_source.get('tariff_type')
        else:
            raise TypeError("config_source must be MdShavingConfig instance or dictionary")
    
    def calculate_excess_demand(self):
        """
        Calculate MD excess demand by taking the difference between actual demand and target series.
        
        This method follows the V3 logic for calculating excess demand:
        - Uses df_sim[power_col] as actual demand
        - Uses target_series as dynamic monthly targets
        - Calculates excess as (actual - target).clip(lower=0)
        
        Returns:
            pd.Series: Excess demand for each timestamp, with negative values clipped to 0
        """
        # Validate required data is available
        if self.df_sim is None or self.df_sim.empty:
            raise ValueError("df_sim is not available or empty")
        
        if self.power_col is None or self.power_col not in self.df_sim.columns:
            raise ValueError(f"Power column '{self.power_col}' not found in df_sim")
        
        if self.target_series is None:
            raise ValueError("target_series is not available")
        
        # Get actual demand from df_sim
        actual_demand = self.df_sim[self.power_col]
        
        # Ensure target_series matches df_sim index
        if not actual_demand.index.equals(self.target_series.index):
            # Align target_series to match df_sim index
            target_aligned = self.target_series.reindex(actual_demand.index, method='ffill')
        else:
            target_aligned = self.target_series
        
        # Calculate excess demand (following V3 logic)
        # excess = max(0, current_demand - monthly_target)
        excess_demand = (actual_demand - target_aligned).clip(lower=0)
        
        return excess_demand
    

        """
        Calculate summary statistics for MD excess demand.
        
        Returns:
            dict: Dictionary containing excess demand statistics
        """
        try:
            excess_demand = self.calculate_excess_demand()
            
            # Calculate statistics
            statistics = {
                'total_excess_events': (excess_demand > 0).sum(),
                'max_excess_kw': excess_demand.max(),
                'total_excess_kwh': excess_demand.sum() * self.interval_hours if self.interval_hours else excess_demand.sum() * 0.25,
                'avg_excess_when_active': excess_demand[excess_demand > 0].mean() if (excess_demand > 0).any() else 0.0,
                'excess_percentage': (excess_demand > 0).sum() / len(excess_demand) * 100,
                'timestamps_with_excess': excess_demand[excess_demand > 0].index.tolist()
            }
            
            return statistics
            
        except Exception as e:
            return {
                'error': f"Failed to calculate excess statistics: {str(e)}",
                'total_excess_events': 0,
                'max_excess_kw': 0.0,
                'total_excess_kwh': 0.0,
                'avg_excess_when_active': 0.0,
                'excess_percentage': 0.0,
                'timestamps_with_excess': []
            }
    


#TO-DO: Verify the size of monthly targets and refer to how excess is calculated in v3
#TO-DO: streamline debugging feature (discuss best practice with ChatGPT)