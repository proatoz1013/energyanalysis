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
    NORMAL = "normal"
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
    is_event: bool = False
    start_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    smoothed_excess_kw: float = 0.0
    above_trigger_count: int = 0
    below_trigger_count: int = 0
    max_excess_kw: float = 0.0
    severity_score: float = 0.0
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
            'work_day_midpoint_factor': 0.5,
            'default_md_start_hour': 14,
            'default_md_end_hour': 22        # 50% split for early/late window
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
    def get_rp4_peak_constants(cls):
        """
        Get RP4 peak period constants from PEAK_OFFPEAK_LOGIC_COMPARISON.md.
        
        Universal RP4 peak definition (applies to all voltage levels):
        - Peak: Monday-Friday, 2:00 PM - 10:00 PM (excluding public holidays)
        - Off-Peak: All other times (weekends, holidays, weekday nights/mornings)
        
        Source: tariffs/peak_logic.py - is_peak_rp4() function
        
        Returns:
            dict: RP4 peak period configuration
        """
        return {
            'peak_days': {0, 1, 2, 3, 4},    # Monday=0 to Friday=4
            'weekend_days': {5, 6},           # Saturday=5, Sunday=6
            'peak_start_hour': 14,            # 2:00 PM
            'peak_end_hour': 22,              # 10:00 PM
            'exclude_holidays': True          # Don't count holidays as peak
        }
    
    @classmethod
    def is_weekend(cls, timestamp):
        """
        Check if timestamp falls on a weekend (Saturday or Sunday).
        
        Args:
            timestamp (datetime): Timestamp to check
            
        Returns:
            bool: True if Saturday (5) or Sunday (6)
        """
        rp4_constants = cls.get_rp4_peak_constants()
        return timestamp.weekday() in rp4_constants['weekend_days']
    
    @classmethod
    def is_weekday(cls, timestamp):
        """
        Check if timestamp falls on a weekday (Monday-Friday).
        
        Args:
            timestamp (datetime): Timestamp to check
            
        Returns:
            bool: True if Monday (0) through Friday (4)
        """
        rp4_constants = cls.get_rp4_peak_constants()
        return timestamp.weekday() in rp4_constants['peak_days']
    
    @classmethod
    def is_public_holiday(cls, timestamp, holidays=None):
        """
        Check if timestamp falls on a public holiday.
        
        Args:
            timestamp (datetime): Timestamp to check
            holidays (set/list/dict): Holiday dates from config
            
        Returns:
            bool: True if it's a public holiday
        """
        if not holidays:
            return False
        
        timestamp_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        
        if isinstance(holidays, (set, list)):
            return timestamp_date in holidays
        elif isinstance(holidays, dict):
            return timestamp_date in holidays.keys()
        else:
            return False
    
    @classmethod
    def is_peak_rp4(cls, timestamp, holidays=None):
        """
        Check if timestamp is in RP4 peak period using universal logic.
        
        Implements the same logic as tariffs/peak_logic.py is_peak_rp4():
        Peak = Weekday (Mon-Fri) AND Peak Hours (2PM-10PM) AND Not Holiday
        
        From PEAK_OFFPEAK_LOGIC_COMPARISON.md:
        - Peak: Monday-Friday, 2:00 PM - 10:00 PM (excluding public holidays)
        - Off-Peak: All other times (weekends, holidays, weekday nights/mornings)
        
        Args:
            timestamp (datetime): Timestamp to check
            holidays (set/list/dict): Public holiday dates
            
        Returns:
            bool: True if in RP4 peak period
        """
        # 1. HOLIDAY CHECK (first priority)
        if cls.is_public_holiday(timestamp, holidays):
            return False
        
        # 2. WEEKDAY CHECK (Mon-Fri = 0-4)
        if not cls.is_weekday(timestamp):
            return False
        
        # 3. HOUR CHECK (2PM-10PM = 14-22)
        rp4_constants = cls.get_rp4_peak_constants()
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
        return rp4_constants['peak_start_hour'] <= hour < rp4_constants['peak_end_hour']
    
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

    def is_md_active(self, timestamp, config_data):
        """
        Check if MD recording is active based on tariff type and RP4 peak period logic.
        
        For TOU tariffs: MD is recorded only during RP4 peak periods (weekday 2PM-10PM, excluding holidays)
        For General tariffs: MD is recorded 24/7
        
        Args:
            timestamp: The timestamp to check for MD recording activity
            config_data: Configuration dictionary with tariff_type and holidays
            
        Returns:
            bool: True if MD recording is active, False otherwise
        """
        # Get tariff type from passed config_data
        tariff_type = config_data.get('tariff_type', 'general')
        
        # Check if TOU tariff
        if SmartConstants.is_tou_tariff(tariff_type):
            # For TOU tariffs, MD is only active during RP4 peak periods
            holidays = config_data.get('holidays', set())
            return SmartConstants.is_peak_rp4(timestamp, holidays)
        else:
            # For General tariffs, MD recording is always active (24/7)
            return True

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
        
        # Get holidays from config (supporting multiple formats: set, list, dict)
        holidays = self.get_config_param('holidays', set())
        
        # Initialize result dictionary
        result = {
            'tariff_type': tariff_type,
            'selected_tariff': selected_tariff,
            'is_tou': SmartConstants.is_tou_tariff(tariff_type),
            'inside_md_window': False,
            'window_rules': 'standard',
            'is_early_window': False,
            'is_late_window': False,
            'soc_reserve_percent': tariff_constants['default_soc_reserve'],
            # Add RP4 peak logic classifications
            'is_weekend': False,
            'is_weekday': False,
            'is_holiday': False,
            'is_peak_rp4': False
        }
        
        # Add RP4 peak classifications if timestamp is valid
        if current_timestamp and hasattr(current_timestamp, 'weekday'):
            result['is_weekend'] = SmartConstants.is_weekend(current_timestamp)
            result['is_weekday'] = SmartConstants.is_weekday(current_timestamp)
            result['is_holiday'] = SmartConstants.is_public_holiday(current_timestamp, holidays)
            result['is_peak_rp4'] = SmartConstants.is_peak_rp4(current_timestamp, holidays)
        
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

    def _is_md_window(self):
       """
       
       
       
       
       """

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
            target_series = self.controller.get_config_param('target_series', None)
            power_col = self.controller.get_config_param('power_col', None)
            print(f"   Target series available: {target_series is not None}")
            print(f"   Power column: {power_col}")
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
        };
        
        return {
            'data': analysis_data,
            'summary': summary_stats,
            'metadata': metadata
        }

    def get_analysis_function(self, analysis_type):
        """
        Registry of available analysis functions for dynamic dispatch.
        
        This method serves as the central registry that maps analysis type strings
        to their corresponding analysis functions. It enables the consolidator
        function to dynamically select the appropriate analysis method without
        hardcoding function calls.
        
        Args:
            analysis_type (str): The type of analysis to perform
                Available types:
                - 'excess_demand': MD excess demand analysis using analyze_md_excess_demand
                - 'md_excess_demand': MD excess demand analysis using analyze_md_excess_demand
                - 'window_analysis': Tariff window and conservation state analysis  
                - 'battery_performance': Battery performance analysis (future)
                - 'tariff_analysis': Tariff optimization analysis (future)
                
        Returns:
            callable: The analysis function corresponding to the requested type
                     Returns generate_window_analysis_table as default fallback
        """
        # Central registry of all available analysis functions
        registry = {
            'excess_demand': self.analyze_md_excess_demand,
            'md_excess_demand': self.analyze_md_excess_demand,
            'window_analysis': self.generate_window_analysis_table,
            'event_status': self.analyze_event_status,
            'trigger_events': self.analyze_event_status,
            'historical_events': self.analyze_historical_events,
            'process_historical_events': self.analyze_historical_events,
            # Future analysis functions can be added here without changing V3:
            # 'battery_performance': self.analyze_battery_performance_for_display,
            # 'tariff_optimization': self.analyze_tariff_optimization_for_display,
            # 'conservation_efficiency': self.analyze_conservation_efficiency_for_display,
            # 'cost_savings': self.analyze_cost_savings_for_display,
        }
        
        # Return requested function or fallback to window analysis
        return registry.get(analysis_type, self.generate_window_analysis_table)

    def display_any_analysis(self, analysis_type="window_analysis", display_config=None, **kwargs):
        """
        Main consolidator function that handles any type of analysis display.
        
        This is the single entry point that V3 calls for all analysis display needs.
        It dynamically selects the appropriate analysis function based on type,
        then uses the existing orchestrator infrastructure to format and display results.
        
        This function ensures that V3 never needs to change - all new analysis types
        are handled by simply adding new functions to the registry and calling this
        method with the appropriate analysis_type parameter.
        
        Args:
            analysis_type (str): Type of analysis to perform (default: "window_analysis")
                - 'excess_demand': Display MD excess demand analysis
                - 'window_analysis': Display tariff window analysis
                - Additional types can be added without changing this interface
            display_config (dict, optional): Display configuration parameters
                - 'max_rows': Maximum rows to display (default: 10)
                - 'debug_output': Show debug information (default: True)  
                - 'show_summary': Include summary statistics (default: True)
            **kwargs: Additional arguments passed to the analysis function
                
        Returns:
            dict: Complete analysis result with dataframe, summary, and metadata
                Same format as display_analysis_table() for consistency
                
        Examples:
            # Excess demand analysis - now uses analyze_md_excess_demand
            result = debugger.display_any_analysis("excess_demand")
            
            # Window analysis with custom config
            result = debugger.display_any_analysis(
                "window_analysis", 
                display_config={'max_rows': 20, 'debug_output': False}
            )
            
            # In V3 conservation tab (this never changes):
            st.dataframe(debugger.display_any_analysis("excess_demand")['dataframe'])
        """
        # Step 1: Get the appropriate analysis function from registry
        analysis_function = self.get_analysis_function(analysis_type)
        
        # Step 2: Use existing orchestrator infrastructure to handle everything
        # This leverages the complete existing architecture:
        # - prepare_analysis_function_for_display() for function adaptation
        # - create_dynamic_analysis_table() for dynamic formatting  
        # - Full error handling and data validation
        result = self.display_analysis_table(
            analysis_function=analysis_function,
            display_config=display_config,
            **kwargs
        )
        
        # Step 3: Add consolidator metadata for tracking
        result.update({
            'requested_analysis_type': analysis_type,
            'registry_function_used': getattr(analysis_function, '__name__', 'unknown'),
            'consolidator_version': '1.0'
        })
        
        return result

    def analyze_md_excess_demand(self, display_config=None):
        """
        Analyze MD excess demand using MdExcess class and display using existing infrastructure.
        
        This method demonstrates the complete workflow:
        1. Creates MdExcess instance from controller configuration
        2. Calls calculate_excess_demand() to get excess demand data
        3. Uses existing format_excess_demand_analysis() to format the data
        4. Uses existing display infrastructure to create the table
        
        This is a complete implementation that utilizes all existing methods
        without creating any redundancy.
        
        Args:
            display_config (dict, optional): Display configuration parameters
                - 'max_rows': Maximum rows to display (default: 10)
                - 'debug_output': Show debug information (default: True)
                - 'show_summary': Include summary statistics (default: True)
                
        Returns:
            dict: Complete analysis result with dataframe, summary, and metadata
                Same format as display_analysis_table() for consistency
                
        Examples:
            # Basic usage
            result = debugger.analyze_md_excess_demand()
            
            # With custom configuration
            result = debugger.analyze_md_excess_demand(
                display_config={'max_rows': 20, 'debug_output': False}
            )
            
            # Display the result in Streamlit
            st.dataframe(result['dataframe'])
            st.json(result['summary'])
        """
        # Set default display configuration
        config = {
            'max_rows': 10,
            'debug_output': True,  # ✅ ENABLE DEBUG for troubleshooting
            'show_summary': True
        }
        if display_config:
            config.update(display_config)
        
        try:
            # 🔍 DEBUG CHECKPOINT 1: Configuration data
            config_data = self.controller.config_data
            if config['debug_output']:
                print(f"🔍 DEBUG 1: config_data available: {config_data is not None}")
                if config_data:
                    print(f"   config_data keys: {list(config_data.keys())}")
            
            if not config_data:
                if config['debug_output']:
                    print("❌ FAILURE: No configuration data available in controller")
                return {
                    'dataframe': pd.DataFrame(),
                    'summary': {'error': 'No configuration data available in controller'},
                    'metadata': {'error': 'Missing configuration', 'analysis_type': 'md_excess_demand'},
                    'analysis_function': 'analyze_md_excess_demand',
                    'display_config': config
                }
            
            # 🔍 DEBUG CHECKPOINT 2: MdExcess instantiation
            if config['debug_output']:
                print(f"🔍 DEBUG 2: Creating MdExcess instance...")
            
            md_excess = MdExcess(config_data)
            
            if config['debug_output']:
                print(f"   MdExcess created successfully: {md_excess is not None}")
            
            # 🔍 DEBUG CHECKPOINT 3: Calculate excess demand
            if config['debug_output']:
                print(f"🔍 DEBUG 3: Calling calculate_excess_demand()...")
            
            excess_demand = md_excess.calculate_excess_demand()
            
            if config['debug_output']:
                print(f"   Excess demand calculated: {excess_demand is not None}")
                if excess_demand is not None:
                    print(f"   Excess demand length: {len(excess_demand)}")
                    print(f"   Max excess: {excess_demand.max():.2f} kW")
                    print(f"   Events with excess: {(excess_demand > 0).sum()}")
            
            # 🔍 DEBUG CHECKPOINT 4: Format the data
            if config['debug_output']:
                print(f"🔍 DEBUG 4: Calling format_excess_demand_analysis()...")
            
            formatted_result = self.format_excess_demand_analysis(
                excess_demand=excess_demand
            )
            
            if config['debug_output']:
                print(f"   Formatting result: {formatted_result is not None}")
                if formatted_result:
                    print(f"   Formatted data length: {len(formatted_result.get('data', []))}")
            
            # Check if formatting was successful
            if not formatted_result['data']:
                if config['debug_output']:
                    print("❌ FAILURE: Formatting failed - no data")
                return {
                    'dataframe': pd.DataFrame(),
                    'summary': formatted_result.get('summary', {'error': 'Formatting failed'}),
                    'metadata': formatted_result.get('metadata', {'error': 'No formatted data'}),
                    'analysis_function': 'analyze_md_excess_demand',
                    'display_config': config
                }
            
            # 🔍 DEBUG CHECKPOINT 5: Create dynamic table
            if config['debug_output']:
                print(f"🔍 DEBUG 5: Calling create_dynamic_analysis_table()...")
            
            table_result = self.create_dynamic_analysis_table(
                data_records=formatted_result['data'],
                summary_stats=formatted_result['summary'],
                metadata=formatted_result['metadata'],
                max_rows=config['max_rows']
            )
            
            if config['debug_output']:
                print(f"   Table creation result: {table_result is not None}")
                if table_result:
                    print(f"   Table rows: {table_result.get('displayed_records', 0)}")
            
            # Step 5: Add method-specific metadata
            table_result.update({
                'analysis_function': 'analyze_md_excess_demand',
                'md_excess_used': True,
                'excess_calculation_method': 'MdExcess.calculate_excess_demand',
                'display_config': config,
                'workflow_steps': [
                    'Created MdExcess instance',
                    'Called calculate_excess_demand()',
                    'Used format_excess_demand_analysis()',
                    'Used create_dynamic_analysis_table()',
                    'Added metadata and returned result'
                ]
            })
            
            if config['debug_output']:
                print(f"✅ SUCCESS: Analysis complete - {table_result['displayed_records']} rows displayed")
            
            return table_result
            
        except Exception as e:
            error_message = f'MD excess analysis failed: {str(e)}'
            if config.get('debug_output', False):
                print(f"❌ EXCEPTION in analyze_md_excess_demand: {error_message}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
            
            return {
                'dataframe': pd.DataFrame(),
                'summary': {'error': error_message},
                'metadata': {
                    'error': 'Method execution failed',
                    'analysis_type': 'md_excess_demand',
                    'exception_type': type(e).__name__,
                    'exception_details': str(e)
                },
                'analysis_function': 'analyze_md_excess_demand',
                'display_config': config
            }
    
   
        """
        Analyze MD event status using TriggerEvents with real simulation data.
        
        This method creates a TriggerEvents instance, processes real excess demand data
        through the trigger counter logic, and displays the results using existing
        infrastructure - just like analyze_md_excess_demand but for trigger events.
        
        Args:
            display_config (dict, optional): Display configuration parameters
                - 'max_rows': Maximum rows to display (default: 10)
                - 'debug_output': Show debug information (default: True)
                - 'show_summary': Include summary statistics (default: True)
            trigger_threshold_kw (float): Trigger threshold for TriggerEvents (default: 50.0)
                
        Returns:
            dict: Complete analysis result with dataframe, summary, and metadata
                Same format as display_analysis_table() for consistency
        """
        # Set default display configuration
        config = {
            'max_rows': 10,
            'debug_output': True,
            'show_summary': True
        }
        if display_config:
            config.update(display_config)
        
        try:
            # 🔍 DEBUG CHECKPOINT 1: Configuration data
            config_data = self.controller.config_data
            if config['debug_output']:
                print(f"🔍 DEBUG 1: config_data available: {config_data is not None}")
            
            if not config_data:
                if config['debug_output']:
                    print("❌ FAILURE: No configuration data available in controller")
                return {
                    'dataframe': pd.DataFrame(),
                    'summary': {'error': 'No configuration data available in controller'},
                    'metadata': {'error': 'Missing configuration', 'analysis_type': 'event_status'},
                    'analysis_function': 'analyze_event_status',
                    'display_config': config
                }
            
            # 🔍 DEBUG CHECKPOINT 2: Get excess demand data (same as MD excess analysis)
            if config['debug_output']:
                print(f"🔍 DEBUG 2: Getting excess demand data...")
            
            md_excess = MdExcess(config_data)
            excess_demand = md_excess.calculate_excess_demand()
            
            if config['debug_output']:
                print(f"   Excess demand calculated: {excess_demand is not None}")
                if excess_demand is not None:
                    print(f"   Excess demand length: {len(excess_demand)}")
                    print(f"   Max excess: {excess_demand.max():.2f} kW")
            
            # 🔍 DEBUG CHECKPOINT 3: Process through TriggerEvents
            if config['debug_output']:
                print(f"🔍 DEBUG 3: Processing trigger events (threshold: {trigger_threshold_kw} kW)...")
            
            trigger_events = TriggerEvents(trigger_threshold_kw=trigger_threshold_kw)
            event_state = _MdEventState()
            
            # Process each excess demand value through trigger logic
            trigger_results = []
            for i, (timestamp, excess_val) in enumerate(excess_demand.items()):
                # Update trigger counters based on excess demand
                trigger_result = trigger_events.update_trigger_counters(
                    smoothed_excess_kw=excess_val,
                    event_state=event_state
                )
                
                # Get current event status
                event_status = trigger_events.get_event_status(event_state)
                
                # Add timestamp and processing info
                event_status.update({
                    'timestamp': timestamp,
                    'excess_demand_kw': excess_val,
                    'trigger_result': trigger_result['trigger_status'],
                    'counter_changed': trigger_result.get('counter_reset', False)
                })
                
                trigger_results.append(event_status)
                
                # Only keep recent results to avoid too much data
                if len(trigger_results) > config['max_rows'] * 10:  # Keep 10x more than display
                    trigger_results = trigger_results[-config['max_rows'] * 5:]  # Keep 5x for variety
            
            if config['debug_output']:
                print(f"   Processed {len(excess_demand)} excess values")
                print(f"   Final trigger results: {len(trigger_results)} records")
                print(f"   Final above_trigger_count: {event_state.above_trigger_count}")
                print(f"   Final below_trigger_count: {event_state.below_trigger_count}")
            
            # 🔍 DEBUG CHECKPOINT 4: Format the data
            if config['debug_output']:
                print(f"🔍 DEBUG 4: Formatting trigger event data...")
            
            formatted_result = self.format_event_status_analysis(trigger_results)
            
            if config['debug_output']:
                print(f"   Formatting result: {formatted_result is not None}")
                if formatted_result:
                    print(f"   Formatted data length: {len(formatted_result.get('data', []))}")
            
            # 🔍 DEBUG CHECKPOINT 5: Create dynamic table
            if config['debug_output']:
                print(f"🔍 DEBUG 5: Creating trigger event table...")
            
            table_result = self.create_dynamic_analysis_table(
                data_records=formatted_result['data'],
                summary_stats=formatted_result['summary'],
                metadata=formatted_result['metadata'],
                max_rows=config['max_rows']
            )
            
            if config['debug_output']:
                print(f"   Table creation successful: {table_result is not None}")
                if table_result:
                    print(f"   Table rows: {table_result.get('displayed_records', 0)}")
            
            # Add method-specific metadata
            table_result.update({
                'analysis_function': 'analyze_event_status',
                'trigger_events_used': True,
                'trigger_threshold_kw': trigger_threshold_kw,
                'display_config': config,
                'workflow_steps': [
                    'Created MdExcess instance',
                    'Calculated excess demand',
                    'Created TriggerEvents instance',
                    'Processed through trigger counter logic',
                    'Used format_event_status_analysis()',
                    'Used create_dynamic_analysis_table()'
                ]
            })
            
            if config['debug_output']:
                print(f"✅ SUCCESS: Event status analysis complete - {table_result['displayed_records']} rows displayed")
            
            return table_result
            
        except Exception as e:
            error_message = f'Event status analysis failed: {str(e)}'
            if config.get('debug_output', False):
                print(f"❌ EXCEPTION in analyze_event_status: {error_message}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
            
            return {
                'dataframe': pd.DataFrame(),
                'summary': {'error': error_message},
                'metadata': {
                    'error': 'Method execution failed',
                    'analysis_type': 'event_status',
                    'exception_type': type(e).__name__,
                    'exception_details': str(e)
                },
                'analysis_function': 'analyze_event_status',
                'display_config': config
            }
    
    def analyze_historical_events(self, display_config=None):
        """
        Analyze complete historical dataset using process_historical_events method.
        
        This method uses TriggerEvents.process_historical_events() to classify the entire
        dataset into events and non-events using the simplified logic: excess_demand > 0.
        It reuses existing display infrastructure for consistent formatting and presentation.
        
        Args:
            display_config (dict, optional): Display configuration parameters
                - 'max_rows': Maximum rows to display (default: 100)
                - 'debug_output': Show debug information (default: False)
                - 'show_summary': Include summary statistics (default: True)
                
        Returns:
            dict: Complete analysis result with dataframe, summary, and metadata
        """
        # Set default display configuration
        config = {
            'max_rows': 100,
            'debug_output': False,
            'show_summary': True
        }
        if display_config:
            config.update(display_config)
        
        try:
            # Get configuration data
            config_data = self.controller.config_data
            if not config_data:
                return {
                    'dataframe': pd.DataFrame(),
                    'summary': {'error': 'No configuration data available'},
                    'metadata': {'error': 'Missing configuration', 'analysis_type': 'historical_events'},
                    'analysis_function': 'analyze_historical_events',
                    'display_config': config
                }
            
            # Process historical events using TriggerEvents with battery state tracking
            trigger_events = MdOrchestrator()
            enhanced_df = trigger_events.process_events_with_battery_state(config_data, initial_soc_percent=95.0)
            
            # Convert enhanced dataframe to display format
            display_data = []
            for idx, row in enhanced_df.head(config['max_rows'] * 2).iterrows():
                display_row = {
                    'timestamp': idx,
                    'current_power_kw': round(row[config_data['power_col']], 2),
                    'target_kw': round(config_data['target_series'].loc[idx] if idx in config_data['target_series'].index else 0, 2),
                    'excess_demand_kw': round(row['excess_demand_kw'], 2),
                    'is_event': 'Yes' if row['is_event'] else 'No',
                    'event_id': row['event_id'],
                    'event_start': 'Yes' if row['event_start'] else 'No',
                    'event_duration_min': round(row['event_duration'], 1),
                    'severity_score': round(row['severity_score'], 2),
                    'battery_soc_kwh': round(row['battery_soc_kwh'], 2),
                    'battery_soc_percent': round(row['battery_soc_percent'], 2)
                }
                display_data.append(display_row)
            
            # Calculate summary statistics
            summary_stats = {
                'total_timestamps': len(enhanced_df),
                'total_event_timestamps': int(enhanced_df['is_event'].sum()),
                'total_non_event_timestamps': int((~enhanced_df['is_event']).sum()),
                'unique_events_detected': int(enhanced_df[enhanced_df['event_id'] > 0]['event_id'].nunique()),
                'max_excess_demand_kw': float(enhanced_df['excess_demand_kw'].max()),
                'min_excess_demand_kw': float(enhanced_df['excess_demand_kw'].min()),
                'avg_excess_during_events': float(enhanced_df[enhanced_df['is_event']]['excess_demand_kw'].mean()) if enhanced_df['is_event'].any() else 0,
                'event_percentage': round(enhanced_df['is_event'].sum() / len(enhanced_df) * 100, 1)
            }
            
            # Create table using existing infrastructure
            table_result = self.create_dynamic_analysis_table(
                data_records=display_data,
                summary_stats=summary_stats,
                metadata={
                    'analysis_type': 'historical_events',
                    'method_used': 'process_events_with_battery_state',
                    'event_logic': 'excess_demand > 0 with battery SOC tracking',
                    'trigger_source': 'target_series (monthly target)'
                },
                max_rows=config['max_rows']
            )
            
            # Add method-specific metadata
            table_result.update({
                'analysis_function': 'analyze_historical_events',
                'enhanced_dataframe': enhanced_df,  # Include full processed dataset
                'display_config': config,
                'workflow_steps': [
                    'Created MdOrchestrator instance',
                    'Called process_events_with_battery_state() with 95% initial SOC',
                    'Classified events as excess_demand > 0 with battery state tracking',
                    'Used create_dynamic_analysis_table() for display'
                ]
            })
            
            return table_result
            
        except Exception as e:
            return {
                'dataframe': pd.DataFrame(),
                'summary': {'error': f'Historical events analysis failed: {str(e)}'},
                'metadata': {
                    'error': 'Method execution failed',
                    'analysis_type': 'historical_events',
                    'exception_type': type(e).__name__,
                    'exception_details': str(e)
                },
                'analysis_function': 'analyze_historical_events',
                'display_config': config
            }
    
    
        """
        Format event status data for table display using existing infrastructure.
        
        This method takes event status data from TriggerEvents.get_event_status()
        and formats it for display in conservation_tab2 using the same structure
        as other analysis methods.
        
        Args:
            event_scenarios: List of event status dictionaries from get_event_status()
            
        Returns:
            dict: Dictionary with 'data', 'summary', and 'metadata' for table display
        """
        if not event_scenarios:
            return {
                'data': [],
                'summary': {'error': 'No event scenarios provided'},
                'metadata': {'error': 'Missing event data', 'analysis_type': 'event_status_analysis'}
            }
        
        # Format data for table display
        analysis_data = []
        summary_stats = {
            'total_scenarios': len(event_scenarios),
            'active_events': 0,
            'conservation_events': 0,
            'max_excess_kw': 0.0,
            'max_severity': 0.0,
            'total_discharged_kwh': 0.0,
            'avg_above_trigger_count': 0.0,
            'avg_below_trigger_count': 0.0
        }
        
        above_counts = []
        below_counts = []
        
        # Process each event scenario
        for i, event_status in enumerate(event_scenarios):
            # Create formatted record
            record = {
                'scenario_id': i + 1,
                'scenario': event_status.get('scenario', f'Scenario {i + 1}'),
                'timestamp': event_status.get('timestamp', 'N/A'),
                'description': event_status.get('description', 'No description'),
                'event_active': 'Yes' if event_status.get('event_active', False) else 'No',
                'event_id': event_status.get('event_id', 0),
                'duration_minutes': round(event_status.get('duration_minutes', 0.0), 1),
                'smoothed_excess_kw': round(event_status.get('smoothed_excess_kw', 0.0), 2),
                'above_trigger_count': event_status.get('above_trigger_count', 0),
                'below_trigger_count': event_status.get('below_trigger_count', 0),
                'max_excess_kw': round(event_status.get('max_excess_kw', 0.0), 2),
                'max_severity': round(event_status.get('max_severity', 0.0), 2),
                'total_discharged_kwh': round(event_status.get('total_discharged_kwh', 0.0), 2),
                'entered_conservation': 'Yes' if event_status.get('entered_conservation', False) else 'No',
                'trigger_threshold_kw': event_status.get('trigger_threshold_kw', 50.0),
                'controller_mode': event_status.get('controller_mode', 'Unknown')
            }
            
            analysis_data.append(record)
            
            # Update summary statistics
            if event_status.get('event_active', False):
                summary_stats['active_events'] += 1
            
            if event_status.get('entered_conservation', False):
                summary_stats['conservation_events'] += 1
            
            # Track numeric values for averages and maximums
            max_excess = event_status.get('max_excess_kw', 0.0)
            if max_excess > summary_stats['max_excess_kw']:
                summary_stats['max_excess_kw'] = max_excess
            
            max_sev = event_status.get('max_severity', 0.0)
            if max_sev > summary_stats['max_severity']:
                summary_stats['max_severity'] = max_sev
            
            summary_stats['total_discharged_kwh'] += event_status.get('total_discharged_kwh', 0.0)
            
            above_counts.append(event_status.get('above_trigger_count', 0))
            below_counts.append(event_status.get('below_trigger_count', 0))
        
        # Calculate averages
        if above_counts:
            summary_stats['avg_above_trigger_count'] = sum(above_counts) / len(above_counts)
        if below_counts:
            summary_stats['avg_below_trigger_count'] = sum(below_counts) / len(below_counts)
        
        # Add percentage calculations
        summary_stats.update({
            'active_event_percentage': (summary_stats['active_events'] / summary_stats['total_scenarios'] * 100) if summary_stats['total_scenarios'] > 0 else 0,
            'conservation_percentage': (summary_stats['conservation_events'] / summary_stats['total_scenarios'] * 100) if summary_stats['total_scenarios'] > 0 else 0
        })
        
        # Create metadata
        metadata = {
            'analysis_type': 'event_status_analysis',
            'total_records': len(analysis_data),
            'columns': [
                'scenario_id', 'scenario', 'timestamp', 'description', 'event_active', 
                'event_id', 'duration_minutes', 'smoothed_excess_kw', 'above_trigger_count',
                'below_trigger_count', 'max_excess_kw', 'max_severity', 
                'total_discharged_kwh', 'entered_conservation', 'trigger_threshold_kw',
                'controller_mode'
            ],
            'data_types': {
                'scenario_id': 'integer',
                'scenario': 'string',
                'timestamp': 'datetime',
                'description': 'string',
                'event_active': 'boolean_string',
                'event_id': 'integer',
                'duration_minutes': 'float',
                'smoothed_excess_kw': 'float',
                'above_trigger_count': 'integer',
                'below_trigger_count': 'integer',
                'max_excess_kw': 'float',
                'max_severity': 'float',
                'total_discharged_kwh': 'float',
                'entered_conservation': 'boolean_string',
                'trigger_threshold_kw': 'float',
                'controller_mode': 'string'
            },
            'units': {
                'duration_minutes': 'minutes',
                'smoothed_excess_kw': 'kW',
                'max_excess_kw': 'kW',
                'total_discharged_kwh': 'kWh',
                'trigger_threshold_kw': 'kW'
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
    
class TriggerEvents:
    """
    Event trigger management for MD excess demand monitoring.
    
    This class handles the logic for tracking trigger events based on
    excess demand relative to monthly targets. Events are classified as
    any timestamp where MD excess > 0 (exceeding the target).
    """
    
    def __init__(self, trigger_threshold_kw=50.0):
        """
        Initialize TriggerEvents with a configurable trigger threshold.
        
        Args:
            trigger_threshold_kw (float): The trigger level in kW above which
                                        excess demand is considered significant
        """
        self.trigger_threshold_kw = trigger_threshold_kw
    
    def set_event_state(self, current_excess, inside_md_window, event_state):
        """
        Simple boolean logic to determine if event_state.active is True or False
        If excess demand is not zero AND in MD window, set event_state.active = True
        Else, set event_state.active = False
        """
        # Simple boolean logic: excess demand not zero AND in MD window
        if (current_excess > 0) and inside_md_window:
            event_state.active = True
        else:
            event_state.active = False
            
        return event_state.active
        
    def set_event_id(self, event_state):
        """
        Create persistent event counter and increment it for new events.
        
        This method only handles the persistent counter increment logic.
        All 4-case conditional logic is handled in the orchestrator loop.
        
        Args:
            event_state (_MdEventState): Event state object to update
            
        Returns:
            int: The new event ID assigned to this event
        """
        # Initialize persistent counter if it doesn't exist
        if not hasattr(event_state, 'persistent_event_counter'):
            event_state.persistent_event_counter = 0
        
        # Increment the persistent counter and assign to event_id
        event_state.persistent_event_counter += 1
        event_state.event_id = event_state.persistent_event_counter
        
        return event_state.event_id
  
    def _reset_event_statistics(self, event_state):
        """
        Reset event duration and per-event statistics for a new event.
        
        This private method clears all per-event tracking variables to ensure
        a clean state for the new event being started.
        
        Args:
            event_state (_MdEventState): Event state object to reset
        """
        # Reset duration tracking
        event_state.duration_minutes = 0.0
        
        # Reset per-event statistics
        event_state.max_excess_kw = 0.0
        event_state.max_severity = 0.0
        event_state.total_discharged_kwh = 0.0
        
        # Reset conservation flag
        event_state.entered_conservation = False
        
        # Reset smoothed excess (will be updated by trigger processing)
        event_state.smoothed_excess_kw = 0.0
        
        # Note: Trigger counters (above_trigger_count, below_trigger_count) 
        # are NOT reset here as they continue across events for pattern tracking
       
    def _set_event_mode(self, controller_state):
        """
        Set the MD shaving controller mode when an event is detected.
        
        This method updates the controller state to set the appropriate
        MD shaving mode when an event becomes active. It sets the mode
        to NORMAL to indicate active MD shaving operations.
        
        Args:
            controller_state (_MdControllerState): Controller state object to update
        """
        # Import the enum here to avoid circular imports
        from enum import Enum
        
        # Define MdShavingMode if not already defined
        class MdShavingMode(Enum):
            IDLE = "idle"
            NORMAL = "normal" 
            CONSERVATION = "conservation"
            EMERGENCY = "emergency"
        
        # Set controller mode to NORMAL for active MD shaving
        controller_state.mode = MdShavingMode.NORMAL
        
    def get_event_status(self, event_state):
        """
        Get comprehensive status information for the current event.
        
        This method provides detailed information about the current event state,
        useful for monitoring and debugging event progression.
        
        Args:
            event_state (_MdEventState): Event state to analyze
            
        Returns:
            dict: Comprehensive event status information
        """
        return {
            'event_active': event_state.active,
            'event_id': event_state.event_id,
            'start_time': event_state.start_time,
            'duration_minutes': event_state.duration_minutes,
            'smoothed_excess_kw': event_state.smoothed_excess_kw,
            'above_trigger_count': event_state.above_trigger_count,
            'below_trigger_count': event_state.below_trigger_count,
            'max_excess_kw': event_state.max_excess_kw,
            'max_severity': event_state.max_severity,
            'total_discharged_kwh': event_state.total_discharged_kwh,
            'entered_conservation': event_state.entered_conservation,
            'trigger_threshold_kw': self.trigger_threshold_kw
        }

class SeverityScore:
    """
    Utility class for MD window status checking.
    
    This class provides simplified methods to check if timestamps fall within
    MD windows using the existing comprehensive _check_tariff_window_conditions method.
    """
    
    def __init__(self, controller):
        """
        Initialize MdWindowType with a controller instance.
        
        Args:
            controller (MdShavingController): Controller instance with _check_tariff_window_conditions method
        """
        self.controller = controller
    
    def _get_event_duration_from_dataframe(self, current_timestamp):
        """
        Retrieve event duration for current timestamp from df_sim.
        
        This helper method reads the 'event_duration' column from the simulation
        dataframe to get the current event's duration in minutes.
        
        Args:
            current_timestamp (datetime): Timestamp to look up
            
        Returns:
            float: Event duration in minutes, or 0.0 if not found/not in event
        """
        try:
            # Get df_sim from controller config
            config_data = self.controller.config_data
            if not config_data:
                return 0.0
            
            df_sim = config_data.get('df_sim')
            if df_sim is None:
                return 0.0
            
            # Check if timestamp exists in dataframe
            if current_timestamp not in df_sim.index:
                return 0.0
            
            # Get event_duration column value
            if 'event_duration' not in df_sim.columns:
                return 0.0
            
            duration_minutes = df_sim.loc[current_timestamp, 'event_duration']
            
            # Return as float, handling any NaN or invalid values
            return float(duration_minutes) if duration_minutes is not None else 0.0
            
        except Exception as e:
            # If any error occurs, return 0.0 as safe default
            return 0.0
    
    def check_md_window_status(self, df_sim=None):
        """
        Check MD window status for all timestamps in df_sim using existing _check_tariff_window_conditions.
        
        This method provides a simple wrapper around the comprehensive _check_tariff_window_conditions
        method, returning just the MD window state (True/False) for each timestamp in df_sim.
        
        Args:
            df_sim (pd.DataFrame, optional): DataFrame with timestamps to check
                                           Uses controller's df_sim if None
                                           
        Returns:
            pd.Series: Boolean series indicating MD window status for each timestamp
                      True = inside MD window, False = outside MD window
                      
        Example:
            md_window = MdWindowType(controller)
            window_status = md_window.check_md_window_status()
            print(f"Timestamps in MD window: {window_status.sum()}")
        """
        # Use provided df_sim or get from controller
        if df_sim is None:
            df_sim = self.controller.df_sim
        
        if df_sim is None or df_sim.empty:
            return pd.Series(dtype=bool, name='inside_md_window')
        
        # Check MD window status for each timestamp
        md_window_status = []
        for timestamp in df_sim.index:
            try:
                # Call existing comprehensive method
                window_conditions = self.controller._check_tariff_window_conditions(timestamp)
                
                # Extract just the MD window boolean
                inside_md_window = window_conditions.get('inside_md_window', False)
                md_window_status.append(inside_md_window)
                
            except Exception:
                # Default to False if checking fails
                md_window_status.append(False)
        
        # Return as pandas Series with same index as df_sim
        return pd.Series(md_window_status, index=df_sim.index, name='inside_md_window')
    
    def severity_score(self, current_timestamp, event_state, controller_state):
        """
        Set event status based on MD window conditions, overriding previous event active logic.
        
        This method calls check_md_window_status to determine if the current timestamp
        is inside an MD window. If not inside MD window, it sets event_state.active = False
        to override previous event logic that doesn't consider tariff or MD window conditions.
        
        If active_event is active, it calculates severity score based on formula:
        Args:
            current_timestamp (datetime): Current timestamp to check
            event_state (_MdEventState): Event state object to modify
            controller_state (_MdControllerState): Controller state for event management
            
        Returns:
            dict: Summary of event status actions performed
        """
        trigger_events = TriggerEvents()
        
        # ✅ UPDATED: Pass current_timestamp to severity_params to retrieve event duration from dataframe
        active_event_data = self.severity_params(current_timestamp=current_timestamp)
        
        battery_excess_diff = active_event_data.get('battery_excess_difference_kw', 0.0)
        tightness_value = active_event_data.get('tightness_value', 0.5)

        # Calculate severity score: w1*excess_diff + w2*event_duration_score + w3*tightness_value
        # Assign weights as variables for easy tuning later
        w1 = 1.0  # Weight for excess difference
        w2 = 1.0  # Weight for event duration
        w3 = 2.0  # Weight for tightness value
                
        # Get event duration and normalize to score (0-1 range)
        event_duration_minutes = active_event_data.get('event_duration_minutes', 0)
        event_duration_score = min(event_duration_minutes / 60.0, 1.0)  # Normalize to hour, cap at 1.0
                
        # Calculate severity score
        severity_score = w1 * battery_excess_diff + w2 * event_duration_score + w3 * tightness_value
                
        # Update event state with calculated severity
        event_state.severity_score = severity_score
             

        return severity_score        
                      
    def severity_params(self, current_timestamp=None):
        """
        Retrieve battery maximum discharge capacity and calculate difference with current excess. 
        Return as a variable for use in conservation logic. 

        Retrieve active event duration from df_sim dataframe using helper method.

        Retrieve battery SOC and assign a tightness value.

        Return a dictionary with these values.
        
        Args:
            current_timestamp (datetime, optional): Timestamp to retrieve event duration for
        
        Returns:
            dict: Dictionary containing severity parameters including event duration from dataframe
        """
        # Get battery capacity from controller configuration
        battery_capacity = self.controller.get_config_param('battery_capacity', 0.0)
        battery_kw_conserved = self.controller.get_config_param('battery_kw_conserved', 0.0)
        
        # Calculate current excess demand using existing MdExcess method
                #compare current excess to battery capacity and cap to a reasonable upperbound
                #compare event duration against a chosen worrying duration threshold L_ref
                #cap event duration score to a reasonable upperbound (~1.0 so very long doesn't dominate)
                #SOC tightness between 0-1.0, where 0 is comfortable and 1 is at or below reserve
        try:
            config_data = self.controller.config_data
            if config_data:
                md_excess = MdExcess(config_data)
                excess_demand = md_excess.calculate_excess_demand()
                current_excess = excess_demand.iloc[-1] if not excess_demand.empty else 0.0
            else:
                current_excess = 0.0
        except Exception:
            current_excess = 0.0
        
        # Calculate difference between battery capacity and current excess
        raw_battery_excess_difference = battery_kw_conserved - current_excess
        
        # Normalize battery excess difference to 0-1 range for severity score
        # Positive difference (battery can handle excess) -> lower severity (closer to 0)
        # Negative difference (excess exceeds battery) -> higher severity (closer to 1)
        if raw_battery_excess_difference >= 0:
            # Battery can handle the excess - lower severity
            battery_excess_difference = max(0.0, 1.0 - (raw_battery_excess_difference / max(battery_kw_conserved, 1.0)))
        else:
            # Excess exceeds battery capacity - higher severity, capped at 1.0
            battery_excess_difference = min(1.0, 0.5 + abs(raw_battery_excess_difference) / max(battery_kw_conserved, 1.0))
        
        # ✅ NEW: Retrieve event duration from df_sim dataframe using helper method
        if current_timestamp is not None:
            event_duration_minutes = int(self._get_event_duration_from_dataframe(current_timestamp))
        else:
            event_duration_minutes = 0
        
        # Retrieve battery SOC and assign tightness value
        # SOC tightness: High SOC = loose (0.8-1.0), Medium SOC = moderate (0.4-0.6), Low SOC = tight (0.0-0.3)
        current_soc = self.controller.get_config_param('battery_soc_percent', 50.0)  # Default 50%
        
        if current_soc >= 80.0:
            soc_tightness = 'loose'
            tightness_value = 0.9
        elif current_soc >= 40.0:
            soc_tightness = 'moderate'
            tightness_value = 0.5
        else:
            soc_tightness = 'tight'
            tightness_value = 0.2
        
        return {
            'battery_max_discharge_kw': battery_kw_conserved,
            'current_excess_demand_kw': current_excess,
            'raw_battery_excess_difference_kw': raw_battery_excess_difference,
            'battery_excess_difference_kw': battery_excess_difference,
            'event_duration_minutes': event_duration_minutes,
            'battery_soc_percent': current_soc,
            'soc_tightness': soc_tightness,
            'tightness_value': tightness_value
        } 

class DecisionMaker:
    """
    Decision maker using four-case conditional logic for event management.
    
    Determines when to calculate severity score. This is the bridge between 
    the smart_conservation module and the smart_battery_executor module. 

    It is a refactor of the existing logic in process_historical_events and 
    exists as a method to be called in future orchestrator functions. 
    """

    def four_case_event_logic(self, previous_event_active, current_event_active, 
                                current_timestamp, event_state, controller_state, 
                                severity_calculator, df_sim, row_index,
                                battery_soc_kwh=None, battery_soc_percent=None):
        """
        Apply four-case conditional logic to manage event state and calculate severity score.
        
        This method is called by the battery executor to determine discharge strategy based
        on event transitions and severity assessment. It bridges Smart Conservation (System 1)
        and Smart Battery Executor (System 2).
        
        Args:
            previous_event_active (bool): Event active state from previous timestamp
            current_event_active (bool): Event active state for current timestamp
            current_timestamp (datetime): Current timestamp being processed
            event_state (_MdEventState): Event state object tracking current event
            controller_state (_MdControllerState): Controller state for mode management
            severity_calculator (SeverityScore): Calculator for severity assessment
            df_sim (pd.DataFrame): Simulation dataframe for storing results
            row_index (int): Current row index in df_sim (i)
            battery_soc_kwh (float, optional): Current battery SOC in kWh (from executor)
            battery_soc_percent (float, optional): Current battery SOC as percentage (from executor)
            
        Returns:
            dict: Event processing result with keys:
                - 'event_case': str ('new_event', 'continue_event', 'event_ended', 'no_event')
                - 'severity_score': float (calculated severity for discharge strategy)
                - 'event_id': int (current event ID)
                - 'event_duration_minutes': float (duration since event start)
                - 'should_calculate_severity': bool (whether severity was calculated)
                - 'discharge_recommendation': dict (recommended discharge strategy)
                
        Example:
            result = decision_maker.four_case_event_logic(
                previous_event_active=False,
                current_event_active=True,
                current_timestamp=timestamp,
                event_state=event_state,
                controller_state=controller_state,
                severity_calculator=severity_calc,
                df_sim=df_sim,
                row_index=i,
                battery_soc_kwh=450.0,  # From battery executor
                battery_soc_percent=75.0  # From battery executor
            )
        """
        # Initialize return structure
        result = {
            'event_case': None,
            'severity_score': 0.0,
            'event_id': 0,
            'event_duration_minutes': 0.0,
            'should_calculate_severity': False,
            'discharge_recommendation': {}
        }
        
        # Create trigger events instance for event management methods
        trigger_events = TriggerEvents()
        
        # Update controller config with current battery SOC if provided
        # This allows severity_params to use actual battery state instead of static config
        if battery_soc_percent is not None:
            severity_calculator.controller.config_data['battery_soc_percent'] = battery_soc_percent
        if battery_soc_kwh is not None:
            severity_calculator.controller.config_data['battery_soc_kwh'] = battery_soc_kwh
        
        # CASE 1: NEW event starts (previous=False, current=True)
        if not previous_event_active and current_event_active:
            # Set event case type
            result['event_case'] = 'new_event'
            
            # Update event state
            event_state.is_event = current_event_active
            event_state.event_id = trigger_events.set_event_id(event_state)
            event_state.start_time = current_timestamp
            trigger_events._set_event_mode(controller_state)
            
            # Store in dataframe
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_start')] = True
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_duration')] = 0.0
            
            # Calculate severity score for new event start
            severity = severity_calculator.severity_score(current_timestamp, event_state, controller_state)
            df_sim.iloc[row_index, df_sim.columns.get_loc('severity_score')] = severity
            
            # Update result
            result['severity_score'] = severity
            result['event_id'] = event_state.event_id
            result['event_duration_minutes'] = 0.0
            result['should_calculate_severity'] = True
            
            # Get discharge recommendation based on severity
            result['discharge_recommendation'] = self._get_discharge_recommendation_from_severity(
                severity, battery_soc_percent
            )
        
        # CASE 2: CONTINUE existing event (previous=True, current=True)
        elif previous_event_active and current_event_active:
            # Set event case type
            result['event_case'] = 'continue_event'
            
            # Update event state
            event_state.is_event = current_event_active
            
            # Calculate duration
            if event_state.start_time:
                time_diff = current_timestamp - event_state.start_time
                event_state.duration_minutes = time_diff.total_seconds() / 60.0
            
            # Store in dataframe
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_start')] = False
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_duration')] = event_state.duration_minutes
            
            # Calculate severity score for continuing event
            severity = severity_calculator.severity_score(current_timestamp, event_state, controller_state)
            df_sim.iloc[row_index, df_sim.columns.get_loc('severity_score')] = severity
            
            # Update result
            result['severity_score'] = severity
            result['event_id'] = event_state.event_id
            result['event_duration_minutes'] = event_state.duration_minutes
            result['should_calculate_severity'] = True
            
            # Get discharge recommendation based on severity
            result['discharge_recommendation'] = self._get_discharge_recommendation_from_severity(
                severity, battery_soc_percent
            )
        
        # CASE 3: Event ENDED (previous=True, current=False)
        elif previous_event_active and not current_event_active:
            # Set event case type
            result['event_case'] = 'event_ended'
            
            # Update event state - reset for next event
            event_state.is_event = current_event_active
            event_state.event_id = 0
            trigger_events._reset_event_statistics(event_state)
            
            # Store in dataframe
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_start')] = False
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_duration')] = 0.0
            df_sim.iloc[row_index, df_sim.columns.get_loc('severity_score')] = 0.0
            
            # Update result - no discharge during non-events
            result['severity_score'] = 0.0
            result['event_id'] = 0
            result['event_duration_minutes'] = 0.0
            result['should_calculate_severity'] = False
            result['discharge_recommendation'] = {
                'action': 'idle',
                'discharge_multiplier': 0.0,
                'conservation_level': 'none',
                'reasoning': 'Event ended - entering idle/charge mode'
            }
        
        # CASE 4: No event (previous=False, current=False)
        else:
            # Set event case type
            result['event_case'] = 'no_event'
            
            # Update event state
            event_state.is_event = current_event_active
            event_state.event_id = 0
            
            # Store in dataframe
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_start')] = False
            df_sim.iloc[row_index, df_sim.columns.get_loc('event_duration')] = 0.0
            df_sim.iloc[row_index, df_sim.columns.get_loc('severity_score')] = 0.0
            
            # Update result - no discharge during non-events
            result['severity_score'] = 0.0
            result['event_id'] = 0
            result['event_duration_minutes'] = 0.0
            result['should_calculate_severity'] = False
            result['discharge_recommendation'] = {
                'action': 'charge',
                'discharge_multiplier': 0.0,
                'conservation_level': 'none',
                'reasoning': 'No event - charging mode if SOC < 95%'
            }
        
        # Derive displayable is_event from event_state.active
        df_sim.iloc[row_index, df_sim.columns.get_loc('is_event')] = event_state.is_event
        
        return result
    
    def _get_discharge_recommendation_from_severity(self, severity_score, battery_soc_percent=None):
        """
        Convert severity score into discharge recommendation for battery executor.
        
        This method translates the severity score (0-5+ range) into actionable
        discharge multipliers using the severity-based conservation strategy.
        
        Args:
            severity_score (float): Calculated severity score
            battery_soc_percent (float, optional): Current battery SOC percentage
            
        Returns:
            dict: Discharge recommendation with keys:
                - 'action': str ('discharge', 'conserve', 'idle')
                - 'discharge_multiplier': float (0.0-1.0, percentage of full discharge)
                - 'conservation_level': str ('normal', 'moderate', 'strong', 'emergency')
                - 'reasoning': str (explanation of recommendation)
        """
        # Severity-based discharge multipliers (from Option B discussion)
        if severity_score < 1.0:
            return {
                'action': 'discharge',
                'discharge_multiplier': 1.0,  # Full discharge capability
                'conservation_level': 'normal',
                'reasoning': f"Low severity ({severity_score:.2f}) - full discharge capacity"
            }
        
        elif 1.0 <= severity_score < 2.0:
            return {
                'action': 'discharge',
                'discharge_multiplier': 0.75,  # Reduce discharge 25%
                'conservation_level': 'moderate',
                'reasoning': f"Moderate severity ({severity_score:.2f}) - conserving 25% capacity"
            }
        
        elif 2.0 <= severity_score < 3.0:
            return {
                'action': 'discharge',
                'discharge_multiplier': 0.50,  # Reduce discharge 50%
                'conservation_level': 'strong',
                'reasoning': f"Strong severity ({severity_score:.2f}) - conserving 50% capacity"
            }
        
        else:  # severity >= 3.0
            # Emergency mode - minimal discharge
            multiplier = 0.25 if battery_soc_percent and battery_soc_percent > 20 else 0.1
            return {
                'action': 'conserve',
                'discharge_multiplier': multiplier,
                'conservation_level': 'emergency',
                'reasoning': f"Emergency severity ({severity_score:.2f}) - conserving 75%+ capacity"
            }

class MdOrchestrator:

    """
    Orchestrator for MD excess demand event analysis.

    This class compiles event status data and computes summary statistics. 

    SmartConservationDebugger calls from here to display data on v3.

    """
    def process_historical_events(self, config_data):
        """
        Process entire df_sim dataset to classify each timestamp as event/non-event.
        
        This method processes the complete historical dataset using the simplified logic:
        - Trigger = target_series (monthly target) 
        - Event = any timestamp where MD excess > 0
        - Assigns sequential event IDs to continuous event periods
        
        Args:
            config_data (dict): Configuration containing df_sim, power_col, target_series
            
        Returns:
            pd.DataFrame: Enhanced dataset with event classification columns:
                - 'excess_demand_kw': calculated excess (current_power - target)
                - 'is_event': boolean event classification (excess > 0)
                - 'event_id': sequential event ID (0 for non-events)
                - 'event_start': boolean marking start of new events
                - 'event_duration': minutes since event started
        """
        import pandas as pd
        
        # Extract required data from config
        df_sim = config_data['df_sim'].copy()
        
        # Use existing MdExcess method to calculate excess demand
        md_excess = MdExcess(config_data)
        excess_demand = md_excess.calculate_excess_demand()
        
        # Add excess demand to the dataframe
        df_sim['excess_demand_kw'] = excess_demand
        
        # Initialize event tracking columns (is_event will be derived from event_state.active)
        df_sim['is_event'] = False
        df_sim['event_id'] = 0
        df_sim['event_start'] = False
        df_sim['event_duration'] = 0.0
        df_sim['severity_score'] = 0.0
        
        # Use existing methods for event processing
        event_state = _MdEventState()
        controller_state = _MdControllerState()

        # Create trigger events instance
        trigger_events = TriggerEvents()

        # create smart constants instance
        sc = SmartConstants()
        
        # Create controller and window type for severity score calculation
        controller = MdShavingController(df_sim)
        controller.import_config(config_data)

        # Create instance of SmartConstants to access RP4 peak methods
        smart_constants = SmartConstants()

        # Create severity score calculator
        severity_calculator = SeverityScore(controller)
        
        # Process each timestamp using existing event management methods
        for i in range(len(df_sim)):
            current_row = df_sim.iloc[i]
            current_timestamp = current_row.name
            current_excess = current_row['excess_demand_kw']
            
            # Get MD window status for this timestamp
            # Use is_md_active() which checks tariff type and RP4 peak period logic
            inside_md_window = sc.is_md_active(current_timestamp, config_data)
               
            # Check if current event is active by calling set_event_state
            current_event_active = trigger_events.set_event_state(current_excess, inside_md_window, event_state)
            
            # Calculate previous_event_active based on row i-1
            if i > 0:
                # Get previous row data
                previous_row = df_sim.iloc[i-1]
                previous_excess = previous_row['excess_demand_kw']
                
                # Get previous MD window status
                previous_timestamp = previous_row.name
                previous_inside_md_window = sc.is_md_active(previous_timestamp, config_data)
                
                # Calculate previous event state using same logic
                previous_event_active = trigger_events.set_event_state(previous_excess, previous_inside_md_window, event_state)
            else:
                # For first row (i=0), there is no previous event
                previous_event_active = False
            
            # Apply concrete 4-case logic for event ID assignment
            if not previous_event_active and current_event_active:
                # Case 1: NEW event starts (previous=False, current=True)
                # Increment event ID and initialize
                event_state.is_event = current_event_active
                event_state.event_id = trigger_events.set_event_id(event_state)
                event_state.start_time = current_timestamp
                trigger_events._set_event_mode(controller_state)
                df_sim.iloc[i, df_sim.columns.get_loc('event_start')] = True
                df_sim.iloc[i, df_sim.columns.get_loc('event_duration')] = 0.0
                
                # Calculate severity score for new event start
                severity = severity_calculator.severity_score(current_timestamp, event_state, controller_state)
                df_sim.iloc[i, df_sim.columns.get_loc('severity_score')] = severity
                
            elif previous_event_active and current_event_active:
                # Case 2: CONTINUE existing event (previous=True, current=True)
                # Calculate duration, maintain same event_id
                event_state.is_event = current_event_active
               
                if event_state.start_time:
                    time_diff = current_timestamp - event_state.start_time
                    event_state.duration_minutes = time_diff.total_seconds() / 60.0
                df_sim.iloc[i, df_sim.columns.get_loc('event_start')] = False
                df_sim.iloc[i, df_sim.columns.get_loc('event_duration')] = event_state.duration_minutes
                
                # Calculate severity score for continuing event
                severity = severity_calculator.severity_score(current_timestamp, event_state, controller_state)
                df_sim.iloc[i, df_sim.columns.get_loc('severity_score')] = severity
                
            elif previous_event_active and not current_event_active:
                # Case 3: Event ENDED (previous=True, current=False)
                # Reset event_id to 0, but keep persistent counter for next event
                event_state.is_event = current_event_active
                event_state.event_id = 0
                trigger_events._reset_event_statistics(event_state)
                df_sim.iloc[i, df_sim.columns.get_loc('event_start')] = False
                df_sim.iloc[i, df_sim.columns.get_loc('event_duration')] = 0.0
                df_sim.iloc[i, df_sim.columns.get_loc('severity_score')] = 0.0
                
            else:
                # Case 4: No event (previous=False, current=False)
                # Keep event_id at 0, no change to persistent counter
                event_state.is_event = current_event_active
                event_state.event_id = 0
                df_sim.iloc[i, df_sim.columns.get_loc('event_start')] = False
                df_sim.iloc[i, df_sim.columns.get_loc('event_duration')] = 0.0
                df_sim.iloc[i, df_sim.columns.get_loc('severity_score')] = 0.0
          
            # Derive the displayable is_event from event_state.active
            df_sim.iloc[i, df_sim.columns.get_loc('is_event')] = event_state.is_event
            
            # Append all relevant data to df_sim
            df_sim.iloc[i, df_sim.columns.get_loc('event_id')] = event_state.event_id

        return df_sim
    
    def process_events_with_battery_state(self, config_data, initial_soc_percent=95.0):
        """
        Process historical events using DecisionMaker.four_case_event_logic() with battery state tracking.
        
        This orchestrator method reproduces the logic of process_historical_events() but delegates
        the four-case conditional logic to DecisionMaker.four_case_event_logic(). This enables
        integration with the Smart Battery Executor by passing dynamic battery SOC state.
        
        Key differences from process_historical_events():
        - Uses DecisionMaker.four_case_event_logic() instead of inline 4-case logic
        - Tracks battery SOC state across timestamps (placeholder for now)
        - Passes battery state to severity calculation for dynamic recommendations
        - Prepares for future Smart Battery Executor integration
        
        Args:
            config_data (dict): Configuration containing df_sim, power_col, target_series, battery params
            initial_soc_percent (float, optional): Starting battery SOC percentage (default: 95.0)
            
        Returns:
            pd.DataFrame: Enhanced dataset with event classification columns:
                - 'excess_demand_kw': calculated excess (current_power - target)
                - 'is_event': boolean event classification (excess > 0)
                - 'event_id': sequential event ID (0 for non-events)
                - 'event_start': boolean marking start of new events
                - 'event_duration': minutes since event started
                - 'severity_score': calculated severity score
                - 'battery_soc_percent': tracked battery SOC (placeholder)
        """
        import pandas as pd
        
        # Extract required data from config
        df_sim = config_data['df_sim'].copy()
        
        # Use existing MdExcess method to calculate excess demand
        md_excess = MdExcess(config_data)
        excess_demand = md_excess.calculate_excess_demand()
        
        # Add excess demand to the dataframe
        df_sim['excess_demand_kw'] = excess_demand
        
        # Initialize event tracking columns
        df_sim['is_event'] = False
        df_sim['event_id'] = 0
        df_sim['event_start'] = False
        df_sim['event_duration'] = 0.0
        df_sim['severity_score'] = 0.0
        df_sim['battery_soc_kwh'] = 0.0  # Track battery SOC in kWh
        df_sim['battery_soc_percent'] = 0.0  # Track battery SOC in percentage
        
        # Initialize state objects
        event_state = _MdEventState()
        controller_state = _MdControllerState()
        
        # Create required instances
        trigger_events = TriggerEvents()
        sc = SmartConstants()
        
        # Create controller for severity calculation
        controller = MdShavingController(df_sim)
        controller.import_config(config_data)
        
        # Create severity score calculator
        severity_calculator = SeverityScore(controller)
        
        # Create decision maker instance
        decision_maker = DecisionMaker()
        
        # Initialize battery state tracking
        battery_capacity_kwh = config_data.get('battery_capacity', 600.0)  # Default 600 kWh
        current_soc_kwh = battery_capacity_kwh * (initial_soc_percent / 100.0)
        current_soc_percent = initial_soc_percent
        
        # Process each timestamp using DecisionMaker.four_case_event_logic()
        for i in range(len(df_sim)):
            current_row = df_sim.iloc[i]
            current_timestamp = current_row.name
            current_excess = current_row['excess_demand_kw']
            
            # Get MD window status for this timestamp
            inside_md_window = sc.is_md_active(current_timestamp, config_data)
            
            # Check if current event is active
            current_event_active = trigger_events.set_event_state(current_excess, inside_md_window, event_state)
            
            # Calculate previous_event_active
            if i > 0:
                previous_row = df_sim.iloc[i-1]
                previous_excess = previous_row['excess_demand_kw']
                previous_timestamp = previous_row.name
                previous_inside_md_window = sc.is_md_active(previous_timestamp, config_data)
                previous_event_active = trigger_events.set_event_state(previous_excess, previous_inside_md_window, event_state)
            else:
                previous_event_active = False
            
            # Call DecisionMaker.four_case_event_logic() instead of inline 4-case logic
            event_result = decision_maker.four_case_event_logic(
                previous_event_active=previous_event_active,
                current_event_active=current_event_active,
                current_timestamp=current_timestamp,
                event_state=event_state,
                controller_state=controller_state,
                severity_calculator=severity_calculator,
                df_sim=df_sim,
                row_index=i,
                battery_soc_kwh=current_soc_kwh,      # Pass current battery state
                battery_soc_percent=current_soc_percent  # Pass current SOC percentage
            )
            
            # Store battery SOC in dataframe
            df_sim.iloc[i, df_sim.columns.get_loc('battery_soc_kwh')] = current_soc_kwh
            df_sim.iloc[i, df_sim.columns.get_loc('battery_soc_percent')] = current_soc_percent
            
            # Store event_id in dataframe
            df_sim.iloc[i, df_sim.columns.get_loc('event_id')] = event_result['event_id']
            
            # TODO: Update battery SOC based on discharge/charge actions
            # This will be implemented when Smart Battery Executor is integrated
            # For now, SOC remains constant (placeholder)
            # Future implementation will call Smart Battery Executor here to:
            # 1. Execute discharge based on event_result['discharge_recommendation']
            # 2. Update current_soc_kwh and current_soc_percent
            # 3. Track energy discharged/charged
        
        return df_sim