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
        self.event_state = None
        self.mode_state = None
        self.config_data = None

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

