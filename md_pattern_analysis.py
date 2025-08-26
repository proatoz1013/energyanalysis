"""
MD Pattern Analysis Module
==========================

This module provides comprehensive Maximum Demand (MD) pattern analysis with:
- Multi-format file upload support (CSV, Excel .xls/.xlsx)
- Advanced pattern recognition and identification
- Load profile trend analysis
- Peak demand timing analysis
- Monthly and seasonal pattern detection
- Statistical analysis of demand patterns
- Comprehensive RP4 General and TOU tariff integration
- Malaysia holiday compliance with auto-detection
- Enhanced tariff selection interface with Medium Voltage TOU default
- Integrated MD shaving difficulty scoring with RP4 peak logic

Features:
- ✅ RP4 Peak Period Detection (2PM-10PM weekdays)
- ✅ Malaysia Public Holiday Integration
- ✅ TOU vs General Tariff Differentiation
- ✅ Comprehensive Tariff Selection Interface
- ✅ Holiday vs Non-Holiday Demand Analysis
- ✅ Peak/Off-Peak Ratio Calculations
- ✅ RP4-Aware MD Difficulty Scoring
- ✅ Cost Analysis with Current RP4 Rates
- ✅ Pattern Visualizations with RP4 Overlay

Author: Energy Analysis Team
Version: 1.2 (Complete RP4 Integration with Enhanced Tariff Selection)
Date: January 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import calendar
from typing import Optional, Dict, List, Tuple

# --- Reuse shared logic when available ---
try:
    from md_shaving_solution_v2 import _process_dataframe as _process_df_v2
    from md_shaving_solution_v2 import render_md_shaving_v2
except Exception:
    _process_df_v2 = None
    render_md_shaving_v2 = None

try:
    from tariffs.peak_logic import (
        is_peak_rp4, 
        get_period_classification, 
        get_malaysia_holidays,
        detect_holidays_from_data,
        classify_peak_period,
        is_public_holiday
    )
    from tariffs.rp4_tariffs import TARIFF_DATA
except Exception:
    is_peak_rp4 = None
    get_period_classification = None
    get_malaysia_holidays = None
    detect_holidays_from_data = None
    classify_peak_period = None
    is_public_holiday = None
    TARIFF_DATA = None

try:
    from battery_algorithms import get_battery_parameters_ui, perform_comprehensive_battery_analysis
except Exception:
    get_battery_parameters_ui = None
    perform_comprehensive_battery_analysis = None

try:
    from advanced_energy_analysis import _detect_peak_events as _detect_peak_events_advanced
except Exception:
    _detect_peak_events_advanced = None


def read_uploaded_file(file) -> pd.DataFrame:
    """Read uploaded file based on its extension with comprehensive error handling."""
    try:
        if file.name.endswith('.csv'):
            # Try different encodings for CSV files
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)  # Reset file pointer
                return pd.read_csv(file, encoding='latin-1')
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame()


def detect_data_interval(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Detect the data interval from timestamp differences.
    
    Returns:
        Tuple of (interval_minutes, interval_hours, consistency_percentage)
    """
    if len(df) < 2:
        return 15.0, 0.25, 0.0
    
    time_diffs = df.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return 15.0, 0.25, 0.0
    
    # Get the most common time interval
    most_common_interval = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=15)
    interval_minutes = most_common_interval.total_seconds() / 60
    interval_hours = most_common_interval.total_seconds() / 3600
    
    # Check consistency
    unique_intervals = time_diffs.value_counts()
    consistency_percentage = (unique_intervals.iloc[0] / len(time_diffs)) * 100 if len(unique_intervals) > 0 else 100
    
    return interval_minutes, interval_hours, consistency_percentage


def analyze_daily_patterns(df: pd.DataFrame, power_col: str) -> Dict:
    """Analyze daily demand patterns."""
    
    # Add time-based columns
    df_analysis = df.copy()
    df_analysis['Hour'] = df_analysis.index.hour
    df_analysis['DayOfWeek'] = df_analysis.index.dayofweek
    df_analysis['Month'] = df_analysis.index.month
    df_analysis['Date'] = df_analysis.index.date
    
    # Daily pattern analysis
    hourly_avg = df_analysis.groupby('Hour')[power_col].agg(['mean', 'min', 'max', 'std']).round(2)
    
    # Weekly pattern analysis
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = df_analysis.groupby('DayOfWeek')[power_col].agg(['mean', 'min', 'max', 'std']).round(2)
    weekly_avg.index = [weekday_names[i] for i in weekly_avg.index]
    
    # Monthly pattern analysis
    monthly_avg = df_analysis.groupby('Month')[power_col].agg(['mean', 'min', 'max', 'std']).round(2)
    monthly_avg.index = [calendar.month_name[i] for i in monthly_avg.index]
    
    # Peak hours identification
    peak_hours = hourly_avg.nlargest(5, 'mean').index.tolist()
    off_peak_hours = hourly_avg.nsmallest(5, 'mean').index.tolist()
    
    return {
        'hourly_patterns': hourly_avg,
        'weekly_patterns': weekly_avg,
        'monthly_patterns': monthly_avg,
        'peak_hours': peak_hours,
        'off_peak_hours': off_peak_hours,
        'analysis_df': df_analysis
    }


def identify_peak_events(df: pd.DataFrame, power_col: str, threshold_percentile: float = 95) -> pd.DataFrame:
    """Identify peak demand events based on percentile threshold."""
    
    threshold_value = np.percentile(df[power_col], threshold_percentile)
    
    # Find peak events
    peak_mask = df[power_col] > threshold_value
    
    # Group consecutive peak periods
    df_peaks = df[peak_mask].copy()
    
    if df_peaks.empty:
        return pd.DataFrame()
    
    # Create events by grouping consecutive time periods
    events = []
    current_event_start = None
    current_event_data = []
    
    for idx, (timestamp, row) in enumerate(df_peaks.iterrows()):
        if current_event_start is None:
            # Start new event
            current_event_start = timestamp
            current_event_data = [row]
        else:
            # Check if this continues the current event (within 1 hour gap)
            time_gap = timestamp - df_peaks.index[idx-1]
            if time_gap <= pd.Timedelta(hours=1):
                current_event_data.append(row)
            else:
                # End current event and start new one
                if current_event_data:
                    events.append(_create_event_summary(current_event_start, 
                                                      df_peaks.index[idx-1], 
                                                      current_event_data, 
                                                      power_col))
                current_event_start = timestamp
                current_event_data = [row]
    
    # Don't forget the last event
    if current_event_data:
        events.append(_create_event_summary(current_event_start, 
                                          df_peaks.index[-1], 
                                          current_event_data, 
                                          power_col))
    
    return pd.DataFrame(events) if events else pd.DataFrame()


def _create_event_summary(start_time: pd.Timestamp, end_time: pd.Timestamp, 
                         event_data: List, power_col: str) -> Dict:
    """Create summary for a peak event."""
    power_values = [row[power_col] for row in event_data]
    
    return {
        'Start Time': start_time,
        'End Time': end_time,
        'Duration (hours)': (end_time - start_time).total_seconds() / 3600,
        'Peak Demand (kW)': max(power_values),
        'Average Demand (kW)': np.mean(power_values),
        'Min Demand (kW)': min(power_values),
        'Date': start_time.date(),
        'Start Hour': start_time.hour,
        'Day of Week': start_time.strftime('%A')
    }


def prepare_timeseries_reliably(df: pd.DataFrame, timestamp_col: str, power_col: str) -> pd.DataFrame:
    """
    Prepare timeseries data reliably with comprehensive validation and processing.
    
    Returns:
        Processed DataFrame with datetime index and validated power data
    """
    try:
        # Use existing v2 processing if available
        if _process_df_v2 is not None:
            processed_df = _process_df_v2(df.copy(), timestamp_col)
            if not processed_df.empty and power_col in processed_df.columns:
                return processed_df
        
        # Fallback to local processing
        df_clean = df.copy()
        
        # Convert timestamp column
        df_clean['parsed_timestamp'] = pd.to_datetime(df_clean[timestamp_col], errors='coerce')
        
        # Remove rows with invalid timestamps or power values
        df_clean = df_clean.dropna(subset=['parsed_timestamp', power_col])
        
        # Set timestamp as index
        df_clean.set_index('parsed_timestamp', inplace=True)
        df_clean.sort_index(inplace=True)
        
        # Validate power column is numeric
        df_clean[power_col] = pd.to_numeric(df_clean[power_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[power_col])
        
        # Remove obvious outliers (values outside 3 standard deviations)
        power_mean = df_clean[power_col].mean()
        power_std = df_clean[power_col].std()
        outlier_threshold = power_mean + (3 * power_std)
        df_clean = df_clean[df_clean[power_col] <= outlier_threshold]
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error preparing timeseries data: {str(e)}")
        return pd.DataFrame()


def compute_monthly_baseline_features_with_rp4(df: pd.DataFrame, power_col: str, 
                                            selected_tariff: Optional[Dict] = None) -> Dict:
    """
    Enhanced monthly baseline features computation with comprehensive RP4 integration.
    
    Args:
        df: Processed DataFrame with datetime index
        power_col: Power column name  
        selected_tariff: Selected tariff information (General vs TOU)
    
    Returns:
        Dictionary containing monthly statistics and RP4-aware features
    """
    if df.empty:
        return {}
    
    # Auto-detect holidays from data years
    holidays = set()
    if detect_holidays_from_data is not None:
        temp_df = df.reset_index()
        holidays = detect_holidays_from_data(temp_df, temp_df.columns[0])
    
    # Group by month
    df_monthly = df.groupby(df.index.to_period('M'))
    
    monthly_features = {}
    
    for month_period, month_data in df_monthly:
        month_str = str(month_period)
        
        # Basic statistics
        power_values = month_data[power_col]
        
        features = {
            'month_period': month_period,
            'data_points': len(power_values),
            'mean_demand': power_values.mean(),
            'max_demand': power_values.max(),
            'min_demand': power_values.min(),
            'std_demand': power_values.std(),
            'p95_demand': power_values.quantile(0.95),
            'p90_demand': power_values.quantile(0.90),
            'p75_demand': power_values.quantile(0.75),
            'load_factor': power_values.mean() / power_values.max() if power_values.max() > 0 else 0,
        }
        
        # Enhanced RP4 peak period analysis
        if is_peak_rp4 is not None and holidays is not None:
            try:
                # Classify each timestamp using proper holidays
                peak_mask = month_data.index.map(lambda x: is_peak_rp4(x, holidays))
                peak_data = month_data[peak_mask]
                off_peak_data = month_data[~peak_mask]
                
                # RP4-specific features
                features.update({
                    'rp4_peak_max': peak_data[power_col].max() if not peak_data.empty else 0,
                    'rp4_peak_mean': peak_data[power_col].mean() if not peak_data.empty else 0,
                    'rp4_peak_std': peak_data[power_col].std() if not peak_data.empty else 0,
                    'rp4_off_peak_max': off_peak_data[power_col].max() if not off_peak_data.empty else 0,
                    'rp4_off_peak_mean': off_peak_data[power_col].mean() if not off_peak_data.empty else 0,
                    'rp4_peak_to_offpeak_ratio': (peak_data[power_col].max() / off_peak_data[power_col].max()) if not off_peak_data.empty and off_peak_data[power_col].max() > 0 else 1,
                    'rp4_peak_hours_count': len(peak_data),
                    'rp4_off_peak_hours_count': len(off_peak_data),
                    'rp4_peak_load_factor': peak_data[power_col].mean() / peak_data[power_col].max() if not peak_data.empty and peak_data[power_col].max() > 0 else 0,
                })
                
                # Tariff-specific analysis
                if selected_tariff:
                    tariff_type = selected_tariff.get('Type', 'General')
                    
                    if tariff_type == 'TOU':
                        # TOU-specific features
                        features.update({
                            'tou_peak_energy': peak_data[power_col].sum() if not peak_data.empty else 0,
                            'tou_off_peak_energy': off_peak_data[power_col].sum() if not off_peak_data.empty else 0,
                            'tou_energy_ratio': (peak_data[power_col].sum() / off_peak_data[power_col].sum()) if not off_peak_data.empty and off_peak_data[power_col].sum() > 0 else 1,
                        })
                    
                    # MD charge considerations
                    if 'Rates' in selected_tariff:
                        rates = selected_tariff['Rates']
                        if 'Capacity Rate' in rates:
                            features['md_charge_rate'] = rates['Capacity Rate']
                            features['monthly_md_cost'] = features['max_demand'] * rates['Capacity Rate']
                            
                            if 'rp4_peak_max' in features:
                                features['peak_period_md_cost'] = features['rp4_peak_max'] * rates['Capacity Rate']
                
            except Exception as e:
                st.warning(f"RP4 analysis failed for {month_str}: {str(e)}")
                # Fallback to simple time-based analysis
                business_hours = month_data.between_time('08:00', '18:00')
                features.update({
                    'peak_period_max': business_hours[power_col].max() if not business_hours.empty else 0,
                    'peak_period_mean': business_hours[power_col].mean() if not business_hours.empty else 0,
                })
        else:
            # Fallback when RP4 logic not available
            business_hours = month_data.between_time('08:00', '18:00')
            features.update({
                'peak_period_max': business_hours[power_col].max() if not business_hours.empty else 0,
                'peak_period_mean': business_hours[power_col].mean() if not business_hours.empty else 0,
            })
        
        # Enhanced variability metrics with RP4 context
        features['coefficient_of_variation'] = features['std_demand'] / features['mean_demand'] if features['mean_demand'] > 0 else 0
        features['peak_diversity'] = 1 - (features['mean_demand'] / features['max_demand']) if features['max_demand'] > 0 else 0
        
        # Time-based patterns
        hourly_max = month_data.groupby(month_data.index.hour)[power_col].max()
        features['peak_hour_concentration'] = (hourly_max.max() - hourly_max.median()) / hourly_max.max() if hourly_max.max() > 0 else 0
        
        # Holiday vs non-holiday analysis
        if holidays:
            try:
                holiday_mask = month_data.index.map(lambda x: is_public_holiday(x, holidays) if is_public_holiday else False)
                non_holiday_data = month_data[~holiday_mask]
                holiday_data = month_data[holiday_mask]
                
                features.update({
                    'non_holiday_max': non_holiday_data[power_col].max() if not non_holiday_data.empty else 0,
                    'holiday_max': holiday_data[power_col].max() if not holiday_data.empty else 0,
                    'holiday_impact': (non_holiday_data[power_col].max() / holiday_data[power_col].max()) if not holiday_data.empty and holiday_data[power_col].max() > 0 else 1,
                })
            except Exception:
                pass  # Skip if holiday analysis fails
        
        monthly_features[month_str] = features
    
    return monthly_features


def compute_monthly_baseline_features(df: pd.DataFrame, power_col: str) -> Dict:
    """
    Compute comprehensive monthly baseline features for MD analysis.
    
    Returns:
        Dictionary containing monthly statistics and features
    """
    if df.empty:
        return {}
    
    # Group by month
    df_monthly = df.groupby(df.index.to_period('M'))
    
    monthly_features = {}
    
    for month_period, month_data in df_monthly:
        month_str = str(month_period)
        
        # Basic statistics
        power_values = month_data[power_col]
        
        features = {
            'month_period': month_period,
            'data_points': len(power_values),
            'mean_demand': power_values.mean(),
            'max_demand': power_values.max(),
            'min_demand': power_values.min(),
            'std_demand': power_values.std(),
            'p95_demand': power_values.quantile(0.95),
            'p90_demand': power_values.quantile(0.90),
            'p75_demand': power_values.quantile(0.75),
            'load_factor': power_values.mean() / power_values.max() if power_values.max() > 0 else 0,
        }
        
        # Peak period analysis (if RP4 logic available)
        if is_peak_rp4 is not None:
            try:
                peak_mask = month_data.index.map(lambda x: is_peak_rp4(x, holidays=[]))
                peak_data = month_data[peak_mask]
                off_peak_data = month_data[~peak_mask]
                
                features.update({
                    'peak_period_max': peak_data[power_col].max() if not peak_data.empty else 0,
                    'peak_period_mean': peak_data[power_col].mean() if not peak_data.empty else 0,
                    'off_peak_max': off_peak_data[power_col].max() if not off_peak_data.empty else 0,
                    'peak_to_offpeak_ratio': (peak_data[power_col].max() / off_peak_data[power_col].max()) if not off_peak_data.empty and off_peak_data[power_col].max() > 0 else 1,
                })
            except Exception:
                # Fallback to simple time-based peak detection
                business_hours = month_data.between_time('08:00', '18:00')
                features.update({
                    'peak_period_max': business_hours[power_col].max() if not business_hours.empty else 0,
                    'peak_period_mean': business_hours[power_col].mean() if not business_hours.empty else 0,
                })
        
        # Variability metrics
        features['coefficient_of_variation'] = features['std_demand'] / features['mean_demand'] if features['mean_demand'] > 0 else 0
        features['peak_diversity'] = 1 - (features['mean_demand'] / features['max_demand']) if features['max_demand'] > 0 else 0
        
        # Time-based patterns
        hourly_max = month_data.groupby(month_data.index.hour)[power_col].max()
        features['peak_hour_concentration'] = (hourly_max.max() - hourly_max.median()) / hourly_max.max() if hourly_max.max() > 0 else 0
        
        monthly_features[month_str] = features
    
    return monthly_features


def tag_load_profile_patterns(monthly_features: Dict) -> Dict:
    """
    Tag load profile patterns using rule-based classification.
    
    Returns:
        Dictionary with pattern tags for each month
    """
    pattern_tags = {}
    
    for month_str, features in monthly_features.items():
        tags = []
        
        # Load factor based patterns
        load_factor = features.get('load_factor', 0)
        if load_factor > 0.8:
            tags.append('BASE_LOAD')
        elif load_factor > 0.6:
            tags.append('MODERATE_VARIABILITY')
        else:
            tags.append('HIGH_VARIABILITY')
        
        # Peak concentration patterns
        peak_concentration = features.get('peak_hour_concentration', 0)
        if peak_concentration > 0.3:
            tags.append('CONCENTRATED_PEAKS')
        elif peak_concentration > 0.15:
            tags.append('MODERATE_PEAKS')
        else:
            tags.append('DISTRIBUTED_LOAD')
        
        # Coefficient of variation patterns
        cv = features.get('coefficient_of_variation', 0)
        if cv > 0.4:
            tags.append('HIGHLY_VARIABLE')
        elif cv > 0.2:
            tags.append('MODERATELY_STABLE')
        else:
            tags.append('VERY_STABLE')
        
        # Peak diversity patterns
        peak_diversity = features.get('peak_diversity', 0)
        if peak_diversity > 0.6:
            tags.append('SPIKY_PROFILE')
        elif peak_diversity > 0.3:
            tags.append('MODERATE_SPIKES')
        else:
            tags.append('FLAT_PROFILE')
        
        # Peak to off-peak ratio (if available)
        peak_ratio = features.get('peak_to_offpeak_ratio', 1)
        if peak_ratio > 1.5:
            tags.append('STRONG_TOU_PATTERN')
        elif peak_ratio > 1.2:
            tags.append('MODERATE_TOU_PATTERN')
        else:
            tags.append('WEAK_TOU_PATTERN')
        
        pattern_tags[month_str] = tags
    
    return pattern_tags


def detect_contiguous_peak_events_vs_monthly_targets(df: pd.DataFrame, power_col: str, 
                                                   monthly_features: Dict, 
                                                   shaving_percentage: float = 15.0) -> Dict:
    """
    Detect contiguous peak events versus monthly MD targets.
    
    Args:
        df: Processed DataFrame with datetime index
        power_col: Power column name
        monthly_features: Monthly baseline features
        shaving_percentage: Target shaving percentage (default 15%)
    
    Returns:
        Dictionary containing peak events analysis for each month
    """
    monthly_events = {}
    
    # Group by month
    df_monthly = df.groupby(df.index.to_period('M'))
    
    for month_period, month_data in df_monthly:
        month_str = str(month_period)
        
        if month_str not in monthly_features:
            continue
        
        # Calculate monthly target based on shaving percentage
        monthly_max = monthly_features[month_str]['max_demand']
        target_demand = monthly_max * (1 - shaving_percentage / 100)
        
        # Find peak events above target
        peak_mask = month_data[power_col] > target_demand
        
        if not peak_mask.any():
            monthly_events[month_str] = {
                'target_demand': target_demand,
                'events': [],
                'total_events': 0,
                'total_duration_hours': 0,
                'max_excess_kw': 0,
                'total_energy_above_target': 0
            }
            continue
        
        # Group contiguous peak events
        peak_data = month_data[peak_mask]
        events = []
        
        if not peak_data.empty:
            # Detect contiguous events (within 1 hour gaps)
            event_groups = []
            current_group = [peak_data.index[0]]
            
            for i in range(1, len(peak_data.index)):
                time_gap = peak_data.index[i] - peak_data.index[i-1]
                if time_gap <= pd.Timedelta(hours=1):
                    current_group.append(peak_data.index[i])
                else:
                    event_groups.append(current_group)
                    current_group = [peak_data.index[i]]
            
            if current_group:
                event_groups.append(current_group)
            
            # Create event summaries
            for i, group in enumerate(event_groups):
                if len(group) > 0:
                    start_time = group[0]
                    end_time = group[-1]
                    event_data = month_data.loc[group]
                    
                    peak_power = event_data[power_col].max()
                    avg_power = event_data[power_col].mean()
                    excess_kw = peak_power - target_demand
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    
                    # Calculate energy above target
                    energy_above = ((event_data[power_col] - target_demand) * 
                                  (event_data.index.to_series().diff().dt.total_seconds() / 3600)).sum()
                    
                    events.append({
                        'event_id': i + 1,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_hours': duration_hours,
                        'peak_demand_kw': peak_power,
                        'average_demand_kw': avg_power,
                        'excess_kw': excess_kw,
                        'energy_above_target_kwh': max(0, energy_above),
                        'target_demand': target_demand
                    })
        
        # Calculate summary statistics
        total_energy_above = sum([e['energy_above_target_kwh'] for e in events])
        max_excess = max([e['excess_kw'] for e in events]) if events else 0
        total_duration = sum([e['duration_hours'] for e in events])
        
        monthly_events[month_str] = {
            'target_demand': target_demand,
            'events': events,
            'total_events': len(events),
            'total_duration_hours': total_duration,
            'max_excess_kw': max_excess,
            'total_energy_above_target': total_energy_above
        }
    
    return monthly_events


def compute_md_shaving_difficulty_score_with_rp4(monthly_features: Dict, monthly_events: Dict, 
                                               pattern_tags: Dict, selected_tariff: Optional[Dict] = None) -> Dict:
    """
    Enhanced MD-shaving difficulty score computation with RP4 tariff considerations.
    
    Returns:
        Dictionary with difficulty scores, grades, and RP4-specific insights
    """
    difficulty_analysis = {}
    
    for month_str in monthly_features.keys():
        features = monthly_features[month_str]
        events = monthly_events.get(month_str, {})
        tags = pattern_tags.get(month_str, [])
        
        # Initialize base score
        difficulty_score = 0
        score_components = {}
        rp4_insights = {}
        
        # Component 1: Load Factor (0-25 points)
        # Use RP4 peak load factor if available, otherwise general load factor
        load_factor = features.get('rp4_peak_load_factor', features.get('load_factor', 0))
        load_factor_score = max(0, 25 * (1 - load_factor))
        difficulty_score += load_factor_score
        score_components['load_factor'] = load_factor_score
        
        # Component 2: Peak Concentration (0-20 points) - Enhanced with RP4
        peak_concentration = features.get('peak_hour_concentration', 0)
        
        # Bonus difficulty for high RP4 peak-to-off-peak ratios
        rp4_ratio = features.get('rp4_peak_to_offpeak_ratio', 1)
        if rp4_ratio > 1.5:
            concentration_bonus = 5  # Extra difficulty for high TOU differentiation
        else:
            concentration_bonus = 0
        
        concentration_score = min(20, 20 * peak_concentration + concentration_bonus)
        difficulty_score += concentration_score
        score_components['peak_concentration'] = concentration_score
        
        # Component 3: Variability (0-20 points)
        cv = features.get('coefficient_of_variation', 0)
        variability_score = min(20, 20 * cv / 0.5)
        difficulty_score += variability_score
        score_components['variability'] = variability_score
        
        # Component 4: Peak Events Frequency (0-15 points)
        total_events = events.get('total_events', 0)
        data_days = features.get('data_points', 0) / 96
        event_frequency = total_events / max(1, data_days) * 30
        frequency_score = min(15, event_frequency * 2)
        difficulty_score += frequency_score
        score_components['event_frequency'] = frequency_score
        
        # Component 5: Peak Severity (0-10 points) - Enhanced with RP4
        max_excess = events.get('max_excess_kw', 0)
        monthly_max = features.get('max_demand', 1)
        
        # Use RP4 peak max if available for more accurate assessment
        rp4_peak_max = features.get('rp4_peak_max', monthly_max)
        peak_reference = max(rp4_peak_max, monthly_max)
        
        excess_ratio = max_excess / peak_reference if peak_reference > 0 else 0
        severity_score = min(10, 10 * excess_ratio / 0.3)
        difficulty_score += severity_score
        score_components['peak_severity'] = severity_score
        
        # Component 6: Pattern Complexity (0-10 points) - Enhanced with RP4
        complexity_score = 0
        if 'SPIKY_PROFILE' in tags:
            complexity_score += 4
        if 'HIGH_VARIABILITY' in tags:
            complexity_score += 3
        if 'CONCENTRATED_PEAKS' in tags:
            complexity_score += 3
        
        # RP4-specific complexity considerations
        if rp4_ratio > 2.0:  # Very high peak/off-peak differential
            complexity_score += 2
        
        difficulty_score += complexity_score
        score_components['pattern_complexity'] = complexity_score
        
        # Normalize to 0-100 scale
        difficulty_score = min(100, difficulty_score)
        
        # Assign letter grade
        if difficulty_score >= 85:
            grade = 'F'
            grade_description = 'Extremely Difficult'
        elif difficulty_score >= 75:
            grade = 'D'
            grade_description = 'Very Difficult'
        elif difficulty_score >= 65:
            grade = 'C'
            grade_description = 'Difficult'
        elif difficulty_score >= 55:
            grade = 'C+'
            grade_description = 'Moderately Difficult'
        elif difficulty_score >= 45:
            grade = 'B'
            grade_description = 'Moderate'
        elif difficulty_score >= 35:
            grade = 'B+'
            grade_description = 'Manageable'
        elif difficulty_score >= 25:
            grade = 'A'
            grade_description = 'Easy'
        else:
            grade = 'A+'
            grade_description = 'Very Easy'
        
        # RP4-specific insights
        if 'rp4_peak_max' in features:
            rp4_insights = {
                'rp4_compliance': True,
                'rp4_peak_max': features['rp4_peak_max'],
                'rp4_off_peak_max': features.get('rp4_off_peak_max', 0),
                'peak_to_offpeak_ratio': rp4_ratio,
                'peak_hours_percentage': (features.get('rp4_peak_hours_count', 0) / features.get('data_points', 1)) * 100,
                'tariff_type': selected_tariff.get('Type', 'General') if selected_tariff else 'Unknown',
            }
            
            # Holiday impact analysis
            if 'holiday_impact' in features:
                rp4_insights['holiday_impact_ratio'] = features['holiday_impact']
        
        difficulty_analysis[month_str] = {
            'difficulty_score': round(difficulty_score, 1),
            'grade': grade,
            'grade_description': grade_description,
            'score_components': score_components,
            'rp4_insights': rp4_insights,
            'recommendations': _generate_rp4_difficulty_recommendations(
                difficulty_score, tags, score_components, rp4_insights, selected_tariff
            )
        }
    
    return difficulty_analysis


def _generate_rp4_difficulty_recommendations(score: float, tags: List[str], components: Dict, 
                                           rp4_insights: Dict, selected_tariff: Optional[Dict] = None) -> List[str]:
    """Generate RP4-aware recommendations based on difficulty analysis."""
    recommendations = []
    
    # Base recommendations by difficulty score
    if score >= 75:
        recommendations.append("🔋 Consider advanced battery energy storage systems with 4+ hour duration")
        recommendations.append("⚡ Implement demand response programs with automated load shedding")
        recommendations.append("🔄 Evaluate comprehensive load shifting to off-peak hours")
        if rp4_insights.get('rp4_compliance'):
            recommendations.append("📊 Critical: RP4 peak periods show very high demand - urgent intervention needed")
    elif score >= 50:
        recommendations.append("🔋 Standard battery systems (2-3 hour duration) should be effective")
        recommendations.append("📈 Implement peak demand monitoring and alerts")
        recommendations.append("💰 Consider time-of-use tariff optimization")
        if rp4_insights.get('rp4_compliance'):
            recommendations.append("⏰ Focus MD shaving during RP4 peak hours (2PM-10PM, weekdays)")
    else:
        recommendations.append("🎯 Simple demand management strategies may be sufficient")
        recommendations.append("⚙️ Focus on operational efficiency improvements")
        recommendations.append("📊 Monitor for seasonal pattern changes")
    
    # RP4-specific recommendations
    if rp4_insights.get('rp4_compliance'):
        tariff_type = rp4_insights.get('tariff_type', 'General')
        peak_ratio = rp4_insights.get('peak_to_offpeak_ratio', 1)
        
        if tariff_type == 'TOU' and peak_ratio > 1.5:
            recommendations.append("💡 Strong TOU pattern detected - excellent candidate for RP4 TOU tariff")
            recommendations.append("🔄 Prioritize load shifting from peak (2PM-10PM) to off-peak periods")
            recommendations.append("🏭 Consider automated systems to avoid peak period operations")
        elif tariff_type == 'General':
            recommendations.append("📋 Currently on RP4 General - evaluate TOU tariff for potential savings")
            recommendations.append("⏰ MD occurs during RP4 peak hours - focus on 2PM-10PM demand reduction")
        
        # Holiday impact recommendations
        holiday_impact = rp4_insights.get('holiday_impact_ratio', 1)
        if holiday_impact > 1.3:
            recommendations.append("🎉 Lower demand on holidays - potential for holiday-specific strategies")
    
    # Pattern-specific recommendations
    if 'SPIKY_PROFILE' in tags:
        recommendations.append("🎯 Spiky profile detected - focus on peak clipping rather than load shifting")
    if 'BASE_LOAD' in tags:
        recommendations.append("⚖️ Base load operation - limited MD shaving potential, focus on efficiency")
    if 'STRONG_TOU_PATTERN' in tags:
        recommendations.append("🌟 Excellent candidate for RP4 TOU optimization")
    
    # Cost-based recommendations
    if selected_tariff and 'Rates' in selected_tariff:
        capacity_rate = selected_tariff['Rates'].get('Capacity Rate', 0)
        if capacity_rate > 0.08:  # High capacity charge
            recommendations.append(f"💰 High capacity rate (RM {capacity_rate:.4f}/kWh) - MD reduction highly beneficial")
    
    return recommendations


def compute_md_shaving_difficulty_score(monthly_features: Dict, monthly_events: Dict, 
                                      pattern_tags: Dict) -> Dict:
    """
    Compute MD-shaving difficulty score (0-100) and grade for each month.
    
    Returns:
        Dictionary with difficulty scores and grades
    """
    difficulty_analysis = {}
    
    for month_str in monthly_features.keys():
        features = monthly_features[month_str]
        events = monthly_events.get(month_str, {})
        tags = pattern_tags.get(month_str, [])
        
        # Initialize base score
        difficulty_score = 0
        score_components = {}
        
        # Component 1: Load Factor (0-25 points)
        # Lower load factor = higher difficulty
        load_factor = features.get('load_factor', 0)
        load_factor_score = max(0, 25 * (1 - load_factor))
        difficulty_score += load_factor_score
        score_components['load_factor'] = load_factor_score
        
        # Component 2: Peak Concentration (0-20 points)
        # Higher concentration = higher difficulty
        peak_concentration = features.get('peak_hour_concentration', 0)
        concentration_score = 20 * peak_concentration
        difficulty_score += concentration_score
        score_components['peak_concentration'] = concentration_score
        
        # Component 3: Variability (0-20 points)
        # Higher variability = higher difficulty
        cv = features.get('coefficient_of_variation', 0)
        variability_score = min(20, 20 * cv / 0.5)  # Normalize to 0.5 CV max
        difficulty_score += variability_score
        score_components['variability'] = variability_score
        
        # Component 4: Peak Events Frequency (0-15 points)
        total_events = events.get('total_events', 0)
        data_days = features.get('data_points', 0) / 96  # Assuming 15-min intervals
        event_frequency = total_events / max(1, data_days) * 30  # Events per month
        frequency_score = min(15, event_frequency * 2)  # Scale appropriately
        difficulty_score += frequency_score
        score_components['event_frequency'] = frequency_score
        
        # Component 5: Peak Severity (0-10 points)
        max_excess = events.get('max_excess_kw', 0)
        monthly_max = features.get('max_demand', 1)
        excess_ratio = max_excess / monthly_max if monthly_max > 0 else 0
        severity_score = min(10, 10 * excess_ratio / 0.3)  # Normalize to 30% excess max
        difficulty_score += severity_score
        score_components['peak_severity'] = severity_score
        
        # Component 6: Pattern Complexity (0-10 points)
        complexity_score = 0
        if 'SPIKY_PROFILE' in tags:
            complexity_score += 4
        if 'HIGH_VARIABILITY' in tags:
            complexity_score += 3
        if 'CONCENTRATED_PEAKS' in tags:
            complexity_score += 3
        difficulty_score += complexity_score
        score_components['pattern_complexity'] = complexity_score
        
        # Normalize to 0-100 scale
        difficulty_score = min(100, difficulty_score)
        
        # Assign letter grade (INVERTED: Lower difficulty = better grade)
        if difficulty_score >= 85:
            grade = 'F'
            grade_description = 'Extremely Difficult'
        elif difficulty_score >= 75:
            grade = 'D'
            grade_description = 'Very Difficult'
        elif difficulty_score >= 65:
            grade = 'C'
            grade_description = 'Difficult'
        elif difficulty_score >= 55:
            grade = 'C+'
            grade_description = 'Moderately Difficult'
        elif difficulty_score >= 45:
            grade = 'B'
            grade_description = 'Moderate'
        elif difficulty_score >= 35:
            grade = 'B+'
            grade_description = 'Manageable'
        elif difficulty_score >= 25:
            grade = 'A'
            grade_description = 'Easy'
        else:
            grade = 'A+'
            grade_description = 'Very Easy'
        
        difficulty_analysis[month_str] = {
            'difficulty_score': round(difficulty_score, 1),
            'grade': grade,
            'grade_description': grade_description,
            'score_components': score_components,
            'recommendations': _generate_difficulty_recommendations(difficulty_score, tags, score_components)
        }
    
    return difficulty_analysis


def _generate_difficulty_recommendations(score: float, tags: List[str], components: Dict) -> List[str]:
    """Generate recommendations based on difficulty analysis."""
    recommendations = []
    
    if score >= 75:
        recommendations.append("Consider advanced battery energy storage systems with 4+ hour duration")
        recommendations.append("Implement demand response programs with automated load shedding")
        recommendations.append("Evaluate load shifting opportunities to off-peak hours")
    elif score >= 50:
        recommendations.append("Standard battery systems (2-3 hour duration) should be effective")
        recommendations.append("Implement peak demand monitoring and alerts")
        recommendations.append("Consider time-of-use tariff optimization")
    else:
        recommendations.append("Simple demand management strategies may be sufficient")
        recommendations.append("Focus on operational efficiency improvements")
        recommendations.append("Monitor for seasonal pattern changes")
    
    # Specific recommendations based on pattern tags
    if 'SPIKY_PROFILE' in tags:
        recommendations.append("Focus on peak clipping rather than load shifting")
    if 'BASE_LOAD' in tags:
        recommendations.append("Limited MD shaving potential - focus on efficiency")
    if 'STRONG_TOU_PATTERN' in tags:
        recommendations.append("Excellent candidate for time-of-use optimization")
    
    return recommendations


def create_pattern_visualizations(patterns: Dict, power_col: str, df_clean: Optional[pd.DataFrame] = None) -> Dict[str, go.Figure]:
    """Create comprehensive pattern visualization charts with RP4 overlay."""
    
    figures = {}
    
    # 1. Hourly Pattern Chart with RP4 Peak Period Overlay
    hourly_data = patterns['hourly_patterns']
    fig_hourly = go.Figure()
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_data.index,
        y=hourly_data['mean'],
        mode='lines+markers',
        name='Average Demand',
        line=dict(color='blue', width=2)
    ))
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_data.index,
        y=hourly_data['max'],
        mode='lines',
        name='Maximum Demand',
        line=dict(color='red', dash='dash'),
        opacity=0.7
    ))
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_data.index,
        y=hourly_data['min'],
        mode='lines',
        name='Minimum Demand',
        line=dict(color='green', dash='dash'),
        opacity=0.7
    ))
    
    # Add RP4 peak period overlay (2PM-10PM, weekdays)
    if is_peak_rp4 is not None and df_clean is not None:
        try:
            # Highlight RP4 peak hours (14-22)
            peak_hours = list(range(14, 22))
            peak_y_values = [hourly_data.loc[h, 'max'] if h in hourly_data.index else 0 for h in peak_hours]
            
            fig_hourly.add_trace(go.Scatter(
                x=peak_hours,
                y=peak_y_values,
                fill=None,
                mode='none',
                showlegend=False
            ))
            
            # Add shaded region for RP4 peak hours
            fig_hourly.add_shape(
                type="rect",
                x0=14, x1=22,
                y0=0, y1=max(hourly_data['max']) * 1.1,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )
            
            fig_hourly.add_annotation(
                x=18, y=max(hourly_data['max']) * 0.95,
                text="RP4 Peak Period<br>(2PM-10PM)",
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=1
            )
        except Exception:
            pass  # Skip RP4 overlay if it fails
    
    fig_hourly.update_layout(
        title='Daily Demand Pattern (24-Hour Profile) with RP4 Peak Periods',
        xaxis_title='Hour of Day',
        yaxis_title=f'Power Demand (kW)',
        hovermode='x unified',
        height=500
    )
    
    figures['hourly_pattern'] = fig_hourly
    
    # 2. Weekly Pattern Chart
    weekly_data = patterns['weekly_patterns']
    fig_weekly = px.bar(
        x=weekly_data.index,
        y=weekly_data['mean'],
        title='Weekly Demand Pattern (Average by Day of Week)',
        labels={'x': 'Day of Week', 'y': 'Average Power Demand (kW)'},
        color=weekly_data['mean'],
        color_continuous_scale='viridis'
    )
    fig_weekly.update_layout(height=400)
    figures['weekly_pattern'] = fig_weekly
    
    # 3. Monthly Pattern Chart
    monthly_data = patterns['monthly_patterns']
    fig_monthly = px.line(
        x=monthly_data.index,
        y=monthly_data['mean'],
        title='Monthly Demand Pattern (Average by Month)',
        labels={'x': 'Month', 'y': 'Average Power Demand (kW)'},
        markers=True
    )
    fig_monthly.update_layout(height=400)
    figures['monthly_pattern'] = fig_monthly
    
    # 4. Heatmap - Hour vs Day of Week
    analysis_df = patterns['analysis_df']
    pivot_data = analysis_df.pivot_table(
        values=power_col, 
        index='Hour', 
        columns='DayOfWeek', 
        aggfunc='mean'
    )
    
    # Map day numbers to names
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_data.columns = [day_names[int(col)] for col in pivot_data.columns]
    
    fig_heatmap = px.imshow(
        pivot_data,
        title='Demand Pattern Heatmap (Hour vs Day of Week)',
        labels=dict(x="Day of Week", y="Hour of Day", color="Average Demand (kW)"),
        aspect="auto",
        color_continuous_scale="viridis"
    )
    fig_heatmap.update_layout(height=600)
    figures['demand_heatmap'] = fig_heatmap
    
    return figures


def create_md_difficulty_visualizations(difficulty_analysis: Dict, monthly_features: Dict, 
                                      monthly_events: Dict) -> Dict[str, go.Figure]:
    """Create MD difficulty analysis visualization charts."""
    
    figures = {}
    
    if not difficulty_analysis:
        return figures
    
    # 1. Monthly Difficulty Score Chart
    months = list(difficulty_analysis.keys())
    scores = [difficulty_analysis[m]['difficulty_score'] for m in months]
    grades = [difficulty_analysis[m]['grade'] for m in months]
    
    # Color map for grades (CORRECTED: Green for easy, Red for difficult)
    grade_colors = {
        'A+': '#2E8B57',   # Dark Green - Very Easy
        'A': '#32CD32',    # Lime Green - Easy  
        'B+': '#FFD700',   # Gold - Manageable
        'B': '#FFA500',    # Orange - Moderate
        'C+': '#FF6347',   # Tomato - Moderately Difficult
        'C': '#FF4500',    # Red Orange - Difficult
        'D': '#DC143C',    # Crimson - Very Difficult
        'F': '#8B0000'     # Dark Red - Extremely Difficult
    }
    
    colors = [grade_colors.get(grade, '#1f77b4') for grade in grades]
    
    fig_scores = go.Figure(data=[
        go.Bar(
            x=months,
            y=scores,
            text=[f"{score:.1f}<br>({grade})" for score, grade in zip(scores, grades)],
            textposition='auto',
            marker_color=colors,
            name='Difficulty Score'
        )
    ])
    
    fig_scores.update_layout(
        title='MD Shaving Difficulty Score by Month',
        xaxis_title='Month',
        yaxis_title='Difficulty Score (0-100)',
        yaxis=dict(range=[0, 100]),
        height=500
    )
    
    # Add grade bands as horizontal lines
    fig_scores.add_hline(y=85, line_dash="dash", line_color="darkred", 
                        annotation_text="F (Extremely Difficult)")
    fig_scores.add_hline(y=75, line_dash="dash", line_color="red", 
                        annotation_text="D (Very Difficult)")
    fig_scores.add_hline(y=55, line_dash="dash", line_color="orange", 
                        annotation_text="C+ (Moderately Difficult)")
    fig_scores.add_hline(y=35, line_dash="dash", line_color="gold", 
                        annotation_text="B+ (Manageable)")
    fig_scores.add_hline(y=25, line_dash="dash", line_color="green", 
                        annotation_text="A (Easy)")
    
    figures['difficulty_scores'] = fig_scores
    
    # 2. Score Components Breakdown (Stacked Bar)
    if months:
        components = ['load_factor', 'peak_concentration', 'variability', 
                     'event_frequency', 'peak_severity', 'pattern_complexity']
        component_labels = ['Load Factor', 'Peak Concentration', 'Variability',
                          'Event Frequency', 'Peak Severity', 'Pattern Complexity']
        
        fig_components = go.Figure()
        
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            values = [difficulty_analysis[m]['score_components'].get(comp, 0) for m in months]
            fig_components.add_trace(go.Bar(
                name=label,
                x=months,
                y=values,
                offsetgroup=0
            ))
        
        fig_components.update_layout(
            title='Difficulty Score Components Breakdown',
            xaxis_title='Month',
            yaxis_title='Component Score',
            barmode='stack',
            height=500
        )
        
        figures['score_components'] = fig_components
    
    # 3. Monthly Peak Events Summary
    if monthly_events:
        event_counts = [monthly_events[m].get('total_events', 0) for m in months]
        max_excess = [monthly_events[m].get('max_excess_kw', 0) for m in months]
        
        fig_events = go.Figure()
        
        fig_events.add_trace(go.Bar(
            name='Peak Events Count',
            x=months,
            y=event_counts,
            yaxis='y',
            marker_color='lightblue'
        ))
        
        fig_events.add_trace(go.Scatter(
            name='Max Excess (kW)',
            x=months,
            y=max_excess,
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig_events.update_layout(
            title='Monthly Peak Events and Maximum Excess',
            xaxis_title='Month',
            yaxis=dict(title='Number of Peak Events', side='left'),
            yaxis2=dict(title='Maximum Excess (kW)', side='right', overlaying='y'),
            height=500
        )
        
        figures['peak_events_summary'] = fig_events
    
    # 4. Load Factor vs Difficulty Score Scatter
    if monthly_features:
        load_factors = [monthly_features[m].get('load_factor', 0) for m in months]
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=load_factors,
            y=scores,
            mode='markers+text',
            text=[m.split('-')[1] if '-' in m else m for m in months],  # Show month only
            textposition='top center',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.7
            ),
            name='Monthly Data'
        ))
        
        fig_scatter.update_layout(
            title='Load Factor vs MD Shaving Difficulty',
            xaxis_title='Load Factor',
            yaxis_title='Difficulty Score',
            height=500
        )
        
        figures['load_factor_vs_difficulty'] = fig_scatter
    
    return figures


def show_md_pattern_analysis():
    """Main function to render the MD Pattern Analysis tab."""
    
    st.markdown("## 📊 MD Pattern Analysis")
    st.markdown("""
    **Comprehensive Maximum Demand Pattern Analysis with MD Shaving Difficulty Assessment**
    
    Upload your load profile data to identify and analyze demand patterns including:
    - 📈 **Daily patterns** - 24-hour demand profiles
    - 📅 **Weekly patterns** - Day-of-week variations  
    - 🗓️ **Monthly patterns** - Seasonal demand trends
    - ⚡ **Peak events** - High demand period identification
    - 🎯 **Pattern insights** - Statistical analysis and recommendations
    - 🔋 **MD Shaving Difficulty** - Monthly difficulty scoring (0-100) with grades
    """)
    
    # Check for existing session state data first
    use_existing_data = False
    if hasattr(st.session_state, 'processed_df') and st.session_state.processed_df is not None:
        if hasattr(st.session_state, 'power_column') and st.session_state.power_column:
            with st.expander("💡 Use Existing Data", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"Found existing data with {len(st.session_state.processed_df)} rows. "
                           f"Power column: '{st.session_state.power_column}'")
                with col2:
                    if st.button("Use Existing Data", type="primary"):
                        use_existing_data = True
    
    df_clean = None
    power_col = None
    
    if use_existing_data:
        # Use existing session state data
        df_clean = st.session_state.processed_df.copy()
        power_col = st.session_state.power_column
        st.success("✅ Using existing processed data from session state")
        
    else:
        # File Upload Section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your load profile data file",
            type=["csv", "xls", "xlsx"],
            help="Upload CSV or Excel file containing timestamp and power demand data",
            key="md_pattern_file_uploader"
        )
        
        if uploaded_file is not None:
            # Read and process the file
            with st.spinner("Reading and processing file..."):
                df = read_uploaded_file(uploaded_file)
            
            if df.empty:
                st.error("Failed to read the uploaded file. Please check the file format and try again.")
                return
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows of data.")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column Selection
            st.subheader("🔧 Column Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp_col = st.selectbox(
                    "Select timestamp/datetime column:",
                    options=df.columns.tolist(),
                    key="md_pattern_timestamp_col"
                )
            
            with col2:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_columns:
                    st.error("No numeric columns found for power data!")
                    return
                
                power_col = st.selectbox(
                    "Select power demand column (kW):",
                    options=numeric_columns,
                    key="md_pattern_power_col"
                )
            
            if timestamp_col and power_col:
                # Process the data using the new reliable function
                with st.spinner("Processing timeseries data..."):
                    df_clean = prepare_timeseries_reliably(df, timestamp_col, power_col)
                
                if df_clean.empty:
                    st.error("No valid data found after cleaning. Please check your timestamp and power columns.")
                    return
                
                st.success(f"✅ Data processed successfully! {len(df_clean)} valid data points.")
        else:
            # Show sample data format when no file is uploaded
            st.info("👆 **Upload your load profile data file to begin pattern analysis**")
            
            with st.expander("📋 Expected Data Format", expanded=True):
                st.markdown("""
                **Your data file should contain:**
                - **Timestamp column**: Date and time information (various formats supported)
                - **Power column**: Power consumption/demand values in kW
                
                **Supported file formats:**
                - CSV files (.csv)
                - Excel files (.xls, .xlsx)
                
                **Sample data structure:**
                """)
                
                # Create sample data
                sample_dates = pd.date_range('2024-01-01', periods=10, freq='15T')
                sample_data = {
                    'Timestamp': sample_dates,
                    'Power_kW': [245.2, 250.1, 248.7, 255.3, 260.8, 258.4, 252.1, 247.9, 249.5, 251.2],
                    'Additional_Column': ['Optional'] * 10
                }
                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df, use_container_width=True)
                
                st.markdown("""
                **Tips for best results:**
                - Include at least several days of data for meaningful pattern analysis
                - Consistent time intervals (e.g., 15-minute, 30-minute, hourly) work best
                - Remove any obvious data errors or outliers before upload
                - Include weekdays and weekends for comprehensive weekly pattern analysis
                """)
            return
    
    # Proceed with analysis if we have valid data
    if df_clean is not None and not df_clean.empty and power_col:
        
        # Data Quality Analysis
        st.subheader("📊 Data Quality Analysis")
        interval_minutes, interval_hours, consistency = detect_data_interval(df_clean)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", f"{len(df_clean):,}")
        with col2:
            if interval_minutes < 60:
                st.metric("Detected Interval", f"{interval_minutes:.0f} min")
            else:
                st.metric("Detected Interval", f"{interval_hours:.1f} hrs")
        with col3:
            st.metric("Data Consistency", f"{consistency:.1f}%")
        with col4:
            duration_days = (df_clean.index.max() - df_clean.index.min()).days
            st.metric("Data Duration", f"{duration_days} days")
        
        # Quality indicator
        if consistency >= 95:
            st.success(f"✅ Excellent data quality - {consistency:.1f}% consistent intervals")
        elif consistency >= 85:
            st.warning(f"⚠️ Good data quality - {consistency:.1f}% consistent intervals")
        else:
            st.error(f"❌ Poor data quality - {consistency:.1f}% consistent intervals")
        
        # **NEW: Enhanced RP4 Tariff Selection Interface with Medium Voltage TOU Default**
        st.subheader("🏛️ RP4 Tariff Configuration")
        
        # Helper function to find default tariff (Medium Voltage TOU)
        def _find_medium_voltage_tou_default():
            """Find Medium Voltage TOU tariff as default selection."""
            if TARIFF_DATA is None:
                return None, None, None, None
                
            for user_type, user_data in TARIFF_DATA.items():
                if "Business" in user_type:  # Prefer Business users
                    for group_name, group_data in user_data.get("Tariff Groups", {}).items():
                        for tariff in group_data.get("Tariffs", []):
                            tariff_name = tariff.get("Tariff", "")
                            if ("Medium Voltage" in tariff_name and 
                                "TOU" in tariff_name and 
                                tariff.get("Type") == "TOU"):
                                return user_type, group_name, tariff_name, tariff
            
            # Fallback: find any TOU tariff
            for user_type, user_data in TARIFF_DATA.items():
                for group_name, group_data in user_data.get("Tariff Groups", {}).items():
                    for tariff in group_data.get("Tariffs", []):
                        if tariff.get("Type") == "TOU":
                            return user_type, group_name, tariff.get("Tariff"), tariff
            
            return None, None, None, None
        
        # Initialize default tariff if no selection exists
        if (not hasattr(st.session_state, 'selected_tariff') or 
            st.session_state.selected_tariff is None):
            if TARIFF_DATA is not None:
                default_user, default_group, default_tariff_name, default_tariff = _find_medium_voltage_tou_default()
                if default_tariff:
                    st.session_state.selected_tariff = default_tariff
                    st.session_state.default_tariff_user = default_user
                    st.session_state.default_tariff_group = default_group
                    st.session_state.default_tariff_name = default_tariff_name
        
        # Show current tariff selection with comprehensive interface
        if TARIFF_DATA is not None:
            # Get current selections (either from session state or defaults)
            current_tariff = getattr(st.session_state, 'selected_tariff', None)
            
            # Always show the tariff selection interface (no confirmation button needed)
            st.markdown("**RP4 Tariff Selection - Automatically Applied:**")
            
            # User Type Selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                user_types = list(TARIFF_DATA.keys())
                default_user_idx = 0
                if hasattr(st.session_state, 'default_tariff_user'):
                    try:
                        default_user_idx = user_types.index(st.session_state.default_tariff_user)
                    except ValueError:
                        default_user_idx = 1 if len(user_types) > 1 else 0
                
                selected_user_type = st.selectbox(
                    "User Type",
                    options=user_types,
                    index=default_user_idx,
                    key="rp4_user_type",
                    help="Select your electricity user category"
                )
            
            # Tariff Group Selection
            with col2:
                if selected_user_type and selected_user_type in TARIFF_DATA:
                    tariff_groups = list(TARIFF_DATA[selected_user_type]["Tariff Groups"].keys())
                    default_group_idx = 0
                    if hasattr(st.session_state, 'default_tariff_group'):
                        try:
                            default_group_idx = tariff_groups.index(st.session_state.default_tariff_group)
                        except ValueError:
                            default_group_idx = 0
                    
                    selected_group = st.selectbox(
                        "Tariff Group",
                        options=tariff_groups,
                        index=default_group_idx,
                        key="rp4_tariff_group",
                        help="Select your voltage/demand category"
                    )
                else:
                    selected_group = None
                    st.selectbox("Tariff Group", options=[], key="rp4_tariff_group_empty", disabled=True)
            
            # Specific Tariff Selection
            with col3:
                if selected_group and selected_user_type in TARIFF_DATA:
                    available_tariffs = TARIFF_DATA[selected_user_type]["Tariff Groups"][selected_group]["Tariffs"]
                    tariff_options = [t["Tariff"] for t in available_tariffs]
                    
                    default_tariff_idx = 0
                    if hasattr(st.session_state, 'default_tariff_name'):
                        try:
                            default_tariff_idx = tariff_options.index(st.session_state.default_tariff_name)
                        except ValueError:
                            # If default not found, prefer TOU tariffs
                            for i, tariff in enumerate(available_tariffs):
                                if tariff.get("Type") == "TOU":
                                    default_tariff_idx = i
                                    break
                    
                    selected_tariff_name = st.selectbox(
                        "Specific Tariff",
                        options=tariff_options,
                        index=default_tariff_idx,
                        key="rp4_specific_tariff",
                        help="Select your exact tariff rate"
                    )
                    
                    # Find the selected tariff data and auto-update session state
                    selected_tariff = None
                    for tariff in available_tariffs:
                        if tariff["Tariff"] == selected_tariff_name:
                            selected_tariff = tariff.copy()
                            break
                    
                    # Automatically update session state (no confirmation button)
                    if selected_tariff:
                        st.session_state.selected_tariff = selected_tariff
                else:
                    selected_tariff = None
                    st.selectbox("Specific Tariff", options=[], key="rp4_specific_tariff_empty", disabled=True)
            
            # Display comprehensive tariff information
            if selected_tariff:
                with st.expander("📋 Selected Tariff Details & Analysis Configuration", expanded=True):
                    # Header with tariff path
                    st.markdown(f"**🎯 Active Selection:** {selected_user_type} → {selected_group} → {selected_tariff['Tariff']}")
                    
                    # Main tariff info in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**⚡ Tariff Configuration:**")
                        voltage = selected_tariff.get('Voltage', 'N/A')
                        tariff_type = selected_tariff.get('Type', 'N/A')
                        st.info(f"**Voltage Level:** {voltage}")
                        st.info(f"**Rate Type:** {tariff_type}")
                        
                        # TOU indicator with enhanced messaging
                        if tariff_type == 'TOU':
                            st.success("🎯 **TOU (Time-of-Use) Tariff Active**")
                            st.caption("⏰ Peak periods: Mon-Fri 2PM-10PM")
                        else:
                            st.info("📊 **General Tariff Active**")
                            st.caption("📈 Standard peak period analysis")
                    
                    with col2:
                        st.markdown("**💰 Rate Information:**")
                        if "Rates" in selected_tariff:
                            rates = selected_tariff["Rates"]
                            capacity_rate = rates.get("Capacity Rate", 0)
                            peak_rate = rates.get("Peak Rate", rates.get("Energy Rate", 0))
                            
                            st.metric("Capacity Rate", f"RM {capacity_rate:.4f}")
                            if tariff_type == 'TOU' and "Peak Rate" in rates:
                                st.metric("Peak Rate", f"RM {peak_rate:.4f}")
                                off_peak_rate = rates.get("Off Peak Rate", 0)
                                st.metric("Off-Peak Rate", f"RM {off_peak_rate:.4f}")
                            else:
                                st.metric("Energy Rate", f"RM {peak_rate:.4f}")
                    
                    with col3:
                        st.markdown("**🔍 Analysis Features:**")
                        features = []
                        if tariff_type == 'TOU':
                            features.extend([
                                "✅ RP4 Peak Period Detection",
                                "✅ TOU Energy Cost Analysis", 
                                "✅ Peak vs Off-Peak Ratios",
                                "✅ Holiday Impact Analysis"
                            ])
                        else:
                            features.extend([
                                "✅ General MD Analysis",
                                "✅ Standard Peak Detection",
                                "✅ Load Pattern Recognition",
                                "✅ Basic Cost Estimation"
                            ])
                        
                        for feature in features:
                            st.write(feature)
                    
                    # Quick change option
                    if st.button("🔄 Reset to Medium Voltage TOU Default", key="reset_to_default_btn"):
                        default_user, default_group, default_tariff_name, default_tariff = _find_medium_voltage_tou_default()
                        if default_tariff:
                            st.session_state.selected_tariff = default_tariff
                            st.session_state.default_tariff_user = default_user
                            st.session_state.default_tariff_group = default_group
                            st.session_state.default_tariff_name = default_tariff_name
                            st.success("✅ Reset to Medium Voltage TOU default tariff")
                            st.rerun()
        else:
            st.warning("⚠️ RP4 tariff data not available. Analysis will use simplified peak period detection.")
            selected_tariff = None
        
        # Set final selected tariff for analysis
        selected_tariff = getattr(st.session_state, 'selected_tariff', None)
        
        # Display active tariff status
        if selected_tariff:
            st.success(f"🎯 **Active Tariff:** {selected_tariff.get('Tariff', 'Unknown')} | **Type:** {selected_tariff.get('Type', 'Unknown')} | **Auto-Applied ✅**")
        
        # **NEW: MD Shaving Difficulty Analysis**
        st.subheader("🎯 MD Shaving Difficulty Analysis")
        
        # Configuration for MD shaving analysis with enhanced context
        col1, col2 = st.columns(2)
        with col1:
            shaving_percentage = st.slider(
                "Target MD Shaving Percentage",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                help="Target percentage to reduce monthly maximum demand",
                key="md_shaving_target_percent"
            )
        
        with col2:
            # Display selected tariff information if available
            if selected_tariff:
                tariff_type = selected_tariff.get('Type', 'General')
                voltage = selected_tariff.get('Voltage', 'Unknown')
                st.info(f"**Analysis Mode:** {tariff_type} Tariff")
                st.info(f"**Voltage:** {voltage}")
                
                if tariff_type == 'TOU':
                    st.success("🎯 Enhanced RP4 peak period analysis enabled")
                    st.caption("Peak periods: Mon-Fri 2PM-10PM")
                else:
                    st.info("📊 Standard peak period analysis")
            else:
                st.warning("No tariff selected. Using simplified analysis.")
        
        # Compute monthly baseline features with RP4 integration
        with st.spinner("Computing monthly baseline features with RP4 analysis..."):
            if selected_tariff and is_peak_rp4 is not None:
                # Use enhanced RP4-aware analysis
                monthly_features = compute_monthly_baseline_features_with_rp4(
                    df_clean, power_col, selected_tariff
                )
                st.success("✅ Using RP4-compliant peak period analysis with Malaysia holidays")
            else:
                # Fallback to standard analysis
                monthly_features = compute_monthly_baseline_features(df_clean, power_col)
                if is_peak_rp4 is None:
                    st.info("ℹ️ RP4 peak logic not available - using simplified analysis")
        
        if not monthly_features:
            st.warning("⚠️ Unable to compute monthly features. Need at least 1 month of data.")
        else:
            # Tag load profile patterns
            pattern_tags = tag_load_profile_patterns(monthly_features)
            
            # Detect peak events vs monthly targets
            monthly_events = detect_contiguous_peak_events_vs_monthly_targets(
                df_clean, power_col, monthly_features, shaving_percentage
            )
            
            # Compute difficulty scores with RP4 integration
            if selected_tariff and is_peak_rp4 is not None:
                difficulty_analysis = compute_md_shaving_difficulty_score_with_rp4(
                    monthly_features, monthly_events, pattern_tags, selected_tariff
                )
            else:
                difficulty_analysis = compute_md_shaving_difficulty_score(
                    monthly_features, monthly_events, pattern_tags
                )
            
            # Display MD Difficulty Results
            st.markdown("#### 🏆 Monthly MD Shaving Difficulty Scores")
            
            if difficulty_analysis:
                # Create summary metrics
                all_scores = [d['difficulty_score'] for d in difficulty_analysis.values()]
                avg_score = np.mean(all_scores)
                max_score = max(all_scores)
                min_score = min(all_scores)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Difficulty", f"{avg_score:.1f}/100")
                with col2:
                    st.metric("Hardest Month", f"{max_score:.1f}/100")
                with col3:
                    st.metric("Easiest Month", f"{min_score:.1f}/100")
                
                # Monthly difficulty table with RP4 insights
                table_data = []
                for month, data in difficulty_analysis.items():
                    row = {
                        'Month': month,
                        'Difficulty Score': data['difficulty_score'],
                        'Grade': data['grade'],
                        'Description': data['grade_description'],
                        'Peak Events': monthly_events.get(month, {}).get('total_events', 0),
                        'Max Excess (kW)': monthly_events.get(month, {}).get('max_excess_kw', 0)
                    }
                    
                    # Add RP4-specific columns if available
                    if 'rp4_insights' in data and data['rp4_insights'].get('rp4_compliance'):
                        rp4_data = data['rp4_insights']
                        row.update({
                            'RP4 Peak Max (kW)': round(rp4_data.get('rp4_peak_max', 0), 1),
                            'Peak/Off-Peak Ratio': round(rp4_data.get('peak_to_offpeak_ratio', 1), 2),
                            'Tariff Type': rp4_data.get('tariff_type', 'Unknown')
                        })
                    
                    table_data.append(row)
                
                difficulty_df = pd.DataFrame(table_data)
                st.dataframe(difficulty_df, use_container_width=True)
                
                # RP4 Compliance Summary
                rp4_compliant_months = [m for m, d in difficulty_analysis.items() 
                                      if d.get('rp4_insights', {}).get('rp4_compliance', False)]
                
                if rp4_compliant_months:
                    with st.expander("🏛️ RP4 Tariff Analysis Summary", expanded=False):
                        st.markdown("#### RP4 Peak Period Compliance Analysis")
                        
                        # Calculate RP4 statistics
                        avg_peak_ratio = np.mean([
                            difficulty_analysis[m]['rp4_insights']['peak_to_offpeak_ratio'] 
                            for m in rp4_compliant_months
                        ])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RP4 Compliant Months", len(rp4_compliant_months))
                        with col2:
                            st.metric("Avg Peak/Off-Peak Ratio", f"{avg_peak_ratio:.2f}")
                        with col3:
                            tariff_type = difficulty_analysis[rp4_compliant_months[0]]['rp4_insights'].get('tariff_type', 'Unknown')
                            st.metric("Current Tariff Type", tariff_type)
                        
                        # RP4 recommendations
                        st.markdown("**RP4-Specific Insights:**")
                        if avg_peak_ratio > 1.5:
                            st.success("✅ Strong peak/off-peak differentiation - excellent for TOU optimization")
                        elif avg_peak_ratio > 1.2:
                            st.warning("⚠️ Moderate peak/off-peak differentiation - some TOU potential")
                        else:
                            st.info("ℹ️ Low peak/off-peak differentiation - focus on general MD reduction")
                
                # Display enhanced monthly difficulty table
                st.dataframe(difficulty_df, use_container_width=True)
                
                # MD Difficulty Visualizations
                st.markdown("#### 📈 MD Difficulty Visualizations")
                
                difficulty_figures = create_md_difficulty_visualizations(
                    difficulty_analysis, monthly_features, monthly_events
                )
                
                if difficulty_figures:
                    difficulty_tabs = st.tabs([
                        "Difficulty Scores", "Score Components", "Peak Events", "Load Factor Analysis"
                    ])
                    
                    with difficulty_tabs[0]:
                        if 'difficulty_scores' in difficulty_figures:
                            st.plotly_chart(difficulty_figures['difficulty_scores'], use_container_width=True)
                    
                    with difficulty_tabs[1]:
                        if 'score_components' in difficulty_figures:
                            st.plotly_chart(difficulty_figures['score_components'], use_container_width=True)
                    
                    with difficulty_tabs[2]:
                        if 'peak_events_summary' in difficulty_figures:
                            st.plotly_chart(difficulty_figures['peak_events_summary'], use_container_width=True)
                    
                    with difficulty_tabs[3]:
                        if 'load_factor_vs_difficulty' in difficulty_figures:
                            st.plotly_chart(difficulty_figures['load_factor_vs_difficulty'], use_container_width=True)
                
                # Show recommendations for the most difficult month
                if difficulty_analysis:
                    hardest_month = max(difficulty_analysis.keys(), 
                                      key=lambda x: difficulty_analysis[x]['difficulty_score'])
                    hardest_data = difficulty_analysis[hardest_month]
                    
                    st.markdown(f"#### 💡 Recommendations for {hardest_month} (Hardest Month)")
                    st.markdown(f"**Difficulty Score: {hardest_data['difficulty_score']:.1f}/100 (Grade: {hardest_data['grade']})**")
                    
                    for rec in hardest_data['recommendations']:
                        st.markdown(f"- 💡 {rec}")
        
        # Standard Pattern Analysis
        st.subheader("🔍 Standard Pattern Analysis")
        
        with st.spinner("Analyzing demand patterns..."):
            patterns = analyze_daily_patterns(df_clean, power_col)
        
        # Display key insights
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peak Hours", ", ".join([f"{h:02d}:00" for h in patterns['peak_hours'][:3]]))
        with col2:
            st.metric("Off-Peak Hours", ", ".join([f"{h:02d}:00" for h in patterns['off_peak_hours'][:3]]))
        
        # Pattern Visualizations
        st.subheader("📈 Pattern Visualizations")
        
        figures = create_pattern_visualizations(patterns, power_col, df_clean)
        
        # Display charts in tabs
        chart_tabs = st.tabs(["Daily Pattern", "Weekly Pattern", "Monthly Pattern", "Demand Heatmap"])
        
        with chart_tabs[0]:
            st.plotly_chart(figures['hourly_pattern'], use_container_width=True)
            
            # Show hourly statistics table
            with st.expander("📊 Hourly Statistics"):
                st.dataframe(patterns['hourly_patterns'], use_container_width=True)
        
        with chart_tabs[1]:
            st.plotly_chart(figures['weekly_pattern'], use_container_width=True)
            
            # Show weekly statistics table
            with st.expander("📊 Weekly Statistics"):
                st.dataframe(patterns['weekly_patterns'], use_container_width=True)
        
        with chart_tabs[2]:
            st.plotly_chart(figures['monthly_pattern'], use_container_width=True)
            
            # Show monthly statistics table
            with st.expander("📊 Monthly Statistics"):
                st.dataframe(patterns['monthly_patterns'], use_container_width=True)
        
        with chart_tabs[3]:
            st.plotly_chart(figures['demand_heatmap'], use_container_width=True)
            st.info("💡 Darker colors indicate higher demand. This heatmap helps identify peak demand times across different days of the week.")
        
        # Peak Events Analysis
        st.subheader("⚡ Peak Events Analysis")
        
        # Peak threshold configuration
        threshold_percentile = st.slider(
            "Peak Event Threshold (Percentile)",
            min_value=85,
            max_value=99,
            value=95,
            step=1,
            help="Define peak events as demand above this percentile threshold",
            key="peak_threshold_slider"
        )
        
        with st.spinner("Identifying peak events..."):
            peak_events = identify_peak_events(df_clean, power_col, threshold_percentile)
        
        if not peak_events.empty:
            st.success(f"🎯 Found {len(peak_events)} peak events above {threshold_percentile}th percentile")
            
            # Peak events summary
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_duration = peak_events['Duration (hours)'].mean()
                st.metric("Avg Event Duration", f"{avg_duration:.1f} hrs")
            with col2:
                max_peak = peak_events['Peak Demand (kW)'].max()
                st.metric("Highest Peak", f"{max_peak:.1f} kW")
            with col3:
                most_common_day = peak_events['Day of Week'].mode().iloc[0] if not peak_events.empty else "N/A"
                st.metric("Most Common Day", most_common_day)
            
            # Show peak events table
            with st.expander("📋 Peak Events Details"):
                display_columns = ['Date', 'Start Time', 'Duration (hours)', 'Peak Demand (kW)', 
                                 'Average Demand (kW)', 'Day of Week', 'Start Hour']
                
                formatted_events = peak_events[display_columns].copy()
                formatted_events['Start Time'] = formatted_events['Start Time'].dt.strftime('%H:%M')
                formatted_events['Duration (hours)'] = formatted_events['Duration (hours)'].round(2)
                formatted_events['Peak Demand (kW)'] = formatted_events['Peak Demand (kW)'].round(1)
                formatted_events['Average Demand (kW)'] = formatted_events['Average Demand (kW)'].round(1)
                
                st.dataframe(formatted_events, use_container_width=True)
        else:
            st.info(f"No peak events found above {threshold_percentile}th percentile threshold.")
        
        # Pattern Insights and Recommendations
        st.subheader("💡 Pattern Insights & Recommendations")
        
        # Generate insights
        insights = []
        
        # Peak hour insights
        peak_hours_str = ", ".join([f"{h:02d}:00" for h in patterns['peak_hours']])
        insights.append(f"🕐 **Peak demand typically occurs during:** {peak_hours_str}")
        
        # Weekly pattern insights
        weekly_data = patterns['weekly_patterns']
        busiest_day = weekly_data['mean'].idxmax()
        quietest_day = weekly_data['mean'].idxmin()
        insights.append(f"📅 **Busiest day:** {busiest_day} | **Quietest day:** {quietest_day}")
        
        # Variability insights
        daily_std = patterns['hourly_patterns']['std'].mean()
        overall_mean = df_clean[power_col].mean()
        variability_ratio = daily_std / overall_mean * 100
        
        if variability_ratio > 30:
            insights.append(f"📊 **High demand variability** ({variability_ratio:.1f}%) - Consider demand management strategies")
        elif variability_ratio > 15:
            insights.append(f"📊 **Moderate demand variability** ({variability_ratio:.1f}%) - Some optimization potential")
        else:
            insights.append(f"📊 **Low demand variability** ({variability_ratio:.1f}%) - Stable demand profile")
        
        # Peak events insights
        if not peak_events.empty:
            event_count = len(peak_events)
            total_days = (df_clean.index.max() - df_clean.index.min()).days
            event_frequency = event_count / total_days * 30  # Events per month
            insights.append(f"⚡ **Peak events occur approximately {event_frequency:.1f} times per month**")
        
        # Display insights
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Recommendations
        st.markdown("#### 🎯 General Recommendations")
        recommendations = []
        
        if not peak_events.empty:
            recommendations.append("Consider demand response strategies during identified peak hours")
            recommendations.append("Implement load shifting to move demand from peak to off-peak periods")
            recommendations.append("Evaluate battery energy storage systems for peak shaving")
        
        if variability_ratio > 20:
            recommendations.append("High demand variability suggests potential for load optimization")
            recommendations.append("Consider automated demand management systems")
        
        recommendations.append("Monitor and track demand patterns regularly for continuous optimization")
        recommendations.append("Consider time-of-use tariffs to benefit from off-peak pricing")
        
        for rec in recommendations:
            st.markdown(f"- 💡 {rec}")


if __name__ == "__main__":
    show_md_pattern_analysis()
