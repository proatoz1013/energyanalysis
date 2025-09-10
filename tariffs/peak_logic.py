import pandas as pd
import datetime
from datetime import date

# Malaysia Public Holidays 2024 (existing)
MALAYSIA_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 2, 10),  # Chinese New Year (1st Day)
    date(2024, 2, 11),  # Chinese New Year (2nd Day)
    date(2024, 4, 10),  # Hari Raya Aidilfitri (1st Day)
    date(2024, 4, 11),  # Hari Raya Aidilfitri (2nd Day)
    date(2024, 5, 1),   # Labour Day
    date(2024, 5, 22),  # Wesak Day
    date(2024, 6, 3),   # Agong's Birthday
    date(2024, 6, 17),  # Hari Raya Aidiladha
    date(2024, 7, 7),   # Awal Muharram
    date(2024, 8, 31),  # Independence Day
    date(2024, 9, 15),  # Maulidur Rasul
    date(2024, 9, 16),  # Malaysia Day
    date(2024, 11, 1),  # Deepavali
    date(2024, 12, 25), # Christmas Day
}

# Malaysia Public Holidays 2025 (comprehensive list)
MALAYSIA_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 29),  # Chinese New Year (1st Day)
    date(2025, 1, 30),  # Chinese New Year (2nd Day)
    date(2025, 3, 31),  # Hari Raya Aidilfitri (1st Day)
    date(2025, 4, 1),   # Hari Raya Aidilfitri (2nd Day)
    date(2025, 5, 1),   # Labour Day
    date(2025, 5, 12),  # Wesak Day
    date(2025, 6, 2),   # Agong's Birthday
    date(2025, 6, 7),   # Hari Raya Aidiladha
    date(2025, 6, 27),  # Awal Muharram
    date(2025, 8, 31),  # Independence Day
    date(2025, 9, 5),   # Maulidur Rasul
    date(2025, 9, 16),  # Malaysia Day
    date(2025, 10, 20), # Deepavali
    date(2025, 12, 25), # Christmas Day
}

# Combined holidays dictionary for multi-year support
MALAYSIA_HOLIDAYS = {
    2024: MALAYSIA_HOLIDAYS_2024,
    2025: MALAYSIA_HOLIDAYS_2025,
}

def get_malaysia_holidays(year=None):
    """
    Get Malaysia public holidays for a specific year or all available years.
    
    Args:
        year (int, optional): Specific year to get holidays for. If None, returns all holidays.
    
    Returns:
        set: Set of date objects representing public holidays.
    """
    if year is None:
        # Return all holidays from all years
        all_holidays = set()
        for year_holidays in MALAYSIA_HOLIDAYS.values():
            all_holidays.update(year_holidays)
        return all_holidays
    
    return MALAYSIA_HOLIDAYS.get(year, set())

def detect_holidays_from_data(df, timestamp_col):
    """
    Automatically detect which years are present in the data and return appropriate holidays.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp data
        timestamp_col (str): Name of the timestamp column
    
    Returns:
        set: Set of date objects for holidays covering the data's time range
    """
    if df.empty:
        return set()
    
    # Get the year range from the data
    years = df[timestamp_col].dt.year.unique()
    
    holidays = set()
    for year in years:
        holidays.update(get_malaysia_holidays(year))
    
    return holidays

def is_public_holiday(dt, holidays):
    """
    Returns True if the datetime falls on a public holiday (matching by date).
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return dt.date() in holidays

def is_peak_hour(dt, peak_start=14, peak_end=22):
    """
    Check if a datetime falls within peak hours (default 2 PM to 10 PM).
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return peak_start <= dt.hour < peak_end

def is_peak_rp4(dt, holidays, peak_days={0, 1, 2, 3, 4}, peak_start=14, peak_end=22):
    """
    RP4 peak period rule:
    - Peak: Mon–Fri, 14:00–22:00 (excluding public holidays)
    - Off-Peak: All other times
    
    Improved Hierarchy: Holiday Check → Weekday Check → Hour Check
    This clearer flow makes the logic more maintainable for both General and TOU tariffs.
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    
    # 1. HOLIDAY CHECK (first priority - clearest exclusion)
    if is_public_holiday(dt, holidays):
        return False
    
    # 2. WEEKDAY CHECK (second priority - excludes weekends)
    if dt.weekday() not in peak_days:
        return False
    
    # 3. HOUR CHECK (final constraint - MD recording window)
    return is_peak_hour(dt, peak_start, peak_end)

def classify_peak_period(df, timestamp_col, holidays=None, label_col="Period"):
    """
    Add a column (default: 'Period') indicating whether each timestamp is in Peak or Off-Peak.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp data
        timestamp_col (str): Name of the timestamp column
        holidays (set, optional): Set of holiday dates. If None, auto-detects from data years.
        label_col (str): Name of the column to add for period classification
    
    Returns:
        pd.DataFrame: DataFrame with added period classification column
    """
    df = df.copy()
    
    # Auto-detect holidays if not provided
    if holidays is None:
        holidays = detect_holidays_from_data(df, timestamp_col)
    
    df[label_col] = df[timestamp_col].apply(
        lambda ts: "Peak" if is_peak_rp4(ts, holidays) else "Off-Peak"
    )
    return df

def get_period_classification(dt, holidays=None):
    """
    Get the period classification (Peak/Off-Peak) for a single datetime.
    
    Args:
        dt (datetime-like): The datetime to classify
        holidays (set, optional): Set of holiday dates. If None, uses Malaysia 2025 holidays.
    
    Returns:
        str: "Peak" or "Off-Peak"
    """
    if holidays is None:
        # Default to Malaysia 2025 holidays for single datetime classification
        holidays = get_malaysia_holidays(2025)
    
    return "Peak" if is_peak_rp4(dt, holidays) else "Off-Peak"
