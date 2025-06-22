import pandas as pd
import datetime

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
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    if is_public_holiday(dt, holidays):
        return False
    if dt.weekday() not in peak_days:
        return False
    return is_peak_hour(dt, peak_start, peak_end)

def classify_peak_period(df, timestamp_col, holidays, label_col="Period"):
    """
    Add a column (default: 'Period') indicating whether each timestamp is in Peak or Off-Peak.
    """
    df = df.copy()
    df[label_col] = df[timestamp_col].apply(
        lambda ts: "Peak" if is_peak_rp4(ts, holidays) else "Off-Peak"
    )
    return df
