import pandas as pd
import datetime

def is_peak_rp4(ts, holidays):
    """
    Returns True if the timestamp is in peak period according to RP4 rules:
    - Peak: Monday–Friday, 8:00–22:00, excluding public holidays.
    - Off-peak: All other times (including weekends and public holidays).
    Args:
        ts (datetime.datetime): Timestamp to check.
        holidays (set of datetime.date): Set of public holiday dates.
    Returns:
        bool: True if peak, False if off-peak.
    """
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    date = ts.date()
    # If public holiday, always off-peak
    if date in holidays:
        return False
    # If weekend, always off-peak
    if ts.weekday() >= 5:
        return False
    # Peak hours: 8:00 to 22:00 (8am to 10pm, inclusive of 8:00, exclusive of 22:00)
    if 8 <= ts.hour < 22:
        return True
    return False
