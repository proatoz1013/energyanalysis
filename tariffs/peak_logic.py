import datetime

def is_peak_rp4(timestamp: datetime.datetime, holidays: set = None) -> bool:
    """Return True if timestamp is in RP4 Peak period; else False."""
    weekday = timestamp.weekday()  # 0=Mon, 6=Sun
    # Off-peak if weekend
    if weekday >= 5:
        return False
    # Off-peak if public holiday
    if holidays and timestamp.date() in holidays:
        return False
    # Peak = Weekday between 14:00–21:59
    return 14 <= timestamp.hour < 22
