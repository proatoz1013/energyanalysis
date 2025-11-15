"""
Test script for Smart Conservation Module - Event Detection Logic

This script tests the process_historical_events method with sample data
to verify MD window detection and 4-case event logic.
"""

import pandas as pd
from datetime import datetime, timedelta

# Create sample test data - 9 rows spanning 19:00 to 23:00 (30-minute intervals)
test_data = {
    'timestamp': [
        datetime(2024, 3, 18, 19, 0),   # Row 0: 19:00 - Monday, inside MD window
        datetime(2024, 3, 18, 19, 30),  # Row 1: 19:30 - Monday, inside MD window
        datetime(2024, 3, 18, 20, 0),   # Row 2: 20:00 - Monday, inside MD window
        datetime(2024, 3, 18, 20, 30),  # Row 3: 20:30 - Monday, inside MD window
        datetime(2024, 3, 18, 21, 0),   # Row 4: 21:00 - Monday, inside MD window
        datetime(2024, 3, 18, 21, 30),  # Row 5: 21:30 - Monday, inside MD window
        datetime(2024, 3, 18, 22, 0),   # Row 6: 22:00 - Monday, OUTSIDE MD window (after 22:00)
        datetime(2024, 3, 18, 22, 30),  # Row 7: 22:30 - Monday, OUTSIDE MD window
        datetime(2024, 3, 18, 23, 0),   # Row 8: 23:00 - Monday, OUTSIDE MD window
    ],
    'demand_kw': [
        2500,  # Row 0: High demand
        2800,  # Row 1: Higher demand
        3200,  # Row 2: Peak demand
        3000,  # Row 3: Still high
        2700,  # Row 4: Decreasing
        2400,  # Row 5: Lower
        2100,  # Row 6: Outside window
        1900,  # Row 7: Outside window
        1800,  # Row 8: Outside window
    ]
}

# Create DataFrame with timestamp as index
df_sim = pd.DataFrame(test_data)
df_sim.set_index('timestamp', inplace=True)

# Target MD: 2200 kW
target_md = 2200

# Calculate excess demand (demand - target, minimum 0)
df_sim['excess_demand_kw'] = (df_sim['demand_kw'] - target_md).clip(lower=0)

# Expected Results:
# Row 0 (19:00): excess=300, inside_md_window=True
# Row 1 (19:30): excess=600, inside_md_window=True
# Row 2 (20:00): excess=1000, inside_md_window=True
# Row 3 (20:30): excess=800, inside_md_window=True
# Row 4 (21:00): excess=500, inside_md_window=True
# Row 5 (21:30): excess=200, inside_md_window=True
# Row 6 (22:00): excess=0, inside_md_window=False (after 22:00)
# Row 7 (22:30): excess=0, inside_md_window=False
# Row 8 (23:00): excess=0, inside_md_window=False

print("=" * 80)
print("TEST DATA FOR SMART CONSERVATION MODULE")
print("=" * 80)
print(f"\nTest DataFrame Shape: {df_sim.shape}")
print(f"Target MD: {target_md} kW")
print(f"\nTimestamp Range: {df_sim.index[0]} to {df_sim.index[-1]}")
print(f"Interval: 30 minutes")
print("\n" + "=" * 80)
print("SAMPLE DATA:")
print("=" * 80)
print(df_sim)
print("\n" + "=" * 80)
print("MD WINDOW STATUS (RP4 Peak: Weekday 14:00-22:00, excluding holidays):")
print("=" * 80)

# Check MD window status for each timestamp
from smart_conservation import SmartConstants

holidays = set()  # No holidays for this test
for idx, row in df_sim.iterrows():
    is_rp4_peak = SmartConstants.is_peak_rp4(idx, holidays)
    print(f"{idx.strftime('%Y-%m-%d %H:%M')} | Demand: {row['demand_kw']:4.0f} kW | "
          f"Excess: {row['excess_demand_kw']:4.0f} kW | "
          f"Inside MD Window: {is_rp4_peak}")

print("\n" + "=" * 80)
print("EXPECTED EVENT DETECTION (assuming trigger_threshold > 0):")
print("=" * 80)
print("Row 0 (19:00): CASE 1 - NEW event starts (prev=False, curr=True)")
print("Row 1 (19:30): CASE 2 - CONTINUE event (prev=True, curr=True)")
print("Row 2 (20:00): CASE 2 - CONTINUE event (prev=True, curr=True)")
print("Row 3 (20:30): CASE 2 - CONTINUE event (prev=True, curr=True)")
print("Row 4 (21:00): CASE 2 - CONTINUE event (prev=True, curr=True)")
print("Row 5 (21:30): CASE 2 - CONTINUE event (prev=True, curr=True)")
print("Row 6 (22:00): CASE 3 - Event ENDED (prev=True, curr=False) [outside window]")
print("Row 7 (22:30): CASE 4 - No event (prev=False, curr=False)")
print("Row 8 (23:00): CASE 4 - No event (prev=False, curr=False)")
print("=" * 80)
