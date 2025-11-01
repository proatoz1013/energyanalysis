# MD Saving Algorithm Critical Fix

## Problem Identified
The monthly MD saving calculation in `_create_monthly_summary_table()` contained a critical logic error that caused approximately **70% overestimation** of savings.

## Root Cause
The algorithm was incorrectly identifying the "peak day" for monthly billing by finding the day with the **highest MD excess** instead of the day with the **highest original demand**.

### Wrong Logic (Before Fix)
```python
# INCORRECT: Finding day with maximum MD excess
peak_excess_day_idx = month_data['MD_Excess_Numeric'].idxmax()
```

### Correct Logic (After Fix)
```python
# CORRECT: Finding day with maximum original demand (billing peak day)
peak_billing_day_idx = month_data['Max_Original_Demand_Numeric'].idxmax()
```

## Why This Matters
In utility billing, the **monthly maximum demand charge** is determined by the day with the **highest original demand**, regardless of whether that was the day with the best battery performance.

### Example from User Data (August 2025)
- **August 19th**: 975.0 kW original demand, 63.7 kW MD excess → High battery savings day
- **August 25th**: 1,038.6 kW original demand, 19.3 kW MD excess → **Actual billing peak day**

### Impact of Fix
- **Before Fix (Wrong)**: RM 6,182.72 savings (based on August 19th's 63.7 kW excess)
- **After Fix (Correct)**: RM 1,873.26 savings (based on August 25th's 19.3 kW shaving)
- **Correction**: ~70% reduction in overestimated savings

## Implementation Details
The fix was implemented in `/Users/chyeap89/Documents/energyanalysis/energyanalysis/md_shaving_solution_v2.py` at line ~7076:

1. Added extraction of `Max_Original_Demand_Numeric` column
2. Changed peak day identification from MD excess to original demand
3. Updated comments to explain the critical nature of this fix

## Validation Required
After this fix, users should see:
1. More realistic monthly savings calculations
2. Savings based on actual utility billing methodology
3. Proper identification of billing peak days in monthly summaries

## Status
✅ **FIXED** - Algorithm now correctly identifies billing peak days based on maximum original demand
