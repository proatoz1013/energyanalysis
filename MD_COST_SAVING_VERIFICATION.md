# MD Cost Saving Calculation Verification

## Overview
This document verifies the implementation of Maximum Demand (MD) cost savings calculation in the Monthly Summary section of the Battery Simulation Data Tables.

## Cost Saving Formula
```
Cost_Saving_RM = Success_Shaved_kW × MD_Rate_RM_per_kW
```

Where:
- **Success_Shaved_kW**: Actual MD shave achieved on the peak billing day
- **MD_Rate_RM_per_kW**: The MD rate in RM per kW from the tariff structure

## Algorithm Implementation (FIXED)

### Step 1: Daily Summary Analysis
The algorithm first creates a daily summary table containing:
- Date
- Max Original Demand (kW)
- Actual MD Shave (kW)
- MD Excess (kW)

### Step 2: Peak Billing Day Identification (CRITICAL FIX)
For each month, the algorithm identifies the **peak billing day** using:

```python
# CORRECT IMPLEMENTATION (Fixed)
peak_billing_day_idx = month_data['Max_Original_Demand_Numeric'].idxmax()
```

**Key Fix**: The algorithm now correctly identifies the day with the **highest original demand** as the peak billing day, not the day with the highest MD excess.

### Step 3: Monthly Success Shaved Calculation
```python
actual_shave_on_billing_day = month_data.loc[peak_billing_day_idx, 'Actual_MD_Shave_Numeric']
monthly_success_shaved[month_period] = actual_shave_on_billing_day
```

The monthly success shaved value is taken from the **specific day** that determines the monthly MD billing.

### Step 4: Cost Saving Calculation
```python
monthly_data['Cost_Saving_RM'] = monthly_data['Success_Shaved_kW'] * md_rate_rm_per_kw
```

## Why This Fix Was Critical

### Previous (Wrong) Logic:
- Found day with **highest MD excess above target**
- Could select a day with high excess but lower original demand
- **Result**: ~70% overestimation of savings

### Current (Correct) Logic:
- Finds day with **highest original demand** (determines billing)
- Uses actual MD shave from that specific day
- **Result**: Accurate monthly savings calculation

## Example Correction
**August Data Example:**
- **Day A (Aug 19th)**: 950 kW original, 63.7 kW excess → Wrong selection
- **Day B (Aug 25th)**: 1,038.6 kW original, 19.3 kW actual shave → Correct selection

**Impact**: 19.3 kW actual vs 63.7 kW overestimated = 70% reduction in overestimation

## Verification Points

✅ **Peak Day Selection**: Uses `Max_Original_Demand_Numeric.idxmax()`
✅ **Cost Formula**: `Success_Shaved_kW × MD_Rate_RM_per_kW`
✅ **Monthly Aggregation**: Takes MD shave from billing day, not monthly average
✅ **Data Flow**: Daily → Monthly → Cost calculation chain is correct

## Monthly Summary Table Columns
1. **Month**: Period identifier
2. **MD Excess (kW)**: Original demand minus target
3. **Success Shaved (kW)**: Actual MD shave from peak billing day
4. **Cost Saving (RM)**: Success Shaved × MD Rate
5. **Total EFC**: Equivalent Full Cycles for the month
6. **Accumulating Charging Cycles**: Cumulative EFC cycles

## Conclusion
The MD cost saving calculation has been corrected to accurately:
1. Identify the true peak billing day (highest original demand)
2. Extract the actual MD shave from that specific day
3. Calculate monthly cost savings based on billing reality
4. Prevent overestimation that was occurring with the previous logic

This ensures that the Monthly Summary provides realistic and actionable cost saving projections for battery MD shaving applications.
