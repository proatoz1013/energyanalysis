# SOC Initialization Update - Double Limitation Fix

## Summary

Successfully updated the V2 MD Shaving Solution to address the "double limitation" issue in battery SOC initialization.

## Changes Made

### 1. SOC Initialization (Lines 5255 & 5297)
**Before:**
```python
current_soc_kwh = soc[i-1] if i > 0 else usable_capacity * 0.80  # Start at 80% SOC (within 5%-95% range)
# and
soc[i] = usable_capacity * 0.8
```

**After:**
```python
current_soc_kwh = soc[i-1] if i > 0 else battery_capacity * 0.95  # Start at 95% of total capacity
# and  
soc[i] = battery_capacity * 0.95  # Start at 95% of total capacity
```

### 2. SOC Percentage Calculations
Updated all SOC percentage calculations to use `battery_capacity` instead of `usable_capacity`:

**Line 5256:** Initial SOC percentage for C-rate calculations
**Line 5305:** SOC percentage for charging logic  
**Line 5481:** SOC feedback message
**Line 5502:** Final SOC percentage assignment to dataframe

**Before:**
```python
soc_percent = (soc_kwh / usable_capacity) * 100
```

**After:**
```python
soc_percent = (soc_kwh / battery_capacity) * 100  # Use total capacity for percentage
```

### 3. Battery_SOC_kWh Values
**Updated**: Both initialization points for the `soc` array now use `battery_capacity * 0.95`
- **Discharge path** (Line 5255): When battery is discharging during first interval
- **Charging path** (Line 5297): When battery is charging/idle during first interval

## Impact Analysis

### ✅ Benefits
1. **Removes Double Limitation**: Battery now starts at 95% of total capacity instead of 80% of usable capacity
2. **Consistent Energy Flow**: Efficiency adjustments and DoD limitations are now applied separately
3. **Proper SOC Representation**: SOC percentages now reflect actual battery state relative to total capacity  
4. **Correct Battery_SOC_kWh Values**: The actual energy values in the dataframe now start at the correct level
5. **Maintains Safety**: DoD protection still enforced through usable_capacity constraints (5%-95% limits)

### ✅ Preserved Functionality
1. **C-rate Calculations**: Still use total battery capacity (correctly implemented)
2. **Safety Constraints**: 5%-95% operational limits maintained using usable_capacity
3. **Charging Logic**: All charging algorithms work with updated SOC calculations
4. **Conservation Features**: SOC improvement calculations preserved for delta calculations

## Validation

- ✅ No syntax errors introduced
- ✅ All SOC percentage calculations updated consistently  
- ✅ Safety constraints maintained
- ✅ DoD protection applied only once (as intended)

## Result

The battery simulation now:
- Starts with **95% of total energy capacity** instead of **80% of usable capacity**
- Calculates SOC percentages based on **total capacity**
- Eliminates the double limitation effect while maintaining proper safety boundaries
- Provides more accurate energy flow representation

This change addresses the user's concern about potential double application of DoD limitations in the energy flow calculations.
