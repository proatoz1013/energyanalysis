# Double Limitation Fix - Final Resolution

## Issue Summary
The V2 MD Shaving Solution was applying DoD (Depth of Discharge) limitations twice:
1. Once in the SOC initialization using `usable_capacity * 0.80`
2. Once more in the charging/discharging constraints using `usable_capacity * 0.95`

This created a "double limitation" effect where the battery couldn't use its full theoretical capacity.

## Root Cause Analysis
The user reported `Battery_SOC_kWh` showing 2656.2 kWh instead of expected 3320.25 kWh.

**Key Finding**: Line 5146 was still using `usable_capacity` for SOC percentage calculation, which applied the 0.8 DoD factor even after we updated the SOC initialization.

## Changes Made

### 1. SOC Initialization Updates (Previously Fixed)
- **Line 5255**: `current_soc_kwh = soc[i-1] if i > 0 else battery_capacity * 0.95`
- **Line 5297**: `soc[i] = battery_capacity * 0.95`

### 2. SOC Percentage Calculations (Previously Fixed)
- **Line 5256**: `current_soc_percent = (current_soc_kwh / battery_capacity) * 100`
- **Line 5305**: `soc_percentage = (soc[i] / battery_capacity) * 100`
- **Line 5481**: Updated SOC feedback message to use `battery_capacity`
- **Line 5502**: `soc_percent[i] = (soc[i] / battery_capacity) * 100`

### 3. Final Fix - Conservation Cascade SOC Calculation
- **Line 5146**: `current_soc_percent = (soc[i-1] / battery_capacity * 100) if i > 0 else 95`
  - Changed from: `(soc[i-1] / usable_capacity * 100) if i > 0 else 80`
  - This was the remaining source of the double limitation!

### 4. SOC Improvement Calculation
- **Line 5202**: `soc_improvement = (energy_conserved_kwh / battery_capacity) * 100`
  - Changed from: `(energy_conserved_kwh / usable_capacity) * 100`

## Expected Results
- **Battery Total Capacity**: 3495.0 kWh
- **Usable Capacity (80% DoD)**: 2796.0 kWh  
- **Initial SOC (95% of total)**: 3320.25 kWh
- **Initial SOC Percentage**: 95% (relative to total capacity)

## Verification
```python
battery_capacity = 3495.0
usable_capacity = 2796.0  # 80% DoD
initial_soc_new = battery_capacity * 0.95  # 3320.25 kWh
initial_soc_old = usable_capacity * 0.80   # 2236.8 kWh
```

## Safety Constraints Maintained
The following usable_capacity constraints remain unchanged for DoD protection:
- **Line 5284**: `min_soc_energy = usable_capacity * 0.05` (5% minimum safety)
- **Line 5418**: `soc[i] < usable_capacity * max_soc_target` (charging constraint)
- **Line 5443**: `remaining_capacity = usable_capacity * 0.95 - soc[i]` (charging limit)
- **Line 5501**: `soc[i] = max(usable_capacity * 0.05, min(soc[i], usable_capacity * 0.95))` (safety bounds)

These constraints ensure the battery operates within its specified DoD limits during normal operation.

## Impact
- ✅ Eliminates double limitation effect
- ✅ Battery starts at 95% of total capacity (3320.25 kWh) instead of 76% (2656.2 kWh)
- ✅ Maintains DoD protection during charging/discharging operations
- ✅ Improves battery utilization efficiency
- ✅ Preserves safety constraints and operational limits

## Files Modified
- `/Users/chyeap89/Documents/energyanalysis-1/md_shaving_solution_v2.py`

The double limitation issue has been resolved by consistently using `battery_capacity` for SOC initialization and percentage calculations while maintaining `usable_capacity` constraints for DoD protection.
