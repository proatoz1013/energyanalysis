# TOU Event Detection Implementation

## Summary
Implemented TOU-aware event detection in `battery_performance_comparator.py` by reusing existing logic from `smart_conservation.py`.

## Changes Made

### 1. Added Imports
```python
from smart_conservation import MdOrchestrator, MdExcess, SmartConstants, TriggerEvents
```

### 2. Created `_detect_events_with_tou_logic()` Method
**Location**: Lines 90-136 in `battery_performance_comparator.py`

**Purpose**: Detect events following TOU tariff logic where events are active when BOTH:
1. `excess_demand > 0` (current demand exceeds target)
2. `inside_md_window = True` (currently in MD recording period)

**TOU Logic**:
- For TOU tariffs: MD window is only 2PM-10PM weekdays (excluding holidays)
- For General tariffs: MD window is 24/7

**Implementation**:
- Reuses `SmartConstants.is_md_active()` for MD window checking
- Reuses `TriggerEvents.set_event_state()` for event state logic
- Chains existing methods from `smart_conservation.py` (no duplication)

### 3. Updated Event Detection in Methods

#### `_run_default_shaving()` (Line 458)
**Before**:
```python
result_df['is_event'] = excess_demand > 0
```

**After**:
```python
# Use TOU-aware event detection (respects tariff type and MD window)
result_df['is_event'] = self._detect_events_with_tou_logic(result_df, excess_demand)
```

#### `_run_simple_conservation()` (Line 583)
**Before**:
```python
result_df['is_event'] = excess_demand > 0
```

**After**:
```python
# Use TOU-aware event detection (respects tariff type and MD window)
result_df['is_event'] = self._detect_events_with_tou_logic(result_df, excess_demand)
```

## Technical Details

### Method Chain from `smart_conservation.py`
1. **SmartConstants.is_md_active()**: Checks tariff type and determines MD window status
   - Uses `SmartConstants.is_tou_tariff()` to identify TOU tariffs
   - Uses `SmartConstants.is_peak_rp4()` for RP4 peak period logic (weekday 2PM-10PM, excluding holidays)
   - Returns True for General tariffs (24/7 MD recording)

2. **TriggerEvents.set_event_state()**: Applies boolean logic to determine event status
   - Returns True only when `excess > 0 AND inside_md_window`
   - Updates `_MdEventState` object with event status

### Benefits
- **Consistency**: Uses same TOU logic as V3 smart conservation
- **Reusability**: Chains existing methods (no code duplication)
- **Maintainability**: Changes to TOU logic in `smart_conservation.py` automatically propagate
- **Accuracy**: Respects tariff type and holiday calendars for event detection

## Verification Checklist
- ✅ Imports added: `SmartConstants`, `TriggerEvents`
- ✅ New method created: `_detect_events_with_tou_logic()`
- ✅ Updated `_run_default_shaving()` to use TOU-aware detection
- ✅ Updated `_run_simple_conservation()` to use TOU-aware detection
- ✅ No syntax errors
- ✅ Method chains existing `smart_conservation.py` functions
- ✅ Respects tariff type (TOU vs General)
- ✅ Respects MD window periods (RP4 peak logic)

## Testing Recommendations
1. Test with TOU tariff data (verify events only during 2PM-10PM weekdays)
2. Test with General tariff data (verify events 24/7)
3. Test with holiday calendar (verify holidays excluded for TOU)
4. Compare results with V3 smart conservation event detection
5. Verify comparison table shows correct event counts
