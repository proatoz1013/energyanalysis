# Peak and Off-Peak Calculation Logic: High Voltage vs Medium Voltage Comparison

## 📊 Executive Summary

**KEY FINDING: The peak/off-peak calculation logic is IDENTICAL for both High Voltage and Medium Voltage tariffs. The only differences are in the rates and charging rules, NOT in the time-based classification logic.**

## 🕐 Peak Period Definition (Universal)

All tariffs use the **same RP4 peak period rule**:
- **Peak**: Monday-Friday, 2:00 PM - 10:00 PM (excluding public holidays)  
- **Off-Peak**: All other times (weekends, holidays, weekday nights/mornings)

**Source**: `tariffs/peak_logic.py` - `is_peak_rp4()` function

---

## 📋 Side-by-Side Comparison Table

| **Aspect** | **Medium Voltage General** | **Medium Voltage TOU** | **High Voltage General** | **High Voltage TOU** |
|------------|---------------------------|------------------------|--------------------------|----------------------|
| **Peak Time Definition** | ❌ No peak/off-peak periods<br>(24/7 flat rate) | ✅ Mon-Fri 2PM-10PM<br>(excl. holidays) | ❌ No peak/off-peak periods<br>(24/7 flat rate) | ✅ Mon-Fri 2PM-10PM<br>(excl. holidays) |
| **Off-Peak Time Definition** | ❌ No off-peak concept | ✅ All other times<br>(weekends, holidays, nights) | ❌ No off-peak concept | ✅ All other times<br>(weekends, holidays, nights) |
| **Energy Rate Structure** | **Single Rate**: RM 0.2983/kWh | **Split Rate**:<br>Peak: RM 0.3132/kWh<br>Off-Peak: RM 0.2723/kWh | **Single Rate**: RM 0.4303/kWh | **Split Rate**:<br>Peak: RM 0.4452/kWh<br>Off-Peak: RM 0.4043/kWh |
| **MD Recording Window** | **24/7 Continuous**<br>(All times count) | **Peak Hours Only**<br>(Mon-Fri 2PM-10PM) | **24/7 Continuous**<br>(All times count) | **Peak Hours Only**<br>(Mon-Fri 2PM-10PM) |
| **Capacity Charge Basis** | kW (24/7 max demand) | kW (peak period max only) | kW (24/7 max demand) | kW (peak period max only) |
| **Network Charge Basis** | kW (24/7 max demand) | kW (peak period max only) | kW (24/7 max demand) | kW (peak period max only) |
| **Peak Detection Logic** | `return 'Peak'` (always) | `is_peak_rp4(timestamp, holidays)` | `return 'Peak'` (always) | `is_peak_rp4(timestamp, holidays)` |

---

## 🔍 Detailed Logic Analysis

### 1. **Peak/Off-Peak Time Classification**

**Function**: `is_peak_rp4(dt, holidays)` in `tariffs/peak_logic.py`

```python
def is_peak_rp4(dt, holidays, peak_days={0, 1, 2, 3, 4}, peak_start=14, peak_end=22):
    # 1. HOLIDAY CHECK (first priority)
    if is_public_holiday(dt, holidays):
        return False
    
    # 2. WEEKDAY CHECK (Mon-Fri = 0-4)
    if dt.weekday() not in peak_days:
        return False
    
    # 3. HOUR CHECK (2PM-10PM = 14-22)
    return is_peak_hour(dt, peak_start, peak_end)
```

**Result**: Same logic applied to ALL voltage levels and tariff types.

### 2. **Tariff-Specific Period Classification**

**General Tariffs** (both Medium & High Voltage):
```python
def _classify_general_tariff_periods(timestamp):
    return 'Peak'  # Always Peak (MD applies 24/7)
```

**TOU Tariffs** (both Medium & High Voltage):
```python
def _classify_tou_tariff_periods(timestamp):
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    if weekday < 5 and 14 <= hour < 22:
        return 'Peak'    # High energy rate + MD recording
    else:
        return 'Off-Peak'  # Low energy rate
```

### 3. **MD Recording Window Logic**

**General Tariffs**:
- **Medium Voltage**: Records MD 24/7 (any time above previous max = new MD)
- **High Voltage**: Records MD 24/7 (any time above previous max = new MD)

**TOU Tariffs**:
- **Medium Voltage**: Records MD only during peak hours (Mon-Fri 2PM-10PM)
- **High Voltage**: Records MD only during peak hours (Mon-Fri 2PM-10PM)

---

## 💰 Rate Structure Differences

### **Medium Voltage Rates** (RM/kWh)
| Tariff Type | Energy/Peak Rate | Off-Peak Rate | Capacity Rate | Network Rate |
|-------------|------------------|---------------|---------------|--------------|
| **General** | 0.2983 | N/A | 29.43 | 59.84 |
| **TOU** | 0.3132 | 0.2723 | 30.19 | 66.87 |

### **High Voltage Rates** (RM/kWh)
| Tariff Type | Energy/Peak Rate | Off-Peak Rate | Capacity Rate | Network Rate |
|-------------|------------------|---------------|---------------|--------------|
| **General** | 0.4303 | N/A | 16.68 | 14.53 |
| **TOU** | 0.4452 | 0.4043 | 21.76 | 23.06 |

**Key Observations**:
1. **High Voltage energy rates are higher** than Medium Voltage
2. **Medium Voltage capacity/network rates are higher** than High Voltage
3. **Off-peak savings**: MV TOU = 13.1%, HV TOU = 9.2%

---

## 🔧 Implementation Details

### **Cost Calculator Logic** (`utils/cost_calculator.py`)

**Same logic for all voltage levels**:

1. **Determine if TOU tariff**:
   ```python
   if not rules.get("has_peak_split", False):
       # General tariff logic
   else:
       # TOU tariff logic
   ```

2. **Calculate peak periods**:
   ```python
   is_peak = df["Parsed Timestamp"].apply(lambda ts: is_peak_rp4(ts, holidays))
   ```

3. **Split energy consumption**:
   ```python
   # For TOU only
   peak_kwh = interval_kwh[is_peak].sum()
   offpeak_kwh = interval_kwh[~is_peak].sum()
   ```

### **Battery Algorithm Logic** (`battery_algorithms.py`)

**Same discharge compliance rules**:

**TOU Tariffs** (both MV & HV):
```python
if is_peak_period:
    # ✅ COMPLIANT: Discharge during peak hours
    battery_action = calculate_discharge()
else:
    # ❌ VIOLATION: No discharge during off-peak
    log_violation()
```

**General Tariffs** (both MV & HV):
```python
# ✅ COMPLIANT: Discharge anytime above target
battery_action = calculate_discharge()
```

---

## ✅ Conclusion

**NO DIFFERENCES** in peak/off-peak calculation logic between:
- Medium Voltage vs High Voltage
- Within same tariff type (General vs General, TOU vs TOU)

**The ONLY differences are**:
1. **Rate values** (energy, capacity, network rates)
2. **Charging rule specifics** (kW vs kWh basis)
3. **Cost calculations** (due to different rates)

**The time-based classification (when is peak vs off-peak) is universal across all RP4 tariffs.**
