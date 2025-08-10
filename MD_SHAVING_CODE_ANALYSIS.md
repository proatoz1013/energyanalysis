# MD Shaving Solution - Code Structure and Flow Analysis

## Overview
The MD Shaving Solution provides comprehensive Maximum Demand (MD) shaving analysis with tariff-aware color logic and battery simulation capabilities. This document explains the code structure, flow of logic, and implementation details.

## Main Components

### 1. Entry Point (`show()` function)
- **Location**: `md_shaving_solution.py` lines 177-450
- **Purpose**: Main interface function that handles file upload, configuration, and analysis initiation
- **Key Features**:
  - Configurable default values for shaving parameters
  - Three target setting methods: Percentage to Shave, Percentage of Current Max, Manual Target
  - File upload and data validation
  - Tariff selection integration

### 2. Color Logic System

#### Core Function: `create_conditional_demand_line_with_peak_logic()`
- **Location**: `md_shaving_solution.py` lines 46-150
- **Purpose**: Creates color-coded demand lines based on tariff-specific peak period logic
- **Color Logic**:
  - **Red**: Above target during Peak Periods (MD cost impact)
  - **Green**: Above target during Off-Peak Periods (No MD cost impact for TOU)
  - **Blue**: Below target (Within acceptable limits)

#### Tariff Classification Functions
- **`get_tariff_period_classification()`** (line 2846): Main classifier that routes to tariff-specific logic
- **`_classify_tou_tariff_periods()`** (line 2878): TOU tariff logic (2PM-10PM weekdays = Peak)
- **`_classify_general_tariff_periods()`** (line 2895): General tariff logic (always Peak for MD purposes)
- **`_get_tariff_description()`** (line 2910): Helper for descriptive text

### 3. Chart Implementation

#### A. Battery Simulation Chart (`_display_battery_simulation_chart()`)
- **Location**: `md_shaving_solution.py` lines 2203-2800
- **Purpose**: Shows battery operation effectiveness with 5 panels
- **Panels**:
  1. **MD Shaving Effectiveness**: Original vs Net demand with target line
  2. **SOC vs Battery Power**: Combined chart showing state of charge and power usage
  3. **Battery Power Utilization Heatmap**: Hourly utilization patterns
  4. **Daily Peak Shave Effectiveness**: Success/failure analysis for MD periods only
  5. **Cumulative Energy Analysis**: Energy discharged vs required over time

#### B. Peak Events Timeline Chart (`_display_peak_events_chart()`)
- **Location**: `md_shaving_solution.py` lines 1129-1200
- **Purpose**: Visualizes peak events with color-coded highlighting
- **Features**:
  - Uses same color logic as battery simulation
  - Filled areas for peak events (red for peak period, green for off-peak)
  - Target demand line overlay

### 4. Analysis Flow

#### Main Analysis Function: `_perform_md_shaving_analysis()`
- **Location**: `md_shaving_solution.py` lines 765-890
- **Flow**:
  1. Detect data interval
  2. Display tariff-specific color explanations
  3. Extract MD rates from selected tariff
  4. Detect peak events using `_detect_peak_events()`
  5. Calculate potential savings
  6. Display battery sizing recommendations
  7. Perform battery analysis if requested

#### Peak Event Detection
- **Function**: `_detect_peak_events()` (referenced from `advanced_energy_analysis.py`)
- **Logic**: Groups consecutive intervals above target, calculates energy and MD costs
- **Output**: Event summaries with timing, energy, and cost impact data

### 5. Battery Analysis Integration

#### Battery Algorithms Integration
- **Import**: `from battery_algorithms import (...)`
- **Functions Used**:
  - `get_battery_parameters_ui()`: UI for battery configuration
  - `perform_comprehensive_battery_analysis()`: Main analysis engine
  - `create_battery_algorithms()`: Algorithm creation

#### Simulation Function: `_simulate_battery_operation()`
- **Location**: `md_shaving_solution.py` lines 1926-2020
- **Purpose**: Simulates battery charge/discharge cycles
- **Key Features**:
  - MD-focused success rate calculation
  - Peak reduction analysis
  - SOC management
  - Performance metrics calculation

## Color Logic Implementation Details

### TOU (Time of Use) Tariffs
```python
if is_tou_tariff:
    if weekday < 5 and 14 <= hour < 22:
        return 'Peak'    # High energy rate + MD recording
    else:
        return 'Off-Peak'  # Low energy rate
```

### General Tariffs
```python
# For General tariffs, everything is "Peak" for MD visualization
# since MD charges apply 24/7 regardless of time
return 'Peak'
```

### Color Application Logic
```python
if demand_value > target_demand:
    if period_type == 'Peak':
        color_class = 'red'    # MD cost impact
    else:
        color_class = 'green'  # No MD cost impact (TOU only)
else:
    color_class = 'blue'      # Within target
```

## Data Flow Diagram

```
File Upload
    ↓
Data Processing & Validation
    ↓
Tariff Selection
    ↓
Target Demand Setting
    ↓
Peak Event Detection
    ↓
Color Logic Application
    ↓
Chart Generation (Peak Events + Battery Simulation)
    ↓
Battery Analysis (Optional)
    ↓
Results Display
```

## Key Functions and Their Relationships

### Chart Generation Flow
1. **Main Analysis** → `_perform_md_shaving_analysis()`
2. **Event Display** → `_display_peak_event_results()` → `_display_peak_events_chart()`
3. **Battery Analysis** → `perform_comprehensive_battery_analysis()` → `_display_battery_analysis()`
4. **Color Logic** → `create_conditional_demand_line_with_peak_logic()` (used in both charts)

### Tariff-Aware Logic Flow
1. **Tariff Selection** → `_configure_tariff_selection()`
2. **Period Classification** → `get_tariff_period_classification()`
3. **Color Assignment** → `create_conditional_demand_line_with_peak_logic()`
4. **Chart Rendering** → Both peak events and battery simulation charts

## Configuration and Customization

### Sidebar Configuration
- **Default Values**: Configurable shaving percentages and targets
- **Quick Presets**: Conservative (5%), Moderate (10%), Aggressive (20%)
- **Tariff Selection**: RP4 tariff integration with rate display

### Analysis Options
- **Event Filtering**: All, Peak Period Only, Off-Peak Period Only, MD Cost Impact Events
- **Detailed Analysis**: Toggle for comprehensive results
- **Threshold Sensitivity**: Analysis of different target thresholds

## Implementation Status

### Working Components
✅ **Core Functions**: All main functions are syntactically correct and importable
✅ **Color Logic**: Tariff-aware period classification working correctly  
✅ **Data Structures**: Event summaries and analysis data properly structured
✅ **Integration**: Battery algorithms and tariff modules properly imported

### Potential Issues Identified
⚠️ **Incomplete Functions**: Some functions have partial implementations
⚠️ **Missing Error Handling**: Limited error handling in chart generation
⚠️ **Streamlit Context**: Chart functions require Streamlit runtime context
⚠️ **Data Dependencies**: Charts depend on properly formatted event summaries

## Usage Examples

### Basic Usage
```python
# Import and run MD Shaving Solution
from md_shaving_solution import show
show()
```

### Function Testing
```python
# Test color logic
result = get_tariff_period_classification(timestamp, selected_tariff)
# Test chart creation
fig = create_conditional_demand_line_with_peak_logic(fig, df, 'demand', target)
```

## Troubleshooting Guide

### Common Issues
1. **Charts Not Displaying**: Ensure Streamlit context is available
2. **Color Logic Not Working**: Verify tariff selection is properly configured
3. **No Peak Events**: Check target demand setting and data format
4. **Battery Analysis Fails**: Verify event summaries are properly formatted

### Debug Steps
1. Check imports: All required modules should import without errors
2. Verify data format: DataFrame must have proper datetime index
3. Confirm tariff selection: Selected tariff object must have required structure
4. Test functions individually: Use Python console to test specific functions

## Technical Dependencies

### Required Modules
- `streamlit`: UI framework
- `pandas`: Data manipulation
- `plotly`: Chart generation  
- `numpy`: Numerical operations
- Custom modules: `tariffs.rp4_tariffs`, `tariffs.peak_logic`, `battery_algorithms`

### Data Requirements
- **Input Data**: CSV/Excel with timestamp and power columns
- **Tariff Data**: RP4 tariff structure with rates and classifications
- **Holiday Data**: Optional for accurate peak period classification

This analysis shows that the MD Shaving Solution has a well-structured codebase with comprehensive functionality. The color logic implementation is sophisticated and properly handles both TOU and General tariffs. Both chart sections (Battery Simulation and Peak Events Timeline) use the same underlying color logic system for consistency.
