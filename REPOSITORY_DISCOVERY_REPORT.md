# Load Profile & MD Shaving Repository Discovery Report
**Generated:** December 28, 2024  
**Version:** 1.0

## Executive Summary

This repository contains a comprehensive energy analysis platform built with Streamlit, focused on Maximum Demand (MD) shaving and load profile analysis. The system integrates Malaysia's RP4 tariff structure with advanced battery energy storage system (BESS) analysis capabilities.

## Project Architecture Overview

### Core Application Structure
```
streamlit_app.py (Main Entry Point)
├── 9 Tabs: TNB Comparison, Load Profile, Advanced Analysis, Monthly Impact, 
│            MD Shaving, MD Shaving v2, MD Patterns, Advanced MD Shaving, Chiller Dashboard
├── Session State Management
└── Cross-tab Data Sharing
```

### Key Technology Stack
- **Frontend:** Streamlit with Plotly visualizations
- **Data Processing:** Pandas, NumPy
- **Analysis:** Custom algorithms for peak detection, battery simulation
- **File Support:** CSV, Excel (.xls/.xlsx), multi-format handling
- **Visualization:** Plotly (charts, heatmaps, time series)

## Module Analysis

### 1. Main Application Hub (`streamlit_app.py`)
**Purpose:** Central orchestrator and primary user interface
**Key Features:**
- 9 integrated analysis tabs
- Cross-module data sharing via session state
- Battery investment configuration
- Global MD shaving configuration sidebar
- File upload handling with robust error management

**Data Flow:**
```
File Upload → Session State Storage → Module Distribution → Analysis Results
```

### 2. MD Shaving Solution (`md_shaving_solution.py`)
**Purpose:** Core MD shaving analysis with RP4 tariff integration
**Key Components:**
- **File Upload**: Multi-format support (CSV, Excel)
- **Data Processing**: Automatic interval detection, timestamp handling
- **Peak Event Detection**: RP4-aware peak period logic (2PM-10PM weekdays)
- **Battery Analysis**: Comprehensive BESS sizing and simulation
- **Tariff Integration**: RP4 capacity + network rates
- **Financial Analysis**: ROI, payback period, lifecycle costs

**Architecture:**
```python
show() → File Upload → Data Configuration → Tariff Selection → 
Peak Event Detection → Battery Analysis → Financial Modeling
```

### 3. MD Shaving Solution v2 (`md_shaving_solution_v2.py`)
**Purpose:** Enhanced MD shaving with monthly-based calculations
**Key Enhancements:**
- Monthly target calculation (vs overall target in v1)
- Battery database integration
- Interactive capacity selection
- Enhanced visualization with peak events timeline

**Inheritance Pattern:**
```python
# Reuses v1 components:
from md_shaving_solution import (
    read_uploaded_file, _configure_data_inputs, 
    _process_dataframe, _configure_tariff_selection,
    create_conditional_demand_line_with_peak_logic
)
```

### 4. MD Pattern Analysis (`md_pattern_analysis.py`)
**Purpose:** Advanced pattern recognition and analysis
**Core Features:**
- Daily, weekly, monthly pattern detection
- Peak event identification with statistical analysis
- Demand forecasting capabilities
- Battery sizing recommendations
- Investment analysis integration

**New Implementation:**
- Built from scratch with comprehensive file upload functionality
- Clones MD Shaving v2 capabilities
- Focuses on pattern recognition vs immediate peak shaving

### 5. Advanced Energy Analysis (`advanced_energy_analysis.py`)
**Purpose:** Comprehensive energy analysis with RP4 integration
**Key Components:**
- Peak/Off-Peak analysis with RP4 logic
- Cost analysis with current RP4 rates
- Advanced peak event detection
- Load Duration Curve (LDC) analysis
- Threshold sensitivity analysis

### 6. Battery Algorithms (`battery_algorithms.py`)
**Purpose:** Sophisticated battery simulation and optimization
**Core Algorithms:**
- **Tariff-Aware Discharge**: TOU vs General tariff compliance
- **Smart Charging**: RP4 period-aware charging strategies
- **Financial Modeling**: NPV, IRR, payback calculations
- **Compliance Validation**: Peak period discharge verification

**Algorithm Structure:**
```python
class BatteryAlgorithms:
    ├── calculate_optimal_sizing()
    ├── simulate_battery_operation()  # Tariff-aware
    ├── calculate_financial_metrics()
    └── _validate_tariff_compliance()
```

### 7. Tariff & Utility Modules
**Structure:**
```
tariffs/
├── rp4_tariffs.py      # RP4 tariff definitions and rates
├── peak_logic.py       # Malaysia holidays and peak period logic
└── __init__.py

utils/
├── cost_calculator.py     # New RP4 cost calculations
├── old_cost_calculator.py # Legacy tariff calculations
└── holiday_api.py         # Holiday management
```

## Data Flow Architecture

### 1. File Upload → Processing Pipeline
```
User Upload → read_uploaded_file() → Format Detection (CSV/Excel) → 
DataFrame Creation → Validation → Session State Storage
```

### 2. Data Configuration Flow
```
Column Selection → Timestamp Processing → Index Setting → 
Holiday Configuration → Interval Detection → Validation
```

### 3. Analysis Execution Flow
```
Processed Data → Tariff Selection → Peak Event Detection → 
Battery Simulation → Financial Analysis → Visualization
```

### 4. Cross-Module Integration
```
Session State Keys:
├── uploaded_file
├── processed_df  
├── power_column
├── timestamp_column
├── battery_* (investment parameters)
└── md_* (shaving configuration)
```

## Key Technical Features

### 1. Robust File Handling
- **Multi-format Support**: CSV, .xls, .xlsx
- **Error Recovery**: Encoding fallbacks, validation layers
- **Data Validation**: Column existence, format verification
- **Preprocessing**: Timestamp normalization, month name conversion

### 2. Advanced Peak Detection
- **RP4 Logic**: Malaysia-specific peak period rules (2PM-10PM weekdays)
- **Holiday Integration**: Automatic holiday detection and management
- **Event Classification**: TOU vs General tariff impacts
- **Energy Calculations**: Interval-aware energy consumption

### 3. Battery Simulation Engine
- **Tariff-Aware Discharge**: TOU (peak-only) vs General (24/7) strategies
- **Smart Charging**: Period-aware charging optimization
- **SOC Management**: Depth of discharge and efficiency modeling
- **Compliance Tracking**: Violation detection and reporting

### 4. Financial Modeling
- **ROI Calculations**: Annual return on investment
- **NPV Analysis**: Net present value with discount rates
- **Payback Periods**: Simple and discounted payback
- **Lifecycle Costs**: 15-20 year total cost of ownership

## Integration Points for New MD Analysis Module

### 1. File Upload Integration
**Existing Pattern:**
```python
def read_uploaded_file(file):
    """Standard file reading with multi-format support"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
```

**Recommendation:** Use existing `read_uploaded_file()` from any MD module for consistency.

### 2. Data Processing Hooks
**Existing Function:**
```python
def _process_dataframe(df, timestamp_col):
    """Standard preprocessing pipeline"""
    # Timestamp parsing, index setting, sorting
```

**Integration Point:** All modules use this pattern for data standardization.

### 3. Battery Interface Integration
**Existing Interface:**
```python
from battery_algorithms import (
    get_battery_parameters_ui,
    perform_comprehensive_battery_analysis,
    create_battery_algorithms
)
```

**Integration Strategy:** New analysis modules can directly plug into existing battery algorithms.

### 4. Tariff System Integration
**Existing Pattern:**
```python
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4
from utils.cost_calculator import calculate_cost
```

**Usage:** All MD analysis modules use these for consistent tariff handling.

### 5. Session State Integration
**Current Keys:**
```python
st.session_state.processed_df     # Processed data
st.session_state.power_column     # Selected power column
st.session_state.battery_*        # Investment parameters
st.session_state.md_*             # MD configuration
```

**Integration Point:** New modules can share data through established session state pattern.

## Current Module Dependencies

### MD Shaving Solution Dependencies
```python
# Core dependencies
from tariffs.rp4_tariffs import get_tariff_data
from tariffs.peak_logic import is_peak_rp4, get_period_classification  
from utils.cost_calculator import calculate_cost
from battery_algorithms import (
    get_battery_parameters_ui,
    perform_comprehensive_battery_analysis,
    create_battery_algorithms
)
```

### Cross-Module Shared Functions
1. **`read_uploaded_file()`** - Used by all file upload modules
2. **`_process_dataframe()`** - Standard data preprocessing
3. **`fmt()`** - Number formatting for display
4. **RP4 tariff functions** - Shared tariff logic
5. **Battery algorithms** - Centralized battery analysis

## Recommendations for New Analysis Integration

### 1. Follow Established Patterns
- Use existing file upload functions for consistency
- Leverage battery_algorithms module for BESS analysis
- Integrate with RP4 tariff system for cost calculations
- Follow session state patterns for data sharing

### 2. Integration Strategy
```python
# New analysis module template
from md_shaving_solution import read_uploaded_file, _process_dataframe
from battery_algorithms import perform_comprehensive_battery_analysis
from tariffs.rp4_tariffs import get_tariff_data
from utils.cost_calculator import calculate_cost

def new_analysis_function():
    # Use established upload pattern
    df = read_uploaded_file(uploaded_file)
    df = _process_dataframe(df, timestamp_col)
    
    # Integrate with existing battery analysis
    battery_analysis = perform_comprehensive_battery_analysis(...)
    
    # Use existing tariff system
    cost_breakdown = calculate_cost(df, selected_tariff, ...)
```

### 3. Enhancement Opportunities
- **Data Caching**: Implement caching for large dataset processing
- **Parallel Processing**: Optimize battery simulation algorithms
- **Export Functions**: Standardized data export across modules
- **Validation Framework**: Enhanced data validation patterns

## Technical Specifications

### Supported File Formats
- **CSV**: UTF-8, Latin-1 encoding support
- **Excel**: .xls (legacy), .xlsx (modern)
- **Data Intervals**: 15-min, 30-min, hourly (auto-detected)

### Battery Analysis Capabilities
- **Technologies**: Li-ion, LiFePO4, Sodium-ion
- **Sizing Methods**: Auto-sizing, manual capacity, duration-based
- **Financial Analysis**: 15-20 year lifecycle modeling
- **Compliance**: RP4 TOU vs General tariff strategies

### Visualization Features
- **Time Series**: Load profiles with peak highlighting
- **Heatmaps**: Battery utilization patterns
- **Financial Charts**: ROI and payback visualizations
- **Comparative Analysis**: Before/after MD shaving scenarios

## Conclusion

The repository represents a mature, well-architected energy analysis platform with sophisticated MD shaving capabilities. The modular design enables easy integration of new analysis features while maintaining consistency across the application. The existing battery algorithms, tariff integration, and data processing frameworks provide a solid foundation for extending functionality.

Key strengths include robust file handling, comprehensive battery simulation, accurate RP4 tariff integration, and sophisticated financial modeling. The architecture supports both immediate analysis needs and long-term system expansion.
