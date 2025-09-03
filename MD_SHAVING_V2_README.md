# MD Shaving Solution V2 - Enhanced MD Shaving Analysis

## 📋 Overview

MD Shaving Solution V2 is a next-generation Maximum Demand (MD) shaving analysis tool that provides advanced battery optimization, vendor database integration, and comprehensive financial analysis for energy management systems.

### 🆕 V2 Key Features
- **Monthly-based target calculation** with dynamic user settings
- **Battery database integration** with vendor specifications
- **Enhanced timeline visualization** with peak events clustering
- **Interactive battery capacity selection** interface
- **Advanced dispatch simulation** with engineering constraints
- **Comprehensive financial analysis** with ROI calculations

---

## 🏗️ Architecture & Code Structure

### **Section A: Data Foundation & Setup**

#### A1. Data Upload
📁 **Data Upload Interface**
- File upload functionality for energy consumption data
- Support for CSV, XLS, XLSX formats
- Data validation and preprocessing

**Functions:**
- `read_uploaded_file()` ⚠️ **External from md_shaving_solution.py**
- `st.file_uploader()` ⚠️ **External from streamlit**

**Dependencies:**
- pandas, streamlit
- External file: `md_shaving_solution.py`

#### A2. Configuration Setup
📋 **Data Configuration**
⚡ **Tariff Configuration**
🔧 **RP4 Tariff Configuration**

**Functions:**
- `_configure_data_inputs()` ⚠️ **External from md_shaving_solution.py**
- `_process_dataframe()` ⚠️ **External from md_shaving_solution.py**
- `_configure_tariff_selection()` ⚠️ **External from md_shaving_solution.py**

**Dependencies:**
- A1 functions (file upload results)
- External file: `md_shaving_solution.py`

#### A3. Target Definition
🎯 **Target Setting (V2)**
- Three targeting methods: Manual kW, Percentage to Shave, Percentage of Current Max
- Tariff-specific calculations (General vs TOU)
- Monthly-based optimization

**Functions:**
- `_calculate_monthly_targets_v2()` ✅ **Internal V2 function**
- `_calculate_tariff_specific_monthly_peaks()` ✅ **Internal V2 function**

**Dependencies:**
- A2 (processed dataframe, tariff config)
- `is_peak_rp4()` ⚠️ **External from tariffs.peak_logic**

### **Section B: Peak Analysis & Detection**

#### B1. Peak Event Analysis
📊 **Peak Events Timeline**
📋 **Monthly Target Calculation Summary**
⚡ **Peak Event Detection Results**

**Functions:**
- `_render_v2_peak_events_timeline()` ✅ **Internal V2 function**
- `_infer_interval_hours()` ✅ **Internal V2 function**
- `_detect_peak_events()` ⚠️ **External from md_shaving_solution.py**
- `create_conditional_demand_line_with_peak_logic()` ⚠️ **External from md_shaving_solution.py**

**Dependencies:**
- A3 (monthly targets)
- External file: `md_shaving_solution.py`
- External file: `tariffs/peak_logic.py`

#### B2. Peak Clustering & Power Analysis
🔗 **Peak Event Clusters**
⚡ **Peak Power & Energy Analysis**

**Functions:**
- `cluster_peak_events()` ✅ **Internal V2 function**
- `build_daily_simulator_structure()` ✅ **Internal V2 function**

**Dependencies:**
- B1 (peak events data)
- `is_peak_rp4()` ⚠️ **External from tariffs.peak_logic**

### **Section C: Battery Database & Recommendations**

#### C1. Battery Database Integration
🔋 **Recommended Battery Capacity**
💡 **Battery Capacity Recommendation**
🔋 **Battery Unit Requirements**

**Functions:**
- `load_vendor_battery_database()` ✅ **Internal V2 function**
- `get_battery_capacity_range()` ✅ **Internal V2 function**
- `get_battery_options_for_capacity()` ✅ **Internal V2 function**

**Dependencies:**
- B2 (clustering results for capacity calculations)
- **External file**: `vendor_battery_database.json` ⚠️ **Required JSON file**

#### C2. Battery Selection & Configuration
📋 **Tabled Analysis**
🔋 **Battery Sizing & Financial Analysis**

**Functions:**
- `_render_battery_selection_dropdown()` ✅ **Internal V2 function**
- `_render_battery_sizing_analysis()` ✅ **Internal V2 function**
- `_render_v2_battery_controls()` ✅ **Internal V2 function**

**Dependencies:**
- C1 (battery database, capacity calculations)
- B2 (max power and energy requirements from clustering)
- **External file**: `vendor_battery_database.json` ⚠️ **Required JSON file**

### **Section D: Battery Simulation & Performance**

#### D1. Core Simulation
🔋 **Battery Simulation Analysis**
📊 **Battery Operation Simulation**

**Functions:**
- `_simulate_battery_operation()` ⚠️ **External from md_shaving_solution.py**
- `_display_battery_simulation_chart()` ⚠️ **External from md_shaving_solution.py**

**Dependencies:**
- C2 (battery specifications and configurations)
- A2 (processed dataframe)
- External file: `md_shaving_solution.py`

#### D2. Enhanced Performance Visualization
1️⃣ **MD Shaving Effectiveness: Demand vs Battery vs Target**
2️⃣ **Combined SOC and Battery Power Chart**
3️⃣ **Battery Power Utilization Heatmap**

**Functions:**
- `_render_battery_impact_timeline()` ✅ **Internal V2 function**
- `render_battery_impact_visualization()` ✅ **Internal V2 function**

**Dependencies:**
- D1 (simulation results)
- A3 (monthly targets)
- C2 (battery configuration)

#### D3. Success Analysis & Optimization
4️⃣ **Daily Peak Shave Effectiveness & Success Analysis (MD Peak Periods Only)**
🔍 **Detailed Success/Failure Analysis**
5️⃣ **Cumulative Energy Discharged vs Required (MD Peak Periods Only)**

**Functions:**
- `_determine_constraint_type()` ✅ **Internal V2 function**

**Dependencies:**
- D1 (simulation data)
- B2 (clustering data for constraint analysis)
- C2 (battery specifications for power/energy limits)

### **Section E: Comprehensive Analysis & Financial Results**

#### E1. Enhanced Insights & Analytics
🔍 **Key Insights from Enhanced Analysis**

**Functions:**
- Complex chart generation functions using **plotly.graph_objects** and **plotly.express**
- Advanced data analytics embedded in `_render_v2_peak_events_timeline()`

**Dependencies:**
- All previous sections (A-D)
- External libraries: `plotly.graph_objects`, `plotly.express`
- `numpy` for statistical calculations

#### E2. BESS Dispatch & Financial Analysis
🔋 **BESS Dispatch Simulation & Comprehensive Analysis**
📊 **Dispatch Simulation Results**
💰 **Monthly Savings Analysis**

**Functions:**
- Complex dispatch simulation logic embedded in `_render_v2_peak_events_timeline()`
- Monthly savings calculation algorithms

**Dependencies:**
- All previous sections (A-D)
- C1 (battery database for unit costs)
- D1 (simulation framework for dispatch modeling)
- B2 (clustering results for dispatch optimization)

### **Section F: Main Interface & Coordination**

#### F1. Main Application Function
**Main entry point and workflow coordination**

**Functions:**
- `render_md_shaving_v2()` ✅ **Internal V2 function**
- `show()` ✅ **Internal V2 function (compatibility)**

**Dependencies:**
- All functions from Sections A-E
- `streamlit` library components for UI coordination

---

## 🔗 Dependencies & Requirements

### **Critical External Files**
1. **`md_shaving_solution.py`** ⚠️ **CRITICAL DEPENDENCY**
   - Contains 8 essential functions reused by V2
   - Provides proven data processing and simulation logic
   - Required for: file upload, data processing, tariff config, peak detection, battery simulation

2. **`tariffs/peak_logic.py`** ⚠️ **CRITICAL DEPENDENCY**
   - Contains `is_peak_rp4()` function
   - Required for TOU tariff calculations
   - Used in monthly peak calculations

3. **`vendor_battery_database.json`** ⚠️ **REQUIRED DATA FILE**
   - Battery specifications database
   - Vendor information and technical parameters
   - Cost information for financial analysis

### **External Libraries**
```python
import streamlit as st           # UI framework
import pandas as pd             # Data processing and analysis
import numpy as np              # Numerical computations
import plotly.graph_objects as go  # Advanced visualizations
import plotly.express as px     # Quick visualizations
from datetime import datetime, timedelta  # Time handling
import json                     # Database file parsing
```

### **Function Execution Dependencies**
```
External Files → A1 → A2 → A3 → B1 → B2 → C1 → C2 → D1 → D2/D3 → E1 → E2 → F1
```

### **Section-Level Dependencies**
- **Section A** depends on: External files only
- **Section B** depends on: Section A + external peak logic
- **Section C** depends on: Section B (for sizing calculations) + JSON database
- **Section D** depends on: Sections B & C + V1 simulation functions
- **Section E** depends on: All sections A-D for comprehensive analysis
- **Section F** depends on: All sections A-E as main coordinator

---

## 🚀 Installation & Setup

### **1. Environment Setup**
```bash
# Clone the repository
cd /Users/chyeap89/Documents/energyanalysis

# Install required packages
pip install streamlit pandas numpy plotly openpyxl matplotlib seaborn
```

### **2. Required Files**
Ensure these files are present in your workspace:
- `md_shaving_solution.py` (V1 components)
- `tariffs/peak_logic.py` (Peak logic functions)
- `vendor_battery_database.json` (Battery database)

### **3. Running the Application**
```bash
# Launch the main Streamlit app
streamlit run streamlit_app.py

# Or run V2 directly (if configured as standalone)
streamlit run md_shaving_solution_v2.py
```

### **4. Accessing V2 Features**
1. Navigate to the main application at `http://localhost:8501`
2. Select "🔋 MD Shaving Solution (v2)" from the sidebar
3. Upload your energy data file (CSV, XLS, XLSX)
4. Configure data columns and tariff settings
5. Set monthly targets using V2 targeting methods
6. Select batteries from the vendor database
7. Run comprehensive analysis and simulations

---

## 📊 Usage Guide

### **Data Upload Requirements**
Your data file should contain:
- **Timestamp column**: Date and time information
- **Power column**: Power consumption values in kW
- **Supported formats**: CSV, Excel (.xls, .xlsx)

### **Target Setting Methods**
1. **Manual Target (kW)**: Set specific target values for all months
2. **Percentage to Shave**: Reduce current peaks by specified percentage
3. **Percentage of Current Max**: Set targets as percentage of monthly peaks

### **Battery Selection Options**
1. **By Capacity**: Use slider to select desired capacity, view matching batteries
2. **By Specific Model**: Choose exact battery model from vendor database

### **Analysis Outputs**
- Peak events timeline with monthly targets
- Battery sizing recommendations
- Financial analysis with investment costs
- Dispatch simulation results
- Success/failure analysis
- Monthly savings projections

---

## 🔧 Development Status

### **Completed Features**
- ✅ UI Framework and basic structure
- ✅ Integration with existing V1 data processing
- ✅ Enhanced interface design
- ✅ Battery database integration with vendor specifications
- ✅ Monthly-based target calculation
- ✅ Interactive battery capacity selection
- ✅ Peak event clustering algorithms
- ✅ Daily simulation structure
- ✅ Financial analysis framework

### **In Development**
- 🔄 Advanced battery optimization algorithms
- 🔄 Multi-scenario comparison engine
- 🔄 Enhanced cost analysis and ROI calculations
- 🔄 Advanced visualization suite
- 🔄 Complete dispatch simulation implementation

### **Planned Features**
- 📋 AI-powered battery sizing recommendations
- 📋 Real-time optimization suggestions
- 📋 Advanced reporting and export capabilities
- 📋 Extended battery vendor database integration
- 📋 Multi-site analysis capabilities

---

## ⚠️ Risk Assessment

### **High Risk Dependencies**
- **Missing `md_shaving_solution.py`**: Breaks 75% of functionality
- Core data processing, simulation, and visualization functions

### **Medium Risk Dependencies**
- **Missing `vendor_battery_database.json`**: Breaks battery analysis (Sections C-E)
- Battery selection, sizing, and financial analysis unavailable

### **Low Risk Dependencies**
- **Missing `tariffs/peak_logic.py`**: Affects TOU calculations only
- General tariff calculations still functional

### **Very Low Risk Dependencies**
- **Missing plotly libraries**: Affects visualizations only
- Core analysis functions remain operational

---

## 🆕 V2 Innovations

### **Key V2 Enhancements**
- **50%+ new internal functions** for enhanced capabilities
- **Advanced battery database integration** with real vendor specifications
- **Sophisticated clustering algorithms** for optimal battery dispatch
- **Comprehensive financial analysis** with monthly breakdown
- **Engineering constraint modeling** for realistic performance analysis
- **Strategic reuse of V1 components** for proven reliability

### **Technical Improvements**
- Tariff-specific monthly peak calculations
- Peak event clustering for battery optimization
- Daily simulation framework with TOU considerations
- Advanced constraint analysis for engineering limits
- Interactive battery selection with real specifications
- Enhanced financial modeling with ROI calculations

---

## 📞 Support & Contact

### **Development Team**
- **Author**: Enhanced MD Shaving Team
- **Version**: 2.0
- **Date**: August 2025

### **Documentation**
- Main README: This file
- Code Analysis: `MD_SHAVING_CODE_ANALYSIS.md`
- Implementation Status: `IMPLEMENTATION_STATUS_REPORT.md`
- Enhancement Summary: `ENHANCEMENT_SUMMARY.md`

### **File Structure**
```
/Users/chyeap89/Documents/energyanalysis/
├── md_shaving_solution_v2.py          # Main V2 application
├── md_shaving_solution.py             # V1 components (dependency)
├── vendor_battery_database.json       # Battery database
├── tariffs/peak_logic.py              # Peak logic functions
└── [other supporting files]
```

---

## 🔄 Version History

- **V2.0** (August 2025): Enhanced MD shaving with battery database integration
- **V1.0** (Previous): Original MD shaving solution with basic battery analysis

---

*This README provides comprehensive documentation for the MD Shaving Solution V2 architecture, dependencies, and usage guidelines based on detailed code structure analysis.*
