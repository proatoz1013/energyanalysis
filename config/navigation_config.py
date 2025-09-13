
"""
Navigation configuration for the Streamlit Energy Analysis Dashboard.
This file contains all navigation sections and their configurations.
"""

# Navigation sections with structured format matching MD Shaving v2 workflow
NAVIGATION_SECTIONS = [
    # General sections (available in multiple tabs)
    {
        "name": "📊 Load Profile Analysis",
        "anchor": "#load-profile-analysis",
        "category": "general",
        "description": "Analyze load patterns and consumption trends"
    },
    {
        "name": "💰 Cost Analysis", 
        "anchor": "#cost-analysis",
        "category": "general",
        "description": "Compare costs across different tariffs"
    },
    {
        "name": "📈 Results Summary",
        "anchor": "#results-summary", 
        "category": "general",
        "description": "Summary of analysis results and recommendations"
    },
    
    # MD Shaving (v2) structured sections - matching actual implementation
    {
        "name": "🔋 1. MD Shaving Solution (v2)",
        "anchor": "#1-md-shaving-solution-v2",
        "category": "md_shaving_v2",
        "order": 1,
        "description": "Overview of MD Shaving version 2 solution"
    },
    {
        "name": "📁 2. Data Upload",
        "anchor": "#2-data-upload",
        "category": "md_shaving_v2", 
        "order": 2,
        "description": "Upload and validate energy consumption data"
    },
    {
        "name": "📋 3. Data Configuration",
        "anchor": "#3-data-configuration",
        "category": "md_shaving_v2",
        "order": 3,
        "description": "Configure data processing parameters"
    },
    {
        "name": "⚡ 4. Tariff Configuration",
        "anchor": "#4-tariff-configuration",
        "category": "md_shaving_v2",
        "order": 4,
        "description": "Configure tariff settings and parameters"
    },
    {
        "name": "🎯 5. Target Setting (V2)",
        "anchor": "#5-target-setting-v2",
        "category": "md_shaving_v2",
        "order": 5,
        "description": "Set MD shaving targets and strategy"
    },
    {
        "name": "📊 6. Peak Events Timeline",
        "anchor": "#6-peak-events-timeline",
        "category": "md_shaving_v2",
        "order": 6,
        "description": "Timeline analysis of peak demand events"
    },
    {
        "name": "📋 6.1 Monthly Target Calculation",
        "anchor": "#6-1-monthly-target-calculation-summary",
        "category": "md_shaving_v2",
        "order": 6.1,
        "description": "Monthly target calculation summary"
    },
    {
        "name": "⚡ 6.2 Peak Event Detection",
        "anchor": "#6-2-peak-event-detection-results",
        "category": "md_shaving_v2",
        "order": 6.2,
        "description": "Peak event detection results"
    },
    {
        "name": "🔗 6.3 Peak Event Clusters",
        "anchor": "#6-3-peak-event-clusters",
        "category": "md_shaving_v2",
        "order": 6.3,
        "description": "Peak event clustering analysis"
    },
    {
        "name": "⚡ 6.4 Peak Power & Energy Analysis",
        "anchor": "#64-peak-power-energy-analysis",
        "category": "md_shaving_v2",
        "order": 6.4,
        "description": "Peak power and energy analysis"
    },
    {
        "name": "🔋 6.5 Battery Sizing Analysis",
        "anchor": "#65-battery-sizing-analysis",
        "category": "md_shaving_v2",
        "order": 6.5,
        "description": "Battery sizing and capacity analysis"
    },
    {
        "name": "🔋 6.6 Battery Simulation Analysis",
        "anchor": "#66-battery-simulation-analysis",
        "category": "md_shaving_v2",
        "order": 6.6,
        "description": "Battery operation simulation"
    },
    {
        "name": "🔋 6.7 BESS Dispatch Simulation",
        "anchor": "#67-bess-dispatch-simulation-comprehensive-analysis",
        "category": "md_shaving_v2",
        "order": 6.7,
        "description": "BESS dispatch simulation and comprehensive analysis"
    },
    {
        "name": "💰 6.7.1 Monthly Savings Analysis",
        "anchor": "#671-monthly-savings-analysis",
        "category": "md_shaving_v2",
        "order": 6.71,
        "description": "Monthly savings analysis"
    },
    {
        "name": "📋 7. Tabled Analysis",
        "anchor": "#7-tabled-analysis",
        "category": "md_shaving_v2",
        "order": 7,
        "description": "Tabulated analysis results"
    },
    {
        "name": "🔢 7.1 Battery Quantity Recommendation",
        "anchor": "#7-1-battery-quantity-recommendation",
        "category": "md_shaving_v2",
        "order": 7.1,
        "description": "Battery quantity recommendations"
    },
    {
        "name": "🔋 7.2 Battery Sizing & Financial Analysis",
        "anchor": "#7-2-battery-sizing-financial-analysis",
        "category": "md_shaving_v2",
        "order": 7.2,
        "description": "Financial analysis of battery investment"
    },
    {
        "name": "📊 8. Battery Impact on Energy Consumption",
        "anchor": "#8-battery-impact-on-energy-consumption",
        "category": "md_shaving_v2",
        "order": 8,
        "description": "Analysis of battery impact on energy consumption"
    },
    {
        "name": "📊 Battery Operation Simulation",
        "anchor": "#battery-operation-simulation",
        "category": "md_shaving_v2",
        "order": 9,
        "description": "Battery operation simulation results"
    },
    {
        "name": "🌅 TOU Tariff Performance Analysis",
        "anchor": "#tou-tariff-performance-analysis",
        "category": "md_shaving_v2",
        "order": 10,
        "description": "Time-of-Use tariff performance analysis"
    },
    {
        "name": "🔋 TOU Tariff Performance Summary",
        "anchor": "#tou-tariff-performance-summary",
        "category": "md_shaving_v2",
        "order": 11,
        "description": "TOU tariff performance summary"
    },
    {
        "name": "📊 TOU vs General Tariff Comparison",
        "anchor": "#tou-vs-general-tariff-comparison",
        "category": "md_shaving_v2",
        "order": 12,
        "description": "Comparison between TOU and General tariffs"
    }
]

# Tab configuration - maps tab names to their content sections
TAB_NAVIGATION_MAP = {
    "TNB New Tariff Comparison": ["general"],
    "Load Profile Analysis": ["general"],
    "Advanced Energy Analysis": ["general"],
    "Monthly Rate Impact Analysis": ["general"],
    "MD Shaving Solution": ["general"],
    "🔋 MD Shaving (v2)": ["general", "md_shaving_v2"],
    "🔋 MD Shaving (v3)": ["general"],
    "📊 MD Patterns": ["general"],
    "🔋 Advanced MD Shaving": ["general"],
    "❄️ Chiller Energy Dashboard": ["general"]
}

def get_navigation_for_tab(tab_name):
    """
    Get navigation sections for a specific tab.
    
    Args:
        tab_name (str): Name of the tab
        
    Returns:
        list: List of navigation sections for the tab
    """
    categories = TAB_NAVIGATION_MAP.get(tab_name, ["general"])
    sections = []
    
    for section in NAVIGATION_SECTIONS:
        if section.get("category") in categories:
            sections.append(section)
    
    # Sort MD Shaving v2 sections by order (handles decimal ordering like 6.1, 6.2, etc.)
    def sort_key(section):
        if section.get("category") == "md_shaving_v2":
            return (0, section.get("order", 0))
        else:
            return (1, 0)
    
    sections.sort(key=sort_key)
    return sections

def get_all_navigation_sections():
    """
    Get all navigation sections.
    
    Returns:
        list: All navigation sections
    """
    return NAVIGATION_SECTIONS

def get_sections_by_category(category):
    """
    Get navigation sections by category.
    
    Args:
        category (str): Category name (e.g., 'general', 'md_shaving_v2')
        
    Returns:
        list: List of sections in the category
    """
    return [section for section in NAVIGATION_SECTIONS if section.get("category") == category]
