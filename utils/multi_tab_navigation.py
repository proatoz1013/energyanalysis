"""
Multi-tab navigation system for Streamlit Energy Analysis Dashboard.
This module provides navigation that shows all tabs and their sections.
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config.navigation_config import get_navigation_for_tab, TAB_NAVIGATION_MAP
except ImportError:
    # Fallback: direct import using absolute path
    config_path = os.path.join(parent_dir, 'config')
    sys.path.insert(0, config_path)
    from navigation_config import get_navigation_for_tab, TAB_NAVIGATION_MAP

def render_multi_tab_navigation():
    """
    Render navigation for all tabs in an organized way
    """
    st.sidebar.markdown("### 📑 Quick Navigation")
    st.sidebar.markdown("Navigate to any section across all tabs:")
    
    # Tab names
    tab_names = [
        "TNB New Tariff Comparison", 
        "Load Profile Analysis", 
        "Advanced Energy Analysis", 
        "Monthly Rate Impact Analysis", 
        "MD Shaving Solution", 
        "🔋 MD Shaving (v2)", 
        "🔋 MD Shaving (v3)", 
        "📊 MD Patterns", 
        "🔋 Advanced MD Shaving", 
        "❄️ Chiller Energy Dashboard"
    ]
    
    # Render navigation for each tab
    for tab_name in tab_names:
        nav_items = get_navigation_for_tab(tab_name)
        
        if nav_items:
            # Create collapsible section for each tab
            with st.sidebar.expander(f"🔍 {tab_name}", expanded=tab_name=="🔋 MD Shaving (v2)"):
                for item in nav_items:
                    anchor = item['anchor']
                    name = item['name']
                    description = item.get('description', '')
                    
                    # Simplified navigation link
                    st.markdown(f"[{name}]({anchor})")
                    if description:
                        st.markdown(f"<small style='color: #666;'>{description}</small>", 
                                  unsafe_allow_html=True)

def render_focused_tab_navigation(current_tab):
    """
    Render navigation for a specific tab only
    
    Args:
        current_tab (str): Name of the current tab
    """
    # Only render if we haven't already rendered navigation
    if "navigation_rendered" not in st.session_state:
        st.session_state.navigation_rendered = set()
    
    if current_tab in st.session_state.navigation_rendered:
        return
    
    nav_items = get_navigation_for_tab(current_tab)
    
    if not nav_items:
        return
    
    # Clear previous navigation
    st.session_state.navigation_rendered.clear()
    st.session_state.navigation_rendered.add(current_tab)
    
    # Navigation header
    st.sidebar.markdown("### 📑 Quick Navigation")
    st.sidebar.markdown(f"**{current_tab}**")
    
    # Show navigation items
    for item in nav_items:
        anchor = item['anchor']
        name = item['name']
        description = item.get('description', '')
        
        # Navigation link with description
        link_html = f"""
        <div style="margin: 5px 0;">
            <a href="#{anchor}" style="
                color: #1f77b4; 
                text-decoration: none; 
                font-weight: 500;
                display: block;
                padding: 5px 10px;
                border-left: 3px solid #1f77b4;
                background-color: rgba(31, 119, 180, 0.1);
                border-radius: 4px;
                transition: all 0.3s ease;
            " 
            onmouseover="this.style.backgroundColor='rgba(31, 119, 180, 0.2)'"
            onmouseout="this.style.backgroundColor='rgba(31, 119, 180, 0.1)'">
                {name}
            </a>
            {f'<small style="color: #666; margin-left: 10px; display: block; margin-top: 2px;">{description}</small>' if description else ''}
        </div>
        """
        st.sidebar.markdown(link_html, unsafe_allow_html=True)
