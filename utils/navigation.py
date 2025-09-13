"""
Navigation utilities for the Streamlit Energy Analysis Dashboard.
This module handles the rendering and logic for sidebar navigation.
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config.navigation_config import get_navigation_for_tab, get_all_navigation_sections
except ImportError:
    # Fallback: direct import using absolute path
    config_path = os.path.join(parent_dir, 'config')
    sys.path.insert(0, config_path)
    from navigation_config import get_navigation_for_tab, get_all_navigation_sections

def render_sidebar_navigation(current_tab):
    """
    Render navigation for the current tab in the sidebar
    
    Args:
        current_tab (str): Name of the current tab
    """
    # Create a unique identifier for this tab's navigation
    nav_id = f"nav_{current_tab.replace(' ', '_').replace('(', '').replace(')', '')}"
    
    # Check if this navigation is already rendered to avoid duplicates
    if f"rendered_{nav_id}" in st.session_state:
        return
    
    # Get navigation items for current tab
    nav_items = get_tab_navigation_info(current_tab)
    
    if not nav_items:
        return
    
    # Mark this navigation as rendered
    st.session_state[f"rendered_{nav_id}"] = True
    
    # Navigation header
    st.sidebar.markdown("### 📑 Quick Navigation")
    st.sidebar.markdown(f"**{current_tab}**")
    
    # Show all navigation items directly
    for item in nav_items:
        # Create anchor link
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

def render_navigation_link(section):
    """
    Render a single navigation link with tooltip.
    
    Args:
        section (dict): Section configuration with name, anchor, description
    """
    name = section.get("name", "Unknown Section")
    anchor = section.get("anchor", "#")
    description = section.get("description", "")
    
    # Create the markdown link with tooltip
    if description:
        link_text = f"[{name}]({anchor})"
        st.sidebar.markdown(f"{link_text}")
        # Add description as smaller text
        st.sidebar.markdown(f"<small style='color: #666; margin-left: 1em;'>{description}</small>", 
                          unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"[{name}]({anchor})")

def get_tab_navigation_info(tab_name):
    """
    Get navigation items for a specific tab.
    
    Args:
        tab_name (str): Name of the tab
        
    Returns:
        list: List of navigation items for the tab, sorted by order
    """
    sections = get_navigation_for_tab(tab_name)
    
    # Sort by order to ensure proper sequence
    sections.sort(key=lambda x: x.get("order", 0))
    
    return sections

def create_navigation_overview():
    """
    Create an overview of all available navigation sections.
    Useful for debugging or showing users what's available.
    
    Returns:
        dict: Overview of navigation structure
    """
    all_sections = get_all_navigation_sections()
    
    overview = {
        "total_sections": len(all_sections),
        "categories": {},
        "sections": all_sections
    }
    
    # Group by category
    for section in all_sections:
        category = section.get("category", "uncategorized")
        if category not in overview["categories"]:
            overview["categories"][category] = []
        overview["categories"][category].append(section)
    
    return overview

def inject_navigation_css():
    """
    Inject custom CSS for better navigation styling.
    """
    st.markdown("""
    <style>
    /* Custom styles for navigation */
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    
    /* Style navigation links */
    .sidebar .markdown-text-container a {
        text-decoration: none;
        color: #1f77b4;
        transition: color 0.2s ease;
    }
    
    .sidebar .markdown-text-container a:hover {
        color: #ff7f0e;
        text-decoration: underline;
    }
    
    /* Style section descriptions */
    .sidebar small {
        display: block;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    /* Add some spacing between sections */
    .sidebar .markdown-text-container p {
        margin-bottom: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)
