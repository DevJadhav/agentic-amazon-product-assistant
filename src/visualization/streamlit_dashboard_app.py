"""
Streamlit Dashboard Application
Interactive visualization app for Amazon product analytics.
"""

import streamlit as st
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.interactive_dashboards import InteractiveDashboard

def main():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Amazon Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run dashboard
    dashboard = InteractiveDashboard()
    dashboard.render_streamlit_dashboard()

if __name__ == "__main__":
    main()