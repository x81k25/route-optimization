"""
Streamlit Route Optimization Dashboard - Main Application

This is the main entry point that orchestrates the different pages.
"""

# standard library imports
import sys
from pathlib import Path

# 3rd-party imports
import streamlit as st

# add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

# local imports
from aggregate_report import show_aggregate_report
from utils import load_data
from zone_details import show_zone_details

st.set_page_config(
    page_title="Route Optimization Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """
    Main application function.
    
    :return: None
    """
    # load data
    with st.spinner("loading data..."):
        itinerary_df, locations = load_data()
        
    if itinerary_df is None:
        st.error("unable to load data. please check that the output files exist.")
        return
    
    # sidebar for navigation
    st.sidebar.title("navigation")
    page = st.sidebar.selectbox("choose a page", ["aggregate report", "zone details"])
    
    if page == "aggregate report":
        show_aggregate_report(itinerary_df, locations)
    elif page == "zone details":
        show_zone_details(itinerary_df, locations)


if __name__ == "__main__":
    main()