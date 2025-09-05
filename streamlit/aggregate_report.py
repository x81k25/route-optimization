"""
Aggregate Report Page for Route Optimization Dashboard.

Displays zone aggregate metrics, summary statistics, and all-zones map.
"""

# standard library imports
import sys
from pathlib import Path

# 3rd-party imports
import streamlit as st

# add current directory to path to import local modules  
sys.path.insert(0, str(Path(__file__).parent))

# local imports
from utils import (
    calculate_zone_metrics,
    create_zones_map,
    load_aggregate_metrics
)


def show_aggregate_report(itinerary_df, locations):
    """
    Display the aggregate report page.
    
    :param itinerary_df: DataFrame with route optimization results
    :param locations: Dictionary of location data
    :return: None
    """
    st.header("zone aggregate report")
    
    # load pre-calculated aggregate summary
    with st.spinner("loading aggregate metrics..."):
        summary_stats = load_aggregate_metrics()
        zone_metrics_df = calculate_zone_metrics(itinerary_df)
    
    if summary_stats is None or zone_metrics_df is None:
        st.error("unable to load aggregate metrics.")
        return
    
    # Summary statistics from aggregate_summary.jsonl
    st.subheader("summary statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("total zones", len(zone_metrics_df))
    
    with col2:
        avg_duration = summary_stats.get('average_weekly_duration', 0)
        st.metric("avg weekly duration", f"{avg_duration:.1f} hrs")
    
    with col3:
        avg_utilization = summary_stats.get('average_utilization', 0)
        st.metric("avg utilization", f"{avg_utilization:.1f}%")
    
    with col4:
        avg_pos_time = summary_stats.get('average_daily_pos_time', 0)
        st.metric("avg POS time", f"{avg_pos_time:.1f} hrs")
    
    with col5:
        avg_drive_time = summary_stats.get('average_daily_drive_time', 0)
        st.metric("avg drive time", f"{avg_drive_time:.1f} hrs")
    
    st.markdown("---")
    
    # Zone-level metrics table
    st.subheader("zone-level metrics")
    
    # Format the dataframe for display
    display_df = zone_metrics_df.copy()
    display_df.columns = [
        'zone ID', 'primary locations', 'secondary locations', 
        'weekly duration (hrs)', 'utilization (%)', 'overutilized days',
        'underutilized days', 'total POS time (hrs)', 'total drive time (hrs)', 'sec std dev_hrs'
    ]
    
    # Display with formatting - using shorter column labels
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "zone ID": st.column_config.TextColumn("zone_id", width="small"),
            "primary locations": st.column_config.NumberColumn("prime_pos", width="small"),
            "secondary locations": st.column_config.NumberColumn("sec_pos", width="small"),
            "weekly duration (hrs)": st.column_config.NumberColumn("weekly_hrs", format="%.2f"),
            "utilization (%)": st.column_config.NumberColumn("util_%", format="%.1f"),
            "overutilized days": st.column_config.NumberColumn("days_over", width="small"),
            "underutilized days": st.column_config.NumberColumn("days_under", width="small"),
            "total POS time (hrs)": st.column_config.NumberColumn("POS_hrs", format="%.2f"),
            "total drive time (hrs)": st.column_config.NumberColumn("drive_hrs", format="%.2f"),
            "sec std dev_hrs": st.column_config.NumberColumn("sec_std", format="%.2f", width="small")
        }
    )
    
    st.markdown("---")
    
    # Map
    st.subheader("all zones map")
    
    with st.spinner("creating map..."):
        map_fig = create_zones_map(itinerary_df, locations)
    
    if map_fig:
        # Add styling for the map
        st.markdown("""
        <style>
        div[data-testid="stPlotlyChart"] {
            border: 2px solid #666666;
            border-radius: 15px;
            overflow: hidden;
            margin: 1rem 0;
            padding: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[data-testid="stPlotlyChart"] > div {
            border-radius: 15px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(map_fig, width='stretch')
    else:
        st.error("unable to create map.")