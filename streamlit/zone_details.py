"""
Zone Details Page for Route Optimization Dashboard.

Displays detailed metrics and daily itinerary for individual zones.
"""

# standard library imports
import sys
from pathlib import Path

# 3rd-party imports
import pandas as pd
import streamlit as st

# add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

# local imports
from utils import (
    calculate_zone_metrics,
    create_zone_specific_map,
    generate_detailed_itinerary_table
)


def show_zone_details(itinerary_df, locations):
    """
    Display zone details page.
    
    :param itinerary_df: DataFrame with route optimization results
    :param locations: Dictionary of location data
    :return: None
    """
    st.header("zone details")
    
    # zone selector
    zones = sorted(itinerary_df['zone_id'].unique())
    selected_zone = st.selectbox("select a zone:", zones)
    
    if not selected_zone:
        return
    
    # Filter data for selected zone
    zone_data = itinerary_df.filter(itinerary_df['zone_id'] == selected_zone)
    
    # Calculate zone-specific metrics
    zone_metrics_df = calculate_zone_metrics(itinerary_df)
    zone_metrics = zone_metrics_df[zone_metrics_df['zone_id'] == selected_zone].iloc[0]
    
    # Display zone summary
    st.subheader(f"zone {selected_zone} summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("primary locations", int(zone_metrics['primary_pos_count']))
    
    with col2:
        st.metric("secondary locations", int(zone_metrics['secondary_pos_count']))
    
    with col3:
        st.metric("weekly duration", f"{zone_metrics['weekly_duration']:.2f} hrs")
    
    with col4:
        st.metric("utilization", f"{zone_metrics['utilization']:.1f}%")
    
    # Zone-specific map
    st.subheader(f"zone {selected_zone} map")
    
    with st.spinner("creating zone map..."):
        zone_map_fig = create_zone_specific_map(itinerary_df, locations, selected_zone)
    
    if zone_map_fig:
        # Add styling for the zone map  
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
        
        st.plotly_chart(zone_map_fig, width='stretch')
    else:
        st.warning(f"no map data available for {selected_zone}")
    
    # Daily summary table
    st.subheader("daily summary")
    
    daily_data = []
    for row in zone_data.iter_rows(named=True):
        pos_ids = row['pos_id'] or []
        pos_classes = row['pos_class'] or []
        pos_durations = row['pos_duration'] or []
        
        daily_data.append({
            'day': row['day'],
            'locations': len(pos_ids),
            'primary': sum(1 for pc in pos_classes if pc == 'primary'),
            'secondary': sum(1 for pc in pos_classes if pc == 'secondary'),
            'duration (min)': row['duration'],
            'POS time (min)': sum(pos_durations) if pos_durations else 0,
            'drive time (min)': max(0, row['duration'] - (sum(pos_durations) if pos_durations else 0))
        })
    
    daily_df = pd.DataFrame(daily_data)
    
    st.dataframe(
        daily_df,
        width='stretch',
        hide_index=True,
        column_config={
            "day": st.column_config.NumberColumn("day", width="small"),
            "locations": st.column_config.NumberColumn("locations", width="small"),
            "primary": st.column_config.NumberColumn("prime_pos", width="small"),
            "secondary": st.column_config.NumberColumn("sec_pos", width="small"),
            "duration (min)": st.column_config.NumberColumn("duration_min", format="%.0f"),
            "POS time (min)": st.column_config.NumberColumn("pos_time_min", format="%.0f"),
            "drive time (min)": st.column_config.NumberColumn("drive_time_min", format="%.0f")
        }
    )
    
    # Daily itinerary table
    st.subheader("daily itinerary")
    
    # Generate detailed itinerary table data
    detailed_table_data = generate_detailed_itinerary_table(zone_data, locations)
    
    if detailed_table_data:
        # Create DataFrame from detailed table data
        detailed_df = pd.DataFrame(detailed_table_data)
        
        st.dataframe(
            detailed_df,
            width='stretch',
            hide_index=True,
            column_config={
                "day": st.column_config.TextColumn("day", width="small"),
                "stop_num": st.column_config.TextColumn("#", width="small"),
                "time": st.column_config.TextColumn("time", width="medium"),
                "location": st.column_config.TextColumn("location", width="large"),
                "activity": st.column_config.TextColumn("activity", width="small"),
                "duration": st.column_config.TextColumn("duration", width="small")
            }
        )
    else:
        st.info("no itinerary data available for this zone")