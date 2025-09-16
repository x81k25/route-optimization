"""
Zone Summary Page for Route Optimization Dashboard.

Displays zone-level metrics with algorithm filtering, summary statistics, and all-zones map.
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
    filter_by_algorithm,
    get_latest_timestamp_data,
    get_unique_algorithms,
    load_aggregate_metrics
)


def show_zone_summary(itinerary_df) -> None:
    """
    Display the zone summary page with algorithm filtering.

    :param itinerary_df: DataFrame with route optimization results
    :return: None
    """
    st.header("Zone Summary")

    # load all data from Parquet files
    with st.spinner("loading zone metrics..."):
        # load all zone metrics data (no filtering yet)
        zone_metrics_df_full = calculate_zone_metrics(itinerary_df)

    if zone_metrics_df_full is None:
        st.error("unable to load zone metrics.")
        return

    # convert to polars for filtering if needed
    import polars as pl
    if hasattr(zone_metrics_df_full, 'to_pandas'):
        zone_metrics_pl = pl.from_pandas(zone_metrics_df_full)
    else:
        zone_metrics_pl = pl.DataFrame(zone_metrics_df_full)

    # get unique algorithm values for filters
    clusterers, balancers = get_unique_algorithms(zone_metrics_pl)

    # initialize session state for first load
    if 'zone_first_load' not in st.session_state:
        st.session_state.zone_first_load = True

    # on first load, get latest timestamp data and defaults
    if st.session_state.zone_first_load:
        zone_latest_df, default_clusterer, default_balancer = get_latest_timestamp_data(zone_metrics_pl)
        st.session_state.zone_first_load = False
        # store defaults in session state
        if 'zone_clusterer' not in st.session_state:
            st.session_state.zone_clusterer = default_clusterer
        if 'zone_balancer' not in st.session_state:
            st.session_state.zone_balancer = default_balancer

    # filters in sidebar
    st.sidebar.subheader("Zone Summary Filters")

    clusterer_filter = st.sidebar.selectbox(
        "Clusterer",
        options=clusterers,
        index=clusterers.index(st.session_state.zone_clusterer) if st.session_state.zone_clusterer in clusterers else 0,
        key="zone_clusterer_select"
    )

    balancer_filter = st.sidebar.selectbox(
        "Balancer",
        options=balancers,
        index=balancers.index(st.session_state.zone_balancer) if st.session_state.zone_balancer in balancers else 0,
        key="zone_balancer_select"
    )

    # update session state
    st.session_state.zone_clusterer = clusterer_filter
    st.session_state.zone_balancer = balancer_filter

    # filter data based on selections (ignores timestamp)
    filtered_zone_df = filter_by_algorithm(zone_metrics_pl, clusterer_filter, balancer_filter)
    zone_metrics_df = filtered_zone_df.to_pandas() if filtered_zone_df is not None else None

    if zone_metrics_df is None or len(zone_metrics_df) == 0:
        st.warning(f"No data found for {clusterer_filter} + {balancer_filter}")
        return

    # filter itinerary data for map
    filtered_itinerary_df = filter_by_algorithm(itinerary_df, clusterer_filter, balancer_filter)

    # show selected algorithm combination
    st.info(f"Showing results for: **{clusterer_filter}** + **{balancer_filter}**")
    
    # Summary statistics calculated from filtered zone data
    st.subheader("Summary Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Zones", len(zone_metrics_df))

    with col2:
        avg_duration = zone_metrics_df['weekly_duration'].mean() if 'weekly_duration' in zone_metrics_df.columns else 0
        st.metric("Avg Weekly Duration", f"{avg_duration:.1f} hrs")

    with col3:
        avg_utilization = zone_metrics_df['utilization'].mean() if 'utilization' in zone_metrics_df.columns else 0
        st.metric("Avg Utilization", f"{avg_utilization:.1f}%")

    with col4:
        avg_pos_time = zone_metrics_df['total_pos_time'].mean() if 'total_pos_time' in zone_metrics_df.columns else 0
        st.metric("Avg POS Time", f"{avg_pos_time:.1f} hrs")

    with col5:
        avg_drive_time = zone_metrics_df['total_drive_time'].mean() if 'total_drive_time' in zone_metrics_df.columns else 0
        st.metric("Avg Drive Time", f"{avg_drive_time:.1f} hrs")
    
    st.markdown("---")
    
    # Zone-level metrics table
    st.subheader("Zone-Level Metrics")

    # Format the dataframe for display
    display_df = zone_metrics_df.copy()

    # Remove unwanted columns
    columns_to_remove = ['clusterer', 'router', 'balancer', 'created_on']
    for col in columns_to_remove:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])

    # handle different possible column names from Parquet and JSONL
    column_mapping = {}
    if 'zone_id' in display_df.columns:
        column_mapping['zone_id'] = 'Zone ID'
    # Handle both naming conventions for primary/secondary counts
    if 'primary_pos_count' in display_df.columns:
        column_mapping['primary_pos_count'] = 'Primaries'
    elif 'primary_count' in display_df.columns:
        column_mapping['primary_count'] = 'Primaries'
    if 'secondary_pos_count' in display_df.columns:
        column_mapping['secondary_pos_count'] = 'Secondaries'
    elif 'secondary_count' in display_df.columns:
        column_mapping['secondary_count'] = 'Secondaries'
    if 'weekly_duration' in display_df.columns:
        column_mapping['weekly_duration'] = 'Weekly Duration (hrs)'
    if 'utilization' in display_df.columns:
        column_mapping['utilization'] = 'Utilization (%)'
    if 'overutilized_days' in display_df.columns:
        column_mapping['overutilized_days'] = 'Overutilized Days'
    if 'underutilized_days' in display_df.columns:
        column_mapping['underutilized_days'] = 'Underutilized Days'
    if 'total_pos_time' in display_df.columns:
        column_mapping['total_pos_time'] = 'Total POS Time (hrs)'
    if 'total_drive_time' in display_df.columns:
        column_mapping['total_drive_time'] = 'Total Drive Time (hrs)'
    if 'duration_std' in display_df.columns:
        column_mapping['duration_std'] = 'Duration Std Dev (hrs)'

    display_df = display_df.rename(columns=column_mapping)
    
    # Display with formatting
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Zone ID": st.column_config.TextColumn(width="small"),
            "Primaries": st.column_config.NumberColumn(width="small"),
            "Secondaries": st.column_config.NumberColumn(width="small"),
            "Weekly Duration (hrs)": st.column_config.NumberColumn(format="%.2f"),
            "Utilization (%)": st.column_config.NumberColumn(format="%.1f"),
            "Overutilized Days": st.column_config.NumberColumn(width="small"),
            "Underutilized Days": st.column_config.NumberColumn(width="small"),
            "Total POS Time (hrs)": st.column_config.NumberColumn(format="%.2f"),
            "Total Drive Time (hrs)": st.column_config.NumberColumn(format="%.2f"),
            "Duration Std Dev (hrs)": st.column_config.NumberColumn(format="%.2f", width="small")
        }
    )
    
    st.markdown("---")
    
    # Map
    st.subheader("All Zones Map")

    with st.spinner("creating map..."):
        map_fig = create_zones_map(filtered_itinerary_df)
    
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