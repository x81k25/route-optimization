"""
Zone Details Page for Route Optimization Dashboard.

Displays detailed metrics and daily itinerary for individual zones.
"""

# standard library imports
import sys
from pathlib import Path

# 3rd-party imports
import pandas as pd
import polars as pl
import streamlit as st

# add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

# local imports
from utils import (
    calculate_zone_metrics,
    create_zone_specific_map,
    filter_by_algorithm,
    get_latest_timestamp_data,
    get_unique_algorithms,
    load_daily_summary
)


def get_day_color(day) -> str:
    """Get the color for a specific day matching the map colors."""
    day_colors = [
        '#FF0040',  # Vibrant Red (Day 1)
        '#00FF80',  # Vibrant Green (Day 2)
        '#0080FF',  # Vibrant Blue (Day 3)
        '#FF8000',  # Vibrant Orange (Day 4)
        '#8000FF'   # Vibrant Purple (Day 5)
    ]
    # Convert to int to handle numpy float64 values
    day_int = int(day)
    return day_colors[(day_int - 1) % len(day_colors)]


def style_dataframe_by_day(
    df,
    day_column='day'
) -> object:
    """Apply day-based color styling to dataframe rows."""
    def apply_day_color(row):
        day = row[day_column]
        color = get_day_color(day)
        # Convert hex to RGB and add alpha transparency
        hex_color = color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Increase opacity to match map colors more closely (35% instead of 15%)
        background_color = f'rgba({r}, {g}, {b}, 0.35)'
        return [f'background-color: {background_color}' for _ in row]

    return df.style.apply(apply_day_color, axis=1)


def show_zone_details(itinerary_df) -> None:
    """
    Display zone details page with algorithm filtering.
    Shows 1:1 data from itinerary and daily-summary tables.

    :param itinerary_df: DataFrame with route optimization results
    :return: None
    """
    st.header("Zone Details")

    # get unique algorithm values for filters
    clusterers, balancers = get_unique_algorithms(itinerary_df)

    # initialize session state for first load
    if 'details_first_load' not in st.session_state:
        st.session_state.details_first_load = True

    # on first load, get latest timestamp data and defaults
    if st.session_state.details_first_load:
        latest_df, default_clusterer, default_balancer = get_latest_timestamp_data(itinerary_df)
        st.session_state.details_first_load = False
        # store defaults in session state
        if 'details_clusterer' not in st.session_state:
            st.session_state.details_clusterer = default_clusterer
        if 'details_balancer' not in st.session_state:
            st.session_state.details_balancer = default_balancer

    # filters in sidebar
    st.sidebar.subheader("Zone Details Filters")

    clusterer_filter = st.sidebar.selectbox(
        "Clusterer",
        options=clusterers,
        index=clusterers.index(st.session_state.details_clusterer) if st.session_state.details_clusterer in clusterers else 0,
        key="details_clusterer_select"
    )

    balancer_filter = st.sidebar.selectbox(
        "Balancer",
        options=balancers,
        index=balancers.index(st.session_state.details_balancer) if st.session_state.details_balancer in balancers else 0,
        key="details_balancer_select"
    )

    # update session state
    st.session_state.details_clusterer = clusterer_filter
    st.session_state.details_balancer = balancer_filter

    # filter data based on selections (ignores timestamp)
    filtered_itinerary_df = filter_by_algorithm(itinerary_df, clusterer_filter, balancer_filter)

    if filtered_itinerary_df is None or len(filtered_itinerary_df) == 0:
        st.warning(f"No itinerary data found for {clusterer_filter} + {balancer_filter}")
        return

    # show selected algorithm combination
    st.info(f"Showing results for: **{clusterer_filter}** + **{balancer_filter}**")

    # zone selector
    zones = sorted(filtered_itinerary_df['zone_id'].unique())
    selected_zone = st.selectbox("Select a zone:", zones, key="zone_selector")

    if not selected_zone:
        return

    # day filter - multiselect for the selected zone
    zone_itinerary_all = filtered_itinerary_df.filter(filtered_itinerary_df['zone_id'] == selected_zone)
    available_days = sorted(zone_itinerary_all['day'].unique())

    # Add custom CSS for colored day options
    day_colors_css = """
    <style>
    /* Style the multiselect selected tags (the items under the dropdown) */
    span[data-baseweb="tag"] {
        background-color: #4A5568 !important;
        color: white !important;
    }

    span[data-baseweb="tag"] > span {
        color: white !important;
    }

    /* Color reference badges above selector */
    .day-selector {
        margin-bottom: 0.2rem;
        margin-top: 0.7rem;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .day-option {
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-size: 1.1em;
    }
    .day-1 { background-color: #FF0040; }  /* Red */
    .day-2 { background-color: #00FF80; }  /* Green */
    .day-3 { background-color: #0080FF; }  /* Blue */
    .day-4 { background-color: #FF8000; }  /* Orange */
    .day-5 { background-color: #8000FF; }  /* Purple */

    /* Card styling for metrics section */
    .metrics-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """
    st.markdown(day_colors_css, unsafe_allow_html=True)

    selected_days = st.multiselect(
        "Select days to display:",
        options=available_days,
        default=available_days,  # All days selected by default
        key="day_filter",
        format_func=lambda x: f"Day {x}"
    )

    if not selected_days:
        st.warning("Please select at least one day to display.")
        return

    # Filter data for selected zone and days
    zone_itinerary = filtered_itinerary_df.filter(
        (filtered_itinerary_df['zone_id'] == selected_zone) &
        (filtered_itinerary_df['day'].is_in(selected_days))
    )

    # Load zone summary data
    zone_summary_df = calculate_zone_metrics(filtered_itinerary_df)
    if zone_summary_df is not None:
        # Filter zone summary by algorithm and selected zone
        # Handle null values in clusterer and balancer columns
        if 'clusterer' in zone_summary_df.columns and 'balancer' in zone_summary_df.columns:
            # Check if columns have null values
            if zone_summary_df['clusterer'].isna().all() and zone_summary_df['balancer'].isna().all():
                # If all values are null, just filter by zone_id
                zone_summary_filtered = zone_summary_df[
                    zone_summary_df['zone_id'] == selected_zone
                ]
            else:
                # Normal filtering with algorithm values
                zone_summary_filtered = zone_summary_df[
                    (zone_summary_df['clusterer'] == clusterer_filter) &
                    (zone_summary_df['balancer'] == balancer_filter) &
                    (zone_summary_df['zone_id'] == selected_zone)
                ]
        else:
            # If columns don't exist, just filter by zone_id
            zone_summary_filtered = zone_summary_df[
                zone_summary_df['zone_id'] == selected_zone
            ]
    else:
        zone_summary_filtered = None

    # Load daily summary data for the detailed table
    daily_summary_df = load_daily_summary()
    if daily_summary_df is not None:
        filtered_daily_summary = filter_by_algorithm(daily_summary_df, clusterer_filter, balancer_filter)
        zone_daily_summary = filtered_daily_summary.filter(
            (filtered_daily_summary['zone_id'] == selected_zone) &
            (filtered_daily_summary['day'].is_in(selected_days))
        ) if filtered_daily_summary is not None else None
    else:
        zone_daily_summary = None

    # Display zone summary using zone-summary data
    st.subheader(f"Zone {selected_zone} Summary")

    if zone_summary_filtered is not None and len(zone_summary_filtered) > 0:
        # Read summary metrics directly from zone-summary data
        zone_record = zone_summary_filtered.iloc[0]

        # Use expander for card-like effect
        with st.expander("Zone Summary Metrics", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                # Handle both naming conventions for primary count
                primary_col = 'primary_count' if 'primary_count' in zone_record else 'primary_pos_count'
                st.metric("Primary Locations", int(zone_record[primary_col]))
                st.metric("Weekly Duration", f"{zone_record['weekly_duration']:.1f} min")

            with col2:
                # Handle both naming conventions for secondary count
                secondary_col = 'secondary_count' if 'secondary_count' in zone_record else 'secondary_pos_count'
                st.metric("Secondary Locations", int(zone_record[secondary_col]))
                st.metric("Utilization", f"{zone_record['utilization']:.1f}%")

            with col3:
                st.metric("Total POS Time", f"{zone_record['total_pos_time']:.1f} min")
                st.metric("Total Drive Time", f"{zone_record['total_drive_time']:.1f} min")

            # Additional metrics row
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Overutilized Days", int(zone_record['overutilized_days']))
            with col5:
                st.metric("Underutilized Days", int(zone_record['underutilized_days']))
            with col6:
                st.metric("Duration Std Dev", f"{zone_record['duration_std']:.3f}")
    else:
        st.warning("No zone summary data available for this zone")
    
    # Zone-specific map
    day_label = "All Days" if len(selected_days) == len(available_days) else f"Days {min(selected_days)}-{max(selected_days)}"
    st.subheader(f"Zone {selected_zone} Map ({day_label})")

    # Create colored day labels for reference above the map (only for selected days, in order)
    day_color_display = "".join([
        f'<span class="day-option day-{day}">Day {day}</span>'
        for day in sorted(selected_days)
    ])
    st.markdown(f'<div class="day-selector">{day_color_display}</div>', unsafe_allow_html=True)

    # Filter itinerary data for map (by zone and selected days)
    map_itinerary_data = filtered_itinerary_df.filter(
        (filtered_itinerary_df['zone_id'] == selected_zone) &
        (filtered_itinerary_df['day'].is_in(selected_days))
    )

    with st.spinner("creating zone map..."):
        zone_map_fig = create_zone_specific_map(map_itinerary_data, selected_zone)
    
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
    
    # Display Daily Summary Table (1:1 with daily-summary data)
    if zone_daily_summary is not None and len(zone_daily_summary) > 0:
        st.subheader("Daily Summary Table")

        # Convert to pandas for better display
        daily_summary_pandas = zone_daily_summary.to_pandas()

        # Select only the relevant columns for display
        display_columns = [
            "day", "primary_locations", "secondary_locations",
            "duration", "utilization_percentage", "total_pos_time", "total_drive_time"
        ]
        daily_summary_display = daily_summary_pandas[display_columns]

        # Apply day-based color styling
        styled_daily_summary = style_dataframe_by_day(daily_summary_display)

        st.dataframe(
            styled_daily_summary,
            width='stretch',
            hide_index=True,
            column_config={
                "day": st.column_config.NumberColumn("Day", width=60),
                "primary_locations": st.column_config.NumberColumn("Primary", width=80),
                "secondary_locations": st.column_config.NumberColumn("Secondary", width=90),
                "duration": st.column_config.NumberColumn("Duration (min)", width=110, format="%.2f"),
                "utilization_percentage": st.column_config.NumberColumn("Utilization %", width=110, format="%.2f%%"),
                "total_pos_time": st.column_config.NumberColumn("POS Time (min)", width=120, format="%.2f"),
                "total_drive_time": st.column_config.NumberColumn("Drive Time (min)", width=130, format="%.2f")
            }
        )

    # Display Itinerary Table (1:1 with itinerary data)
    st.subheader("Route Itinerary Table")

    if zone_itinerary is not None and len(zone_itinerary) > 0:
        # Convert to pandas and add location names
        itinerary_data = []

        for row in zone_itinerary.iter_rows(named=True):
            pos_id = row['pos_id']

            # Get location name for display
            location_name = row.get('pos_name')

            # Format route coordinates for display (show first few points)
            route_str = ""
            if row['route']:
                try:
                    route_coords = row['route']
                    if isinstance(route_coords, (list, tuple)) and len(route_coords) > 0:
                        first_coord = route_coords[0]
                        if isinstance(first_coord, (list, tuple)) and len(first_coord) >= 2:
                            # Only show "..." if there are more than 1 coordinate pair
                            if len(route_coords) > 1:
                                route_str = f"[{first_coord[0]:.4f}, {first_coord[1]:.4f}]..."
                            else:
                                route_str = f"[{first_coord[0]:.4f}, {first_coord[1]:.4f}]"
                        else:
                            route_str = str(route_coords)[:50] + "..."
                except:
                    route_str = "[route data]"

            itinerary_data.append({
                'day': row['day'],
                'pos_id': pos_id,
                'location_name': location_name,
                'pos_class': row['pos_class'],
                'route_preview': route_str,
                'action': row['action'],
                'schedule': row['schedule'],
                'duration': row['duration']
            })

        itinerary_df_display = pd.DataFrame(itinerary_data)

        # Apply day-based color styling
        styled_itinerary = style_dataframe_by_day(itinerary_df_display)

        st.dataframe(
            styled_itinerary,
            width='stretch',
            hide_index=True,
            column_config={
                "day": st.column_config.NumberColumn("Day", width=60),
                "pos_id": st.column_config.TextColumn("POS ID", width=80),
                "location_name": st.column_config.TextColumn("POS Name", width=150),
                "pos_class": st.column_config.TextColumn("Class", width=80),
                "route_preview": st.column_config.TextColumn("Coordinates", width=140),
                "action": st.column_config.TextColumn("Action", width=80),
                "schedule": st.column_config.NumberColumn("Schedule (min)", width=100, format="%.2f"),
                "duration": st.column_config.NumberColumn("Duration (min)", width=100, format="%.2f")
            }
        )
    else:
        st.info("No itinerary data available for this zone")
    
