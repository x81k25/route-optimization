"""
Shared utilities for the Streamlit Route Optimization Dashboard.

Contains data loading functions and map generation utilities used by multiple pages.
"""

# standard library imports
import json
import sys
from pathlib import Path

# 3rd-party imports
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
import yaml

# add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_data() -> tuple:
    """
    Load itinerary data from Parquet files (primary) with JSONL fallback.

    :return: itinerary_df or None if failed
    """
    try:
        # load itinerary data - prefer Parquet
        itinerary_parquet = Path(__file__).parent.parent / "output" / "itinerary.parquet"
        itinerary_jsonl = Path(__file__).parent.parent / "output" / "itinerary.jsonl"

        itinerary_df = None

        if itinerary_parquet.exists():
            try:
                itinerary_df = pl.read_parquet(itinerary_parquet)
            except Exception as e:
                st.warning(f"Could not read Parquet file, trying JSONL: {e}")

        if itinerary_df is None and itinerary_jsonl.exists():
            itinerary_data = []
            with open(itinerary_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        itinerary_data.append(json.loads(line))
            itinerary_df = pl.DataFrame(itinerary_data)

        if itinerary_df is None:
            st.error("No itinerary data found in Parquet or JSONL format")
            return None

        return itinerary_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_aggregate_metrics() -> tuple:
    """Load aggregate summary data from Parquet (primary) with JSONL fallback."""
    try:
        # prefer Parquet format
        summary_parquet = Path(__file__).parent.parent / "output" / "aggregate-summary.parquet"
        summary_jsonl = Path(__file__).parent.parent / "output" / "aggregate-summary.jsonl"

        summary_df = None

        if summary_parquet.exists():
            try:
                summary_df = pl.read_parquet(summary_parquet)
                return summary_df
            except Exception as e:
                st.warning(f"Could not read summary Parquet file, trying JSONL: {e}")

        if summary_jsonl.exists():
            # fallback to JSONL
            summary_data = []
            with open(summary_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        summary_data.append(json.loads(line))
            if summary_data:
                return pl.DataFrame(summary_data)

        st.error("No aggregate summary data found in Parquet or JSONL format")
        return None

    except Exception as e:
        st.error(f"Error loading aggregate summary: {e}")
        return None

def calculate_zone_metrics(itinerary_df) -> tuple:
    """Load zone-level metrics from Parquet (primary) with JSONL fallback."""
    try:
        # prefer Parquet format
        aggregate_parquet = Path(__file__).parent.parent / "output" / "zone-summary.parquet"
        aggregate_jsonl = Path(__file__).parent.parent / "output" / "zone-summary.jsonl"

        aggregate_df = None

        if aggregate_parquet.exists():
            try:
                aggregate_df = pl.read_parquet(aggregate_parquet)
                return aggregate_df.to_pandas().sort_values('zone_id')
            except Exception as e:
                st.warning(f"Could not read aggregate Parquet file, trying JSONL: {e}")

        if aggregate_jsonl.exists():
            # fallback to JSONL
            aggregate_data = []
            with open(aggregate_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        aggregate_data.append(json.loads(line))

            if aggregate_data:
                return pd.DataFrame(aggregate_data).sort_values('zone_id')

        st.error("No zone summary data found in Parquet or JSONL format")
        return None

    except Exception as e:
        st.error(f"Error loading zone summary: {e}")
        return None

def load_daily_summary() -> tuple:
    """Load daily-summary data from Parquet (primary) with JSONL fallback."""
    try:
        # prefer Parquet format
        daily_parquet = Path(__file__).parent.parent / "output" / "daily-summary.parquet"
        daily_jsonl = Path(__file__).parent.parent / "output" / "daily-summary.jsonl"

        daily_df = None

        if daily_parquet.exists():
            try:
                daily_df = pl.read_parquet(daily_parquet)
                return daily_df
            except Exception as e:
                st.warning(f"Could not read daily-summary Parquet file, trying JSONL: {e}")

        if daily_jsonl.exists():
            # fallback to JSONL
            daily_data = []
            with open(daily_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        daily_data.append(json.loads(line))
            if daily_data:
                return pl.DataFrame(daily_data)

        st.error("No daily summary data found in Parquet or JSONL format")
        return None

    except Exception as e:
        st.error(f"Error loading daily summary: {e}")
        return None

@st.cache_data
def create_zones_map(itinerary_df) -> object:
    """Create a map showing all zone locations."""
    if itinerary_df is None:
        return None
        
    # Color palette for different zones
    zone_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C0C8', '#D7DBDD', '#FADBD8',
        '#D5DBDB', '#EBF5FB', '#E8F8F5', '#FDF2E9', '#F4F6F6', '#FDEDEC', '#EAF2F8', '#E9F7EF'
    ]
    
    fig = go.Figure()
    
    # Get unique zones and assign colors
    zones = sorted(itinerary_df['zone_id'].unique())
    zone_color_map = {}
    for i, zone_id in enumerate(zones):
        zone_color_map[zone_id] = zone_colors[i % len(zone_colors)]
    
    # Collect all coordinates for centering
    all_lats = []
    all_lons = []
    
    # Process each zone
    for zone_id in zones:
        zone_data = itinerary_df.filter(pl.col('zone_id') == zone_id)
        zone_color = zone_color_map[zone_id]
        
        # Track unique locations to avoid duplicates
        seen_locations = set()
        primary_locations = {'lat': [], 'lon': [], 'text': []}
        secondary_locations = {'lat': [], 'lon': [], 'text': []}
        
        # Get unique positions for this zone across all days and actions
        zone_positions = zone_data.filter(
            (pl.col('pos_id').is_not_null()) & (pl.col('action') == 'arriving')
        ).unique(subset=['pos_id'])

        for row in zone_positions.iter_rows(named=True):
            pos_id = row['pos_id']
            pos_class = row['pos_class']
            route_coords = row['route']

            if pos_id and pos_id not in seen_locations and route_coords:
                seen_locations.add(pos_id)

                # Extract coordinates from route (first coordinate)
                try:
                    if isinstance(route_coords, list) and len(route_coords) > 0:
                        coord = route_coords[0]
                        if isinstance(coord, list) and len(coord) >= 2:
                            lon, lat = float(coord[0]), float(coord[1])
                        else:
                            continue
                    else:
                        continue

                    all_lats.append(lat)
                    all_lons.append(lon)

                    # Get location name from pos_name field or fallback
                    name = row.get('pos_name', f"Location {pos_id}")
                    hover_text = f"{zone_id} - {name} ({'primary' if pos_class == 'primary' else 'secondary'})"

                    if pos_class == 'primary':
                        primary_locations['lat'].append(lat)
                        primary_locations['lon'].append(lon)
                        primary_locations['text'].append(hover_text)
                    else:
                        secondary_locations['lat'].append(lat)
                        secondary_locations['lon'].append(lon)
                        secondary_locations['text'].append(hover_text)
                except Exception as e:
                    continue
        
        # Add primary locations as star markers
        if primary_locations['lat']:
            fig.add_trace(go.Scattermapbox(
                lat=primary_locations['lat'],
                lon=primary_locations['lon'],
                mode='markers',
                marker=dict(size=16, color=zone_color, symbol='star'),
                name=f'{zone_id} Primary',
                text=primary_locations['text'],
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ))
        
        # Add secondary locations as circle markers
        if secondary_locations['lat']:
            fig.add_trace(go.Scattermapbox(
                lat=secondary_locations['lat'],
                lon=secondary_locations['lon'],
                mode='markers',
                marker=dict(size=8, color=zone_color),
                name=f'{zone_id} Secondary',
                text=secondary_locations['text'],
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ))
    
    # Calculate center and zoom
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        max_range = max(lat_range, lon_range)
        
        if max_range > 15:      zoom = 4
        elif max_range > 8:     zoom = 5
        elif max_range > 4:     zoom = 6
        elif max_range > 2:     zoom = 7
        elif max_range > 1:     zoom = 8
        elif max_range > 0.5:   zoom = 9
        else:                   zoom = 10
    else:
        center_lat, center_lon, zoom = 37.7749, -122.4194, 11
    
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

def get_gradient_opacity(
    stop_index,
    total_stops,
    min_opacity=0.3,
    max_opacity=1.0
):
    """Generate an opacity value based on stop position for gradient effect."""
    if total_stops <= 1:
        return max_opacity
    
    # Linear interpolation from min to max opacity
    progress = stop_index / (total_stops - 1)
    return min_opacity + (max_opacity - min_opacity) * progress

def create_zone_specific_map(
    itinerary_df,
    zone_id
):
    """Create a map showing detailed routes for a specific zone with gradient colors."""
    if itinerary_df is None:
        return None
    
    # Filter data for the specific zone
    zone_data = itinerary_df.filter(pl.col('zone_id') == zone_id)
    
    if len(zone_data) == 0:
        return None
    
    fig = go.Figure()
    
    # Define vibrant base colors for different days (matching HTML version)
    day_colors = [
        '#FF0040',  # Vibrant Red
        '#00FF80',  # Vibrant Green
        '#0080FF',  # Vibrant Blue
        '#FF8000',  # Vibrant Orange
        '#8000FF'   # Vibrant Purple
    ]
    
    all_lats = []
    all_lons = []
    
    # Process each day's route - group by day and collect all positions
    days = zone_data.select('day').unique().sort('day').to_series().to_list()

    for day in days:
        day_data = zone_data.filter(pl.col('day') == day)

        # Get all positions for this day (including centroid and regular stops)
        day_positions = day_data.filter(
            ((pl.col('pos_id').is_not_null()) & (pl.col('action') == 'arriving')) |
            ((pl.col('pos_id').is_null()) & (pl.col('pos_class') == 'centroid'))
        ).sort('schedule')

        if len(day_positions) == 0:
            continue

        # Get base color for this day
        base_color = day_colors[(day - 1) % len(day_colors)]
        num_stops = len(day_positions)

        # Collect route geometry from driving actions
        driving_data = day_data.filter(pl.col('action') == 'driving')
        route_coords = []
        for driving_row in driving_data.iter_rows(named=True):
            route = driving_row['route'] or []
            for coord in route:
                if isinstance(coord, list) and len(coord) >= 2:
                    route_coords.append((float(coord[1]), float(coord[0])))  # lat, lon

        if route_coords:
            route_lats, route_lons = zip(*route_coords)
            all_lats.extend(route_lats)
            all_lons.extend(route_lons)
        else:
            route_lats, route_lons = [], []

        # Collect stop locations
        stop_coords = []
        pos_ids = []
        pos_classes = []
        pos_names = []

        for pos_row in day_positions.iter_rows(named=True):
            pos_id = pos_row['pos_id']
            pos_class = pos_row['pos_class']
            pos_name = pos_row.get('pos_name', None)
            route = pos_row['route'] or []

            if route and isinstance(route, list) and len(route) > 0:
                coord = route[0]
                if isinstance(coord, list) and len(coord) >= 2:
                    lat, lon = float(coord[1]), float(coord[0])
                    stop_coords.append((lat, lon))
                    pos_ids.append(pos_id)
                    pos_classes.append(pos_class)
                    pos_names.append(pos_name)

                    if not route_coords:  # Only add if we don't have detailed route
                        all_lats.append(lat)
                        all_lons.append(lon)
        
        
        if num_stops <= 1:
            # Single stop - use base color with standard opacity
            if stop_coords and pos_ids:
                lat, lon = stop_coords[0]
                pos_id = pos_ids[0]
                pos_class = pos_classes[0]
                pos_name = pos_names[0] if pos_names else None

                if pos_id is None:
                    name = "Zone Centroid"
                    marker_size = 30
                    marker_symbol = 'star'
                else:
                    name = pos_name or f'Location {pos_id}'
                    marker_size = 24 if pos_class == 'primary' else 20
                    marker_symbol = 'circle'

                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=base_color,
                        symbol=marker_symbol
                    ),
                    opacity=1.0,  # Full opacity for single stops (and legend)
                    hovertemplate=f'<b>{name}</b><br>Day: {day}<br>Stop: 1<br>ID: {pos_id or "Centroid"}<br>Type: {pos_class.title()}<extra></extra>',
                    name=f'Day {day}',
                    showlegend=True,
                    legendgroup=f'day_{day}'
                ))
        else:
            # Multiple stops - create gradient segments and markers
            
            # Draw route segments with gradient if we have detailed route data
            if route_lats and route_lons and len(stop_coords) > 1:
                # Find the closest route point to each stop location
                stop_indices = []
                for stop_lat, stop_lon in stop_coords:
                    min_dist = float('inf')
                    closest_idx = 0
                    
                    for idx, (route_lat, route_lon) in enumerate(zip(route_lats, route_lons)):
                        dist = ((route_lat - stop_lat) ** 2 + (route_lon - stop_lon) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    stop_indices.append(closest_idx)
                
                # Sort indices to ensure they're in route order
                stop_indices = sorted(stop_indices)
                
                # Create gradient segments between consecutive stops
                for seg_idx in range(len(stop_indices) - 1):
                    segment_opacity = get_gradient_opacity(seg_idx, num_stops)
                    
                    start_idx = stop_indices[seg_idx]
                    end_idx = stop_indices[seg_idx + 1] + 1
                    
                    if start_idx < len(route_lats) and end_idx <= len(route_lats) and start_idx < end_idx:
                        segment_lats = route_lats[start_idx:end_idx]
                        segment_lons = route_lons[start_idx:end_idx]
                        
                        if segment_lats and segment_lons:
                            fig.add_trace(go.Scattermapbox(
                                lat=segment_lats,
                                lon=segment_lons,
                                mode='lines',
                                line=dict(
                                    width=3,
                                    color=base_color
                                ),
                                opacity=segment_opacity,
                                hovertemplate=f'<b>Day {day} Route</b><br>Segment: Stop {seg_idx + 1} → Stop {seg_idx + 2}<br>Zone: {zone_id}<extra></extra>',
                                showlegend=False,
                                legendgroup=f'day_{day}'
                            ))
            elif len(stop_coords) > 1:
                # No detailed route - fallback to straight lines between stops
                for i in range(len(stop_coords) - 1):
                    segment_opacity = get_gradient_opacity(i, num_stops)
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[stop_coords[i][0], stop_coords[i+1][0]],
                        lon=[stop_coords[i][1], stop_coords[i+1][1]],
                        mode='lines',
                        line=dict(width=3, color=base_color),
                        opacity=segment_opacity,
                        showlegend=False,
                        legendgroup=f'day_{day}'
                    ))
            
            # Note: Legend removed - using colored day selector instead

            # Add gradient markers for each stop
            for i, (pos_id, pos_class, pos_name) in enumerate(zip(pos_ids, pos_classes, pos_names)):
                if i < len(stop_coords):
                    lat, lon = stop_coords[i]

                    # Get location details
                    if pos_id is None:
                        name = "Zone Centroid"
                        marker_size = 30  # Larger size for centroid
                        marker_symbol = 'star'
                    else:
                        name = pos_name or f'Location {pos_id}'
                        marker_size = 24 if pos_class == 'primary' else 20
                        marker_symbol = 'circle'

                    # Calculate gradient opacity
                    if i == 0:
                        stop_opacity = 0.2  # First stop gets low opacity
                    else:
                        stop_opacity = get_gradient_opacity(i, num_stops)

                    # Create marker
                    marker_dict = dict(
                        size=marker_size,
                        color=base_color,
                        symbol=marker_symbol
                    )

                    fig.add_trace(go.Scattermapbox(
                        lat=[lat],
                        lon=[lon],
                        mode='markers',
                        marker=marker_dict,
                        opacity=stop_opacity,
                        hovertemplate=f'<b>{name}</b><br>Day: {day}<br>Stop: {i+1}<br>ID: {pos_id or "Centroid"}<br>Type: {pos_class.title()}<extra></extra>',
                        name="",  # No name for individual markers
                        showlegend=False,  # Don't show in legend
                        legendgroup=f'day_{day}'
                    ))
    
    # Calculate center and zoom based on all coordinates
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Calculate zoom with better granularity
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        max_range = max(lat_range, lon_range)
        
        # More precise zoom levels matching HTML version
        if max_range > 5:       zoom = 4
        elif max_range > 1:     zoom = 6
        elif max_range > 0.5:   zoom = 8
        elif max_range > 0.1:   zoom = 10
        else:                   zoom = 12
    else:
        center_lat, center_lon, zoom = 37.7749, -122.4194, 11
    
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=800,  # Increased height to match HTML version
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#333333"),
            bordercolor="#cccccc",
            borderwidth=2,
            itemsizing="constant",  # Keep legend marker size constant
            itemwidth=30  # Set legend marker width
        )
    )
    
    return fig


def generate_detailed_itinerary_table(zone_data) -> object:
    """Generate detailed itinerary table with separate rows for stops and travel segments."""
    table_data = []

    # Get unique days and process each one
    days = zone_data.select('day').unique().sort('day').to_series().to_list()

    for day in days:
        day_data = zone_data.filter(pl.col('day') == day)

        # Get all actions for this day, sorted by schedule
        day_actions = day_data.sort('schedule')

        if len(day_actions) == 0:
            table_data.append({
                'day': f"Day {day}",
                'type': 'summary',
                'stop_num': '',
                'time': '',
                'location': 'No locations scheduled',
                'activity': '',
                'duration': '0 hrs'
            })
            continue

        stop_num = 0
        for row in day_actions.iter_rows(named=True):
            action = row['action']
            pos_id = row['pos_id']
            pos_class = row['pos_class']
            schedule_time = row['schedule']

            # Convert schedule to hours:minutes
            hours = int(schedule_time // 60)
            mins = int(schedule_time % 60)
            time_str = f"{hours:02d}:{mins:02d}"

            if action == 'driving':
                if pos_id is None:  # Starting from centroid
                    table_data.append({
                        'day': f"Day {day}",
                        'type': 'travel',
                        'stop_num': 'start',
                        'time': time_str,
                        'location': 'Departing from zone centroid',
                        'activity': 'driving',
                        'duration': '--'
                    })
                else:
                    # This would be driving between locations, but we handle that in departing
                    pass

            elif action == 'arriving':
                stop_num += 1
                pos_name = row.get('pos_name')
                store_name = pos_name or f"Store {pos_id}"
                table_data.append({
                    'day': f"Day {day}",
                    'type': 'stop',
                    'stop_num': str(stop_num),
                    'time': time_str,
                    'location': f"Arrive at {store_name}",
                    'activity': pos_class if pos_class else 'location',
                    'duration': '--'
                })

            elif action == 'departing':
                pos_name = row.get('pos_name')
                store_name = pos_name or f"Store {pos_id}"
                table_data.append({
                    'day': f"Day {day}",
                    'type': 'travel',
                    'stop_num': f"← {stop_num}",
                    'time': time_str,
                    'location': f"Depart from {store_name}",
                    'activity': 'driving',
                    'duration': '--'
                })

    return table_data


def get_latest_timestamp_data(df) -> object:
    """
    Get data with the latest timestamp and return default clusterer/balancer values.
    Used only on page load.

    :param df: Polars DataFrame with created_on column
    :return: tuple of (filtered_df, default_clusterer, default_balancer)
    """
    if df is None or len(df) == 0:
        return None, "mds_kmeans", "greedy"

    # convert created_on to datetime for proper sorting
    df_with_datetime = df.with_columns(
        pl.col('created_on').str.to_datetime('%Y-%m-%d %H:%M:%S').alias('created_on_dt')
    )

    # get the latest timestamp
    latest_timestamp = df_with_datetime.select(pl.col('created_on_dt').max()).item()

    # filter to only records with latest timestamp
    latest_df = df_with_datetime.filter(pl.col('created_on_dt') == latest_timestamp)

    # get default clusterer and balancer (first record from latest data)
    first_record = latest_df.row(0, named=True)
    default_clusterer = first_record.get('clusterer', 'mds_kmeans')
    default_balancer = first_record.get('balancer', 'greedy')

    # return original df without datetime column for compatibility
    latest_df_clean = latest_df.drop('created_on_dt')

    return latest_df_clean, default_clusterer, default_balancer


def filter_by_algorithm(
    df,
    clusterer=None,
    balancer=None
):
    """
    Filter DataFrame by clusterer and balancer.
    Used for subsequent filter selections (ignores timestamp).

    :param df: Polars DataFrame
    :param clusterer: clusterer to filter by (None = no filter)
    :param balancer: balancer to filter by (None = no filter)
    :return: filtered Polars DataFrame
    """
    if df is None:
        return None

    filtered_df = df

    if clusterer and 'clusterer' in df.columns:
        filtered_df = filtered_df.filter(pl.col('clusterer') == clusterer)

    if balancer and 'balancer' in df.columns:
        filtered_df = filtered_df.filter(pl.col('balancer') == balancer)

    return filtered_df


def get_unique_algorithms(df) -> tuple:
    """
    Get unique clusterer and balancer values from DataFrame.

    :param df: Polars DataFrame
    :return: tuple of (clusterers_list, balancers_list)
    """
    if df is None or len(df) == 0:
        # Return default algorithm options when no data
        return ['mds_kmeans'], ['greedy']

    clusterers = []
    balancers = []

    if 'clusterer' in df.columns:
        clusterer_values = df.select(pl.col('clusterer').unique()).to_series().to_list()
        # Filter out None/null values and sort, but keep "none" strings
        clusterers = sorted([c for c in clusterer_values if c is not None])

    if 'balancer' in df.columns:
        balancer_values = df.select(pl.col('balancer').unique()).to_series().to_list()
        # Filter out None/null values and sort, but keep "none" strings
        balancers = sorted([b for b in balancer_values if b is not None])

    # Ensure we always have at least one option
    if not clusterers:
        clusterers = ['none']  # Use 'none' as default to match data
    if not balancers:
        balancers = ['none']   # Use 'none' as default to match data

    return clusterers, balancers