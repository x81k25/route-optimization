"""
Shared utilities for the Streamlit Route Optimization Dashboard.

Contains data loading functions and map generation utilities used by multiple pages.
"""

import streamlit as st
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import json
import yaml
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_data():
    """Load itinerary and aggregate data."""
    try:
        # Load itinerary data
        itinerary_path = Path(__file__).parent.parent / "output" / "itinerary.jsonl"
        if not itinerary_path.exists():
            st.error(f"Itinerary file not found: {itinerary_path}")
            return None, None
            
        itinerary_data = []
        with open(itinerary_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    itinerary_data.append(json.loads(line))
        
        itinerary_df = pl.DataFrame(itinerary_data)
        
        # Load locations data
        locations_path = Path(__file__).parent.parent / "data" / "locations.jsonl"
        locations = {}
        if locations_path.exists():
            with open(locations_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        location = json.loads(line)
                        locations[location['pos_id']] = location
        
        return itinerary_df, locations
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def load_aggregate_metrics():
    """Load pre-calculated aggregate metrics from aggregate_summary.jsonl."""
    try:
        # Load aggregate summary
        summary_path = Path(__file__).parent.parent / "output" / "aggregate_summary.jsonl"
        if not summary_path.exists():
            st.error(f"Aggregate summary file not found: {summary_path}")
            return None
        
        # Read the summary stats (first line of the file)
        with open(summary_path, 'r') as f:
            summary_stats = json.loads(f.readline().strip())
        
        return summary_stats
        
    except Exception as e:
        st.error(f"Error loading aggregate summary: {e}")
        return None

def calculate_zone_metrics(itinerary_df):
    """Load zone-level metrics from aggregate-report.jsonl file."""
    try:
        # Load aggregate report directly from file
        aggregate_path = Path(__file__).parent.parent / "output" / "aggregate-report.jsonl"
        if not aggregate_path.exists():
            st.error(f"Aggregate report file not found: {aggregate_path}")
            return None
        
        # Read the aggregate report
        aggregate_data = []
        with open(aggregate_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    aggregate_data.append(json.loads(line))
        
        if not aggregate_data:
            st.error("No data found in aggregate report file")
            return None
        
        return pd.DataFrame(aggregate_data).sort_values('zone_id')
        
    except Exception as e:
        st.error(f"Error loading aggregate report: {e}")
        return None

def create_zones_map(itinerary_df, locations):
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
        
        for row in zone_data.iter_rows(named=True):
            pos_ids = row['pos_id'] or []
            pos_locations = row['pos_locations'] or []
            pos_classes = row['pos_class'] or []
            
            for pos_id, pos_location_str, pos_class in zip(pos_ids, pos_locations, pos_classes):
                if pos_id not in seen_locations:
                    seen_locations.add(pos_id)
                    
                    # Parse coordinates from string format
                    try:
                        # Remove brackets and split
                        coords_str = pos_location_str.strip('[]')
                        lon, lat = map(float, coords_str.split())
                        
                        all_lats.append(lat)
                        all_lons.append(lon)
                        
                        # Get location name
                        name = locations.get(pos_id, {}).get('name', f"Location {pos_id}")
                        hover_text = f"{zone_id} - {name} ({'PRIMARY' if pos_class == 'primary' else 'SECONDARY'})"
                        
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

def get_gradient_opacity(stop_index, total_stops, min_opacity=0.3, max_opacity=1.0):
    """Generate an opacity value based on stop position for gradient effect."""
    if total_stops <= 1:
        return max_opacity
    
    # Linear interpolation from min to max opacity
    progress = stop_index / (total_stops - 1)
    return min_opacity + (max_opacity - min_opacity) * progress

def create_zone_specific_map(itinerary_df, locations, zone_id):
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
    
    # Process each day's route
    for row in zone_data.iter_rows(named=True):
        day = row['day']
        pos_ids = row['pos_id'] or []
        pos_locations = row['pos_locations'] or []
        pos_classes = row['pos_class'] or []
        route = row['route'] or []
        
        if not pos_locations:
            continue
            
        # Get base color for this day
        base_color = day_colors[(day - 1) % len(day_colors)]
        num_stops = len(pos_locations)
        
        # Parse route geometry (detailed path from OSRM)
        route_lats = []
        route_lons = []
        
        if route:
            # Route contains the actual driving path with many waypoints
            for point_str in route:
                try:
                    # Parse route point coordinates
                    coords_str = point_str.strip('[]') if isinstance(point_str, str) else str(point_str).strip('[]')
                    parts = coords_str.split()
                    if len(parts) >= 2:
                        lon, lat = map(float, parts[:2])
                        route_lats.append(lat)
                        route_lons.append(lon)
                        all_lats.append(lat)
                        all_lons.append(lon)
                except:
                    continue
        
        # Parse stop locations
        stop_coords = []
        for pos_location_str in pos_locations:
            try:
                coords_str = pos_location_str.strip('[]')
                lon, lat = map(float, coords_str.split())
                stop_coords.append((lat, lon))
                if not route:  # Only add to all_lats/lons if we don't have route data
                    all_lats.append(lat)
                    all_lons.append(lon)
            except:
                continue
        
        if num_stops <= 1:
            # Single stop - use base color with standard opacity
            if stop_coords and pos_ids:
                lat, lon = stop_coords[0]
                location_info = locations.get(pos_ids[0], {})
                name = location_info.get('name', f'Location {pos_ids[0]}')
                marker_size = 24 if pos_classes[0] == 'primary' else 20
                
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=base_color,
                        symbol='circle'
                    ),
                    opacity=0.7,
                    hovertemplate=f'<b>{name}</b><br>Day: {day}<br>Stop: 1<br>ID: {pos_ids[0]}<br>Type: {pos_classes[0].title()}<extra></extra>',
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
            
            # Add gradient markers for each stop
            for i, (pos_id, pos_class) in enumerate(zip(pos_ids, pos_classes)):
                if i < len(stop_coords):
                    lat, lon = stop_coords[i]
                    
                    # Get location details
                    location_info = locations.get(pos_id, {})
                    name = location_info.get('name', f'Location {pos_id}')
                    
                    # Calculate gradient opacity
                    if i == 0:
                        stop_opacity = 0.2  # First stop gets low opacity
                    else:
                        stop_opacity = get_gradient_opacity(i, num_stops)
                    
                    # Determine marker style
                    marker_size = 24 if pos_class == 'primary' else 20
                    
                    # Show legend only for first marker of this day
                    show_in_legend = (i == 0)
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[lat],
                        lon=[lon],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=base_color,
                            symbol='circle'
                        ),
                        opacity=stop_opacity,
                        hovertemplate=f'<b>{name}</b><br>Day: {day}<br>Stop: {i+1}<br>ID: {pos_id}<br>Type: {pos_class.title()}<extra></extra>',
                        name=f'Day {day}' if show_in_legend else "",
                        showlegend=show_in_legend,
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
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#333333"),
            bordercolor="#cccccc",
            borderwidth=2
        )
    )
    
    return fig


def generate_detailed_itinerary_table(zone_data, locations):
    """Generate detailed itinerary table with separate rows for stops and travel segments."""
    table_data = []
    
    for row in zone_data.iter_rows(named=True):
        day = row['day']
        pos_ids = row['pos_id'] or []
        pos_locations = row['pos_locations'] or []
        pos_classes = row['pos_class'] or []
        pos_durations = row['pos_duration'] or []
        schedule = row['schedule'] or []
        duration = row['duration'] or 0.0
        
        # If no locations for this day, add a summary row
        if not pos_ids:
            table_data.append({
                'day': f"Day {day}",
                'type': 'summary',
                'stop_num': '',
                'time': '',
                'location': 'No locations scheduled',
                'activity': '',
                'duration': f"{duration / 60.0:.1f} hrs"
            })
            continue
        
        # Add rows for each stop and travel segment
        for i, pos_id in enumerate(pos_ids):
            # Get store information
            store_name = locations.get(pos_id, {}).get('name', f"Store {pos_id}")
            pos_class = pos_classes[i] if i < len(pos_classes) else 'unknown'
            pos_duration = pos_durations[i] if i < len(pos_durations) else 0
            
            # Get arrival and departure times
            if i * 2 < len(schedule):
                arrival_minutes = schedule[i * 2]
                arrival_hours = int(arrival_minutes // 60)
                arrival_mins = int(arrival_minutes % 60)
                arrival_time = f"{arrival_hours:02d}:{arrival_mins:02d}"
            else:
                arrival_time = "--:--"
            
            if i * 2 + 1 < len(schedule):
                departure_minutes = schedule[i * 2 + 1]
                departure_hours = int(departure_minutes // 60)
                departure_mins = int(departure_minutes % 60)
                departure_time = f"{departure_hours:02d}:{departure_mins:02d}"
            else:
                departure_time = "--:--"
            
            # Add travel segment row (except for first location)
            if i > 0 and (i * 2 - 1) < len(schedule) and (i * 2) < len(schedule):
                previous_departure = schedule[i * 2 - 1]
                current_arrival = schedule[i * 2]
                drive_minutes = current_arrival - previous_departure
                
                if drive_minutes > 0:
                    prev_dep_hours = int(previous_departure // 60)
                    prev_dep_mins = int(previous_departure % 60)
                    prev_departure_time = f"{prev_dep_hours:02d}:{prev_dep_mins:02d}"
                    
                    drive_hours = int(drive_minutes // 60)
                    drive_mins_remainder = int(drive_minutes % 60)
                    
                    if drive_hours > 0:
                        drive_duration = f"{drive_hours}h {drive_mins_remainder}m"
                    else:
                        drive_duration = f"{drive_mins_remainder}m"
                    
                    table_data.append({
                        'day': f"Day {day}",
                        'type': 'travel',
                        'stop_num': f"→ {i+1}",
                        'time': f"{prev_departure_time} - {arrival_time}",
                        'location': f"Drive to {store_name}",
                        'activity': 'driving',
                        'duration': drive_duration
                    })
            
            # Add stop row
            service_hours = int(pos_duration // 60)
            service_mins = int(pos_duration % 60)
            if service_hours > 0:
                service_duration = f"{service_hours}h {service_mins}m"
            else:
                service_duration = f"{service_mins}m"
            
            table_data.append({
                'day': f"Day {day}",
                'type': 'stop',
                'stop_num': str(i + 1),
                'time': f"{arrival_time} - {departure_time}" if departure_time != "--:--" else arrival_time,
                'location': store_name,
                'activity': pos_class.upper() if pos_class else 'LOCATION',
                'duration': service_duration
            })
    
    return table_data