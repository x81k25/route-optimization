"""
Route optimization visualization utilities using Plotly.
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path
import polyline
from loguru import logger


class RouteVisualizer:
    """Visualizes optimized routes on a map using Plotly."""
    
    def __init__(self, locations_path: str = "data/subway_locations.jsonl"):
        """
        Initialize visualizer with location data.
        
        Args:
            locations_path: Path to locations JSON or JSONL file
        """
        if locations_path.endswith('.jsonl'):
            # Handle JSONL format - one JSON object per line
            locations_data = []
            with open(locations_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        locations_data.append(json.loads(line))
        else:
            # Handle traditional JSON format
            with open(locations_path, 'r') as f:
                data = json.load(f)
            
            # Handle both old and new dataset formats
            if 'subway_locations_california' in data:
                locations_data = data['subway_locations_california'] 
            elif 'subway_locations_san_francisco' in data:
                locations_data = data['subway_locations_san_francisco']
            else:
                raise ValueError("Could not find locations in dataset")
            
        self.locations = {
            loc['id']: loc 
            for loc in locations_data
        }
        
        # Color palette for different days
        self.day_colors = {
            1: '#FF6B6B',  # Red for primary
            2: '#4ECDC4',  # Teal
            3: '#45B7D1',  # Blue
            4: '#96CEB4',  # Green
            5: '#FFEAA7',  # Yellow
            6: '#DDA0DD',  # Plum
            7: '#98D8C8'   # Mint
        }
    
    def load_optimization_results(self, results_path: str) -> Dict[str, Any]:
        """Load optimization results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def create_overview_map(self, results: Dict[str, Any]) -> go.Figure:
        """Create overview map showing all routes."""
        fig = go.Figure()
        
        # Add map background
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',  # Clean, minimalist style with light colors
                center=dict(lat=37.7749, lon=-122.4194),  # San Francisco center
                zoom=11
            ),
            title="Route Optimization - All Days Overview",
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Plot primary locations
        for day, primary_id in results['primary_assignments'].items():
            location = self.locations[primary_id]
            fig.add_trace(go.Scattermapbox(
                lat=[location['latitude']],
                lon=[location['longitude']],
                mode='markers',
                marker=dict(
                    size=25,
                    color=self.day_colors[int(day)],
                    symbol='circle'
                ),
                text=f"Day {day}: {location['name']} (PRIMARY)",
                name=f"Day {day} - Primary",
                hovertemplate="<b>%{text}</b><br>" +
                            "Address: " + location['address'] + 
                            "<extra></extra>"
            ))
        
        # Plot secondary routes
        for day, location_ids in results['secondary_assignments'].items():
            day_int = int(day)
            route = results['daily_routes'][day]
            drive_time = results['daily_drive_times'][day]
            
            # Plot locations
            lats = [self.locations[loc_id]['latitude'] for loc_id in route]
            lons = [self.locations[loc_id]['longitude'] for loc_id in route]
            names = [self.locations[loc_id]['name'] for loc_id in route]
            
            # Add location markers
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.day_colors[day_int],
                ),
                text=[f"Day {day}: {name}" for name in names],
                name=f"Day {day} - Locations ({len(route)})",
                hovertemplate="<b>%{text}</b><extra></extra>"
            ))
            
            # Add route lines with real geometry if available
            if len(route) > 1:
                # Check if we have route geometry data
                route_geometry = results.get('route_geometries', {}).get(day)
                if route_geometry and route_geometry.get('geometry_polyline'):
                    # Decode polyline to get actual route path
                    try:
                        polyline_coords = polyline.decode(route_geometry['geometry_polyline'])
                        route_lats = [coord[0] for coord in polyline_coords]
                        route_lons = [coord[1] for coord in polyline_coords]
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=route_lats,
                            lon=route_lons,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=self.day_colors[day_int]
                            ),
                            name=f"Day {day} - Route ({drive_time:.1f} min)",
                            hoverinfo='skip',
                            showlegend=True
                        ))
                    except Exception as e:
                        # Fallback to straight lines if polyline decoding fails
                        logger.warning(f"Failed to decode polyline for day {day}, using straight lines: {e}")
                        fig.add_trace(go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode='lines',
                            line=dict(
                                width=3,
                                color=self.day_colors[day_int]
                            ),
                            name=f"Day {day} - Route ({drive_time:.1f} min)",
                            hoverinfo='skip',
                            showlegend=True
                        ))
                else:
                    # No route geometry available, use straight lines
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        line=dict(
                            width=3,
                            color=self.day_colors[day_int]
                        ),
                        name=f"Day {day} - Route ({drive_time:.1f} min)",
                        hoverinfo='skip',
                        showlegend=True
                    ))
        
        return fig
    
    def create_daily_map(self, results: Dict[str, Any], day: str) -> go.Figure:
        """Create detailed map for a specific day."""
        day_int = int(day)
        fig = go.Figure()
        
        # Add map background
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',  # Clean, minimalist style with light colors
                zoom=12
            ),
            title=f"Route Optimization - Day {day}",
            width=1000,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Handle primary day
        if day in results['primary_assignments']:
            primary_id = results['primary_assignments'][day]
            location = self.locations[primary_id]
            
            fig.add_trace(go.Scattermapbox(
                lat=[location['latitude']],
                lon=[location['longitude']],
                mode='markers',
                marker=dict(
                    size=30,
                    color=self.day_colors[day_int],
                    symbol='circle'
                ),
                text=f"{location['name']} (PRIMARY - Full Day)",
                name="Primary Location",
                hovertemplate="<b>%{text}</b><br>" +
                            "Address: " + location['address'] + 
                            "<extra></extra>"
            ))
            
            # Center map on primary location
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=location['latitude'], lon=location['longitude']),
                    zoom=14
                )
            )
        
        # Handle secondary day
        elif day in results['secondary_assignments']:
            route = results['daily_routes'][day]
            drive_time = results['daily_drive_times'][day]
            
            lats = [self.locations[loc_id]['latitude'] for loc_id in route]
            lons = [self.locations[loc_id]['longitude'] for loc_id in route]
            names = [self.locations[loc_id]['name'] for loc_id in route]
            addresses = [self.locations[loc_id]['address'] for loc_id in route]
            
            # Add numbered location markers
            for i, (lat, lon, name, addr) in enumerate(zip(lats, lons, names, addresses)):
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=self.day_colors[day_int],
                    ),
                    text=str(i + 1),
                    textposition='middle center',
                    textfont=dict(color='white', size=12),
                    name=f"Stop {i + 1}: {name}",
                    hovertemplate=f"<b>Stop {i + 1}: {name}</b><br>" +
                                f"Address: {addr}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ))
            
            # Add route lines with real geometry if available
            if len(route) > 1:
                # Check if we have route geometry data
                route_geometry = results.get('route_geometries', {}).get(day)
                if route_geometry and route_geometry.get('geometry_polyline'):
                    # Decode polyline to get actual route path
                    try:
                        polyline_coords = polyline.decode(route_geometry['geometry_polyline'])
                        route_lats = [coord[0] for coord in polyline_coords]
                        route_lons = [coord[1] for coord in polyline_coords]
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=route_lats,
                            lon=route_lons,
                            mode='lines',
                            line=dict(
                                width=5,
                                color=self.day_colors[day_int]
                            ),
                            name=f"Route (Total: {drive_time:.1f} min)",
                            hoverinfo='skip'
                        ))
                    except Exception as e:
                        # Fallback to straight lines if polyline decoding fails
                        logger.warning(f"Failed to decode polyline for day {day}, using straight lines: {e}")
                        fig.add_trace(go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=self.day_colors[day_int]
                            ),
                            name=f"Route (Total: {drive_time:.1f} min)",
                            hoverinfo='skip'
                        ))
                else:
                    # No route geometry available, use straight lines
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        line=dict(
                            width=4,
                            color=self.day_colors[day_int]
                        ),
                        name=f"Route (Total: {drive_time:.1f} min)",
                        hoverinfo='skip'
                    ))
            
            # Center map on route
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon)
                )
            )
            
            # Add route details as annotation
            route_details = []
            if 'route_details' in results or len(route) > 1:
                for i in range(len(route) - 1):
                    from_name = self.locations[route[i]]['name']
                    to_name = self.locations[route[i + 1]]['name']
                    # Estimate drive time (we could get this from results if stored)
                    route_details.append(f"{i + 1}→{i + 2}: {from_name} → {to_name}")
            
            if route_details:
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text="<br>".join(route_details),
                    showarrow=False,
                    align='left',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filepath: str, width: int = 1200, height: int = 800):
        """Save figure as PNG or HTML."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # Try to save as PNG
            fig.write_image(filepath, width=width, height=height)
            logger.info(f"Saved visualization: {filepath}")
        except Exception as e:
            # Fallback to HTML if image export fails
            html_filepath = filepath.replace('.png', '.html')
            fig.write_html(html_filepath)
            logger.warning(f"PNG export failed, saved HTML instead: {html_filepath}")
            logger.error(f"PNG export error: {e}")
    
    def generate_all_visualizations(
        self, 
        results_path: str = "output/optimization_result.json",
        output_dir: str = "output"
    ):
        """Generate and save all route visualizations."""
        # Load results
        results = self.load_optimization_results(results_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating route visualizations...")
        
        # Generate overview map
        overview_fig = self.create_overview_map(results)
        self.save_visualization(
            overview_fig, 
            f"{output_dir}/routes_overview.png",
            width=1400,
            height=900
        )
        
        # Generate daily maps
        all_days = set(results['primary_assignments'].keys()) | set(results['secondary_assignments'].keys())
        
        for day in sorted(all_days, key=int):
            daily_fig = self.create_daily_map(results, day)
            self.save_visualization(
                daily_fig,
                f"{output_dir}/route_day_{day}.png",
                width=1000,
                height=700
            )
        
        logger.info(f"Generated {len(all_days) + 1} visualization files in {output_dir}/")
        
        # Print summary
        logger.info("Visualization Summary:")
        logger.info("- Overview map: routes_overview.png")
        for day in sorted(all_days, key=int):
            if day in results['primary_assignments']:
                loc_name = self.locations[results['primary_assignments'][day]]['name']
                logger.info(f"- Day {day}: route_day_{day}.png (Primary: {loc_name})")
            else:
                n_locations = len(results['secondary_assignments'][day])
                drive_time = results['daily_drive_times'][day]
                logger.info(f"- Day {day}: route_day_{day}.png ({n_locations} locations, {drive_time:.1f} min)")


def visualize_routes(
    results_path: str = "output/optimization_result.json",
    locations_path: str = "data/subway_locations.jsonl",
    output_dir: str = "output"
):
    """
    Convenience function to generate all route visualizations.
    
    Args:
        results_path: Path to optimization results JSON
        locations_path: Path to locations JSON
        output_dir: Output directory for PNG files
    """
    visualizer = RouteVisualizer(locations_path)
    visualizer.generate_all_visualizations(results_path, output_dir)


# Functional programming versions for the new architecture

def create_route_map(
    locations_df,
    daily_routes: Dict[int, List[int]],
    route_geometries: Dict[int, Any] = None,
    zone_id: str = "zone"
) -> go.Figure:
    """
    Create route map using functional approach.
    
    Args:
        locations_df: Polars DataFrame with location data
        daily_routes: Dictionary mapping day -> list of location IDs
        route_geometries: Optional route geometry data
        zone_id: Zone identifier for title
        
    Returns:
        Plotly figure object
    """
    # Convert DataFrame to lookup dict for compatibility
    locations_dict = {row['location_id']: row for row in locations_df.to_dicts()}
    
    fig = go.Figure()
    
    # Calculate optimal map bounds and zoom level
    all_lats = [row['latitude'] for row in locations_df.to_dicts()]
    all_lons = [row['longitude'] for row in locations_df.to_dicts()]
    
    if all_lats and all_lons:
        # Calculate center point
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Calculate bounding box with padding
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        
        # Add padding (10% of range or minimum 0.001 degrees)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        lat_padding = max(lat_range * 0.1, 0.001)
        lon_padding = max(lon_range * 0.1, 0.001)
        
        # Padded bounds
        padded_min_lat = min_lat - lat_padding
        padded_max_lat = max_lat + lat_padding
        padded_min_lon = min_lon - lon_padding
        padded_max_lon = max_lon + lon_padding
        
        # Calculate the maximum distance between any two points (in degrees)
        import math
        max_distance_deg = 0
        for i in range(len(all_lats)):
            for j in range(i + 1, len(all_lats)):
                # Calculate great circle distance approximation
                lat_diff = all_lats[i] - all_lats[j]
                lon_diff = all_lons[i] - all_lons[j]
                distance = math.sqrt(lat_diff**2 + lon_diff**2)
                max_distance_deg = max(max_distance_deg, distance)
        
        # Calculate zoom level based on maximum distance
        # These thresholds are calibrated for Mapbox/Plotly zoom levels
        if max_distance_deg > 1.0:
            zoom = 8   # Very wide area (multiple cities)
        elif max_distance_deg > 0.5:
            zoom = 9   # Wide metro area
        elif max_distance_deg > 0.2:
            zoom = 10  # Metro area
        elif max_distance_deg > 0.1:
            zoom = 11  # Large city
        elif max_distance_deg > 0.05:
            zoom = 12  # City district
        elif max_distance_deg > 0.02:
            zoom = 13  # Neighborhood
        elif max_distance_deg > 0.01:
            zoom = 14  # Close neighborhood
        else:
            zoom = 15  # Very close locations
        
        logger.info(f"Map bounds: lat [{min_lat:.4f}, {max_lat:.4f}], "
                   f"lon [{min_lon:.4f}, {max_lon:.4f}], "
                   f"max_distance: {max_distance_deg:.4f}°, zoom: {zoom}")
    else:
        # Fallback to San Francisco if no locations
        center_lat, center_lon, zoom = 37.7749, -122.4194, 11
    
    # Color palette for different days
    day_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1',  # Blue
        4: '#96CEB4',  # Green
        5: '#FFEAA7',  # Yellow
        6: '#DDA0DD',  # Plum
        7: '#98D8C8'   # Mint
    }
    
    # Add map background - just set initial center and zoom, no navigation restrictions
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        title=f"Route Optimization - Zone {zone_id}",
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Plot routes for each day
    for day, route in daily_routes.items():
        if not route:
            continue
            
        day_int = int(day)
        color = day_colors.get(day_int, '#999999')
        
        # Get location coordinates
        lats = []
        lons = []
        names = []
        
        for loc_id in route:
            if loc_id in locations_dict:
                loc = locations_dict[loc_id]
                lats.append(loc['latitude'])
                lons.append(loc['longitude'])
                names.append(loc['name'])
        
        if lats:
            # Add location markers
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                ),
                text=[f"Day {day}: {name}" for name in names],
                name=f"Day {day} - Locations ({len(route)})",
                hovertemplate="<b>%{text}</b><extra></extra>"
            ))
            
            # Add route lines if multiple locations
            if len(route) > 1:
                # Use actual OSRM route geometry if available
                if route_geometries and day in route_geometries and route_geometries[day]:
                    try:
                        import polyline
                        route_geom = route_geometries[day]
                        
                        # Decode the polyline geometry from OSRM
                        decoded_coords = polyline.decode(route_geom.geometry_polyline)
                        route_lats = [coord[0] for coord in decoded_coords]
                        route_lons = [coord[1] for coord in decoded_coords]
                        
                        # Add the detailed route path
                        fig.add_trace(go.Scattermapbox(
                            lat=route_lats,
                            lon=route_lons,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=color
                            ),
                            name=f"Day {day} - Route",
                            hoverinfo='skip'
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to decode route geometry for day {day}: {e}")
                        # Fallback to straight lines
                        fig.add_trace(go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode='lines',
                            line=dict(
                                width=4,
                                color=color,
                                dash='dash'  # Use dashed line to indicate fallback
                            ),
                            name=f"Day {day} - Route (fallback)",
                            hoverinfo='skip'
                        ))
                else:
                    # Fallback to straight lines when no route geometry available
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        line=dict(
                            width=4,
                            color=color,
                            dash='dash'  # Use dashed line to indicate fallback
                        ),
                        name=f"Day {day} - Route (direct)",
                        hoverinfo='skip'
                    ))
    
    return fig


def save_route_visualization(
    route_map: go.Figure,
    zone_id: str,
    output_dir: str = "output/visualizations",
    optimization_package=None
) -> str:
    """
    Save route visualization with itinerary to file.
    
    Args:
        route_map: Plotly figure
        zone_id: Zone identifier for filename
        output_dir: Output directory
        optimization_package: Zone optimization data for itinerary
        
    Returns:
        Path to saved file
    """
    import os
    from pathlib import Path
    
    # Ensure directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filepath = f"{output_dir}/route_map_{zone_id}.html"
    
    try:
        # Generate the map HTML
        map_html = route_map.to_html(include_plotlyjs=True, div_id="map-container")
        
        # Generate itinerary HTML
        itinerary_html = generate_itinerary_html(optimization_package) if optimization_package else ""
        
        # Create complete HTML page
        full_html = create_complete_html_page(zone_id, map_html, itinerary_html)
        
        # Save the HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        logger.info(f"Saved route visualization: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        return ""


def generate_itinerary_html(optimization_package) -> str:
    """Generate HTML table for the daily itinerary."""
    if not optimization_package or 'daily_routes' not in optimization_package:
        return ""
    
    daily_routes = optimization_package['daily_routes']
    locations_df = optimization_package['locations_df']
    locations_dict = {row['location_id']: row for row in locations_df.to_dicts()}
    route_geometries = optimization_package.get('route_geometries', {})
    
    # Build itinerary data
    itinerary_rows = []
    
    for day, route in daily_routes.items():
        if not route:
            continue
            
        day_int = int(day)
        route_geom = route_geometries.get(day)
        total_distance = route_geom.total_distance_meters / 1000 if route_geom and hasattr(route_geom, 'total_distance_meters') else 0
        total_duration = route_geom.total_duration_seconds / 60 if route_geom and hasattr(route_geom, 'total_duration_seconds') else 0
        
        # Add day header
        itinerary_rows.append({
            'type': 'day_header',
            'day': day_int,
            'locations': len(route),
            'distance': f"{total_distance:.1f} km" if total_distance > 0 else "N/A",
            'duration': f"{total_duration:.1f} min" if total_duration > 0 else "N/A"
        })
        
        # Add locations for this day
        for i, loc_id in enumerate(route):
            if loc_id in locations_dict:
                loc = locations_dict[loc_id]
                itinerary_rows.append({
                    'type': 'location',
                    'day': day_int,
                    'stop_number': i + 1,
                    'name': loc['name'],
                    'address': loc.get('address', 'N/A'),
                    'location_class': loc.get('location_class', loc.get('class', 'secondary')),
                    'coordinates': f"{loc['latitude']:.4f}, {loc['longitude']:.4f}"
                })
    
    # Build daily summary data
    daily_summary_rows = []
    for row in itinerary_rows:
        if row['type'] == 'day_header':
            daily_summary_rows.append({
                'day': row['day'],
                'locations': row['locations'],
                'distance': row['distance'],
                'duration': row['duration']
            })
    
    # Generate Daily Summary HTML table
    summary_html = """
    <div class="summary-container" style="margin-bottom: 30px;">
        <h3>Daily Summary</h3>
        <table id="summary-table" class="display" style="width:100%">
            <thead>
                <tr>
                    <th>Day</th>
                    <th>Locations</th>
                    <th>Distance</th>
                    <th>Drive Time</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for row in daily_summary_rows:
        summary_html += f"""
                <tr>
                    <td>Day {row['day']}</td>
                    <td>{row['locations']}</td>
                    <td>{row['distance']}</td>
                    <td>{row['duration']}</td>
                </tr>
        """
    
    summary_html += """
            </tbody>
        </table>
    </div>
    """
    
    # Generate Detailed Itinerary HTML table
    itinerary_html = """
    <div class="itinerary-container">
        <h3>Daily Itinerary</h3>
        <table id="itinerary-table" class="display" style="width:100%">
            <thead>
                <tr>
                    <th>Day</th>
                    <th>Stop</th>
                    <th>Time</th>
                    <th>Location</th>
                    <th>Type</th>
                    <th>Address</th>
                    <th>Coordinates</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Calculate drive times between stops for each day
    def get_leg_drive_times(day_str):
        """Get drive times for each leg (stop-to-stop) for a given day."""
        day_geom = route_geometries.get(day_str, {})
        instructions = day_geom.get('turn_by_turn_instructions', [])
        
        if not instructions:
            return {}
            
        # Group instructions by leg_index and sum duration for each leg
        leg_times = {}
        for instruction in instructions:
            leg_idx = instruction.get('leg_index', 0)
            duration_minutes = instruction.get('duration_seconds', 0) / 60.0
            leg_times[leg_idx] = leg_times.get(leg_idx, 0) + duration_minutes
            
        return leg_times
    
    # Track time for each day (starting from 00:00)
    current_day = None
    current_time_minutes = 0
    
    for row in itinerary_rows:
        if row['type'] == 'day_header':
            # Reset time for new day
            current_day = row['day']
            current_time_minutes = 0
            
            # Add empty row with just day number, leave other cells empty
            itinerary_html += f"""
                <tr class="day-header" style="background-color: #f0f8ff; font-weight: bold;">
                    <td>Day {row['day']}</td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                </tr>
            """
        else:
            # Format time as HH:MM
            hours = current_time_minutes // 60
            minutes = current_time_minutes % 60
            time_str = f"{hours:02d}:{minutes:02d}"
            
            class_badge = "primary" if row['location_class'] == 'primary' else "secondary"
            itinerary_html += f"""
                <tr class="location-row">
                    <td>{row['day']}</td>
                    <td>{row['stop_number']}</td>
                    <td>{time_str}</td>
                    <td>{row['name']}</td>
                    <td><span class="badge {class_badge}">{row['location_class']}</span></td>
                    <td>{row['address']}</td>
                    <td>{row['coordinates']}</td>
                </tr>
            """
            
            # Increment time: stop time + drive time to next stop
            if row['location_class'] == 'primary':
                # Primary stores: full day (8 hours = 480 minutes)
                current_time_minutes += 480
            else:
                # Secondary stores: 1 hour (60 minutes)
                current_time_minutes += 60
            
            # Add drive time to next stop (if not the last stop)
            day_str = str(current_day)
            if day_str in route_geometries:
                leg_times = get_leg_drive_times(day_str)
                # Leg index is stop_number - 1 (0-based for drive TO next stop)
                leg_idx = row['stop_number'] - 1
                drive_time = leg_times.get(leg_idx, 0)
                current_time_minutes += drive_time
    
    itinerary_html += """
            </tbody>
        </table>
    </div>
    """
    
    # Combine both tables
    html = summary_html + itinerary_html
    
    return html


def create_complete_html_page(zone_id: str, map_html: str, itinerary_html: str) -> str:
    """Create a complete HTML page with map and itinerary."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Route Optimization - {zone_id}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css">
    
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container-fluid {{
            padding: 20px;
        }}
        #map-container {{
            height: 600px;
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .itinerary-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .badge.primary {{
            background-color: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .badge.secondary {{
            background-color: #6c757d;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .day-header {{
            border-top: 2px solid #007bff !important;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        h3 {{
            color: #34495e;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <h1>Route Optimization - {zone_id.upper()}</h1>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div id="map-section">
                    {map_html}
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                {itinerary_html}
            </div>
        </div>
    </div>
    
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- DataTables JS -->
    <script type="text/javascript" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
    
    <script>
        $(document).ready(function() {{
            // Initialize Summary Table - minimal features
            $('#summary-table').DataTable({{
                "paging": false,
                "searching": false,
                "info": false,
                "ordering": false
            }});
            
            // Initialize Itinerary Table - minimal features  
            $('#itinerary-table').DataTable({{
                "paging": false,
                "searching": false,
                "info": false,
                "ordering": false
            }});
        }});
    </script>
</body>
</html>
    """


def create_geocoding_results_map(locations_data: List[Dict], output_path: str = "output/geocoding_results_map.html") -> str:
    """
    Create an interactive map showing all geocoded locations with before/after comparison.
    
    Args:
        locations_data: List of location dictionaries with lat/lon coordinates
        output_path: Path to save the HTML map
        
    Returns:
        Path to the saved HTML file
    """
    import plotly.graph_objects as go
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract coordinates and metadata
    all_lats = [loc['latitude'] for loc in locations_data]
    all_lons = [loc['longitude'] for loc in locations_data]
    
    if not all_lats or not all_lons:
        logger.warning("No location data provided for geocoding results map")
        return ""
    
    # Calculate optimal map bounds and zoom
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Calculate maximum distance for zoom level
    import math
    max_distance_deg = 0
    for i in range(len(all_lats)):
        for j in range(i + 1, len(all_lats)):
            lat_diff = all_lats[i] - all_lats[j]
            lon_diff = all_lons[i] - all_lons[j]
            distance = math.sqrt(lat_diff**2 + lon_diff**2)
            max_distance_deg = max(max_distance_deg, distance)
    
    # Set zoom level based on geographic spread
    if max_distance_deg > 10.0:
        zoom = 5   # Very wide area (multi-state)
    elif max_distance_deg > 5.0:
        zoom = 6   # Multi-state area
    elif max_distance_deg > 2.0:
        zoom = 7   # State-wide
    elif max_distance_deg > 1.0:
        zoom = 8   # Large region
    elif max_distance_deg > 0.5:
        zoom = 9   # Metro area
    elif max_distance_deg > 0.2:
        zoom = 10  # City area
    else:
        zoom = 11  # Local area
    
    logger.info(f"Geocoding map bounds: lat [{min(all_lats):.4f}, {max(all_lats):.4f}], "
               f"lon [{min(all_lons):.4f}, {max(all_lons):.4f}], "
               f"max_distance: {max_distance_deg:.4f}°, zoom: {zoom}")
    
    fig = go.Figure()
    
    # Color coding for different location types
    primary_lats = []
    primary_lons = []
    primary_names = []
    primary_zones = []
    
    secondary_lats = []
    secondary_lons = []
    secondary_names = []  
    secondary_zones = []
    
    # Separate primary and secondary locations
    for loc in locations_data:
        if loc.get('class') == 'primary':
            primary_lats.append(loc['latitude'])
            primary_lons.append(loc['longitude'])
            primary_names.append(loc['name'])
            primary_zones.append(loc.get('zone_id', 'Unknown'))
        else:
            secondary_lats.append(loc['latitude'])
            secondary_lons.append(loc['longitude']) 
            secondary_names.append(loc['name'])
            secondary_zones.append(loc.get('zone_id', 'Unknown'))
    
    # Add primary locations (larger red markers)
    if primary_lats:
        fig.add_trace(go.Scattermapbox(
            lat=primary_lats,
            lon=primary_lons,
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star'
            ),
            text=[f"<b>{name}</b><br>Zone: {zone}<br>Type: Primary" 
                  for name, zone in zip(primary_names, primary_zones)],
            hovertemplate='%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
            name=f"Primary Locations ({len(primary_lats)})"
        ))
    
    # Add secondary locations (smaller blue markers)
    if secondary_lats:
        fig.add_trace(go.Scattermapbox(
            lat=secondary_lats,
            lon=secondary_lons,
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                symbol='circle'
            ),
            text=[f"<b>{name}</b><br>Zone: {zone}<br>Type: Secondary"
                  for name, zone in zip(secondary_names, secondary_zones)],
            hovertemplate='%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
            name=f"Secondary Locations ({len(secondary_lats)})"
        ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        title=dict(
            text=f"Geocoding Results - {len(locations_data)} Locations",
            font=dict(size=20)
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Generate summary statistics HTML
    zone_counts = {}
    for loc in locations_data:
        zone = loc.get('zone_id', 'Unknown')
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    summary_html = f"""
    <div style="margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
        <h3>Geocoding Summary</h3>
        <p><strong>Total Locations:</strong> {len(locations_data)}</p>
        <p><strong>Primary Locations:</strong> {len(primary_lats)}</p>
        <p><strong>Secondary Locations:</strong> {len(secondary_lats)}</p>
        <p><strong>Zones:</strong> {len(zone_counts)}</p>
        <p><strong>Geographic Bounds:</strong></p>
        <ul style="margin-left: 20px;">
            <li>Latitude: {min(all_lats):.4f}° to {max(all_lats):.4f}°</li>
            <li>Longitude: {min(all_lons):.4f}° to {max(all_lons):.4f}°</li>
            <li>Span: {max_distance_deg:.4f}° (~{max_distance_deg*69:.1f} miles)</li>
        </ul>
    </div>
    """
    
    # Create complete HTML page
    map_html = fig.to_html(include_plotlyjs=True, div_id="map-container")
    
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Geocoding Results Map</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }}
        .header {{
            background-color: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        #map-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geocoding Results</h1>
        <p>Interactive map showing all geocoded locations</p>
    </div>
    
    <div class="container">
        {summary_html}
        
        <div id="map-section">
            {map_html}
        </div>
    </div>
</body>
</html>
    """
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    logger.info(f"Saved geocoding results map: {output_path}")
    return output_path


def create_zone_visualization_map(locations_path: str = "data/subway_locations.jsonl", output_path: str = "output/zone_visualization_map.html") -> str:
    """
    Create an interactive map showing clustered zones with different colors
    
    Args:
        locations_path: Path to JSONL file with clustered locations
        output_path: Output HTML file path
    
    Returns:
        Path to saved HTML file
    """
    import polars as pl
    import plotly.graph_objects as go
    import plotly.express as px
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating zone visualization map from {locations_path}")
    
    # Load locations
    locations_df = pl.read_ndjson(locations_path)
    
    # Filter out locations with null coordinates and zone_ids
    valid_locations = locations_df.filter(
        (pl.col('latitude').is_not_null()) & 
        (pl.col('longitude').is_not_null()) &
        (pl.col('zone_id').is_not_null())
    )
    
    if len(valid_locations) == 0:
        logger.error("No valid clustered locations found!")
        return None
    
    logger.info(f"Visualizing {len(valid_locations)} clustered locations across {valid_locations['zone_id'].n_unique()} zones")
    
    # Convert to pandas for easier plotting
    locations_pd = valid_locations.to_pandas()
    
    # Sort by zone_id numerically for proper legend ordering
    locations_pd['zone_sort_key'] = locations_pd['zone_id'].str.extract(r'(\d+)').astype(int)
    locations_pd = locations_pd.sort_values('zone_sort_key')
    
    # Calculate map bounds
    lat_min = locations_pd['latitude'].min()
    lat_max = locations_pd['latitude'].max()
    lon_min = locations_pd['longitude'].min()
    lon_max = locations_pd['longitude'].max()
    
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Calculate appropriate zoom level using the same method as geocoding map
    import math
    max_distance_deg = 0
    lats = locations_pd['latitude'].tolist()
    lons = locations_pd['longitude'].tolist()
    
    for i in range(len(lats)):
        for j in range(i + 1, len(lats)):
            lat_diff = lats[i] - lats[j]
            lon_diff = lons[i] - lons[j]
            distance = math.sqrt(lat_diff**2 + lon_diff**2)
            max_distance_deg = max(max_distance_deg, distance)
    
    # Set zoom level based on geographic spread (same as geocoding map)
    if max_distance_deg > 10.0:
        zoom = 5   # Very wide area (multi-state)
    elif max_distance_deg > 5.0:
        zoom = 6   # Multi-state area
    elif max_distance_deg > 2.0:
        zoom = 7   # State-wide
    elif max_distance_deg > 1.0:
        zoom = 8   # Large region
    elif max_distance_deg > 0.5:
        zoom = 9   # Metro area
    elif max_distance_deg > 0.2:
        zoom = 10  # City area
    else:
        zoom = 11  # Local area
    
    # Create scatter plot with zone colors
    fig = px.scatter_mapbox(
        locations_pd,
        lat="latitude",
        lon="longitude",
        color="zone_id",
        size_max=15,
        zoom=zoom,
        center={"lat": lat_center, "lon": lon_center},
        mapbox_style="carto-positron",
        hover_data=["name", "address", "class"],
        title="Zone Clustering Results",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        margin={"r":0,"t":50,"l":0,"b":0},
        font=dict(size=14),
        title={
            'text': 'Zone Clustering Results',
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    # Create summary statistics
    zone_stats = locations_pd.groupby(['zone_id', 'zone_sort_key']).agg({
        'name': 'count',
        'class': lambda x: (x == 'primary').sum()
    }).rename(columns={'name': 'total_locations', 'class': 'primary_stores'})
    zone_stats['secondary_stores'] = zone_stats['total_locations'] - zone_stats['primary_stores']
    zone_stats = zone_stats.reset_index().sort_values('zone_sort_key')
    
    # Create summary HTML
    summary_html = f"""
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h2>Zone Clustering Summary</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 6px;">
                    <h3 style="margin: 0; color: #495057;">{len(valid_locations)}</h3>
                    <p style="margin: 5px 0; color: #6c757d;">Total Locations</p>
                </div>
                <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 6px;">
                    <h3 style="margin: 0; color: #495057;">{valid_locations['zone_id'].n_unique()}</h3>
                    <p style="margin: 5px 0; color: #6c757d;">Zones Created</p>
                </div>
                <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 6px;">
                    <h3 style="margin: 0; color: #495057;">{locations_pd[locations_pd['class'] == 'primary'].shape[0]}</h3>
                    <p style="margin: 5px 0; color: #6c757d;">Primary Stores</p>
                </div>
                <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 6px;">
                    <h3 style="margin: 0; color: #495057;">{locations_pd[locations_pd['class'] == 'secondary'].shape[0]}</h3>
                    <p style="margin: 5px 0; color: #6c757d;">Secondary Stores</p>
                </div>
            </div>
            
            <h3>Zone Details</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background-color: #f8f9fa;">
                            <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Zone ID</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Total Locations</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Primary Stores</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">Secondary Stores</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for _, row in zone_stats.iterrows():
        summary_html += f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #dee2e6;">{row['zone_id']}</td>
                            <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{row['total_locations']}</td>
                            <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{row['primary_stores']}</td>
                            <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{row['secondary_stores']}</td>
                        </tr>
        """
    
    summary_html += """
                    </tbody>
                </table>
            </div>
        </div>
    """
    
    # Create complete HTML page
    map_html = fig.to_html(include_plotlyjs=True, div_id="map-container")
    
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zone Clustering Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }}
        .header {{
            background-color: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        #map-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Zone Clustering Results</h1>
        <p>Interactive map showing clustered zones with different colors</p>
    </div>
    
    <div class="container">
        {summary_html}
        
        <div id="map-section">
            {map_html}
        </div>
    </div>
</body>
</html>
    """
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    logger.info(f"Saved zone visualization map: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate visualizations for existing results
    visualize_routes()