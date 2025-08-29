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


if __name__ == "__main__":
    # Generate visualizations for existing results
    visualize_routes()