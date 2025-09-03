"""
Main Route Optimization Orchestrator

Functional programming implementation that coordinates Stage 1 (day assignment) 
and Stage 2 (route optimization) to solve the complete route optimization problem.
"""

import json
import yaml
import polars as pl
from typing import Dict, List, Tuple, Any
from datetime import datetime
from loguru import logger

from .stage1_assignment import assign_days_to_secondary_locations, get_od_matrix_polars
from .stage2_routing import optimize_daily_route
from ..utils.osrm_utils import fetch_route_geometry, convert_locations_from_polars


def load_config(config_path: str = "config/model-params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['model_params']


def load_locations_from_jsonl(locations_path: str, zone_id: str = None) -> pl.DataFrame:
    """Load location data from JSONL file and return as Polars DataFrame."""
    data = []
    
    if locations_path.endswith('.jsonl'):
        with open(locations_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    loc_data = json.loads(line)
                    
                    # Filter by zone_id if specified
                    if zone_id and 'zone_id' in loc_data:
                        if loc_data.get('zone_id') != zone_id:
                            continue
                    
                    data.append({
                        'location_id': loc_data['id'],
                        'zone_id': loc_data.get('zone_id', zone_id or 'default'),
                        'name': loc_data['name'],
                        'location_class': loc_data['class'],
                        'address': loc_data['address'],
                        'latitude': loc_data['latitude'],
                        'longitude': loc_data['longitude'],
                        'source_system': 'jsonl_file'
                    })
    else:
        # Handle traditional JSON format
        with open(locations_path, 'r') as f:
            json_data = json.load(f)
        
        locations_key = None
        if 'subway_locations_california' in json_data:
            locations_key = 'subway_locations_california'
        elif 'subway_locations_san_francisco' in json_data:
            locations_key = 'subway_locations_san_francisco'
        else:
            raise ValueError("Could not find locations in dataset")
        
        for loc_data in json_data[locations_key]:
            if zone_id and 'zone_id' in loc_data:
                if loc_data.get('zone_id') != zone_id:
                    continue
            
            data.append({
                'location_id': loc_data['id'],
                'zone_id': loc_data.get('zone_id', zone_id or 'default'),
                'name': loc_data['name'],
                'location_class': loc_data['class'],
                'address': loc_data['address'],
                'latitude': loc_data['latitude'],
                'longitude': loc_data['longitude'],
                'source_system': 'json_file'
            })
    
    return pl.DataFrame(data)


def separate_primary_secondary_locations(locations_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Separate locations into primary and secondary DataFrames."""
    primary_df = locations_df.filter(pl.col('location_class') == 'primary')
    secondary_df = locations_df.filter(pl.col('location_class') == 'secondary')
    return primary_df, secondary_df


def assign_primary_locations_to_days(primary_df: pl.DataFrame) -> Dict[int, int]:
    """
    Assign primary locations to days.
    Simple assignment: one primary location per day.
    
    Returns:
        Dictionary mapping day -> primary_location_id
    """
    primary_assignments = {}
    primary_locations = primary_df.to_dicts()
    
    for i, location in enumerate(primary_locations):
        day = i + 1  # Days start from 1
        primary_assignments[day] = location['location_id']
    
    return primary_assignments


def calculate_available_secondary_days(primary_count: int, days_per_week: int) -> int:
    """Calculate how many days are available for secondary locations."""
    return max(0, days_per_week - primary_count)


def calculate_total_drive_time(daily_drive_times: Dict[int, float]) -> float:
    """Calculate total drive time across all days."""
    return sum(daily_drive_times.values())


def calculate_total_locations_visited(primary_assignments: Dict[int, int], 
                                    secondary_assignments: Dict[int, List[int]]) -> int:
    """Calculate total number of locations visited."""
    total = len(primary_assignments)  # Primary locations
    total += sum(len(locs) for locs in secondary_assignments.values())  # Secondary locations
    return total


def calculate_primary_store_count(primary_assignments: Dict[int, int]) -> int:
    """Calculate total number of primary locations."""
    return len(primary_assignments)


def calculate_non_primary_store_count(secondary_assignments: Dict[int, List[int]]) -> int:
    """Calculate total number of secondary/non-primary locations."""
    return sum(len(locs) for locs in secondary_assignments.values())


def calculate_total_primary_hours(config: Dict[str, Any]) -> float:
    """Calculate total hours spent at primary locations."""
    return config.get('primary_hours_per_week', 24)


def calculate_total_non_primary_hours(secondary_assignments: Dict[int, List[int]], 
                                    config: Dict[str, Any]) -> float:
    """Calculate total hours spent at non-primary locations."""
    hours_per_non_primary = config.get('hours_per_non_primary', 1)
    non_primary_count = calculate_non_primary_store_count(secondary_assignments)
    return non_primary_count * hours_per_non_primary


def calculate_unutilized_time(primary_assignments: Dict[int, int],
                            secondary_assignments: Dict[int, List[int]],
                            daily_drive_times: Dict[int, float],
                            config: Dict[str, Any]) -> float:
    """Calculate unutilized time in hours per week, including drive time only for secondary days."""
    days_per_week = config.get('days_per_week', 5)
    utilization = config.get('utilization', 100) / 100.0
    
    # Total available hours per week (assuming 8 hours per working day)
    total_available_hours = days_per_week * 8.0 * utilization
    
    # Used hours: location time + drive time (only for secondary days)
    location_hours = (calculate_total_primary_hours(config) + 
                     calculate_total_non_primary_hours(secondary_assignments, config))
    
    # Calculate drive time only for secondary days (not primary days)
    secondary_drive_time = 0.0
    for day, drive_time in daily_drive_times.items():
        # Only count drive time if this day has secondary assignments (not primary)
        if day in secondary_assignments:
            secondary_drive_time += drive_time
    
    drive_hours = secondary_drive_time / 60.0  # Convert minutes to hours
    used_hours = location_hours + drive_hours
    
    # Unutilized time (can be negative if over-utilized)
    return float(total_available_hours - used_hours)


def optimize_zone(locations_df: pl.DataFrame, config: Dict[str, Any], zone_id: str) -> Dict[str, Any]:
    """
    Main optimization function that coordinates both stages.
    
    Args:
        locations_df: DataFrame containing location data
        config: Configuration dictionary
        zone_id: Zone identifier
        
    Returns:
        Dictionary containing complete optimization result
    """
    start_time = datetime.now()
    
    # Separate primary and secondary locations
    primary_df, secondary_df = separate_primary_secondary_locations(locations_df)
    
    # Stage 1: Assign primary locations to days
    primary_assignments = assign_primary_locations_to_days(primary_df)
    
    # Calculate available days for secondary locations
    available_secondary_days = calculate_available_secondary_days(
        len(primary_assignments), config['days_per_week']
    )
    
    if available_secondary_days <= 0:
        # No days available for secondary locations
        secondary_assignments = {}
        daily_routes = {day: [primary_id] for day, primary_id in primary_assignments.items()}
        daily_drive_times = {day: 0.0 for day in primary_assignments}
        route_geometries = {day: None for day in primary_assignments}
    else:
        # Get OD matrix for secondary locations
        od_matrix_df = get_od_matrix_polars(zone_id, secondary_df)
        
        # Stage 1: Assign secondary locations to available days
        secondary_clusters = assign_days_to_secondary_locations(
            secondary_df=secondary_df,
            zone_id=zone_id,
            available_secondary_days=available_secondary_days,
            max_locations_per_day=config['locations_per_day_max'],
            use_swap_optimization=True
        )
        
        # Convert cluster IDs to actual day numbers
        secondary_assignments = {}
        available_days = [d for d in range(1, config['days_per_week'] + 1) 
                        if d not in primary_assignments]
        
        for cluster_id, location_ids in secondary_clusters.items():
            if cluster_id - 1 < len(available_days):  # cluster_id starts from 1
                day = available_days[cluster_id - 1]
                secondary_assignments[day] = location_ids
        
        # Stage 2: Optimize routes for each day
        daily_routes = {}
        daily_drive_times = {}
        route_geometries = {}
        
        # Primary days (single location, no routing needed)
        for day, primary_id in primary_assignments.items():
            daily_routes[day] = [primary_id]
            daily_drive_times[day] = 0.0
            route_geometries[day] = None
        
        # Secondary days (optimize routes)
        for day, location_ids in secondary_assignments.items():
            route, drive_time, route_metadata = optimize_daily_route(
                location_ids=location_ids,
                od_matrix_df=od_matrix_df,
                use_exhaustive_if_small=True
            )
            daily_routes[day] = route
            daily_drive_times[day] = drive_time
            
            # Fetch detailed route geometry
            if len(route) > 1:
                route_locations_df = secondary_df.filter(
                    pl.col('location_id').is_in(route)
                )
                route_locations = convert_locations_from_polars(route_locations_df)
                
                # Reorder locations according to optimized route
                ordered_locations = []
                for loc_id in route:
                    for loc in route_locations:
                        if loc.location_id == loc_id:
                            ordered_locations.append(loc)
                            break
                
                route_geometry = fetch_route_geometry(
                    zone_id=zone_id,
                    day_number=day,
                    route_locations=ordered_locations,
                    include_steps=True
                )
                route_geometries[day] = route_geometry
            else:
                route_geometries[day] = None
    
    end_time = datetime.now()
    
    # Calculate quality metrics for secondary clustering
    stage1_quality_metrics = {}
    if available_secondary_days > 0 and secondary_clusters:
        from .stage1_assignment import calculate_cluster_quality
        stage1_quality_metrics = calculate_cluster_quality(secondary_clusters, secondary_df)
    
    # Compile metadata
    metadata = {
        'optimization_start_time': start_time.isoformat(),
        'optimization_end_time': end_time.isoformat(),
        'optimization_duration_seconds': (end_time - start_time).total_seconds(),
        'config': config,
        'zone_id': zone_id,
        'n_primary_locations': len(primary_df),
        'n_secondary_locations': len(secondary_df),
        'available_secondary_days': available_secondary_days,
        'stage1_quality_metrics': stage1_quality_metrics,
        'osrm_integration': True,
        'route_geometries_fetched': sum(1 for rg in route_geometries.values() if rg is not None)
    }
    
    return {
        'primary_assignments': primary_assignments,
        'secondary_assignments': secondary_assignments,
        'daily_routes': daily_routes,
        'daily_drive_times': daily_drive_times,
        'route_geometries': route_geometries,
        'metadata': metadata
    }


def print_optimization_solution(optimization_result: Dict[str, Any], locations_df: pl.DataFrame) -> None:
    """Print human-readable optimization results."""
    logger.info("Route Optimization Results")
    logger.info("=" * 50)
    
    # Extract data
    primary_assignments = optimization_result['primary_assignments']
    secondary_assignments = optimization_result['secondary_assignments']
    daily_routes = optimization_result['daily_routes']
    daily_drive_times = optimization_result['daily_drive_times']
    route_geometries = optimization_result['route_geometries']
    metadata = optimization_result['metadata']
    
    # Summary statistics
    total_locations = calculate_total_locations_visited(primary_assignments, secondary_assignments)
    total_drive_time = calculate_total_drive_time(daily_drive_times)
    
    logger.info(f"Total locations: {total_locations}")
    logger.info(f"Total drive time: {total_drive_time:.1f} minutes")
    logger.info(f"Optimization time: {metadata['optimization_duration_seconds']:.2f} seconds")
    logger.info(f"Zone ID: {metadata.get('zone_id', 'N/A')}")
    logger.info(f"Route geometries fetched: {metadata.get('route_geometries_fetched', 0)}")
    logger.info("")
    
    # Create location lookup
    location_lookup = {row['location_id']: row['name'] for row in locations_df.to_dicts()}
    
    # Print daily schedules
    all_days = sorted(set(primary_assignments.keys()) | set(secondary_assignments.keys()))
    
    for day in all_days:
        logger.info(f"Day {day}:")
        
        if day in primary_assignments:
            primary_id = primary_assignments[day]
            logger.info(f"  PRIMARY: {location_lookup[primary_id]} (full day)")
            logger.info(f"  Drive time: 0.0 minutes")
        
        elif day in secondary_assignments:
            location_ids = secondary_assignments[day]
            route = daily_routes[day]
            drive_time = daily_drive_times[day]
            
            logger.info(f"  SECONDARY ({len(location_ids)} locations):")
            logger.info(f"  Route: {' → '.join([location_lookup[loc_id] for loc_id in route])}")
            logger.info(f"  Drive time: {drive_time:.1f} minutes")
            
            # Show route geometry info if available
            if day in route_geometries and route_geometries[day]:
                route_geom = route_geometries[day]
                logger.info(f"  Route geometry: {len(route_geom.turn_by_turn_instructions)} turn instructions")
                logger.info(f"  Total distance: {route_geom.total_distance_meters:.0f} meters")
        
        logger.info("")
    
    # Quality metrics
    if metadata['stage1_quality_metrics']:
        logger.info("Clustering Quality Metrics:")
        for metric, value in metadata['stage1_quality_metrics'].items():
            logger.info(f"  {metric}: {value:.2f}")


def save_optimization_solution(optimization_result: Dict[str, Any], output_path: str) -> None:
    """Save optimization results to JSON file."""
    # Convert route geometries to serializable format
    route_geometries_serializable = {}
    for day, route_geom in optimization_result['route_geometries'].items():
        if route_geom:
            route_geometries_serializable[day] = {
                'zone_id': route_geom.zone_id,
                'day_number': route_geom.day_number,
                'route_location_ids': route_geom.route_location_ids,
                'geometry_polyline': route_geom.geometry_polyline,
                'total_distance_meters': route_geom.total_distance_meters,
                'total_duration_seconds': route_geom.total_duration_seconds,
                'turn_by_turn_instructions': route_geom.turn_by_turn_instructions,
                'osrm_response_code': route_geom.osrm_response_code,
                'api_call_timestamp': route_geom.api_call_timestamp.isoformat()
            }
        else:
            route_geometries_serializable[day] = None
    
    # Extract data for summary
    primary_assignments = optimization_result['primary_assignments']
    secondary_assignments = optimization_result['secondary_assignments']
    daily_drive_times = optimization_result['daily_drive_times']
    
    solution_data = {
        'primary_assignments': primary_assignments,
        'secondary_assignments': secondary_assignments,
        'daily_routes': optimization_result['daily_routes'],
        'daily_drive_times': daily_drive_times,
        'route_geometries': route_geometries_serializable,
        'metadata': optimization_result['metadata'],
        'summary': {
            'total_locations_visited': calculate_total_locations_visited(primary_assignments, secondary_assignments),
            'total_drive_time_minutes': calculate_total_drive_time(daily_drive_times)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(solution_data, f, indent=2, default=str)
    
    logger.info(f"Solution saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    zone_id = "sf_subway_zone"
    locations_df = load_locations_from_jsonl("data/subway_locations.jsonl", zone_id)
    
    logger.info("Starting route optimization with OSRM integration...")
    result = optimize_zone(locations_df, config, zone_id)
    
    # Display results
    print_optimization_solution(result, locations_df)
    
    # Save results
    save_optimization_solution(result, "output/optimization_result.json")