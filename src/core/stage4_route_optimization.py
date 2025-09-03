"""
Core Route Optimization Pipeline

Handles Stage 4 from the architecture:
- Individual Day Route Optimization (TSP per assigned day)
- Converts the class-based DailyRouteOptimizer to functional approach
"""

import numpy as np
import polars as pl
from typing import List, Tuple, Dict, Any
import itertools
from loguru import logger

from ..utils.osrm_utils import fetch_route_geometry, convert_locations_from_polars


def create_drive_time_lookup(od_matrix_df: pl.DataFrame) -> Dict[Tuple[int, int], float]:
    """
    Create lookup dictionary for quick drive time access from OD matrix.
    
    Args:
        od_matrix_df: Polars DataFrame with OD matrix data
        
    Returns:
        Dictionary mapping (origin_id, dest_id) -> drive_time_minutes
    """
    drive_time_lookup = {}
    
    if od_matrix_df.is_empty():
        return drive_time_lookup
        
    for row in od_matrix_df.iter_rows():
        origin_id, dest_id = row[1], row[2]  # origin_id, destination_id from OSRM format
        drive_time = row[5]  # duration_minutes
        drive_time_lookup[(origin_id, dest_id)] = drive_time
    
    return drive_time_lookup


def get_drive_time(origin_id: int, dest_id: int, drive_time_lookup: Dict[Tuple[int, int], float]) -> float:
    """Get drive time between two locations."""
    if origin_id == dest_id:
        return 0.0
    return drive_time_lookup.get((origin_id, dest_id), float('inf'))


def calculate_route_time(route: List[int], drive_time_lookup: Dict[Tuple[int, int], float]) -> float:
    """Calculate total drive time for a route."""
    if len(route) <= 1:
        return 0.0
    
    total_time = 0.0
    for i in range(len(route) - 1):
        total_time += get_drive_time(route[i], route[i + 1], drive_time_lookup)
    
    return total_time


def greedy_nearest_neighbor(
    location_ids: List[int], 
    drive_time_lookup: Dict[Tuple[int, int], float],
    start_location_id: int = None
) -> Tuple[List[int], float]:
    """
    Solve TSP using Greedy Nearest Neighbor algorithm.
    
    Args:
        location_ids: List of location IDs to visit
        drive_time_lookup: Dictionary for drive time lookups
        start_location_id: Starting location (if None, uses first location)
        
    Returns:
        Tuple of (route, total_drive_time)
    """
    if len(location_ids) <= 1:
        return location_ids, 0.0
    
    # Choose starting location
    current_location = start_location_id if start_location_id is not None else location_ids[0]
    unvisited = set(location_ids) - {current_location}
    route = [current_location]
    total_time = 0.0
    
    # Greedy selection: always visit nearest unvisited location
    while unvisited:
        nearest_location = min(
            unvisited,
            key=lambda loc: get_drive_time(current_location, loc, drive_time_lookup)
        )
        
        drive_time = get_drive_time(current_location, nearest_location, drive_time_lookup)
        total_time += drive_time
        
        route.append(nearest_location)
        unvisited.remove(nearest_location)
        current_location = nearest_location
    
    return route, total_time


def two_opt_improvement(
    route: List[int], 
    drive_time_lookup: Dict[Tuple[int, int], float],
    max_iterations: int = 100
) -> Tuple[List[int], float]:
    """
    Improve route using 2-opt local search.
    
    Args:
        route: Initial route as list of location IDs
        drive_time_lookup: Dictionary for drive time lookups
        max_iterations: Maximum number of improvement iterations
        
    Returns:
        Tuple of (improved_route, total_drive_time)
    """
    if len(route) <= 3:
        return route, calculate_route_time(route, drive_time_lookup)
    
    best_route = route.copy()
    best_time = calculate_route_time(best_route, drive_time_lookup)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try all possible 2-opt swaps
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:  # Skip adjacent edges
                    continue
                
                # Create new route by reversing segment between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_time = calculate_route_time(new_route, drive_time_lookup)
                
                if new_time < best_time:
                    best_route = new_route.copy()
                    best_time = new_time
                    improved = True
        
        if improved:
            route = best_route.copy()
        else:
            break  # No more improvements found
    
    return best_route, best_time


def exhaustive_search(
    location_ids: List[int], 
    drive_time_lookup: Dict[Tuple[int, int], float]
) -> Tuple[List[int], float]:
    """
    Solve TSP using exhaustive search (brute force).
    Only use for small problems (≤ 8 locations).
    
    Args:
        location_ids: List of location IDs to visit
        drive_time_lookup: Dictionary for drive time lookups
        
    Returns:
        Tuple of (optimal_route, total_drive_time)
    """
    if len(location_ids) > 8:
        raise ValueError("Exhaustive search only supported for ≤ 8 locations")
    
    if len(location_ids) <= 1:
        return location_ids, 0.0
    
    best_route = None
    best_time = float('inf')
    
    # Fix first location, permute the rest
    first_location = location_ids[0]
    remaining_locations = location_ids[1:]
    
    for perm in itertools.permutations(remaining_locations):
        route = [first_location] + list(perm)
        route_time = calculate_route_time(route, drive_time_lookup)
        
        if route_time < best_time:
            best_route = route
            best_time = route_time
    
    return best_route, best_time


def optimize_daily_route(
    location_ids: List[int],
    od_matrix_df: pl.DataFrame,
    start_location_id: int = None,
    use_exhaustive_if_small: bool = True
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Main function to optimize a route for given locations on a single day.
    
    Args:
        location_ids: List of location IDs to visit
        od_matrix_df: Polars DataFrame containing OD matrix
        start_location_id: Starting location (if None, uses first location)
        use_exhaustive_if_small: Use exhaustive search for ≤ 5 locations
        
    Returns:
        Tuple of (route, total_time, metadata)
    """
    if not location_ids:
        return [], 0.0, {'algorithm': 'empty'}
    
    # Create drive time lookup
    drive_time_lookup = create_drive_time_lookup(od_matrix_df)
    
    metadata = {
        'n_locations': len(location_ids),
        'start_location': start_location_id or location_ids[0]
    }
    
    # Choose algorithm based on problem size
    if use_exhaustive_if_small and len(location_ids) <= 5:
        # Use exhaustive search for small problems
        route, total_time = exhaustive_search(location_ids, drive_time_lookup)
        metadata['algorithm'] = 'exhaustive_search'
        metadata['optimal'] = True
    else:
        # Use greedy + 2-opt for larger problems
        route, _ = greedy_nearest_neighbor(location_ids, drive_time_lookup, start_location_id)
        route, total_time = two_opt_improvement(route, drive_time_lookup)
        metadata['algorithm'] = 'greedy_plus_2opt'
        metadata['optimal'] = False
    
    metadata['total_drive_time_minutes'] = total_time
    
    return route, total_time, metadata


def get_route_details(route: List[int], drive_time_lookup: Dict[Tuple[int, int], float]) -> List[Dict[str, Any]]:
    """
    Get detailed information about each step in the route.
    
    Args:
        route: List of location IDs in order
        drive_time_lookup: Dictionary for drive time lookups
        
    Returns:
        List of step details including drive times
    """
    if len(route) <= 1:
        return []
    
    details = []
    cumulative_time = 0.0
    
    for i in range(len(route) - 1):
        from_id = route[i]
        to_id = route[i + 1]
        drive_time = get_drive_time(from_id, to_id, drive_time_lookup)
        cumulative_time += drive_time
        
        details.append({
            'step': i + 1,
            'from_location_id': from_id,
            'to_location_id': to_id,
            'drive_time_minutes': drive_time,
            'cumulative_time_minutes': cumulative_time
        })
    
    return details


def optimize_all_daily_routes(optimization_package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize routes for all days in a zone optimization package.
    
    Args:
        optimization_package: Package from data_processing.py containing all zone data
        
    Returns:
        Updated optimization package with route optimization results
    """
    zone_id = optimization_package['zone_id']
    primary_assignments = optimization_package['primary_assignments']
    secondary_assignments = optimization_package['secondary_assignments']
    od_matrix_df = optimization_package['od_matrix_df']
    secondary_df = optimization_package['secondary_df']
    
    logger.info(f"Optimizing routes for zone {zone_id}")
    
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
        
        logger.info(f"Day {day}: {route_metadata['algorithm']}, {drive_time:.1f} min, {len(route)} locations")
        
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
    
    # Add route optimization results to package
    optimization_package.update({
        'daily_routes': daily_routes,
        'daily_drive_times': daily_drive_times,
        'route_geometries': route_geometries
    })
    
    return optimization_package


if __name__ == "__main__":
    # Example usage with functional approach
    from .stage0_data_ingestion import load_locations_from_jsonl, create_zone_optimization_package
    import yaml
    
    # Load configuration
    with open("config/model-params.yaml", 'r') as f:
        config = yaml.safe_load(f)['model_params']
    
    zone_id = "test_zone"
    locations_df = load_locations_from_jsonl("data/subway_locations.json", zone_id)
    
    logger.info("Starting route optimization pipeline...")
    
    # Create optimization package with data processing
    optimization_package = create_zone_optimization_package(locations_df, config, zone_id)
    
    # Optimize all routes
    optimization_package = optimize_all_daily_routes(optimization_package)
    
    logger.info(f"Route optimization complete:")
    logger.info(f"- Daily routes: {len(optimization_package['daily_routes'])} days")
    logger.info(f"- Total drive time: {sum(optimization_package['daily_drive_times'].values()):.1f} minutes")
    logger.info(f"- Route geometries: {sum(1 for rg in optimization_package['route_geometries'].values() if rg is not None)}")