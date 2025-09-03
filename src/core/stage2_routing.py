"""
Stage 2: Daily Route Optimization

Functional programming implementation for route optimization.
Converted from class-based DailyRouteOptimizer to functional approach.
"""

import polars as pl
from typing import List, Tuple, Dict, Any
from loguru import logger

# Import functional implementations from stage4_route_optimization.py
from .stage4_route_optimization import (
    create_drive_time_lookup,
    get_drive_time,
    calculate_route_time,
    greedy_nearest_neighbor,
    two_opt_improvement,
    exhaustive_search,
    optimize_daily_route,
    get_route_details
)

# Re-export functions for backward compatibility
__all__ = [
    'create_drive_time_lookup',
    'get_drive_time', 
    'calculate_route_time',
    'greedy_nearest_neighbor',
    'two_opt_improvement', 
    'exhaustive_search',
    'optimize_daily_route',
    'get_route_details'
]


def load_location_names(locations_path: str = "data/subway_locations.json") -> Dict[int, str]:
    """Load location ID to name mapping."""
    import json
    
    with open(locations_path, 'r') as f:
        data = json.load(f)
    
    return {
        loc['id']: loc['name'] 
        for loc in data['subway_locations_san_francisco']
    }


if __name__ == "__main__":
    # Example usage with functional approach
    import polars as pl
    
    # Create mock OD matrix for testing
    mock_od_data = [
        {'zone_id': 'test', 'origin_id': 2, 'destination_id': 3, 'distance_meters': 1000, 'duration_seconds': 180, 'duration_minutes': 3.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None},
        {'zone_id': 'test', 'origin_id': 2, 'destination_id': 4, 'distance_meters': 1500, 'duration_seconds': 300, 'duration_minutes': 5.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None},
        {'zone_id': 'test', 'origin_id': 3, 'destination_id': 2, 'distance_meters': 1000, 'duration_seconds': 180, 'duration_minutes': 3.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None},
        {'zone_id': 'test', 'origin_id': 3, 'destination_id': 4, 'distance_meters': 800, 'duration_seconds': 120, 'duration_minutes': 2.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None},
        {'zone_id': 'test', 'origin_id': 4, 'destination_id': 2, 'distance_meters': 1500, 'duration_seconds': 300, 'duration_minutes': 5.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None},
        {'zone_id': 'test', 'origin_id': 4, 'destination_id': 3, 'distance_meters': 800, 'duration_seconds': 120, 'duration_minutes': 2.0, 'osrm_response_code': 'Ok', 'api_call_timestamp': None}
    ]
    od_matrix_df = pl.DataFrame(mock_od_data)
    
    # Test with a sample set of locations
    test_locations = [2, 3, 4]  # Sample location IDs
    
    logger.info("Route Optimization Example (Functional):")
    logger.info("=======================================")
    
    # Optimize route using functional approach
    route, total_time, metadata = optimize_daily_route(test_locations, od_matrix_df)
    
    # Load location names for display
    location_names = load_location_names()
    
    logger.info(f"Algorithm used: {metadata['algorithm']}")
    logger.info(f"Optimal solution: {metadata['optimal']}")
    logger.info(f"Total drive time: {total_time:.1f} minutes")
    logger.info(f"\nOptimized route:")
    
    for i, location_id in enumerate(route):
        logger.info(f"  {i + 1}. {location_names.get(location_id, f'Location {location_id}')}")
    
    # Show step-by-step details
    logger.info(f"\nRoute details:")
    drive_time_lookup = create_drive_time_lookup(od_matrix_df)
    details = get_route_details(route, drive_time_lookup)
    for step in details:
        from_name = location_names.get(step['from_location_id'], f"Location {step['from_location_id']}")
        to_name = location_names.get(step['to_location_id'], f"Location {step['to_location_id']}")
        logger.info(f"  Step {step['step']}: {from_name} → {to_name}")
        logger.info(f"    Drive time: {step['drive_time_minutes']:.1f} min, Cumulative: {step['cumulative_time_minutes']:.1f} min")