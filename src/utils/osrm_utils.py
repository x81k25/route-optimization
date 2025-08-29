"""
OSRM API utilities for route optimization.

Provides functions for:
1. OD matrix generation via OSRM Table API
2. Detailed route geometry via OSRM Route API
"""

import requests
import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# OSRM server configuration
OSRM_BASE_URL = "http://192.168.50.2:32050"


@dataclass
class Location:
    """Location data structure."""
    location_id: int
    latitude: float
    longitude: float
    name: Optional[str] = None
    address: Optional[str] = None


@dataclass
class ODMatrixResult:
    """Result from OSRM Table API call."""
    zone_id: str
    distance_matrix: np.ndarray  # N×N matrix in meters
    duration_matrix: np.ndarray  # N×N matrix in seconds
    location_ids: List[int]
    osrm_response_code: str
    api_call_timestamp: datetime
    

@dataclass  
class RouteGeometry:
    """Result from OSRM Route API call."""
    zone_id: str
    day_number: int
    route_location_ids: List[int]
    geometry_polyline: str
    turn_by_turn_instructions: List[Dict[str, Any]]
    total_distance_meters: float
    total_duration_seconds: float
    osrm_response_code: str
    api_call_timestamp: datetime


def build_coordinates_string(locations: List[Location]) -> str:
    """
    Build OSRM-compatible coordinate string from locations.
    
    Args:
        locations: List of Location objects
        
    Returns:
        Semicolon-separated coordinate string: "lon1,lat1;lon2,lat2;..."
    """
    coords = []
    for loc in locations:
        coords.append(f"{loc.longitude},{loc.latitude}")
    return ";".join(coords)


def fetch_od_matrix(
    zone_id: str, 
    locations: List[Location],
    annotations: List[str] = None
) -> ODMatrixResult:
    """
    Fetch origin-destination matrix from OSRM Table API.
    
    Args:
        zone_id: Zone identifier for this request
        locations: List of locations (≤25 for optimal performance)
        annotations: List of desired annotations ['duration', 'distance']
        
    Returns:
        ODMatrixResult with distance and duration matrices
        
    Raises:
        requests.RequestException: If API call fails
        ValueError: If response format is invalid
    """
    if len(locations) > 25:
        logger.warning(f"Zone {zone_id}: {len(locations)} locations exceeds recommended 25 limit")
    
    # Build coordinate string
    coordinates = build_coordinates_string(locations)
    
    # Build request URL
    endpoint = f"{OSRM_BASE_URL}/table/v1/driving/{coordinates}"
    
    # Add annotations parameter
    if annotations is None:
        annotations = ["duration", "distance"]
    params = {"annotations": ",".join(annotations)}
    
    # Make API call
    logger.info(f"Fetching OD matrix for zone {zone_id} with {len(locations)} locations")
    api_call_time = datetime.now()
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
    except requests.RequestException as e:
        logger.error(f"OSRM API request failed for zone {zone_id}: {e}")
        raise
        
    # Validate response
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM API returned error for zone {zone_id}: {data.get('message', 'Unknown error')}")
    
    # Extract matrices
    n_locations = len(locations)
    location_ids = [loc.location_id for loc in locations]
    
    # Duration matrix (seconds)
    durations = data.get("durations", [])
    if not durations or len(durations) != n_locations:
        raise ValueError(f"Invalid duration matrix size for zone {zone_id}")
    duration_matrix = np.array(durations, dtype=np.float64)
    
    # Ensure matrix is symmetric (OSRM should return symmetric matrix, but enforce it)
    duration_matrix = (duration_matrix + duration_matrix.T) / 2
    
    # Distance matrix (meters) - may not be present in all responses
    distances = data.get("distances", [])
    if distances and len(distances) == n_locations:
        distance_matrix = np.array(distances, dtype=np.float64)
        # Ensure distance matrix is also symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    else:
        # Estimate distances from durations (rough approximation)
        logger.warning(f"No distance matrix in response for zone {zone_id}, estimating from durations")
        # Assume average speed of 30 km/h = 8.33 m/s for city driving
        distance_matrix = duration_matrix * 8.33
    
    logger.info(f"Successfully fetched OD matrix for zone {zone_id}")
    
    return ODMatrixResult(
        zone_id=zone_id,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        location_ids=location_ids,
        osrm_response_code=data["code"],
        api_call_timestamp=api_call_time
    )


def fetch_route_geometry(
    zone_id: str,
    day_number: int, 
    route_locations: List[Location],
    include_steps: bool = True,
    include_alternatives: bool = False
) -> RouteGeometry:
    """
    Fetch detailed route geometry from OSRM Route API.
    
    Args:
        zone_id: Zone identifier
        day_number: Day number (1-7)
        route_locations: Ordered list of locations for the route
        include_steps: Include turn-by-turn instructions
        include_alternatives: Include alternative routes
        
    Returns:
        RouteGeometry with polyline and turn-by-turn instructions
        
    Raises:
        requests.RequestException: If API call fails
        ValueError: If response format is invalid
    """
    if len(route_locations) < 2:
        raise ValueError(f"Route must have at least 2 locations, got {len(route_locations)}")
    
    # Build coordinate string
    coordinates = build_coordinates_string(route_locations)
    
    # Build request URL
    endpoint = f"{OSRM_BASE_URL}/route/v1/driving/{coordinates}"
    
    # Build parameters
    params = {
        "overview": "full",  # Full geometry
        "geometries": "polyline",  # Polyline format
        "steps": "true" if include_steps else "false",
        "alternatives": "true" if include_alternatives else "false"
    }
    
    # Make API call
    logger.info(f"Fetching route geometry for zone {zone_id}, day {day_number} with {len(route_locations)} stops")
    api_call_time = datetime.now()
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
    except requests.RequestException as e:
        logger.error(f"OSRM Route API request failed for zone {zone_id}, day {day_number}: {e}")
        raise
        
    # Validate response
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM Route API returned error for zone {zone_id}, day {day_number}: {data.get('message', 'Unknown error')}")
    
    # Extract route information
    routes = data.get("routes", [])
    if not routes:
        raise ValueError(f"No routes returned for zone {zone_id}, day {day_number}")
    
    # Take the first (primary) route
    route = routes[0]
    
    # Extract geometry
    geometry_polyline = route.get("geometry", "")
    
    # Extract distance and duration
    total_distance = route.get("distance", 0.0)  # meters
    total_duration = route.get("duration", 0.0)  # seconds
    
    # Extract turn-by-turn instructions
    instructions = []
    if include_steps:
        legs = route.get("legs", [])
        for leg_idx, leg in enumerate(legs):
            steps = leg.get("steps", [])
            for step_idx, step in enumerate(steps):
                instruction = {
                    "leg_index": leg_idx,
                    "step_index": step_idx,
                    "instruction": step.get("maneuver", {}).get("instruction", ""),
                    "distance_meters": step.get("distance", 0.0),
                    "duration_seconds": step.get("duration", 0.0),
                    "maneuver_type": step.get("maneuver", {}).get("type", ""),
                    "geometry": step.get("geometry", "")
                }
                instructions.append(instruction)
    
    logger.info(f"Successfully fetched route geometry for zone {zone_id}, day {day_number}")
    
    return RouteGeometry(
        zone_id=zone_id,
        day_number=day_number,
        route_location_ids=[loc.location_id for loc in route_locations],
        geometry_polyline=geometry_polyline,
        turn_by_turn_instructions=instructions,
        total_distance_meters=total_distance,
        total_duration_seconds=total_duration,
        osrm_response_code=data["code"],
        api_call_timestamp=api_call_time
    )


def convert_locations_from_polars(zone_df: pl.DataFrame) -> List[Location]:
    """
    Convert Polars DataFrame to list of Location objects.
    
    Args:
        zone_df: Polars DataFrame with location data for single zone
        
    Returns:
        List of Location objects
    """
    locations = []
    for row in zone_df.iter_rows(named=True):
        location = Location(
            location_id=row["location_id"],
            latitude=row["latitude"], 
            longitude=row["longitude"],
            name=row.get("name"),
            address=row.get("address")
        )
        locations.append(location)
    
    return locations


def od_matrix_to_polars(od_result: ODMatrixResult) -> pl.DataFrame:
    """
    Convert ODMatrixResult to Polars DataFrame format.
    
    Args:
        od_result: ODMatrixResult from OSRM API
        
    Returns:
        Polars DataFrame with OD matrix in long format
    """
    n_locations = len(od_result.location_ids)
    
    # Build long-format data
    rows = []
    for i in range(n_locations):
        for j in range(n_locations):
            row = {
                "zone_id": od_result.zone_id,
                "origin_id": od_result.location_ids[i],
                "destination_id": od_result.location_ids[j],
                "distance_meters": od_result.distance_matrix[i, j],
                "duration_seconds": od_result.duration_matrix[i, j],
                "duration_minutes": od_result.duration_matrix[i, j] / 60.0,
                "osrm_response_code": od_result.osrm_response_code,
                "api_call_timestamp": od_result.api_call_timestamp
            }
            rows.append(row)
    
    return pl.DataFrame(rows)


if __name__ == "__main__":
    # Example usage
    sample_locations = [
        Location(1, 37.7749, -122.4194, "SF Location 1"),
        Location(2, 37.7849, -122.4094, "SF Location 2"),
        Location(3, 37.7649, -122.4294, "SF Location 3")
    ]
    
    # Test OD matrix fetch
    try:
        od_result = fetch_od_matrix("test_zone", sample_locations)
        logger.info(f"OD Matrix shape: {od_result.duration_matrix.shape}")
        logger.info(f"Sample duration (Location 1→2): {od_result.duration_matrix[0, 1]:.1f} seconds")
        
        # Test route geometry fetch  
        route_result = fetch_route_geometry("test_zone", 1, sample_locations)
        logger.info(f"Route distance: {route_result.total_distance_meters:.0f} meters")
        logger.info(f"Route duration: {route_result.total_duration_seconds:.0f} seconds")
        logger.info(f"Turn-by-turn steps: {len(route_result.turn_by_turn_instructions)}")
        
    except Exception as e:
        logger.warning(f"Example failed (expected if OSRM server not running): {e}")