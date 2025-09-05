"""
OSRM API utilities for route optimization.

Provides functions for:
1. OD matrix generation via OSRM Table API
2. Detailed route geometry via OSRM Route API
"""

# standard library imports
from dataclasses import dataclass
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from typing import Any, Dict, List, Optional, Tuple

# 3rd-party imports
import numpy as np
import polars as pl
import requests
from loguru import logger

# OSRM server configuration
OSRM_BASE_URL = "http://192.168.50.2:32050"


@dataclass
class Location:
    """
    Location data structure.
    
    :param location_id: Unique identifier for the location
    :param latitude: Latitude coordinate
    :param longitude: Longitude coordinate
    :param name: Optional name of the location
    :param address: Optional address of the location
    """
    location_id: int
    latitude: float
    longitude: float
    name: Optional[str] = None
    address: Optional[str] = None


@dataclass
class ODMatrixResult:
    """
    Result from OSRM Table API call.
    
    :param zone_id: Zone identifier
    :param distance_matrix: N×N matrix in meters
    :param duration_matrix: N×N matrix in seconds
    :param location_ids: List of location IDs
    :param osrm_response_code: Response code from OSRM API
    :param api_call_timestamp: Timestamp of API call
    """
    zone_id: str
    distance_matrix: np.ndarray  # N×N matrix in meters
    duration_matrix: np.ndarray  # N×N matrix in seconds
    location_ids: List[int]
    osrm_response_code: str
    api_call_timestamp: datetime
    

@dataclass  
class RouteGeometry:
    """
    Result from OSRM Route API call.
    
    :param zone_id: Zone identifier
    :param day_number: Day number
    :param route_location_ids: List of location IDs in route order
    :param geometry_polyline: Encoded polyline geometry
    :param turn_by_turn_instructions: List of turn-by-turn instructions
    :param total_distance_meters: Total route distance in meters
    :param total_duration_seconds: Total route duration in seconds
    :param osrm_response_code: Response code from OSRM API
    :param api_call_timestamp: Timestamp of API call
    """
    zone_id: str
    day_number: int
    route_location_ids: List[int]
    geometry_polyline: str
    turn_by_turn_instructions: List[Dict[str, Any]]
    total_distance_meters: float
    total_duration_seconds: float
    osrm_response_code: str
    api_call_timestamp: datetime


def validate_california_coordinates(
    locations: List[Location],
    zone_id: str
) -> List[Location]:
    """
    Validate that all locations are within California bounds.
    
    California approximate bounds:
    - Latitude: 32.5° to 42.0° N
    - Longitude: -124.5° to -114.0° W
    
    :param locations: List of Location objects
    :param zone_id: Zone identifier for logging
    :return: List of valid locations within California bounds
    """
    CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
    CA_LON_MIN, CA_LON_MAX = -124.5, -114.0
    
    valid_locations = []
    invalid_count = 0
    
    for loc in locations:
        if (CA_LAT_MIN <= loc.latitude <= CA_LAT_MAX and 
            CA_LON_MIN <= loc.longitude <= CA_LON_MAX):
            valid_locations.append(loc)
        else:
            logger.warning(f"zone {zone_id}: location {loc.location_id} ({loc.name}) at ({loc.latitude}, {loc.longitude}) is outside California bounds")
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"zone {zone_id}: filtered out {invalid_count} locations outside California bounds")
    
    return valid_locations


def build_coordinates_string(locations: List[Location]) -> str:
    """
    Build OSRM-compatible coordinate string from locations.
    
    :param locations: List of Location objects
    :return: Semicolon-separated coordinate string: "lon1,lat1;lon2,lat2;..."
    """
    coords = []
    for loc in locations:
        coords.append(f"{loc.longitude},{loc.latitude}")
    return ";".join(coords)


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate haversine distance between two points in meters.
    
    :param lat1: Latitude of first point
    :param lon1: Longitude of first point
    :param lat2: Latitude of second point
    :param lon2: Longitude of second point
    :return: Distance in meters
    """
    
    # convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # earth radius in meters
    r = 6371000
    return r * c


def generate_fallback_od_matrix(
    locations: List[Location],
    zone_id: str,
    api_call_time: datetime
) -> ODMatrixResult:
    """
    Generate fallback OD matrix using haversine distances when OSRM API fails.
    
    :param locations: List of Location objects
    :param zone_id: Zone identifier
    :param api_call_time: Timestamp of the API call attempt
    :return: ODMatrixResult with estimated distances and durations
    """
    n_locations = len(locations)
    location_ids = [loc.location_id for loc in locations]
    
    # build distance matrix using haversine
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                dist = haversine_distance(
                    locations[i].latitude, locations[i].longitude,
                    locations[j].latitude, locations[j].longitude
                )
                distance_matrix[i, j] = dist
    
    # estimate duration from distance (assume 30 km/h average speed = 8.33 m/s)
    duration_matrix = distance_matrix / 8.33
    
    logger.info(f"generated fallback OD matrix for zone {zone_id}")
    
    return ODMatrixResult(
        zone_id=zone_id,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        location_ids=location_ids,
        osrm_response_code="Fallback",
        api_call_timestamp=api_call_time
    )


# ------------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------------

def generate_od_matrix(
    pos_zone: pl.DataFrame,
    centroid: Tuple = None
) -> pl.DataFrame:
    """
    Generate origin-destination matrix for a single zone using OSRM.
    
    :param pos_zone: Polars DataFrame containing location data for a single zone
        Expected columns: pos_id, latitude, longitude, name, address, zone_id
    :param centroid: Tuple of (longitude, latitude) for zone centroid to include in OD matrix
    :return: Polars DataFrame with OD matrix in long format containing:
        - zone_id, origin_id, destination_id
        - distance_meters, duration_seconds, duration_minutes
        - osrm_response_code, api_call_timestamp
    """
    # validate input
    if pos_zone.is_empty():
        logger.warning("empty DataFrame provided to generate_od_matrix")
        return pl.DataFrame()
    
    # extract zone_id
    zone_ids = pos_zone['zone_id'].unique().to_list()
    if len(zone_ids) != 1:
        logger.error(f"Expected single zone, got {len(zone_ids)} zones: {zone_ids}")
        raise ValueError(f"generate_od_matrix requires data from a single zone")
    zone_id = zone_ids[0]
    
    # filter out rows with null coordinates
    pos_valid = pos_zone.filter(
        pl.col('latitude').is_not_null() & 
        pl.col('longitude').is_not_null()
    )
    
    if pos_valid.is_empty():
        logger.warning(f"no valid coordinates for zone {zone_id}")
        return pl.DataFrame()
    
    logger.info(f"generating OD matrix for zone {zone_id} with {len(pos_valid)} valid locations")
    
    # convert to Location objects
    locations = []
    
    # add centroid as first location with ID -1
    if centroid is not None:
        centroid_location = Location(
            location_id=-1,
            latitude=centroid[1],   # centroid is (lon, lat)
            longitude=centroid[0],  # centroid is (lon, lat)
            name="zone centroid",
            address="zone centroid"
        )
        locations.append(centroid_location)
    
    for row in pos_valid.iter_rows(named=True):
        location = Location(
            location_id=row['pos_id'],
            latitude=row['latitude'], 
            longitude=row['longitude'],
            name=row.get('name'),
            address=row.get('address')
        )
        locations.append(location)
    
    # fetch OD matrix from OSRM
    try:
        od_result = fetch_od_matrix(zone_id, locations)
        
        # convert to Polars DataFrame format
        od_df = od_matrix_to_polars(od_result)
        
        logger.success(f"generated OD matrix for zone {zone_id}: {len(od_df)} pairs")
        return od_df
        
    except Exception as e:
        logger.error(f"failed to generate OD matrix for zone {zone_id}: {e}")
        # return empty DataFrame on failure
        return pl.DataFrame()


def fetch_od_matrix(
    zone_id: str, 
    locations: List[Location],
    annotations: List[str] = None
) -> ODMatrixResult:
    """
    Fetch origin-destination matrix from OSRM Table API.
    
    :param zone_id: Zone identifier for this request
    :param locations: List of locations (≤25 for optimal performance)
    :param annotations: List of desired annotations ['duration', 'distance']
    :return: ODMatrixResult with distance and duration matrices
    :raises requests.RequestException: If API call fails
    :raises ValueError: If response format is invalid
    """
    # validate coordinates are within California bounds
    valid_locations = validate_california_coordinates(locations, zone_id)
    
    if len(valid_locations) == 0:
        raise ValueError(f"Zone {zone_id}: No valid locations within California bounds")
    
    if len(valid_locations) != len(locations):
        logger.warning(f"Zone {zone_id}: Using {len(valid_locations)} valid locations out of {len(locations)} total")
    
    if len(valid_locations) > 25:
        logger.warning(f"Zone {zone_id}: {len(valid_locations)} valid locations exceeds recommended 25 limit")
    
    # build coordinate string from valid locations
    coordinates = build_coordinates_string(valid_locations)
    
    # build request URL
    endpoint = f"{OSRM_BASE_URL}/table/v1/driving/{coordinates}"
    
    # add annotations parameter
    if annotations is None:
        annotations = ["duration", "distance"]
    params = {"annotations": ",".join(annotations)}
    
    # make API call
    logger.info(f"Fetching OD matrix for zone {zone_id} with {len(locations)} locations")
    api_call_time = datetime.now()
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
    except requests.Timeout as e:
        logger.error(f"OSRM Table API timeout (30s) for zone {zone_id} with {len(valid_locations)} locations: {e}")
        logger.error(f"Endpoint: {endpoint}")
        logger.error(f"Coordinates: {coordinates}")
        
        # generate dummy OD matrix with estimated distances
        logger.warning(f"Generating fallback OD matrix using haversine distances for zone {zone_id}")
        return generate_fallback_od_matrix(valid_locations, zone_id, api_call_time)
        
    except requests.RequestException as e:
        logger.error(f"OSRM Table API request failed for zone {zone_id}: {e}")
        logger.error(f"Endpoint: {endpoint}")
        
        # generate dummy OD matrix as fallback
        logger.warning(f"Generating fallback OD matrix using haversine distances for zone {zone_id}")
        return generate_fallback_od_matrix(valid_locations, zone_id, api_call_time)
        
    # validate response
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM API returned error for zone {zone_id}: {data.get('message', 'Unknown error')}")
    
    # extract matrices (use valid locations only)
    n_locations = len(valid_locations)
    location_ids = [loc.location_id for loc in valid_locations]
    
    # duration matrix (seconds)
    durations = data.get("durations", [])
    if not durations or len(durations) != n_locations:
        raise ValueError(f"Invalid duration matrix size for zone {zone_id}")
    duration_matrix = np.array(durations, dtype=np.float64)
    
    # ensure matrix is symmetric (OSRM should return symmetric matrix, but enforce it)
    duration_matrix = (duration_matrix + duration_matrix.T) / 2
    
    # distance matrix (meters) - may not be present in all responses
    distances = data.get("distances", [])
    if distances and len(distances) == n_locations:
        distance_matrix = np.array(distances, dtype=np.float64)
        # ensure distance matrix is also symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    else:
        # estimate distances from durations (rough approximation)
        logger.warning(f"No distance matrix in response for zone {zone_id}, estimating from durations")
        # assume average speed of 30 km/h = 8.33 m/s for city driving
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


def generate_fallback_route_geometry(
    locations: List[Location],
    zone_id: str,
    day_number: int,
    api_call_time: datetime
) -> RouteGeometry:
    """
    Generate fallback route geometry using straight-line estimates when OSRM API fails.
    
    :param locations: Ordered list of locations for the route
    :param zone_id: Zone identifier
    :param day_number: Day number
    :param api_call_time: Timestamp of the API call attempt
    :return: RouteGeometry with estimated distances and dummy polyline
    """
    if len(locations) < 2:
        raise ValueError(f"Need at least 2 locations for fallback route")
    
    # calculate total distance using haversine
    total_distance = 0.0
    for i in range(len(locations) - 1):
        dist = haversine_distance(
            locations[i].latitude, locations[i].longitude,
            locations[i + 1].latitude, locations[i + 1].longitude
        )
        total_distance += dist
    
    # estimate duration (assume 30 km/h average speed = 8.33 m/s)
    total_duration = total_distance / 8.33
    
    # generate dummy polyline and basic turn instructions
    dummy_polyline = "o|sbFx}liU" + "?" * (len(locations) * 2)
    instructions = []
    
    for i in range(len(locations) - 1):
        # depart instruction
        instructions.append({
            "leg_index": i,
            "step_index": 0,
            "instruction": "",
            "distance_meters": 0,
            "duration_seconds": 0,
            "maneuver_type": "depart",
            "geometry": "o|sbFx}liU??"
        })
        # arrive instruction
        instructions.append({
            "leg_index": i,
            "step_index": 1,
            "instruction": "",
            "distance_meters": 0,
            "duration_seconds": 0,
            "maneuver_type": "arrive",
            "geometry": "o|sbFx}liU"
        })
    
    logger.info(f"Generated fallback route geometry for zone {zone_id}, day {day_number}")
    
    return RouteGeometry(
        zone_id=zone_id,
        day_number=day_number,
        route_location_ids=[loc.location_id for loc in locations],
        geometry_polyline=dummy_polyline,
        turn_by_turn_instructions=instructions,
        total_distance_meters=total_distance,
        total_duration_seconds=total_duration,
        osrm_response_code="Fallback",
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
    
    :param zone_id: Zone identifier
    :param day_number: Day number (1-7)
    :param route_locations: Ordered list of locations for the route
    :param include_steps: Include turn-by-turn instructions
    :param include_alternatives: Include alternative routes
    :return: RouteGeometry with polyline and turn-by-turn instructions
    :raises requests.RequestException: If API call fails
    :raises ValueError: If response format is invalid
    """
    if len(route_locations) < 2:
        raise ValueError(f"Route must have at least 2 locations, got {len(route_locations)}")
    
    # validate coordinates are within California bounds
    valid_locations = validate_california_coordinates(route_locations, zone_id)
    
    if len(valid_locations) == 0:
        raise ValueError(f"Zone {zone_id}, day {day_number}: No valid locations within California bounds")
    
    if len(valid_locations) < 2:
        raise ValueError(f"Zone {zone_id}, day {day_number}: Need at least 2 valid locations for route, got {len(valid_locations)}")
    
    if len(valid_locations) != len(route_locations):
        logger.warning(f"Zone {zone_id}, day {day_number}: Using {len(valid_locations)} valid locations out of {len(route_locations)} total")
    
    # build coordinate string from valid locations
    coordinates = build_coordinates_string(valid_locations)
    
    # build request URL
    endpoint = f"{OSRM_BASE_URL}/route/v1/driving/{coordinates}"
    
    # build parameters
    params = {
        "overview": "full",  # full geometry
        "geometries": "polyline",  # polyline format
        "steps": "true" if include_steps else "false",
        "alternatives": "true" if include_alternatives else "false"
    }
    
    # make API call
    logger.info(f"Fetching route geometry for zone {zone_id}, day {day_number} with {len(route_locations)} stops")
    api_call_time = datetime.now()
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
    except requests.Timeout as e:
        logger.error(f"OSRM Route API timeout (30s) for zone {zone_id}, day {day_number} with {len(valid_locations)} stops: {e}")
        logger.error(f"Endpoint: {endpoint}")
        logger.error(f"Coordinates: {coordinates}")
        
        # generate dummy route geometry as fallback
        logger.warning(f"Generating fallback route geometry for zone {zone_id}, day {day_number}")
        return generate_fallback_route_geometry(valid_locations, zone_id, day_number, api_call_time)
        
    except requests.RequestException as e:
        logger.error(f"OSRM Route API request failed for zone {zone_id}, day {day_number}: {e}")
        logger.error(f"Endpoint: {endpoint}")
        
        # generate dummy route geometry as fallback
        logger.warning(f"Generating fallback route geometry for zone {zone_id}, day {day_number}")
        return generate_fallback_route_geometry(valid_locations, zone_id, day_number, api_call_time)
        
    # validate response
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM Route API returned error for zone {zone_id}, day {day_number}: {data.get('message', 'Unknown error')}")
    
    # extract route information
    routes = data.get("routes", [])
    if not routes:
        raise ValueError(f"No routes returned for zone {zone_id}, day {day_number}")
    
    # take the first (primary) route
    route = routes[0]
    
    # extract geometry
    geometry_polyline = route.get("geometry", "")
    
    # extract distance and duration
    total_distance = route.get("distance", 0.0)  # meters
    total_duration = route.get("duration", 0.0)  # seconds
    
    # extract turn-by-turn instructions
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
        route_location_ids=[loc.location_id for loc in valid_locations],
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
    
    :param zone_df: Polars DataFrame with location data for single zone
    :return: List of Location objects
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
    
    :param od_result: ODMatrixResult from OSRM API
    :return: Polars DataFrame with OD matrix in long format
    """
    n_locations = len(od_result.location_ids)
    
    # build long-format data
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
    # example usage
    sample_locations = [
        Location(1, 37.7749, -122.4194, "SF Location 1"),
        Location(2, 37.7849, -122.4094, "SF Location 2"),
        Location(3, 37.7649, -122.4294, "SF Location 3")
    ]
    
    # test OD matrix fetch
    try:
        od_result = fetch_od_matrix("test_zone", sample_locations)
        logger.info(f"OD Matrix shape: {od_result.duration_matrix.shape}")
        logger.info(f"Sample duration (Location 1→2): {od_result.duration_matrix[0, 1]:.1f} seconds")
        
        # test route geometry fetch  
        route_result = fetch_route_geometry("test_zone", 1, sample_locations)
        logger.info(f"Route distance: {route_result.total_distance_meters:.0f} meters")
        logger.info(f"Route duration: {route_result.total_duration_seconds:.0f} seconds")
        logger.info(f"Turn-by-turn steps: {len(route_result.turn_by_turn_instructions)}")
        
    except Exception as e:
        logger.warning(f"example failed (expected if OSRM server not running): {e}")


# ------------------------------------------------------------------------------
# end of osrm_utils.py
# ------------------------------------------------------------------------------