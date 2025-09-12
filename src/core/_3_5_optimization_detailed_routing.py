"""
Stage 3.5: Detailed Routing
Fetch detailed route geometry and turn-by-turn directions from OSRM
"""

# standard library imports
from typing import Any, Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
from src.utils.osrm_utils import fetch_route_geometry


def get_detailed_route_information(
    optimized_routes: Dict[int, Tuple[List[int], float]],
    df: pl.DataFrame,
    zone_id: str
) -> pl.DataFrame:
    """
    Fetch detailed route geometry and timing for all optimized routes.
    
    This is substage 3.5 where we:
    1. Get turn-by-turn directions from OSRM
    2. Extract route geometry polylines
    3. Calculate accurate segment times
    4. Build final itinerary DataFrame
    
    :param optimized_routes: Dictionary mapping days to (route, cost) tuples
    :param df: Location DataFrame
    :param zone_id: Zone identifier
    :return: DataFrame with detailed route information
    """
    logger.info(f"Stage 3.5: DETAILED ROUTING - Zone {zone_id}")
    logger.info("Getting detailed routes and timing for all days")
    
    detailed_routes = []
    
    for day, (route, _) in optimized_routes.items():
        if not route:
            continue
            
        logger.info(f"Fetching detailed route for zone {zone_id}, day {day} with {len(route)} locations")
        
        # get route geometry and timing from OSRM
        route_details = fetch_detailed_route_osrm(route, df, zone_id, day)
        
        if route_details:
            # build route record
            route_record = build_route_record(
                zone_id=zone_id,
                day=day,
                route=route,
                route_details=route_details,
                df=df
            )
            
            detailed_routes.append(route_record)
            logger.info(f"Zone {zone_id}, day {day}: {len(route_details.get('geometry', []))} route points, "
                       f"{route_details.get('total_duration', 0):.1f} min total")
    
    # convert to DataFrame
    if detailed_routes:
        result_df = pl.DataFrame(detailed_routes)
        logger.success(f"Detailed routes completed for {len(detailed_routes)} days")
        return result_df
    else:
        # return empty DataFrame with expected schema
        return create_empty_route_dataframe()


def fetch_detailed_route_osrm(
    route: List[int],
    df: pl.DataFrame,
    zone_id: str,
    day: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed route information from OSRM Route API.
    
    :param route: List of location IDs in visit order
    :param df: Location DataFrame
    :param zone_id: Zone identifier
    :param day: Day number
    :return: Dictionary with route geometry and timing details
    """
    try:
        # convert location IDs to Location objects for OSRM
        from src.utils.osrm_utils import Location
        
        route_locations = []
        for loc_id in route:
            if loc_id == -1:  # centroid
                # create a dummy location for centroid 
                route_locations.append(Location(
                    location_id=-1,
                    latitude=0.0,  # will be set by OSRM
                    longitude=0.0, # will be set by OSRM
                    name="zone_centroid",
                    address="zone_centroid"
                ))
            else:
                # find location in DataFrame
                loc_row = df.filter(pl.col("pos_id") == loc_id)
                if len(loc_row) > 0:
                    row_data = loc_row.row(0, named=True)
                    route_locations.append(Location(
                        location_id=loc_id,
                        latitude=row_data["latitude"],
                        longitude=row_data["longitude"],
                        name=row_data.get("name", f"Location {loc_id}"),
                        address=row_data.get("address", "")
                    ))
        
        # call OSRM function with correct signature
        geometry_result = fetch_route_geometry(
            zone_id=zone_id,
            day_number=day,
            route_locations=route_locations,
            include_steps=True,
            include_alternatives=False
        )
        
        # convert to expected format
        return {
            'geometry': geometry_result.encoded_polyline if hasattr(geometry_result, 'encoded_polyline') else [],
            'total_duration': geometry_result.total_duration_minutes if hasattr(geometry_result, 'total_duration_minutes') else 0.0,
            'legs': geometry_result.route_legs if hasattr(geometry_result, 'route_legs') else [],
            'centroid_coords': [0.0, 0.0]  # placeholder
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch route geometry for zone {zone_id}, day {day}: {e}")
        return None


def build_route_record(
    zone_id: str,
    day: int,
    route: List[int],
    route_details: Dict[str, Any],
    df: pl.DataFrame
) -> Dict[str, Any]:
    """
    Build a route record for the final DataFrame.
    
    :param zone_id: Zone identifier
    :param day: Day number
    :param route: List of location IDs
    :param route_details: OSRM route details
    :param df: Location DataFrame
    :return: Dictionary representing a route record
    """
    # get location information
    locations = []
    pos_classes = []
    
    for pos_id in route:
        if pos_id == -1:  # centroid
            locations.append(route_details.get('centroid_coords', [0.0, 0.0]))
            pos_classes.append("centroid")
        else:
            loc_row = df.filter(pl.col("pos_id") == pos_id)
            if len(loc_row) > 0:
                row_data = loc_row.row(0, named=True)
                locations.append([row_data["longitude"], row_data["latitude"]])
                pos_classes.append(row_data["class"])
            else:
                locations.append([0.0, 0.0])
                pos_classes.append("unknown")
    
    # extract timing information
    geometry = route_details.get('geometry', [])
    total_duration = route_details.get('total_duration', 0.0)
    
    # build schedule (cumulative times)
    schedule = build_schedule(route_details.get('legs', []), len(route))
    
    return {
        'zone_id': zone_id,
        'day': day,
        'pos_id': route,
        'pos_locations': locations,
        'pos_class': pos_classes,
        'route': geometry,
        'schedule': schedule,
        'duration': total_duration
    }


def build_schedule(legs: List[Dict], num_locations: int) -> List[float]:
    """
    Build cumulative timing schedule from OSRM leg information.
    
    :param legs: OSRM route legs
    :param num_locations: Number of locations in route
    :return: List of cumulative times in minutes
    """
    schedule = [0.0]  # start at time 0
    cumulative_time = 0.0
    
    for i, leg in enumerate(legs):
        duration_seconds = leg.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        cumulative_time += duration_minutes
        schedule.append(cumulative_time)
    
    # ensure we have the right number of entries
    while len(schedule) < num_locations:
        schedule.append(schedule[-1])
    
    return schedule[:num_locations]


def create_empty_route_dataframe() -> pl.DataFrame:
    """
    Create empty DataFrame with expected route schema.
    
    :return: Empty DataFrame with route columns
    """
    return pl.DataFrame({
        'zone_id': pl.Series([], dtype=pl.Utf8),
        'day': pl.Series([], dtype=pl.Int64),
        'pos_id': pl.Series([], dtype=pl.List(pl.Int64)),
        'pos_locations': pl.Series([], dtype=pl.List(pl.List(pl.Float64))),
        'pos_class': pl.Series([], dtype=pl.List(pl.Utf8)),
        'route': pl.Series([], dtype=pl.List(pl.List(pl.Float64))),
        'schedule': pl.Series([], dtype=pl.List(pl.Float64)),
        'duration': pl.Series([], dtype=pl.Float64)
    })