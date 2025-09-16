"""
Stage 3.5: Detailed Routing
Add detailed action sequences with driving times between locations
"""

# standard library imports
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
from src.utils.osrm_utils import fetch_route_geometry


def add_detailed_action_sequences(
    itinerary: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """
    Stage 3.5: Add detailed action sequences with driving times between locations.

    Logic:
    1. For each day, process locations in route_order sequence
    2. Start with "departing" from centroid (duration: 0)
    3. Add "driving" action (pos_id=null, pos_name=null, pos_class=null) with od_matrix duration
    4. Add "arriving" at location with original location duration
    5. Add "departing" from location (duration: 0)
    6. Repeat for next location
    7. End with final "departing" from last location

    :param itinerary: Itinerary DataFrame from stage 3.3 route optimization
    :param od_matrix: Distance matrix for driving times between locations
    :return: Updated itinerary with detailed action sequences
    """
    logger.info("Stage 3.5: DETAILED ROUTING")
    logger.info("Adding detailed action sequences with driving times")

    # Get zone_id and metadata from itinerary
    zone_id = itinerary.select("zone_id").unique().to_series().to_list()[0]
    sample_record = itinerary.row(0, named=True)
    clusterer_name = sample_record["clusterer"]
    router_name = sample_record["router"]
    balancer_name = sample_record["balancer"]
    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Group by day and process each day's route
    days = itinerary.select("day").unique().sort("day").to_series().to_list()
    detailed_records = []

    for day in days:
        # Get locations for this day in route_order sequence
        day_data = itinerary.filter(pl.col("day") == day).sort("route_order")
        day_locations = []

        for row in day_data.iter_rows(named=True):
            day_locations.append({
                "pos_id": row["pos_id"],
                "pos_name": row["pos_name"],
                "pos_class": row["pos_class"],
                "route": row["route"],
                "duration": row["duration"],
                "route_order": row["route_order"]
            })

        logger.info(f"Processing day {day}: {len(day_locations)} locations")

        # Process each location in sequence to create full action flow
        for i, location in enumerate(day_locations):
            current_pos_id = location["pos_id"]

            # Convert null pos_id for centroid to -1 for od_matrix lookup
            current_lookup_id = -1 if current_pos_id is None else int(current_pos_id)

            # Add "departing" action from current location
            departing_record = {
                "zone_id": zone_id,
                "day": day,
                "pos_id": location["pos_id"],
                "pos_name": location["pos_name"],
                "pos_class": location["pos_class"],
                "route": location["route"],
                "action": "departing",
                "schedule": None,
                "duration": 0.0,  # departing actions have 0 duration
                "route_order": location["route_order"] * 10,  # multiply by 10 to give room for intermediate actions
                "clusterer": clusterer_name,
                "router": router_name,
                "balancer": balancer_name,
                "created_on": created_on
            }
            detailed_records.append(departing_record)
            logger.debug(f"Added departing record for {location['pos_id']} with route_order {departing_record['route_order']}")

            # If this is not the last location, add driving and arriving actions
            if i < len(day_locations) - 1:
                next_location = day_locations[i + 1]
                next_pos_id = next_location["pos_id"]

                # Convert null pos_id for centroid to -1 for od_matrix lookup
                next_lookup_id = -1 if next_pos_id is None else int(next_pos_id)

                # Get driving time from od_matrix
                driving_time = od_matrix.get((current_lookup_id, next_lookup_id), 0.0)

                # Add "driving" action
                driving_record = {
                    "zone_id": zone_id,
                    "day": day,
                    "pos_id": None,  # driving actions have null pos_id
                    "pos_name": None,  # driving actions have null pos_name
                    "pos_class": None,  # driving actions have null pos_class
                    "route": [next_location["route"][0]],  # route to next location
                    "action": "driving",
                    "schedule": None,
                    "duration": driving_time,
                    "route_order": location["route_order"] * 10 + 1,  # between departing and arriving
                    "clusterer": clusterer_name,
                    "router": router_name,
                    "balancer": balancer_name,
                    "created_on": created_on
                }
                detailed_records.append(driving_record)
                logger.debug(f"Added driving record from {current_pos_id} to {next_pos_id} with route_order {driving_record['route_order']} and duration {driving_time}")

                # Add "arriving" action at next location
                arriving_record = {
                    "zone_id": zone_id,
                    "day": day,
                    "pos_id": next_location["pos_id"],
                    "pos_name": next_location["pos_name"],
                    "pos_class": next_location["pos_class"],
                    "route": next_location["route"],
                    "action": "arriving",
                    "schedule": None,
                    "duration": next_location["duration"],  # use original location duration
                    "route_order": next_location["route_order"] * 10 - 1,  # just before the departing from next location
                    "clusterer": clusterer_name,
                    "router": router_name,
                    "balancer": balancer_name,
                    "created_on": created_on
                }
                detailed_records.append(arriving_record)
                logger.debug(f"Added arriving record for {next_location['pos_id']} with route_order {arriving_record['route_order']}")

    # Create new itinerary DataFrame with detailed actions
    if detailed_records:
        detailed_itinerary = pl.DataFrame(detailed_records, schema=itinerary.schema)
        # Sort by zone_id, day, route_order to maintain proper sequence
        detailed_itinerary = detailed_itinerary.sort(["zone_id", "day", "route_order"])
    else:
        detailed_itinerary = pl.DataFrame(schema=itinerary.schema)

    logger.success(f"Detailed routing complete: {len(detailed_records)} action records created")
    logger.debug(f"Detailed records breakdown: {len([r for r in detailed_records if r['action'] == 'departing'])} departing, {len([r for r in detailed_records if r['action'] == 'driving'])} driving, {len([r for r in detailed_records if r['action'] == 'arriving'])} arriving")

    return detailed_itinerary


def get_detailed_route_information(
    optimized_routes: Dict[int, Tuple[List[int], float]],
    df: pl.DataFrame,
    zone_id: str,
    centroid: Tuple[float, float]
) -> pl.DataFrame:
    """
    Fetch detailed route geometry and timing for all optimized routes.

    This is substage 3.5 where we:
    1. Get turn-by-turn directions from OSRM
    2. Extract route geometry polylines
    3. Calculate accurate segment times
    4. Build individual position records for new itinerary format

    :param optimized_routes: Dictionary mapping days to (route, cost) tuples
    :param df: Location DataFrame
    :param zone_id: Zone identifier
    :param centroid: Zone centroid coordinates (lat, lon)
    :return: DataFrame with individual position records
    """
    logger.info(f"Stage 3.5: DETAILED ROUTING - Zone {zone_id}")
    logger.info("Getting detailed routes and timing for all days")

    detailed_records = []

    for day, (route, _) in optimized_routes.items():
        if not route:
            continue

        logger.info(f"Fetching detailed route for zone {zone_id}, day {day} with {len(route)} locations: {route}")

        # get route geometry and timing from OSRM
        route_details = fetch_detailed_route_osrm(route, df, zone_id, day, centroid)

        if route_details:
            # build individual position records
            position_records = build_position_records(
                zone_id=zone_id,
                day=day,
                route=route,
                route_details=route_details,
                df=df,
                centroid=centroid
            )

            detailed_records.extend(position_records)

            logger.info(f"Zone {zone_id}, day {day}: {len(position_records)} position records, "
                       f"{route_details.get('total_duration', 0):.1f} min total")

    # convert to DataFrame
    if detailed_records:
        result_df = pl.DataFrame(detailed_records)
        logger.success(f"Detailed routes completed with {len(detailed_records)} position records")
        return result_df
    else:
        # return empty DataFrame with expected schema
        return create_empty_itinerary_dataframe()


def fetch_detailed_route_osrm(
    route: List[int],
    df: pl.DataFrame,
    zone_id: str,
    day: int,
    centroid: Tuple[float, float]
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
                # create a location for centroid using actual coordinates
                route_locations.append(Location(
                    location_id=-1,
                    latitude=centroid[0],  # actual centroid latitude
                    longitude=centroid[1], # actual centroid longitude
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
        # Note: we need to make a direct OSRM API call to get the legs data properly
        # The RouteGeometry object doesn't include the raw legs needed for schedule building
        coordinates = []
        for loc in route_locations:
            coordinates.append(f"{loc.longitude},{loc.latitude}")
        coordinate_string = ";".join(coordinates)

        # Make direct OSRM route API call to get legs
        import requests
        from src.utils.osrm_utils import OSRM_BASE_URL

        endpoint = f"{OSRM_BASE_URL}/route/v1/driving/{coordinate_string}"
        params = {"overview": "full", "geometries": "polyline", "steps": "true"}

        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        route_data = response.json()

        if route_data.get("code") != "Ok":
            raise ValueError(f"OSRM API returned error: {route_data.get('message', 'Unknown error')}")

        routes = route_data.get("routes", [])
        if not routes:
            raise ValueError("No routes returned from OSRM API")

        route = routes[0]
        legs = route.get("legs", [])

        # Decode the polyline to coordinate arrays
        import polyline
        encoded_polyline = route.get("geometry", "")
        if encoded_polyline:
            decoded_coords = polyline.decode(encoded_polyline)
            # Convert from [(lat, lon), ...] to [[lon, lat], ...] format
            route_coords = [[coord[1], coord[0]] for coord in decoded_coords]
        else:
            route_coords = []

        return {
            'geometry': route_coords,  # decoded polyline as coordinate arrays
            'total_duration': route.get("duration", 0.0) / 60.0,  # convert seconds to minutes
            'legs': legs,  # actual OSRM legs data for schedule building
            'centroid_coords': [0.0, 0.0]  # placeholder
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch route geometry for zone {zone_id}, day {day}: {e}")
        return None


def build_position_records(
    zone_id: str,
    day: int,
    route: List[int],
    route_details: Dict[str, Any],
    df: pl.DataFrame,
    centroid: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """
    Build individual position records for the new itinerary format.

    :param zone_id: Zone identifier
    :param day: Day number
    :param route: List of location IDs
    :param route_details: OSRM route details
    :param df: Location DataFrame
    :param centroid: Zone centroid coordinates (lat, lon)
    :return: List of individual position records
    """
    records = []

    # get full route geometry and timing
    full_geometry = route_details.get('geometry', [])
    legs = route_details.get('legs', [])

    # build cumulative schedule for arrival/departure times
    cumulative_times = build_schedule(legs, len(route))

    # create individual records for each position
    for i, pos_id in enumerate(route):
        # determine position class, coordinates, and name
        if pos_id == -1:  # centroid
            pos_class = "centroid"
            pos_name = None  # centroid has no name
            pos_coords = [centroid[1], centroid[0]]  # [lon, lat]
        else:
            loc_row = df.filter(pl.col("pos_id") == pos_id)
            if len(loc_row) > 0:
                row_data = loc_row.row(0, named=True)
                pos_class = row_data["class"]
                pos_name = row_data["name"]
                pos_coords = [row_data["longitude"], row_data["latitude"]]
            else:
                pos_class = "unknown"
                pos_name = "Unknown Location"
                pos_coords = [0.0, 0.0]

        # get route segment for this position
        if i < len(legs):
            # extract geometry segment from leg
            route_segment = extract_route_segment(full_geometry, legs, i)
            # calculate duration from leg data
            leg_duration_minutes = legs[i].get('duration', 0) / 60.0
        else:
            route_segment = [pos_coords]  # last position gets its coordinates
            leg_duration_minutes = 0.0

        # start of day: driving from centroid
        if i == 0:
            records.append({
                'zone_id': zone_id,
                'day': day,
                'pos_id': None,  # centroid gets None
                'pos_name': pos_name,  # centroid name is None
                'pos_class': pos_class,
                'route': [pos_coords],  # single coordinate pair for centroid
                'action': 'driving',
                'schedule': 0.0,
                'duration': leg_duration_minutes
            })

        # arriving at each location (except first centroid)
        if i > 0:
            arrival_time = cumulative_times[i]
            departure_time = arrival_time + 60.0  # depart after 60 minutes at location

            records.append({
                'zone_id': zone_id,
                'day': day,
                'pos_id': str(pos_id),
                'pos_name': pos_name,
                'pos_class': pos_class,
                'route': [pos_coords],  # arrival point
                'action': 'arriving',
                'schedule': arrival_time,
                'duration': 60.0  # standard 60 minutes at location
            })

            # departing from each location (all locations have departing action)
            records.append({
                'zone_id': zone_id,
                'day': day,
                'pos_id': str(pos_id),
                'pos_name': pos_name,
                'pos_class': pos_class,
                'route': [pos_coords],  # departure point
                'action': 'departing',
                'schedule': departure_time,
                'duration': 0.0  # departing duration is always 0
            })

            # add driving row after departing (except for last location)
            if i < len(route) - 1:
                next_leg_duration = legs[i].get('duration', 0) / 60.0 if i < len(legs) else 0.0

                records.append({
                    'zone_id': zone_id,
                    'day': day,
                    'pos_id': str(pos_id),
                    'pos_name': pos_name,
                    'pos_class': pos_class,
                    'route': [pos_coords],  # single coordinate pair for departure location
                    'action': 'driving',
                    'schedule': departure_time,  # same time as departing
                    'duration': next_leg_duration
                })

    return records


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


def extract_route_segment(full_geometry: List[List[float]], legs: List[Dict], leg_index: int) -> List[List[float]]:
    """
    Extract route geometry segment for a specific leg.

    :param full_geometry: Complete route geometry
    :param legs: OSRM route legs
    :param leg_index: Index of the leg to extract
    :return: Route segment coordinates
    """
    if leg_index >= len(legs) or not full_geometry:
        return []

    # For now, return full geometry for each segment
    # In a more sophisticated implementation, we would parse the leg steps
    # to extract the specific segment geometry
    return full_geometry


def generate_daily_summary(
    itinerary_df: pl.DataFrame,
    clusterer: str,
    balancer: str,
    router: str
) -> pl.DataFrame:
    """
    Generate daily summary table from individual itinerary records.
    Aggregates position records into daily metrics.

    :param itinerary_df: DataFrame with individual position records
    :param clusterer: Clustering algorithm name
    :param balancer: Balancing algorithm name
    :param router: Routing algorithm name
    :return: DataFrame with daily summary records
    """
    if len(itinerary_df) == 0:
        return create_empty_daily_summary_dataframe()

    logger.info("Generating daily summary from individual position records")

    # Group by zone_id and day to calculate daily metrics
    summary_data = []
    from datetime import datetime
    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get unique combinations of zone_id and day
    groups = itinerary_df.select(['zone_id', 'day']).unique()

    for group_row in groups.iter_rows(named=True):
        zone_id = group_row['zone_id']
        day = group_row['day']

        # Filter to this specific day and zone
        day_data = itinerary_df.filter(
            (pl.col('zone_id') == zone_id) &
            (pl.col('day') == day)
        )

        # Count distinct locations by class (excluding null pos_ids)
        pos_data = day_data.filter(pl.col('pos_id').is_not_null())

        primary_locations = pos_data.filter(pl.col('pos_class') == 'primary').select('pos_id').n_unique()
        secondary_locations = pos_data.filter(pl.col('pos_class') == 'secondary').select('pos_id').n_unique()

        # Calculate POS time (sum of duration for all pos activities)
        total_pos_time = pos_data.select('duration').sum().item() or 0.0

        # Calculate drive time (sum of duration for driving actions)
        driving_data = day_data.filter(pl.col('action') == 'driving')
        total_drive_time = driving_data.select('duration').sum().item() or 0.0

        # Calculate total duration
        # For early pipeline stages (no schedule/action), use sum of all durations
        # For later pipeline stages (with schedule), use schedule range
        schedule_col = day_data.select('schedule')
        min_time = schedule_col.min().item()
        max_time = schedule_col.max().item()

        if min_time is not None and max_time is not None and len(day_data) > 1:
            duration = max_time - min_time  # in minutes
        else:
            # If schedules are null, use sum of pos time + drive time
            duration = total_pos_time + total_drive_time

        # Calculate utilization percentage (assuming 8 hours = 480 minutes per day)
        hours_per_day = 8 * 60  # 480 minutes
        utilization_percentage = (duration / hours_per_day * 100) if hours_per_day > 0 else 0.0

        summary_data.append({
            'zone_id': zone_id,
            'day': day,
            'primary_locations': primary_locations,
            'secondary_locations': secondary_locations,
            'duration': duration,
            'utilization_percentage': utilization_percentage,
            'total_pos_time': total_pos_time,
            'total_drive_time': total_drive_time,
            'clusterer': clusterer,
            'router': router,
            'balancer': balancer,
            'created_on': created_on
        })

    if summary_data:
        result_df = pl.DataFrame(summary_data)
        logger.info(f"Generated {len(result_df)} daily summary records")
        return result_df
    else:
        return create_empty_daily_summary_dataframe()


def create_empty_itinerary_dataframe() -> pl.DataFrame:
    """
    Create empty DataFrame with new itinerary schema.

    :return: Empty DataFrame with individual position record columns
    """
    return pl.DataFrame({
        'zone_id': pl.Series([], dtype=pl.Utf8),
        'day': pl.Series([], dtype=pl.Int64),
        'pos_id': pl.Series([], dtype=pl.Utf8),  # now string, nullable for centroid
        'pos_name': pl.Series([], dtype=pl.Utf8),  # name from locations table, nullable for centroid
        'pos_class': pl.Series([], dtype=pl.Utf8),  # individual string
        'route': pl.Series([], dtype=pl.List(pl.List(pl.Float64))),
        'action': pl.Series([], dtype=pl.Utf8),  # driving/arriving/departing
        'schedule': pl.Series([], dtype=pl.Float64),  # individual float time
        'duration': pl.Series([], dtype=pl.Float64)  # duration of each step
    })


def create_empty_daily_summary_dataframe() -> pl.DataFrame:
    """
    Create empty DataFrame with daily summary schema.

    :return: Empty DataFrame with daily summary columns
    """
    return pl.DataFrame({
        'zone_id': pl.Series([], dtype=pl.Utf8),
        'day': pl.Series([], dtype=pl.Int64),
        'primary_locations': pl.Series([], dtype=pl.Int64),
        'secondary_locations': pl.Series([], dtype=pl.Int64),
        'duration': pl.Series([], dtype=pl.Float64),
        'utilization_percentage': pl.Series([], dtype=pl.Float64),
        'total_pos_time': pl.Series([], dtype=pl.Float64),
        'total_drive_time': pl.Series([], dtype=pl.Float64),
        'clusterer': pl.Series([], dtype=pl.Utf8),
        'router': pl.Series([], dtype=pl.Utf8),
        'balancer': pl.Series([], dtype=pl.Utf8),
        'created_on': pl.Series([], dtype=pl.Utf8)
    })