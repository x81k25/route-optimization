# standard library imports
import yaml
from pathlib import Path
from typing import Any, Dict, List

# 3rd-party imports
import numpy as np
import polars as pl
import polyline
from loguru import logger
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# ------------------------------------------------------------------------------
# supporting functions  
# ------------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """
    Load configuration from model-params.yaml file.
    
    :return: configuration dictionary
    """
    config_path = Path("config/model-params.yaml")
    if not config_path.exists():
        logger.warning(f"config file not found at {config_path}, using defaults")
        return {"model_params": {"days_per_week": 5}}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get("model_params", {"days_per_week": 5})

# ------------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------------

def assign_anchor_days(
    pos: pl.DataFrame
) -> pl.DataFrame:
    """
    Assign locations to anchor days with primary locations taking full days.
    
    :param pos: DataFrame containing all position/location data for a single zone
    :return: itinerary DataFrame with zone_id, day, and pos_id columns
    """
    # load config
    config = load_config()
    days_per_week = config.get("days_per_week", 5)
    hours_per_day = config.get("hours_per_day", 8)
    primary_hours_per_week = config.get("primary_hours_per_week", 24)
    
    # get zone information
    zone_ids = pos['zone_id'].unique().to_list()
    if len(zone_ids) != 1:
        raise ValueError(f"expected single zone, got {len(zone_ids)} zones: {zone_ids}")
    
    zone_id = zone_ids[0]
    logger.info(f"assigning anchor days for zone: {zone_id} ({days_per_week} days per week)")
    logger.info(f"config: {hours_per_day}h/day, {primary_hours_per_week}h/week primary")
    
    # separate primary and secondary locations
    primary_df = pos.filter(pl.col('class') == 'primary')
    secondary_df = pos.filter(pl.col('class') == 'secondary')
    
    logger.info(f"found {len(primary_df)} primary, {len(secondary_df)} secondary locations")
    
    # create itinerary DataFrame structure
    itinerary_data = []
    
    # initialize all days as empty
    for day in range(1, days_per_week + 1):
        itinerary_data.append({
            'zone_id': zone_id,
            'day': day,
            'pos_id': [],
            'pos_locations': [],
            'pos_duration': [],
            'pos_class': []
        })
    
    # assign primary locations based on hours distribution
    primary_locations = primary_df.to_dicts()
    
    if len(primary_locations) > 0:
        # calculate hours per primary location
        hours_per_primary = primary_hours_per_week / len(primary_locations)
        logger.info(f"distributing {hours_per_primary:.1f} hours per primary location")
        
        current_day = 1
        remaining_hours_today = hours_per_day
        
        for location in primary_locations:
            pos_id = location['pos_id']
            longitude = location['longitude'] 
            latitude = location['latitude']
            
            hours_left_for_location = hours_per_primary
            
            # distribute this location's hours across days
            while hours_left_for_location > 0 and current_day <= days_per_week:
                # hours to assign to current day
                hours_for_today = min(hours_left_for_location, remaining_hours_today)
                minutes_for_today = int(hours_for_today * 60)
                
                # add/update current day's assignment
                day_idx = current_day - 1  # convert to 0-based index
                
                # add this location to today's schedule
                itinerary_data[day_idx]['pos_id'].append(pos_id)
                itinerary_data[day_idx]['pos_locations'].append([longitude, latitude])
                itinerary_data[day_idx]['pos_duration'].append(minutes_for_today)
                itinerary_data[day_idx]['pos_class'].append('primary')
                
                # update counters
                hours_left_for_location -= hours_for_today
                remaining_hours_today -= hours_for_today
                
                # move to next day if current day is full
                if remaining_hours_today <= 0:
                    current_day += 1
                    remaining_hours_today = hours_per_day
    
    # create polars DataFrame
    itinerary_df = pl.DataFrame(itinerary_data)
    
    logger.success(f"anchor days assigned for zone {zone_id}: {len(itinerary_df)} days created")
    return itinerary_df


def cluster_secondary_pos(
    pos: pl.DataFrame,
    itinerary: pl.DataFrame,
    od_matrix: pl.DataFrame
) -> pl.DataFrame:
    """
    Cluster secondary locations and assign them to available days in the itinerary.
    
    Args:
        pos: DataFrame containing all position/location data for a single zone
        itinerary: DataFrame with zone_id, day, pos_id, pos_locations, pos_duration columns
        od_matrix: DataFrame with OD matrix data (origin_id, destination_id, duration_seconds, etc.)
        
    Returns:
        Updated itinerary DataFrame with secondary locations assigned to available days
    """
    from ..utils.clustering_utils import haversine_distance
    
    # Load config
    config = load_config()
    hours_per_non_primary = config.get("hours_per_non_primary", 1)
    locations_per_day_max = config.get("locations_per_day_max", 7)
    
    # Get zone information
    zone_ids = pos['zone_id'].unique().to_list()
    if len(zone_ids) != 1:
        raise ValueError(f"expected single zone, got {len(zone_ids)} zones: {zone_ids}")
    zone_id = zone_ids[0]
    
    # Separate secondary locations
    secondary_df = pos.filter(pl.col('class') == 'secondary')
    
    if secondary_df.is_empty():
        logger.info(f"No secondary locations found for zone {zone_id}")
        return itinerary
    
    logger.info(f"Clustering {len(secondary_df)} secondary locations for zone {zone_id}")
    
    # Find available days (days with empty pos_id lists)
    available_days = []
    for row in itinerary.iter_rows(named=True):
        if not row['pos_id']:  # empty list
            available_days.append(row['day'])
    
    n_available_days = len(available_days)
    if n_available_days == 0:
        logger.warning(f"No available days for secondary locations in zone {zone_id}")
        return itinerary
    
    logger.info(f"Found {n_available_days} available days for secondary clustering")
    
    # Extract secondary location IDs and coordinates
    secondary_ids = secondary_df['pos_id'].to_list()
    secondary_coords = secondary_df.select(['latitude', 'longitude']).to_numpy()
    
    # Build distance matrix using haversine distance (in kilometers)
    distance_matrix = np.zeros((len(secondary_ids), len(secondary_ids)))
    
    for i in range(len(secondary_ids)):
        for j in range(len(secondary_ids)):
            if i != j:
                # Calculate haversine distance between locations
                distance_km = haversine_distance(
                    secondary_coords[i, 0], secondary_coords[i, 1],  # lat1, lon1
                    secondary_coords[j, 0], secondary_coords[j, 1]   # lat2, lon2
                )
                distance_matrix[i, j] = distance_km
    
    # Perform K-means clustering
    clusters = _cluster_locations_kmeans(
        secondary_coords, secondary_ids, n_available_days, locations_per_day_max
    )
    
    # Assign clusters to available days and update itinerary
    updated_itinerary = itinerary.clone()
    
    for cluster_id, (day, location_ids) in enumerate(zip(available_days, clusters.values())):
        if not location_ids:
            continue
            
        # Get location coordinates for these pos_ids
        cluster_locations = secondary_df.filter(pl.col('pos_id').is_in(location_ids))
        
        pos_locations = []
        durations = []
        pos_classes = []
        
        for loc_row in cluster_locations.iter_rows(named=True):
            pos_locations.append([loc_row['longitude'], loc_row['latitude']])
            durations.append(int(hours_per_non_primary * 60))  # convert hours to minutes
            pos_classes.append('secondary')
        
        # Update the itinerary for this day using direct row updates
        for i, row in enumerate(updated_itinerary.rows()):
            if row[1] == day:  # day column is at index 1
                # Create new row data
                new_row = {
                    'zone_id': row[0],
                    'day': row[1], 
                    'pos_id': location_ids,
                    'pos_locations': pos_locations,
                    'pos_duration': durations,
                    'pos_class': pos_classes
                }
                
                # Replace this row in the DataFrame
                rows_before = updated_itinerary[:i]
                rows_after = updated_itinerary[i+1:]
                new_row_df = pl.DataFrame([new_row])
                
                if len(rows_before) == 0:
                    updated_itinerary = pl.concat([new_row_df, rows_after])
                elif len(rows_after) == 0:
                    updated_itinerary = pl.concat([rows_before, new_row_df])
                else:
                    updated_itinerary = pl.concat([rows_before, new_row_df, rows_after])
                break
    
    logger.success(f"Assigned {len(secondary_df)} secondary locations to {len(clusters)} days")
    return updated_itinerary



def _cluster_locations_hierarchical(
    distance_matrix: np.ndarray, 
    location_ids: List[int], 
    n_clusters: int, 
    max_locations_per_cluster: int = 7
) -> Dict[int, List[int]]:
    """
    Cluster locations into days using hierarchical clustering.
    
    Args:
        distance_matrix: square distance matrix between locations
        location_ids: list of location IDs
        n_clusters: number of clusters (available secondary days)
        max_locations_per_cluster: maximum locations per cluster
        
    Returns:
        dictionary mapping cluster_id to list of location_ids
    """
    n_locations = len(location_ids)
    
    # Handle empty case
    if n_locations == 0:
        return {}
    
    # Handle single location case
    if n_locations == 1:
        return {1: location_ids.copy()}
    
    # Handle case where we have more clusters than locations
    if n_locations <= n_clusters:
        return {i+1: [location_ids[i]] for i in range(n_locations)}
    
    # Convert distance matrix to condensed form for scipy
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering with average linkage
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Get cluster assignments
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Group locations by cluster
    cluster_assignments = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append(location_ids[idx])
    
    # Handle constraint violations (clusters too large)
    cluster_assignments = _enforce_cluster_size_constraints(
        cluster_assignments, max_locations_per_cluster
    )
    
    return cluster_assignments


def _enforce_cluster_size_constraints(
    clusters: Dict[int, List[int]], 
    max_size: int
) -> Dict[int, List[int]]:
    """
    Split clusters that exceed size limit.
    
    Args:
        clusters: dictionary of cluster assignments
        max_size: maximum allowed cluster size
        
    Returns:
        adjusted clusters dictionary
    """
    adjusted_clusters = {}
    cluster_counter = 1
    
    for cluster_id, locations in clusters.items():
        if len(locations) <= max_size:
            # cluster is within size limit
            adjusted_clusters[cluster_counter] = locations
            cluster_counter += 1
        else:
            # split oversized cluster
            n_splits = (len(locations) + max_size - 1) // max_size  # ceiling division
            
            for i in range(n_splits):
                start_idx = i * max_size
                end_idx = min((i + 1) * max_size, len(locations))
                adjusted_clusters[cluster_counter] = locations[start_idx:end_idx]
                cluster_counter += 1
    
    return adjusted_clusters


def _cluster_locations_kmeans(
    coordinates: np.ndarray,
    location_ids: List[int], 
    n_clusters: int, 
    max_locations_per_cluster: int = 7
) -> Dict[int, List[int]]:
    """
    Cluster locations into days using K-means clustering.
    
    Args:
        coordinates: array of [lat, lon] coordinates for locations
        location_ids: list of location IDs
        n_clusters: number of clusters (available secondary days)
        max_locations_per_cluster: maximum locations per cluster
        
    Returns:
        dictionary mapping cluster_id to list of location_ids
    """
    from sklearn.cluster import KMeans
    
    n_locations = len(location_ids)
    
    # Handle empty case
    if n_locations == 0:
        return {}
    
    # Handle single location case
    if n_locations == 1:
        return {1: location_ids.copy()}
    
    # Handle case where we have more clusters than locations
    if n_locations <= n_clusters:
        return {i+1: [location_ids[i]] for i in range(n_locations)}
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Group locations by cluster
    cluster_assignments = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append(location_ids[idx])
    
    # Handle constraint violations (clusters too large)
    cluster_assignments = _enforce_cluster_size_constraints(
        cluster_assignments, max_locations_per_cluster
    )
    
    return cluster_assignments


def gen_secondary_routes(
    itinerary: pl.DataFrame,
    od_matrix: pl.DataFrame,
    centroid: tuple = None
) -> pl.DataFrame:
    """
    Generate optimized routes for days with secondary locations using TSP algorithms.
    
    Args:
        itinerary: DataFrame with zone_id, day, pos_id, pos_locations, pos_duration columns
        od_matrix: DataFrame with OD matrix data (origin_id, destination_id, duration_minutes, etc.)
        centroid: Tuple of (longitude, latitude) for zone centroid starting point
        
    Returns:
        Updated itinerary DataFrame with optimized route order for secondary location days
    """
    logger.info("Generating optimized routes for secondary locations")
    
    # Create drive time lookup from OD matrix
    drive_time_lookup = {}
    for row in od_matrix.iter_rows(named=True):
        origin_id = row['origin_id']
        dest_id = row['destination_id'] 
        drive_time = row['duration_minutes']
        drive_time_lookup[(origin_id, dest_id)] = drive_time
    
    updated_itinerary = itinerary.clone()
    
    # Process each day that has locations
    for day_row in itinerary.iter_rows(named=True):
        day = day_row['day']
        pos_ids = day_row['pos_id']
        pos_locations = day_row['pos_locations']
        pos_durations = day_row['pos_duration']
        pos_classes = day_row['pos_class']
        
        # Skip days with 0 locations (no routing needed)
        if len(pos_ids) == 0:
            continue
            
        # Prepare locations for routing with centroid as starting point
        route_pos_ids = pos_ids.copy()
        route_locations = pos_locations.copy()
        route_durations = pos_durations.copy()
        route_classes = pos_classes.copy()
        
        # Add centroid as starting point if provided and day has secondary locations
        centroid_pos_id = None
        if centroid is not None and len(pos_ids) > 0:
            # Use negative ID for centroid to avoid conflicts with actual pos_ids
            centroid_pos_id = -1
            route_pos_ids.insert(0, centroid_pos_id)
            route_locations.insert(0, [centroid[0], centroid[1]])  # [lon, lat]
            route_durations.insert(0, 0)  # 0 duration at centroid
            route_classes.insert(0, 'centroid')
            
        logger.info(f"Optimizing route for day {day} with {len(route_pos_ids)} locations (including centroid)")
        
        # Skip optimization if only centroid (no secondary locations)
        if len(route_pos_ids) <= 1:
            continue
            
        # Optimize route using appropriate algorithm with centroid as fixed start
        optimized_route, total_drive_time, metadata = _optimize_daily_route(
            location_ids=route_pos_ids,
            drive_time_lookup=drive_time_lookup,
            start_location_id=centroid_pos_id
        )
        
        logger.info(f"Day {day}: {metadata['algorithm']}, {total_drive_time:.1f} min drive time")
        
        # Reorder locations, durations, and classes to match optimized route
        optimized_locations = []
        optimized_durations = []
        optimized_classes = []
        
        # Keep optimized route including centroid as first point
        final_pos_ids = []
        
        for optimized_id in optimized_route:
            if optimized_id == centroid_pos_id:
                # Keep centroid as first point in final output
                final_pos_ids.append(optimized_id)
                optimized_locations.append([centroid[0], centroid[1]])
                optimized_durations.append(0)  # 0 duration at centroid
                optimized_classes.append('centroid')
            else:
                # Find the index of this ID in original order (before centroid was added)
                original_idx = pos_ids.index(optimized_id)
                final_pos_ids.append(optimized_id)
                optimized_locations.append(pos_locations[original_idx])
                optimized_durations.append(pos_durations[original_idx])
                optimized_classes.append(pos_classes[original_idx])
        
        # Update the itinerary for this day using direct row updates
        for i, row in enumerate(updated_itinerary.rows()):
            if row[1] == day:  # day column is at index 1
                # Create new row data
                new_row = {
                    'zone_id': row[0],
                    'day': row[1],
                    'pos_id': final_pos_ids,
                    'pos_locations': optimized_locations,
                    'pos_duration': optimized_durations,
                    'pos_class': optimized_classes
                }
                
                # Replace this row in the DataFrame
                rows_before = updated_itinerary[:i]
                rows_after = updated_itinerary[i+1:]
                new_row_df = pl.DataFrame([new_row])
                
                if len(rows_before) == 0:
                    updated_itinerary = pl.concat([new_row_df, rows_after])
                elif len(rows_after) == 0:
                    updated_itinerary = pl.concat([rows_before, new_row_df])
                else:
                    updated_itinerary = pl.concat([rows_before, new_row_df, rows_after])
                break
    
    logger.success("Route optimization completed for all secondary location days")
    return updated_itinerary


def _optimize_daily_route(
    location_ids: List[int],
    drive_time_lookup: Dict,
    start_location_id: int = None,
    use_exhaustive_if_small: bool = True
) -> tuple:
    """
    Main function to optimize a route for given locations on a single day.
    
    Args:
        location_ids: List of location IDs to visit
        drive_time_lookup: Dictionary for drive time lookups
        start_location_id: Starting location (if None, uses first location)
        use_exhaustive_if_small: Use exhaustive search for ≤ 5 locations
        
    Returns:
        Tuple of (route, total_time, metadata)
    """
    if not location_ids:
        return [], 0.0, {'algorithm': 'empty'}
    
    metadata = {
        'n_locations': len(location_ids),
        'start_location': start_location_id or location_ids[0]
    }
    
    # Choose algorithm based on problem size
    if use_exhaustive_if_small and len(location_ids) <= 5:
        # Use exhaustive search for small problems
        route, total_time = _exhaustive_search(location_ids, drive_time_lookup)
        metadata['algorithm'] = 'exhaustive_search'
        metadata['optimal'] = True
    else:
        # Use greedy + 2-opt for larger problems
        route, _ = _greedy_nearest_neighbor(location_ids, drive_time_lookup, start_location_id)
        route, total_time = _two_opt_improvement(route, drive_time_lookup)
        metadata['algorithm'] = 'greedy_plus_2opt'
        metadata['optimal'] = False
    
    metadata['total_drive_time_minutes'] = total_time
    
    return route, total_time, metadata


def _get_drive_time(origin_id: int, dest_id: int, drive_time_lookup: Dict) -> float:
    """Get drive time between two locations."""
    if origin_id == dest_id:
        return 0.0
    
    # Handle centroid (ID = -1) by using a reasonable default drive time
    if origin_id == -1 or dest_id == -1:
        # Assume 5 minutes average drive time from/to centroid
        return 5.0
    
    return drive_time_lookup.get((origin_id, dest_id), float('inf'))


def _calculate_route_time(route: List[int], drive_time_lookup: Dict) -> float:
    """Calculate total drive time for a route."""
    if len(route) <= 1:
        return 0.0
    
    total_time = 0.0
    for i in range(len(route) - 1):
        total_time += _get_drive_time(route[i], route[i + 1], drive_time_lookup)
    
    return total_time


def _greedy_nearest_neighbor(
    location_ids: List[int], 
    drive_time_lookup: Dict,
    start_location_id: int = None
) -> tuple:
    """
    Solve TSP using Greedy Nearest Neighbor algorithm.
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
            key=lambda loc: _get_drive_time(current_location, loc, drive_time_lookup)
        )
        
        drive_time = _get_drive_time(current_location, nearest_location, drive_time_lookup)
        total_time += drive_time
        
        route.append(nearest_location)
        unvisited.remove(nearest_location)
        current_location = nearest_location
    
    return route, total_time


def _two_opt_improvement(
    route: List[int], 
    drive_time_lookup: Dict,
    max_iterations: int = 100
) -> tuple:
    """
    Improve route using 2-opt local search.
    """
    if len(route) <= 3:
        return route, _calculate_route_time(route, drive_time_lookup)
    
    best_route = route.copy()
    best_time = _calculate_route_time(best_route, drive_time_lookup)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try all possible 2-opt swaps
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:  # Skip adjacent edges
                    continue
                
                # Create new route by reversing segment between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_time = _calculate_route_time(new_route, drive_time_lookup)
                
                if new_time < best_time:
                    best_route = new_route.copy()
                    best_time = new_time
                    improved = True
        
        if improved:
            route = best_route.copy()
        else:
            break  # No more improvements found
    
    return best_route, best_time


def _exhaustive_search(
    location_ids: List[int], 
    drive_time_lookup: Dict
) -> tuple:
    """
    Solve TSP using exhaustive search (brute force).
    Only use for small problems (≤ 5 locations).
    """
    import itertools
    
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
        route_time = _calculate_route_time(route, drive_time_lookup)
        
        if route_time < best_time:
            best_route = route
            best_time = route_time
    
    return best_route, best_time


def get_detailed_routes(
    itinerary: pl.DataFrame,
    od_matrix: pl.DataFrame
) -> pl.DataFrame:
    """
    Get detailed route geometry and timing for each day using OSRM Route API.
    
    Args:
        itinerary: DataFrame with optimized routes for each day
        od_matrix: DataFrame with OD matrix data for drive time lookups
        
    Returns:
        Updated itinerary DataFrame with new columns:
        - route: Array of [lon, lat] points along the route
        - schedule: Array of cumulative times at each point (float minutes)
        - duration: Total route duration for the day (float minutes)
    """
    from ..utils.osrm_utils import fetch_route_geometry, Location
    
    logger.info("Getting detailed routes and timing for all days")
    
    updated_itinerary = itinerary.clone()
    
    # Add new columns with default empty values
    route_data = []
    schedule_data = []
    duration_data = []
    
    for day_row in itinerary.iter_rows(named=True):
        zone_id = day_row['zone_id']
        day = day_row['day']
        pos_ids = day_row['pos_id']
        pos_locations = day_row['pos_locations']
        pos_durations = day_row['pos_duration']
        
        # Skip days with 0 or 1 locations (no routing needed)
        if len(pos_ids) <= 1:
            if len(pos_ids) == 1:
                # Single location - just the point and duration
                route_data.append([pos_locations[0]])
                schedule_data.append([0.0, float(pos_durations[0])])
                duration_data.append(float(pos_durations[0]))
            else:
                # No locations
                route_data.append([])
                schedule_data.append([])
                duration_data.append(0.0)
            continue
        
        logger.info(f"Fetching detailed route for zone {zone_id}, day {day} with {len(pos_ids)} locations")
        
        try:
            # Convert pos_locations to Location objects for OSRM API
            locations = []
            for i, coords in enumerate(pos_locations):
                location = Location(
                    location_id=pos_ids[i],
                    latitude=coords[1],  # lat
                    longitude=coords[0], # lon
                    name=f"Stop_{i+1}"
                )
                locations.append(location)
            
            # Fetch route geometry from OSRM
            route_geometry = fetch_route_geometry(
                zone_id=zone_id,
                day_number=day,
                route_locations=locations,
                include_steps=True
            )
            
            # Extract route points from polyline geometry
            route_points = _decode_polyline_to_points(route_geometry.geometry_polyline)
            
            # Build schedule with cumulative timing
            schedule = []
            total_duration = 0.0
            
            # Start at time 0 (at centroid)
            schedule.append(0.0)
            
            # Add drive times between locations and service times at each location
            for i in range(len(pos_ids) - 1):
                # Add service time at current location
                service_time = float(pos_durations[i])
                total_duration += service_time
                schedule.append(total_duration)
                
                # Add drive time to next location
                origin_id = pos_ids[i]
                dest_id = pos_ids[i + 1]
                
                # Handle drive time lookup (centroid uses default 5 min)
                if origin_id == -1 or dest_id == -1:
                    # Use default 5 minutes for centroid connections
                    drive_time = 5.0
                else:
                    # Look up drive time from OD matrix
                    drive_time_row = od_matrix.filter(
                        (pl.col('origin_id') == origin_id) & 
                        (pl.col('destination_id') == dest_id)
                    )
                    
                    if not drive_time_row.is_empty():
                        drive_time = float(drive_time_row['duration_minutes'].item())
                    else:
                        drive_time = 5.0  # fallback
                
                total_duration += drive_time
                schedule.append(total_duration)
            
            # Add final service time at last location
            final_service_time = float(pos_durations[-1])
            total_duration += final_service_time
            schedule.append(total_duration)
            
            route_data.append(route_points)
            schedule_data.append(schedule)
            duration_data.append(total_duration)
            
            logger.info(f"Zone {zone_id}, day {day}: {len(route_points)} route points, {total_duration:.1f} min total")
            
        except Exception as e:
            logger.error(f"Failed to get detailed route for zone {zone_id}, day {day}: {e}")
            # Fallback: use simple point-to-point route
            fallback_route = pos_locations.copy()
            fallback_schedule = [0.0] + [float(sum(pos_durations[:i+1])) for i in range(len(pos_durations))]
            fallback_duration = float(sum(pos_durations))
            
            route_data.append(fallback_route)
            schedule_data.append(fallback_schedule)
            duration_data.append(fallback_duration)
    
    # Add new columns to itinerary
    updated_itinerary = updated_itinerary.with_columns([
        pl.Series("route", route_data),
        pl.Series("schedule", schedule_data), 
        pl.Series("duration", duration_data)
    ])
    
    logger.success("Detailed routes and timing completed for all days")
    return updated_itinerary


def _decode_polyline_to_points(polyline_string: str) -> List[List[float]]:
    """
    Decode OSRM polyline string to array of [lon, lat] points.
    
    Args:
        polyline_string: Encoded polyline string from OSRM
        
    Returns:
        List of [longitude, latitude] coordinate pairs
    """
    try:
        # Decode polyline to [(lat, lon), ...] tuples
        decoded_coords = polyline.decode(polyline_string)
        
        # Convert to [[lon, lat], ...] format
        route_points = [[float(coord[1]), float(coord[0])] for coord in decoded_coords]
        
        return route_points
        
    except Exception as e:
        logger.warning(f"Failed to decode polyline: {e}")
        return []

# ------------------------------------------------------------------------------
# end of optimize.py
# ------------------------------------------------------------------------------