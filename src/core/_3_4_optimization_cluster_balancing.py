"""
Stage 3.4: Cluster Balancing
Balance workloads across clusters using organic duration rebalancing
"""

# standard library imports
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import numpy as np
import polars as pl

# local imports
from src.utils.clustering_utils import haversine_distance


def balance_cluster_workloads(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    duration_threshold_min: float = 60.0,
    max_iterations: int = 5
) -> Dict[int, List[int]]:
    """
    Balance workloads across clusters using organic duration rebalancing.
    
    This is substage 3.4 where we:
    1. Calculate duration per cluster
    2. Identify imbalances
    3. Move locations to balance workloads
    4. Iterate until convergence
    
    :param day_assignments: Initial day assignments
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param duration_threshold_min: Threshold for rebalancing
    :param max_iterations: Maximum rebalancing iterations
    :return: Balanced day assignments
    """
    logger.info(f"Stage 3.4: CLUSTER BALANCING - Zone {zone_id}")
    logger.info(f"Starting organic duration rebalancing (threshold: {duration_threshold_min} min)")
    
    balanced_assignments = day_assignments.copy()
    
    for iteration in range(max_iterations):
        # calculate current cluster durations
        cluster_durations = calculate_cluster_durations(
            balanced_assignments, df, od_matrix
        )
        
        if not cluster_durations:
            break
            
        # check if rebalancing is needed
        min_duration = min(cluster_durations.values())
        max_duration = max(cluster_durations.values())
        duration_gap = max_duration - min_duration
        
        logger.info(f"Iteration {iteration + 1}: Duration gap = {duration_gap:.1f} min "
                   f"(max: {max_duration:.1f}, min: {min_duration:.1f})")
        
        if duration_gap <= duration_threshold_min:
            logger.info(f"Converged! Duration gap {duration_gap:.1f} <= {duration_threshold_min} threshold")
            break
        
        # find clusters to balance
        overloaded_day = max(cluster_durations, key=cluster_durations.get)
        underloaded_day = min(cluster_durations, key=cluster_durations.get)
        
        # move location from overloaded to underloaded
        moved = move_location_between_clusters(
            balanced_assignments, 
            overloaded_day, 
            underloaded_day,
            df, 
            od_matrix
        )
        
        if not moved:
            logger.info("No beneficial moves found, stopping rebalancing")
            break
            
        logger.info(f"Moved location from day {overloaded_day} to day {underloaded_day}")
    
    logger.success("Organic duration rebalancing completed")
    
    return balanced_assignments


def calculate_cluster_durations(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    service_time_min: float = 60.0
) -> Dict[int, float]:
    """
    Calculate total duration (service + drive time) for each cluster.
    
    :param day_assignments: Day assignments
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :param service_time_min: Service time per location
    :return: Dictionary mapping days to total durations
    """
    cluster_durations = {}
    
    for day, location_ids in day_assignments.items():
        if not location_ids:
            continue
            
        # service time
        service_duration = len(location_ids) * service_time_min
        
        # drive time (approximate using pairwise distances)
        drive_duration = 0.0
        for i in range(len(location_ids)):
            for j in range(i + 1, len(location_ids)):
                loc1, loc2 = location_ids[i], location_ids[j]
                drive_duration += od_matrix.get((loc1, loc2), 0.0)
        
        # average drive time for TSP approximation
        if len(location_ids) > 1:
            drive_duration = drive_duration / len(location_ids)
        
        total_duration = service_duration + drive_duration
        cluster_durations[day] = total_duration
    
    return cluster_durations


def move_location_between_clusters(
    day_assignments: Dict[int, List[int]],
    from_day: int,
    to_day: int,
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float]
) -> bool:
    """
    Move one location from overloaded to underloaded cluster.
    
    :param day_assignments: Current assignments
    :param from_day: Source day (overloaded)
    :param to_day: Target day (underloaded)  
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :return: True if a move was made, False otherwise
    """
    from_locations = day_assignments.get(from_day, [])
    to_locations = day_assignments.get(to_day, [])
    
    if not from_locations:
        return False
    
    # calculate centroids
    from_centroid = calculate_cluster_centroid(from_locations, df)
    to_centroid = calculate_cluster_centroid(to_locations, df) if to_locations else from_centroid
    
    # find location closest to target centroid
    best_location = None
    best_distance = float('inf')
    
    for loc_id in from_locations:
        loc_row = df.filter(pl.col("pos_id") == loc_id).row(0, named=True)
        loc_coords = (loc_row["latitude"], loc_row["longitude"])
        
        distance = haversine_distance(loc_coords[0], loc_coords[1], to_centroid[0], to_centroid[1])
        
        if distance < best_distance:
            best_distance = distance
            best_location = loc_id
    
    if best_location is not None:
        # move the location
        day_assignments[from_day].remove(best_location)
        if to_day not in day_assignments:
            day_assignments[to_day] = []
        day_assignments[to_day].append(best_location)
        
        logger.info(f"Moved location {best_location} from day {from_day} to day {to_day} "
                   f"(distance improvement: {best_distance:.1f} km)")
        return True
    
    return False


def calculate_cluster_centroid(location_ids: List[int], df: pl.DataFrame) -> Tuple[float, float]:
    """
    Calculate centroid for a cluster of locations.
    
    :param location_ids: List of location IDs
    :param df: Location DataFrame
    :return: Centroid coordinates (lat, lon)
    """
    if not location_ids:
        return (0.0, 0.0)
    
    cluster_df = df.filter(pl.col("pos_id").is_in(location_ids))
    
    if len(cluster_df) == 0:
        return (0.0, 0.0)
    
    avg_lat = cluster_df["latitude"].mean()
    avg_lon = cluster_df["longitude"].mean()
    
    return (avg_lat, avg_lon)