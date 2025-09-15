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
    secondary_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    balancer: str = "greedy",
    duration_threshold_min: float = 60.0,
    max_iterations: int = 5
) -> Dict[int, List[int]]:
    """
    Balance workloads across secondary clusters only.

    This is substage 3.4 where we:
    1. Only rebalance secondary day clusters
    2. Never touch primary store assignments
    3. Calculate duration per secondary cluster
    4. Apply selected balancing algorithm only to secondary locations

    :param secondary_assignments: Secondary cluster assignments (cluster_id -> location_ids)
    :param df: Location DataFrame
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param balancer: Balancing algorithm to use
    :param duration_threshold_min: Threshold for rebalancing
    :param max_iterations: Maximum rebalancing iterations
    :return: Balanced secondary cluster assignments
    """
    logger.info(f"Stage 3.4: CLUSTER BALANCING - Zone {zone_id}")
    logger.info(f"Starting {balancer} balancing on secondary clusters only (threshold: {duration_threshold_min} min)")

    if not secondary_assignments:
        logger.info("No secondary clusters to balance")
        return secondary_assignments

    if balancer == "greedy":
        return apply_greedy_transfer_balancing(
            secondary_assignments, df, od_matrix, zone_id, duration_threshold_min, max_iterations
        )
    elif balancer == "local_search":
        return apply_local_search_balancing(
            secondary_assignments, df, od_matrix, zone_id, duration_threshold_min, max_iterations
        )
    elif balancer == "simulated_annealing":
        return apply_simulated_annealing_balancing(
            secondary_assignments, df, od_matrix, zone_id, duration_threshold_min
        )
    elif balancer == "min_max":
        return apply_min_max_balancing(
            secondary_assignments, df, od_matrix, zone_id, duration_threshold_min, max_iterations
        )
    elif balancer == "network_flow":
        return apply_network_flow_balancing(
            secondary_assignments, df, od_matrix, zone_id
        )
    else:
        logger.warning(f"Unknown balancer {balancer}, falling back to greedy")
        return apply_greedy_transfer_balancing(
            secondary_assignments, df, od_matrix, zone_id, duration_threshold_min, max_iterations
        )


def apply_greedy_transfer_balancing(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    duration_threshold_min: float,
    max_iterations: int
) -> Dict[int, List[int]]:
    """Apply enhanced greedy transfer balancing."""
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
    
    logger.success("Enhanced greedy transfer balancing completed")
    return balanced_assignments


def apply_local_search_balancing(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    duration_threshold_min: float,
    max_iterations: int
) -> Dict[int, List[int]]:
    """Apply local search with swap operations."""
    balanced_assignments = day_assignments.copy()
    
    for iteration in range(max_iterations):
        improved = False
        cluster_durations = calculate_cluster_durations(balanced_assignments, df, od_matrix)
        
        if not cluster_durations:
            break
        
        current_variance = calculate_duration_variance(cluster_durations)
        logger.info(f"Iteration {iteration + 1}: Duration variance = {current_variance:.1f}")
        
        # Try all possible swaps between clusters
        days = list(cluster_durations.keys())
        for i, day1 in enumerate(days):
            for j, day2 in enumerate(days[i+1:], i+1):
                locations1 = balanced_assignments.get(day1, [])
                locations2 = balanced_assignments.get(day2, [])
                
                if not locations1 or not locations2:
                    continue
                
                # Try swapping each location from day1 with each location from day2
                for loc1 in locations1[:]:
                    for loc2 in locations2[:]:
                        # Make the swap temporarily
                        balanced_assignments[day1].remove(loc1)
                        balanced_assignments[day2].remove(loc2)
                        balanced_assignments[day1].append(loc2)
                        balanced_assignments[day2].append(loc1)
                        
                        # Check if this improves balance
                        new_durations = calculate_cluster_durations(balanced_assignments, df, od_matrix)
                        new_variance = calculate_duration_variance(new_durations)
                        
                        if new_variance < current_variance - duration_threshold_min:
                            logger.info(f"Swap improved variance: {current_variance:.1f} → {new_variance:.1f}")
                            improved = True
                            current_variance = new_variance
                            break
                        else:
                            # Revert the swap
                            balanced_assignments[day1].remove(loc2)
                            balanced_assignments[day2].remove(loc1)
                            balanced_assignments[day1].append(loc1)
                            balanced_assignments[day2].append(loc2)
                    
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            logger.info("No improving swaps found, stopping")
            break
    
    logger.success("Local search balancing completed")
    return balanced_assignments


def apply_simulated_annealing_balancing(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    duration_threshold_min: float
) -> Dict[int, List[int]]:
    """Apply simulated annealing balancing."""
    import random
    import math
    
    balanced_assignments = day_assignments.copy()
    current_cost = calculate_balance_cost(balanced_assignments, df, od_matrix)
    
    # Simulated annealing parameters
    initial_temp = 100.0
    cooling_rate = 0.95
    min_temp = 1.0
    iterations_per_temp = 20
    
    temperature = initial_temp
    
    while temperature > min_temp:
        for _ in range(iterations_per_temp):
            # Generate random move (swap two locations between different days)
            days = list(balanced_assignments.keys())
            if len(days) < 2:
                break
            
            day1, day2 = random.sample(days, 2)
            locations1 = balanced_assignments.get(day1, [])
            locations2 = balanced_assignments.get(day2, [])
            
            if not locations1 or not locations2:
                continue
            
            loc1 = random.choice(locations1)
            loc2 = random.choice(locations2)
            
            # Make the move
            balanced_assignments[day1].remove(loc1)
            balanced_assignments[day2].remove(loc2)
            balanced_assignments[day1].append(loc2)
            balanced_assignments[day2].append(loc1)
            
            # Calculate new cost
            new_cost = calculate_balance_cost(balanced_assignments, df, od_matrix)
            cost_diff = new_cost - current_cost
            
            # Accept or reject the move
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_cost = new_cost
            else:
                # Revert the move
                balanced_assignments[day1].remove(loc2)
                balanced_assignments[day2].remove(loc1)
                balanced_assignments[day1].append(loc1)
                balanced_assignments[day2].append(loc2)
        
        temperature *= cooling_rate
    
    logger.success("Simulated annealing balancing completed")
    return balanced_assignments


def apply_min_max_balancing(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    duration_threshold_min: float,
    max_iterations: int
) -> Dict[int, List[int]]:
    """Apply min-max optimization balancing."""
    balanced_assignments = day_assignments.copy()
    
    for iteration in range(max_iterations):
        cluster_durations = calculate_cluster_durations(balanced_assignments, df, od_matrix)
        
        if not cluster_durations:
            break
        
        min_duration = min(cluster_durations.values())
        max_duration = max(cluster_durations.values())
        duration_gap = max_duration - min_duration
        
        logger.info(f"Iteration {iteration + 1}: Min-Max gap = {duration_gap:.1f} min")
        
        if duration_gap <= duration_threshold_min:
            break
        
        # Focus on moving from max to min
        max_day = max(cluster_durations, key=cluster_durations.get)
        min_day = min(cluster_durations, key=cluster_durations.get)
        
        # Move the location that most reduces the max-min gap
        moved = move_location_between_clusters(
            balanced_assignments, max_day, min_day, df, od_matrix
        )
        
        if not moved:
            break
    
    logger.success("Min-max balancing completed")
    return balanced_assignments


def apply_network_flow_balancing(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str
) -> Dict[int, List[int]]:
    """Apply network flow formulation balancing (simplified implementation)."""
    # For now, use greedy approach as placeholder
    # Full network flow implementation would require specialized solvers
    logger.warning("Network flow balancing not fully implemented, using greedy fallback")
    return apply_greedy_transfer_balancing(
        day_assignments, df, od_matrix, zone_id, 60.0, 5
    )


def calculate_duration_variance(cluster_durations: Dict[int, float]) -> float:
    """Calculate variance of cluster durations."""
    if not cluster_durations:
        return 0.0
    
    values = list(cluster_durations.values())
    mean_duration = sum(values) / len(values)
    variance = sum((x - mean_duration) ** 2 for x in values) / len(values)
    return variance


def calculate_balance_cost(
    day_assignments: Dict[int, List[int]],
    df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float]
) -> float:
    """Calculate balance cost (variance of durations)."""
    cluster_durations = calculate_cluster_durations(day_assignments, df, od_matrix)
    return calculate_duration_variance(cluster_durations)


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