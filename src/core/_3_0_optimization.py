"""
Stage 3: Optimization Master Module
Coordinates all optimization substages
"""

# standard library imports
from typing import Dict, List, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
from ._3_1_optimization_primary_day_assignment import assign_primary_days
from ._3_2_optimization_secondary_day_clustering import cluster_secondary_locations
from ._3_3_optimization_route_optimization import optimize_daily_routes
from ._3_4_optimization_cluster_balancing import balance_cluster_workloads
from ._3_5_optimization_detailed_routing import get_detailed_route_information


def optimize_zone(
    df: pl.DataFrame,
    zone_id: str,
    od_matrix: Dict[Tuple[int, int], float],
    clusterer: str = "mds_kmeans",
    balancer: str = "greedy",
    centroid: Tuple[float, float] = (0.0, 0.0),
    config_path: str = "./config/model-params.yaml"
) -> pl.DataFrame:
    """
    Execute complete optimization pipeline for a single zone.
    
    This is the master Stage 3 that coordinates all substages:
    3.1. Primary day assignment
    3.2. Secondary day clustering  
    3.3. Route optimization
    3.4. Cluster balancing
    3.5. Detailed routing
    
    :param df: Location DataFrame for zone
    :param zone_id: Zone identifier
    :param od_matrix: Distance matrix
    :param clusterer: Clustering algorithm for secondary locations
    :param balancer: Balancing approach for workload equalization
    :param config_path: Configuration file path
    :return: DataFrame with detailed route information
    """
    logger.info(f"Stage 3: OPTIMIZATION - Zone {zone_id}")
    logger.info("Executing complete optimization pipeline")
    
    # stage 3.1: Primary Day Assignment
    primary_assignments, available_days, primary_df, secondary_df = assign_primary_days(
        df, zone_id, config_path
    )
    
    # stage 3.2: Secondary Day Clustering
    secondary_assignments = cluster_secondary_locations(
        secondary_df, available_days, od_matrix, zone_id, clusterer
    )
    
    # combine assignments
    all_assignments = combine_assignments(
        primary_assignments, secondary_assignments
    )
    
    # stage 3.3: Route Optimization
    optimized_routes = optimize_daily_routes(
        all_assignments, df, od_matrix, zone_id
    )
    
    # stage 3.4: Cluster Balancing
    balanced_assignments = balance_cluster_workloads(
        all_assignments, df, od_matrix, zone_id, balancer
    )
    
    # re-optimize routes after balancing
    final_routes = optimize_daily_routes(
        balanced_assignments, df, od_matrix, zone_id
    )
    
    # stage 3.5: Detailed Routing
    result_df = get_detailed_route_information(
        final_routes, df, zone_id, centroid
    )
    
    logger.success(f"Optimization complete for zone {zone_id}")
    
    return result_df


def combine_assignments(
    primary_assignments: Dict[int, int],
    secondary_assignments: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    Combine primary and secondary assignments into unified day assignments.
    
    :param primary_assignments: Mapping location_id -> day
    :param secondary_assignments: Mapping day -> [location_ids]
    :return: Combined day assignments
    """
    combined = {}
    
    # add primary assignments
    for location_id, day in primary_assignments.items():
        if day not in combined:
            combined[day] = []
        combined[day].append(location_id)
    
    # add secondary assignments
    for day, location_ids in secondary_assignments.items():
        if day not in combined:
            combined[day] = []
        combined[day].extend(location_ids)
    
    return combined