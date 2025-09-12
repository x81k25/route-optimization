"""
Stage 3: Optimization Master Module
Coordinates all optimization substages
"""

import polars as pl
from typing import Dict, List, Tuple
from loguru import logger

from ._3_1_optimization_primary_day_assignment import assign_primary_days
from ._3_2_optimization_secondary_day_clustering import cluster_secondary_locations
from ._3_3_optimization_route_optimization import optimize_daily_routes
from ._3_4_optimization_cluster_balancing import balance_cluster_workloads
from ._3_5_optimization_detailed_routing import get_detailed_route_information


def optimize_zone(
    df: pl.DataFrame,
    zone_id: str,
    od_matrix: Dict[Tuple[int, int], float],
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
    
    Args:
        df: Location DataFrame for zone
        zone_id: Zone identifier
        od_matrix: Distance matrix
        config_path: Configuration file path
        
    Returns:
        DataFrame with detailed route information
    """
    logger.info(f"Stage 3: OPTIMIZATION - Zone {zone_id}")
    logger.info("Executing complete optimization pipeline")
    
    # Stage 3.1: Primary Day Assignment
    primary_assignments, available_days, primary_df, secondary_df = assign_primary_days(
        df, zone_id, config_path
    )
    
    # Stage 3.2: Secondary Day Clustering
    secondary_assignments = cluster_secondary_locations(
        secondary_df, available_days, od_matrix, zone_id
    )
    
    # Combine assignments
    all_assignments = combine_assignments(
        primary_assignments, secondary_assignments
    )
    
    # Stage 3.3: Route Optimization
    optimized_routes = optimize_daily_routes(
        all_assignments, df, od_matrix, zone_id
    )
    
    # Stage 3.4: Cluster Balancing
    balanced_assignments = balance_cluster_workloads(
        all_assignments, df, od_matrix, zone_id
    )
    
    # Re-optimize routes after balancing
    final_routes = optimize_daily_routes(
        balanced_assignments, df, od_matrix, zone_id
    )
    
    # Stage 3.5: Detailed Routing
    result_df = get_detailed_route_information(
        final_routes, df, zone_id
    )
    
    logger.success(f"Optimization complete for zone {zone_id}")
    
    return result_df


def combine_assignments(
    primary_assignments: Dict[int, int],
    secondary_assignments: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    Combine primary and secondary assignments into unified day assignments.
    
    Args:
        primary_assignments: Mapping location_id -> day
        secondary_assignments: Mapping day -> [location_ids]
        
    Returns:
        Combined day assignments
    """
    combined = {}
    
    # Add primary assignments
    for location_id, day in primary_assignments.items():
        if day not in combined:
            combined[day] = []
        combined[day].append(location_id)
    
    # Add secondary assignments
    for day, location_ids in secondary_assignments.items():
        if day not in combined:
            combined[day] = []
        combined[day].extend(location_ids)
    
    return combined