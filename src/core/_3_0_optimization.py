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
from ._3_2_optimization_secondary_day_clustering import cluster_secondary_days
from ._3_3_optimization_route_optimization import optimize_itinerary_routes
from ._3_4_optimization_cluster_balancing import balance_cluster_workloads
from ._3_5_optimization_detailed_routing import add_detailed_action_sequences
from src.utils.geo_utils import get_centroid
from src.utils.osrm_utils import generate_od_matrix


def calculate_zone_centroid(df: pl.DataFrame) -> Tuple[float, float]:
    """
    Calculate geographic centroid for a zone.

    :param df: Location DataFrame
    :return: Tuple of (latitude, longitude) for centroid
    """
    return get_centroid(df)


def build_distance_matrix(
    df: pl.DataFrame,
    zone_id: str,
    centroid: Tuple[float, float]
) -> Dict[Tuple[int, int], float]:
    """
    Build origin-destination distance matrix using OSRM.

    :param df: Location DataFrame
    :param zone_id: Zone identifier
    :param centroid: Zone centroid coordinates
    :return: Dictionary mapping location pairs to drive times
    """
    # the generate_od_matrix function expects (longitude, latitude) format for centroid
    centroid_lonlat = (centroid[1], centroid[0])  # convert from (lat, lon) to (lon, lat)
    od_df = generate_od_matrix(df, centroid_lonlat)

    # convert DataFrame to dictionary format expected by optimization functions
    od_dict = {}
    for row in od_df.iter_rows(named=True):
        origin_id = row["origin_id"]
        dest_id = row["destination_id"]
        duration_min = row["duration_minutes"]
        od_dict[(origin_id, dest_id)] = duration_min

    return od_dict


def optimize_zone(
    zone_df: pl.DataFrame,
    zone_id: str,
    model_params: dict,
    clusterer: str,
    balancer: str
) -> pl.DataFrame:
    """
    Execute complete optimization pipeline for a single zone.

    This is the master Stage 3 that coordinates all substages:
    3.0. Calculate centroid and distance matrix (moved from preprocessing)
    3.1. Primary day assignment
    3.2. Secondary day clustering
    3.3. Route optimization
    3.4. Cluster balancing
    3.5. Detailed routing

    :param zone_df: Zone-specific location DataFrame (pre-filtered)
    :param zone_id: Zone identifier
    :param model_params: Model parameters configuration dictionary
    :param clusterer: Clustering algorithm for secondary locations
    :param balancer: Balancing approach for workload equalization
    :return: DataFrame with detailed route information
    """
    logger.info(f"stage 3: optimization - zone {zone_id}")
    logger.info("executing complete optimization pipeline")

    # Stage 3.0: Calculate centroid and OD matrix (moved from preprocessing)
    logger.info(f"processing {len(zone_df)} locations for zone {zone_id}")

    # Calculate centroid
    centroid = calculate_zone_centroid(zone_df)
    logger.info(f"centroid calculated: ({centroid[0]:.4f}, {centroid[1]:.4f})")

    # Generate distance matrix
    od_matrix = build_distance_matrix(zone_df, zone_id, centroid)
    logger.info(f"distance matrix generated: {len(od_matrix)} pairs")
    
    # Stage 3.1: Primary Day Assignment
    itinerary = assign_primary_days(zone_df, model_params)
    logger.debug(f"after assign_primary_days - clusterer: {itinerary['clusterer'].unique()}, balancer: {itinerary['balancer'].unique()}")

    # DEBUG: Print primary assignments for zone_000
    if zone_id == "zone_000":
        primary_assignments = {}
        for row in itinerary.filter(pl.col("pos_class") == "primary").iter_rows(named=True):
            pos_id = row["pos_id"]
            day = row["day"]
            if pos_id not in primary_assignments:
                primary_assignments[pos_id] = []
            primary_assignments[pos_id].append(day)
        logger.debug(f"zone {zone_id} primary assignments: {primary_assignments}")

    # Stage 3.2: Secondary Day Clustering
    if clusterer == "none":
        logger.info("clusterer set to 'none' - skipping secondary day clustering")
    else:
        itinerary = cluster_secondary_days(itinerary, zone_df, od_matrix, centroid, model_params, clusterer)
        logger.debug(f"after cluster_secondary_days - clusterer: {itinerary['clusterer'].unique()}, balancer: {itinerary['balancer'].unique()}")

    # Stage 3.3: Route Optimization
    itinerary = optimize_itinerary_routes(itinerary, zone_df, od_matrix, centroid)
    logger.debug(f"after optimize_itinerary_routes - clusterer: {itinerary['clusterer'].unique()}, balancer: {itinerary['balancer'].unique()}")

    # TODO: Implement remaining optimization stages to work with itinerary pattern
    # Stage 3.4: Cluster Balancing
    # itinerary = balance_cluster_workloads(itinerary, od_matrix, model_params, balancer)

    # Stage 3.5: Detailed Routing
    itinerary = add_detailed_action_sequences(itinerary, od_matrix)
    logger.debug(f"after add_detailed_action_sequences - clusterer: {itinerary['clusterer'].unique()}, balancer: {itinerary['balancer'].unique()}")

    logger.success(f"optimization complete for zone {zone_id}")

    return itinerary


# Note: combine_assignments function removed as primary and secondary
# assignments are now handled separately throughout the pipeline


# ------------------------------------------------------------------------------
# end of _3_0_optimization.py
# ------------------------------------------------------------------------------