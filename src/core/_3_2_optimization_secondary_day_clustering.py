"""
Stage 3.2: Secondary Day Clustering
Cluster secondary locations into days using K-means
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.cluster import KMeans

from src.utils.clustering_utils import (
    kmeans_cluster_locations,
    identify_noise_points
)


def cluster_secondary_locations(
    secondary_df: pl.DataFrame,
    available_days: List[int],
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    noise_threshold_km: float = 150.0
) -> Dict[int, List[int]]:
    """
    Cluster secondary locations into available days.
    
    This is substage 3.2 where we:
    1. Apply K-means clustering
    2. Detect and handle noise points
    3. Create day assignments
    
    Args:
        secondary_df: DataFrame of secondary locations
        available_days: Days available for secondary locations
        od_matrix: Distance matrix
        zone_id: Zone identifier
        noise_threshold_km: Distance threshold for noise detection
        
    Returns:
        Dictionary mapping days to location IDs
    """
    logger.info(f"Stage 3.2: SECONDARY CLUSTERING - Zone {zone_id}")
    logger.info(f"Clustering {len(secondary_df)} locations into {len(available_days)} days")
    
    if len(secondary_df) == 0:
        logger.warning("No secondary locations to cluster")
        return {}
    
    if len(available_days) == 0:
        logger.warning("No available days for secondary locations")
        return {}
    
    # Detect noise points  
    noise_points = identify_noise_points(
        secondary_df,  # Keep as polars DataFrame  
        noise_threshold_km=noise_threshold_km
    )
    
    n_noise = len(noise_points)
    if n_noise > 0:
        logger.warning(f"Detected {n_noise} noise points (isolated locations)")
    
    # Apply K-means clustering
    n_clusters = min(len(available_days), len(secondary_df))
    
    # Use the existing kmeans function with Polars DataFrame
    locations_data = secondary_df.select(['pos_id', 'latitude', 'longitude'])
    
    clustered_df = kmeans_cluster_locations(
        locations_data, 
        n_clusters=n_clusters,
        random_seed=42
    )
    
    # Extract cluster assignments
    cluster_assignments = clustered_df['cluster_id'].to_list()
    
    # Map clusters to days
    day_assignments = {}
    pos_ids = clustered_df['pos_id'].to_list()
    
    for day_idx, day in enumerate(available_days[:n_clusters]):
        cluster_locations = []
        for idx, cluster_id in enumerate(cluster_assignments):
            if cluster_id == day_idx:
                pos_id = pos_ids[idx]
                cluster_locations.append(pos_id)
        
        if cluster_locations:
            day_assignments[day] = cluster_locations
            logger.info(f"Day {day}: {len(cluster_locations)} locations")
    
    logger.success(f"Clustering complete: {len(day_assignments)} clusters created")
    
    return day_assignments