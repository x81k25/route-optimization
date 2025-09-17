"""
Stage 3.2: Secondary Day Clustering
Cluster secondary locations into days using various clustering algorithms
"""

# standard library imports
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import numpy as np
import polars as pl
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# local imports
from src.utils.clustering_utils import (
    identify_noise_points,
    kmeans_cluster_locations
)
from src.data_models.itinerary_frame import ItineraryFrame


def cluster_secondary_days(
    itinerary: pl.DataFrame,
    zone_df: pl.DataFrame,
    od_matrix: Dict[Tuple[int, int], float],
    centroid: Tuple[float, float],
    model_params: Dict[str, any],
    clusterer: str
) -> pl.DataFrame:
    """
    Stage 3.2: Cluster secondary locations into available secondary days.

    Logic:
    1. Count days with no primary class POS - this is the number of output clusters
    2. Collect all secondary stores and cluster them
    3. Strip original arbitrary day assignments from stage 3.1
    4. Apply clustering algorithm to secondary locations only

    :param itinerary: ItineraryFrame from stage 3.1 with primary day assignments
    :param zone_df: Zone location data
    :param od_matrix: Distance matrix for clustering
    :param centroid: Zone centroid coordinates
    :param model_params: Model parameters configuration dictionary
    :param clusterer: Clustering algorithm to use
    :return: Updated itinerary with secondary day assignments
    """
    logger.info("stage 3.2: secondary day clustering")

    # Get zone_id from itinerary (all records will have same zone_id)
    zone_id = itinerary.select("zone_id").unique().to_series().to_list()[0]
    logger.info(f"processing zone {zone_id}")

    # Step 1: Count days with no primary class POS
    days_per_week = model_params.get("days_per_week", 5)
    hours_per_non_primary = model_params.get("hours_per_non_primary", 1)
    primary_days = itinerary.filter(pl.col("pos_class") == "primary").select("day").unique().to_series().to_list()

    all_days = set(range(1, days_per_week + 1))
    primary_days_set = set(primary_days)
    secondary_days_available = list(all_days - primary_days_set)
    n_secondary_clusters = len(secondary_days_available)

    logger.info(f"primary days: {sorted(primary_days)}")
    logger.info(f"secondary days available: {sorted(secondary_days_available)}")
    logger.info(f"number of secondary clusters needed: {n_secondary_clusters}")

    # Step 2: Collect all secondary stores from zone_df (ignore existing day assignments)
    secondary_df = zone_df.filter(pl.col("class") == "secondary")

    if len(secondary_df) == 0:
        logger.info("no secondary locations to cluster")
        # Return only primary assignments
        return itinerary.filter(pl.col("pos_class") == "primary")

    if n_secondary_clusters == 0:
        logger.warning("no secondary days available - all days are primary days")
        # Return only primary assignments
        return itinerary.filter(pl.col("pos_class") == "primary")

    # Step 3 & 4: Cluster secondary locations
    cluster_assignments = cluster_secondary_locations(
        secondary_df=secondary_df,
        secondary_days=n_secondary_clusters,
        od_matrix=od_matrix,
        zone_id=zone_id,
        clusterer=clusterer
    )

    # DEBUG: Print cluster assignments for zone_000
    if zone_id == "zone_000":
        logger.info(f"debug: zone {zone_id} cluster assignments: {cluster_assignments}")
        logger.info(f"debug: zone {zone_id} secondary days available: {secondary_days_available}")

    # Step 5: Create new secondary itinerary records
    secondary_records = []
    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get metadata from existing itinerary
    sample_record = itinerary.row(0, named=True)
    clusterer_name = sample_record["clusterer"]
    router_name = sample_record["router"]
    balancer_name = sample_record["balancer"]

    for cluster_id, pos_ids in cluster_assignments.items():
        if cluster_id < len(secondary_days_available):
            assigned_day = secondary_days_available[cluster_id]

            for pos_id in pos_ids:
                # Get location data
                pos_data = secondary_df.filter(pl.col("pos_id") == pos_id).row(0, named=True)

                secondary_records.append({
                    "zone_id": zone_id,
                    "day": assigned_day,
                    "pos_id": str(pos_id),
                    "pos_name": pos_data["name"],
                    "pos_class": "secondary",
                    "route": [[pos_data["longitude"], pos_data["latitude"]]],
                    "action": None,
                    "schedule": None,
                    "duration": hours_per_non_primary * 60.0,  # convert hours to minutes
                    "route_order": None,  # Will be populated in route optimization stage
                    "clusterer": clusterer_name,
                    "router": router_name,
                    "balancer": balancer_name,
                    "created_on": created_on
                })

    # Step 6: Combine primary and secondary assignments
    primary_itinerary = itinerary.filter(pl.col("pos_class") == "primary")

    if secondary_records:
        secondary_itinerary = pl.DataFrame(secondary_records, schema=itinerary.schema)
        final_itinerary = pl.concat([primary_itinerary, secondary_itinerary], how="vertical")
    else:
        final_itinerary = primary_itinerary

    # assign clusterer value
    final_itinerary = final_itinerary.with_columns(clusterer = pl.lit(clusterer))

    logger.success(f"secondary clustering complete: {len(secondary_records)} secondary assignments created")

    return final_itinerary


def cluster_secondary_locations(
    secondary_df: pl.DataFrame,
    secondary_days: int,
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    clusterer: str,
    noise_threshold_km: float = 150.0
) -> Dict[int, List[int]]:
    """
    Cluster secondary locations using the calculated secondary_days count.

    This is substage 3.2 where we:
    1. Use secondary_days as the number of clusters to create
    2. Apply selected clustering algorithm to secondary POS only
    3. Detect and handle noise points
    4. Create day assignments for secondary clusters

    :param secondary_df: DataFrame of secondary locations
    :param secondary_days: Number of secondary days (clusters) to create
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param clusterer: Clustering algorithm to use
    :param noise_threshold_km: Distance threshold for noise detection
    :return: Dictionary mapping cluster IDs to location IDs
    """
    logger.info(f"stage 3.2: secondary clustering - zone {zone_id}")
    logger.info(f"clustering {len(secondary_df)} locations into {secondary_days} clusters using {clusterer}")

    if len(secondary_df) == 0:
        logger.warning("no secondary locations to cluster")
        return {}

    if secondary_days <= 0:
        logger.warning("no secondary days available for clustering")
        return {}

    # detect noise points
    noise_points = identify_noise_points(
        secondary_df,  # keep as polars DataFrame
        noise_threshold_km=noise_threshold_km
    )

    n_noise = len(noise_points)
    if n_noise > 0:
        logger.warning(f"detected {n_noise} noise points (isolated locations)")

    # apply selected clustering algorithm using secondary_days as cluster count
    n_clusters = min(secondary_days, len(secondary_df))
    locations_data = secondary_df.select(['pos_id', 'latitude', 'longitude'])

    if clusterer == "mds_kmeans":
        clustered_df = apply_mds_kmeans_clustering(locations_data, n_clusters, od_matrix)
    elif clusterer == "dbscan":
        clustered_df = apply_dbscan_clustering(locations_data, od_matrix)
    elif clusterer == "hierarchical":
        clustered_df = apply_hierarchical_clustering(locations_data, n_clusters, od_matrix)
    elif clusterer == "spectral":
        clustered_df = apply_spectral_clustering(locations_data, n_clusters, od_matrix)
    elif clusterer == "balanced":
        clustered_df = apply_balanced_clustering(locations_data, n_clusters, od_matrix)
    else:
        logger.warning(f"unknown clusterer {clusterer}, falling back to mds_kmeans")
        clustered_df = apply_mds_kmeans_clustering(locations_data, n_clusters, od_matrix)

    # extract cluster assignments
    cluster_assignments = clustered_df['cluster_id'].to_list()

    # map clusters to cluster IDs (0, 1, 2, etc.)
    cluster_assignments_dict = {}
    pos_ids = clustered_df['pos_id'].to_list()

    for cluster_id in range(n_clusters):
        cluster_locations = []
        for idx, assigned_cluster in enumerate(cluster_assignments):
            if assigned_cluster == cluster_id:
                pos_id = pos_ids[idx]
                cluster_locations.append(pos_id)

        if cluster_locations:
            cluster_assignments_dict[cluster_id] = cluster_locations
            logger.info(f"cluster {cluster_id}: {len(cluster_locations)} locations")

    logger.success(f"clustering complete: {len(cluster_assignments_dict)} clusters created")

    return cluster_assignments_dict


def apply_mds_kmeans_clustering(
    locations_data: pl.DataFrame, 
    n_clusters: int, 
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """Apply MDS + K-means clustering with drive times."""
    pos_ids = locations_data['pos_id'].to_list()
    n_locations = len(pos_ids)
    
    # Build distance matrix from od_matrix
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = od_matrix.get((pos_ids[i], pos_ids[j]), 0.0)
    
    # Apply MDS for dimensionality reduction
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
    coords_2d = mds.fit_transform(distance_matrix)
    
    # Apply K-means on the 2D coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords_2d)
    
    # Create result DataFrame
    return locations_data.with_columns([
        pl.Series("cluster_id", cluster_labels)
    ])


def apply_dbscan_clustering(
    locations_data: pl.DataFrame, 
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """Apply DBSCAN clustering on MDS-transformed drive times."""
    pos_ids = locations_data['pos_id'].to_list()
    n_locations = len(pos_ids)
    
    # Build distance matrix from od_matrix
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = od_matrix.get((pos_ids[i], pos_ids[j]), 0.0)
    
    # Apply MDS for dimensionality reduction
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
    coords_2d = mds.fit_transform(distance_matrix)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(coords_2d)
    
    # Handle noise points by assigning them to nearest cluster
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    for i, label in enumerate(cluster_labels):
        if label == -1:  # noise point
            if len(unique_labels) > 0:
                cluster_labels[i] = min(unique_labels)  # assign to first valid cluster
            else:
                cluster_labels[i] = 0  # fallback
    
    # Create result DataFrame
    return locations_data.with_columns([
        pl.Series("cluster_id", cluster_labels)
    ])


def apply_hierarchical_clustering(
    locations_data: pl.DataFrame, 
    n_clusters: int, 
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """Apply hierarchical clustering on drive time matrix."""
    pos_ids = locations_data['pos_id'].to_list()
    n_locations = len(pos_ids)
    
    # Build distance matrix from od_matrix
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = od_matrix.get((pos_ids[i], pos_ids[j]), 0.0)
    
    # Apply hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='complete',
        metric='precomputed'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Create result DataFrame
    return locations_data.with_columns([
        pl.Series("cluster_id", cluster_labels)
    ])


def apply_spectral_clustering(
    locations_data: pl.DataFrame, 
    n_clusters: int, 
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """Apply spectral clustering on drive time matrix."""
    pos_ids = locations_data['pos_id'].to_list()
    n_locations = len(pos_ids)
    
    # Build distance matrix from od_matrix
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = od_matrix.get((pos_ids[i], pos_ids[j]), 0.0)
    
    # Convert distance to similarity (affinity)
    max_dist = np.max(distance_matrix)
    if max_dist > 0:
        affinity_matrix = 1.0 - (distance_matrix / max_dist)
    else:
        affinity_matrix = np.ones_like(distance_matrix)
    np.fill_diagonal(affinity_matrix, 1.0)
    
    # Apply spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    cluster_labels = clustering.fit_predict(affinity_matrix)
    
    # Create result DataFrame
    return locations_data.with_columns([
        pl.Series("cluster_id", cluster_labels)
    ])


def apply_balanced_clustering(
    locations_data: pl.DataFrame, 
    n_clusters: int, 
    od_matrix: Dict[Tuple[int, int], float]
) -> pl.DataFrame:
    """Apply balanced partitioning with drive times (simplified implementation)."""
    # For now, use K-means as baseline for balanced clustering
    # This is a simplified implementation - full balanced partitioning would be more complex
    pos_ids = locations_data['pos_id'].to_list()
    n_locations = len(pos_ids)
    
    # Build distance matrix from od_matrix
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = od_matrix.get((pos_ids[i], pos_ids[j]), 0.0)
    
    # Apply MDS for dimensionality reduction
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
    coords_2d = mds.fit_transform(distance_matrix)
    
    # Apply K-means with balanced initialization
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(coords_2d)
    
    # Simple rebalancing: ensure each cluster has roughly equal size
    locations_per_cluster = n_locations // n_clusters
    remainder = n_locations % n_clusters
    
    # Sort locations by distance to their assigned cluster center
    balanced_labels = np.zeros_like(cluster_labels)
    assigned_count = np.zeros(n_clusters, dtype=int)
    
    # Process locations in order of confidence (distance to cluster center)
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            continue
        
        cluster_indices = np.where(cluster_mask)[0]
        center = coords_2d[cluster_indices].mean(axis=0)
        distances = np.linalg.norm(coords_2d[cluster_indices] - center, axis=1)
        sorted_indices = cluster_indices[np.argsort(distances)]
        
        target_size = locations_per_cluster + (1 if cluster_id < remainder else 0)
        for i, idx in enumerate(sorted_indices[:target_size]):
            balanced_labels[idx] = cluster_id
            assigned_count[cluster_id] += 1
    
    # Assign any remaining locations to least full clusters
    unassigned = np.where(balanced_labels == 0)[0]
    for idx in unassigned:
        least_full_cluster = np.argmin(assigned_count)
        balanced_labels[idx] = least_full_cluster
        assigned_count[least_full_cluster] += 1
    
    # Create result DataFrame
    return locations_data.with_columns([
        pl.Series("cluster_id", balanced_labels)
    ])