"""
Stage 3.2: Secondary Day Clustering
Cluster secondary locations into days using K-means
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

# local imports
from src.utils.clustering_utils import (
    identify_noise_points,
    kmeans_cluster_locations
)


def cluster_secondary_locations(
    secondary_df: pl.DataFrame,
    available_days: List[int],
    od_matrix: Dict[Tuple[int, int], float],
    zone_id: str,
    clusterer: str = "mds_kmeans",
    noise_threshold_km: float = 150.0
) -> Dict[int, List[int]]:
    """
    Cluster secondary locations into available days.
    
    This is substage 3.2 where we:
    1. Apply selected clustering algorithm
    2. Detect and handle noise points
    3. Create day assignments
    
    :param secondary_df: DataFrame of secondary locations
    :param available_days: Days available for secondary locations
    :param od_matrix: Distance matrix
    :param zone_id: Zone identifier
    :param clusterer: Clustering algorithm to use
    :param noise_threshold_km: Distance threshold for noise detection
    :return: Dictionary mapping days to location IDs
    """
    logger.info(f"Stage 3.2: SECONDARY CLUSTERING - Zone {zone_id}")
    logger.info(f"Clustering {len(secondary_df)} locations into {len(available_days)} days using {clusterer}")
    
    if len(secondary_df) == 0:
        logger.warning("No secondary locations to cluster")
        return {}
    
    if len(available_days) == 0:
        logger.warning("No available days for secondary locations")
        return {}
    
    # detect noise points  
    noise_points = identify_noise_points(
        secondary_df,  # keep as polars DataFrame  
        noise_threshold_km=noise_threshold_km
    )
    
    n_noise = len(noise_points)
    if n_noise > 0:
        logger.warning(f"Detected {n_noise} noise points (isolated locations)")
    
    # apply selected clustering algorithm
    n_clusters = min(len(available_days), len(secondary_df))
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
        logger.warning(f"Unknown clusterer {clusterer}, falling back to mds_kmeans")
        clustered_df = apply_mds_kmeans_clustering(locations_data, n_clusters, od_matrix)
    
    # extract cluster assignments
    cluster_assignments = clustered_df['cluster_id'].to_list()
    
    # map clusters to days
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