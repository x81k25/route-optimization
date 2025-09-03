"""
Stage 1: Day Assignment Algorithm

Functional programming implementation that assigns secondary locations 
to available days using drive time-based clustering.
"""

import numpy as np
import polars as pl
from typing import List, Dict, Tuple, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from loguru import logger

from ..utils.osrm_utils import fetch_od_matrix, convert_locations_from_polars, od_matrix_to_polars


def get_od_matrix_polars(zone_id: str, secondary_df: pl.DataFrame) -> pl.DataFrame:
    """Get OD matrix in Polars DataFrame format from secondary locations."""
    locations = convert_locations_from_polars(secondary_df)
    od_result = fetch_od_matrix(zone_id, locations)
    return od_matrix_to_polars(od_result)


def get_distance_matrix_from_od_result(od_result) -> Tuple[np.ndarray, List[int]]:
    """Extract distance matrix and location IDs from OD matrix result."""
    location_ids = od_result.location_ids
    # Use duration matrix for clustering (in minutes)
    distance_matrix = od_result.duration_matrix / 60.0
    return distance_matrix, location_ids


def cluster_locations_hierarchical(
    distance_matrix: np.ndarray, 
    location_ids: List[int], 
    n_clusters: int, 
    max_locations_per_cluster: int = 7
) -> Dict[int, List[int]]:
    """
    Cluster locations into days using hierarchical clustering.
    
    Args:
        distance_matrix: Square distance matrix between locations
        location_ids: List of location IDs
        n_clusters: Number of clusters (available secondary days)
        max_locations_per_cluster: Maximum locations per cluster
        
    Returns:
        Dictionary mapping cluster_id to list of location_ids
    """
    n_locations = len(location_ids)
    
    # Handle empty case
    if n_locations == 0:
        return {}
    
    # Handle single location case
    if n_locations == 1:
        return {1: location_ids.copy()}
    
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
    cluster_assignments = enforce_cluster_size_constraints(
        cluster_assignments, max_locations_per_cluster
    )
    
    return cluster_assignments


def enforce_cluster_size_constraints(
    clusters: Dict[int, List[int]], 
    max_size: int
) -> Dict[int, List[int]]:
    """Split clusters that exceed size limit."""
    adjusted_clusters = {}
    cluster_counter = 1
    
    for cluster_id, locations in clusters.items():
        if len(locations) <= max_size:
            # Cluster is within size limit
            adjusted_clusters[cluster_counter] = locations
            cluster_counter += 1
        else:
            # Split oversized cluster
            n_splits = (len(locations) + max_size - 1) // max_size  # Ceiling division
            
            for i in range(n_splits):
                start_idx = i * max_size
                end_idx = min((i + 1) * max_size, len(locations))
                adjusted_clusters[cluster_counter] = locations[start_idx:end_idx]
                cluster_counter += 1
    
    return adjusted_clusters


def calculate_cluster_quality(
    clusters: Dict[int, List[int]], 
    distance_matrix: np.ndarray, 
    location_ids: List[int]
) -> Dict[str, float]:
    """Calculate quality metrics for clustering solution."""
    total_intra_cluster_distance = 0
    total_pairs = 0
    cluster_sizes = []
    
    for cluster_id, locations in clusters.items():
        cluster_sizes.append(len(locations))
        
        # Calculate average intra-cluster distance
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i < j:  # Avoid double counting
                    idx1 = location_ids.index(loc1)
                    idx2 = location_ids.index(loc2)
                    distance = distance_matrix[idx1, idx2]
                    total_intra_cluster_distance += distance
                    total_pairs += 1
    
    avg_intra_cluster_distance = total_intra_cluster_distance / max(total_pairs, 1)
    
    return {
        'avg_intra_cluster_distance': avg_intra_cluster_distance,
        'cluster_size_std': np.std(cluster_sizes),
        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
        'n_clusters': len(clusters)
    }


def calculate_cluster_quality_from_secondary_df(
    clusters: Dict[int, List[int]], 
    secondary_df: pl.DataFrame
) -> Dict[str, float]:
    """Calculate quality metrics for clustering solution using secondary DataFrame."""
    # Get locations and create OD matrix
    locations = convert_locations_from_polars(secondary_df)
    if not locations:
        return {
            'avg_intra_cluster_distance': 0,
            'cluster_size_std': 0,
            'min_cluster_size': 0,
            'max_cluster_size': 0,
            'n_clusters': 0
        }
    
    # Create a minimal OD result for quality calculation
    location_ids = [loc.location_id for loc in locations]
    
    # Create a simple distance matrix based on haversine distance for quality calculation
    # This is a fallback when we don't have the full OD matrix
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points."""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    n_locations = len(locations)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i != j:
                distance = haversine_distance(
                    loc1.latitude, loc1.longitude, 
                    loc2.latitude, loc2.longitude
                )
                distance_matrix[i, j] = distance
    
    return calculate_cluster_quality(clusters, distance_matrix, location_ids)


def optimize_clusters_with_swaps(
    clusters: Dict[int, List[int]], 
    distance_matrix: np.ndarray, 
    location_ids: List[int],
    max_iterations: int = 100
) -> Dict[int, List[int]]:
    """
    Improve clustering by swapping locations between clusters.
    
    Args:
        clusters: Initial cluster assignments
        distance_matrix: Distance matrix between locations
        location_ids: List of location IDs
        max_iterations: Maximum number of swap attempts
        
    Returns:
        Improved cluster assignments
    """
    current_clusters = {k: v.copy() for k, v in clusters.items()}
    best_quality = calculate_cluster_quality(current_clusters, distance_matrix, location_ids)['avg_intra_cluster_distance']
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping each location to each other cluster
        for cluster_id, locations in list(current_clusters.items()):
            for location in locations.copy():
                for other_cluster_id in current_clusters:
                    if other_cluster_id == cluster_id:
                        continue
                    
                    # Skip if target cluster would exceed size limit
                    if len(current_clusters[other_cluster_id]) >= 7:
                        continue
                    
                    # Try the swap
                    current_clusters[cluster_id].remove(location)
                    current_clusters[other_cluster_id].append(location)
                    
                    # Check if improvement
                    new_quality = calculate_cluster_quality(current_clusters, distance_matrix, location_ids)['avg_intra_cluster_distance']
                    
                    if new_quality < best_quality:
                        # Keep the swap
                        best_quality = new_quality
                        improved = True
                        break
                    else:
                        # Revert the swap
                        current_clusters[other_cluster_id].remove(location)
                        current_clusters[cluster_id].append(location)
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break  # No more improvements possible
    
    return current_clusters


def assign_days_to_secondary_locations(
    secondary_df: pl.DataFrame,
    zone_id: str,
    available_secondary_days: int,
    max_locations_per_day: int = 7,
    use_swap_optimization: bool = True
) -> Dict[int, List[int]]:
    """
    Main function to assign secondary locations to days.
    
    Args:
        secondary_df: DataFrame containing secondary location data
        zone_id: Zone identifier
        available_secondary_days: Number of days available for secondary locations
        max_locations_per_day: Maximum locations per day
        use_swap_optimization: Whether to apply swap optimization
        
    Returns:
        Dictionary mapping day_id to list of location_ids
    """
    # Convert to Location objects and fetch OD matrix
    locations = convert_locations_from_polars(secondary_df)
    
    if not locations:
        return {}
    
    od_result = fetch_od_matrix(zone_id, locations)
    distance_matrix, location_ids = get_distance_matrix_from_od_result(od_result)
    
    # Initial clustering
    clusters = cluster_locations_hierarchical(
        distance_matrix, location_ids, available_secondary_days, max_locations_per_day
    )
    
    # Optional swap optimization
    if use_swap_optimization:
        clusters = optimize_clusters_with_swaps(clusters, distance_matrix, location_ids)
    
    return clusters


def load_secondary_locations_data(locations_path: str = "data/subway_locations.json") -> List[Dict]:
    """Load and return secondary locations data."""
    import json
    
    with open(locations_path, 'r') as f:
        data = json.load(f)
    
    return [
        loc for loc in data['subway_locations_san_francisco'] 
        if loc['class'] == 'secondary'
    ]


if __name__ == "__main__":
    # Example usage with functional approach
    import polars as pl
    
    # Load sample data
    locations_data = load_secondary_locations_data()
    
    # Convert to DataFrame
    secondary_df = pl.DataFrame([
        {
            'location_id': loc['id'],
            'zone_id': 'test_zone',
            'name': loc['name'],
            'location_class': 'secondary',
            'address': loc['address'],
            'latitude': loc['latitude'],
            'longitude': loc['longitude'],
            'source_system': 'test'
        }
        for loc in locations_data
    ])
    
    # Assign to 3 days (assuming 5 work days - 2 primary days = 3 secondary days)
    day_assignments = assign_days_to_secondary_locations(
        secondary_df=secondary_df,
        zone_id='test_zone',
        available_secondary_days=3
    )
    
    # Display results
    id_to_name = {loc['id']: loc['name'] for loc in locations_data}
    
    logger.info("Day Assignments:")
    logger.info("================")
    for day_id, location_ids in day_assignments.items():
        logger.info(f"\nDay {day_id} ({len(location_ids)} locations):")
        for loc_id in location_ids:
            logger.info(f"  - {id_to_name[loc_id]}")
    
    # Show quality metrics
    quality = calculate_cluster_quality_from_secondary_df(day_assignments, secondary_df)
    logger.info(f"\nCluster Quality Metrics:")
    for metric, value in quality.items():
        logger.info(f"  {metric}: {value:.2f}")