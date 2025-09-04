"""
Functional clustering utility for route optimization.

Creates geographic zones from location data with configurable cluster sizes.
Uses pure functions and Polars DataFrames for data processing.
"""

# standard library imports
import json
import random
from typing import Any, Dict, Tuple

# 3rd-party imports
import numpy as np
import polars as pl
from loguru import logger
from sklearn.cluster import KMeans


# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

def default_cluster_config() -> Dict[str, Any]:
    """
    Default configuration for clustering algorithm.
    
    :return: dictionary with default clustering configuration
    """
    return {
        'min_locations_per_cluster': 3,
        'max_locations_per_cluster': 25,
        'method': 'kmeans',  # "kmeans", "geographic"
        'random_seed': 42,
        'primary_store_min': 0,
        'primary_store_max': 3
    }


def haversine_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> float:
    """
    Calculate haversine distance between two points in kilometers.
    
    :param lat1: latitude of first point
    :param lon1: longitude of first point
    :param lat2: latitude of second point
    :param lon2: longitude of second point
    :return: distance in kilometers
    """
    R = 6371.0  # earth radius in kilometers
    
    # convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_distance_matrix(
    locations_df: pl.DataFrame
) -> np.ndarray:
    """
    Calculate distance matrix between all location pairs.
    
    :param locations_df: DataFrame with 'latitude' and 'longitude' columns
    :return: distance matrix (n x n)
    """
    coords = locations_df.select(['latitude', 'longitude']).to_numpy()
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix


def kmeans_cluster_locations(
    locations_df: pl.DataFrame, 
    n_clusters: int, 
    random_seed: int = 42
) -> pl.DataFrame:
    """
    Perform K-means clustering on locations.
    
    :param locations_df: DataFrame with location data including 'latitude', 'longitude'
    :param n_clusters: number of clusters
    :param random_seed: random seed for reproducibility
    :return: DataFrame with additional 'cluster_id' column
    """
    # filter out locations with null coordinates
    valid_locations = locations_df.filter(
        (pl.col('latitude').is_not_null()) & (pl.col('longitude').is_not_null())
    )
    
    null_locations = locations_df.filter(
        (pl.col('latitude').is_null()) | (pl.col('longitude').is_null())
    )
    
    logger.info(f"k-means clustering: {len(valid_locations)} valid locations, {len(null_locations)} excluded (null coordinates)")
    
    if len(valid_locations) == 0:
        logger.error("no valid coordinates found for clustering")
        return locations_df.with_columns(pl.lit(None).alias('cluster_id'))
    
    if len(valid_locations) < n_clusters:
        logger.warning(f"fewer valid locations ({len(valid_locations)}) than clusters ({n_clusters})")
        # assign each valid location to its own cluster
        valid_with_clusters = valid_locations.with_columns(
            pl.arange(0, len(valid_locations)).alias('cluster_id')
        )
    else:
        # extract coordinates from valid locations only
        coordinates = valid_locations.select(['latitude', 'longitude']).to_numpy()
        
        # perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # add cluster labels to valid locations
        valid_with_clusters = valid_locations.with_columns(
            pl.Series('cluster_id', cluster_labels)
        )
    
    # add null cluster_id to null locations
    null_with_clusters = null_locations.with_columns(
        pl.lit(None).alias('cluster_id')
    )
    
    # combine back together, preserving original order
    result = pl.concat([valid_with_clusters, null_with_clusters], how="vertical")
    
    # sort by original index if available, otherwise by id
    if 'id' in result.columns:
        result = result.sort('id')
    
    return result


def geographic_cluster_locations(
    locations_df: pl.DataFrame, 
    target_size: int
) -> pl.DataFrame:
    """
    Simple geographic clustering based on latitude bands.
    
    :param locations_df: DataFrame with location data
    :param target_size: target number of locations per cluster
    :return: DataFrame with additional 'cluster_id' column
    """
    # sort by latitude
    sorted_df = locations_df.sort('latitude')
    
    # calculate number of clusters
    n_clusters = max(1, len(sorted_df) // target_size)
    locations_per_cluster = len(sorted_df) // n_clusters
    remainder = len(sorted_df) % n_clusters
    
    # assign cluster IDs
    cluster_ids = []
    current_cluster = 0
    locations_in_current = 0
    
    for i in range(len(sorted_df)):
        cluster_ids.append(current_cluster)
        locations_in_current += 1
        
        # determine when to move to next cluster
        cluster_size_limit = locations_per_cluster + (1 if current_cluster < remainder else 0)
        
        if locations_in_current >= cluster_size_limit and current_cluster < n_clusters - 1:
            current_cluster += 1
            locations_in_current = 0
    
    return sorted_df.with_columns(
        pl.Series('cluster_id', cluster_ids)
    )


def balance_cluster_sizes(
    clustered_df: pl.DataFrame, 
    min_size: int, 
    max_size: int
) -> pl.DataFrame:
    """
    Balance cluster sizes to respect min/max constraints.
    
    :param clustered_df: DataFrame with 'cluster_id' column
    :param min_size: minimum locations per cluster
    :param max_size: maximum locations per cluster
    :return: DataFrame with rebalanced cluster assignments
    """
    # get cluster sizes
    cluster_sizes = (
        clustered_df
        .group_by('cluster_id')
        .agg(pl.count().alias('size'))
        .sort('cluster_id')
    )
    
    # identify problematic clusters
    oversized = cluster_sizes.filter(pl.col('size') > max_size)['cluster_id'].to_list()
    undersized = cluster_sizes.filter(pl.col('size') < min_size)['cluster_id'].to_list()
    
    if not oversized and not undersized:
        # already balanced, just convert cluster_ids to zone_ids
        return clustered_df.with_columns(
            pl.col('cluster_id').map_elements(
                lambda x: f"zone_{x:03d}", 
                return_dtype=pl.String
            ).alias('zone_id')
        ).drop('cluster_id')
    
    # collect all locations from problematic clusters
    problem_locations = clustered_df.filter(
        pl.col('cluster_id').is_in(oversized + undersized)
    ).drop('cluster_id')
    
    # keep good clusters
    good_clusters = clustered_df.filter(
        ~pl.col('cluster_id').is_in(oversized + undersized)
    )
    
    # re-cluster problematic locations
    if len(problem_locations) > 0:
        target_size = (min_size + max_size) // 2
        n_new_clusters = max(1, len(problem_locations) // target_size)
        
        # use K-means for re-clustering
        rebalanced = kmeans_cluster_locations(problem_locations, n_new_clusters)
        
        # adjust cluster IDs to not conflict with good clusters
        max_good_cluster = good_clusters['cluster_id'].max() if len(good_clusters) > 0 else -1
        rebalanced = rebalanced.with_columns(
            (pl.col('cluster_id') + max_good_cluster + 1).alias('cluster_id')
        )
        
        # combine good and rebalanced clusters
        result_df = pl.concat([good_clusters, rebalanced])
    else:
        result_df = good_clusters
    
    # convert to zone_ids
    return result_df.with_columns(
        pl.col('cluster_id').map_elements(
            lambda x: f"zone_{x:03d}", 
            return_dtype=pl.String
        ).alias('zone_id')
    ).drop('cluster_id')


def calculate_cluster_centers(
    clustered_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate geographic centers for each cluster.
    
    :param clustered_df: DataFrame with 'zone_id', 'latitude', 'longitude' columns
    :return: DataFrame with zone centers
    """
    return (
        clustered_df
        .group_by('zone_id')
        .agg([
            pl.col('latitude').mean().alias('center_latitude'),
            pl.col('longitude').mean().alias('center_longitude'),
            pl.count().alias('cluster_size')
        ])
        .sort('zone_id')
    )


def assign_primary_stores(
    clustered_df: pl.DataFrame, 
    config: Dict[str, Any]
) -> pl.DataFrame:
    """
    Randomly assign primary store status to locations within each cluster.
    
    :param clustered_df: DataFrame with 'zone_id' column
    :param config: configuration with primary_store_min and primary_store_max
    :return: DataFrame with 'class' column updated for primary stores
    """
    random.seed(config['random_seed'])
    
    result_rows = []
    
    for zone_id in clustered_df['zone_id'].unique():
        zone_locations = clustered_df.filter(pl.col('zone_id') == zone_id)
        
        # determine number of primary stores for this zone
        min_primary = config['primary_store_min']
        max_primary = min(config['primary_store_max'], len(zone_locations))
        n_primary = random.randint(min_primary, max_primary)
        
        # randomly select primary stores
        zone_data = zone_locations.to_dicts()
        if n_primary > 0:
            primary_indices = random.sample(range(len(zone_data)), n_primary)
            for i, row in enumerate(zone_data):
                if i in primary_indices:
                    row['class'] = 'primary'
                else:
                    row['class'] = 'secondary'
        else:
            # all secondary if n_primary = 0
            for row in zone_data:
                row['class'] = 'secondary'
        
        result_rows.extend(zone_data)
    
    return pl.DataFrame(result_rows)


def calculate_quality_metrics(clustered_df: pl.DataFrame) -> Dict[str, float]:
    """
    Calculate clustering quality metrics.
    
    Args:
        clustered_df: DataFrame with clustered locations
        
    Returns:
        Quality metrics dictionary
    """
    if len(clustered_df) == 0:
        return {}
    
    # Basic statistics
    cluster_sizes = (
        clustered_df
        .group_by('zone_id')
        .agg(pl.count().alias('size'))['size']
        .to_numpy()
    )
    
    metrics = {
        'n_clusters': len(cluster_sizes),
        'total_locations': len(clustered_df),
        'avg_cluster_size': float(np.mean(cluster_sizes)),
        'min_cluster_size': int(np.min(cluster_sizes)),
        'max_cluster_size': int(np.max(cluster_sizes)),
        'std_cluster_size': float(np.std(cluster_sizes))
    }
    
    # Calculate average intra-cluster distance
    intra_distances = []
    
    for zone_id in clustered_df['zone_id'].unique():
        zone_locations = clustered_df.filter(pl.col('zone_id') == zone_id)
        
        if len(zone_locations) > 1:
            coords = zone_locations.select(['latitude', 'longitude']).to_numpy()
            
            # Calculate all pairwise distances within cluster
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = haversine_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                    intra_distances.append(dist)
    
    if intra_distances:
        metrics['avg_intra_cluster_distance'] = float(np.mean(intra_distances))
        metrics['std_intra_cluster_distance'] = float(np.std(intra_distances))
    else:
        metrics['avg_intra_cluster_distance'] = 0.0
        metrics['std_intra_cluster_distance'] = 0.0
    
    return metrics


def cluster_locations(locations_df: pl.DataFrame, config: Dict[str, any]) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    Main function to cluster locations into geographic zones.
    
    Args:
        locations_df: DataFrame with location data including 'latitude', 'longitude'
        config: Clustering configuration dictionary
        
    Returns:
        Tuple of (clustered_df_with_zone_ids, quality_metrics)
    """
    logger.info(f"Clustering {len(locations_df)} locations with method: {config['method']}")
    
    # Set random seed
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    # Determine number of clusters
    target_size = (config['min_locations_per_cluster'] + config['max_locations_per_cluster']) // 2
    n_clusters = max(1, len(locations_df) // target_size)
    
    logger.info(f"Target cluster size: {target_size}, creating {n_clusters} initial clusters")
    
    # Perform initial clustering
    if config['method'] == "kmeans":
        clustered_df = kmeans_cluster_locations(locations_df, n_clusters, config['random_seed'])
    elif config['method'] == "geographic":
        clustered_df = geographic_cluster_locations(locations_df, target_size)
    else:
        logger.warning(f"Unknown method {config['method']}, using kmeans")
        clustered_df = kmeans_cluster_locations(locations_df, n_clusters, config['random_seed'])
    
    # Balance cluster sizes and assign zone_ids
    balanced_df = balance_cluster_sizes(clustered_df, config['min_locations_per_cluster'], config['max_locations_per_cluster'])
    
    # Assign primary store status
    final_df = assign_primary_stores(balanced_df, config)
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(final_df)
    
    logger.info(f"Created {quality_metrics['n_clusters']} clusters")
    logger.info(f"Cluster sizes: min={quality_metrics['min_cluster_size']}, "
               f"max={quality_metrics['max_cluster_size']}, "
               f"avg={quality_metrics['avg_cluster_size']:.1f}")
    logger.info(f"Average intra-cluster distance: {quality_metrics['avg_intra_cluster_distance']:.2f} km")
    
    # Count primary stores assigned
    primary_count = len(final_df.filter(pl.col('class') == 'primary'))
    secondary_count = len(final_df.filter(pl.col('class') == 'secondary'))
    logger.info(f"Assigned {primary_count} primary stores and {secondary_count} secondary stores")
    
    return final_df, quality_metrics


def add_zone_ids_to_jsonl_dataset(
    input_file: str,
    output_file: str,
    config: Dict[str, any] = None
) -> Dict[str, float]:
    """
    Add zone_id assignments to a JSONL location dataset.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file with zone_ids
        config: Clustering configuration dictionary
        
    Returns:
        Quality metrics dictionary
    """
    config = config or default_cluster_config()
    logger.info(f"Adding zone_ids to JSONL dataset: {input_file} -> {output_file}")
    
    # Load dataset as DataFrame
    locations_df = pl.read_ndjson(input_file)
    
    # Perform clustering
    clustered_df, quality_metrics = cluster_locations(locations_df, config)
    
    # Write to JSONL with zone_ids
    clustered_df.write_ndjson(output_file)
    
    logger.info(f"saved dataset with zone_ids to {output_file}")
    return quality_metrics


def add_zone_ids_to_json_dataset(
    input_file: str,
    output_file: str,
    config: Dict[str, any] = None
) -> Dict[str, float]:
    """
    Add zone_id assignments to a JSON location dataset.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file with zone_ids
        config: Clustering configuration dictionary
        
    Returns:
        Quality metrics dictionary
    """
    config = config or default_cluster_config()
    logger.info(f"Adding zone_ids to JSON dataset: {input_file} -> {output_file}")
    
    # Load dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract locations (handle different JSON structures)
    locations_data = None
    if 'subway_locations_california' in data:
        locations_data = data['subway_locations_california']
    elif 'subway_locations_san_francisco' in data:
        locations_data = data['subway_locations_san_francisco']
    elif isinstance(data, list):
        locations_data = data
    else:
        raise ValueError("Could not find locations in dataset")
    
    # Convert to DataFrame
    locations_df = pl.DataFrame(locations_data)
    
    # Perform clustering
    clustered_df, quality_metrics = cluster_locations(locations_df, config)
    
    # Calculate cluster centers
    cluster_centers = calculate_cluster_centers(clustered_df)
    
    # Convert back to list of dicts for JSON output
    updated_locations = clustered_df.to_dicts()
    
    # Save updated dataset
    output_data = {
        'subway_locations_california': updated_locations,
        'clustering_info': {
            'n_clusters': quality_metrics['n_clusters'],
            'avg_cluster_size': quality_metrics['avg_cluster_size'],
            'method': config['method'],
            'min_locations_per_cluster': config['min_locations_per_cluster'],
            'max_locations_per_cluster': config['max_locations_per_cluster']
        },
        'cluster_centers': {
            row['zone_id']: {
                'latitude': row['center_latitude'], 
                'longitude': row['center_longitude']
            } for row in cluster_centers.to_dicts()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"saved dataset with zone_ids to {output_file}")
    return quality_metrics


if __name__ == "__main__":
    # Configuration for California Subway locations
    config = {
        'min_locations_per_cluster': 3,
        'max_locations_per_cluster': 15,  # Good for route optimization (< 25 limit)
        'method': 'kmeans',
        'random_seed': 42,
        'primary_store_min': 0,
        'primary_store_max': 3
    }
    
    # Add zone_ids to the JSONL dataset
    quality_metrics = add_zone_ids_to_jsonl_dataset(
        input_file="data/subway_locations.jsonl",
        output_file="data/subway_locations.jsonl",
        config=config
    )
    
    print(f"clustering results:")
    print(f"created {quality_metrics['n_clusters']} zones")
    print(f"average locations per zone: {quality_metrics['avg_cluster_size']:.1f}")
    print(f"zone size range: {quality_metrics['min_cluster_size']} - {quality_metrics['max_cluster_size']}")
    print(f"average intra-cluster distance: {quality_metrics['avg_intra_cluster_distance']:.1f} km")


# ------------------------------------------------------------------------------
# end of clustering_utils.py
# ------------------------------------------------------------------------------