"""
Core Data Processing Pipeline

Handles the primary processing stages from the architecture:
- Stage 0: Data Ingestion (JSON/JSONL to Polars DataFrame)
- Stage 1: OSRM OD Matrix Generation
- Stage 2: Primary Location Assignment to Days
- Stage 3: Secondary Location Clustering to Days
"""

import json
import polars as pl
from typing import Dict, List, Tuple, Any
from loguru import logger

from ..utils.osrm_utils import fetch_od_matrix, convert_locations_from_polars, od_matrix_to_polars
from .stage1_assignment import assign_days_to_secondary_locations


def load_locations_from_jsonl(locations_path: str, zone_id: str = None) -> pl.DataFrame:
    """
    Stage 0: Data Ingestion - Load location data from JSONL file and return as Polars DataFrame.
    
    Args:
        locations_path: Path to JSONL or JSON file
        zone_id: Optional zone filter
        
    Returns:
        Polars DataFrame with standardized location schema
    """
    data = []
    
    if locations_path.endswith('.jsonl'):
        with open(locations_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    loc_data = json.loads(line)
                    
                    # Filter by zone_id if specified
                    if zone_id and 'zone_id' in loc_data:
                        if loc_data.get('zone_id') != zone_id:
                            continue
                    
                    data.append({
                        'location_id': loc_data['id'],
                        'zone_id': loc_data.get('zone_id', zone_id or 'default'),
                        'name': loc_data['name'],
                        'location_class': loc_data['class'],
                        'address': loc_data['address'],
                        'latitude': loc_data['latitude'],
                        'longitude': loc_data['longitude'],
                        'source_system': 'jsonl_file'
                    })
    else:
        # Handle traditional JSON format
        with open(locations_path, 'r') as f:
            json_data = json.load(f)
        
        locations_key = None
        if 'subway_locations_california' in json_data:
            locations_key = 'subway_locations_california'
        elif 'subway_locations_san_francisco' in json_data:
            locations_key = 'subway_locations_san_francisco'
        else:
            raise ValueError("Could not find locations in dataset")
        
        for loc_data in json_data[locations_key]:
            if zone_id and 'zone_id' in loc_data:
                if loc_data.get('zone_id') != zone_id:
                    continue
            
            data.append({
                'location_id': loc_data['id'],
                'zone_id': loc_data.get('zone_id', zone_id or 'default'),
                'name': loc_data['name'],
                'location_class': loc_data['class'],
                'address': loc_data['address'],
                'latitude': loc_data['latitude'],
                'longitude': loc_data['longitude'],
                'source_system': 'json_file'
            })
    
    return pl.DataFrame(data)


def separate_primary_secondary_locations(locations_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Separate locations into primary and secondary DataFrames.
    
    Args:
        locations_df: Combined location DataFrame
        
    Returns:
        Tuple of (primary_df, secondary_df)
    """
    primary_df = locations_df.filter(pl.col('location_class') == 'primary')
    secondary_df = locations_df.filter(pl.col('location_class') == 'secondary')
    return primary_df, secondary_df


def generate_od_matrix_for_zone(zone_id: str, secondary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Stage 1: OSRM OD Matrix Generation - Get OD matrix for secondary locations in a zone.
    
    Args:
        zone_id: Zone identifier for OSRM API
        secondary_df: DataFrame containing secondary locations
        
    Returns:
        Polars DataFrame containing OD matrix with drive times and distances
    """
    locations = convert_locations_from_polars(secondary_df)
    if not locations:
        # Return empty DataFrame with correct schema
        return pl.DataFrame({
            'zone_id': [],
            'origin_id': [],
            'destination_id': [],
            'distance_meters': [],
            'duration_seconds': [],
            'duration_minutes': [],
            'osrm_response_code': [],
            'api_call_timestamp': []
        })
    
    od_result = fetch_od_matrix(zone_id, locations)
    return od_matrix_to_polars(od_result)


def assign_primary_locations_to_days(primary_df: pl.DataFrame) -> Dict[int, int]:
    """
    Stage 2: Primary Location Assignment - Assign primary locations to days.
    Simple assignment: one primary location per day.
    
    Args:
        primary_df: DataFrame containing primary locations
        
    Returns:
        Dictionary mapping day -> primary_location_id
    """
    primary_assignments = {}
    primary_locations = primary_df.to_dicts()
    
    for i, location in enumerate(primary_locations):
        day = i + 1  # Days start from 1
        primary_assignments[day] = location['location_id']
    
    return primary_assignments


def calculate_available_secondary_days(primary_count: int, days_per_week: int) -> int:
    """Calculate how many days are available for secondary locations."""
    return max(0, days_per_week - primary_count)


def assign_secondary_locations_to_days(
    secondary_df: pl.DataFrame,
    zone_id: str,
    available_secondary_days: int,
    max_locations_per_day: int = 7,
    use_swap_optimization: bool = True
) -> Dict[int, List[int]]:
    """
    Stage 3: Secondary Location Clustering - Assign secondary locations to available days using clustering.
    
    Args:
        secondary_df: DataFrame containing secondary location data
        zone_id: Zone identifier
        available_secondary_days: Number of days available for secondary locations
        max_locations_per_day: Maximum locations per day
        use_swap_optimization: Whether to apply swap optimization
        
    Returns:
        Dictionary mapping day_id to list of location_ids
    """
    return assign_days_to_secondary_locations(
        secondary_df=secondary_df,
        zone_id=zone_id,
        available_secondary_days=available_secondary_days,
        max_locations_per_day=max_locations_per_day,
        use_swap_optimization=use_swap_optimization
    )


def create_zone_optimization_package(
    locations_df: pl.DataFrame,
    config: Dict[str, Any],
    zone_id: str
) -> Dict[str, Any]:
    """
    Create complete zone optimization package with all processed data.
    
    This function orchestrates all primary processing stages:
    1. Data separation (primary/secondary)
    2. OD matrix generation 
    3. Primary location assignment
    4. Secondary location clustering
    
    Args:
        locations_df: DataFrame containing all location data for the zone
        config: Configuration dictionary
        zone_id: Zone identifier
        
    Returns:
        Dictionary containing all processed data for route optimization
    """
    logger.info(f"Creating zone optimization package for {zone_id}")
    
    # Stage 0: Separate primary and secondary locations
    primary_df, secondary_df = separate_primary_secondary_locations(locations_df)
    
    # Stage 2: Assign primary locations to days
    primary_assignments = assign_primary_locations_to_days(primary_df)
    
    # Calculate available days for secondary locations
    available_secondary_days = calculate_available_secondary_days(
        len(primary_assignments), config['days_per_week']
    )
    
    logger.info(f"Zone {zone_id}: {len(primary_df)} primary, {len(secondary_df)} secondary locations")
    logger.info(f"Zone {zone_id}: {available_secondary_days} days available for secondary locations")
    
    if available_secondary_days <= 0:
        # No days available for secondary locations
        secondary_assignments = {}
        od_matrix_df = pl.DataFrame()
    else:
        # Stage 1: Generate OD matrix for secondary locations
        od_matrix_df = generate_od_matrix_for_zone(zone_id, secondary_df)
        
        # Stage 3: Assign secondary locations to available days
        secondary_clusters = assign_secondary_locations_to_days(
            secondary_df=secondary_df,
            zone_id=zone_id,
            available_secondary_days=available_secondary_days,
            max_locations_per_day=config['locations_per_day_max'],
            use_swap_optimization=True
        )
        
        # Convert cluster IDs to actual day numbers
        secondary_assignments = {}
        available_days = [d for d in range(1, config['days_per_week'] + 1) 
                        if d not in primary_assignments]
        
        for cluster_id, location_ids in secondary_clusters.items():
            if cluster_id - 1 < len(available_days):  # cluster_id starts from 1
                day = available_days[cluster_id - 1]
                secondary_assignments[day] = location_ids
    
    return {
        'zone_id': zone_id,
        'config': config,
        'locations_df': locations_df,
        'primary_df': primary_df,
        'secondary_df': secondary_df,
        'od_matrix_df': od_matrix_df,
        'primary_assignments': primary_assignments,
        'secondary_assignments': secondary_assignments,
        'available_secondary_days': available_secondary_days
    }


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open("config/model-params.yaml", 'r') as f:
        config = yaml.safe_load(f)['model_params']
    
    zone_id = "test_zone"
    locations_df = load_locations_from_jsonl("data/subway_locations.json", zone_id)
    
    logger.info("Starting data processing pipeline...")
    optimization_package = create_zone_optimization_package(locations_df, config, zone_id)
    
    logger.info(f"Zone optimization package created:")
    logger.info(f"- Primary assignments: {optimization_package['primary_assignments']}")
    logger.info(f"- Secondary assignments: {optimization_package['secondary_assignments']}")
    logger.info(f"- OD matrix rows: {len(optimization_package['od_matrix_df'])}")