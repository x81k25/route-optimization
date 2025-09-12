"""
Stage 2: Preprocessing Module
Clean, normalize, and prepare data for optimization
"""

# standard library imports
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
from src.utils.geo_utils import get_centroid
from src.utils.osrm_utils import generate_od_matrix


def preprocess_zone_data(
    df: pl.DataFrame,
    zone_id: str
) -> Tuple[pl.DataFrame, Tuple[float, float], Dict[Tuple[int, int], float]]:
    """
    Preprocess data for a single zone.
    
    This is the second stage of the pipeline where we:
    1. Calculate zone centroid
    2. Generate distance matrix via OSRM
    3. Prepare data structures for optimization
    
    :param df: Location DataFrame for the zone
    :param zone_id: Zone identifier
    :return: Tuple of (filtered_df, centroid, distance_matrix)
    """
    logger.info(f"Stage 2: PREPROCESSING - Zone {zone_id}")
    
    # filter to specific zone
    zone_df = df.filter(pl.col("zone_id") == zone_id)
    logger.info(f"Processing {len(zone_df)} locations for zone {zone_id}")
    
    # calculate centroid
    centroid = calculate_zone_centroid(zone_df)
    logger.info(f"Centroid calculated: ({centroid[0]:.4f}, {centroid[1]:.4f})")
    
    # generate distance matrix
    od_matrix = build_distance_matrix(zone_df, zone_id, centroid)
    logger.info(f"Distance matrix generated: {len(od_matrix)} pairs")
    
    return zone_df, centroid, od_matrix


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


def normalize_location_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize and clean location data.
    
    Operations:
    - Standardize column names
    - Handle missing values
    - Ensure consistent data types
    
    :param df: Raw location DataFrame
    :return: Normalized DataFrame
    """
    # ensure consistent column types
    df = df.with_columns([
        pl.col("pos_id").cast(pl.Int64),
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
        pl.col("zone_id").cast(pl.Utf8),
        pl.col("class").cast(pl.Utf8),
    ])
    
    # fill missing addresses with empty string
    if "address" in df.columns:
        df = df.with_columns(
            pl.col("address").fill_null("")
        )
    
    return df


def group_by_zones(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Group locations by zone for parallel processing.
    
    :param df: Location DataFrame with zone assignments
    :return: Dictionary mapping zone_id to location DataFrame
    """
    zones = {}
    for zone_id in df["zone_id"].unique().to_list():
        if zone_id is not None:
            zones[zone_id] = df.filter(pl.col("zone_id") == zone_id)
    
    logger.info(f"Grouped into {len(zones)} zones for processing")
    return zones