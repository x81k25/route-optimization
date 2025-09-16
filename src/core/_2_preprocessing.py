"""
Stage 2: Preprocessing Module
Clean, normalize, and prepare data for optimization
"""

# standard library imports
from typing import Dict, List, Optional, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl




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
    
    logger.info(f"grouped into {len(zones)} zones for processing")
    return zones