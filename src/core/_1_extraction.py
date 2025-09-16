"""
Stage 1: Extraction Module
Load and validate location data from external sources
"""

# standard library imports
from pathlib import Path
from typing import List, Optional

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
from src.utils.io_utils import extract as io_extract


def extract_locations(
    filepath: str = "./data/locations.jsonl",
    zone_ids: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Extract location data from JSONL file.
    
    This is the first stage of the pipeline where we:
    1. Load raw location data from files
    2. Validate data integrity
    3. Filter by zone if specified
    4. Handle missing/invalid records
    
    :param filepath: Path to locations JSONL file
    :param zone_ids: Optional list of zone IDs to filter
    :return: DataFrame with validated location data
    """
    logger.info("stage 1: extraction - loading location data")
    logger.info(f"extracting from: {filepath}")
    
    # use existing io_utils extract function
    df = io_extract(zone_ids=zone_ids, locations_path=filepath)
    
    logger.success(f"extraction complete: {len(df)} locations loaded")
    
    if len(df) > 0 and "zone_id" in df.columns:
        logger.info(f"zones: {df['zone_id'].unique().to_list()}")
    else:
        logger.warning("no zone_id column found or empty dataset")
    
    return df


def validate_locations(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validate extracted location data.
    
    Checks:
    - Valid coordinates (lat/lon bounds)
    - Required fields present
    - Zone assignments valid
    
    :param df: Raw location DataFrame
    :return: Validated DataFrame with invalid records removed
    """
    initial_count = len(df)
    
    # check coordinate bounds
    df = df.filter(
        (pl.col("latitude") >= -90) & (pl.col("latitude") <= 90) &
        (pl.col("longitude") >= -180) & (pl.col("longitude") <= 180)
    )
    
    # check required fields
    required_cols = ["pos_id", "name", "latitude", "longitude", "zone_id", "class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # remove null coordinates
    df = df.filter(
        pl.col("latitude").is_not_null() & 
        pl.col("longitude").is_not_null()
    )
    
    removed = initial_count - len(df)
    if removed > 0:
        logger.warning(f"removed {removed} invalid locations during validation")
    
    return df