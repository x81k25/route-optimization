# standard library imports
from typing import List, Optional

# 3rd-party imports
import polars as pl
from loguru import logger

# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

def extract(
    zone_ids: Optional[List[str]] = None,
    locations_path: str = "./data/locations.jsonl"
) -> pl.DataFrame:
    """
    Extract location data from JSONL file and optionally filter by zone IDs.
    
    :param zone_ids: optional list of zone IDs to filter by
    :param locations_path: path to the locations JSONL file
    :return: polars DataFrame containing location data
    """
    logger.info(f"extracting location data from {locations_path}")
    
    # read the JSONL file using polars
    try:
        df = pl.read_ndjson(locations_path)
        logger.info(f"loaded {len(df)} locations from file")
    except Exception as e:
        logger.error(f"failed to read {locations_path}: {e}")
        return pl.DataFrame()
    
    # filter out entries with null zone_ids
    original_count = len(df)
    df = df.filter(pl.col('zone_id').is_not_null())
    null_count = original_count - len(df)
    if null_count > 0:
        logger.info(f"filtered out {null_count} entries with null zone_ids")
    
    # filter by zone_ids if provided
    if zone_ids:
        original_count = len(df)
        # get all unique zones in the data
        available_zones = set(df['zone_id'].drop_nulls().unique().to_list())
        requested_zones = set(zone_ids)
        
        # find valid and invalid zones
        valid_zones = requested_zones.intersection(available_zones)
        invalid_zones = requested_zones - available_zones
        
        if invalid_zones:
            logger.warning(f"the following zone_ids do not exist in the data: {sorted(invalid_zones)}")
        
        if valid_zones:
            logger.info(f"filtering for valid zones: {sorted(valid_zones)}")
            df = df.filter(pl.col('zone_id').is_in(list(valid_zones)))
            logger.info(f"filtered to {len(df)} locations from {original_count} total")
        else:
            logger.error(f"none of the requested zones exist: {zone_ids}")
            logger.info(f"available zones: {sorted(available_zones)}")
            return pl.DataFrame()
    
    # validate required columns exist (using pos_id instead of id)
    required_columns = ['pos_id', 'name', 'class', 'latitude', 'longitude', 'address']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"missing required columns: {missing_columns}")
        return pl.DataFrame()
    
    logger.info(f"successfully extracted {len(df)} locations")
    return df

# ------------------------------------------------------------------------------
# end of io_utils.py
# ------------------------------------------------------------------------------