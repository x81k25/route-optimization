"""
Stage 3.1: Primary Day Assignment
Assign primary (high-value) locations to dedicated days
"""

import polars as pl
from typing import Dict, List, Tuple
from loguru import logger
import yaml


def assign_primary_days(
    df: pl.DataFrame,
    zone_id: str,
    config_path: str = "./config/model-params.yaml"
) -> Tuple[Dict[int, int], List[int], pl.DataFrame, pl.DataFrame]:
    """
    Assign primary locations to dedicated days.
    
    This is substage 3.1 where we:
    1. Identify primary locations
    2. Calculate hours per primary
    3. Distribute across available days
    
    Args:
        df: Location DataFrame for zone
        zone_id: Zone identifier
        config_path: Path to configuration file
        
    Returns:
        Tuple of (primary_assignments, available_days, primary_df, secondary_df)
    """
    logger.info(f"Stage 3.1: PRIMARY DAY ASSIGNMENT - Zone {zone_id}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_params = config["model_params"]
    days_per_week = model_params["days_per_week"]
    hours_per_day = model_params["hours_per_day"]
    primary_hours_per_week = model_params["primary_hours_per_week"]
    
    # Separate primary and secondary locations
    primary_df = df.filter(pl.col("class") == "primary")
    secondary_df = df.filter(pl.col("class") == "secondary")
    
    logger.info(f"Found {len(primary_df)} primary, {len(secondary_df)} secondary locations")
    
    # Calculate hours per primary location
    num_primary = len(primary_df)
    if num_primary > 0:
        hours_per_primary = primary_hours_per_week / num_primary
        logger.info(f"Allocating {hours_per_primary:.1f} hours per primary location")
    else:
        hours_per_primary = 0
    
    # Assign primary locations to days
    primary_assignments = {}
    days_used = []
    
    if num_primary > 0:
        current_day = 1
        current_hours = 0
        
        for row in primary_df.iter_rows(named=True):
            pos_id = row["pos_id"]
            remaining_hours = hours_per_primary
            
            while remaining_hours > 0 and current_day <= days_per_week:
                available_hours = hours_per_day - current_hours
                hours_to_assign = min(remaining_hours, available_hours)
                
                primary_assignments[pos_id] = current_day
                days_used.append(current_day)
                
                current_hours += hours_to_assign
                remaining_hours -= hours_to_assign
                
                if current_hours >= hours_per_day or remaining_hours == 0:
                    current_day += 1
                    current_hours = 0
    
    # Determine available days for secondary locations
    all_days = set(range(1, days_per_week + 1))
    primary_days = set(days_used)
    available_days = sorted(list(all_days - primary_days))
    
    if not available_days:
        available_days = list(range(1, days_per_week + 1))
    
    logger.success(f"Primary assignment complete: {len(primary_days)} days used, {len(available_days)} available for secondary")
    
    return primary_assignments, available_days, primary_df, secondary_df