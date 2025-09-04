"""
Core Reporting Module

Handles aggregation and reporting of route optimization results.
Generates zone-level metrics from detailed daily itineraries.
"""

import yaml
from typing import Dict, Any
import polars as pl
from loguru import logger


def load_config() -> Dict[str, Any]:
    """Load configuration parameters from model-params.yaml."""
    with open("config/model-params.yaml", 'r') as f:
        return yaml.safe_load(f)['model_params']


def aggregate(itinerary: pl.DataFrame, local: bool = False) -> pl.DataFrame:
    """
    Generate zone-level aggregate metrics from detailed daily itinerary.
    
    Args:
        itinerary: DataFrame containing detailed daily routes for all zones
        local: If True, export HTML table to output/ directory
        
    Returns:
        DataFrame with one row per zone containing aggregate metrics:
        - zone_id: Zone identifier
        - primary_pos_count: Count of primary locations in zone
        - secondary_pos_count: Count of secondary locations in zone  
        - weekly_duration: Sum of all daily durations (hours)
        - utilization: (weekly_duration / target_capacity) * 100
        - overutilized_days: Count of days where duration > hours_per_day
        - underutilized_days: Count of days where duration < hours_per_day
        - total_pos_time: Total time spent at locations (service time, hours)
        - total_drive_time: Total time spent driving between locations (hours)
    """
    logger.info("Generating zone-level aggregate metrics")
    
    # Load config
    config = load_config()
    days_per_week = config.get("days_per_week", 5)
    hours_per_day = config.get("hours_per_day", 8)
    hours_per_non_primary = config.get("hours_per_non_primary", 1)
    
    # Calculate target weekly capacity in hours
    target_weekly_hours = days_per_week * hours_per_day
    daily_capacity_hours = hours_per_day
    
    # Group by zone and calculate metrics
    zone_metrics = []
    
    for zone_id in itinerary['zone_id'].unique():
        zone_data = itinerary.filter(pl.col('zone_id') == zone_id)
        
        # Count primary and secondary locations
        primary_count = 0
        secondary_count = 0
        total_pos_time = 0.0
        total_drive_time = 0.0
        
        # Collect all unique locations across all days
        all_pos_ids = set()
        all_pos_classes = []
        
        for row in zone_data.iter_rows(named=True):
            pos_ids = row['pos_id'] or []
            pos_classes = row['pos_class'] or []
            pos_durations = row['pos_duration'] or []
            day_duration = row['duration'] or 0.0
            
            # Track unique locations and their classes
            for pos_id, pos_class in zip(pos_ids, pos_classes):
                if pos_id not in all_pos_ids:
                    all_pos_ids.add(pos_id)
                    all_pos_classes.append(pos_class)
            
            # Sum position time (service time at locations) - convert minutes to hours
            if pos_durations:
                total_pos_time += sum(pos_durations) / 60.0
            
            # Calculate drive time (total duration - service time) - convert minutes to hours
            service_time_for_day = sum(pos_durations) if pos_durations else 0.0
            drive_time_for_day = max(0.0, day_duration - service_time_for_day)
            total_drive_time += drive_time_for_day / 60.0
        
        # Count primary and secondary from unique locations
        for pos_class in all_pos_classes:
            if pos_class == 'primary':
                primary_count += 1
            elif pos_class == 'secondary':
                secondary_count += 1
        
        # Calculate weekly metrics - convert minutes to hours
        weekly_duration = zone_data['duration'].sum() / 60.0
        utilization = (weekly_duration / target_weekly_hours) * 100 if target_weekly_hours > 0 else 0
        
        # Count over/under-utilized days
        overutilized_days = 0
        underutilized_days = 0
        
        for row in zone_data.iter_rows(named=True):
            day_duration_minutes = row['duration'] or 0.0
            day_duration_hours = day_duration_minutes / 60.0
            pos_ids = row['pos_id'] or []
            
            # Over-utilized: duration exceeds daily capacity
            if day_duration_hours > daily_capacity_hours:
                overutilized_days += 1
            
            # Under-utilized: duration is less than hours_per_day (if has any locations)
            if len(pos_ids) > 0 and day_duration_hours < daily_capacity_hours:
                underutilized_days += 1
        
        zone_metrics.append({
            'zone_id': zone_id,
            'primary_pos_count': primary_count,
            'secondary_pos_count': secondary_count,
            'weekly_duration': weekly_duration,
            'utilization': round(utilization, 2),
            'overutilized_days': overutilized_days,
            'underutilized_days': underutilized_days,
            'total_pos_time': round(total_pos_time, 2),
            'total_drive_time': round(total_drive_time, 2)
        })
    
    # Create DataFrame from metrics
    aggregate_df = pl.DataFrame(zone_metrics)
    
    # Sort by zone_id for consistent output
    aggregate_df = aggregate_df.sort('zone_id')
    
    logger.success(f"Generated aggregate metrics for {len(aggregate_df)} zones")
    
    # Export HTML table if local is True
    if local:
        from ..utils import html_utils
        html_utils.create_aggregate_report_html(aggregate_df, itinerary, "output/aggregate_report.html")
    
    return aggregate_df


def zone(itinerary: pl.DataFrame, local: bool = True) -> None:
    """
    Generate individual zone reports as HTML files.
    
    Args:
        itinerary: DataFrame containing detailed daily routes for all zones
        local: If True, export HTML files to output/zone/ directory
    """
    if not local:
        logger.info("Local flag is False, skipping zone report generation")
        return
        
    logger.info("Generating individual zone reports")
    
    # Get unique zones
    zones = itinerary['zone_id'].unique().sort()
    logger.info(f"Found {len(zones)} zones to process: {list(zones)}")
    
    for zone_id in zones:
        logger.info(f"Generating report for {zone_id}")
        
        # Filter itinerary for this zone
        zone_itinerary = itinerary.filter(pl.col('zone_id') == zone_id)
        
        # Generate individual zone report HTML
        from ..utils import html_utils
        output_path = f"output/zone/{zone_id}_report.html"
        html_utils.create_individual_zone_report(zone_itinerary, zone_id, output_path)
        
        logger.success(f"Generated report for {zone_id} at {output_path}")
    
    logger.success(f"Generated individual zone reports for {len(zones)} zones")


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing report.aggregate() function")
    
    # Create mock itinerary data for testing
    mock_data = [
        {
            'zone_id': 'zone_001',
            'day': 1,
            'pos_id': [101],
            'pos_locations': [[-122.0, 37.0]], 
            'pos_duration': [480],
            'pos_class': ['primary'],
            'route': [[-122.0, 37.0]],
            'schedule': [0.0, 480.0],
            'duration': 480.0
        },
        {
            'zone_id': 'zone_001', 
            'day': 2,
            'pos_id': [102, 103],
            'pos_locations': [[-122.1, 37.1], [-122.2, 37.2]],
            'pos_duration': [60, 60], 
            'pos_class': ['secondary', 'secondary'],
            'route': [[-122.1, 37.1], [-122.2, 37.2]],
            'schedule': [0.0, 60.0, 75.0, 135.0],
            'duration': 135.0
        }
    ]
    
    mock_itinerary = pl.DataFrame(mock_data)
    logger.info("Mock itinerary:")
    logger.info(mock_itinerary)
    
    # Test aggregate function
    result = aggregate(mock_itinerary)
    logger.info("Aggregate result:")
    logger.info(result)


# ------------------------------------------------------------------------------
# end of report.py
# ------------------------------------------------------------------------------