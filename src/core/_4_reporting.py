# standard library imports
from typing import Any, Dict

# 3rd-party imports
import polars as pl
import yaml
from loguru import logger


def load_config() -> Dict[str, Any]:
    """
    Load configuration parameters from model-params.yaml.
    
    :return: Configuration dictionary
    """
    with open("config/model-params.yaml", 'r') as f:
        return yaml.safe_load(f)['model_params']


def aggregate(
    itinerary: pl.DataFrame,
    local: bool = False
) -> pl.DataFrame:
    """
    Generate zone-level aggregate metrics from detailed daily itinerary.
    
    :param itinerary: DataFrame containing detailed daily routes for all zones
    :param local: If True, export HTML table to output/ directory
    :return: DataFrame with one row per zone containing aggregate metrics:
        - zone_id: Zone identifier
        - primary_pos_count: Count of primary locations in zone
        - secondary_pos_count: Count of secondary locations in zone
        - weekly_duration: Sum of all daily durations (hours)
        - utilization: (weekly_duration / target_capacity) * 100
        - overutilized_days: Count of days where duration > hours_per_day
        - underutilized_days: Count of days where duration < hours_per_day
        - total_pos_time: Total time spent at locations (service time, hours)
        - total_drive_time: Total time spent driving between locations (hours)
        - sec_std: Standard deviation of duration for all days with secondary class stores
    """
    logger.info("generating zone-level aggregate metrics")
    
    # load config
    config = load_config()
    days_per_week = config.get("days_per_week", 5)
    hours_per_day = config.get("hours_per_day", 8)
    hours_per_non_primary = config.get("hours_per_non_primary", 1)
    
    # calculate target weekly capacity in hours
    target_weekly_hours = days_per_week * hours_per_day
    daily_capacity_hours = hours_per_day
    
    # group by zone and calculate metrics
    zone_metrics = []
    
    for zone_id in itinerary['zone_id'].unique():
        zone_data = itinerary.filter(pl.col('zone_id') == zone_id)
        
        # count primary and secondary locations
        primary_count = 0
        secondary_count = 0
        total_pos_time = 0.0
        total_drive_time = 0.0
        
        # collect all unique locations across all days
        all_pos_ids = set()
        all_pos_classes = []
        
        for row in zone_data.iter_rows(named=True):
            pos_ids = row['pos_id'] or []
            pos_classes = row['pos_class'] or []
            # use schedule for individual position durations if available, otherwise estimate
            schedule = row.get('schedule', [])
            day_duration = row['duration'] or 0.0
            
            # track unique locations and their classes
            for pos_id, pos_class in zip(pos_ids, pos_classes):
                if pos_id not in all_pos_ids:
                    all_pos_ids.add(pos_id)
                    all_pos_classes.append(pos_class)
            
            # sum position time (service time at locations) - convert minutes to hours
            # estimate 60 minutes per location (excluding centroid)
            estimated_service_time = len([p for p in pos_ids if p != -1]) * 60.0
            total_pos_time += estimated_service_time / 60.0
            
            # calculate drive time (total duration - service time) - convert minutes to hours
            service_time_for_day = estimated_service_time
            drive_time_for_day = max(0.0, day_duration - service_time_for_day)
            total_drive_time += drive_time_for_day / 60.0
        
        # count primary and secondary from unique locations
        for pos_class in all_pos_classes:
            if pos_class == 'primary':
                primary_count += 1
            elif pos_class == 'secondary':
                secondary_count += 1
        
        # calculate weekly metrics - convert minutes to hours
        weekly_duration = zone_data['duration'].sum() / 60.0
        utilization = (weekly_duration / target_weekly_hours) * 100 if target_weekly_hours > 0 else 0
        
        # count over/under-utilized days and collect secondary days durations
        overutilized_days = 0
        underutilized_days = 0
        secondary_day_durations = []
        
        for row in zone_data.iter_rows(named=True):
            day_duration_minutes = row['duration'] or 0.0
            day_duration_hours = day_duration_minutes / 60.0
            pos_ids = row['pos_id'] or []
            pos_classes = row['pos_class'] or []
            
            # check if this day has any secondary class stores
            has_secondary = 'secondary' in pos_classes
            if has_secondary:
                secondary_day_durations.append(day_duration_hours)
            
            # over-utilized: duration exceeds daily capacity
            if day_duration_hours > daily_capacity_hours:
                overutilized_days += 1
            
            # under-utilized: duration is less than hours_per_day (if has any locations)
            if len(pos_ids) > 0 and day_duration_hours < daily_capacity_hours:
                underutilized_days += 1
        
        # calculate standard deviation for secondary days
        sec_std = 0.0
        if len(secondary_day_durations) > 1:
            # calculate standard deviation manually
            mean_duration = sum(secondary_day_durations) / len(secondary_day_durations)
            variance = sum((x - mean_duration) ** 2 for x in secondary_day_durations) / len(secondary_day_durations)
            sec_std = variance ** 0.5
        
        zone_metrics.append({
            'zone_id': zone_id,
            'primary_pos_count': primary_count,
            'secondary_pos_count': secondary_count,
            'weekly_duration': weekly_duration,
            'utilization': round(utilization, 2),
            'overutilized_days': overutilized_days,
            'underutilized_days': underutilized_days,
            'total_pos_time': round(total_pos_time, 2),
            'total_drive_time': round(total_drive_time, 2),
            'sec_std': round(sec_std, 2)
        })
    
    # create DataFrame from metrics
    aggregate_df = pl.DataFrame(zone_metrics)
    
    # sort by zone_id for consistent output
    aggregate_df = aggregate_df.sort('zone_id')
    
    logger.success(f"generated aggregate metrics for {len(aggregate_df)} zones")
        
    return aggregate_df


if __name__ == "__main__":
    # example usage and testing
    logger.info("testing report.aggregate() function")
    
    # create mock itinerary data for testing
    mock_data = [
        {
            'zone_id': 'zone_001',
            'day': 1,
            'pos_id': [101],
            'pos_locations': [[-122.0, 37.0]], 
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
            'pos_class': ['secondary', 'secondary'],
            'route': [[-122.1, 37.1], [-122.2, 37.2]],
            'schedule': [0.0, 60.0, 120.0],
            'duration': 135.0
        }
    ]
    
    mock_itinerary = pl.DataFrame(mock_data)
    logger.info("mock itinerary:")
    logger.info(mock_itinerary)
    
    # test aggregate function
    result = aggregate(mock_itinerary)
    logger.info("aggregate result:")
    logger.info(result)


# ------------------------------------------------------------------------------
# end of report.py
# ------------------------------------------------------------------------------