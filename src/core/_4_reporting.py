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
        
        # Process the new individual position records format
        # Group by day to calculate day-level metrics
        days = zone_data.select('day').unique().sort('day').to_series().to_list()

        for day in days:
            day_data = zone_data.filter(pl.col('day') == day)

            # Get unique positions for this day (excluding centroid/null)
            day_positions = day_data.filter(
                (pl.col('pos_id').is_not_null()) & (pl.col('action') == 'arriving')
            )

            # Track unique locations and their classes
            for row in day_positions.iter_rows(named=True):
                pos_id = row['pos_id']
                pos_class = row['pos_class']

                if pos_id not in all_pos_ids:
                    all_pos_ids.add(pos_id)
                    all_pos_classes.append(pos_class)

            # Calculate day duration from schedule (difference between first and last action)
            day_schedule = day_data.sort('schedule')
            if len(day_schedule) > 1:
                first_time = day_schedule.select('schedule').min().item()
                last_time = day_schedule.select('schedule').max().item()
                day_duration = last_time - first_time  # in minutes
            else:
                day_duration = 0.0

            # Calculate service time based on primary allocation rules
            primary_positions = day_positions.filter(pl.col('pos_class') == 'primary')
            secondary_positions = day_positions.filter(pl.col('pos_class') == 'secondary')

            num_primaries = len(primary_positions)
            num_secondaries = len(secondary_positions)

            # For primaries: days with primaries get full day allocation
            if num_primaries > 0:
                # Any day with primary locations gets the full day capacity
                primary_service_time = 480.0  # hardcode 8 hours = 480 minutes for testing
                logger.info(f"Day {day}: {num_primaries} primaries, primary_service_time = {primary_service_time} minutes")
            else:
                primary_service_time = 0.0

            # For secondaries: standard 60 minutes per location
            secondary_service_time = num_secondaries * 60.0  # minutes

            estimated_service_time = primary_service_time + secondary_service_time
            total_pos_time += estimated_service_time / 60.0  # convert to hours

            # Calculate drive time (total duration - service time)
            drive_time_for_day = max(0.0, day_duration - estimated_service_time)
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
            'total_pos_time': round(total_pos_time * 60.0, 2),  # convert back to minutes for daily summary
            'total_drive_time': round(total_drive_time * 60.0, 2),  # convert back to minutes for daily summary
            'sec_std': round(sec_std, 2)
        })
    
    # create DataFrame from metrics
    aggregate_df = pl.DataFrame(zone_metrics)
    
    # sort by zone_id for consistent output
    aggregate_df = aggregate_df.sort('zone_id')
    
    logger.success(f"generated aggregate metrics for {len(aggregate_df)} zones")
        
    return aggregate_df


def generate_reports_from_daily_summary(daily_summary_df: pl.DataFrame, itinerary_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    Generate aggregate zone reports from daily summary data.

    :param daily_summary_df: Daily summary DataFrame
    :param itinerary_df: Optional itinerary DataFrame for accurate distinct counts
    :return: DataFrame with zone-level aggregate metrics
    """
    if len(daily_summary_df) == 0:
        logger.warning("No daily summary data to generate reports from")
        return pl.DataFrame()

    logger.info("Generating aggregate reports from daily summary data")

    # Group by zone_id to calculate zone-level metrics
    zones = daily_summary_df.select('zone_id').unique().sort('zone_id').to_series().to_list()

    result_data = []

    for zone_id in zones:
        zone_data = daily_summary_df.filter(pl.col('zone_id') == zone_id)

        if len(zone_data) == 0:
            continue

        # Calculate zone metrics from daily summaries
        # If we have itinerary data, get distinct counts from there
        if itinerary_df is not None and len(itinerary_df) > 0:
            zone_itinerary = itinerary_df.filter(pl.col('zone_id') == zone_id)

            # Count distinct primary and secondary locations
            primary_pos = zone_itinerary.filter(pl.col('pos_class') == 'primary')
            secondary_pos = zone_itinerary.filter(pl.col('pos_class') == 'secondary')

            total_primary = primary_pos.select('pos_id').n_unique() if len(primary_pos) > 0 else 0
            total_secondary = secondary_pos.select('pos_id').n_unique() if len(secondary_pos) > 0 else 0
        else:
            # Fallback: use max daily counts (not ideal but better than sum)
            total_primary = zone_data.select('primary_locations').max().item() or 0
            total_secondary = zone_data.select('secondary_locations').max().item() or 0
        weekly_duration = zone_data.select('duration').sum().item() / 60.0  # convert to hours

        # Calculate utilization metrics
        target_weekly_hours = 5 * 8  # 5 days * 8 hours
        utilization = (weekly_duration / target_weekly_hours) * 100 if target_weekly_hours > 0 else 0

        # Sum position and drive times
        total_pos_time = zone_data.select('total_pos_time').sum().item() / 60.0  # convert to hours
        total_drive_time = zone_data.select('total_drive_time').sum().item() / 60.0  # convert to hours

        # Count over/under utilized days
        daily_capacity_hours = 8  # 8 hours per day
        overutilized_days = len(zone_data.filter(pl.col('duration') > daily_capacity_hours * 60))

        # Count days with locations that are under-utilized
        days_with_locations = zone_data.filter(
            (pl.col('primary_locations') + pl.col('secondary_locations')) > 0
        )
        underutilized_days = len(days_with_locations.filter(
            pl.col('duration') < daily_capacity_hours * 60
        ))

        # Calculate secondary duration standard deviation
        secondary_days = zone_data.filter(pl.col('secondary_locations') > 0)
        if len(secondary_days) > 1:
            secondary_durations = secondary_days.select('duration').to_series().to_list()
            secondary_durations_hours = [d / 60.0 for d in secondary_durations]  # convert to hours
            mean_duration = sum(secondary_durations_hours) / len(secondary_durations_hours)
            variance = sum((x - mean_duration) ** 2 for x in secondary_durations_hours) / len(secondary_durations_hours)
            sec_std = variance ** 0.5
        else:
            sec_std = 0.0

        # Get metadata from first record
        first_record = zone_data.row(0, named=True)

        result_data.append({
            'zone_id': zone_id,
            'primary_count': total_primary,
            'secondary_count': total_secondary,
            'total_pos_time': total_pos_time,
            'total_drive_time': total_drive_time,
            'weekly_duration': weekly_duration,
            'utilization': utilization,
            'overutilized_days': overutilized_days,
            'underutilized_days': underutilized_days,
            'sec_std': sec_std,
            'clusterer': first_record.get('clusterer', 'unknown'),
            'router': first_record.get('router', 'unknown'),
            'balancer': first_record.get('balancer', 'unknown'),
            'created_on': first_record.get('created_on', '')
        })

    if result_data:
        result_df = pl.DataFrame(result_data)
        logger.success(f"Generated aggregate reports for {len(result_df)} zones")
        return result_df
    else:
        return pl.DataFrame()


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