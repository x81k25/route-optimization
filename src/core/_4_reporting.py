# standard library imports
from datetime import datetime
from typing import Any, Dict

# 3rd-party imports
import polars as pl
from loguru import logger


def gen_daily_summary(
    itinerary_df: pl.DataFrame,
    model_params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Generate daily summary table from individual itinerary records.
    Aggregates position records into daily metrics and rolls up metadata from itinerary.

    :param itinerary_df: DataFrame with individual position records
    :param model_params: Model configuration parameters
    :return: DataFrame with daily summary records including rolled-up metadata
    """
    if len(itinerary_df) == 0:
        logger.warning("no itinerary data to generate daily summary from")
        return pl.DataFrame()

    logger.info("generating daily summary from itinerary")

    # Roll up clusterer, balancer, and router from itinerary using Polars
    metadata_df = itinerary_df.select([
        pl.col("clusterer").unique(),
        pl.col("balancer").unique(),
        pl.col("router").unique()
    ])

    clusterer_values = metadata_df["clusterer"].to_list()
    balancer_values = metadata_df["balancer"].to_list()
    router_values = metadata_df["router"].to_list()

    # Use first value or 'unknown' if multiple/none exist
    clusterer = clusterer_values[0] if len(clusterer_values) == 1 else "unknown"
    balancer = balancer_values[0] if len(balancer_values) == 1 else "unknown"
    router = router_values[0] if len(router_values) == 1 else "unknown"

    if len(clusterer_values) > 1:
        logger.warning(f"multiple clusterers found in itinerary: {clusterer_values}")
    if len(balancer_values) > 1:
        logger.warning(f"multiple balancers found in itinerary: {balancer_values}")
    if len(router_values) > 1:
        logger.warning(f"multiple routers found in itinerary: {router_values}")

    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hours_per_day_minutes = model_params.get("hours_per_day", 8) * 60  # convert to minutes

    # Use Polars group_by and aggregation instead of iteration
    daily_summary = (
        itinerary_df
        .group_by(['zone_id', 'day'])
        .agg([
            # Count distinct primary locations (excluding nulls)
            (pl.col('pos_id')
             .filter((pl.col('pos_id').is_not_null()) & (pl.col('pos_class') == 'primary'))
             .n_unique()
             .alias('primary_locations')),

            # Count distinct secondary locations (excluding nulls)
            (pl.col('pos_id')
             .filter((pl.col('pos_id').is_not_null()) & (pl.col('pos_class') == 'secondary'))
             .n_unique()
             .alias('secondary_locations')),

            # Sum duration for non-null pos_ids (total POS time)
            (pl.col('duration')
             .filter(pl.col('pos_id').is_not_null())
             .sum()
             .fill_null(0.0)
             .alias('total_pos_time')),

            # Sum duration for driving actions (total drive time)
            (pl.col('duration')
             .filter(pl.col('action') == 'driving')
             .sum()
             .fill_null(0.0)
             .alias('total_drive_time')),

            # Get min and max schedule times for duration calculation
            pl.col('schedule').min().alias('min_schedule'),
            pl.col('schedule').max().alias('max_schedule'),

            # Count of records for this group (to check if > 1)
            pl.count().alias('record_count')
        ])
        .with_columns([
            # Calculate duration based on schedule or sum of times
            pl.when(
                (pl.col('min_schedule').is_not_null()) &
                (pl.col('max_schedule').is_not_null()) &
                (pl.col('record_count') > 1)
            )
            .then(pl.col('max_schedule') - pl.col('min_schedule'))
            .otherwise(pl.col('total_pos_time') + pl.col('total_drive_time'))
            .alias('duration'),

            # Add metadata columns
            pl.lit(clusterer).alias('clusterer'),
            pl.lit(router).alias('router'),
            pl.lit(balancer).alias('balancer'),
            pl.lit(created_on).alias('created_on')
        ])
        .with_columns([
            # Calculate utilization percentage
            ((pl.col('duration') / hours_per_day_minutes) * 100).alias('utilization_percentage')
        ])
        .select([
            'zone_id',
            'day',
            pl.col('primary_locations').cast(pl.Int64),
            pl.col('secondary_locations').cast(pl.Int64),
            'duration',
            'utilization_percentage',
            'total_pos_time',
            'total_drive_time',
            'clusterer',
            'router',
            'balancer',
            'created_on'
        ])
        .sort(['zone_id', 'day'])
    )

    if len(daily_summary) > 0:
        logger.success(f"generated daily summary for {len(daily_summary)} zone-days")
        return daily_summary
    else:
        logger.warning("no daily summary data generated")
        return pl.DataFrame()


def gen_zone_summary(
    daily_summary_df: pl.DataFrame,
    model_params: Dict[str, Any] = None
) -> pl.DataFrame:
    """
    Generate zone-level summary from daily summary data using Polars methods.

    :param daily_summary_df: Daily summary DataFrame
    :param model_params: Model configuration parameters
    :return: DataFrame with zone-level aggregate metrics
    """
    if len(daily_summary_df) == 0:
        logger.warning("no daily summary data to generate zone summary from")
        return pl.DataFrame()

    logger.info("generating zone summary from daily summary data")

    # Get model parameters with defaults
    if model_params is None:
        model_params = {}
    days_per_week = model_params.get("days_per_week", 5)
    hours_per_day = model_params.get("hours_per_day", 8)
    target_weekly_hours = days_per_week * hours_per_day
    daily_capacity_minutes = hours_per_day * 60

    # Generate zone summary using Polars aggregation
    zone_summary = (
        daily_summary_df
        .group_by('zone_id')
        .agg([
            # Count distinct primary and secondary locations (max per day)
            pl.col('primary_locations').max().alias('primary_count'),
            pl.col('secondary_locations').max().alias('secondary_count'),

            # Sum position and drive times (convert to hours)
            (pl.col('total_pos_time').sum() / 60.0).alias('total_pos_time'),
            (pl.col('total_drive_time').sum() / 60.0).alias('total_drive_time'),

            # Sum weekly duration (convert to hours)
            (pl.col('duration').sum() / 60.0).alias('weekly_duration'),

            # Count over/under utilized days
            (pl.col('duration') > daily_capacity_minutes).sum().alias('overutilized_days'),
            ((pl.col('primary_locations') + pl.col('secondary_locations') > 0) &
             (pl.col('duration') < daily_capacity_minutes)).sum().alias('underutilized_days'),

            # Calculate duration standard deviation for all days
            (pl.col('duration') / 60.0).std().fill_null(0.0).alias('duration_std'),

            # Get metadata from first record
            pl.col('clusterer').first().alias('clusterer'),
            pl.col('router').first().alias('router'),
            pl.col('balancer').first().alias('balancer'),
            pl.col('created_on').first().alias('created_on')
        ])
        .with_columns([
            # Calculate utilization percentage
            ((pl.col('weekly_duration') / target_weekly_hours) * 100).alias('utilization')
        ])
        .select([
            'zone_id',
            pl.col('primary_count').cast(pl.Int64),
            pl.col('secondary_count').cast(pl.Int64),
            'total_pos_time',
            'total_drive_time',
            'weekly_duration',
            'utilization',
            pl.col('overutilized_days').cast(pl.Int64),
            pl.col('underutilized_days').cast(pl.Int64),
            'duration_std',
            'clusterer',
            'router',
            'balancer',
            'created_on'
        ])
        .sort('zone_id')
    )

    if len(zone_summary) > 0:
        logger.success(f"generated zone summary for {len(zone_summary)} zones")
        return zone_summary
    else:
        logger.warning("no zone summary data generated")
        return pl.DataFrame()


def gen_aggregate_summary(zone_summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate aggregate summary statistics from zone summary data using Polars methods.

    :param zone_summary_df: Zone summary DataFrame
    :return: DataFrame with aggregate statistics across all zones
    """
    if len(zone_summary_df) == 0:
        logger.warning("no zone summary data to generate aggregate summary from")
        return pl.DataFrame()

    logger.info("generating aggregate summary from zone summary data")

    # Generate aggregate statistics using Polars aggregation
    aggregate_summary = (
        zone_summary_df
        .select([
            pl.col('weekly_duration').mean().alias('average_weekly_duration'),
            pl.col('utilization').mean().alias('average_utilization'),
            pl.col('overutilized_days').mean().alias('average_overutilized_days'),
            pl.col('underutilized_days').mean().alias('average_underutilized_days'),
            pl.col('total_pos_time').mean().alias('average_daily_pos_time'),
            pl.col('total_drive_time').mean().alias('average_daily_drive_time'),
            pl.col('duration_std').mean().alias('average_duration_standard_deviation'),

            # Get metadata from first record
            pl.col('clusterer').first().alias('clusterer'),
            pl.col('router').first().alias('router'),
            pl.col('balancer').first().alias('balancer'),
            pl.col('created_on').first().alias('created_on')
        ])
    )

    if len(aggregate_summary) > 0:
        logger.success("generated aggregate summary statistics")
        return aggregate_summary
    else:
        logger.warning("no aggregate summary data generated")
        return pl.DataFrame()


if __name__ == "__main__":
    # example usage and testing
    logger.info("testing report module")
    
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

    # test parameters
    test_model_params = {
        "days_per_week": 5,
        "hours_per_day": 8
    }

    # test daily summary generation
    daily_summary = gen_daily_summary(mock_itinerary, test_model_params)
    logger.info("daily summary result:")
    logger.info(daily_summary)

    # test zone summary generation
    zone_summary = gen_zone_summary(daily_summary, test_model_params)
    logger.info("zone summary result:")
    logger.info(zone_summary)

    # test aggregate summary generation
    aggregate_summary = gen_aggregate_summary(zone_summary)
    logger.info("aggregate summary result:")
    logger.info(aggregate_summary)


# ------------------------------------------------------------------------------
# end of report.py
# ------------------------------------------------------------------------------