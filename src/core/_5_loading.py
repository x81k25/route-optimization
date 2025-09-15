"""
Stage 5: Loading
Export optimized routes and results to output files
"""

# standard library imports
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 3rd-party imports
from loguru import logger
import polars as pl


def load_results_to_files(
    itinerary: pl.DataFrame,
    daily_summary: Optional[pl.DataFrame] = None,
    zone_summary: Optional[pl.DataFrame] = None,
    aggregate_summary: Optional[pl.DataFrame] = None,
    output_dir: str = "./output",
    clusterer: str = "mds_kmeans",
    balancer: str = "greedy",
    router: str = "exhaustive"
) -> None:
    """
    Export all results to output files.

    This is stage 5 where we:
    1. Write optimized routes to JSONL and Parquet files
    2. Write daily summaries to JSONL and Parquet files
    3. Generate HTML visualizations
    4. Create CSV exports for external systems
    5. Organize output directory structure

    :param itinerary: Detailed route itineraries (individual position records)
    :param daily_summary: Daily summary data
    :param zone_summary: Zone analytics data
    :param aggregate_summary: Overall summary statistics
    :param output_dir: Output directory path
    :param clusterer: Clustering algorithm used
    :param balancer: Balancing method used
    :param router: Routing algorithm used
    """
    logger.info("Stage 5: LOADING - Exporting results to files")

    # ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # get current timestamp
    created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # export itinerary data (individual position records)
    export_itinerary_data(itinerary, output_path, clusterer, balancer, router, created_on)

    # export daily summary data if provided
    if daily_summary is not None:
        export_daily_summary_data(daily_summary, output_path, clusterer, balancer, router, created_on)

    # export zone summary data if provided
    if zone_summary is not None:
        export_zone_summary_data(zone_summary, output_path, clusterer, balancer, router, created_on)

    # export aggregate summary data if provided
    if aggregate_summary is not None:
        export_aggregate_summary_data(aggregate_summary, output_path, clusterer, balancer, router, created_on)

    logger.success(f"Results exported to {output_dir}")


def export_itinerary_data(
    itinerary: pl.DataFrame,
    output_path: Path,
    clusterer: str,
    balancer: str,
    router: str,
    created_on: str
) -> None:
    """
    Export detailed itinerary data to JSONL and Parquet files.
    Merges with existing data and keeps newest records based on uniqueness constraints.
    Unique on: (zone_id, pos_id, day, clusterer, balancer)

    :param itinerary: Itinerary DataFrame
    :param output_path: Output directory path
    :param clusterer: Clustering algorithm used
    :param balancer: Balancing method used
    :param router: Routing algorithm used
    :param created_on: Timestamp of creation
    """
    if len(itinerary) == 0:
        logger.warning("No itinerary data to export")
        return

    output_file = output_path / "itinerary.jsonl"

    # add metadata to new dataframe
    new_data = itinerary.with_columns([
        pl.lit(clusterer).alias('clusterer'),
        pl.lit(router).alias('router'),
        pl.lit(balancer).alias('balancer'),
        pl.lit(created_on).alias('created_on')
    ])

    # read existing data from Parquet (primary format) with JSONL fallback for migration
    parquet_file = output_path / "itinerary.parquet"
    existing_data = None

    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            logger.debug(f"Read {len(existing_data)} existing records from Parquet")
        except Exception as e:
            logger.warning(f"Could not read existing Parquet file {parquet_file}: {e}. Trying JSONL migration.")

    # fallback to JSONL only for migration from legacy format
    if existing_data is None and output_file.exists():
        try:
            existing_data = read_jsonl_to_dataframe(output_file)
            logger.info(f"Migrating {len(existing_data)} records from JSONL to Parquet format")
        except Exception as e:
            logger.warning(f"Could not read existing JSONL file {output_file}: {e}. Using new data only.")

    # perform upsert operation using Polars
    if existing_data is not None and len(existing_data) > 0:
        # handle schema migration - add missing columns
        migration_needed = False

        if 'duration' not in existing_data.columns:
            logger.info("Migrating existing data: adding duration column with default values")
            existing_data = existing_data.with_columns([
                pl.when(pl.col('action') == 'departing')
                .then(0.0)
                .when(pl.col('action') == 'arriving')
                .then(60.0)  # standard 60 minutes at location
                .otherwise(10.0)  # default driving time for legacy data
                .alias('duration')
            ])
            migration_needed = True

        if 'pos_name' not in existing_data.columns:
            logger.info("Migrating existing data: adding pos_name column with default values")
            existing_data = existing_data.with_columns([
                pl.when(pl.col('pos_id').is_null())
                .then(pl.lit(None))  # centroid has no name
                .otherwise(pl.lit("Legacy Location"))  # placeholder for existing data
                .alias('pos_name')
            ])
            migration_needed = True

        if 'route_order' not in existing_data.columns:
            logger.info("Migrating existing data: adding route_order column with default values")
            existing_data = existing_data.with_columns([
                pl.lit(None, dtype=pl.Int64).alias('route_order')  # Will be populated in route optimization stage
            ])
            migration_needed = True

        if migration_needed:
            # reorder columns to match new schema
            expected_cols = ['zone_id', 'day', 'pos_id', 'pos_name', 'pos_class', 'route', 'action', 'schedule', 'duration',
                           'route_order', 'clusterer', 'router', 'balancer', 'created_on']
            existing_data = existing_data.select([col for col in expected_cols if col in existing_data.columns])

        # combine existing and new data
        combined_data = pl.concat([existing_data, new_data], how='vertical')

        # keep newest records based on uniqueness constraint
        # unique on: (zone_id, pos_id, day, clusterer, balancer)
        # Note: action is excluded since early pipeline stages have null actions
        final_data = (combined_data
            .with_columns(pl.col('created_on').str.to_datetime('%Y-%m-%d %H:%M:%S'))
            .sort('created_on', descending=True)
            .unique(subset=['zone_id', 'pos_id', 'day', 'clusterer', 'balancer'], keep='first')
            .with_columns(pl.col('created_on').dt.strftime('%Y-%m-%d %H:%M:%S'))
            .sort(['zone_id', 'day', 'route_order'])
        )
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"Exported {len(final_data)} itinerary records to {output_file} and {parquet_file} (merged and deduplicated)")


def export_zone_summary_data(
    zone_summary: pl.DataFrame,
    output_path: Path,
    clusterer: str,
    balancer: str,
    router: str,
    created_on: str
) -> None:
    """
    Export zone summary analytics data to JSONL and Parquet files.

    :param zone_summary: Zone summary DataFrame
    :param output_path: Output directory path
    :param clusterer: Clustering algorithm used
    :param balancer: Balancing method used
    :param router: Routing algorithm used
    :param created_on: Timestamp of creation
    """
    if len(zone_summary) == 0:
        logger.warning("No zone summary data to export")
        return

    output_file = output_path / "zone-summary.jsonl"

    # add metadata to new dataframe
    new_data = zone_summary.with_columns([
        pl.lit(clusterer).alias('clusterer'),
        pl.lit(router).alias('router'),
        pl.lit(balancer).alias('balancer'),
        pl.lit(created_on).alias('created_on')
    ])

    # read existing data from Parquet (primary format) with JSONL fallback for migration
    parquet_file = output_path / "zone-summary.parquet"
    existing_data = None

    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            logger.debug(f"Read {len(existing_data)} existing zone summary records from Parquet")
        except Exception as e:
            logger.warning(f"Could not read existing Parquet file {parquet_file}: {e}. Trying JSONL migration.")

    # fallback to JSONL only for migration from legacy format
    if existing_data is None and output_file.exists():
        try:
            existing_data = read_jsonl_to_dataframe(output_file)
            logger.info(f"Migrating {len(existing_data)} zone summary records from JSONL to Parquet format")
        except Exception as e:
            logger.warning(f"Could not read existing JSONL file {output_file}: {e}. Using new data only.")

    # perform upsert operation using Polars
    if existing_data is not None and len(existing_data) > 0:
        # combine existing and new data
        combined_data = pl.concat([existing_data, new_data], how='vertical')

        # keep newest records based on uniqueness constraint
        # unique on: (zone_id, clusterer, balancer)
        final_data = (combined_data
            .with_columns(pl.col('created_on').str.to_datetime('%Y-%m-%d %H:%M:%S'))
            .sort('created_on', descending=True)
            .unique(subset=['zone_id', 'clusterer', 'balancer'], keep='first')
            .with_columns(pl.col('created_on').dt.strftime('%Y-%m-%d %H:%M:%S'))
            .sort(['zone_id'])
        )
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"Exported {len(final_data)} zone summary records to {output_file} and {parquet_file} (merged and deduplicated)")


def export_daily_summary_data(
    daily_summary: pl.DataFrame,
    output_path: Path,
    clusterer: str,
    balancer: str,
    router: str,
    created_on: str
) -> None:
    """
    Export daily summary data to JSONL and Parquet files.
    Merges with existing data and keeps newest records based on uniqueness constraints.
    Unique on: (zone_id, day, clusterer, balancer)

    :param daily_summary: Daily summary DataFrame
    :param output_path: Output directory path
    :param clusterer: Clustering algorithm used
    :param balancer: Balancing method used
    :param router: Routing algorithm used
    :param created_on: Timestamp of creation
    """
    if len(daily_summary) == 0:
        logger.warning("No daily summary data to export")
        return

    output_file = output_path / "daily-summary.jsonl"

    # add metadata to new dataframe (if not already present)
    new_data = daily_summary
    if 'clusterer' not in daily_summary.columns:
        new_data = new_data.with_columns([
            pl.lit(clusterer).alias('clusterer'),
            pl.lit(router).alias('router'),
            pl.lit(balancer).alias('balancer'),
            pl.lit(created_on).alias('created_on')
        ])

    # read existing data from Parquet (primary format) with JSONL fallback for migration
    parquet_file = output_path / "daily-summary.parquet"
    existing_data = None

    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            logger.debug(f"Read {len(existing_data)} existing daily summary records from Parquet")
        except Exception as e:
            logger.warning(f"Could not read existing Parquet file {parquet_file}: {e}. Trying JSONL migration.")

    # fallback to JSONL only for migration from legacy format
    if existing_data is None and output_file.exists():
        try:
            existing_data = read_jsonl_to_dataframe(output_file)
            logger.info(f"Migrating {len(existing_data)} daily summary records from JSONL to Parquet format")
        except Exception as e:
            logger.warning(f"Could not read existing JSONL file {output_file}: {e}. Using new data only.")

    # perform upsert operation using Polars
    if existing_data is not None and len(existing_data) > 0:
        # combine existing and new data
        combined_data = pl.concat([existing_data, new_data], how='vertical')

        # keep newest records based on uniqueness constraint
        # unique on: (zone_id, day, clusterer, balancer)
        final_data = (combined_data
            .with_columns(pl.col('created_on').str.to_datetime('%Y-%m-%d %H:%M:%S'))
            .sort('created_on', descending=True)
            .unique(subset=['zone_id', 'day', 'clusterer', 'balancer'], keep='first')
            .with_columns(pl.col('created_on').dt.strftime('%Y-%m-%d %H:%M:%S'))
            .sort(['zone_id', 'day'])
        )
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"Exported {len(final_data)} daily summary records to {output_file} and {parquet_file} (merged and deduplicated)")


def export_aggregate_summary_data(
    aggregate_summary: pl.DataFrame,
    output_path: Path,
    clusterer: str,
    balancer: str,
    router: str,
    created_on: str
) -> None:
    """
    Export aggregate summary statistics to JSONL and Parquet files.
    Merges with existing data and keeps newest records based on uniqueness constraints.
    Unique on: (clusterer, balancer)

    :param aggregate_summary: Aggregate summary DataFrame
    :param output_path: Output directory path
    :param clusterer: Clustering algorithm used
    :param balancer: Balancing method used
    :param router: Routing algorithm used
    :param created_on: Timestamp of creation
    """
    if len(aggregate_summary) == 0:
        logger.warning("No aggregate summary data to export")
        return

    output_file = output_path / "aggregate-summary.jsonl"

    # add metadata to new dataframe
    new_data = aggregate_summary.with_columns([
        pl.lit(clusterer).alias('clusterer'),
        pl.lit(router).alias('router'),
        pl.lit(balancer).alias('balancer'),
        pl.lit(created_on).alias('created_on')
    ])

    # read existing data from Parquet (primary format) with JSONL fallback for migration
    parquet_file = output_path / "aggregate-summary.parquet"
    existing_data = None

    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            logger.debug(f"Read {len(existing_data)} existing aggregate summary records from Parquet")
        except Exception as e:
            logger.warning(f"Could not read existing Parquet file {parquet_file}: {e}. Trying JSONL migration.")

    # fallback to JSONL only for migration from legacy format
    if existing_data is None and output_file.exists():
        try:
            existing_data = read_jsonl_to_dataframe(output_file)
            logger.info(f"Migrating {len(existing_data)} aggregate summary records from JSONL to Parquet format")
        except Exception as e:
            logger.warning(f"Could not read existing JSONL file {output_file}: {e}. Using new data only.")

    # perform upsert operation using Polars
    if existing_data is not None and len(existing_data) > 0:
        # combine existing and new data
        combined_data = pl.concat([existing_data, new_data], how='vertical')

        # keep newest records based on uniqueness constraint
        # unique on: (clusterer, balancer)
        final_data = (combined_data
            .with_columns(pl.col('created_on').str.to_datetime('%Y-%m-%d %H:%M:%S'))
            .sort('created_on', descending=True)
            .unique(subset=['clusterer', 'balancer'], keep='first')
            .with_columns(pl.col('created_on').dt.strftime('%Y-%m-%d %H:%M:%S'))
        )
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"Exported {len(final_data)} aggregate summary records to {output_file} and {parquet_file} (merged and deduplicated)")


def export_csv_files(
    itinerary: pl.DataFrame,
    output_path: Path,
    flatten_routes: bool = True
) -> None:
    """
    Export data in CSV format for external systems.

    :param itinerary: Itinerary DataFrame with individual position records
    :param output_path: Output directory path
    :param flatten_routes: Whether to flatten route geometry
    """
    if len(itinerary) == 0:
        logger.warning("No data to export as CSV")
        return

    # create simplified version for CSV export with new schema
    csv_data = itinerary.select([
        "zone_id",
        "day",
        "pos_id",
        "pos_class",
        "action",
        "schedule"
    ])

    output_file = output_path / "routes_summary.csv"
    csv_data.write_csv(output_file)

    logger.info(f"Exported CSV summary to {output_file}")


def clean_record_for_json(record: Dict) -> Dict:
    """
    Clean a record for JSON serialization by converting numpy types.
    
    :param record: Dictionary record
    :return: Cleaned record with JSON-serializable types
    """
    cleaned = {}
    
    for key, value in record.items():
        # handle numpy types and convert to Python types
        if hasattr(value, 'item'):  # numpy scalars
            cleaned[key] = value.item()
        elif isinstance(value, list):
            cleaned[key] = [clean_list_item(item) for item in value]
        else:
            cleaned[key] = value
    
    return cleaned


def clean_list_item(item):
    """
    Clean individual list items for JSON serialization.
    
    :param item: List item to clean
    :return: JSON-serializable item
    """
    if hasattr(item, 'item'):  # numpy scalar
        return item.item()
    elif isinstance(item, list):
        return [clean_list_item(subitem) for subitem in item]
    else:
        return item


def create_visualization_files(
    itinerary_df: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Create HTML visualization files for routes.
    
    :param itinerary_df: Itinerary DataFrame
    :param output_path: Output directory path
    """
    vis_path = output_path / "visualizations"
    vis_path.mkdir(exist_ok=True)
    
    # create visualizations for each zone
    zones = itinerary_df["zone_id"].unique().to_list()
    
    for zone_id in zones:
        zone_df = itinerary_df.filter(pl.col("zone_id") == zone_id)
        
        if len(zone_df) > 0:
            vis_file = vis_path / f"route_map_{zone_id}.html"
            create_zone_visualization(zone_df, vis_file, zone_id)


def create_zone_visualization(
    zone_df: pl.DataFrame,
    output_file: Path,
    zone_id: str
) -> None:
    """
    Create HTML visualization for a single zone.

    :param zone_df: DataFrame for the zone
    :param output_file: Output HTML file path
    :param zone_id: Zone identifier
    """
    # this would typically use a mapping library like Folium
    # for now, create a simple HTML placeholder

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Map - {zone_id}</title>
    </head>
    <body>
        <h1>Route Optimization - {zone_id}</h1>
        <p>Total days: {len(zone_df)}</p>
        <p>Interactive map visualization would be generated here using the route geometry data.</p>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Created visualization placeholder for {zone_id} at {output_file}")


def read_jsonl_to_dataframe(file_path: Path) -> pl.DataFrame:
    """
    Read JSONL file into a Polars DataFrame.

    :param file_path: Path to JSONL file
    :return: Polars DataFrame
    """
    try:
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if records:
            return pl.DataFrame(records)
        else:
            return pl.DataFrame()
    except Exception as e:
        logger.warning(f"Error reading JSONL file {file_path}: {e}")
        return pl.DataFrame()


def write_dataframe_to_jsonl(data: pl.DataFrame, file_path: Path) -> None:
    """
    Write Polars DataFrame to JSONL file.

    :param data: Polars DataFrame
    :param file_path: Path to output JSONL file
    """
    records = data.to_dicts()

    with open(file_path, 'w') as f:
        for record in records:
            cleaned_record = clean_record_for_json(record)
            f.write(json.dumps(cleaned_record) + '\n')


def write_dataframe_to_parquet(data: pl.DataFrame, file_path: Path) -> None:
    """
    Write Polars DataFrame to Parquet file.

    :param data: Polars DataFrame
    :param file_path: Path to output Parquet file
    """
    data.write_parquet(file_path)