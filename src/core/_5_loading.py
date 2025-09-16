# standard library imports
import json
from pathlib import Path
from typing import Dict, Optional

# 3rd-party imports
from loguru import logger
import polars as pl


def load_results_to_files(
    itinerary: pl.DataFrame,
    daily_summary: Optional[pl.DataFrame] = None,
    zone_summary: Optional[pl.DataFrame] = None,
    aggregate_summary: Optional[pl.DataFrame] = None,
    output_dir: str = "./output"
) -> None:
    """
    Export all results to output files.

    This is stage 5 where we:
    1. Write optimized routes to JSONL and Parquet files
    2. Write daily summaries to JSONL and Parquet files
    3. Write zone summaries to JSONL and Parquet files
    4. Write aggregate summaries to JSONL and Parquet files

    :param itinerary: Detailed route itineraries (individual position records)
    :param daily_summary: Daily summary data
    :param zone_summary: Zone analytics data
    :param aggregate_summary: Overall summary statistics
    :param output_dir: Output directory path
    """
    logger.info("stage 5: loading - exporting results to files")

    # ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # export itinerary data (individual position records)
    export_itinerary_data(itinerary, output_path)

    # export daily summary data if provided
    if daily_summary is not None:
        export_daily_summary_data(daily_summary, output_path)

    # export zone summary data if provided
    if zone_summary is not None:
        export_zone_summary_data(zone_summary, output_path)

    # export aggregate summary data if provided
    if aggregate_summary is not None:
        export_aggregate_summary_data(aggregate_summary, output_path)

    logger.success(f"results exported to {output_dir}")


def export_itinerary_data(
    itinerary: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Export detailed itinerary data by appending to JSONL and Parquet files.

    :param itinerary: Itinerary DataFrame
    :param output_path: Output directory path
    """
    if len(itinerary) == 0:
        logger.warning("no itinerary data to export")
        return

    output_file = output_path / "itinerary.jsonl"
    parquet_file = output_path / "itinerary.parquet"

    new_data = itinerary

    # read existing data and append new rows
    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            final_data = pl.concat([existing_data, new_data], how='vertical')
            logger.debug(f"appending {len(new_data)} records to existing {len(existing_data)} records")
        except Exception as e:
            logger.warning(f"could not read existing Parquet file: {e}. creating new file.")
            final_data = new_data
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"exported {len(new_data)} itinerary records (total: {len(final_data)})")


def export_daily_summary_data(
    daily_summary: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Export daily summary data by appending to JSONL and Parquet files.

    :param daily_summary: Daily summary DataFrame
    :param output_path: Output directory path
    """
    if len(daily_summary) == 0:
        logger.warning("no daily summary data to export")
        return

    output_file = output_path / "daily-summary.jsonl"
    parquet_file = output_path / "daily-summary.parquet"

    new_data = daily_summary

    # read existing data and append new rows
    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            final_data = pl.concat([existing_data, new_data], how='vertical')
            logger.debug(f"appending {len(new_data)} records to existing {len(existing_data)} records")
        except Exception as e:
            logger.warning(f"could not read existing Parquet file: {e}. creating new file.")
            final_data = new_data
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"exported {len(new_data)} daily summary records (total: {len(final_data)})")


def export_zone_summary_data(
    zone_summary: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Export zone summary analytics data by appending to JSONL and Parquet files.

    :param zone_summary: Zone summary DataFrame
    :param output_path: Output directory path
    """
    if len(zone_summary) == 0:
        logger.warning("no zone summary data to export")
        return

    output_file = output_path / "zone-summary.jsonl"
    parquet_file = output_path / "zone-summary.parquet"

    new_data = zone_summary

    # read existing data and append new rows
    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            final_data = pl.concat([existing_data, new_data], how='vertical')
            logger.debug(f"appending {len(new_data)} records to existing {len(existing_data)} records")
        except Exception as e:
            logger.warning(f"could not read existing Parquet file: {e}. creating new file.")
            final_data = new_data
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"exported {len(new_data)} zone summary records (total: {len(final_data)})")



def export_aggregate_summary_data(
    aggregate_summary: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Export aggregate summary statistics by appending to JSONL and Parquet files.

    :param aggregate_summary: Aggregate summary DataFrame
    :param output_path: Output directory path
    """
    if len(aggregate_summary) == 0:
        logger.warning("no aggregate summary data to export")
        return

    output_file = output_path / "aggregate-summary.jsonl"
    parquet_file = output_path / "aggregate-summary.parquet"

    new_data = aggregate_summary

    # read existing data and append new rows
    if parquet_file.exists():
        try:
            existing_data = pl.read_parquet(parquet_file)
            final_data = pl.concat([existing_data, new_data], how='vertical')
            logger.debug(f"appending {len(new_data)} records to existing {len(existing_data)} records")
        except Exception as e:
            logger.warning(f"could not read existing Parquet file: {e}. creating new file.")
            final_data = new_data
    else:
        final_data = new_data

    # write Parquet first (primary format for fast processing)
    write_dataframe_to_parquet(final_data, parquet_file)

    # write JSONL second (human-readable backup)
    write_dataframe_to_jsonl(final_data, output_file)

    logger.info(f"exported {len(new_data)} aggregate summary records (total: {len(final_data)})")




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
            cleaned[key] = [item.item() if hasattr(item, 'item') else item for item in value]
        else:
            cleaned[key] = value

    return cleaned




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
        logger.warning(f"error reading JSONL file {file_path}: {e}")
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