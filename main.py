#!/usr/bin/env python3

# standard library imports
import argparse
import json
import os
import sys
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
import src.core as core
from src.utils import io_utils, osrm_utils, geo_utils

# ------------------------------------------------------------------------------
# extract
# ------------------------------------------------------------------------------

def extract(
    zone_ids: Optional[List[str]],
    local: bool = True,
    pos_path: str = "./data/locations.jsonl"
) -> pl.DataFrame:
    """
    extract pos data to be optimized

    :param zone_ids: list of zone_ids to be optimized
    :local: whether the data will be returned via local files
    """
    logger.info("extracting pos data")

    if local:
        logger.info(f"extracting pos data from {pos_path}")
        pos = io_utils.extract(
            zone_ids=zone_ids,
            locations_path=pos_path
        ).sort("zone_id", descending=True)

    if pos.height == 0:
        logger.error("no valid zones given for processing")
        return None
    else:
        logger.success("pos data extracted")
        logger.info(pos.head())
        return pos


# ------------------------------------------------------------------------------
# optimizization 
# ------------------------------------------------------------------------------

def optimization_orchstrator(
    pos: pl.DataFrame
) -> pl.DataFrame:
    """
    runs all zone level optimization for routes using parallel processing

    :param pos: DataFrame containing all pos information
    :return: DataFrame containing a detailed route itinerary for all zones
    """
    pos_optimize = pos.clone()

    # get zone count and determine worker threads
    zone_count = pos['zone_id'].n_unique()
    zones_ids = pos['zone_id'].unique()
    max_workers = max(1, os.cpu_count() // 2)
    logger.info(f"optimizing {zone_count} zone(s) using {max_workers} threads")

    # prepare zone data for parallel processing
    zone_data_list = []
    for zone_id in zones_ids:
        pos_zone = pos_optimize.filter(pl.col("zone_id") == zone_id)
        zone_data_list.append(pos_zone)

    # optimize zones in parallel
    itinerary_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all optimization tasks
        future_to_zone = {executor.submit(optimize, pos_zone): pos_zone['zone_id'].unique()[0] 
                         for pos_zone in zone_data_list}
        
        # collect results as they complete
        for future in as_completed(future_to_zone):
            zone_id = future_to_zone[future]
            try:
                itinerary_zone = future.result()
                itinerary_list.append(itinerary_zone)
                logger.info(f"completed optimization for zone {zone_id}")
            except Exception as exc:
                logger.error(f"zone {zone_id} generated an exception: {exc}")
                raise

    # combine all results
    itinerary = pl.concat(itinerary_list, how='vertical')        
    return itinerary


def optimize(
    pos_zone: pl.DataFrame
) -> pl.DataFrame:
    """
    runs all zone level optimization for routes

    :param pos: DataFrame containing pos information for 1 zone
    :return: DataFrame containing a detailed itinerary for 1 zone
    """       
    zone_id = pos_zone['zone_id'].unique()
    if len(zone_id) > 1: 
        logger.error(f"multiple zone_ids: {zone_id} ingested")
    else:
        zone_id = zone_id[0]

    # get centroid
    centroid = geo_utils.get_centroid(
        pos_zone
    )

    # get od matrix
    od_matrix = osrm_utils.generate_od_matrix(
        pos_zone=pos_zone
    )

    # assign anchor days
    itinerary = core.optimize.assign_anchor_days(
        pos=pos_zone
    )

    # create clusters for secondary locations
    itinerary = core.optimize.cluster_secondary_pos(
        pos=pos_zone,
        itinerary=itinerary,
        od_matrix=od_matrix
    )

    # optimize individual days
    itinerary = core.optimize.gen_secondary_routes(
        itinerary=itinerary,
        od_matrix=od_matrix,
        centroid = centroid
    )

    # get detailed routes, and times
    itinerary = core.optimize.get_detailed_routes(
        itinerary=itinerary,
        od_matrix=od_matrix
    )

    # assign remaining days to 
    logger.success(f"{zone_id} optimized")
    logger.info(itinerary)

    return itinerary


# ------------------------------------------------------------------------------
# generate metrics
# ------------------------------------------------------------------------------

def report(
    itinerary: pl.DataFrame,
    local: bool
):
    """
    generates all zone level and aggregates metrics for testing and reporting

    :param itenerary: itenerary object containing daily routes for all zones
    :param local: indciates whether file operations will be performed locally
    """
    itinerary_reporting = itinerary.clone()

    zone_count = itinerary_reporting['zone_id'].n_unique()
    logger.info(f"generating reports for {zone_count} zone(s)")
    
    # generate aggregate report
    aggregate_report = core.report.aggregate(
        itinerary=itinerary_reporting,
        local=local
    )

    logger.info("aggregate report generated")
    logger.info(aggregate_report)

    # save aggregate report to JSONL
    aggregate_records = aggregate_report.to_pandas().to_dict(orient='records')
    with open('./output/aggregate-report.jsonl', 'w') as f:
        for record in aggregate_records:
            f.write(json.dumps(record) + '\n')
    
    # generate aggregate summary for all zones
    aggregate_report_summary = aggregate_report.select([
        pl.col('weekly_duration').mean().alias("average_weekly_duration"),
        pl.col('utilization').mean().alias("average_utilization"),
        pl.col('overutilized_days').mean().alias("average_overutilized_days"),
        pl.col('underutilized_days').mean().alias("average_underutilized_days"),
        pl.col('total_pos_time').mean().alias("average_daily_pos_time"),
        pl.col('total_drive_time').mean().alias("average_daily_drive_time"),  
        pl.col('sec_std').mean().alias("average_secondary_duration_standard_deviation"),  
    ])

    logger.info("aggregate_report_summary:")
    print(aggregate_report_summary)

    # save aggregate summary to JSONL
    summary_records = aggregate_report_summary.to_pandas().to_dict(orient='records')
    with open('./output/aggregate-summary.jsonl', 'w') as f:
        for record in summary_records:
            f.write(json.dumps(record) + '\n')

    return


# ------------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------------

def main(
    zone_ids: Optional[List[str]] = None,
    local: bool = True
) -> None:
    """
    primary orchestration function for the route-optimization

    :param zone_ids: list of zone_ids to be optimized
    :param local: whether operations will be performed with local files
    :return: None
    """

    # extract data
    pos = extract(
        zone_ids=zone_ids,
        local=local
    )

    if pos is None or pos.is_empty():
        return

    # perform zone optimization(s)
    itinerary = optimization_orchstrator(
        pos=pos
    )

    # generate all metrics
    report(
        itinerary=itinerary,
        local=local
    )

    # save itinerary to JSONL when local is True
    itinerary_pandas = itinerary.to_pandas()
    # Convert any numpy arrays to lists for JSON serialization
    for col in itinerary_pandas.columns:
        if itinerary_pandas[col].dtype == 'object':
            itinerary_pandas[col] = itinerary_pandas[col].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
    
    itinerary_records = itinerary_pandas.to_dict(orient='records')
    with open('./output/itinerary.jsonl', 'w') as f:
        for record in itinerary_records:
            f.write(json.dumps(record, default=str) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script with command line arguments')
    
    parser.add_argument(
        '-z', 
        '--zone_ids', 
        nargs="*",
        default=None,
        type=str, 
        help='one or more zone_ids to process (default:None)'
    )
    
    parser.add_argument(
        '-l',
        '--local',
        action='store_true',
        default=True,
        help="toggle local processing of files (default:True)"
    )

    args = parser.parse_args()

    main(
        zone_ids=args.zone_ids,
        local=args.local
    )


# ------------------------------------------------------------------------------
# end of main.py
# ------------------------------------------------------------------------------