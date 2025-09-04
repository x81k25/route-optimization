#!/usr/bin/env python3

# standard library imports
import argparse
import json
import sys
from typing import List, Optional, Set

# 3rd-party imports
from loguru import logger
import polars as pl

# local imports
import src.core as core
from src.utils import io_utils, osrm_utils

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
# optimizization orchestrator
# ------------------------------------------------------------------------------

def optimization_orchstrator(
    pos: pl.DataFrame
) -> pl.DataFrame:
    """
    runs all zone level optimization for routes

    :param pos: DataFrame containing all pos information
    :return: DataFrame containing a detailed route itinerary for all zones
    """
    pos_optimize = pos.clone()

    # get zone count
    zone_count = pos['zone_id'].n_unique()
    zones_ids = pos['zone_id'].unique()
    logger.info(f"optimizing {zone_count} zone(s)")

    # optimize zones
    itinerary = pl.DataFrame()

    for zone_id in zones_ids:
        pos_zone = pos_optimize.filter(pl.col("zone_id") == zone_id)
        itinerary_zone = optimize(
            pos_zone=pos_zone
        )
        itinerary = pl.concat([itinerary, itinerary_zone], how = 'vertical')        

    return itinerary


# ------------------------------------------------------------------------------
# optimize zones
# ------------------------------------------------------------------------------

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
        od_matrix=od_matrix
    )

    # get detailed routes, and timesii
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

    if local:
        core.report.zone(
            itinerary=itinerary_reporting            
        )


    logger.info("aggregate report generated")
    logger.info(aggregate_report)

    return

# ------------------------------------------------------------------------------
# generate ad-hoc outputs
# ------------------------------------------------------------------------------



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

    # save itinerary to JSON when local is True
    if local:
        itinerary_json = itinerary.to_pandas().to_json(orient='records', indent=2)
        with open('./output/itinerary.json', 'w') as f:
            f.write(itinerary_json)

    # commit data to permanent data store

    # generate ad-hoc outputs
#    generate(
#        zones=zones,
#        report=report
#    )


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